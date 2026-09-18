// libtierinfer_mmap.so — give llama.cpp a mapping TierInfer owns, without changing llama.cpp.
//
// Preloaded into an unmodified llama.cpp process (LD_PRELOAD), this does three things:
//
//   1. Interposes mmap(). A read-only file mapping of one of the files named in
//      TIERINFER_FILES becomes an anonymous region registered with a userfaultfd
//      for MISSING faults. The descriptor, the path, the base address and the
//      length go to the TierInfer server over the unix socket TIERINFER_SOCK
//      (SCM_RIGHTS for the fd), and mmap() returns only after the server has
//      acknowledged — so no page is touched before someone can answer for it.
//      From then on the first touch of any page of an expert is one fault the
//      server answers with the whole expert; nothing else in the process knows.
//
//   2. Interposes llama_init_from_model(). The context parameters get a cb_eval
//      that reads every ffn_moe_topk-<layer> tensor row by row (it is a strided
//      view from b10482) and sends the routing to the server as text. A cb_eval
//      the caller already set is chained, so tierinfer-trace keeps working.
//
//   3. Runs an eviction thread. The server writes "EVICT <addr> <len>\n" on a
//      second socket and the thread calls madvise(MADV_DONTNEED) on the range,
//      which is the only way pages in *this* process can be dropped; the next
//      touch faults again and the server decides anew.
//
// With TIERINFER_SOCK unset, or the server unreachable, every call falls
// through to the real implementation: the clean baseline is the default, and
// the shim says on stderr that it stood aside.
//
// Nothing here decides anything. Which expert to keep, evict or prefetch is
// the server's; this is the pair of hands inside the process.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#ifndef UFFD_USER_MODE_ONLY
#define UFFD_USER_MODE_ONLY 1
#endif

typedef void * (*mmap_fn)(void *, size_t, int, int, int, off_t);
typedef struct llama_context * (*init_fn)(struct llama_model *, struct llama_context_params);

static mmap_fn real_mmap;
static init_fn real_init;

static int  ctl_sock = -1;      // shim -> server: MAP, ROUTE; server -> shim: acks
static int  evict_sock = -1;    // server -> shim: EVICT
static int  uffd = -1;
static bool enabled = false;
static bool announced = false;
static pthread_mutex_t ctl_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_t evict_thread;

static char *files[64];
static int   n_files;

static void say(const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fputs("tierinfer-mmap: ", stderr); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap);
}

static int connect_sock(const char * path) {
    int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return -1;
    struct sockaddr_un a; memset(&a, 0, sizeof a); a.sun_family = AF_UNIX;
    strncpy(a.sun_path, path, sizeof a.sun_path - 1);
    if (connect(s, (struct sockaddr *) &a, sizeof a) < 0) { close(s); return -1; }
    return s;
}

static bool write_all(int s, const void * buf, size_t n) {
    const char * p = buf;
    while (n) {
        ssize_t w = write(s, p, n);
        if (w < 0) { if (errno == EINTR) continue; return false; }
        p += w; n -= (size_t) w;
    }
    return true;
}

static bool read_line(int s, char * buf, size_t cap) {
    size_t i = 0;
    while (i + 1 < cap) {
        char c; ssize_t r = read(s, &c, 1);
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        if (c == '\n') break;
        buf[i++] = c;
    }
    buf[i] = 0;
    return true;
}

static void * evict_loop(void * arg) {
    (void) arg;
    char line[256];
    long done = 0;
    while (read_line(evict_sock, line, sizeof line)) {
        unsigned long long addr, len;
        if (sscanf(line, "EVICT %llx %llu", &addr, &len) == 2) {
            if (madvise((void *) (uintptr_t) addr, (size_t) len, MADV_DONTNEED) != 0) {
                say("madvise(DONTNEED, %llx, %llu) failed: %s", addr, len, strerror(errno));
            } else {
                done++;
            }
        } else if (strcmp(line, "PING") == 0) {
            write_all(evict_sock, "PONG\n", 5);
        }
    }
    say("eviction channel closed after %ld evictions", done);
    return NULL;
}

__attribute__((constructor)) static void setup(void) {
    real_mmap = (mmap_fn) dlsym(RTLD_NEXT, "mmap");
    real_init = (init_fn) dlsym(RTLD_NEXT, "llama_init_from_model");
    const char * sock = getenv("TIERINFER_SOCK");
    const char * list = getenv("TIERINFER_FILES");
    if (!sock || !list) { say("TIERINFER_SOCK/TIERINFER_FILES unset — standing aside (native mmap)"); return; }
    char * dup = strdup(list);
    for (char * tok = strtok(dup, ":"); tok && n_files < 64; tok = strtok(NULL, ":")) {
        char * rp = realpath(tok, NULL);
        files[n_files++] = rp ? rp : strdup(tok);
    }
    ctl_sock = connect_sock(sock);
    evict_sock = connect_sock(sock);
    if (ctl_sock < 0 || evict_sock < 0) {
        say("cannot reach the server at %s — standing aside (native mmap)", sock);
        if (ctl_sock >= 0) close(ctl_sock);
        if (evict_sock >= 0) close(evict_sock);
        ctl_sock = evict_sock = -1;
        return;
    }
    char hello[64];
    snprintf(hello, sizeof hello, "HELLO ctl %d\n", (int) getpid());
    write_all(ctl_sock, hello, strlen(hello));
    snprintf(hello, sizeof hello, "HELLO evict %d\n", (int) getpid());
    write_all(evict_sock, hello, strlen(hello));
    pthread_create(&evict_thread, NULL, evict_loop, NULL);
    enabled = true;
    say("connected to %s for %d file(s)", sock, n_files);
}

static bool is_model_fd(int fd, char * path_out, size_t cap) {
    char link[64]; snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(link, path_out, cap - 1);
    if (n <= 0) return false;
    path_out[n] = 0;
    for (int i = 0; i < n_files; i++) if (strcmp(files[i], path_out) == 0) return true;
    return false;
}

static bool ensure_uffd(void) {
    if (uffd >= 0) return true;
    int fd = (int) syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
    if (fd < 0) { say("userfaultfd: %s", strerror(errno)); return false; }
    struct uffdio_api api = { .api = UFFD_API, .features = 0 };
    if (ioctl(fd, UFFDIO_API, &api) < 0) { say("UFFDIO_API: %s", strerror(errno)); close(fd); return false; }
    uffd = fd;
    return true;
}

// Send "MAP <path> <base> <len>\n" with the uffd attached, wait for "OK".
static bool announce_mapping(const char * path, void * base, size_t len) {
    char msg[4200];
    int n = snprintf(msg, sizeof msg, "MAP %s %llx %llu\n", path,
                     (unsigned long long) (uintptr_t) base, (unsigned long long) len);
    struct iovec iov = { .iov_base = msg, .iov_len = (size_t) n };
    char cbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr mh; memset(&mh, 0, sizeof mh);
    mh.msg_iov = &iov; mh.msg_iovlen = 1;
    mh.msg_control = cbuf; mh.msg_controllen = sizeof cbuf;
    struct cmsghdr * cm = CMSG_FIRSTHDR(&mh);
    cm->cmsg_level = SOL_SOCKET; cm->cmsg_type = SCM_RIGHTS; cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &uffd, sizeof(int));
    pthread_mutex_lock(&ctl_lock);
    bool ok = sendmsg(ctl_sock, &mh, 0) == n;
    char reply[64] = "";
    if (ok) ok = read_line(ctl_sock, reply, sizeof reply) && strncmp(reply, "OK", 2) == 0;
    pthread_mutex_unlock(&ctl_lock);
    if (!ok) say("server did not accept the mapping of %s: %s", path, reply);
    return ok;
}

void * mmap(void * addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) real_mmap = (mmap_fn) dlsym(RTLD_NEXT, "mmap");
    char path[4096];
    if (!enabled || fd < 0 || off != 0 || (prot & PROT_WRITE) || !is_model_fd(fd, path, sizeof path)) {
        return real_mmap(addr, len, prot, flags, fd, off);
    }
    if (!ensure_uffd()) return real_mmap(addr, len, prot, flags, fd, off);
    // Anonymous, private, no reservation: the pages arrive by UFFDIO_COPY and
    // leave by MADV_DONTNEED. No MAP_POPULATE whatever llama.cpp asked for —
    // populating would fault the whole file through us at load.
    // The region and the registration are page-granular; a GGUF's size is not.
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const size_t alen = (len + page - 1) & ~(page - 1);
    void * base = real_mmap(NULL, alen, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (base == MAP_FAILED) { say("anonymous mmap of %zu bytes failed: %s", alen, strerror(errno)); return MAP_FAILED; }
    madvise(base, alen, MADV_NOHUGEPAGE);   // keep faults and evictions at 4 KB granularity at slab edges
    struct uffdio_register reg = { .range = { .start = (uintptr_t) base, .len = alen },
                                   .mode = UFFDIO_REGISTER_MODE_MISSING };
    if (ioctl(uffd, UFFDIO_REGISTER, &reg) < 0) {
        say("UFFDIO_REGISTER %zu bytes: %s — falling back to a file mapping", alen, strerror(errno));
        munmap(base, alen);
        return real_mmap(addr, len, prot, flags, fd, off);
    }
    if (!announce_mapping(path, base, len)) {
        munmap(base, alen);
        return real_mmap(addr, len, prot, flags, fd, off);
    }
    if (!announced) { say("serving %s through userfaultfd (%zu bytes at %p)", path, len, base); announced = true; }
    // For tests that need to know where the region landed (a client cannot
    // otherwise tell an anonymous region from any other in its own maps).
    const char * note = getenv("TIERINFER_BASE_FILE");
    if (note) {
        FILE * f = fopen(note, "a");
        if (f) { fprintf(f, "%s %llx %llu\n", path, (unsigned long long) (uintptr_t) base, (unsigned long long) len); fclose(f); }
    }
    return base;
}

// Callers built with _FILE_OFFSET_BITS=64 (Python, llama.cpp) reference the
// mmap64 symbol; on x86_64 glibc it is the same function under another name.
void * mmap64(void * addr, size_t len, int prot, int flags, int fd, off_t off) __attribute__((alias("mmap")));

// -- routing out ------------------------------------------------------------

static const char * TOPK = "ffn_moe_topk-";
static ggml_backend_sched_eval_callback user_cb;
static void * user_cb_data;

static bool on_eval(struct ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    bool mine = strncmp(t->name, TOPK, strlen(TOPK)) == 0;
    bool theirs = user_cb ? user_cb(t, ask, user_cb_data) : false;
    if (ask) return mine || theirs;
    if (!mine || t->type != GGML_TYPE_I32 || t->ne[2] != 1 || t->ne[3] != 1) return true;
    const int layer = atoi(t->name + strlen(TOPK));
    const int64_t n_used = t->ne[0], n_tokens = t->ne[1];
    // ROUTE <layer> <n_tokens> <n_used> e,e,e;e,e,e\n — rows through nb[1]
    size_t cap = 64 + (size_t) n_tokens * (size_t) n_used * 5;
    char * line = malloc(cap);
    if (!line) return true;
    int pos = snprintf(line, cap, "ROUTE %d %lld %lld ", layer, (long long) n_tokens, (long long) n_used);
    int32_t row[64];
    for (int64_t i = 0; i < n_tokens; i++) {
        ggml_backend_tensor_get(t, row, (size_t) (i * t->nb[1]), (size_t) n_used * sizeof(int32_t));
        for (int64_t j = 0; j < n_used && pos < (int) cap - 12; j++) {
            pos += snprintf(line + pos, cap - (size_t) pos, "%s%d", j ? "," : (i ? ";" : ""), row[j]);
        }
    }
    line[pos++] = '\n';
    pthread_mutex_lock(&ctl_lock);
    write_all(ctl_sock, line, (size_t) pos);
    pthread_mutex_unlock(&ctl_lock);
    free(line);
    return true;
}

struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params) {
    if (!real_init) real_init = (init_fn) dlsym(RTLD_NEXT, "llama_init_from_model");
    if (enabled) {
        user_cb = params.cb_eval;
        user_cb_data = params.cb_eval_user_data;
        params.cb_eval = on_eval;
        params.cb_eval_user_data = NULL;
        say("routing callback installed%s", user_cb ? " (chained after the caller's)" : "");
    }
    return real_init(model, params);
}
