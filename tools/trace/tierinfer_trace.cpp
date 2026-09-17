// Capture a MoE routing trace from a real model, without changing llama.cpp.
//
// Everything else in this project reasons about which experts a token needs.
// Until now it reasoned about a trace this project also generated, which
// proves the code runs and nothing about the model. This gets the real thing.
//
// No patch is required. llama.cpp names the top-k selection tensor
// "ffn_moe_topk-<layer>" (llama_context::graph_get_cb formats every callback
// name that way), and exposes ggml_backend_sched_eval_callback through
// llama_context_params::cb_eval. So a trace is a filter on tensor names plus
// a read of int32 data: the inference path is untouched, and with the tool
// not running there is nothing to turn off. That is the clean baseline SCOPE
// goal 6 asks to keep — it is the default, not a mode.
//
// Output is JSONL, one line per decode per layer:
//     {"decode":0,"layer":3,"n_tokens":11,"n_used":8,"experts":[[..],[..]]}
// Assembling per-token routings from that is the reader's job, because the
// reader knows whether it wants prompt tokens, generated tokens, or both.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

const char * const TOPK_PREFIX = "ffn_moe_topk-";

// One expert's slices of the file, loaded from tools/trace/expert_map.py.
struct expert_ranges {
    std::vector<std::pair<long long, long long>> slices;   // offset, bytes
};

struct trace_state {
    FILE * out        = nullptr;
    int    decode     = 0;
    long   lines      = 0;
    long   tensors    = 0;   // how many topk tensors were offered to us
    bool   warned_type = false;

    // -- assist mode (off unless an expert map is given) ---------------
    int    model_fd   = -1;
    int    horizon    = 0;   // how many layers ahead to advise
    long   advised    = 0;   // fadvise calls issued
    long long advised_bytes = 0;
    std::map<std::pair<int,int>, expert_ranges> map;       // (layer, expert)
    std::map<int, std::vector<int>> last_token;            // layer -> experts

    bool assisting() const { return model_fd >= 0 && horizon > 0 && !map.empty(); }

    // Ask the kernel to fetch what the next layers are likely to want.
    //
    // The guess is the plainest one the measurement supports: whatever those
    // layers routed to for the previous token. Consecutive tokens share 38%
    // of their experts on this model, so roughly three of every eight slabs
    // this fetches will be used, and the rest cost bandwidth and nothing
    // else. POSIX_FADV_WILLNEED queues readahead and returns, so the
    // callback does not block the layer it was called from.
    void advise_ahead(int layer) {
        for (int d = 1; d <= horizon; d++) {
            auto it = last_token.find(layer + d);
            if (it == last_token.end()) continue;
            for (int e : it->second) {
                auto r = map.find({layer + d, e});
                if (r == map.end()) continue;
                for (const auto & sl : r->second.slices) {
                    posix_fadvise(model_fd, (off_t) sl.first, (off_t) sl.second,
                                  POSIX_FADV_WILLNEED);
                    advised++;
                    advised_bytes += sl.second;
                }
            }
        }
    }
};

// Reads the flat table expert_map.py writes. Returns false on anything it
// cannot read, rather than assisting from a half-loaded map.
bool load_expert_map(const char * path, trace_state & st) {
    FILE * f = std::fopen(path, "r");
    if (!f) {
        std::fprintf(stderr, "tierinfer-trace: cannot read %s\n", path);
        return false;
    }
    char line[512];
    long n = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int layer, expert;
        long long off, bytes;
        if (std::sscanf(line, "%d %d %lld %lld", &layer, &expert, &off, &bytes) != 4) {
            std::fprintf(stderr, "tierinfer-trace: malformed expert map line: %s", line);
            std::fclose(f);
            return false;
        }
        st.map[{layer, expert}].slices.push_back({off, bytes});
        n++;
    }
    std::fclose(f);
    std::fprintf(stderr, "tierinfer-trace: expert map has %ld ranges over %zu experts\n",
                 n, st.map.size());
    return n > 0;
}

// Called twice per node: once to ask whether we want it, once with the data.
bool on_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<trace_state *>(user_data);

    if (ask) {
        // Returning false for everything else keeps this cheap: the scheduler
        // only materialises what we say yes to.
        return std::strncmp(t->name, TOPK_PREFIX, std::strlen(TOPK_PREFIX)) == 0;
    }

    st->tensors++;

    const int layer = std::atoi(t->name + std::strlen(TOPK_PREFIX));
    const int64_t n_used   = t->ne[0];          // experts routed per token
    const int64_t n_tokens = t->ne[1];          // tokens in this ubatch

    if (t->type != GGML_TYPE_I32) {
        if (!st->warned_type) {
            std::fprintf(stderr, "tierinfer-trace: %s has type %d, expected I32; "
                                 "not guessing at its layout\n", t->name, (int) t->type);
            st->warned_type = true;
        }
        return true;
    }

    // The tensor may live on a device; ggml_backend_tensor_get moves it here.
    std::vector<int32_t> buf((size_t) (n_used * n_tokens));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(int32_t));

    std::fprintf(st->out,
                 "{\"decode\":%d,\"layer\":%d,\"n_tokens\":%lld,\"n_used\":%lld,\"experts\":[",
                 st->decode, layer, (long long) n_tokens, (long long) n_used);
    for (int64_t i = 0; i < n_tokens; i++) {
        std::fputc(i ? ',' : '[', st->out);
        if (i) std::fputc('[', st->out);
        for (int64_t j = 0; j < n_used; j++) {
            if (j) std::fputc(',', st->out);
            std::fprintf(st->out, "%d", buf[(size_t) (i * n_used + j)]);
        }
        std::fputc(']', st->out);
    }
    std::fprintf(st->out, "]}\n");
    st->lines++;

    if (st->assisting()) {
        st->advise_ahead(layer);
        // Remember this layer's routing for the next token's guess. The last
        // token to pass through a layer is the one whose routing is kept,
        // which for a prompt decode means the final prompt token.
        auto & slot = st->last_token[layer];
        slot.clear();
        const int64_t last = n_tokens - 1;
        for (int64_t j = 0; j < n_used; j++) {
            slot.push_back(buf[(size_t) (last * n_used + j)]);
        }
    }
    return true;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL.gguf [-p PROMPT] [-n TOKENS] [-ngl N] [-t THREADS]\n"
        "          [-c CTX] [-o TRACE.jsonl] [--expert-map FILE --horizon N]\n\n"
        "Writes one JSONL line per decode per MoE layer to -o (default stdout).\n"
        "With --expert-map and --horizon > 0 it also advises the page cache for\n"
        "the next N layers, guessing from the previous token's routing.\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, prompt = "Explain what a mixture-of-experts layer does.";
    std::string out_path, map_path;
    int n_predict = 16, n_gpu_layers = 0, n_threads = 8, n_ctx = 2048, horizon = 0;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "tierinfer-trace: %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (a == "-m")   model_path = next("-m");
        else if (a == "-p")   prompt     = next("-p");
        else if (a == "-o")   out_path   = next("-o");
        else if (a == "-n")   n_predict  = std::atoi(next("-n"));
        else if (a == "-ngl") n_gpu_layers = std::atoi(next("-ngl"));
        else if (a == "-t")   n_threads  = std::atoi(next("-t"));
        else if (a == "-c")   n_ctx      = std::atoi(next("-c"));
        else if (a == "--expert-map") map_path = next("--expert-map");
        else if (a == "--horizon")    horizon  = std::atoi(next("--horizon"));
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else {
            std::fprintf(stderr, "tierinfer-trace: unknown argument %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (model_path.empty()) { usage(argv[0]); return 2; }

    trace_state st;
    st.out = out_path.empty() ? stdout : std::fopen(out_path.c_str(), "w");
    if (!st.out) {
        std::fprintf(stderr, "tierinfer-trace: cannot write %s\n", out_path.c_str());
        return 1;
    }

    if (!map_path.empty()) {
        if (horizon <= 0) {
            std::fprintf(stderr, "tierinfer-trace: --expert-map without --horizon does "
                                 "nothing; pass --horizon N to assist\n");
            return 2;
        }
        if (!load_expert_map(map_path.c_str(), st)) return 1;
        st.model_fd = open(model_path.c_str(), O_RDONLY);
        if (st.model_fd < 0) {
            std::fprintf(stderr, "tierinfer-trace: cannot open the model for advising\n");
            return 1;
        }
        st.horizon = horizon;
        std::fprintf(stderr, "tierinfer-trace: assisting %d layers ahead\n", horizon);
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "tierinfer-trace: failed to load %s\n", model_path.c_str());
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = n_ctx;
    cp.n_batch           = n_ctx;
    cp.n_threads         = n_threads;
    cp.n_threads_batch   = n_threads;
    cp.cb_eval           = on_eval;
    cp.cb_eval_user_data = &st;
    // A warmup run would route tokens nobody asked about into the trace.
    cp.op_offload        = false;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "tierinfer-trace: failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens(prompt.size() + 8);
    int n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                               tokens.data(), (int32_t) tokens.size(), true, true);
    if (n_tok < 0) {
        tokens.resize(-n_tok);
        n_tok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                               tokens.data(), (int32_t) tokens.size(), true, true);
    }
    if (n_tok <= 0) {
        std::fprintf(stderr, "tierinfer-trace: the prompt tokenised to nothing\n");
        return 1;
    }
    tokens.resize(n_tok);
    std::fprintf(stderr, "tierinfer-trace: %d prompt tokens, generating %d\n",
                 n_tok, n_predict);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tok)) != 0) {
        std::fprintf(stderr, "tierinfer-trace: prompt decode failed\n");
        return 1;
    }

    for (int i = 0; i < n_predict; i++) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            std::fprintf(stderr, "tierinfer-trace: end of generation at token %d\n", i);
            break;
        }
        st.decode++;
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) {
            std::fprintf(stderr, "tierinfer-trace: decode failed at token %d\n", i);
            break;
        }
    }

    std::fprintf(stderr, "tierinfer-trace: wrote %ld lines from %ld routing tensors\n",
                 st.lines, st.tensors);
    if (st.assisting()) {
        std::fprintf(stderr, "tierinfer-trace: advised %ld ranges, %.2f GB\n",
                     st.advised, (double) st.advised_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    if (st.lines == 0) {
        std::fprintf(stderr, "tierinfer-trace: no routing was captured — either this "
                             "model has no MoE layers, or the tensor name changed\n");
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    if (st.model_fd >= 0) close(st.model_fd);
    if (st.out != stdout) std::fclose(st.out);
    return st.lines > 0 ? 0 : 3;
}
