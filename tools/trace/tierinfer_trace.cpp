// Capture a MoE routing trace from a real model, without changing llama.cpp.
//
// Everything else in this project reasons about which experts a token needs.
// This gets the real thing. No patch is required: llama.cpp names the top-k
// selection tensor "ffn_moe_topk-<layer>" (llama_context::graph_get_cb formats
// every callback name that way) and exposes ggml_backend_sched_eval_callback
// through llama_context_params::cb_eval. So a trace is a filter on tensor
// names plus a read of int32 data; the inference path is untouched, and with
// the tool not running there is nothing to turn off. That is the observability
// mode of the scope, and the clean baseline is the default rather than a mode.
//
// Output is JSONL, one line per decode per layer:
//     {"decode":0,"layer":3,"n_tokens":11,"n_used":8,"experts":[[..],[..]]}
// Assembling per-token routings from that is the reader's job
// (`tierinfer.trace`), because the reader knows whether it wants prompt
// tokens, generated tokens, or both.
//
// Two things this version does that the first did not, both found by the
// 2026-09-18 audit:
//
//   * It reads the routing tensor **row by row through its strides**. From
//     llama.cpp b10482 `ffn_moe_topk` is `ggml_argsort_top_k`, a *view* of the
//     argsort with row stride n_expert*4 bytes, not the contiguous
//     GGML_OP_TOP_K tensor b9888 produced. A contiguous read of n_used*n_tokens
//     ints from that view returns the right experts for the first token and
//     the argsort's ranks 8..15, 16..23, ... for the rest — plausible numbers,
//     all in range, all wrong. Reading through nb[1] is right for both builds.
//
//   * The "assist" mode (posix_fadvise WILLNEED/DONTNEED from inside the
//     callback) is gone. `benchmarks/REAL-ROUTING.md` measured it three ways:
//     616.9 GB of DONTNEED moved residency by nothing, because a live mapping
//     holds its pages. Flags that issue syscalls which cannot achieve what
//     they are named for are flags that do not affect execution, and the
//     audit removed them. The measurement stays in benchmarks/.
//
// `--cpu-moe` keeps the routed experts on the CPU while everything else goes
// to the GPU under -ngl, the same override `llama-cli -cmoe` applies. Routing
// does not depend on placement, so this only makes a trace of a very large
// model take minutes instead of hours.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char * const TOPK_PREFIX = "ffn_moe_topk-";

struct trace_state {
    FILE * out        = nullptr;
    int    decode     = 0;
    long   lines      = 0;
    long   tensors    = 0;   // how many topk tensors were offered to us
    bool   warned_type = false;
    bool   warned_shape = false;
};

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
    if (t->ne[2] != 1 || t->ne[3] != 1) {
        if (!st->warned_shape) {
            std::fprintf(stderr, "tierinfer-trace: %s has shape [%lld,%lld,%lld,%lld]; "
                                 "only two dimensions are understood, refusing\n", t->name,
                         (long long) t->ne[0], (long long) t->ne[1],
                         (long long) t->ne[2], (long long) t->ne[3]);
            st->warned_shape = true;
        }
        return true;
    }

    // One row per token, read at the row's own byte offset. The tensor may
    // live on a device and may be a strided view; ggml_backend_tensor_get
    // copies raw bytes at an offset, so the stride has to be ours to apply.
    std::vector<int32_t> row((size_t) n_used);
    std::fprintf(st->out,
                 "{\"decode\":%d,\"layer\":%d,\"n_tokens\":%lld,\"n_used\":%lld,\"experts\":[",
                 st->decode, layer, (long long) n_tokens, (long long) n_used);
    for (int64_t i = 0; i < n_tokens; i++) {
        ggml_backend_tensor_get(t, row.data(), (size_t) (i * t->nb[1]),
                                row.size() * sizeof(int32_t));
        std::fputc(i ? ',' : '[', st->out);
        if (i) std::fputc('[', st->out);
        for (int64_t j = 0; j < n_used; j++) {
            if (j) std::fputc(',', st->out);
            std::fprintf(st->out, "%d", row[(size_t) j]);
        }
        std::fputc(']', st->out);
    }
    std::fprintf(st->out, "]}\n");
    st->lines++;
    return true;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s -m MODEL.gguf [-p PROMPT] [-f PROMPT_FILE] [-n TOKENS] [-ngl N]\n"
        "          [--cpu-moe] [-t THREADS] [-c CTX] [-b BATCH] [-o TRACE.jsonl]\n\n"
        "Writes one JSONL line per decode per MoE layer to -o (default stdout).\n"
        "--cpu-moe   keep the routed expert tensors on the CPU whatever -ngl says\n"
        "--one-by-one  decode the prompt one token per llama_decode call, so its\n"
        "            routing can be compared with a batched decode of the same prompt\n"
        "Timings (load, prompt, generation) go to stderr.\n",
        argv0);
}

std::string read_file(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "tierinfer-trace: cannot read %s\n", path.c_str()); std::exit(2); }
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, prompt = "Explain what a mixture-of-experts layer does.";
    std::string out_path;
    int n_predict = 16, n_gpu_layers = 0, n_threads = 8, n_ctx = 2048, n_batch = 0;
    bool cpu_moe = false, one_by_one = false;

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
        else if (a == "-f")   prompt     = read_file(next("-f"));
        else if (a == "-o")   out_path   = next("-o");
        else if (a == "-n")   n_predict  = std::atoi(next("-n"));
        else if (a == "-ngl") n_gpu_layers = std::atoi(next("-ngl"));
        else if (a == "-t")   n_threads  = std::atoi(next("-t"));
        else if (a == "-c")   n_ctx      = std::atoi(next("-c"));
        else if (a == "-b")   n_batch    = std::atoi(next("-b"));
        else if (a == "--cpu-moe") cpu_moe = true;
        else if (a == "--one-by-one") one_by_one = true;
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

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    // Same pattern llama-cli's -cmoe uses: the fused expert tensors stay on
    // the CPU buffer type, everything else follows -ngl.
    std::vector<llama_model_tensor_buft_override> overrides;
    if (cpu_moe) {
        overrides.push_back({ "\\.ffn_(up|down|gate)_exps", ggml_backend_cpu_buffer_type() });
        overrides.push_back({ nullptr, nullptr });
        mp.tensor_buft_overrides = overrides.data();
    }

    using clock = std::chrono::steady_clock;
    const auto t_load0 = clock::now();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "tierinfer-trace: failed to load %s\n", model_path.c_str());
        return 1;
    }
    const double load_s = std::chrono::duration<double>(clock::now() - t_load0).count();

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = n_ctx;
    cp.n_batch           = n_batch > 0 ? n_batch : n_ctx;
    cp.n_threads         = n_threads;
    cp.n_threads_batch   = n_threads;
    cp.cb_eval           = on_eval;
    cp.cb_eval_user_data = &st;

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
    std::fprintf(stderr, "tierinfer-trace: model loaded in %.1f s; %d prompt tokens, generating %d\n",
                 load_s, n_tok, n_predict);

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // Fed one token at a time, every prompt token is its own decode (the
    // reader requires each layer once per decode), so a --one-by-one trace
    // numbers its prompt tokens 0..n-1 and its generated tokens after them.
    // Its "prompt"/"generated" split as the reader sees it is therefore not
    // meaningful; the mode exists to compare routings position by position
    // against a batched decode of the same prompt, nothing else.
    const auto t_prompt0 = clock::now();
    if (one_by_one) {
        for (int i = 0; i < n_tok; i++) {
            if (i) st.decode++;
            if (llama_decode(ctx, llama_batch_get_one(&tokens[(size_t) i], 1)) != 0) {
                std::fprintf(stderr, "tierinfer-trace: prompt decode failed at token %d\n", i);
                return 1;
            }
        }
    } else if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tok)) != 0) {
        std::fprintf(stderr, "tierinfer-trace: prompt decode failed\n");
        return 1;
    }
    const double prompt_s = std::chrono::duration<double>(clock::now() - t_prompt0).count();

    const auto t_gen0 = clock::now();
    double first_token_s = 0.0;
    int generated = 0;
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
        generated++;
        if (generated == 1) first_token_s = std::chrono::duration<double>(clock::now() - t_gen0).count();
    }
    const double gen_s = std::chrono::duration<double>(clock::now() - t_gen0).count();

    std::fprintf(stderr, "tierinfer-trace: wrote %ld lines from %ld routing tensors\n",
                 st.lines, st.tensors);
    std::fprintf(stderr, "tierinfer-trace: timing load=%.2fs prompt=%d tokens in %.2fs (%.2f t/s) "
                         "generation=%d tokens in %.2fs (%.2f t/s) first_token=%.3fs\n",
                 load_s, n_tok, prompt_s, prompt_s > 0 ? n_tok / prompt_s : 0.0,
                 generated, gen_s, gen_s > 0 ? generated / gen_s : 0.0, first_token_s);
    llama_perf_context_print(ctx);
    if (st.lines == 0) {
        std::fprintf(stderr, "tierinfer-trace: no routing was captured — either this "
                             "model has no MoE layers, or the tensor name changed\n");
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    if (st.out != stdout) std::fclose(st.out);
    return st.lines > 0 ? 0 : 3;
}
