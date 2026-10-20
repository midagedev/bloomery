// solar2-dump — the oracle of the Solar-Open2 gates: the PAI tree's forward pass (llama.cpp 6ea215d17 plus
// prometheusAIR's solar_open2 patch, the only faithful implementation: it applies beta = 2 sigmoid) dumped as
// dump_ref-format sets (tools/ref/dump_ref.cpp: MANIFEST.tsv, one raw f32 file a node and occurrence, the
// completion trailer), one directory per prompt under --out, from ONE model load.
//
// A port of dump_ref to the mainline `cb_eval` API (llama_context_params.cb_eval), cut to what the Solar gates read:
//  - the nodes a prompt names (--prompt NAME:IDS:NODES:GREEDY, NODES joined by `+`): a name selects the node of that
//    exact name and `<name>-<layer>`, and a node is written whole (every position of its ubatch) at the callback's
//    second call. The graph names are the patch's own `cb()` names: model.embed_tokens, attn_norm, kda_g1 (the
//    decay, -exp(A_log) softplus(f_b f_a x + dt_bias)), kda_beta_neg_eigval (2 sigmoid), kda_out, attn_out (the
//    softmax layer's output before its gate), attn_gate_sigmoid, attn_gated, attn_proj, ffn_inp, ffn_norm,
//    ffn_moe_out, ffn_shexp, ffn_out, l_out, result_norm, result_output;
//  - `result_output` is the last position's whole logits row (the only row the batch asks for);
//  - topk.tsv: the last position's 40 largest logits and their log-softmax in f64 (rank, id, logit, logprob),
//    ties to the lower id;
//  - greedy.tsv: GREEDY argmax tokens continued from the prompt, each with the margin between its logit and the
//    runner-up's (logit units), the first being the prompt's own argmax. The continuation runs without the callback.
// Tokens come in as integers (one id a line), never as text: the tokenizer is not what the gates test.
//
// CPU only (the runner hides CUDA and this tool asks for no GPU layers): the reference is a CPU reduction order,
// as dump.sh's is. n_batch 2048 and n_ubatch 512, the ubatch the card engine's prompt unit (T_MAX) is.
//
// Build: tools/ref/solar2-dump-build.sh   Run: tools/ref/solar2-dump.sh

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "ref_io.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>

struct Prompt {
    std::string              name;
    std::vector<llama_token> ids;
    std::vector<std::string> nodes;
    int                      greedy = 0;
};

struct Dump {
    std::string                dir;
    FILE *                     manifest = nullptr;
    std::vector<std::string>   nodes;       // the prompt's node names
    std::map<std::string, int> seen;        // node name -> times emitted: the occurrence index
    std::vector<uint8_t>       staging;     // a device tensor lands here before the write
    bool                       quiet  = true;  // no node is asked for outside the prompt's decode
    bool                       failed = false;
    int                        written = 0;
    int                        skipped = 0;
};

// `name` is `node` or `node-<digits>`.
static bool is_node(const std::string & node, const char * name) {
    const size_t n = node.size();
    if (strncmp(name, node.c_str(), n) != 0) return false;
    const char * rest = name + n;
    if (*rest == '\0') return true;
    if (*rest++ != '-' || *rest == '\0') return false;
    for (; *rest; ++rest) {
        if (*rest < '0' || *rest > '9') return false;
    }
    return true;
}

static bool wanted(const Dump & d, const char * name) {
    for (const std::string & n : d.nodes) {
        if (is_node(n, name)) return true;
    }
    return false;
}

// One element as f32, exactly: f32, f16 and bf16 only (the Solar graph's activations). False for any other type.
static bool read_f32(ggml_type type, const uint8_t * p, float & f) {
    switch (type) {
        case GGML_TYPE_F32:  { float x;          memcpy(&x, p, sizeof x); f = x;                    return true; }
        case GGML_TYPE_F16:  { ggml_fp16_t x;    memcpy(&x, p, sizeof x); f = ggml_fp16_to_fp32(x); return true; }
        case GGML_TYPE_BF16: { ggml_bf16_t x;    memcpy(&x, p, sizeof x); f = ggml_bf16_to_fp32(x); return true; }
        default:             return false;
    }
}

static std::string safe_name(const char * name) {
    std::string s(name);
    for (char & c : s) {
        if (c == '/' || c == '\\' || c == ' ') c = '_';
    }
    return s;
}

// The file's elements in flat order (memory read contiguously) or in logical order (ggml index order through the
// tensor's own strides): dump_ref's two layouts, so a view or a non-contiguous node gets both files.
static void gather(const ggml_tensor * t, const uint8_t * src, bool logical, std::vector<float> & out) {
    const size_t esize = ggml_type_size(t->type);
    size_t li = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
        const size_t off = logical
            ? (size_t) i0 * t->nb[0] + (size_t) i1 * t->nb[1] + (size_t) i2 * t->nb[2] + (size_t) i3 * t->nb[3]
            : li * esize;
        read_f32(t->type, src + off, out[li]);
        ++li;
    }
}

static bool write_file(const std::string & path, const void * data, size_t size, size_t count) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "solar2-dump: cannot write %s\n", path.c_str());
        return false;
    }
    const bool ok = fwrite(data, size, count, f) == count;
    if (!ok) fprintf(stderr, "solar2-dump: short write on %s\n", path.c_str());
    return (fclose(f) == 0) && ok;
}

static const char * src_name(const ggml_tensor * s) { return s ? (s->name[0] ? s->name : "(unnamed)") : "-"; }

// Write one node: the flat f32 file, the logical twin of a view or non-contiguous tensor, and its manifest row. A
// quantized tensor or a type this tool does not read is a `skip` row, never a missing file read as a tensor that did not run.
static bool dump_one(Dump * d, const ggml_tensor * t) {
    const std::string name   = t->name[0] ? t->name : "(unnamed)";
    const int         occ    = d->seen[name]++;
    float             probe  = 0;
    const uint8_t     zero[8] = {};
    if (!read_f32(t->type, zero, probe) || !t->buffer || !t->data) {
        fprintf(d->manifest, "skip\t%s\t%d\t%s\t%s\n", name.c_str(), occ, ggml_type_name(t->type),
                ggml_is_quantized(t->type) ? "quantized" : (t->buffer && t->data ? "unhandled" : "unallocated"));
        d->skipped++;
        return true;
    }
    const size_t nbytes = ggml_nbytes(t);
    const uint8_t * src;
    if (ggml_backend_buffer_is_host(t->buffer)) {
        src = (const uint8_t *) t->data;
    } else {
        d->staging.resize(nbytes);
        ggml_backend_tensor_get(t, d->staging.data(), 0, nbytes);
        src = d->staging.data();
    }
    const size_t       n = (size_t) ggml_nelements(t);
    std::vector<float> out(n);
    gather(t, src, false, out);
    double sum = 0.0;
    for (float v : out) sum += v;
    const std::string stem = safe_name(name.c_str()) + "." + std::to_string(occ);
    if (!write_file(d->dir + "/" + stem + ".f32", out.data(), sizeof(float), n)) return false;
    const bool contig  = ggml_is_contiguous(t);
    const bool logical = !contig || t->view_src != nullptr;
    if (logical) {
        std::vector<float> lo(n);
        gather(t, src, true, lo);
        if (!write_file(d->dir + "/" + stem + ".logical.f32", lo.data(), sizeof(float), n)) return false;
    }
    fprintf(d->manifest, "tensor\t%s\t%d\t%s\t%lld\t%lld\t%lld\t%lld\t%zu\t%.6f\t%s\t%d\t%d\t%s\t%s\n", name.c_str(), occ,
            ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
            (long long) t->ne[3], n * sizeof(float), sum, ggml_op_desc(t), contig ? 1 : 0, logical ? 1 : 0,
            src_name(t->src[0]), src_name(t->src[1]));
    d->written++;
    return true;
}

static bool on_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    Dump * d = (Dump *) user_data;
    // An empty node (a ubatch with no output row asks the head for zero rows) is no row of the oracle.
    if (d->quiet || d->failed || ggml_nelements(t) == 0 || !wanted(*d, t->name)) return false;
    if (ask) return true;
    if (!dump_one(d, t)) d->failed = true;
    return !d->failed;
}

static bool read_ids(const std::string & path, std::vector<llama_token> & ids) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "solar2-dump: cannot read %s\n", path.c_str());
        return false;
    }
    std::string line;
    long long lines = 0;
    while (std::getline(in, line)) {
        ++lines;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        int32_t id;
        if (line.empty() || !parse_token_id(line, id)) {
            fprintf(stderr, "%s:%lld: invalid token id '%s'\n", path.c_str(), lines, line.c_str());
            return false;
        }
        ids.push_back((llama_token) id);
    }
    if (ids.empty()) fprintf(stderr, "solar2-dump: %s holds no ids\n", path.c_str());
    return !ids.empty();
}

// NAME:IDS:NODES:GREEDY, NODES joined by `+` (empty: logits only through the tool's own result_output).
static bool parse_prompt(const std::string & spec, Prompt & p) {
    std::vector<std::string> f;
    size_t from = 0;
    for (size_t at; (at = spec.find(':', from)) != std::string::npos; from = at + 1) f.push_back(spec.substr(from, at - from));
    f.push_back(spec.substr(from));
    if (f.size() != 4 || f[0].empty()) {
        fprintf(stderr, "solar2-dump: --prompt wants NAME:IDS:NODES:GREEDY, got %s\n", spec.c_str());
        return false;
    }
    p.name = f[0];
    long long g = 0;
    if (!parse_i64("solar2-dump", "--prompt GREEDY", f[3].c_str(), 0, 4096, g)) return false;
    p.greedy = (int) g;
    for (size_t from2 = 0; from2 <= f[2].size();) {
        size_t at = f[2].find('+', from2);
        if (at == std::string::npos) at = f[2].size();
        if (at > from2) p.nodes.push_back(f[2].substr(from2, at - from2));
        from2 = at + 1;
    }
    p.nodes.push_back("result_output");
    return read_ids(f[1], p.ids);
}

struct Top {
    int    id;
    float  logit;
    double logprob;
};

// The k largest logits of a row with their f64 log-softmax; ties go to the lower id.
static std::vector<Top> top_of(const float * row, int n_vocab, int k) {
    double mx = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) mx = std::max(mx, (double) row[i]);
    double se = 0;
    for (int i = 0; i < n_vocab; ++i) se += std::exp((double) row[i] - mx);
    const double       lse = mx + std::log(se);
    std::vector<int>   idx(n_vocab);
    for (int i = 0; i < n_vocab; ++i) idx[i] = i;
    const size_t nt = std::min<size_t>((size_t) k, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + nt, idx.end(), [&](int a, int b) {
        return row[a] != row[b] ? row[a] > row[b] : a < b;
    });
    std::vector<Top> out;
    for (size_t r = 0; r < nt; ++r) out.push_back({idx[r], row[idx[r]], (double) row[idx[r]] - lse});
    return out;
}

static bool finite_row(const float * row, int n_vocab, const char * who) {
    if (!row) {
        fprintf(stderr, "solar2-dump: %s: no last-position logits\n", who);
        return false;
    }
    for (int i = 0; i < n_vocab; ++i) {
        if (!std::isfinite(row[i])) {
            fprintf(stderr, "solar2-dump: %s: logit %d is not finite\n", who, i);
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, out, fixture_line, build;
    int n_ctx = 4096, n_ubatch = 512, threads = 32;
    std::vector<Prompt> prompts;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has = i + 1 < argc;
        if (a == "-m" && has) { model_path = argv[++i]; }
        else if (a == "--out" && has) { out = argv[++i]; }
        else if (a == "--build" && has) { build = argv[++i]; }
        else if (a == "--fixture-line" && has) { fixture_line = argv[++i]; }
        else if (a == "-c" && has) { if (!parse_i32("solar2-dump", "-c", argv[++i], 1, n_ctx)) return 2; }
        else if (a == "-ub" && has) { if (!parse_i32("solar2-dump", "-ub", argv[++i], 1, n_ubatch)) return 2; }
        else if (a == "-t" && has) { if (!parse_i32("solar2-dump", "-t", argv[++i], 1, threads)) return 2; }
        else if (a == "--prompt" && has) {
            Prompt p;
            if (!parse_prompt(argv[++i], p)) return 2;
            prompts.push_back(std::move(p));
        } else {
            fprintf(stderr, "solar2-dump: unknown or incomplete argument %s\n", a.c_str());
            return 2;
        }
    }
    if (model_path.empty() || out.empty() || build.empty() || prompts.empty()) {
        fprintf(stderr, "solar2-dump: wants -m MODEL --out DIR --build BUILD and at least one --prompt\n");
        return 2;
    }
    size_t longest = 0;
    for (const Prompt & p : prompts) longest = std::max(longest, p.ids.size() + (size_t) p.greedy);
    const std::string flags = "-m " + model_path + " -c " + std::to_string(n_ctx) + " -ub " + std::to_string(n_ubatch) +
                              " -t " + std::to_string(threads) + " -b 2048 -ngl 0";
    if (longest > (size_t) n_ctx) {
        fprintf(stderr, "solar2-dump: a prompt with its continuation needs %zu positions, -c is %d\n", longest, n_ctx);
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "solar2-dump: failed to load %s\n", model_path.c_str());
        return 1;
    }
    char arch[128] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0 || strcmp(arch, "solar-open2") != 0) {
        fprintf(stderr, "solar2-dump: general.architecture is '%s', want solar-open2\n", arch);
        return 1;
    }
    Dump d;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = 2048;
    cp.n_ubatch = (uint32_t) n_ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &d;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "solar2-dump: failed to make the context\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (mkdir(out.c_str(), 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "solar2-dump: cannot create %s\n", out.c_str());
        return 1;
    }

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const Prompt & p = prompts[pi];
        if (pi > 0) llama_memory_clear(llama_get_memory(ctx), true);
        d = Dump();
        d.dir = out + "/" + p.name;
        d.nodes = p.nodes;
        mkdir(d.dir.c_str(), 0755);
        const std::string final_path = d.dir + "/MANIFEST.tsv", partial = final_path + ".partial";
        d.manifest = fopen(partial.c_str(), "w");
        if (!d.manifest) {
            fprintf(stderr, "solar2-dump: cannot write %s\n", partial.c_str());
            return 1;
        }
        fprintf(d.manifest, "# solar2-dump — the PAI tree's intermediate tensors, raw f32, little-endian\n");
        fprintf(d.manifest, "# model\t%s\n", model_path.c_str());
        if (!fixture_line.empty()) fprintf(d.manifest, "%s\n", fixture_line.c_str());
        fprintf(d.manifest, "# build\t%s\n", build.c_str());
        fprintf(d.manifest, "# arch\t%s\n", arch);
        const char * slash = strrchr(model_path.c_str(), '/');
        fprintf(d.manifest, "# model_file\t%s\n", slash ? slash + 1 : model_path.c_str());
        fprintf(d.manifest, "# tokens\t");
        for (size_t i = 0; i < p.ids.size(); ++i) fprintf(d.manifest, "%s%d", i ? "," : "", p.ids[i]);
        fprintf(d.manifest, "\n# flags\t%s\n", flags.c_str());
        fprintf(d.manifest, "# topk\ttopk.tsv\t40\n# greedy\tgreedy.tsv\t%d\n", p.greedy);
        fprintf(d.manifest, "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n");

        // The prompt, the callback on: every ubatch asks for its nodes.
        std::vector<llama_token> ids = p.ids;
        d.quiet = false;
        const int rc = llama_decode(ctx, llama_batch_get_one(ids.data(), (int) ids.size()));
        d.quiet = true;
        if (rc != 0 || d.failed) {
            fprintf(stderr, "solar2-dump: decode of %s failed (rc %d, failed %d)\n", p.name.c_str(), rc, (int) d.failed);
            return 1;
        }
        const float * row = llama_get_logits_ith(ctx, -1);
        if (!finite_row(row, n_vocab, p.name.c_str())) return 1;
        const std::vector<Top> top = top_of(row, n_vocab, 40);
        FILE * tk = fopen((d.dir + "/topk.tsv").c_str(), "w");
        if (!tk) return 1;
        for (size_t r = 0; r < top.size(); ++r) fprintf(tk, "%zu\t%d\t%.9g\t%.17g\n", r, top[r].id, top[r].logit, top[r].logprob);
        fclose(tk);

        // The greedy continuation, callback off. Step s is the argmax of the logits after the prompt plus the s tokens
        // before it; its margin is the logit gap to the runner-up.
        FILE * gr = fopen((d.dir + "/greedy.tsv").c_str(), "w");
        if (!gr) return 1;
        fprintf(gr, "# step\tid\tmargin\tlogprob\n");
        for (int s = 0; s < p.greedy; ++s) {
            const float * r = llama_get_logits_ith(ctx, -1);
            if (!finite_row(r, n_vocab, p.name.c_str())) return 1;
            const std::vector<Top> t2 = top_of(r, n_vocab, 2);
            fprintf(gr, "%d\t%d\t%.9g\t%.17g\n", s, t2[0].id, t2[0].logit - t2[1].logit, t2[0].logprob);
            if (s + 1 < p.greedy) {
                llama_token next = t2[0].id;
                if (llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) {
                    fprintf(stderr, "solar2-dump: greedy step %d of %s failed\n", s, p.name.c_str());
                    return 1;
                }
            }
        }
        if (fclose(gr) != 0) return 1;

        fprintf(d.manifest, "# complete\t%d\t%d\n", d.written, d.skipped);
        if (fclose(d.manifest) != 0 || rename(partial.c_str(), final_path.c_str()) != 0) {
            fprintf(stderr, "solar2-dump: cannot install %s\n", final_path.c_str());
            return 1;
        }
        d.manifest = nullptr;
        printf("solar2-dump: %s: %zu ids, argmax %d logprob %.6f, %d nodes written, %d skipped, %d greedy\n",
               p.name.c_str(), p.ids.size(), top[0].id, top[0].logprob, d.written, d.skipped, p.greedy);
        fflush(stdout);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
