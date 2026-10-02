// hidden_ref — write llama.cpp mainline's final-norm hidden state of every prompt position (`result_norm`) as
// raw f32, in dump_ref's set format, so crates/refset reads it through its node-dump family check.
//
// What it writes. One prompt of token ids (from a file, one decimal id per line, the first --tokens-count of them)
// runs through one llama_decode with embeddings on and no pooling, every position an output. For such a context
// llama_get_embeddings_ith(ctx, i) is row i of the graph's `t_embd`, and the qwen35 builder sets `t_embd` to the
// last layer's output RMS-normed by `output_norm` (src/models/qwen35.cpp: `build_norm(cur, model.output_norm, ...)`,
// named `h_nextn`, then `cb(cur, "result_norm", -1); res->t_embd = cur;`), no get_rows in between when every position
// is an output, so the node keeps the name `h_nextn`. The eval callback counts the graph's final-norm nodes — the
// MUL whose second source is `output_norm.weight` — and the name and shape of the last, without asking for any
// node's data (the ask phase answers false), so the schedule is the one the context runs without a callback; the
// header records the count and the name. The set is one `tensor` row, `result_norm` occurrence 0, ne0 the model's width and ne1 the
// positions, and its file `result_norm.0.f32`: position t's row at t·ne0, row-major.
//
// The header carries what the node-dump reader checks: `# model` (the path as given, the family's identity),
// `# arch` (the file's general.architecture), `# build` (BLOOMERY_REF_BUILD, the mainline commit the runner
// names), the tokens file with its sha256 (BLOOMERY_REF_TOKENS_SHA256) and count, the flags, and the
// `# complete` trailer, written only after every file is. The manifest goes to a .partial name and is renamed
// last, so a run that dies leaves the previous set as it was.
//
//   hidden_ref -m <gguf> --tokens-file <ids> --tokens-count <n> --out <dir> [-c <ctx>] [-ub <ubatch>] [-ngl <n>]
//              [-t <threads>]
//
// `-t` sets the context's threads (llama.cpp's default otherwise): the CPU twin of a set, run with no card in view
// (CUDA_VISIBLE_DEVICES empty), is another 8-bit realization of the same rule, the oracle's own floor.
//
// Build: tools/ref/build-hidden.sh   Run: tools/ref/hidden.sh (just dump-hidden-qwen35)

#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

struct seen_ctx {
    int         nodes = 0;   // final-norm nodes: MUL by output_norm.weight
    int64_t     ne0   = 0;   // the last one's shape
    int64_t     ne1   = 0;
    long long   asked = 0;   // every node the scheduler asked about
    std::string last;        // the last one's name and op, for the refusal's message
    std::string name;        // the last final-norm node's name
};

// The ask phase only: count the final-norm nodes and answer false, so no node's data is copied out and the
// scheduler runs the graph as it would with no callback.
static bool on_node(struct ggml_tensor * t, bool ask, void * user) {
    auto * s = static_cast<seen_ctx *>(user);
    if (ask) {
        s->asked++;
        s->last = std::string(t->name) + " (" + ggml_op_desc(t) + ")";
    }
    if (ask && t->op == GGML_OP_MUL && t->src[1] && strcmp(t->src[1]->name, "output_norm.weight") == 0) {
        s->nodes++;
        s->name = t->name;
        s->ne0 = t->ne[0];
        s->ne1 = t->ne[1];
    }
    return false;
}

static bool read_token_file(const std::string & path, long long count, std::vector<llama_token> & ids) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "hidden_ref: cannot read %s\n", path.c_str());
        return false;
    }
    std::string line;
    long long   lines = 0;
    while ((long long) ids.size() < count && std::getline(in, line)) {
        ++lines;
        char * end = nullptr;
        errno = 0;
        const long v = strtol(line.c_str(), &end, 10);
        if (line.empty() || *end != '\0' || errno != 0 || v < 0 || v > INT32_MAX) {
            fprintf(stderr, "hidden_ref: %s:%lld is not a token id: '%s'\n", path.c_str(), lines, line.c_str());
            return false;
        }
        ids.push_back((llama_token) v);
    }
    if ((long long) ids.size() < count) {
        fprintf(stderr, "hidden_ref: %s holds %zu token ids and --tokens-count asks for %lld\n", path.c_str(),
                ids.size(), count);
        return false;
    }
    return true;
}

static bool write_raw(const std::string & path, const float * data, size_t count) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "hidden_ref: cannot write %s\n", path.c_str());
        return false;
    }
    const bool ok = fwrite(data, sizeof(float), count, f) == count;
    if (fclose(f) != 0 || !ok) {
        fprintf(stderr, "hidden_ref: short write on %s\n", path.c_str());
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, tokens_file, out;
    long long   count = -1;
    int         n_ctx = 0, n_ubatch = 512, ngl = 99, threads = 0;
    std::string flags;
    for (int i = 1; i < argc; ++i) {
        flags += (i > 1 ? " " : "") + std::string(argv[i]);
    }
    for (int i = 1; i < argc; ++i) {
        const bool more = i + 1 < argc;
        if (strcmp(argv[i], "-m") == 0 && more) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--tokens-file") == 0 && more) {
            tokens_file = argv[++i];
        } else if (strcmp(argv[i], "--tokens-count") == 0 && more) {
            count = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && more) {
            out = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && more) {
            n_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ub") == 0 && more) {
            n_ubatch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 && more) {
            ngl = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && more) {
            threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "hidden_ref: unknown or incomplete argument '%s'\n", argv[i]);
            return 2;
        }
    }
    if (model_path.empty() || tokens_file.empty() || out.empty() || count <= 0 || n_ubatch <= 0) {
        fprintf(stderr, "usage: hidden_ref -m <gguf> --tokens-file <ids> --tokens-count <n> --out <dir> "
                        "[-c <ctx>] [-ub <ubatch>] [-ngl <n>] [-t <threads>]\n");
        return 2;
    }
    std::vector<llama_token> tokens;
    if (!read_token_file(tokens_file, count, tokens)) return 2;
    const int n = (int) tokens.size();
    if (n_ctx == 0) n_ctx = n;
    if (n_ctx < n) {
        fprintf(stderr, "hidden_ref: %d tokens do not fit in -c %d\n", n, n_ctx);
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers       = ngl;
    llama_model * model   = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "hidden_ref: failed to load %s\n", model_path.c_str());
        return 1;
    }
    seen_ctx seen;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = (uint32_t) n_ctx;
    cp.n_batch           = (uint32_t) n;
    cp.n_ubatch          = (uint32_t) std::min(n_ubatch, n);
    cp.n_seq_max         = 1;
    cp.embeddings        = true;
    cp.pooling_type      = LLAMA_POOLING_TYPE_NONE;
    cp.cb_eval           = on_node;
    cp.cb_eval_user_data = &seen;
    if (threads > 0) {
        cp.n_threads       = threads;
        cp.n_threads_batch = threads;
    }
    llama_context * ctx  = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "hidden_ref: failed to make the context\n");
        return 1;
    }
    const int width = llama_model_n_embd(model);

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "hidden_ref: decode failed\n");
        return 1;
    }
    if (seen.nodes == 0 || seen.ne0 != width) {
        fprintf(stderr, "hidden_ref: the graph ran %d final-norm node(s), the last %lld wide; the model is %d "
                        "(%lld nodes asked, the last %s)\n",
                seen.nodes, (long long) seen.ne0, width, seen.asked, seen.last.c_str());
        return 1;
    }
    std::vector<float> rows((size_t) n * width);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const float * e = llama_get_embeddings_ith(ctx, i);
        if (!e) {
            fprintf(stderr, "hidden_ref: no embedding row for position %d\n", i);
            return 1;
        }
        for (int j = 0; j < width; ++j) {
            if (!std::isfinite(e[j])) {
                fprintf(stderr, "hidden_ref: position %d value %d is not finite\n", i, j);
                return 1;
            }
            rows[(size_t) i * width + j] = e[j];
            sum += e[j];
        }
    }

    mkdir(out.c_str(), 0755);
    if (!write_raw(out + "/result_norm.0.f32", rows.data(), rows.size())) return 1;
    const std::string final_path   = out + "/MANIFEST.tsv";
    const std::string partial_path = final_path + ".partial";
    FILE * m = fopen(partial_path.c_str(), "w");
    if (!m) {
        fprintf(stderr, "hidden_ref: cannot write %s\n", partial_path.c_str());
        return 1;
    }
    char arch[128];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) {
        snprintf(arch, sizeof(arch), "unknown");
    }
    const char * slash = strrchr(model_path.c_str(), '/');
    const char * build = getenv("BLOOMERY_REF_BUILD");
    const char * sha   = getenv("BLOOMERY_REF_TOKENS_SHA256");
    fprintf(m, "# hidden_ref — llama.cpp mainline result_norm of every position, raw f32, little-endian\n");
    fprintf(m, "# model\t%s\n", model_path.c_str());
    if (build && *build) fprintf(m, "# build\t%s\n", build);
    fprintf(m, "# arch\t%s\n", arch);
    fprintf(m, "# model_file\t%s\n", slash ? slash + 1 : model_path.c_str());
    fprintf(m, "# flags\t%s\n", flags.c_str());
    fprintf(m, "# tokens_file\t%s\n", tokens_file.c_str());
    fprintf(m, "# tokens_file_sha256\t%s\n", sha && *sha ? sha : "unknown");
    fprintf(m, "# tokens_count\t%d\n", n);
    fprintf(m, "# source\tllama_get_embeddings_ith over %d positions (t_embd, result_norm); the graph ran %d "
               "final-norm node(s) (MUL by output_norm.weight), the last %s [%lld, %lld]\n", n, seen.nodes,
            seen.name.c_str(), (long long) seen.ne0, (long long) seen.ne1);
    fprintf(m, "# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n");
    fprintf(m, "tensor\tresult_norm\t0\tf32\t%d\t%d\t1\t1\t%zu\t%.6f\tRMS_NORM\t1\t0\t-\t-\n", width, n,
            rows.size() * sizeof(float), sum);
    fprintf(m, "# complete\t1\t0\n");
    if (fclose(m) != 0 || rename(partial_path.c_str(), final_path.c_str()) != 0) {
        fprintf(stderr, "hidden_ref: cannot install %s\n", final_path.c_str());
        return 1;
    }
    fprintf(stderr, "hidden_ref: %d positions x %d into %s (%d final-norm node(s), the last %s)\n", n, width,
            out.c_str(), seen.nodes, seen.name.c_str());
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
