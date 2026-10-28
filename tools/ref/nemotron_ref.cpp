// nemotron_ref — last-position logits of each prompt from llama.cpp mainline, on the CPU.
//
// One llama_decode per prompt, then llama_get_logits_ith(ctx, -1): the last logit row. Only that
// token is marked for output. The row is n_vocab f32, little-endian, plus the prompt's ids.
// Headers are # model, # build (BLOOMERY_REF_BUILD, the tokenizer family's spelling of the
// mainline md5s) and # arch. # complete is written only after every file, by renaming the manifest.
//
//   nemotron_ref -m <gguf> --arch <name> --out <dir> --prompt <name>:<n>:<ids> ... [-c <ctx>] [-ub <ubatch>] [-t <threads>]
//                [--fixture-line <line>]
//
// --arch is the general.architecture the file must state (nemotron_h, nemotron_h_moe or lfm3); a file that states another is
// refused. Nothing else in this harness is a nemotron fact: it decodes the ids it is given on whatever the mainline build loads.
// --fixture-line is the fixture file's `# fixture` line (refset-check --fixture-line), written verbatim beside # model.
// A file with routed experts also gets each MoE layer's selection at the last position: the graph's ffn_moe_topk-<il>
// tensor (the ids ggml_top_k picked over the biased probabilities) read through the eval callback, its last token's
// row; ffn_moe_probs-<il> gives the layer's expert count. One `# route layer <il> <n_expert> <k>` line per layer and a
// routes column (<name>.routes.i32, the layers in order). A file whose graph names no ffn_moe_topk writes neither.
// --prompt is repeated, once a prompt. <ids> is one decimal id a line, at least <n> of them; the
// first <n> are the prompt. n_gpu_layers is 0. The caller also empties CUDA_VISIBLE_DEVICES.

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "ref_io.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>

struct Prompt {
    std::string name;
    int n = 0;
    std::string path;
    std::vector<llama_token> ids;
};

static bool parse_prompt(const char * spec, Prompt & p) {
    const char * c1 = strchr(spec, ':');
    if (!c1 || c1 == spec) return false;
    const char * c2 = strchr(c1 + 1, ':');
    if (!c2 || c2 == c1 + 1 || c2[1] == '\0') return false;
    p.name.assign(spec, c1);
    for (char ch : p.name) {
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9'))) return false;
    }
    const std::string count(c1 + 1, c2);
    long long n = 0;
    if (!parse_i64("nemotron_ref", "--prompt count", count.c_str(), 1, 1000000, n)) return false;
    p.n = (int) n;
    p.path = c2 + 1;
    return true;
}

static bool write_ids(const std::string & path, const std::vector<llama_token> & ids) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "nemotron_ref: cannot write %s\n", path.c_str());
        return false;
    }
    for (llama_token id : ids) {
        if (fprintf(f, "%d\n", (int) id) < 0) {
            fclose(f);
            fprintf(stderr, "nemotron_ref: short write on %s\n", path.c_str());
            return false;
        }
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "nemotron_ref: short write on %s\n", path.c_str());
        return false;
    }
    return true;
}

// The last token's selection of each MoE layer in the last decode, and each layer's expert count. A ubatch overwrites
// the one before it, so after a prompt's decode each row is its last token's.
struct Routes {
    std::map<int, std::vector<int32_t>> ids;
    std::map<int, int64_t> n_expert;
    bool failed = false;
};

// The layer of a tensor named <stem>-<il>, or -1.
static int layer_of(const char * name, const char * stem) {
    const size_t n = strlen(stem);
    if (strncmp(name, stem, n) != 0 || name[n] != '-') return -1;
    char * end = nullptr;
    const long il = strtol(name + n + 1, &end, 10);
    return end && *end == '\0' && il >= 0 && il < INT_MAX ? (int) il : -1;
}

static bool on_eval(ggml_tensor * t, bool ask, void * user) {
    Routes & r = *static_cast<Routes *>(user);
    const int probs = layer_of(t->name, "ffn_moe_probs");
    if (probs >= 0) {
        r.n_expert[probs] = t->ne[0];
        return false;
    }
    const int il = layer_of(t->name, "ffn_moe_topk");
    if (il < 0) return false;
    if (ask) return true;
    if (t->type != GGML_TYPE_I32 || t->ne[1] < 1) {
        fprintf(stderr, "nemotron_ref: %s is %s [%lld, %lld], want i32 [k, tokens]\n", t->name, ggml_type_name(t->type),
                (long long) t->ne[0], (long long) t->ne[1]);
        r.failed = true;
        return false;
    }
    std::vector<int32_t> & row = r.ids[il];
    row.resize((size_t) t->ne[0]);
    ggml_backend_tensor_get(t, row.data(), (size_t) (t->ne[1] - 1) * t->nb[1], row.size() * sizeof(int32_t));
    return true;
}

static void print_top5(const std::string & name, const float * row, int n_vocab) {
    const int k = std::min(n_vocab, 5);
    std::vector<int> order(n_vocab);
    for (int i = 0; i < n_vocab; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) {
        if (row[a] != row[b]) return row[a] > row[b];
        return a < b;
    });
    fprintf(stderr, "nemotron_ref: top5 %s ids=", name.c_str());
    for (int i = 0; i < k; ++i) fprintf(stderr, "%s%d", i ? "," : "", order[i]);
    fprintf(stderr, " logits=");
    for (int i = 0; i < k; ++i) fprintf(stderr, "%s%.6f", i ? "," : "", row[order[i]]);
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    std::string model_path, out, want_arch, fixture_line;
    int n_ctx = 0, n_ubatch = 512, threads = 0;
    std::vector<Prompt> prompts;
    for (int i = 1; i < argc; ++i) {
        const bool more = i + 1 < argc;
        if (strcmp(argv[i], "-m") == 0 && more) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--arch") == 0 && more) {
            want_arch = argv[++i];
        } else if (strcmp(argv[i], "--fixture-line") == 0 && more) {
            fixture_line = argv[++i];
            if (fixture_line.rfind("# fixture\t", 0) != 0 || fixture_line.find('\n') != std::string::npos) {
                fprintf(stderr, "nemotron_ref: --fixture-line wants one '# fixture<TAB>...' line\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--out") == 0 && more) {
            out = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && more) {
            Prompt p;
            if (!parse_prompt(argv[++i], p)) {
                fprintf(stderr, "nemotron_ref: --prompt wants name:count:ids-file, got '%s'\n", argv[i]);
                return 2;
            }
            prompts.push_back(std::move(p));
        } else if (strcmp(argv[i], "-c") == 0 && more) {
            if (!parse_i32("nemotron_ref", "-c", argv[++i], 0, n_ctx)) return 2;
        } else if (strcmp(argv[i], "-ub") == 0 && more) {
            if (!parse_i32("nemotron_ref", "-ub", argv[++i], 1, n_ubatch)) return 2;
        } else if (strcmp(argv[i], "-t") == 0 && more) {
            if (!parse_i32("nemotron_ref", "-t", argv[++i], 0, threads)) return 2;
        } else {
            fprintf(stderr, "nemotron_ref: unknown or incomplete argument '%s'\n", argv[i]);
            return 2;
        }
    }
    const char * build = getenv("BLOOMERY_REF_BUILD");
    if (model_path.empty() || want_arch.empty() || out.empty() || prompts.empty() || n_ubatch <= 0 || !build || !*build) {
        fprintf(stderr, "usage: nemotron_ref -m <gguf> --arch <name> --out <dir> --prompt <name>:<n>:<ids> ... "
                        "[-c <ctx>] [-ub <ubatch>] [-t <threads>] [--fixture-line <line>]\n"
                        "BLOOMERY_REF_BUILD must name the mainline build\n");
        return 2;
    }
    int max_n = 0;
    for (Prompt & p : prompts) {
        if (!read_token_file(p.path, p.n, p.ids)) return 2;
        max_n = std::max(max_n, p.n);
    }
    if (n_ctx == 0) n_ctx = max_n;
    if (n_ctx < max_n) {
        fprintf(stderr, "nemotron_ref: a prompt of %d ids does not fit in -c %d\n", max_n, n_ctx);
        return 2;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "nemotron_ref: failed to load %s\n", model_path.c_str());
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) max_n;
    cp.n_ubatch = (uint32_t) std::min(n_ubatch, max_n);
    cp.n_seq_max = 1;
    Routes routes;
    cp.cb_eval = on_eval;
    cp.cb_eval_user_data = &routes;
    if (threads > 0) {
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "nemotron_ref: failed to make the context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n_vocab <= 0) {
        fprintf(stderr, "nemotron_ref: the model names no vocabulary\n");
        return 1;
    }
    char arch[128] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0 || want_arch != arch) {
        fprintf(stderr, "nemotron_ref: general.architecture is '%s', want %s\n", arch, want_arch.c_str());
        return 1;
    }

    if (mkdir(out.c_str(), 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "nemotron_ref: cannot create %s\n", out.c_str());
        return 1;
    }
    llama_batch batch = llama_batch_init(max_n, 0, 1);
    std::vector<std::pair<int, int64_t>> route_layers;  // (il, n_expert) in layer order, from the first prompt
    size_t route_k = 0;
    for (int i = 0; i < (int) prompts.size(); ++i) {
        const Prompt & p = prompts[i];
        if (i > 0) llama_memory_clear(llama_get_memory(ctx), true);
        for (int t = 0; t < p.n; ++t) {
            batch.token[t] = p.ids[t];
            batch.pos[t] = t;
            batch.n_seq_id[t] = 1;
            batch.seq_id[t][0] = 0;
            batch.logits[t] = t + 1 == p.n;
        }
        batch.n_tokens = p.n;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "nemotron_ref: decode failed for %s\n", p.name.c_str());
            return 1;
        }
        const float * row = llama_get_logits_ith(ctx, -1);
        if (!row) {
            fprintf(stderr, "nemotron_ref: no last-position logits for %s\n", p.name.c_str());
            return 1;
        }
        for (int j = 0; j < n_vocab; ++j) {
            if (!std::isfinite(row[j])) {
                fprintf(stderr, "nemotron_ref: %s logit %d is not finite\n", p.name.c_str(), j);
                return 1;
            }
        }
        if (routes.failed) return 1;
        if (i == 0) {
            for (const auto & [il, row] : routes.ids) {
                const auto n = routes.n_expert.find(il);
                if (n == routes.n_expert.end()) {
                    fprintf(stderr, "nemotron_ref: layer %d has ffn_moe_topk and no ffn_moe_probs\n", il);
                    return 1;
                }
                if (route_k == 0) route_k = row.size();
                if (row.size() != route_k) {
                    fprintf(stderr, "nemotron_ref: layer %d selects %zu experts, layer %d %zu\n", il, row.size(),
                            route_layers.front().first, route_k);
                    return 1;
                }
                route_layers.emplace_back(il, n->second);
            }
        }
        if (routes.ids.size() != route_layers.size()) {
            fprintf(stderr, "nemotron_ref: %s routed %zu layers, the first prompt %zu\n", p.name.c_str(), routes.ids.size(),
                    route_layers.size());
            return 1;
        }
        if (!route_layers.empty()) {
            std::vector<int32_t> flat;
            for (const auto & [il, n_expert] : route_layers) {
                for (int32_t id : routes.ids[il]) {
                    if (id < 0 || id >= n_expert) {
                        fprintf(stderr, "nemotron_ref: %s layer %d selected expert %d of %lld\n", p.name.c_str(), il, id,
                                (long long) n_expert);
                        return 1;
                    }
                    flat.push_back(id);
                }
            }
            if (!write_raw(out + "/" + p.name + ".routes.i32", flat.data(), flat.size())) return 1;
            routes.ids.clear();
        }
        if (!write_ids(out + "/" + p.name + ".ids", p.ids)) return 1;
        if (!write_raw(out + "/" + p.name + ".logits.f32", row, (size_t) n_vocab)) return 1;
        print_top5(p.name, row, n_vocab);
    }

    const std::string final_path = out + "/MANIFEST.tsv";
    const std::string partial_path = final_path + ".partial";
    FILE * m = fopen(partial_path.c_str(), "w");
    if (!m) {
        fprintf(stderr, "nemotron_ref: cannot write %s\n", partial_path.c_str());
        return 1;
    }
    fprintf(m, "# nemotron_ref — last-position logits, raw f32, little-endian\n");
    fprintf(m, "# model\t%s\n", model_path.c_str());
    if (!fixture_line.empty()) fprintf(m, "%s\n", fixture_line.c_str());
    fprintf(m, "# build\t%s\n", build);
    fprintf(m, "# arch\t%s\n", arch);
    fprintf(m, "# source\tllama_get_logits_ith(ctx, -1)\n");
    for (const auto & [il, n_expert] : route_layers) {
        fprintf(m, "# route layer\t%d\t%lld\t%zu\n", il, (long long) n_expert, route_k);
    }
    fprintf(m, "# prompt columns\tname\tn_ids\tn_vocab\tids\tlogits%s\n", route_layers.empty() ? "" : "\troutes");
    for (const Prompt & p : prompts) {
        fprintf(m, "prompt\t%s\t%d\t%d\t%s.ids\t%s.logits.f32", p.name.c_str(), p.n, n_vocab, p.name.c_str(),
                p.name.c_str());
        if (!route_layers.empty()) fprintf(m, "\t%s.routes.i32", p.name.c_str());
        fprintf(m, "\n");
    }
    fprintf(m, "# complete\t%zu\t0\n", prompts.size());
    if (fclose(m) != 0 || rename(partial_path.c_str(), final_path.c_str()) != 0) {
        fprintf(stderr, "nemotron_ref: cannot install %s\n", final_path.c_str());
        return 1;
    }
    fprintf(stderr, "nemotron_ref: %zu prompts, vocab %d, into %s\n", prompts.size(), n_vocab, out.c_str());
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
