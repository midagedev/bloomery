// nemotron_ref — last-position logits of each prompt from llama.cpp mainline, on the CPU.
//
// One llama_decode per prompt, then llama_get_logits_ith(ctx, -1): the last logit row. Only that
// token is marked for output. The row is n_vocab f32, little-endian, plus the prompt's ids.
// Headers are # model, # build (BLOOMERY_REF_BUILD, the tokenizer family's spelling of the
// mainline md5s) and # arch. # complete is written only after every file, by renaming the manifest.
//
//   nemotron_ref -m <gguf> --out <dir> --prompt <name>:<n>:<ids> ... [-c <ctx>] [-ub <ubatch>] [-t <threads>]
//
// --prompt is repeated, once a prompt. <ids> is one decimal id a line, at least <n> of them; the
// first <n> are the prompt. n_gpu_layers is 0. The caller also empties CUDA_VISIBLE_DEVICES.

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
    std::string model_path, out;
    int n_ctx = 0, n_ubatch = 512, threads = 0;
    std::vector<Prompt> prompts;
    for (int i = 1; i < argc; ++i) {
        const bool more = i + 1 < argc;
        if (strcmp(argv[i], "-m") == 0 && more) {
            model_path = argv[++i];
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
    if (model_path.empty() || out.empty() || prompts.empty() || n_ubatch <= 0 || !build || !*build) {
        fprintf(stderr, "usage: nemotron_ref -m <gguf> --out <dir> --prompt <name>:<n>:<ids> ... "
                        "[-c <ctx>] [-ub <ubatch>] [-t <threads>]\n"
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
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0 || strcmp(arch, "nemotron_h") != 0) {
        fprintf(stderr, "nemotron_ref: general.architecture is '%s', want nemotron_h\n", arch);
        return 1;
    }

    if (mkdir(out.c_str(), 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "nemotron_ref: cannot create %s\n", out.c_str());
        return 1;
    }
    llama_batch batch = llama_batch_init(max_n, 0, 1);
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
    fprintf(m, "# build\t%s\n", build);
    fprintf(m, "# arch\t%s\n", arch);
    fprintf(m, "# source\tllama_get_logits_ith(ctx, -1)\n");
    fprintf(m, "# prompt columns\tname\tn_ids\tn_vocab\tids\tlogits\n");
    for (const Prompt & p : prompts) {
        fprintf(m, "prompt\t%s\t%d\t%d\t%s.ids\t%s.logits.f32\n", p.name.c_str(), p.n, n_vocab, p.name.c_str(),
                p.name.c_str());
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
