// argmax_ref — ik_llama.cpp's greedy next token for a fixed prompt set.
//
// The 1-4 gate asks one question the oracle cannot: does our assembled forward pick the
// same token as ik across prompts it was never tuned on? The oracle is one 6-token
// sequence; this is thirty-two, at lengths 4 to 11, which straddles the M <= 7 boundary
// where ik switches activation dialects (crates/model/src/attn.rs, `quantize_act`).
//
// This is NOT dump_ref. It writes no tensors and never touches $BLOOMERY_DATA/ref --
// deliberately, because the one thing that has already destroyed the oracle once is a
// round reaching for "the binary that links ik and takes tokens" (docs/oracle.md). If you
// want ik's answer for a prompt, this is that binary; the dumper stays the oracle's.
//
// It does not tokenize, for the same reason dump_ref does not: the ids come from
// tools/ref/prompts.tsv, read once with llama-tokenize and committed, and BOTH sides read
// that file. A tokenizer difference would otherwise surface as a wrong token id and read
// as a kernel bug.
//
// --gen N appends N greedy decode steps after the prompt's next-token row, one
// llama_decode per token, as two extra columns: gen_ids (the picked tokens) and
// gen_margins (each step's top1-top2 logit margin under the same order as the
// top-5: logit descending, id ascending). The margin is the point: after a first
// difference two continuations are conditioned on different text, so a consumer
// can only judge that first step, and there it needs to tell a near-tie flip
// from a real divergence. Generation stops early on the model's EOS token; the
// EOS token itself is recorded and the shorter gen_ids length is the record of
// the stop. gen_ids[0] is always the row's argmax column. --gen 0 (the default)
// prints exactly the five columns below, byte for byte.
//
// Prompts are isolated by llama_kv_cache_clear AND by a priming decode at startup: ik builds
// a warmup graph -- all experts instead of n_expert_used -- for the first single-token BOS
// decode of a context whose n_eval is still 0, which is exactly the shape of every prompt's
// first step here. See the block in main(). Without it every row is a 64-expert answer.
//
// --step-prefill feeds the prompt one token per llama_decode (the M=1 path)
// instead of one batch. ik's CUDA backend writes prompt-independent logits
// for batch prefill of nine tokens and up in this build — one token at a
// time is the same tokens, positions and KV content through working kernels,
// and the header line records `prefill=step` so a file says how it was made.
//
// Build: tools/ref/build-argmax.sh   Run: $BLOOMERY_DATA/bin/argmax_ref -m <gguf> --prompts tools/ref/prompts.tsv
#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Prompt {
    int                      id;
    std::string              text;
    std::vector<llama_token> tokens;
};

static std::vector<Prompt> read_prompts(const char * path) {
    std::vector<Prompt> out;
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "argmax_ref: cannot read %s\n", path);
        return out;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string id, text, ids;
        if (!std::getline(ss, id, '\t')) continue;
        if (!std::getline(ss, text, '\t')) continue;
        if (!std::getline(ss, ids, '\t')) continue;
        Prompt p;
        p.id   = atoi(id.c_str());
        p.text = text;
        std::stringstream ts(ids);
        std::string tok;
        while (std::getline(ts, tok, ',')) p.tokens.push_back((llama_token) atoi(tok.c_str()));
        out.push_back(p);
    }
    return out;
}

int main(int argc, char ** argv) {
    const char *       prompts_path = nullptr;
    int                gen = 0;
    bool               step_prefill = false;
    std::vector<char *> passthrough;
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--prompts") == 0 && i + 1 < argc) {
            prompts_path = argv[++i];
        } else if (strcmp(argv[i], "--gen") == 0 && i + 1 < argc) {
            gen = atoi(argv[++i]);
            if (gen < 0) {
                fprintf(stderr, "argmax_ref: --gen must be >= 0\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--step-prefill") == 0) {
            step_prefill = true;
        } else {
            passthrough.push_back(argv[i]);
        }
    }
    if (!prompts_path) {
        fprintf(stderr, "argmax_ref: --prompts <tsv> is required; this tool does not tokenize\n");
        return 2;
    }
    std::vector<Prompt> prompts = read_prompts(prompts_path);
    if (prompts.empty()) {
        fprintf(stderr, "argmax_ref: no prompts in %s\n", prompts_path);
        return 2;
    }

    gpt_params params;
    if (!gpt_params_parse((int) passthrough.size(), passthrough.data(), params)) {
        fprintf(stderr, "argmax_ref: bad arguments\n");
        return 2;
    }
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_init_result init = llama_init_from_gpt_params(params);
    if (!init.model || !init.context) {
        fprintf(stderr, "argmax_ref: failed to load the model\n");
        return 1;
    }
    const int n_vocab = llama_n_vocab(init.model);
    const llama_token eos = llama_token_eos(init.model);

    // Prime the context so that no measured sequence is built as a warmup graph.
    //
    // ik decides per graph: is_warming_up = n_eval == 0 && n_tokens == 1 && token[0] == BOS
    // (llama-build-context.cpp), and a warmup graph runs ALL experts instead of the model's
    // n_expert_used. Every prompt here starts with BOS and --step-prefill feeds one token per
    // decode, so the first decode of a sequence matches that shape; n_eval is raised only inside
    // llama_synchronize, and only when a single token was queued. Reading the top-5 row after a
    // whole prompt queues n_tokens > 1, so without this block n_eval stays 0 for the entire run
    // and every row is a 64-expert answer. One lone decode plus one synchronize raises n_eval
    // for the life of the context; llama_reset_timings would put it back to 0, so this tool does
    // not call it.
    {
        llama_token bos = llama_token_bos(init.model);
        if (bos == -1) bos = eos;
        if (llama_decode(init.context, llama_batch_get_one(&bos, 1, 0, 0))) {
            fprintf(stderr, "argmax_ref: priming decode failed\n");
            return 1;
        }
        llama_synchronize(init.context);
        llama_kv_cache_clear(init.context);
    }

    // Top-2 of a logit vector under the row's order (logit desc, id asc): the
    // greedy pick and its margin without the top-5 partial_sort. The caller
    // owns `logits` for the whole call. n_vocab >= 2, as the top-5 already
    // assumes.
    auto top2 = [&](const float * lg, int & best, int & second) {
        auto ahead = [&](int a, int b) {
            if (lg[a] != lg[b]) return lg[a] > lg[b];
            return a < b;
        };
        best = 0;
        second = -1;
        for (int i = 1; i < n_vocab; ++i) {
            if (ahead(i, best)) {
                second = best;
                best = i;
            } else if (second < 0 || ahead(i, second)) {
                second = i;
            }
        }
    };

    // One row per prompt, tab separated, top-5 by (logit desc, id asc) so ties are
    // resolved the same way on both sides. The logit values ride along because the
    // question "did the argmax match" is not enough: a prompt whose top two are 0.1
    // apart is a near-tie, and a near-tie that matched today is not the same evidence
    // as one that is 8 logits clear.
    printf("# argmax_ref\tik_llama.cpp\tmodel=%s", params.model.c_str());
    if (gen > 0) printf("\tgen=%d", gen);
    if (step_prefill) printf("\tprefill=step");
    printf("\n");
    printf("#id\tn_tokens\targmax\ttop5_ids\ttop5_logits");
    if (gen > 0) printf("\tgen_ids\tgen_margins");
    printf("\n");
    for (const Prompt & p : prompts) {
        llama_kv_cache_clear(init.context);
        std::vector<llama_token> toks = p.tokens;
        if (step_prefill) {
            // One token per llama_decode: the M=1 path. ik's CUDA batch prefill
            // writes prompt-independent logits from nine tokens up in this
            // build, so this is the only way to feed a long prompt to the GPU
            // backend and get the model's answer rather than the defect. Same
            // tokens, same positions, same KV cache content as the batch form.
            for (size_t i = 0; i < toks.size(); ++i) {
                if (llama_decode(init.context, llama_batch_get_one(&toks[i], 1, (int) i, 0))) {
                    fprintf(stderr, "argmax_ref: step prefill failed on prompt %d at %zu\n", p.id, i);
                    return 1;
                }
            }
        } else if (llama_decode(init.context, llama_batch_get_one(toks.data(), (int) toks.size(), 0, 0))) {
            fprintf(stderr, "argmax_ref: decode failed on prompt %d\n", p.id);
            return 1;
        }
        // -1 is the last OUTPUT row: llama_batch_get_one asks for last-token
        // logits only, and get_logits_ith(i >= 0) goes through the output_ids
        // map, which need not carry n_tokens entries — a positive index reads
        // whatever row the map names, not this token's logits.
        const float * logits = llama_get_logits_ith(init.context, -1);
        std::vector<int> idx(n_vocab);
        for (int i = 0; i < n_vocab; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) {
            if (logits[a] != logits[b]) return logits[a] > logits[b];
            return a < b;
        });
        printf("%d\t%d\t%d\t", p.id, (int) toks.size(), idx[0]);
        for (int i = 0; i < 5; ++i) printf("%s%d", i ? "," : "", idx[i]);
        printf("\t");
        for (int i = 0; i < 5; ++i) printf("%s%.6f", i ? "," : "", logits[idx[i]]);
        if (gen > 0) {
            // Step 0 is the prompt row's own argmax and margin; every later step
            // feeds the picked token at its position and reads the new logits.
            std::vector<llama_token> gen_ids;
            std::vector<float>       gen_margins;
            llama_token              cur = idx[0];
            float                    cur_margin = logits[idx[0]] - logits[idx[1]];
            int                      pos = (int) toks.size();
            for (int step = 0; step < gen; ++step) {
                gen_ids.push_back(cur);
                gen_margins.push_back(cur_margin);
                if (cur == eos) break;
                llama_token feed = cur;
                if (llama_decode(init.context, llama_batch_get_one(&feed, 1, pos, 0))) {
                    fprintf(stderr, "argmax_ref: decode failed on prompt %d at gen step %d\n", p.id, step);
                    return 1;
                }
                ++pos;
                const float * lg = llama_get_logits_ith(init.context, -1);
                int best = 0, second = -1;
                top2(lg, best, second);
                cur = best;
                cur_margin = lg[best] - lg[second];
            }
            printf("\t");
            for (size_t i = 0; i < gen_ids.size(); ++i) printf("%s%d", i ? "," : "", gen_ids[i]);
            printf("\t");
            for (size_t i = 0; i < gen_margins.size(); ++i) printf("%s%.6f", i ? "," : "", gen_margins[i]);
        }
        printf("\n");
        fflush(stdout);
    }

    llama_free(init.context);
    llama_free_model(init.model);
    llama_backend_free();
    return 0;
}
