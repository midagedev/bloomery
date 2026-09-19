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
    std::vector<char *> passthrough;
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--prompts") == 0 && i + 1 < argc) {
            prompts_path = argv[++i];
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

    // One row per prompt, tab separated, top-5 by (logit desc, id asc) so ties are
    // resolved the same way on both sides. The logit values ride along because the
    // question "did the argmax match" is not enough: a prompt whose top two are 0.1
    // apart is a near-tie, and a near-tie that matched today is not the same evidence
    // as one that is 8 logits clear.
    printf("# argmax_ref\tik_llama.cpp\tmodel=%s\n", params.model.c_str());
    printf("#id\tn_tokens\targmax\ttop5_ids\ttop5_logits\n");
    for (const Prompt & p : prompts) {
        llama_kv_cache_clear(init.context);
        std::vector<llama_token> toks = p.tokens;
        if (llama_decode(init.context, llama_batch_get_one(toks.data(), (int) toks.size(), 0, 0))) {
            fprintf(stderr, "argmax_ref: decode failed on prompt %d\n", p.id);
            return 1;
        }
        const float * logits = llama_get_logits_ith(init.context, (int) toks.size() - 1);
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
        printf("\n");
        fflush(stdout);
    }

    llama_free(init.context);
    llama_free_model(init.model);
    llama_backend_free();
    return 0;
}
