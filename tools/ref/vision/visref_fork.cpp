// visref_fork — the V4.1 vision oracle's fork side (tools/ref/vision/build-fork.sh builds it against smalinin's
// llama.cpp fork). For each case of a cases directory (visref_cases.py writes it) it feeds the case's ids, the
// image span's positions as one embd batch of the span's bf16 rows widened to f32 (exact), then decodes `gen`
// greedy tokens one llama_decode each, and writes the answer's ids and the logits row each of its first `keep`
// tokens was picked from. No chat render and no tokenizer: the ids are the case's, so both engines read the same.
//
//   visref_fork <model.gguf> <cases dir> <out dir> <keep>
//
// cases.tsv rows: name, n_ids, span_at, span_len (0 0: no span), gen. Files: <name>.ids.i32 (n_ids i32, the span's
// positions holding the image token), <name>.rows.bf16 (span_len rows of n_embd_inp bf16). Writes <name>.answer.i32,
// <name>.logits.f32 ([min(keep, answer), n_vocab] f32), <name>.answer.txt, and one `case` line per case. Placement:
// every layer on the card, the routed experts' tensors in host memory, no op offload, flash attention on, 32 threads.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

template <typename T> static std::vector<T> slurp(const std::string & path, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(v.data()), n * sizeof(T)) || f.peek() != EOF) {
        fprintf(stderr, "visref_fork: %s does not hold exactly %zu values of %zu bytes\n", path.c_str(), n, sizeof(T));
        exit(2);
    }
    return v;
}

template <typename T> static void spill(const std::string & path, const std::vector<T> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T))) { fprintf(stderr, "visref_fork: cannot write %s\n", path.c_str()); exit(2); }
}

// Decode positions [from, to) of the case: token ids, or the span's rows when `rows` is given; logits for the last.
static void feed(llama_context * ctx, const std::vector<llama_token> & ids, const float * rows, int n_embd, int from, int to, bool last) {
    const int n_batch = (int) llama_n_batch(ctx);
    for (int at = from; at < to; at += n_batch) {
        const int n = std::min(n_batch, to - at);
        llama_batch b = llama_batch_init(n, rows ? n_embd : 0, 1);
        for (int i = 0; i < n; i++) {
            if (rows) { std::copy(rows + (size_t) (at - from + i) * n_embd, rows + (size_t) (at - from + i + 1) * n_embd, b.embd + (size_t) i * n_embd); }
            else      { b.token[i] = ids[at + i]; }
            b.pos[i] = at + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = last && at + i == to - 1;
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "visref_fork: llama_decode failed at %d..%d\n", at, at + n); exit(1); }
        llama_batch_free(b);
    }
}

int main(int argc, char ** argv) {
    if (argc != 5) { fprintf(stderr, "usage: visref_fork <model.gguf> <cases dir> <out dir> <keep>\n"); return 64; }
    const std::string cases = argv[2], out = argv[3];
    const int keep = atoi(argv[4]);
    llama_backend_init();
    const llama_model_tensor_buft_override host[] = {
        { "_exps\\.", ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) }, { nullptr, nullptr } };
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.tensor_buft_overrides = host;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "visref_fork: cannot load %s\n", argv[1]); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = cp.n_ubatch = 512; cp.n_threads = cp.n_threads_batch = 32;
    cp.op_offload = false; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "visref_fork: cannot make a context\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab), n_embd = llama_model_n_embd_inp(model);
    printf("visref_fork: n_vocab %d n_embd_inp %d n_ctx %u n_batch %u\n", n_vocab, n_embd, llama_n_ctx(ctx), llama_n_batch(ctx));
    std::ifstream list(cases + "/cases.tsv");
    std::string line;
    while (std::getline(list, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        std::string name; int n_ids, at, len, gen;
        if (!(row >> name >> n_ids >> at >> len >> gen) || at < 0 || len < 0 || at + len > n_ids || gen < 1) { fprintf(stderr, "visref_fork: bad case row: %s\n", line.c_str()); return 2; }
        if (n_ids + gen > (int) llama_n_ctx(ctx)) { fprintf(stderr, "visref_fork: %s needs %d positions\n", name.c_str(), n_ids + gen); return 2; }
        std::vector<llama_token> ids = slurp<llama_token>(cases + "/" + name + ".ids.i32", n_ids);
        std::vector<float> rows((size_t) len * n_embd);
        if (len > 0) {
            std::vector<uint16_t> bf = slurp<uint16_t>(cases + "/" + name + ".rows.bf16", (size_t) len * n_embd);
            for (size_t i = 0; i < bf.size(); i++) { uint32_t u = (uint32_t) bf[i] << 16; memcpy(&rows[i], &u, 4); }
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        if (len > 0) {
            feed(ctx, ids, nullptr, n_embd, 0, at, false);
            feed(ctx, ids, rows.data(), n_embd, at, at + len, at + len == n_ids);
        }
        feed(ctx, ids, nullptr, n_embd, at + len, n_ids, true);
        std::vector<llama_token> answer;
        std::vector<float> logits;
        std::string text;
        for (int k = 0; k < gen; k++) {
            const float * l = llama_get_logits_ith(ctx, -1);
            int best = 0;
            for (int v = 1; v < n_vocab; v++) if (l[v] > l[best]) best = v;
            if (k < keep) logits.insert(logits.end(), l, l + n_vocab);
            answer.push_back(best);
            char piece[256];
            const int np = llama_token_to_piece(vocab, best, piece, sizeof piece, 0, true);
            if (np > 0) text.append(piece, np);
            if (llama_vocab_is_eog(vocab, best) || k + 1 == gen) break;
            std::vector<llama_token> one = { best };
            llama_batch b = llama_batch_get_one(one.data(), 1);
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "visref_fork: decode of answer token %d failed\n", k); return 1; }
        }
        spill(out + "/" + name + ".answer.i32", answer);
        spill(out + "/" + name + ".logits.f32", logits);
        std::ofstream(out + "/" + name + ".answer.txt") << text;
        printf("case %s ids %d span %d+%d answer %zu eog %d logits %zu\n", name.c_str(), n_ids, at, len, answer.size(),
               (int) llama_vocab_is_eog(vocab, answer.back()), logits.size() / n_vocab);
        fflush(stdout);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
