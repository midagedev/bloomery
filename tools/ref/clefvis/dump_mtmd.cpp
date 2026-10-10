// dump_mtmd — llama.cpp mainline's image path for a Qwen3-VL family seat (mtmd `qwen3vl_merger` tower + the family's
// text model: Clef-Flash's qwen35, Qwen3.6's qwen35moe, Qwen3.8's qwen4exp), written as oracle sets in dump_ref's set
// format, so crates/refset reads them through its node-dump family check (refset::arch::{qwen35::clefvis, qwen35moe::vis,
// qwen4exp::vis}, refset::clefvis). One source, one binary a mainline tree: the Clef sets are dumped by the binary built
// against the tree that pins them, the Qwen sets by the one built against theirs (tools/ref/clefvis/build-clefvis.sh).
//
//   dump_mtmd tower  --mmproj <gguf> -m <text gguf> --image <name>=<png> ... [--out-preproc <dir>] --out-taps <dir>
//                    [--cpu] [-t <threads>] [--image-min-tokens <n>] [--image-max-tokens <n>] [--tap-patches <n>]
//                    [--taps full|final] [--mmproj-sha256 <hex>] [--card <name>]
//                    (both modes: the header's `# mmproj` digest and `# device` card)
//   dump_mtmd hidden --mmproj <gguf> -m <text gguf> --ids <file> --image <name>=<png> ... --out <dir>
//                    [--rows mtmd|prose|bf16] [--prose-ids <file>] [--image-pad-id <id>]
//                    [-c <ctx>] [-ub <ubatch>] [-ngl <n>] [-ncmoe <n>] [-t <threads>] [--cpu]
//                    [--rows-from embd|node] [--decode <steps> --out-decode <dir>]
//   dump_mtmd chat   --mmproj <gguf> -m <text gguf> --requests <jsonl> --image <name>=<png> ... --out <dir>
//                    [--image-pad-id <id>] [--image-min-tokens <n>] [--image-max-tokens <n>] [-t <threads>]
//                    (only a binary built with DUMP_MTMD_CHAT, which links libllama-common)
//
// tower. Per image, one encode of the image chunk through an mtmd context with no callback (the clean pass: what
// the server runs) and one through a second context whose cb_eval asks for named graph nodes (the tap pass).
//   preproc set (`# clefvis preproc`): per image `<name>/inp_raw`, the graph input of the tower: the normalized f32
//     image after mtmd's preprocess (decode, `calc_size_preserved_ratio`, PAD_CEIL resize on black, (u8/255-0.5)/0.5),
//     channel-planar [W, H, 3, 1]. It is a leaf, so it is read at its first consumer (the IM2COL of the first conv),
//     like dump_ref's `input` rows. The `# image` lines carry the sizes and the token grid; `rgb8_sha256` is the
//     decoded stb_image bytes' digest.
//   taps set (`# clefvis taps`): per image with at most --tap-patches patches, for blocks 0, 1, 13 and 26 the nodes
//     ln1, Qcur_rope, attn_out, ffn_inp, ffn_out and layer_out, plus inp_pos_emb and the post-LN output
//     (`norm_b-27`) from the tap pass, and `embd` (mtmd_get_output_embd of the clean pass, [4096, tokens]). A larger
//     image carries `embd` only. `# tap_effect` says whether the tap pass's final embeddings equal the clean pass's
//     bit for bit (asking for a node splits the scheduler's graph; the clean file is the one to compare against).
//     With `--taps final` every image, whatever its size, carries `embd` and the post-LN output `norm_b-27` only
//     (`# tap_effect … taps final`): the tower's output end, which proves a projector file's load path, not its blocks.
// hidden. The prompt is our ids (one id per line); each maximal run of --image-pad-id is an image span, the n-th run
//   the n-th --image. Text runs go through llama_decode as token batches (every position an output); an image
//   span is mtmd's: mtmd_encode_chunk, then mtmd_helper_decode_image_chunk at the span's n_past, whose post-decode
//   callback collects each batch view's result_norm rows and the exact position arrays the helper fed
//   (embeddings=true makes every row an output, so the helper's logits=false flags are overridden by llama.cpp).
//   --rows mtmd   the image rows are the tower's (set C)
//   --rows bf16   the same rows rounded to bf16 (round to nearest even) and widened (set C'')
//   --rows prose  the span's ids replaced by the first ids of --prose-ids, decoded as text at the same split points,
//                 no tower, positions advancing by the span's length (set C')
// A span's token count must equal mtmd's n_tokens for its image, or the run stops by name. Set rows: `result_norm`
//   [n_embd, ids] and `mrope_pos` [3, ids] (t, y, x per row; a text row's three are its position), with the int twin.
//   --rows-from embd|node  where result_norm comes from. embd (default): llama_get_embeddings_ith of a context with
//                 embeddings on, every row an output. node: the context has embeddings off, every row is flagged an output
//                 (the logits flag) and result_norm is read from the graph node of that name through the context's cb_eval,
//                 a ubatch at a time; the image rows are decoded by this tool's own batch (the helper's: embeddings, the
//                 M-RoPE positions of mtmd_image_tokens_get_decoder_pos, one sequence), the logits flag set on every row
//                 (the helper's sets none). qwen4exp's n_embd_out is hc * n_embd (the MTP hidden width), so its embedding
//                 extraction reads a result_norm tensor past its end: that arch is dumped with node.
//   --decode N --out-decode <dir>  after the prompt, N greedy decode steps in the same context (`# clefvis decode`):
//   step k takes the argmax id of the last row's logits (the first of equal values), decodes it as text at n_past and
//   records `ids` [N], `result_norm` [n_embd, N] (that token's row) and `n_past` [N] (the position after it), all int
//   rows with their i32 twin; the header carries `# n_past_start`, the prompt's tokens file and its spans.
// chat. The llama-server path of a chat request, `server-common.cpp`'s oaicompat_chat_params_parse and
//   tokenize_oai_content_array: each `image_url` part of a message's content array becomes a `media_marker` part (the
//   marker mtmd_default_marker() names, which the context's `media_marker` is too), `common_chat_msgs_parse_oaicompat`
//   flattens the parts (the `\n` join, none beside a marker), `common_chat_templates_apply` renders the model's own
//   template with the server's defaults (use_jinja, add_generation_prompt, thinking when the template supports it,
//   reasoning_format deepseek, prefill of a trailing assistant message), and `mtmd_tokenize` (add_special, parse_special)
//   splits the prompt around the images. The ids of an image chunk are its image-pad id repeated n_tokens times. One set a
//   request in <out>/<id> (`# clefvis chatids`: `ids` [n] with its i32 twin, `# prompt`, `# image` and `# span` lines)
//   and its ids, one a line, in <out>/<id>.ids: the file `hidden` reads. A request is a JSON line
//   {"id": ..., "messages": [{"role": ..., "content": "text" | [{"type": "text", "text": ...} |
//   {"type": "image_url", "image_url": {"url": "<an --image name>"}}]}]}; the n-th image part is the n-th image of the
//   request, in order.
//
// Header lines the node-dump reader checks: `# model` (tower sets: the mmproj as given, the tower sets' identity;
// hidden sets: the text model), `# arch`, `# build` (BLOOMERY_REF_BUILD), `# complete` last. The manifest goes to a
// .partial name and is renamed last.
//
// Build: tools/ref/clefvis/build-clefvis.sh   Run: tools/ref/clefvis/clefvis.sh (just dump-ref-clefvis)

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#ifdef DUMP_MTMD_CHAT
#include "chat.h"
#include "json.h"
#endif

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#define DIE(...) do { fprintf(stderr, "dump_mtmd: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)

// ---- sha256 -----------------------------------------------------------------------------------------------------

struct sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t  buf[64];
    size_t   fill = 0;
    uint64_t total = 0;
    static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 | p[4 * i + 3];
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const uint8_t * p, size_t n) {
        total += n;
        while (n > 0) {
            const size_t take = std::min(n, 64 - fill);
            memcpy(buf + fill, p, take);
            fill += take; p += take; n -= take;
            if (fill == 64) { block(buf); fill = 0; }
        }
    }
    std::string hex() {
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (fill != 56) update(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = (uint8_t) (bits >> (56 - 8 * i));
        update(len, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) snprintf(out + 8 * i, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

static std::string sha256_of(const uint8_t * p, size_t n) {
    sha256 s;
    s.update(p, n);
    return s.hex();
}

static bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

// ---- the set writer (dump_ref's format) ---------------------------------------------------------------------------

static std::string safe_name(const std::string & s) {
    std::string r = s;
    for (char & c : r) if (c == '/' || c == '\\' || c == ' ') c = '_';
    return r;
}

static void write_raw(const std::string & path, const void * data, size_t bytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) DIE("cannot write %s", path.c_str());
    if (fwrite(data, 1, bytes, f) != bytes) DIE("short write on %s", path.c_str());
    if (fclose(f) != 0) DIE("cannot flush %s", path.c_str());
}

struct set_writer {
    std::string dir;
    std::string head;     // header lines
    std::string rows;     // tensor and int rows
    int         written = 0;
    std::map<std::string, int> occ;

    explicit set_writer(const std::string & d) : dir(d) { mkdir(dir.c_str(), 0755); }

    void line(const std::string & l) { head += l + "\n"; }

    // One f32 tensor row and its file.
    void tensor_f32(const std::string & name, const char * op, const int64_t ne[4], const float * data) {
        const int o = occ[name]++;
        const size_t n = (size_t) ne[0] * ne[1] * ne[2] * ne[3];
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(data[i])) DIE("%s: value %zu is not finite", name.c_str(), i);
            sum += data[i];
        }
        write_raw(dir + "/" + safe_name(name) + "." + std::to_string(o) + ".f32", data, n * sizeof(float));
        char buf[512];
        snprintf(buf, sizeof(buf), "tensor\t%s\t%d\tf32\t%lld\t%lld\t%lld\t%lld\t%zu\t%.6f\t%s\t1\t0\t-\t-\n", name.c_str(), o,
                 (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3], n * sizeof(float), sum, op);
        rows += buf;
        written++;
    }

    // An i32 tensor: the f32 file (exact below 2^24, which the writer checks), the lossless i32 twin and the `int` row.
    void tensor_i32(const std::string & name, const char * op, const int64_t ne[4], const std::vector<int32_t> & v) {
        const int o = occ[name]++;
        const size_t n = (size_t) ne[0] * ne[1] * ne[2] * ne[3];
        if (v.size() != n) DIE("%s: %zu ints for shape %lld x %lld", name.c_str(), v.size(), (long long) ne[0], (long long) ne[1]);
        std::vector<float> f(n);
        int64_t sum = 0;
        uint64_t absmax = 0;
        for (size_t i = 0; i < n; ++i) {
            if (v[i] < -(1 << 24) || v[i] > (1 << 24)) DIE("%s: %d does not widen to f32 exactly", name.c_str(), v[i]);
            f[i] = (float) v[i];
            sum += v[i];
            absmax = std::max<uint64_t>(absmax, (uint64_t) (v[i] < 0 ? -(int64_t) v[i] : v[i]));
        }
        const std::string stem = safe_name(name) + "." + std::to_string(o);
        write_raw(dir + "/" + stem + ".f32", f.data(), n * sizeof(float));
        write_raw(dir + "/" + stem + ".i32", v.data(), n * sizeof(int32_t));
        char buf[512];
        snprintf(buf, sizeof(buf), "tensor\t%s\t%d\ti32\t%lld\t%lld\t%lld\t%lld\t%zu\t%.6f\t%s\t1\t0\t-\t-\n", name.c_str(), o,
                 (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3], n * sizeof(float), (double) sum, op);
        rows += buf;
        snprintf(buf, sizeof(buf), "int\t%s\t%d\ttensor\ti32\ti32\tflat\t%zu\t%zu\t%lld\t%llu\t%s.i32\n", name.c_str(), o, n,
                 n * sizeof(int32_t), (long long) sum, (unsigned long long) absmax, stem.c_str());
        rows += buf;
        written++;
    }

    // The manifest, the trailer, and the rename that publishes the set.
    void finish() {
        const std::string final_path = dir + "/MANIFEST.tsv", partial = final_path + ".partial";
        FILE * m = fopen(partial.c_str(), "w");
        if (!m) DIE("cannot write %s", partial.c_str());
        fputs(head.c_str(), m);
        fputs("# kind\tname\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\tsum\top\tcontig\tlogical\tsrc0\tsrc1\n", m);
        fputs("# int\tname\toccurrence\tof\ttype\ttwin\tlayout\tcount\tbytes\tsum\tabsmax\tfile\n", m);
        fputs(rows.c_str(), m);
        fprintf(m, "# complete\t%d\t0\n", written);
        if (fclose(m) != 0 || rename(partial.c_str(), final_path.c_str()) != 0) DIE("cannot install %s", final_path.c_str());
        fprintf(stderr, "dump_mtmd: %d rows into %s\n", written, dir.c_str());
    }
};

// ---- the graph-node callback --------------------------------------------------------------------------------------

struct tap_data {
    int64_t            ne[4];
    std::string        op;
    std::vector<float> v;
};

struct tap_state {
    bool                       armed = false;
    bool                       want_inp = false;   // the graph input `inp_raw`, at its first consumer
    bool                       inp_done = false;
    std::set<std::string>      want;
    std::map<std::string, tap_data> got;
    int                        asked = 0;
};

// A source named `name`, or the scheduler's backend-local copy of it (`CUDA0#name#0`: an input the host sets on the CPU
// buffer reaches a card's split as a copy under that name; its contents are the leaf's).
static const ggml_tensor * src_named(const ggml_tensor * t, const char * name) {
    const std::string hashed = std::string("#") + name + "#";
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (!t->src[i]) continue;
        if (strcmp(t->src[i]->name, name) == 0 || strstr(t->src[i]->name, hashed.c_str())) return t->src[i];
    }
    return nullptr;
}

static void capture(tap_state * s, const std::string & key, const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32) DIE("tap %s is %s, not f32", key.c_str(), ggml_type_name(t->type));
    if (!ggml_is_contiguous(t)) DIE("tap %s is not contiguous", key.c_str());
    tap_data d;
    for (int i = 0; i < 4; ++i) d.ne[i] = t->ne[i];
    d.op = ggml_op_desc(t);
    d.v.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, d.v.data(), 0, ggml_nbytes(t));
    s->got[key] = std::move(d);
}

static bool on_node(struct ggml_tensor * t, bool ask, void * user) {
    auto * s = static_cast<tap_state *>(user);
    static const bool trace = getenv("DUMP_MTMD_TRACE") != nullptr;
    if (trace && ask && s->asked < 12) {
        fprintf(stderr, "dump_mtmd: node %s (%s) armed %d srcs", t->name, ggml_op_desc(t), (int) s->armed);
        for (int i = 0; i < GGML_MAX_SRC; ++i) if (t->src[i]) fprintf(stderr, " [%d]%s", i, t->src[i]->name);
        fputc('\n', stderr);
    }
    if (!s->armed) return false;
    const ggml_tensor * inp = src_named(t, "inp_raw");
    const bool take_inp = inp && s->want_inp && !s->inp_done;
    if (ask) {
        s->asked++;
        return take_inp || (s->want.count(t->name) && !s->got.count(t->name));
    }
    if (take_inp) {
        capture(s, "inp_raw", inp);
        s->got["inp_raw"].op = "NONE";
        s->inp_done = true;
        return true;
    }
    capture(s, t->name, t);
    return true;
}

// ---- helpers -------------------------------------------------------------------------------------------------------

static float bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) DIE("a NaN row value cannot round to bf16");
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    float r;
    memcpy(&r, &u, 4);
    return r;
}

static std::string gguf_arch(const std::string & path) {
    gguf_init_params gp = {true, nullptr};
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) DIE("cannot read the gguf header of %s", path.c_str());
    const int64_t k = gguf_find_key(g, "general.architecture");
    std::string r = k >= 0 ? gguf_get_val_str(g, k) : "unknown";
    gguf_free(g);
    return r;
}

static uint32_t gguf_u32(const std::string & path, const std::string & key) {
    gguf_init_params gp = {true, nullptr};
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) DIE("cannot read the gguf header of %s", path.c_str());
    const int64_t k = gguf_find_key(g, key.c_str());
    if (k < 0) DIE("%s has no key %s", path.c_str(), key.c_str());
    const uint32_t v = gguf_get_val_u32(g, k);
    gguf_free(g);
    return v;
}

// The first GPU device ggml sees: its name and description, as the set's `# device` line records the card the run used.
static std::string ggml_card() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return std::string(ggml_backend_dev_name(d)) + " " + ggml_backend_dev_description(d);
        }
    }
    return "none";
}

static std::string env_or(const char * name, const char * dflt) {
    const char * v = getenv(name);
    return v && *v ? v : dflt;
}

// The header's provenance the caller states: the projector file's sha256 and the card's name (--mmproj-sha256, --card).
static std::string g_mmproj_sha256 = "unknown";
static std::string g_card = "unknown";

struct image_arg {
    std::string name, path;
};

struct image_info {
    std::string name, png_sha, rgb_sha;
    int w = 0, h = 0, best_w = 0, best_h = 0, nx = 0, ny = 0, n_tokens = 0, n_pos = 0;
};

static mtmd_context * make_mtmd(const std::string & mmproj, llama_model * model, bool cpu, int threads, int min_tokens,
                                int max_tokens, tap_state * taps) {
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu          = !cpu;
    mp.print_timings    = false;
    mp.n_threads        = threads;
    mp.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    mp.warmup           = false;
    mp.image_min_tokens = min_tokens;
    mp.image_max_tokens = max_tokens;
    if (taps) {
        mp.cb_eval           = on_node;
        mp.cb_eval_user_data = taps;
    }
    mtmd_context * c = mtmd_init_from_file(mmproj.c_str(), model, mp);
    if (!c) DIE("cannot load the mmproj %s", mmproj.c_str());
    if (!mtmd_support_vision(c)) DIE("%s has no vision tower", mmproj.c_str());
    return c;
}

// The image chunk of one image: decode (stb_image through the helper), tokenize `<__media__>` alone and keep the one
// image chunk; the text chunks around it (`<|vision_start|>`, `<|vision_end|>`) are mtmd's and are dropped here.
struct image_chunk {
    mtmd_bitmap *       bitmap = nullptr;
    mtmd_input_chunks * chunks = nullptr;
    const mtmd_input_chunk * chunk = nullptr;
    int w = 0, h = 0, nx = 0, ny = 0, n_tokens = 0;
    std::string rgb_sha;
    ~image_chunk() {
        if (chunks) mtmd_input_chunks_free(chunks);
        if (bitmap) mtmd_bitmap_free(bitmap);
    }
};

static void tokenize_image(mtmd_context * ctx, const image_arg & a, image_chunk & out) {
    mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(ctx, a.path.c_str(), false, mtmd_helper_init_opt_default());
    if (!w.bitmap) DIE("cannot decode %s", a.path.c_str());
    if (w.video_ctx) DIE("%s decoded as a video", a.path.c_str());
    out.bitmap = w.bitmap;
    out.w = (int) mtmd_bitmap_get_nx(out.bitmap);
    out.h = (int) mtmd_bitmap_get_ny(out.bitmap);
    out.rgb_sha = sha256_of(mtmd_bitmap_get_data(out.bitmap), mtmd_bitmap_get_n_bytes(out.bitmap));
    const std::string marker = mtmd_default_marker();
    mtmd_input_text text = {marker.c_str(), marker.size(), false, true};
    const mtmd_bitmap * bms[1] = {out.bitmap};
    out.chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize(ctx, out.chunks, &text, bms, 1) != 0) DIE("mtmd_tokenize refused %s", a.path.c_str());
    for (size_t i = 0; i < mtmd_input_chunks_size(out.chunks); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(out.chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            if (out.chunk) DIE("%s tokenized to two image chunks", a.path.c_str());
            out.chunk = c;
        }
    }
    if (!out.chunk) DIE("%s tokenized to no image chunk", a.path.c_str());
    out.n_tokens = (int) mtmd_input_chunk_get_n_tokens(out.chunk);
    const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(out.chunk);
    // the grid from the last token's decoder position: (row ny-1, column nx-1) of a raster grid
    const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) out.n_tokens - 1);
    out.nx = (int) last.x + 1;
    out.ny = (int) last.y + 1;
    if (out.nx * out.ny != out.n_tokens) DIE("%s: grid %d x %d is not %d tokens", a.path.c_str(), out.nx, out.ny, out.n_tokens);
    if (mtmd_image_tokens_get_n_pos(it) != std::max(out.nx, out.ny)) DIE("%s: n_pos is not max(nx, ny)", a.path.c_str());
}

static void encode_image(mtmd_context * ctx, const image_chunk & ic, int n_embd, std::vector<float> & out) {
    if (mtmd_encode_chunk(ctx, ic.chunk) != 0) DIE("mtmd_encode_chunk failed");
    const float * e = mtmd_get_output_embd(ctx);
    if (!e) DIE("no output embeddings");
    out.assign(e, e + (size_t) ic.n_tokens * n_embd);
}

static void header_common(set_writer & w, const std::string & kind, const std::string & model, const std::string & arch,
                          const std::string & text_model, const std::string & mmproj, bool cpu, int threads,
                          int min_tokens, int max_tokens, const std::string & flags, const std::string & title) {
    const std::string build = env_or("BLOOMERY_REF_BUILD", "");
    w.line("# " + title);
    w.line("# model\t" + model);
    if (!build.empty()) w.line("# build\t" + build);
    w.line("# arch\t" + arch);
    const size_t slash = model.rfind('/');
    w.line("# model_file\t" + (slash == std::string::npos ? model : model.substr(slash + 1)));
    w.line("# flags\t" + flags);
    w.line("# clefvis\t" + kind);
    w.line("# mmproj\t" + mmproj + "\tsha256\t" + g_mmproj_sha256);
    w.line("# text_model\t" + text_model);
    w.line("# device\t" + std::string(cpu ? "cpu" : "cuda") + "\tthreads\t" + std::to_string(threads) + "\tcard\t" +
           g_card + "\tggml\t" + (cpu ? "none" : ggml_card()));
    w.line("# flash_attn\tenabled\tclip warmup off");
    w.line("# image_tokens\tmin\t" + (min_tokens > 0 ? std::to_string(min_tokens) : std::string("default")) + "\tmax\t" +
           (max_tokens > 0 ? std::to_string(max_tokens) : std::string("default")));
}

static void image_lines(set_writer & w, const std::vector<image_info> & infos) {
    w.line("# image columns\tname png_sha256 rgb8_sha256 w h best_w best_h nx ny n_tokens n_pos");
    for (const auto & i : infos) {
        w.line("# image\t" + i.name + "\t" + i.png_sha + "\t" + i.rgb_sha + "\t" + std::to_string(i.w) + "\t" + std::to_string(i.h) +
               "\t" + std::to_string(i.best_w) + "\t" + std::to_string(i.best_h) + "\t" + std::to_string(i.nx) + "\t" +
               std::to_string(i.ny) + "\t" + std::to_string(i.n_tokens) + "\t" + std::to_string(i.n_pos));
    }
}

// ---- tower ---------------------------------------------------------------------------------------------------------

static const int TAP_BLOCKS[] = {0, 1, 13, 26};
static const char * TAP_NODES[] = {"ln1", "Qcur_rope", "attn_out", "ffn_inp", "ffn_out", "layer_out"};

static int run_tower(const std::string & mmproj, const std::string & text_model, const std::vector<image_arg> & images,
                     const std::string & out_pre, const std::string & out_taps, bool cpu, int threads, int min_tokens,
                     int max_tokens, long tap_patches, bool taps_final, const std::string & flags) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(text_model.c_str(), mp);
    if (!model) DIE("cannot load the vocabulary of %s", text_model.c_str());
    // a vocabulary-only model reports no width: the projector's output width, checked against the text file's header
    const int n_embd = (int) gguf_u32(mmproj, "clip.vision.projection_dim");
    {
        const std::string tarch = gguf_arch(text_model);
        const int n_text = (int) gguf_u32(text_model, tarch + ".embedding_length");
        if (n_text != n_embd) DIE("the mmproj projects to %d, the text model %s is %d wide", n_embd, text_model.c_str(), n_text);
    }
    tap_state taps;
    mtmd_context * clean = make_mtmd(mmproj, model, cpu, threads, min_tokens, max_tokens, nullptr);
    mtmd_context * tapc  = make_mtmd(mmproj, model, cpu, threads, min_tokens, max_tokens, &taps);
    if (!mtmd_decode_use_mrope(clean)) DIE("the text model is not an M-RoPE decoder");

    const std::string arch = gguf_arch(mmproj);
    std::unique_ptr<set_writer> pre;
    if (!out_pre.empty()) pre.reset(new set_writer(out_pre));
    set_writer tw(out_taps);
    std::vector<image_info> infos;
    std::string effect;
    std::vector<std::string> wanted;
    if (!taps_final) wanted.push_back("inp_pos_emb");
    wanted.push_back("norm_b-27");
    if (!taps_final) for (int b : TAP_BLOCKS) for (const char * n : TAP_NODES) wanted.push_back(std::string(n) + "-" + std::to_string(b));

    for (const auto & a : images) {
        fprintf(stderr, "dump_mtmd: tower %s\n", a.name.c_str());
        std::vector<uint8_t> png;
        if (!read_file(a.path, png)) DIE("cannot read %s", a.path.c_str());
        image_info info;
        info.name = a.name;
        info.png_sha = sha256_of(png.data(), png.size());

        image_chunk ic;
        tokenize_image(clean, a, ic);
        info.w = ic.w; info.h = ic.h; info.rgb_sha = ic.rgb_sha;
        info.nx = ic.nx; info.ny = ic.ny; info.n_tokens = ic.n_tokens; info.n_pos = std::max(ic.nx, ic.ny);
        std::vector<float> embd_clean;
        encode_image(clean, ic, n_embd, embd_clean);

        image_chunk it;
        tokenize_image(tapc, a, it);
        if (it.n_tokens != ic.n_tokens || it.w != ic.w || it.h != ic.h) DIE("%s: the two contexts disagree on the plan", a.name.c_str());
        const long patches = 4L * ic.n_tokens;
        const bool full = taps_final || patches <= tap_patches;
        taps.got.clear();
        taps.want.clear();
        taps.want_inp = true;
        taps.inp_done = false;
        if (full) taps.want.insert(wanted.begin(), wanted.end());
        taps.armed = true;
        std::vector<float> embd_tap;
        encode_image(tapc, it, n_embd, embd_tap);
        taps.armed = false;
        if (!taps.inp_done) DIE("%s: the first consumer of inp_raw was never seen", a.name.c_str());
        if (full) for (const auto & n : wanted) if (!taps.got.count(n)) DIE("%s: tap %s was never seen", a.name.c_str(), n.c_str());

        const tap_data & inp = taps.got["inp_raw"];
        info.best_w = (int) inp.ne[0];
        info.best_h = (int) inp.ne[1];
        if (inp.ne[2] != 3 || inp.ne[3] != 1) DIE("%s: inp_raw is not [W, H, 3, 1]", a.name.c_str());
        if ((long) (info.best_w / 32) * (info.best_h / 32) != ic.n_tokens) DIE("%s: inp_raw %d x %d is not %d tokens", a.name.c_str(), info.best_w, info.best_h, ic.n_tokens);
        if (pre) pre->tensor_f32(a.name + "/inp_raw", inp.op.c_str(), inp.ne, inp.v.data());

        size_t diff = 0;
        double maxd = 0.0;
        for (size_t i = 0; i < embd_clean.size(); ++i) {
            if (memcmp(&embd_clean[i], &embd_tap[i], 4) != 0) diff++;
            maxd = std::max(maxd, (double) std::fabs(embd_clean[i] - embd_tap[i]));
        }
        char b[256];
        snprintf(b, sizeof(b), "# tap_effect\t%s\tembd values %zu\tdiffering %zu\tmax_abs_diff %.9g\ttaps %s\n", a.name.c_str(),
                 embd_clean.size(), diff, maxd, taps_final ? "final" : (full ? "full" : "inp_raw only"));
        effect += b;

        const int64_t ne_e[4] = {n_embd, ic.n_tokens, 1, 1};
        if (full) {
            for (const auto & n : wanted) {
                const tap_data & d = taps.got[n];
                tw.tensor_f32(a.name + "/" + n, d.op.c_str(), d.ne, d.v.data());
            }
        }
        tw.tensor_f32(a.name + "/embd", "OUTPUT", ne_e, embd_clean.data());
        infos.push_back(info);
    }

    if (pre) {
        header_common(*pre, "preproc", mmproj, arch, text_model, mmproj, cpu, threads, min_tokens, max_tokens, flags,
                      "dump_mtmd tower — mtmd's preprocessed image (the graph input inp_raw), raw f32 channel-planar [W, H, 3, 1], little-endian");
        image_lines(*pre, infos);
        pre->finish();
    }
    header_common(tw, "taps", mmproj, arch, text_model, mmproj, cpu, threads, min_tokens, max_tokens, flags,
                  "dump_mtmd tower — mtmd's qwen3vl_merger graph nodes and final embeddings, raw f32, little-endian");
    if (taps_final) {
        tw.line("# taps\tnorm_b-27 (the post-LN output) of every image; embd (the clean pass)");
    } else {
        tw.line("# tap_patches\t" + std::to_string(tap_patches));
        tw.line("# taps\tblocks 0,1,13,26 of ln1 Qcur_rope attn_out ffn_inp ffn_out layer_out; inp_pos_emb; norm_b-27 (the post-LN output); embd (the clean pass)");
    }
    image_lines(tw, infos);
    for (size_t p = 0, q; p < effect.size(); p = q + 1) {
        q = effect.find('\n', p);
        tw.line(effect.substr(p, q - p));
    }
    tw.finish();
    mtmd_free(tapc);
    mtmd_free(clean);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ---- hidden --------------------------------------------------------------------------------------------------------

// --rows-from node: the context's cb_eval keeps the `result_norm` node of each ubatch, in order.
struct norm_capture {
    std::vector<float> v;
    int64_t            n_embd = 0;
};

static bool on_result_norm(struct ggml_tensor * t, bool ask, void * user) {
    if (strcmp(t->name, "result_norm") != 0) return false;
    if (ask) return true;
    auto * c = static_cast<norm_capture *>(user);
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || t->ne[0] != c->n_embd) DIE("result_norm is not a contiguous f32 [%lld, rows]", (long long) c->n_embd);
    const size_t at = c->v.size();
    c->v.resize(at + (size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, c->v.data() + at, 0, ggml_nbytes(t));
    return true;
}

struct row_sink {
    llama_context *       lctx = nullptr;
    norm_capture *        cap = nullptr;    // rows come from the captured node instead of the embeddings
    int                   n_embd = 0;
    std::vector<float> *  rows = nullptr;   // n_ids x n_embd
    std::vector<int32_t> * pos3 = nullptr;  // n_ids x 3; none: the positions are not recorded (the decode steps)
    size_t                at = 0;           // the next sequence index to fill
    size_t                limit = 0;
    int                   fed = 0;
    std::vector<float> *  last_logits = nullptr;  // when set, the logits of the last row decoded
};

static void take_rows(row_sink * s, int n, const int32_t * t, const int32_t * y, const int32_t * x) {
    if (s->at + (size_t) n > s->limit) DIE("decode produced rows past the prompt (%zu + %d > %zu)", s->at, n, s->limit);
    if (s->cap && s->cap->v.size() != (size_t) n * s->n_embd) {
        DIE("the batch at sequence index %zu captured %zu result_norm values, want %d x %d", s->at, s->cap->v.size(), n, s->n_embd);
    }
    for (int j = 0; j < n; ++j) {
        const float * e = s->cap ? s->cap->v.data() + (size_t) j * s->n_embd : llama_get_embeddings_ith(s->lctx, j);
        if (!e) DIE("no embedding row %d of the batch at sequence index %zu", j, s->at);
        for (int c = 0; c < s->n_embd; ++c) {
            if (!std::isfinite(e[c])) DIE("sequence index %zu value %d is not finite", s->at + j, c);
        }
        memcpy(s->rows->data() + (s->at + j) * s->n_embd, e, (size_t) s->n_embd * sizeof(float));
        if (s->pos3) {
            (*s->pos3)[(s->at + j) * 3 + 0] = t[j];
            (*s->pos3)[(s->at + j) * 3 + 1] = y[j];
            (*s->pos3)[(s->at + j) * 3 + 2] = x[j];
        }
    }
    if (s->cap) s->cap->v.clear();
    if (s->last_logits) {
        const float * l = llama_get_logits_ith(s->lctx, n - 1);
        if (!l) DIE("no logits for the last row of the batch at sequence index %zu", s->at + n - 1);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(s->lctx)));
        s->last_logits->assign(l, l + n_vocab);
    }
    s->at += (size_t) n;
}

// The positions of one image batch view: n_tokens (t, y, x, z) arrays, section-major, n_tokens long each.
static void take_image_batch(row_sink * s, int n, const llama_pos * pos) {
    std::vector<int32_t> t(pos, pos + n), y(pos + n, pos + 2 * n), x(pos + 2 * n, pos + 3 * n);
    take_rows(s, n, t.data(), y.data(), x.data());
    s->fed += n;
}

// mtmd_helper_post_decode_callback: after each llama_decode of an image batch view. An M-RoPE view lays its
// positions out as four arrays of n_tokens (t, y, x, z). Mainline hands the view as a llama_batch until the helper
// became mtmd_helper_embd_batch (build-clefvis.sh defines DUMP_MTMD_EMBD_BATCH for a tree whose header has it).
#ifdef DUMP_MTMD_EMBD_BATCH
static int32_t after_image_batch(const mtmd_helper_embd_batch * batch, void * user) {
    if (batch->n_pos != 4) DIE("an image batch view of %d position arrays, an M-RoPE view has 4", (int) batch->n_pos);
    take_image_batch(static_cast<row_sink *>(user), batch->n_tokens, batch->pos);
    return 0;
}
#else
static int32_t after_image_batch(llama_batch batch, void * user) {
    take_image_batch(static_cast<row_sink *>(user), batch.n_tokens, batch.pos);
    return 0;
}
#endif

#ifdef DUMP_MTMD_EMBD_BATCH
// The image chunk's rows through llama_process as the helper does (embedding rows, the M-RoPE position of each from
// mtmd_image_tokens_get_decoder_pos at n_past, one sequence), with the output flag on every row.
static void decode_image_rows(llama_context * lctx, row_sink * sink, const mtmd_input_chunk * chunk, const std::vector<float> & embd,
                              int n_embd, llama_pos n_past, int n_batch, llama_pos * new_past) {
    const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(chunk);
    const int n = (int) mtmd_image_tokens_get_n_tokens(it);
    if (n > n_batch) DIE("an image of %d tokens does not fit one batch of %d", n, n_batch);
    llama_batch_ext * b = llama_batch_ext_init(lctx);
    std::vector<int32_t> t(n), y(n), x(n);
    for (int i = 0; i < n; ++i) {
        const llama_embd e = {embd.data() + (size_t) i * n_embd, 1, (size_t) n_embd};
        const int32_t idx = llama_batch_ext_add_embd(b, 0, e);
        if (idx < 0) DIE("llama_batch_ext_add_embd refused row %d (%d)", i, (int) idx);
        const mtmd_decoder_pos d = mtmd_image_tokens_get_decoder_pos(it, n_past, (size_t) i);
        llama_pos p[4] = {(llama_pos) d.t, (llama_pos) d.y, (llama_pos) d.x, (llama_pos) d.z};
        llama_batch_ext_set_pos(b, idx, p);
        llama_batch_ext_set_output_logits(b, idx, true);
        t[i] = p[0]; y[i] = p[1]; x[i] = p[2];
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, b) != 0) DIE("llama_process failed on the image rows at position %d", (int) n_past);
    llama_batch_ext_free(b);
    take_rows(sink, n, t.data(), y.data(), x.data());
    sink->fed += n;
    *new_past = n_past + (llama_pos) mtmd_image_tokens_get_n_pos(it);
}
#endif

static void decode_text(llama_context * lctx, row_sink * sink, const std::vector<llama_token> & ids, size_t from, size_t to,
                        int n_batch, llama_pos & n_past) {
    llama_batch b = llama_batch_init(n_batch, 0, 1);
    size_t i = from;
    while (i < to) {
        b.n_tokens = 0;
        std::vector<int32_t> p;
        for (; i < to && b.n_tokens < n_batch; ++i) {
            const int j = b.n_tokens++;
            b.token[j] = ids[i];
            b.pos[j] = n_past;
            p.push_back(n_past);
            n_past++;
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            b.logits[j] = 1;
        }
        if (llama_decode(lctx, b) != 0) DIE("text decode failed at sequence index %zu", i);
        take_rows(sink, b.n_tokens, p.data(), p.data(), p.data());
    }
    llama_batch_free(b);
}

static std::vector<llama_token> read_ids(const std::string & path, std::string * sha) {
    std::vector<uint8_t> raw;
    if (!read_file(path, raw)) DIE("cannot read %s", path.c_str());
    *sha = sha256_of(raw.data(), raw.size());
    std::vector<llama_token> ids;
    std::istringstream in(std::string(raw.begin(), raw.end()));
    std::string line;
    long n = 0;
    while (std::getline(in, line)) {
        ++n;
        char * end = nullptr;
        errno = 0;
        const long v = strtol(line.c_str(), &end, 10);
        if (line.empty() || *end != '\0' || errno != 0 || v < 0 || v > INT32_MAX) DIE("%s:%ld is not a token id: '%s'", path.c_str(), n, line.c_str());
        ids.push_back((llama_token) v);
    }
    return ids;
}

// The experts of the first n_cpu_moe blocks stay in host memory (the pattern `--n-cpu-moe` of mainline's tools builds), so a
// model whose experts do not fit the card loads with its attention and shared weights on it. The strings outlive the load.
static std::vector<std::string> g_moe_patterns;
static std::vector<llama_model_tensor_buft_override> g_moe_overrides;

static void keep_experts_on_host(llama_model_params & mp, int n_cpu_moe) {
    if (n_cpu_moe <= 0) return;
    g_moe_patterns.reserve((size_t) n_cpu_moe);
    for (int i = 0; i < n_cpu_moe; ++i) {
        char pat[96];
        snprintf(pat, sizeof(pat), "blk\\.%d\\.ffn_(up|down|gate)_(ch|)exps", i);
        g_moe_patterns.emplace_back(pat);
        g_moe_overrides.push_back({g_moe_patterns.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
    g_moe_overrides.push_back({nullptr, nullptr});
    mp.tensor_buft_overrides = g_moe_overrides.data();
}

static int run_hidden(const std::string & mmproj, const std::string & text_model, const std::string & ids_path,
                      const std::vector<image_arg> & images, const std::string & out, const std::string & rows_mode,
                      const std::string & prose_path, int pad_id, int n_ctx, int n_ubatch, int ngl, int ncmoe, bool cpu,
                      int threads, int min_tokens, int max_tokens, int decode_steps, const std::string & out_decode,
                      bool rows_node, const std::string & flags) {
    std::string ids_sha;
    std::vector<llama_token> ids = read_ids(ids_path, &ids_sha);
    const size_t n_ids = ids.size();
    if (n_ids == 0) DIE("%s holds no ids", ids_path.c_str());
    if ((int) n_ids > n_ctx) DIE("%zu ids do not fit in -c %d", n_ids, n_ctx);
    std::vector<std::pair<size_t, size_t>> spans;   // (at, len) of each maximal run of the image pad id
    for (size_t i = 0; i < n_ids;) {
        if (ids[i] != pad_id) { ++i; continue; }
        size_t j = i;
        while (j < n_ids && ids[j] == pad_id) ++j;
        spans.emplace_back(i, j - i);
        i = j;
    }
    if (spans.size() != images.size()) DIE("%zu image span(s) in %s, %zu --image given", spans.size(), ids_path.c_str(), images.size());
    const bool prose = rows_mode == "prose", bf16 = rows_mode == "bf16";
    if (!prose && rows_mode != "mtmd" && !bf16) DIE("--rows is mtmd, bf16 or prose, got %s", rows_mode.c_str());
    std::vector<llama_token> prose_ids;
    std::string prose_sha = "-";
    if (prose) {
        if (prose_path.empty()) DIE("--rows prose needs --prose-ids");
        prose_ids = read_ids(prose_path, &prose_sha);
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    keep_experts_on_host(mp, ngl > 0 ? ncmoe : 0);
    llama_model * model = llama_model_load_from_file(text_model.c_str(), mp);
    if (!model) DIE("failed to load %s", text_model.c_str());
    const int n_embd = llama_model_n_embd_inp(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * pad_text = llama_vocab_get_text(vocab, pad_id);
    if (!pad_text || strcmp(pad_text, "<|image_pad|>") != 0) DIE("id %d is %s, not <|image_pad|>", pad_id, pad_text ? pad_text : "(none)");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) n_ctx;
    cp.n_ubatch = (uint32_t) std::min(n_ubatch, n_ctx);
    cp.n_seq_max = 1;
    cp.embeddings = !rows_node;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    norm_capture cap;
    cap.n_embd = n_embd;
    if (rows_node) {
        cp.cb_eval = on_result_norm;
        cp.cb_eval_user_data = &cap;
    }
    if (threads > 0) { cp.n_threads = threads; cp.n_threads_batch = threads; }
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) DIE("failed to make the context");

    mtmd_context * mctx = nullptr;
    if (!prose) {
        mctx = make_mtmd(mmproj, model, cpu, threads > 0 ? threads : 4, min_tokens, max_tokens, nullptr);
        if (!mtmd_decode_use_mrope(mctx)) DIE("the text model is not an M-RoPE decoder");
    }

    std::vector<float> rows((size_t) n_ids * n_embd);
    std::vector<int32_t> pos3(n_ids * 3);
    row_sink sink;
    sink.lctx = lctx; sink.n_embd = n_embd; sink.rows = &rows; sink.pos3 = &pos3; sink.limit = n_ids;
    if (rows_node) sink.cap = &cap;
    std::vector<float> logits;
    if (decode_steps > 0) sink.last_logits = &logits;

    std::vector<image_info> infos;
    std::string span_lines = "# span columns\tindex image at len nx ny n_pos start_pos end_pos\n";
    llama_pos n_past = 0;
    size_t i = 0, prose_at = 0;
    for (size_t k = 0; k <= spans.size(); ++k) {
        const size_t stop = k < spans.size() ? spans[k].first : n_ids;
        if (stop > i) {
            decode_text(lctx, &sink, ids, i, stop, n_ctx, n_past);
            i = stop;
        }
        if (k == spans.size()) break;
        const size_t len = spans[k].second;
        const image_arg & a = images[k];
        std::vector<uint8_t> png;
        if (!read_file(a.path, png)) DIE("cannot read %s", a.path.c_str());
        image_info info;
        info.name = a.name;
        info.png_sha = sha256_of(png.data(), png.size());
        const llama_pos start = n_past;
        if (prose) {
            if (prose_at + len > prose_ids.size()) DIE("--prose-ids holds %zu ids, the spans need %zu", prose_ids.size(), prose_at + len);
            std::vector<llama_token> span_ids(prose_ids.begin() + prose_at, prose_ids.begin() + prose_at + len);
            prose_at += len;
            decode_text(lctx, &sink, span_ids, 0, len, n_ctx, n_past);
            info.n_tokens = (int) len;
        } else {
            image_chunk ic;
            tokenize_image(mctx, a, ic);
            if ((size_t) ic.n_tokens != len) {
                DIE("%s: mtmd gives %d tokens, the prompt's span %zu holds %zu image pad ids", a.name.c_str(), ic.n_tokens, k, len);
            }
            info.w = ic.w; info.h = ic.h; info.rgb_sha = ic.rgb_sha;
            info.nx = ic.nx; info.ny = ic.ny; info.n_tokens = ic.n_tokens; info.n_pos = std::max(ic.nx, ic.ny);
            info.best_w = ic.nx * 32; info.best_h = ic.ny * 32;
            std::vector<float> embd;
            encode_image(mctx, ic, n_embd, embd);
            if (bf16) for (float & v : embd) v = bf16_round(v);
            llama_pos new_past = n_past;
            sink.fed = 0;
            if (rows_node) {
#ifdef DUMP_MTMD_EMBD_BATCH
                decode_image_rows(lctx, &sink, ic.chunk, embd, n_embd, n_past, n_ctx, &new_past);
#else
                DIE("--rows-from node needs the llama_batch_ext API of a tree that has mtmd_helper_embd_batch");
#endif
            } else {
                const int32_t r = mtmd_helper_decode_image_chunk(mctx, lctx, ic.chunk, embd.data(), n_past, 0, n_ctx, &new_past,
                                                                 after_image_batch, &sink);
                if (r != 0) DIE("mtmd_helper_decode_image_chunk failed (%d)", (int) r);
            }
            if (sink.fed != ic.n_tokens) DIE("the helper fed %d rows of %d", sink.fed, ic.n_tokens);
            n_past = new_past;
        }
        if (sink.at != i + len) DIE("span %zu: %zu rows collected, the span ends at %zu", k, sink.at, i + len);
        char b[256];
        snprintf(b, sizeof(b), "# span\t%zu\t%s\t%zu\t%zu\t%d\t%d\t%d\t%d\t%d\n", k, a.name.c_str(), i, len, info.nx, info.ny, info.n_pos,
                 (int) start, (int) n_past);
        span_lines += b;
        infos.push_back(info);
        i += len;
    }
    if (sink.at != n_ids) DIE("%zu rows collected of %zu ids", sink.at, n_ids);

    // The decode steps: greedy, in the context the prompt left, each token a text row at n_past.
    const llama_pos n_past_start = n_past;
    std::vector<int32_t> dec_ids, dec_past;
    std::vector<float> dec_rows;
    if (decode_steps > 0) {
        dec_rows.resize((size_t) decode_steps * n_embd);
        row_sink dsink;
        dsink.lctx = lctx; dsink.n_embd = n_embd; dsink.rows = &dec_rows; dsink.limit = (size_t) decode_steps;
        if (rows_node) dsink.cap = &cap;
        std::vector<float> dlogits;
        dsink.last_logits = &dlogits;
        const std::vector<float> * cur = &logits;
        for (int k = 0; k < decode_steps; ++k) {
            if (cur->empty()) DIE("decode step %d: no logits", k);
            int best = 0;
            for (size_t v = 0; v < cur->size(); ++v) {
                if (!std::isfinite((*cur)[v])) DIE("decode step %d: logit %zu is not finite", k, v);
                if ((*cur)[v] > (*cur)[best]) best = (int) v;
            }
            dec_ids.push_back(best);
            std::vector<llama_token> one(1, (llama_token) best);
            decode_text(lctx, &dsink, one, 0, 1, n_ctx, n_past);
            dec_past.push_back((int32_t) n_past);
            cur = &dlogits;
        }
        if (dsink.at != (size_t) decode_steps) DIE("%zu decode rows of %d steps", dsink.at, decode_steps);
    }

    set_writer w(out);
    char arch[128];
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) snprintf(arch, sizeof(arch), "unknown");
    const std::string kind = prose ? "prose" : (bf16 ? "bf16rows" : "hidden");
    header_common(w, kind, text_model, arch, text_model, mmproj, cpu, threads, min_tokens, max_tokens, flags,
                  "dump_mtmd hidden — llama.cpp mainline result_norm of every position of a Clef prompt, raw f32, little-endian");
    w.line("# tokens_file\t" + ids_path);
    w.line("# tokens_file_sha256\t" + ids_sha);
    w.line("# tokens_count\t" + std::to_string(n_ids));
    w.line("# rows\t" + rows_mode + "\tprose_ids\t" + prose_path + "\tsha256\t" + prose_sha);
    if (rows_node) w.line("# rows_from\tnode");
    w.line("# image_pad_id\t" + std::to_string(pad_id));
    w.line("# ctx\t" + std::to_string(n_ctx) + "\tubatch\t" + std::to_string(cp.n_ubatch) + "\tn_pos_end\t" + std::to_string((int) n_past));
    w.head += span_lines;
    if (!prose) image_lines(w, infos);
    const int64_t ne_r[4] = {n_embd, (int64_t) n_ids, 1, 1};
    w.tensor_f32("result_norm", "RMS_NORM", ne_r, rows.data());
    const int64_t ne_p[4] = {3, (int64_t) n_ids, 1, 1};
    w.tensor_i32("mrope_pos", "INPUT", ne_p, pos3);
    w.finish();
    if (decode_steps > 0) {
        set_writer dw(out_decode);
        header_common(dw, "decode", text_model, arch, text_model, mmproj, cpu, threads, min_tokens, max_tokens, flags,
                      "dump_mtmd hidden — llama.cpp mainline greedy decode steps after a prompt, raw f32, little-endian");
        dw.line("# tokens_file\t" + ids_path);
        dw.line("# tokens_file_sha256\t" + ids_sha);
        dw.line("# tokens_count\t" + std::to_string(n_ids));
        dw.line("# rows\t" + rows_mode + "\tprose_ids\t" + prose_path + "\tsha256\t" + prose_sha);
        if (rows_node) dw.line("# rows_from\tnode");
        dw.line("# image_pad_id\t" + std::to_string(pad_id));
        dw.line("# ctx\t" + std::to_string(n_ctx) + "\tubatch\t" + std::to_string(cp.n_ubatch) + "\tn_pos_end\t" + std::to_string((int) n_past));
        dw.line("# decode\tsteps\t" + std::to_string(decode_steps) + "\tn_past_start\t" + std::to_string((int) n_past_start) + "\tsampler\tgreedy, the first of equal logits");
        dw.head += span_lines;
        if (!prose) image_lines(dw, infos);
        const int64_t ne_n[4] = {decode_steps, 1, 1, 1};
        const int64_t ne_d[4] = {n_embd, decode_steps, 1, 1};
        dw.tensor_i32("ids", "ARGMAX", ne_n, dec_ids);
        dw.tensor_f32("result_norm", "RMS_NORM", ne_d, dec_rows.data());
        dw.tensor_i32("n_past", "INPUT", ne_n, dec_past);
        dw.finish();
    }
    if (mctx) mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

// ---- chat ----------------------------------------------------------------------------------------------------------

#ifdef DUMP_MTMD_CHAT
static std::string escape_line(const std::string & s) {
    std::string r;
    for (char c : s) {
        if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else if (c == '\t') r += "\\t";
        else if (c == '\r') r += "\\r";
        else r += c;
    }
    return r;
}

static int run_chat(const std::string & mmproj, const std::string & text_model, const std::string & requests_path,
                    const std::vector<image_arg> & images, const std::string & out, int pad_id, int threads, int min_tokens,
                    int max_tokens, const std::string & flags) {
    std::vector<uint8_t> raw;
    if (!read_file(requests_path, raw)) DIE("cannot read %s", requests_path.c_str());
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(text_model.c_str(), mp);
    if (!model) DIE("cannot load the vocabulary of %s", text_model.c_str());
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * pad_text = llama_vocab_get_text(vocab, pad_id);
    if (!pad_text || strcmp(pad_text, "<|image_pad|>") != 0) DIE("id %d is %s, not <|image_pad|>", pad_id, pad_text ? pad_text : "(none)");
    mtmd_context * mctx = make_mtmd(mmproj, model, true, threads, min_tokens, max_tokens, nullptr);
    if (!mtmd_decode_use_mrope(mctx)) DIE("the text model is not an M-RoPE decoder");
    const std::string marker = mtmd_default_marker();
    common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
    if (!tmpls) DIE("no chat template in %s", text_model.c_str());
    const std::string tmpl_src = common_chat_templates_source(tmpls.get());
    const std::string tmpl_sha = sha256_of((const uint8_t *) tmpl_src.data(), tmpl_src.size());
    // llama-server's defaults (server-context.cpp): jinja, thinking when the template has it, deepseek reasoning format
    const bool enable_thinking = common_chat_templates_support_enable_thinking(tmpls.get());
    const std::string arch = gguf_arch(text_model);

    std::map<std::string, std::string> by_name;
    for (const auto & a : images) by_name[a.name] = a.path;
    mkdir(out.c_str(), 0755);
    std::set<std::string> seen;
    std::istringstream in(std::string(raw.begin(), raw.end()));
    std::string line;
    long lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (line.empty()) continue;
        common_json req = common_json::parse(line);
        const std::string id = req.value("id", "");
        if (id.empty() || !seen.insert(id).second) DIE("%s:%ld: no id, or one used twice", requests_path.c_str(), lineno);
        common_json messages = req.at("messages");
        std::vector<std::string> used;   // the image names, in the order of the media markers
        for (size_t m = 0; m < messages.size(); ++m) {
            common_json & content = messages[m].at("content");
            if (!content.is_array()) continue;
            for (size_t k = 0; k < content.size(); ++k) {
                common_json & part = content[k];
                const std::string type = part.value("type", "");
                if (type == "image_url") {
                    const std::string name = part.at("image_url").value("url", "");
                    if (!by_name.count(name)) DIE("%s: request %s names the image %s, no --image gives it", requests_path.c_str(), id.c_str(), name.c_str());
                    used.push_back(name);
                    part["type"] = "media_marker";
                    part["text"] = marker;
                    part.erase("image_url");
                } else if (type != "text") {
                    DIE("request %s: content part type '%s' is not text or image_url", id.c_str(), type.c_str());
                }
            }
        }
        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(messages);
        inputs.use_jinja = true;
        inputs.add_generation_prompt = true;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs.enable_thinking = enable_thinking;
        inputs.force_pure_content = false;
        // prefill_assistant: a trailing assistant message continues instead of opening a new turn
        if (!inputs.messages.empty() && inputs.messages.back().role == "assistant") {
            inputs.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
            inputs.add_generation_prompt = false;
        }
        const std::string prompt = common_chat_templates_apply(tmpls.get(), inputs).prompt;

        std::vector<mtmd_bitmap *> bms;
        std::vector<std::string> rgb_sha, png_sha;
        std::vector<int> bw, bh;
        for (const auto & name : used) {
            const std::string & path = by_name[name];
            std::vector<uint8_t> png;
            if (!read_file(path, png)) DIE("cannot read %s", path.c_str());
            png_sha.push_back(sha256_of(png.data(), png.size()));
            mtmd_helper_bitmap_wrapper w = mtmd_helper_bitmap_init_from_file(mctx, path.c_str(), false, mtmd_helper_init_opt_default());
            if (!w.bitmap) DIE("cannot decode %s", path.c_str());
            if (w.video_ctx) DIE("%s decoded as a video", path.c_str());
            bms.push_back(w.bitmap);
            bw.push_back((int) mtmd_bitmap_get_nx(w.bitmap));
            bh.push_back((int) mtmd_bitmap_get_ny(w.bitmap));
            rgb_sha.push_back(sha256_of(mtmd_bitmap_get_data(w.bitmap), mtmd_bitmap_get_n_bytes(w.bitmap)));
        }
        mtmd_input_text text = {prompt.c_str(), prompt.size(), true, true};
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        std::vector<const mtmd_bitmap *> cbms(bms.begin(), bms.end());
        if (mtmd_tokenize(mctx, chunks, &text, cbms.data(), cbms.size()) != 0) DIE("request %s: mtmd_tokenize refused the prompt", id.c_str());

        std::vector<int32_t> ids;
        std::vector<image_info> infos;
        std::string span_lines = "# span columns\tindex image at len nx ny n_pos start_pos end_pos\n";
        llama_pos n_past = 0;
        size_t n_img = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
            const auto type = mtmd_input_chunk_get_type(c);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0;
                const llama_token * t = mtmd_input_chunk_get_tokens_text(c, &n);
                ids.insert(ids.end(), t, t + n);
                n_past += (llama_pos) n;
            } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                if (n_img >= used.size()) DIE("request %s: more image chunks than image parts", id.c_str());
                const size_t n = mtmd_input_chunk_get_n_tokens(c);
                const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(c);
                const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, n - 1);
                image_info info;
                info.name = used[n_img];
                info.png_sha = png_sha[n_img];
                info.rgb_sha = rgb_sha[n_img];
                info.w = bw[n_img]; info.h = bh[n_img];
                info.nx = (int) last.x + 1; info.ny = (int) last.y + 1;
                if ((size_t) info.nx * info.ny != n) DIE("request %s: grid %d x %d is not %zu tokens", id.c_str(), info.nx, info.ny, n);
                info.n_tokens = (int) n;
                info.n_pos = (int) mtmd_input_chunk_get_n_pos(c);
                if (info.n_pos != std::max(info.nx, info.ny)) DIE("request %s: n_pos is not max(nx, ny)", id.c_str());
                info.best_w = info.nx * 32; info.best_h = info.ny * 32;
                char b[256];
                snprintf(b, sizeof(b), "# span\t%zu\t%s\t%zu\t%zu\t%d\t%d\t%d\t%d\t%d\n", n_img, info.name.c_str(), ids.size(), n, info.nx,
                         info.ny, info.n_pos, (int) n_past, (int) n_past + info.n_pos);
                span_lines += b;
                ids.insert(ids.end(), n, pad_id);
                n_past += info.n_pos;
                infos.push_back(info);
                ++n_img;
            } else {
                DIE("request %s: a chunk that is neither text nor image", id.c_str());
            }
        }
        if (n_img != used.size()) DIE("request %s: %zu image part(s), %zu image chunk(s)", id.c_str(), used.size(), n_img);
        mtmd_input_chunks_free(chunks);
        for (auto * b : bms) mtmd_bitmap_free(b);

        // the ids, one a line: the file `hidden` reads
        {
            std::string txt;
            for (int32_t v : ids) txt += std::to_string(v) + "\n";
            write_raw(out + "/" + id + ".ids", txt.data(), txt.size());
        }
        set_writer w(out + "/" + id);
        header_common(w, "chatids", text_model, arch, text_model, mmproj, true, threads, min_tokens, max_tokens, flags,
                      "dump_mtmd chat — llama.cpp mainline's llama-server ids of a chat request, one id a row");
        w.line("# request_sha256\t" + sha256_of((const uint8_t *) line.data(), line.size()));
        w.line("# chat\ttemplate_sha256 " + tmpl_sha + "\tuse_jinja 1\tenable_thinking " + std::to_string((int) enable_thinking) +
               "\treasoning_format deepseek\tadd_generation_prompt " + std::to_string((int) inputs.add_generation_prompt) +
               "\tcontinue_final_message " + std::string(inputs.continue_final_message == COMMON_CHAT_CONTINUATION_NONE ? "none" : "auto"));
        w.line("# media_marker\t" + marker);
        w.line("# prompt_sha256\t" + sha256_of((const uint8_t *) prompt.data(), prompt.size()));
        w.line("# prompt\t" + escape_line(prompt));
        w.line("# tokens_count\t" + std::to_string(ids.size()));
        w.line("# image_pad_id\t" + std::to_string(pad_id));
        w.head += span_lines;
        image_lines(w, infos);
        const int64_t ne_i[4] = {(int64_t) ids.size(), 1, 1, 1};
        w.tensor_i32("ids", "TOKENS", ne_i, ids);
        w.finish();
    }
    mtmd_free(mctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
#endif

// ---- main ----------------------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    if (argc < 2) DIE("usage: dump_mtmd tower|hidden ... (see the file's header)");
    const std::string mode = argv[1];
    std::string mmproj, model, out, out_pre, ids, prose, rows = "mtmd", requests, out_decode;
    std::vector<image_arg> images;
    int ctx = 4096, ub = 512, ngl = 99, ncmoe = 0, threads = 0, pad = 248056, min_t = -1, max_t = -1, decode_steps = 0;
    long tap_patches = 4096;
    bool cpu = false, taps_final = false, rows_node = false;
    std::string flags;
    for (int i = 1; i < argc; ++i) flags += (i > 1 ? " " : "") + std::string(argv[i]);
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const bool more = i + 1 < argc;
        if (a == "--mmproj" && more) mmproj = argv[++i];
        else if (a == "-m" && more) model = argv[++i];
        else if (a == "--out" && more) out = argv[++i];
        else if (a == "--out-preproc" && more) out_pre = argv[++i];
        else if (a == "--out-taps" && more) out = argv[++i];
        else if (a == "--ids" && more) ids = argv[++i];
        else if (a == "--prose-ids" && more) prose = argv[++i];
        else if (a == "--rows" && more) rows = argv[++i];
        else if (a == "--image" && more) {
            const std::string v = argv[++i];
            const size_t eq = v.find('=');
            if (eq == std::string::npos || eq == 0) DIE("--image wants <name>=<png>, got %s", v.c_str());
            images.push_back({v.substr(0, eq), v.substr(eq + 1)});
        }
        else if (a == "--image-pad-id" && more) pad = atoi(argv[++i]);
        else if (a == "-c" && more) ctx = atoi(argv[++i]);
        else if (a == "-ub" && more) ub = atoi(argv[++i]);
        else if (a == "-ngl" && more) ngl = atoi(argv[++i]);
        else if (a == "-ncmoe" && more) ncmoe = atoi(argv[++i]);
        else if (a == "--requests" && more) requests = argv[++i];
        else if (a == "--decode" && more) decode_steps = atoi(argv[++i]);
        else if (a == "--out-decode" && more) out_decode = argv[++i];
        else if (a == "--rows-from" && more) {
            const std::string v = argv[++i];
            if (v != "embd" && v != "node") DIE("--rows-from is embd or node, got %s", v.c_str());
            rows_node = v == "node";
        }
        else if (a == "--taps" && more) {
            const std::string v = argv[++i];
            if (v != "full" && v != "final") DIE("--taps is full or final, got %s", v.c_str());
            taps_final = v == "final";
        }
        else if (a == "-t" && more) threads = atoi(argv[++i]);
        else if (a == "--image-min-tokens" && more) min_t = atoi(argv[++i]);
        else if (a == "--image-max-tokens" && more) max_t = atoi(argv[++i]);
        else if (a == "--tap-patches" && more) tap_patches = atol(argv[++i]);
        else if (a == "--mmproj-sha256" && more) g_mmproj_sha256 = argv[++i];
        else if (a == "--card" && more) g_card = argv[++i];
        else if (a == "--cpu") { cpu = true; ngl = 0; }
        else DIE("unknown or incomplete argument '%s'", a.c_str());
    }
    if (mmproj.empty() || model.empty() || out.empty() || images.empty()) DIE("--mmproj, -m, --out and at least one --image are required");
    if (mode == "tower") return run_tower(mmproj, model, images, out_pre, out, cpu, threads > 0 ? threads : 4, min_t, max_t, tap_patches, taps_final, flags);
    if (mode == "hidden") {
        if (ids.empty()) DIE("hidden needs --ids");
        if (decode_steps < 0 || (decode_steps > 0) != !out_decode.empty()) DIE("--decode <steps> and --out-decode <dir> go together");
        if (decode_steps > 0 && rows != "mtmd") DIE("--decode follows the tower's own rows: --rows mtmd");
        return run_hidden(mmproj, model, ids, images, out, rows, prose, pad, ctx, ub, ngl, ncmoe, cpu, threads, min_t, max_t, decode_steps, out_decode, rows_node, flags);
    }
    if (mode == "chat") {
#ifdef DUMP_MTMD_CHAT
        if (requests.empty()) DIE("chat needs --requests");
        return run_chat(mmproj, model, requests, images, out, pad, threads > 0 ? threads : 4, min_t, max_t, flags);
#else
        DIE("this binary has no chat mode: build it with DUMP_MTMD_CHAT (build-clefvis.sh does for a tree that has libllama-common)");
#endif
    }
    DIE("the mode is tower, hidden or chat, got %s", mode.c_str());
}
