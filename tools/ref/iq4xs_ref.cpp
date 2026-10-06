// Run ik's OWN IQ4_XS kernel — the pairing the oracle dispatches on this CPU — on the
// first aligned IQ4_XS tensor's first rows, with ik's Q8_K coding of the column the Rust
// gate quantizes. Same dump format as iq4nl_ref.cpp: tensor header, ik's activation bytes
// hex, `w R <hex>` weight rows, `row R <bits>` results.
//
// The pairing: ggml.c:1302 — [GGML_TYPE_IQ4_XS].vec_dot_type = GGML_TYPE_Q8_K;
// iqk_gemm_kquants.cpp:3143-3145 — the entry requires ne00 % QK_K == 0 and typeB = Q8_K,
// and case IQ4_XS (:3174-3175) picks set_functions<DequantizerIQ4XS>, which on a build
// without HAVE_FANCY_SIMD (:1835-1839) is mul_mat_qX_K_q8_K_T<DequantizerIQ4XS, nrc_y>
// (:644) with Q8<nrc_y, block_q8_K>: make_scales over the 6-bit scale split, the +128
// kvalues table, the -128 bsums fold.
//
// The GGUF is argv[1]: build-qdot-ref.sh passes the Qwen3.8 UD-Q3_K_XL second shard
// (BLOOMERY_QWEN_Q3_MODEL), whose IQ4_XS tensors are the one layer's ffn_gate/up_exps.
// That file downloads beside this round: until `<argv[1]>.done` exists the harness dumps
// SYNTHETIC blocks under the name synthetic-iq4_xs and says so — a partially downloaded
// shard has the header but not the rows, and reading past EOF is not a reference. The
// synthetic rows are the same filler as iq4xs_rate.cpp (any bytes are valid codes; each
// block's f16 d masked finite), so the gates still bite on the kernel-vs-mirror and
// kernel-vs-ik contracts until the real rows land.
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_kquants.h"

#include <cerrno>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "ref_paths.h"

static const char *kDataDir = ref_data_dir();
static const int kDotRows = 64;
static const int kSynthK = 2560;

static std::vector<uint8_t> read_range(const char *p, int64_t off, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "iq4xs_ref: open %s failed\n", p); exit(1); }
    std::vector<uint8_t> b(n);
    if (fseeko(f, (off_t)off, SEEK_SET) || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "iq4xs_ref: read failed\n"); exit(1); }
    fclose(f);
    return b;
}

// Write to <path>.tmp.<pid> and rename, as q5k_x4_ref.cpp does.
static bool write_atomic(const std::string &path, const void *data, size_t n) {
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    FILE *out = fopen(tmp.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "iq4xs_ref: cannot open %s for writing: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    const bool wrote = fwrite(data, 1, n, out) == n;
    if (fclose(out) != 0 || !wrote) {
        fprintf(stderr, "iq4xs_ref: cannot write %s: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "iq4xs_ref: cannot rename %s to %s: %s\n", tmp.c_str(), path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: iq4xs_ref <model.gguf>  (a file with an IQ4_XS tensor)\n");
        return 64;
    }
    const char *path = argv[1];
    const std::string done = std::string(path) + ".done";

    std::string tensor_name;
    int k = 0;
    size_t rs = 0;
    int dot_rows = kDotRows;
    std::vector<uint8_t> w;

    if (access(done.c_str(), R_OK) == 0) {
        struct ggml_context *gctx = nullptr;
        struct gguf_init_params gp = {true, &gctx};
        struct gguf_context *gguf = gguf_init_from_file(path, gp);
        if (!gguf || !gctx) { fprintf(stderr, "iq4xs_ref: gguf open failed: %s\n", path); return 1; }
        const size_t data_off = gguf_get_data_offset(gguf);

        // First IQ4_XS tensor with k % 256 == 0 and enough rows.
        int pick = -1;
        for (int i = 0; i < gguf_get_n_tensors(gguf); ++i) {
            struct ggml_tensor *c = ggml_get_tensor(gctx, gguf_get_tensor_name(gguf, i));
            if (c && c->type == GGML_TYPE_IQ4_XS && c->ne[0] % 256 == 0
                    && c->ne[1] * c->ne[2] * c->ne[3] >= 16) { pick = i; break; }
        }
        if (pick < 0) { fprintf(stderr, "iq4xs_ref: no aligned IQ4_XS tensor in %s\n", path); return 1; }
        tensor_name = gguf_get_tensor_name(gguf, pick);
        struct ggml_tensor *t = ggml_get_tensor(gctx, tensor_name.c_str());
        k = (int)t->ne[0];
        rs = ggml_row_size(t->type, k);
        const int nrows = (int)(t->ne[1] * t->ne[2] * t->ne[3]);
        dot_rows = nrows < kDotRows ? nrows : kDotRows;
        fprintf(stderr, "tensor %s k=%d rows=%d rowsz=%zu\n", tensor_name.c_str(), k, nrows, rs);
        w = read_range(path, (int64_t)(data_off + gguf_get_tensor_offset(gguf, pick)), rs * dot_rows);
        gguf_free(gguf);
        ggml_free(gctx);
    } else {
        // Synthetic fallback: the same xorshift64* filler as iq4xs_rate.cpp, every
        // block's f16 d (second byte) masked finite.
        tensor_name = "synthetic-iq4_xs";
        k = kSynthK;
        rs = 136 * (k / 256);
        fprintf(stderr, "iq4xs_ref: %s.done is not there — synthetic blocks, k=%d\n", path, k);
        unsigned long long s = 0x9E3779B97F4A7C15ull;
        auto next = [&]() {
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            return s * 0x2545F4914F6CDD1Dull;
        };
        w.assign((size_t)dot_rows * rs, 0);
        for (size_t i = 0; i + 8 <= w.size(); i += 8) { const unsigned long long v = next(); memcpy(&w[i], &v, 8); }
        for (size_t b = 0; b < w.size(); b += 136) w[b + 1] &= 0x7b;
    }

    FILE *f = fopen((std::string(kDataDir) + "/ref/attn_norm-0.0.f32").c_str(), "rb");
    if (!f) { fprintf(stderr, "iq4xs_ref: no oracle dump\n"); return 1; }
    std::vector<float> x(k);
    if (fread(x.data(), 4, k, f) != (size_t)k) { fprintf(stderr, "iq4xs_ref: short dump\n"); return 1; }
    fclose(f);

    ggml_type_traits_t q8k = ggml_internal_get_type_traits(GGML_TYPE_Q8_K);
    if (!q8k.from_float) { fprintf(stderr, "iq4xs_ref: no from_float for q8_K\n"); return 1; }
    std::vector<uint8_t> y(ggml_row_size(GGML_TYPE_Q8_K, k));
    q8k.from_float(x.data(), y.data(), k);

    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_kquants(k, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_K, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "iq4xs_ref: ik declined the IQ4_XS x Q8_K pairing (k=%d)\n", k);
        return 1;
    }
    std::vector<float> dst(dot_rows);
    DataInfo info;
    info.s = dst.data();
    info.cy = (const char *)y.data();
    info.bs = dot_rows;    // one dst row of dot_rows outputs
    info.by = y.size();    // one activation row
    info.cur_y = 0;
    info.ne11 = 1;
    info.row_mapping = nullptr;
    info.bs2 = 0;
    // bx is the ROW STRIDE (BaseDequantizer::new_row walks vx + bx*ix).
    kernels[0](k, w.data(), rs, info, dot_rows);

    std::string dot = "tensor " + tensor_name + " k " + std::to_string(k) + "\n";
    char line[32];
    for (size_t i = 0; i < y.size(); ++i) {
        snprintf(line, sizeof line, "%02x", y[i]);
        dot += line;
    }
    dot += "\n";
    for (int r = 0; r < dot_rows; ++r) {
        uint32_t bits; memcpy(&bits, &dst[r], 4);
        snprintf(line, sizeof line, "row %d %08x\n", r, bits);
        dot += line;
    }
    for (int r = 0; r < dot_rows; ++r) {
        dot += "w " + std::to_string(r) + " ";
        for (size_t i = 0; i < rs; ++i) {
            snprintf(line, sizeof line, "%02x", w[(size_t)r * rs + i]);
            dot += line;
        }
        dot += "\n";
    }
    if (!write_atomic(std::string(kDataDir) + "/ref/iq4xs-ik-dot.txt", dot.data(), dot.size())) return 1;
    printf("dumped %d rows of %s (q8_K pairing)\n", dot_rows, tensor_name.c_str());
    return 0;
}
