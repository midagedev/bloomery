// Run ik's OWN Q2_K kernel — the pairing the oracle dispatches on this CPU — on the first aligned
// Q2_K tensor's first 64 REAL rows plus 32 synthetic random-code rows, with ik's Q8_K coding of the
// column the Rust gate reads back. Same dump format as iq3s_ref.cpp: tensor header, ik's activation
// bytes hex, `row R <bits>` results, `w R <hex>` weight rows (all 96).
//
// The pairing: ggml.c:895 — [GGML_TYPE_Q2_K].vec_dot_type = GGML_TYPE_Q8_K;
// iqk_set_kernels_kquants (iqk_gemm_kquants.cpp, set_functions :1816) accepts typeB = Q8_K, and
// its kernels[0] is mul_mat_qY_K_q8_K_T<DequantizerQ2K, 1> (iqk_gemm_kquants.cpp:678, :511) on a
// build without AVX512 (the box's znver3). The activation is coded by type_traits[Q8_K].from_float,
// the call the matmul makes: 296-byte block_q8_K {d, sum, qs[256], bsums[16]}, the layout the Rust
// side reads.
//
// The GGUF is argv[1]: build-qdot-ref.sh passes the GLM-5.3-Flash UD-Q2_K_XL fourth shard, whose
// first Q2_K tensor is the NextN layer's blk.45.ffn_gate_exps.weight. The dump's tensor name is that
// real name, though 32 of its 96 rows are not that tensor's: after the 64 real rows come
// kRandomRows rows of synthetic random codes, every byte random, so every scale nibble and every
// min nibble (the 16 scale bytes) and every 2-bit code is hit; then each block's two f16 fields
// (d at byte 80, dmin at byte 82 of the 84-byte block) are replaced by finite positive halves in
// [2^-10, 2^-2). The dump has no per-row marker: rows 64..95 are the synthetic ones. The column is
// the first k f32 of the attn_norm-0 dump the other harnesses read: the real value distribution.
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_kquants.h"

#include <cerrno>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "ref_paths.h"

static const char *kDataDir = ref_data_dir();
static const int kRealRows = 64;
static const int kRandomRows = 32;
static const int kDotRows = kRealRows + kRandomRows;
static const size_t kBlock = 84;  // sizeof(block_q2_K)
static const size_t kDOff = 80;     // block_q2_K::d
static const size_t kDminOff = 82;  // block_q2_K::dmin
static const uint64_t kSeed = 0x1a2b3c4d5e6f7788ull ^ 22;

struct Lcg {
    uint64_t s;
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t)(s >> 32);
    }
};

// A finite positive half in [2^-10, 2^-2): an exponent in -10..-3 and a 10-bit mantissa, which an
// f16 holds exactly. The two draws are separate statements: argument order is unspecified in C++.
static ggml_fp16_t random_scale(Lcg &rng) {
    const int e = -10 + (int)(rng.next() % 8);
    const float m = 1.0f + (float)(rng.next() % 1024) / 1024.0f;
    return ggml_fp32_to_fp16(std::ldexp(m, e));
}

static std::vector<uint8_t> read_range(const char *p, int64_t off, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "q2k_ref: open %s failed\n", p); exit(1); }
    std::vector<uint8_t> b(n);
    if (fseeko(f, (off_t)off, SEEK_SET) || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "q2k_ref: read failed\n"); exit(1); }
    fclose(f);
    return b;
}

// Write to <path>.tmp.<pid> and rename, as q5k_x4_ref.cpp does: a gate on another track
// reading the shared $BLOOMERY_DATA never sees a half-written dump.
static bool write_atomic(const std::string &path, const void *data, size_t n) {
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    FILE *out = fopen(tmp.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "q2k_ref: cannot open %s for writing: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    const bool wrote = fwrite(data, 1, n, out) == n;
    if (fclose(out) != 0 || !wrote) {
        fprintf(stderr, "q2k_ref: cannot write %s: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "q2k_ref: cannot rename %s to %s: %s\n", tmp.c_str(), path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: q2k_ref <model.gguf>  (a file with a Q2_K tensor)\n");
        return 64;
    }
    const char *path = argv[1];

    struct ggml_context *gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context *gguf = gguf_init_from_file(path, gp);
    if (!gguf || !gctx) { fprintf(stderr, "q2k_ref: gguf open failed: %s\n", path); return 1; }
    const size_t data_off = gguf_get_data_offset(gguf);

    // First Q2_K tensor with k % 256 == 0 — the same scan the Rust gate does.
    int pick = -1;
    for (int i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        struct ggml_tensor *c = ggml_get_tensor(gctx, gguf_get_tensor_name(gguf, i));
        if (c && c->type == GGML_TYPE_Q2_K && c->ne[0] % 256 == 0) { pick = i; break; }
    }
    if (pick < 0) { fprintf(stderr, "q2k_ref: no Q2_K tensor in %s\n", path); return 1; }
    // Copy the name BEFORE freeing: it points into the context.
    const std::string tensor_name = gguf_get_tensor_name(gguf, pick);
    struct ggml_tensor *t = ggml_get_tensor(gctx, tensor_name.c_str());
    const int k = (int)t->ne[0];
    const size_t rs = ggml_row_size(t->type, k);
    if (rs != kBlock * (k / 256)) { fprintf(stderr, "q2k_ref: row size %zu, not %zu blocks of %zu\n", rs, (size_t)(k / 256), kBlock); return 1; }
    if (t->ne[1] * t->ne[2] * t->ne[3] < kRealRows) { fprintf(stderr, "q2k_ref: %s has fewer than %d rows\n", tensor_name.c_str(), kRealRows); return 1; }
    fprintf(stderr, "tensor %s k=%d rowsz=%zu\n", tensor_name.c_str(), k, rs);
    std::vector<uint8_t> w = read_range(path, (int64_t)(data_off + gguf_get_tensor_offset(gguf, pick)), rs * kRealRows);
    gguf_free(gguf);
    ggml_free(gctx);

    // ggml_init fills the f16 -> f32 table GGML_FP16_TO_FP32 reads on this build.
    struct ggml_init_params ip = {1024, nullptr, true};
    struct ggml_context *init_ctx = ggml_init(ip);
    if (!init_ctx) { fprintf(stderr, "q2k_ref: ggml_init failed\n"); return 1; }

    // Random-code rows: every byte random, then each block's f16 d and dmin replaced by a finite
    // positive half in [2^-10, 2^-2).
    Lcg rng{kSeed};
    w.resize((size_t)kDotRows * rs);
    for (size_t i = (size_t)kRealRows * rs; i < w.size(); ++i) w[i] = (uint8_t)rng.next();
    for (size_t b = (size_t)kRealRows * rs; b < w.size(); b += kBlock) {
        const ggml_fp16_t d = random_scale(rng);
        memcpy(&w[b + kDOff], &d, sizeof d);
        const ggml_fp16_t dmin = random_scale(rng);
        memcpy(&w[b + kDminOff], &dmin, sizeof dmin);
    }

    FILE *f = fopen((std::string(kDataDir) + "/ref/attn_norm-0.0.f32").c_str(), "rb");
    if (!f) { fprintf(stderr, "q2k_ref: no oracle dump\n"); return 1; }
    std::vector<float> x(k);
    if (fread(x.data(), 4, k, f) != (size_t)k) { fprintf(stderr, "q2k_ref: short dump\n"); return 1; }
    fclose(f);

    ggml_type_traits_t q8k = ggml_internal_get_type_traits(GGML_TYPE_Q8_K);
    if (!q8k.from_float) { fprintf(stderr, "q2k_ref: no from_float for q8_K\n"); return 1; }
    std::vector<uint8_t> y(ggml_row_size(GGML_TYPE_Q8_K, k));
    q8k.from_float(x.data(), y.data(), k);

    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_kquants(k, GGML_TYPE_Q2_K, GGML_TYPE_Q8_K, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "q2k_ref: ik declined the Q2_K x Q8_K pairing (k=%d)\n", k);
        return 1;
    }
    std::vector<float> dst(kDotRows);
    DataInfo info;
    info.s = dst.data();
    info.cy = (const char *)y.data();
    info.bs = kDotRows;    // one dst row of kDotRows outputs
    info.by = y.size();    // one activation row
    info.cur_y = 0;
    info.ne11 = 1;
    info.row_mapping = nullptr;
    info.bs2 = 0;
    // bx is the ROW STRIDE (BaseDequantizer::new_row walks vx + bx*ix).
    kernels[0](k, w.data(), rs, info, kDotRows);

    std::string dot = "tensor " + tensor_name + " k " + std::to_string(k) + "\n";
    char line[32];
    for (size_t i = 0; i < y.size(); ++i) {
        snprintf(line, sizeof line, "%02x", y[i]);
        dot += line;
    }
    dot += "\n";
    for (int r = 0; r < kDotRows; ++r) {
        uint32_t bits; memcpy(&bits, &dst[r], 4);
        snprintf(line, sizeof line, "row %d %08x\n", r, bits);
        dot += line;
    }
    for (int r = 0; r < kDotRows; ++r) {
        dot += "w " + std::to_string(r) + " ";
        for (size_t i = 0; i < rs; ++i) {
            snprintf(line, sizeof line, "%02x", w[(size_t)r * rs + i]);
            dot += line;
        }
        dot += "\n";
    }
    if (!write_atomic(std::string(kDataDir) + "/ref/q2k-ik-dot.txt", dot.data(), dot.size())) return 1;
    printf("dumped %d rows of %s (%d real + %d synthetic random-code), k=%d (q8_K pairing)\n",
           kDotRows, tensor_name.c_str(), kRealRows, kRandomRows, k);
    ggml_free(init_ctx);
    return 0;
}
