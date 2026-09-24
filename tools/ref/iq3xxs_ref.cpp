// Run ik's OWN IQ3_XXS kernel — the pairing the oracle dispatches on this CPU — on the first
// aligned IQ3_XXS tensor's first 64 rows, with ik's own Q8_K coding of the column the Rust
// gate quantizes. Same shape and output format as q5k_x4_ref.cpp: a tensor header, one long
// hex line of ik's activation bytes, then `row R %08x` lines.
//
// The pairing: ggml.c:1116 — [GGML_TYPE_IQ3_XXS].vec_dot_type = GGML_TYPE_Q8_K;
// iqk_gemm_iquants.cpp:2749 — iqk_set_kernels_iquants accepts typeB = Q8_K only, and :2769
// picks set_functions<DequantizerIQ3XXS>, whose kernels[0] is
// mul_mat_qX_K_q8_K_IQ<DequantizerIQ3XXS, 1>, i.e. mul_mat_qX_K_q8_K_IQ_N<..., 1> on a build
// without HAVE_FANCY_SIMD (:1019, the box's znver3). The activation is coded by
// type_traits[Q8_K].from_float, the call the matmul makes (quantize_row_q8_K ->
// iqk_quantize_row_q8_K, iqk_quantize.cpp:3941): 296-byte block_q8_K {d, sum, qs[256],
// bsums[16]}, the layout the Rust side reads.
//
// The GGUF is argv[1]: build-qdot-ref.sh passes the V4-Flash first data shard, whose first
// IQ3_XXS tensor is blk.0.ffn_gate_exps.weight (k = 4096). The column is the first k = 4096
// f32 of the attn_norm-0 dump the other harnesses read (two tokens' worth of that
// 2048-per-token dump: the real value distribution, not a token-aligned column).
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_iquants.h"

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

static std::vector<uint8_t> read_range(const char *p, int64_t off, size_t n) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "iq3xxs_ref: open %s failed\n", p); exit(1); }
    std::vector<uint8_t> b(n);
    if (fseeko(f, (off_t)off, SEEK_SET) || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "iq3xxs_ref: read failed\n"); exit(1); }
    fclose(f);
    return b;
}

// Write to <path>.tmp.<pid> and rename, as q5k_x4_ref.cpp does: a gate on another track
// reading the shared $BLOOMERY_DATA never sees a half-written dump.
static bool write_atomic(const std::string &path, const void *data, size_t n) {
    const std::string tmp = path + ".tmp." + std::to_string(getpid());
    FILE *out = fopen(tmp.c_str(), "wb");
    if (!out) {
        fprintf(stderr, "iq3xxs_ref: cannot open %s for writing: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    const bool wrote = fwrite(data, 1, n, out) == n;
    if (fclose(out) != 0 || !wrote) {
        fprintf(stderr, "iq3xxs_ref: cannot write %s: %s\n", tmp.c_str(), strerror(errno));
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "iq3xxs_ref: cannot rename %s to %s: %s\n", tmp.c_str(), path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: iq3xxs_ref <model.gguf>  (a file with an IQ3_XXS tensor)\n");
        return 64;
    }
    const char *path = argv[1];

    struct ggml_context *gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context *gguf = gguf_init_from_file(path, gp);
    if (!gguf || !gctx) { fprintf(stderr, "iq3xxs_ref: gguf open failed: %s\n", path); return 1; }
    const size_t data_off = gguf_get_data_offset(gguf);

    // First IQ3_XXS tensor with k % 256 == 0 — the same scan the Rust gate does.
    int pick = -1;
    for (int i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        struct ggml_tensor *c = ggml_get_tensor(gctx, gguf_get_tensor_name(gguf, i));
        if (c && c->type == GGML_TYPE_IQ3_XXS && c->ne[0] % 256 == 0) { pick = i; break; }
    }
    if (pick < 0) { fprintf(stderr, "iq3xxs_ref: no IQ3_XXS tensor in %s\n", path); return 1; }
    // Copy the name BEFORE freeing: it points into the context.
    const std::string tensor_name = gguf_get_tensor_name(gguf, pick);
    struct ggml_tensor *t = ggml_get_tensor(gctx, tensor_name.c_str());
    const int k = (int)t->ne[0];
    const size_t rs = ggml_row_size(t->type, k);
    if (t->ne[1] * t->ne[2] * t->ne[3] < kDotRows) { fprintf(stderr, "iq3xxs_ref: %s has fewer than %d rows\n", tensor_name.c_str(), kDotRows); return 1; }
    fprintf(stderr, "tensor %s k=%d rowsz=%zu\n", tensor_name.c_str(), k, rs);
    std::vector<uint8_t> w = read_range(path, (int64_t)(data_off + gguf_get_tensor_offset(gguf, pick)), rs * kDotRows);
    gguf_free(gguf);
    ggml_free(gctx);

    FILE *f = fopen((std::string(kDataDir) + "/ref/attn_norm-0.0.f32").c_str(), "rb");
    if (!f) { fprintf(stderr, "iq3xxs_ref: no oracle dump\n"); return 1; }
    std::vector<float> x(k);
    if (fread(x.data(), 4, k, f) != (size_t)k) { fprintf(stderr, "iq3xxs_ref: short dump\n"); return 1; }
    fclose(f);

    ggml_type_traits_t q8k = ggml_internal_get_type_traits(GGML_TYPE_Q8_K);
    if (!q8k.from_float) { fprintf(stderr, "iq3xxs_ref: no from_float for q8_K\n"); return 1; }
    std::vector<uint8_t> y(ggml_row_size(GGML_TYPE_Q8_K, k));
    q8k.from_float(x.data(), y.data(), k);

    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_iquants(k, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q8_K, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "iq3xxs_ref: ik declined the IQ3_XXS x Q8_K pairing (k=%d)\n", k);
        return 1;
    }
    std::vector<float> dst(kDotRows);
    DataInfo info;
    info.s = dst.data();
    info.cy = (const char *)y.data();
    info.bs = kDotRows;    // one dst row of 64 outputs
    info.by = y.size();    // one activation row
    info.cur_y = 0;
    info.ne11 = 1;
    info.row_mapping = nullptr;
    info.bs2 = 0;
    // bx is the ROW STRIDE (BaseDequantizer::new_row walks vx + bx*ix, iqk_common.h:368).
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
    if (!write_atomic(std::string(kDataDir) + "/ref/iq3xxs-ik-dot.txt", dot.data(), dot.size())) return 1;
    printf("dumped %d rows of %s (q8_K pairing)\n", kDotRows, tensor_name.c_str());
    return 0;
}
