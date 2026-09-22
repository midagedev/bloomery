// MUL-32: run ik's OWN Q5_0 kernel on the first Q5_0 tensor's first 64
// rows, with ik's quantize_row_q8_2_x4 coding the same attn_norm-0 column
// the Rust gate quantizes.
//
// The pairing this dump records (found this round, 2026-09-20):
//   * ggml.c:767 — [GGML_TYPE_Q5_0].vec_dot_type = GGML_TYPE_Q8_2_X4 under
//     __AVX2__ + GGML_USE_IQK_MULMAT (the box's build; aarch64/non-ik
//     builds say Q8_0_X4/Q8_0 and are not this engine's oracle);
//   * iqk_mul_mat.cpp:945 — MulMat::prepare routes Q5_0 to
//     iqk_set_kernels_legacy_quants;
//   * iqk_gemm_legacy_quants.cpp:2482/2494 — expected_type_B = Q8_2_X4 and
//     case Q5_0 picks Q5_0_1_Unpacker, i.e.
//     mul_mat_qX_1_q8_2_T<Q5_0_1_Unpacker, nrc_y> (set_functions, same
//     file, line 2453). Entry requires ne00 % QK8_0(32) == 0 — the model's
//     every Q5_0 tensor is ffn_down_exps with k = 1408 = 32*44, so the
//     256-value super-block contract this crate's other kernels carry
//     never fires here; 1408 % 128 == 0, so the x4 groups are whole too.
//
// Output file format matches q4k-x4-ik-dot.txt (tensor header, one long
// hex line of ik's activation bytes, `row R %08x` lines) so the Rust side
// parses both dumps the same way. Extraction pattern copied from
// q4k_x4_ref.cpp.
#include "ggml.h"
#include "ggml-backend.h"

#define IQK_IMPLEMENT
#include "iqk_gemm_legacy_quants.h"

#include <cerrno>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The data directory the gates read (tools/box.sh exports BLOOMERY_DATA; a parallel
// track moves both this harness's input and its dump by setting it).
static const char *kDataDir = getenv("BLOOMERY_DATA") ? getenv("BLOOMERY_DATA") : "/root/bloomery-data";

static const char *kGgufPath = "/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf";

static std::vector<uint8_t> read_range(const std::string &p, int64_t off, size_t n) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", p.c_str()); exit(1); }
    std::vector<uint8_t> b(n);
    if (fseek(f, off, SEEK_SET) || fread(b.data(), 1, n, f) != n) { fprintf(stderr, "read failed\n"); exit(1); }
    fclose(f);
    return b;
}

int main() {
    struct ggml_context *gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context *gguf = gguf_init_from_file(kGgufPath, gp);
    if (!gguf || !gctx) { fprintf(stderr, "gguf open failed\n"); return 1; }
    const size_t data_off = gguf_get_data_offset(gguf);

    // First Q5_0 tensor with k % 32 == 0 and k % 128 == 0 — the same scan
    // the Rust gate does (every Q5_0 in this model is ffn_down_exps, k=1408).
    int pick = -1;
    for (int i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        const char *nm = gguf_get_tensor_name(gguf, i);
        struct ggml_tensor *c = ggml_get_tensor(gctx, nm);
        if (c && c->type == GGML_TYPE_Q5_0 && c->ne[0] % 32 == 0 && c->ne[0] % 128 == 0) { pick = i; break; }
    }
    if (pick < 0) { fprintf(stderr, "no Q5_0 tensor\n"); return 1; }
    // Copy the name BEFORE freeing: gguf_get_tensor_name points into the
    // context, so a name read after gguf_free is freed memory.
    std::string tensor_name = gguf_get_tensor_name(gguf, pick);
    struct ggml_tensor *t = ggml_get_tensor(gctx, tensor_name.c_str());
    const int k = (int)t->ne[0];
    const size_t rs = ggml_row_size(t->type, k);
    fprintf(stderr, "tensor %s k=%d rowsz=%zu\n", tensor_name.c_str(), k, rs);
    std::vector<uint8_t> w = read_range(kGgufPath, (int64_t)(data_off + gguf_get_tensor_offset(gguf, pick)), rs * 64);
    gguf_free(gguf);
    ggml_free(gctx);
    const char *nm = tensor_name.c_str();

    FILE *f = fopen((std::string(kDataDir) + "/ref/attn_norm-0.0.f32").c_str(), "rb");
    if (!f) { fprintf(stderr, "no oracle dump\n"); return 1; }
    std::vector<float> x(k);
    if (fread(x.data(), 4, k, f) != (size_t)k) { fprintf(stderr, "short dump\n"); return 1; }
    fclose(f);

    // ik's own q8_2_x4 coding of the same column: 144 bytes per 128 values
    // (k % 128 == 0 means no tail blocks).
    std::vector<uint8_t> y(144 * (k / 128));
    quantize_row_q8_2_x4(x.data(), y.data(), k);

    // The Q5_0 kernel through ik's own dispatch table: kernels[0] is the
    // nrc_y = 1 instantiation of mul_mat_qX_1_q8_2_T<Q5_0_1_Unpacker>.
    std::array<mul_mat_t, IQK_MAX_NY> kernels{};
    mul_mat_t func16 = nullptr;
    if (!iqk_set_kernels_legacy_quants(k, GGML_TYPE_Q5_0, GGML_TYPE_Q8_2_X4, kernels, func16) || !kernels[0]) {
        fprintf(stderr, "ik declined the Q5_0 x Q8_2_X4 pairing (k=%d)\n", k);
        return 1;
    }
    std::vector<float> dst(64);
    DataInfo info;
    info.s = dst.data();
    info.cy = (const char *)y.data();
    info.bs = 64;          // one dst row of 64 outputs
    info.by = y.size();    // one activation row
    info.cur_y = 0;
    info.ne11 = 1;
    info.row_mapping = nullptr;
    info.bs2 = 0;
    // bx is the ROW STRIDE, not an offset — BaseDequantizer::new_row walks
    // `vx + bx*ix` (iqk_common.h:368), and the real caller passes the row
    // size (iqk_mul_mat.cpp:96). Zero here computed row 0 sixty-four times.
    kernels[0](k, w.data(), rs, info, 64);

    // Write to <path>.tmp and rename: a reader (gate-qdot on another track, same
    // shared $BLOOMERY_DATA) must never see a half-written dump. rename(2) within
    // one directory is atomic, so the final path is either the old dump or the new one; the pid
    // suffix keeps two tracks running build-ref at once from renaming each other's half file.
    const std::string out_p = std::string(kDataDir) + "/ref/q5f0-ik-dot.txt";
    const std::string tmp_p = out_p + ".tmp." + std::to_string(getpid());
    FILE *out = fopen(tmp_p.c_str(), "w");
    if (!out) {
        fprintf(stderr, "q5f0_ref: cannot open %s for writing: %s\n", tmp_p.c_str(), strerror(errno));
        return 1;
    }
    fprintf(out, "tensor %s k %d\n", nm, k);
    for (size_t i = 0; i < y.size(); ++i) fprintf(out, "%02x", y[i]);
    fprintf(out, "\n");
    for (int r = 0; r < 64; ++r) {
        uint32_t bits; memcpy(&bits, &dst[r], 4);
        fprintf(out, "row %d %08x\n", r, bits);
    }
    if (fclose(out) != 0) {
        fprintf(stderr, "q5f0_ref: cannot write %s: %s\n", tmp_p.c_str(), strerror(errno));
        return 1;
    }
    if (rename(tmp_p.c_str(), out_p.c_str()) != 0) {
        fprintf(stderr, "q5f0_ref: cannot rename %s to %s: %s\n", tmp_p.c_str(), out_p.c_str(), strerror(errno));
        return 1;
    }
    printf("dumped 64 rows of %s (q8_2_x4 pairing)\n", nm);
    return 0;
}
