// iq_quant_ref.cpp — ggml's IQ1_S / IQ2_XXS / Q4_K / Q5_K / Q6_K / Q8_0 quantizer as the oracle of gate_iq_quant
// (bloomery_gpu::iq_quant, gguf::iq_quant and gguf::kquant).
//
// Links the K3.1 quantization tree's libggml-base (tools/ref/build-k3-quant-tree.sh, pin 78feec062): the library
// llama-quantize calls, built for baseline x86-64 (no FMA, no F16C), whose qsort is glibc's stable merge sort.
//
//   iq_quant_ref <in> <out>
//
// <in>, little-endian: the magic "IQQ1"; u32 dst type and u32 src type (ggml ids: IQ2_XXS 16, IQ1_S 19, Q8_0 8, Q4_K 12,
// Q5_K 13, Q6_K 14 / F32 0, F16 1, BF16 30, MXFP4 39); u64 n_per_row, nrows, rows_per_expert, n_expert; then nrows source
// rows of the src type; then n_expert x n_per_row f32 importance values. n_expert 0 means no importance matrix (a null
// quant_weights, ggml's `_ref` rules for the K types); the IQ types refuse it as llama-quantize does. The source rows are widened to f32 by ggml's own to_float (what
// llama-quantize does to a tensor it requantizes), and each expert's rows are quantized by ggml_quantize_chunk with
// that expert's importance row, as llama_tensor_quantize_impl calls it. <out> gets the rows' block bytes in order.
//
// Build: bash tools/ref/build-iq-quant-ref.sh (on the box). Exit: 0 written; 2 a usage or input error, named.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ggml.h"

static int fail(const char * what) {
    std::fprintf(stderr, "iq_quant_ref: %s\n", what);
    return 2;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        return fail("usage: iq_quant_ref <in> <out>");
    }
    FILE * f = std::fopen(argv[1], "rb");
    if (!f) {
        return fail("cannot open the input");
    }
    char magic[4];
    uint32_t dst = 0, src = 0;
    uint64_t n = 0, nrows = 0, per = 0, nexp = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "IQQ1", 4) == 0 &&
              std::fread(&dst, 4, 1, f) == 1 && std::fread(&src, 4, 1, f) == 1 && std::fread(&n, 8, 1, f) == 1 &&
              std::fread(&nrows, 8, 1, f) == 1 && std::fread(&per, 8, 1, f) == 1 && std::fread(&nexp, 8, 1, f) == 1;
    if (!ok) {
        return fail("input header is not IQQ1");
    }
    const auto dt = (ggml_type) dst;
    const auto st = (ggml_type) src;
    const bool iq = dt == GGML_TYPE_IQ1_S || dt == GGML_TYPE_IQ2_XXS;
    if (!iq && dt != GGML_TYPE_Q4_K && dt != GGML_TYPE_Q5_K && dt != GGML_TYPE_Q6_K && dt != GGML_TYPE_Q8_0) {
        return fail("dst type is none of IQ1_S, IQ2_XXS, Q4_K, Q5_K, Q6_K and Q8_0");
    }
    if (iq && nexp == 0) {
        return fail("an IQ dst type needs an importance matrix");
    }
    if (st != GGML_TYPE_F32 && st != GGML_TYPE_F16 && st != GGML_TYPE_BF16 && st != GGML_TYPE_MXFP4) {
        return fail("src type is none of F32, F16, BF16 and MXFP4");
    }
    if (n == 0 || n % ggml_blck_size(dt) != 0 || per == 0 || nrows % per != 0 || (nexp != 0 && nrows / per > nexp)) {
        return fail("geometry: n_per_row a multiple of the dst block, whole experts, enough importance rows");
    }
    const size_t src_bytes = ggml_row_size(st, n) * nrows;
    std::vector<uint8_t> raw(src_bytes);
    std::vector<float> qw(n * nexp);
    if (std::fread(raw.data(), 1, raw.size(), f) != raw.size() ||
        std::fread(qw.data(), 4, qw.size(), f) != qw.size()) {
        return fail("input is shorter than its header says");
    }
    std::fclose(f);
    std::vector<float> x(n * nrows);
    if (st == GGML_TYPE_F32) {
        std::memcpy(x.data(), raw.data(), raw.size());
    } else {
        ggml_get_type_traits(st)->to_float(raw.data(), x.data(), (int64_t) (n * nrows));
    }
    ggml_quantize_init(dt);
    const size_t row = ggml_row_size(dt, n);
    std::vector<uint8_t> out(row * nrows);
    for (uint64_t e = 0; e < nrows / per; ++e) {
        ggml_quantize_chunk(dt, x.data() + e * per * n, out.data() + e * per * row, 0, (int64_t) per, (int64_t) n,
                            nexp ? qw.data() + e * n : nullptr);
    }
    FILE * o = std::fopen(argv[2], "wb");
    if (!o || std::fwrite(out.data(), 1, out.size(), o) != out.size() || std::fclose(o) != 0) {
        return fail("cannot write the output");
    }
    return 0;
}
