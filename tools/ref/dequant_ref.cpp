// dequant_ref.cpp — ggml dequantization oracle for bloomery stage 1, round 1-1.
//
// Opens a GGUF file, finds every distinct ggml type among its tensors, and
// for each type dumps the dequantization of the first rows of one tensor of
// that type, using ggml's own to_float (ggml_internal_get_type_traits):
//   <out>/<type_name>.raw   — f32 reference rows, row-major
//   <out>/<type_name>.meta  — tensor name, type, dims, rows, rowlen
//   <out>/manifest.txt      — "<type_num> <type_name> <count>" per type
// <out> is $BLOOMERY_DATA/ref unless given. Type names (ggml_type_name
// spellings: f32 bf16 q8_0 ...) keep only those types, and each one named
// must be in the file. The Rust gate (crates/gguf/tests/oracle.rs) re-derives
// the same rows from its own loader and compares them per type.
//
// Build: bash tools/ref/build-dequant.sh   (on the box, IK=/home/user/ik_llama.cpp)
// Run:   $BLOOMERY_DATA/bin/dequant_ref [model.gguf [out_dir [type_name ...]]]
//        $BLOOMERY_DATA/bin/dequant_ref --synthetic [out_dir]
//
// --synthetic covers the types no model file on the box holds (q2_K, iq2_xs,
// iq3_xxs, iq4_xs): per type, kSynthQuant rows of kSynthLen pseudo-random
// values quantized by ggml itself (ggml_quantize_chunk with a synthetic
// importance matrix, as the community files are made), then kSynthRandom rows
// whose code bytes are random and whose f16 scales are set to finite values,
// so every grid index and sign pattern is decoded. <out> defaults to
// $BLOOMERY_DATA/ref-synth and gets, per type, <type>.blocks (the quantized
// bytes), <type>.raw (ggml's to_float of them), <type>.meta, and manifest.txt.

#include <algorithm>
#include <cmath>
#include <iterator>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

#include "ggml.h"
#include "ref_paths.h"

static const char * kGgufPath = ref_model_path();
static const char * kDataDir = ref_data_dir();
static const int64_t kRows = 4; // first N rows of the chosen tensor, per type

static void fail(const std::string & msg) {
    std::fprintf(stderr, "dequant_ref FATAL: %s\n", msg.c_str());
    std::exit(1);
}

// Write to <path>.tmp.<pid> and rename, as mxfp4_ref.cpp and q5k_x4_ref.cpp do: a gate on another
// track reading the shared $BLOOMERY_DATA never sees a half-written dump, and two writers never
// share a temporary file.
static void write_file(const std::string & path, const void * data, size_t n) {
    const std::string tmp = path + ".tmp." + std::to_string(::getpid());
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) fail("cannot open for write: " + tmp);
    const bool wrote = std::fwrite(data, 1, n, f) == n;
    if (std::fclose(f) != 0 || !wrote) fail("short write: " + tmp);
    if (std::rename(tmp.c_str(), path.c_str()) != 0) fail("cannot rename " + tmp + " to " + path);
}

// The synthetic set: fixed seed, fixed shapes, so a rerun writes the same bytes.
static const int64_t kSynthLen = 4096;   // values per row: 16 blocks of 256
static const int64_t kSynthQuant = 64;   // rows quantized by ggml
static const int64_t kSynthRandom = 16;  // rows of random codes
static const uint64_t kSynthSeed = 0x1a2b3c4d5e6f7788ull;

struct Lcg {
    uint64_t s;
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t)(s >> 32);
    }
    float unit() { return (next() >> 8) * (1.0f / 16777216.0f); } // [0, 1)
};

// The byte offsets of the f16 scales inside one block, set finite in the random rows.
static std::vector<size_t> scale_offsets(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q2_K:    return {80, 82};
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:  return {0};
        default: fail("no synthetic layout for this type");
    }
    return {};
}

static int synthetic(const std::string & out_dir) {
    // ggml_init fills the f16 -> f32 table that GGML_FP16_TO_FP32 reads on this build; the
    // model path gets it from gguf_init_from_file's context, this path makes its own.
    struct ggml_init_params ip = {1024, nullptr, true};
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) fail("ggml_init failed");
    const enum ggml_type types[] = {GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS};
    std::filesystem::create_directories(out_dir);
    std::string manifest;
    for (const enum ggml_type type : types) {
        Lcg rng{kSynthSeed ^ (uint64_t)type};
        const int64_t rows = kSynthQuant + kSynthRandom;
        const size_t row_bytes = ggml_row_size(type, kSynthLen);

        // Rows of roughly Gaussian values (sum of four uniforms) with a per-row
        // scale over two decades and one outlier per 256, and an importance
        // matrix of positive weights shared by every row.
        std::vector<float> x((size_t)(kSynthQuant * kSynthLen));
        for (int64_t r = 0; r < kSynthQuant; ++r) {
            const float scale = 0.01f * std::pow(100.0f, rng.unit());
            for (int64_t j = 0; j < kSynthLen; ++j) {
                float g = rng.unit() + rng.unit() + rng.unit() + rng.unit() - 2.0f;
                if (j % 256 == 17) g *= 6.0f;
                x[(size_t)(r * kSynthLen + j)] = scale * g;
            }
        }
        std::vector<float> imatrix((size_t)kSynthLen);
        for (float & w : imatrix) w = 0.25f + rng.unit();

        std::vector<uint8_t> blocks((size_t)rows * row_bytes);
        ggml_quantize_init(type);
        const size_t wrote = ggml_quantize_chunk(type, x.data(), blocks.data(), 0, kSynthQuant, kSynthLen,
                                                 imatrix.data(), nullptr);
        if (wrote != (size_t)kSynthQuant * row_bytes) fail("ggml_quantize_chunk wrote a different size");

        // Random-code rows: every byte random, then each f16 scale replaced by a
        // finite positive half in [2^-10, 2^-2).
        const size_t bsz = ggml_type_size(type);
        for (size_t i = (size_t)kSynthQuant * row_bytes; i < blocks.size(); ++i) blocks[i] = (uint8_t)rng.next();
        for (size_t b = (size_t)kSynthQuant * row_bytes; b < blocks.size(); b += bsz) {
            for (size_t off : scale_offsets(type)) {
                const ggml_fp16_t h = ggml_fp32_to_fp16(std::ldexp(1.0f + rng.unit(), -10 + (int)(rng.next() % 8)));
                std::memcpy(&blocks[b + off], &h, sizeof h);
            }
        }

        std::vector<float> out((size_t)(rows * kSynthLen));
        ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
        if (!traits.to_float) fail("no to_float for this type");
        for (int64_t r = 0; r < rows; ++r) {
            traits.to_float(blocks.data() + (size_t)r * row_bytes, out.data() + (size_t)(r * kSynthLen), kSynthLen);
        }

        const char * tname = ggml_type_name(type);
        write_file(out_dir + "/" + tname + ".blocks", blocks.data(), blocks.size());
        write_file(out_dir + "/" + tname + ".raw", out.data(), out.size() * sizeof(float));
        char meta[512];
        const int meta_n = std::snprintf(meta, sizeof meta,
                     "type=%d %s\nrows=%lld\nrowlen=%lld\nquantized_rows=%lld\nrandom_rows=%lld\nseed=%llu\n",
                     (int)type, tname, (long long)rows, (long long)kSynthLen, (long long)kSynthQuant,
                     (long long)kSynthRandom, (unsigned long long)kSynthSeed);
        if (meta_n < 0 || (size_t)meta_n >= sizeof meta) fail("synthetic meta does not fit its buffer");
        write_file(out_dir + "/" + tname + ".meta", meta, (size_t)meta_n);
        manifest += std::to_string((int)type) + " " + tname + " " + std::to_string(rows) + "\n";
        std::printf("%-8s synthetic rows=%lld (%lld quantized + %lld random) rowlen=%lld row_bytes=%zu\n",
                    tname, (long long)rows, (long long)kSynthQuant, (long long)kSynthRandom,
                    (long long)kSynthLen, row_bytes);
    }
    ggml_quantize_free();
    ggml_free(ctx);
    write_file(out_dir + "/manifest.txt", manifest.data(), manifest.size());
    std::printf("wrote %zu synthetic types to %s\n", std::size(types), out_dir.c_str());
    return 0;
}

static std::vector<uint8_t> read_range(const char * path, int64_t off, size_t n) {
    FILE * f = std::fopen(path, "rb");
    if (!f) fail(std::string("cannot open: ") + path);
    if (::fseeko(f, off, SEEK_SET) != 0) fail("seek failed");
    std::vector<uint8_t> out(n);
    if (std::fread(out.data(), 1, n, f) != n) fail("short read");
    std::fclose(f);
    return out;
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--synthetic") {
        return synthetic(argc > 2 ? argv[2] : std::string(kDataDir) + "/ref-synth");
    }
    const char * path = argc > 1 ? argv[1] : kGgufPath;
    const std::string out_dir = argc > 2 ? argv[2] : std::string(kDataDir) + "/ref";
    const std::vector<std::string> only(argv + std::min(argc, 3), argv + argc);

    // ---- open GGUF, tensor metadata only (no_alloc), as q3k_ref.cpp does ----
    struct ggml_context * gctx = nullptr;
    struct gguf_init_params gp = {true, &gctx};
    struct gguf_context * gguf = gguf_init_from_file(path, gp);
    if (!gguf || !gctx) fail("gguf_init_from_file failed");
    const int n_tensors = gguf_get_n_tensors(gguf);
    const size_t data_off = gguf_get_data_offset(gguf);

    // ---- one representative tensor + per-type counts, file order ----
    std::vector<int> first_of_type;      // tidx of first tensor per seen type
    std::vector<std::pair<int,int>> counts; // (type, count), first-seen order
    for (int i = 0; i < n_tensors; ++i) {
        const enum ggml_type t = gguf_get_tensor_type(gguf, i);
        auto it = std::find_if(counts.begin(), counts.end(),
                               [t](const std::pair<int,int> & c) { return c.first == (int)t; });
        if (it == counts.end()) {
            counts.emplace_back((int)t, 1);
            first_of_type.push_back(i);
        } else {
            it->second++;
        }
    }

    // A filter names types the file must hold: an absent one is an error, never
    // a dump that silently lacks it.
    for (const std::string & want : only) {
        auto it = std::find_if(counts.begin(), counts.end(), [&want](const std::pair<int,int> & c) {
            return want == ggml_type_name((enum ggml_type)c.first);
        });
        if (it == counts.end()) fail("type " + want + " is not in " + path);
    }

    std::filesystem::create_directories(out_dir);
    std::string manifest_path = out_dir + "/manifest.txt";
    FILE * mf = std::fopen(manifest_path.c_str(), "wb");
    if (!mf) fail("cannot open for write: " + manifest_path);

    size_t written = 0;
    for (size_t k = 0; k < counts.size(); ++k) {
        const enum ggml_type type = (enum ggml_type)counts[k].first;
        if (!only.empty() && std::find(only.begin(), only.end(), ggml_type_name(type)) == only.end()) {
            continue;
        }
        const int tidx = first_of_type[k];
        const char * name = gguf_get_tensor_name(gguf, tidx); // borrowed, not freed
        struct ggml_tensor * info = ggml_get_tensor(gctx, name);
        if (!info) fail(std::string("ggml_get_tensor failed for ") + name);
        if (info->type != type) fail("type disagreement on " + std::string(name));

        const int64_t ne0 = info->ne[0];
        const int64_t rows_total = info->ne[1] * info->ne[2] * info->ne[3];
        const int64_t rows = std::min(kRows, rows_total);
        const size_t row_bytes = ggml_row_size(type, ne0);
        const size_t n_values = (size_t)rows * (size_t)ne0;

        // raw quantized bytes of the first `rows` rows, via the GGUF offsets
        const size_t tensor_off = gguf_get_tensor_offset(gguf, tidx);
        std::vector<uint8_t> raw =
            read_range(path, (int64_t)(data_off + tensor_off), rows * row_bytes);

        // dequantize with ggml itself. F32 has no to_float in the type
        // traits of this fork (ggml.c:657 leaves it NULL) — f32 rows are
        // their own reference, so copy the bytes.
        std::vector<float> out(n_values);
        ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
        for (int64_t r = 0; r < rows; ++r) {
            const uint8_t * row = raw.data() + (size_t)r * row_bytes;
            if (type == GGML_TYPE_F32) {
                std::memcpy(out.data() + (size_t)r * ne0, row, (size_t)ne0 * sizeof(float));
            } else {
                if (!traits.to_float) fail("no to_float for this type");
                traits.to_float(row, out.data() + (size_t)r * ne0, ne0);
            }
        }

        const char * tname = ggml_type_name(type);
        write_file(out_dir + "/" + tname + ".raw",
                   out.data(), out.size() * sizeof(float));
        char meta[1024];
        const int meta_n = std::snprintf(meta, sizeof meta,
                     "tensor=%s\ntype=%d %s\ndims=%d %lld %lld %lld %lld\nrows=%lld\nrowlen=%lld\n",
                     name, (int)type, tname, ggml_n_dims(info),
                     (long long)info->ne[0], (long long)info->ne[1],
                     (long long)info->ne[2], (long long)info->ne[3],
                     (long long)rows, (long long)ne0);
        if (meta_n < 0 || (size_t)meta_n >= sizeof meta) fail("meta does not fit its buffer: " + std::string(name));
        write_file(out_dir + "/" + tname + ".meta", meta, (size_t)meta_n);
        std::fprintf(mf, "%d %s %d\n", (int)type, tname, counts[k].second);
        std::printf("%-6s tensor=%s rows=%lld rowlen=%lld row_bytes=%zu\n",
                    tname, name, (long long)rows, (long long)ne0, row_bytes);
        ++written;
    }
    std::fclose(mf);
    gguf_free(gguf);
    ggml_free(gctx);
    std::printf("wrote %zu types to %s\n", written, out_dir.c_str());
    return 0;
}
