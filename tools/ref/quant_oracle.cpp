// quant_oracle.cpp — llama.cpp mainline's ggml as the oracle of every GGUF weight type.
//
// For each type in kTypes and each row length K, from seeded rows, it writes what ggml does at each step the engine repeats:
//   <case>.blocks     the weight rows in the type's blocks: kQuantRows rows made by ggml_quantize_chunk, then (for a type with
//                     block codes) kRandomRows rows of random code bytes whose scale fields are set finite, so every code,
//                     grid index and sign pattern is decoded
//   <case>.deq.f32    to_float of every weight row
//   <case>.x.f32      kActRows f32 activation rows (the same for every type of one K)
//   <case>.act        those rows quantized to the type's vec_dot_type by the CPU backend's from_float
//   <case>.dot.f32    vec_dot(weight row w, activation row a), one f32 per pair, w-major
// <case> is "<ggml type name>.<K>". cases.tsv holds one row per case and files.txt the names written, both read by
// tools/ref/quant-oracle.sh, which states the set's identity around them (QUANT.tsv; crates/refset/src/quant.rs reads it).
// cases.tsv also holds one `type` row for every type ggml's own table names (an id below GGML_TYPE_COUNT whose traits state a
// name and a block size), from the traits and never by hand: its name, id, block size and size, and either `dumped` (the
// type has cases) or `skip` with the traits ggml leaves NULL. A type with a to_float and a vec_dot that is not in kTypes ends
// the run, so the set never leaves a type out without ggml's reason.
//
// A K is used for a type when its block and its vec_dot_type's block divide it: 4096 for every type, 2688 too for the
// 32-, 64- and 128-element block types. A type needing an imatrix gets a seeded one. Everything is deterministic: a rerun on the
// same build writes the same bytes.
//
// Nothing here falls back: a type the build lacks, a size that disagrees with the header's block struct, a non-finite
// dequantized or dotted value, or a commit that is not the pin stops the run with a line that names it.
//
// Build: bash tools/ref/quant-oracle.sh build   (links the mainline tree's libggml-base and libggml-cpu; no card is opened)
// Run:   quant_oracle --pin <40-hex commit> --out DIR [type_name ...]   (named types alone write a partial set, which the reader
//        refuses: its table states every other dumped type with no case)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include "ggml.h"
#include "ggml-cpu.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

static void fail(const std::string & msg) {
    std::fprintf(stderr, "quant_oracle FATAL: %s\n", msg.c_str());
    std::exit(1);
}

// Write to <path>.tmp.<pid> and rename: a reader of the shared $BLOOMERY_DATA never sees a half-written file.
static void write_file(const std::string & path, const void * data, size_t n) {
    const std::string tmp = path + ".tmp." + std::to_string(::getpid());
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) fail("cannot open for write: " + tmp);
    const bool wrote = n == 0 || std::fwrite(data, 1, n, f) == n;
    if (std::fclose(f) != 0 || !wrote) fail("short write: " + tmp);
    if (std::rename(tmp.c_str(), path.c_str()) != 0) fail("cannot rename " + tmp + " to " + path);
}

// The types of the set: every weight type the engine's rows name, and Q1_0 beside them. Each is in
// ggml_quantize_chunk's switch (a type outside it would reach its `default`, which only asserts).
static const enum ggml_type kTypes[] = {
    GGML_TYPE_F32,    GGML_TYPE_F16,    GGML_TYPE_BF16,   GGML_TYPE_Q1_0,   GGML_TYPE_Q2_0,   GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,   GGML_TYPE_Q5_0,   GGML_TYPE_Q5_1,   GGML_TYPE_Q8_0,   GGML_TYPE_MXFP4,  GGML_TYPE_NVFP4,
    GGML_TYPE_Q2_K,   GGML_TYPE_Q3_K,   GGML_TYPE_Q4_K,   GGML_TYPE_Q5_K,   GGML_TYPE_Q6_K,   GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,  GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,  GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ1_S,  GGML_TYPE_IQ1_M,  GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
};

static const int64_t kKLong  = 4096;  // every type
static const int64_t kKShort = 2688;  // the 32-, 64- and 128-element block types
static const int64_t kQuantRows  = 32;  // rows ggml_quantize_chunk makes
static const int64_t kRandomRows = 16;  // rows of random codes
static const int64_t kActRows    = 4;
static const uint64_t kSeed = 0x51a7c0de5eed1234ull;
// A random row's dequantized values and a dot over it stay below this: every scale field is set in [2^-10, 2^-2).
static const float kBound = 1.0e5f;

struct Lcg {
    uint64_t s;
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t)(s >> 32);
    }
    float unit() { return (next() >> 8) * (1.0f / 16777216.0f); }  // [0, 1)
};

// A finite positive f16 in [2^-10, 2^-2).
static ggml_half finite_half(Lcg & rng) {
    return ggml_fp32_to_fp16(std::ldexp(1.0f + rng.unit(), -10 + (int)(rng.next() % 8)));
}

static void put_half(uint8_t * block, size_t off, Lcg & rng) {
    const ggml_half h = finite_half(rng);
    std::memcpy(block + off, &h, sizeof h);
}

// Sets the scale fields of one random block finite. The offsets are the header's block structs', so a build whose layout
// differs fails to compile; a field this misses leaves a random scale, which the finite check on to_float names.
static void finite_scales(enum ggml_type type, uint8_t * b, Lcg & rng) {
    constexpr size_t H = sizeof(ggml_half);
    switch (type) {
        case GGML_TYPE_Q1_0:    put_half(b, offsetof(block_q1_0, d), rng); break;
        case GGML_TYPE_Q2_0:    put_half(b, offsetof(block_q2_0, d), rng); break;
        case GGML_TYPE_Q4_0:    put_half(b, offsetof(block_q4_0, d), rng); break;
        case GGML_TYPE_Q4_1:    put_half(b, offsetof(block_q4_1, data), rng); put_half(b, offsetof(block_q4_1, data) + H, rng); break;
        case GGML_TYPE_Q5_0:    put_half(b, offsetof(block_q5_0, d), rng); break;
        case GGML_TYPE_Q5_1:    put_half(b, offsetof(block_q5_1, data), rng); put_half(b, offsetof(block_q5_1, data) + H, rng); break;
        case GGML_TYPE_Q8_0:    put_half(b, offsetof(block_q8_0, d), rng); break;
        case GGML_TYPE_MXFP4:   b[offsetof(block_mxfp4, e)] = (uint8_t)(117 + rng.next() % 8); break;  // E8M0 2^(e-127)
        case GGML_TYPE_NVFP4:   // UE4M3: exponent 1..6, any mantissa (0x7F is NaN)
            for (size_t i = 0; i < sizeof(((block_nvfp4 *)nullptr)->d); ++i) {
                b[offsetof(block_nvfp4, d) + i] = (uint8_t)((1 + rng.next() % 6) << 3 | rng.next() % 8);
            }
            break;
        case GGML_TYPE_Q2_K:    put_half(b, offsetof(block_q2_K, data), rng); put_half(b, offsetof(block_q2_K, data) + H, rng); break;
        case GGML_TYPE_Q3_K:    put_half(b, offsetof(block_q3_K, d), rng); break;
        case GGML_TYPE_Q4_K:    put_half(b, offsetof(block_q4_K, data), rng); put_half(b, offsetof(block_q4_K, data) + H, rng); break;
        case GGML_TYPE_Q5_K:    put_half(b, offsetof(block_q5_K, data), rng); put_half(b, offsetof(block_q5_K, data) + H, rng); break;
        case GGML_TYPE_Q6_K:    put_half(b, offsetof(block_q6_K, d), rng); break;
        case GGML_TYPE_TQ1_0:   put_half(b, offsetof(block_tq1_0, d), rng); break;
        case GGML_TYPE_TQ2_0:   put_half(b, offsetof(block_tq2_0, d), rng); break;
        case GGML_TYPE_IQ2_XXS: put_half(b, offsetof(block_iq2_xxs, d), rng); break;
        case GGML_TYPE_IQ2_XS:  put_half(b, offsetof(block_iq2_xs, d), rng); break;
        case GGML_TYPE_IQ2_S:   put_half(b, offsetof(block_iq2_s, d), rng); break;
        case GGML_TYPE_IQ3_XXS: put_half(b, offsetof(block_iq3_xxs, d), rng); break;
        case GGML_TYPE_IQ3_S:   put_half(b, offsetof(block_iq3_s, d), rng); break;
        case GGML_TYPE_IQ1_S:   put_half(b, offsetof(block_iq1_s, d), rng); break;
        case GGML_TYPE_IQ4_NL:  put_half(b, offsetof(block_iq4_nl, d), rng); break;
        case GGML_TYPE_IQ4_XS:  put_half(b, offsetof(block_iq4_xs, d), rng); break;
        case GGML_TYPE_IQ1_M: {
            // The f16 scale is spread over the top nibbles of the four u16 of scales[]: 16 bits, 4 per u16.
            const ggml_half h = finite_half(rng);
            for (int i = 0; i < 4; ++i) {
                uint8_t & hi = b[offsetof(block_iq1_m, scales) + 2 * i + 1];
                hi = (uint8_t)((hi & 0x0f) | (((h >> (4 * i)) & 0xf) << 4));
            }
            break;
        }
        default: fail(std::string("no finite-scale layout for ") + ggml_type_name(type));
    }
}

// Whether a random row means anything for the type: the float types have no scale and every bit pattern of them can be a NaN.
static bool has_codes(enum ggml_type type) {
    return type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16;
}

// Roughly Gaussian values (sum of four uniforms) with a per-row scale over two decades and one outlier per 256.
static std::vector<float> weight_rows(int64_t rows, int64_t k, Lcg & rng) {
    std::vector<float> x((size_t)(rows * k));
    for (int64_t r = 0; r < rows; ++r) {
        const float scale = 0.01f * std::pow(100.0f, rng.unit());
        for (int64_t j = 0; j < k; ++j) {
            float g = rng.unit() + rng.unit() + rng.unit() + rng.unit() - 2.0f;
            if (j % 256 == 17) g *= 6.0f;
            x[(size_t)(r * k + j)] = scale * g;
        }
    }
    return x;
}

static std::vector<float> act_rows(int64_t k) {
    Lcg rng{kSeed ^ 0xac7ull ^ (uint64_t)k};
    std::vector<float> x((size_t)(kActRows * k));
    for (int64_t r = 0; r < kActRows; ++r) {
        const float scale = std::pow(4.0f, rng.unit() * 2.0f - 1.0f);
        for (int64_t j = 0; j < k; ++j) {
            float g = rng.unit() + rng.unit() + rng.unit() + rng.unit() - 2.0f;
            if (j % 128 == 5) g *= 4.0f;
            x[(size_t)(r * k + j)] = scale * g;
        }
    }
    return x;
}

static const struct ggml_type_traits * traits_of(enum ggml_type t) {
    const struct ggml_type_traits * tr = ggml_get_type_traits(t);
    if (!tr || !tr->type_name) fail("this build has no traits for ggml type " + std::to_string((int)t));
    return tr;
}

static const struct ggml_type_traits_cpu * cpu_traits_of(enum ggml_type t) {
    const struct ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(t);
    if (!tr) fail(std::string("this build has no CPU traits for ") + ggml_type_name(t));
    return tr;
}

// to_float of `rows` rows, every value finite and below kBound.
static std::vector<float> dequantize(enum ggml_type type, const std::vector<uint8_t> & blocks, int64_t rows, int64_t k,
                                     const std::string & what) {
    const struct ggml_type_traits * tr = traits_of(type);
    const size_t row_bytes = ggml_row_size(type, k);
    std::vector<float> out((size_t)(rows * k));
    if (type == GGML_TYPE_F32) {
        // ggml has no to_float for f32: its blocks are the values.
        std::memcpy(out.data(), blocks.data(), blocks.size());
    } else {
        if (!tr->to_float) fail(what + ": no to_float");
        for (int64_t r = 0; r < rows; ++r) {
            tr->to_float(blocks.data() + (size_t)r * row_bytes, out.data() + (size_t)(r * k), k);
        }
    }
    for (size_t i = 0; i < out.size(); ++i) {
        if (!(std::fabs(out[i]) < kBound)) {
            fail(what + ": to_float gives " + std::to_string(out[i]) + " at value " + std::to_string(i) + " (row " +
                 std::to_string(i / (size_t)k) + ")");
        }
    }
    return out;
}

struct Case {
    std::string name;
    size_t type_size;
    int64_t quant_rows, random_rows;
    bool imatrix;
};

static std::string files_written;
static std::string cases_tsv =
    "# case\tname\ttype\tk\tblck\ttype_size\trow_bytes\tvdt\tvdt_name\tact_row_bytes\tquant_rows\trandom_rows\tact_rows\timatrix\n";

static void put(const std::string & dir, const std::string & file, const void * data, size_t n) {
    write_file(dir + "/" + file, data, n);
    files_written += file + "\n";
}

static void run_case(const std::string & dir, enum ggml_type type, int64_t k) {
    const struct ggml_type_traits * tr = traits_of(type);
    const struct ggml_type_traits_cpu * cpu = cpu_traits_of(type);
    const char * tname = ggml_type_name(type);
    const enum ggml_type vdt = cpu->vec_dot_type;
    if (!cpu->vec_dot) fail(std::string(tname) + ": no vec_dot");
    const int64_t blck = ggml_blck_size(type);
    if (tr->blck_size != blck) fail(std::string(tname) + ": the traits' block size is not ggml_blck_size");
    if (k % blck != 0 || k % ggml_blck_size(vdt) != 0) fail(std::string(tname) + ": K does not divide into blocks");
    const size_t row_bytes = ggml_row_size(type, k);
    const size_t act_row_bytes = ggml_row_size(vdt, k);
    if (row_bytes != (size_t)(k / blck) * ggml_type_size(type)) fail(std::string(tname) + ": ggml_row_size disagrees with the block size");

    Case c;
    c.name = std::string(tname) + "." + std::to_string(k);
    c.type_size = ggml_type_size(type);
    c.quant_rows = kQuantRows;
    c.random_rows = has_codes(type) ? kRandomRows : 0;
    c.imatrix = ggml_quantize_requires_imatrix(type);
    const int64_t rows = c.quant_rows + c.random_rows;

    Lcg rng{kSeed ^ ((uint64_t)type * 0x9e3779b97f4a7c15ull) ^ (uint64_t)k};
    std::vector<uint8_t> blocks((size_t)rows * row_bytes);

    // Quantized rows. An imatrix is passed only to a type that cannot be quantized without one.
    const std::vector<float> src = weight_rows(c.quant_rows, k, rng);
    std::vector<float> imatrix((size_t)k);
    for (float & w : imatrix) w = 0.25f + rng.unit();
    const size_t wrote = ggml_quantize_chunk(type, src.data(), blocks.data(), 0, c.quant_rows, k, c.imatrix ? imatrix.data() : nullptr);
    if (wrote != (size_t)c.quant_rows * row_bytes) fail(c.name + ": ggml_quantize_chunk wrote a different size");

    // Random-code rows: every byte random, then the scale fields of every block finite.
    for (size_t i = (size_t)c.quant_rows * row_bytes; i < blocks.size(); ++i) blocks[i] = (uint8_t)rng.next();
    for (size_t b = (size_t)c.quant_rows * row_bytes; b < blocks.size(); b += c.type_size) finite_scales(type, &blocks[b], rng);

    const std::vector<float> deq = dequantize(type, blocks, rows, k, c.name);

    // The activation rows in the type's vec_dot_type, by the CPU backend's own conversion (what a mul_mat does).
    const std::vector<float> x = act_rows(k);
    std::vector<uint8_t> act((size_t)kActRows * act_row_bytes);
    const struct ggml_type_traits_cpu * vcpu = cpu_traits_of(vdt);
    for (int64_t a = 0; a < kActRows; ++a) {
        if (vcpu->from_float) {
            vcpu->from_float(x.data() + (size_t)(a * k), act.data() + (size_t)a * act_row_bytes, k);
        } else if (vdt == GGML_TYPE_F32) {
            std::memcpy(act.data() + (size_t)a * act_row_bytes, x.data() + (size_t)(a * k), (size_t)k * sizeof(float));
        } else {
            fail(c.name + ": no from_float for the vec_dot type " + ggml_type_name(vdt));
        }
    }

    // vec_dot over every (weight row, activation row) pair.
    std::vector<float> dot((size_t)(rows * kActRows));
    for (int64_t w = 0; w < rows; ++w) {
        for (int64_t a = 0; a < kActRows; ++a) {
            float s = 0.0f;
            cpu->vec_dot((int)k, &s, 0, blocks.data() + (size_t)w * row_bytes, 0, act.data() + (size_t)a * act_row_bytes, 0, 1);
            if (!std::isfinite(s)) fail(c.name + ": vec_dot gives " + std::to_string(s) + " at row " + std::to_string(w) + " x act " + std::to_string(a));
            dot[(size_t)(w * kActRows + a)] = s;
        }
    }

    put(dir, c.name + ".blocks", blocks.data(), blocks.size());
    put(dir, c.name + ".deq.f32", deq.data(), deq.size() * sizeof(float));
    put(dir, c.name + ".x.f32", x.data(), x.size() * sizeof(float));
    put(dir, c.name + ".act", act.data(), act.size());
    put(dir, c.name + ".dot.f32", dot.data(), dot.size() * sizeof(float));
    char line[512];
    const int n = std::snprintf(line, sizeof line, "case\t%s\t%d\t%lld\t%lld\t%zu\t%zu\t%d\t%s\t%zu\t%lld\t%lld\t%lld\t%d\n", c.name.c_str(),
                                (int)type, (long long)k, (long long)blck, c.type_size, row_bytes, (int)vdt, ggml_type_name(vdt),
                                act_row_bytes, (long long)c.quant_rows, (long long)c.random_rows, (long long)kActRows, c.imatrix ? 1 : 0);
    if (n < 0 || (size_t)n >= sizeof line) fail(c.name + ": the case row does not fit its buffer");
    cases_tsv += line;
    std::printf("%-8s K=%-5lld block=%-3lld bytes=%-4zu row=%zu vdt=%s imatrix=%d rows=%lld+%lld\n", tname, (long long)k, (long long)blck,
                c.type_size, row_bytes, ggml_type_name(vdt), c.imatrix ? 1 : 0, (long long)c.quant_rows, (long long)c.random_rows);
}

static bool is_dumped(enum ggml_type type) {
    return std::find(std::begin(kTypes), std::end(kTypes), type) != std::end(kTypes);
}

// Why ggml has nothing to dump for a type kTypes leaves out: the traits it states NULL. A type with both a to_float and a
// vec_dot could be dumped, so leaving it out is the harness's omission and stops the run.
static std::string skip_reason(enum ggml_type type) {
    const struct ggml_type_traits * tr = traits_of(type);
    const struct ggml_type_traits_cpu * cpu = ggml_get_type_traits_cpu(type);
    std::string reason;
    if (!tr->to_float) reason += "to_float NULL";
    if (!cpu || !cpu->vec_dot) reason += std::string(reason.empty() ? "" : "; ") + "vec_dot NULL";
    if (reason.empty()) fail(std::string(ggml_type_name(type)) + " has a to_float and a vec_dot but is not in kTypes: add it to kTypes");
    return reason;
}

// One `type` row for every type ggml's table names, in id order. An id whose traits state no name or no block size is one ggml
// no longer names (its slot holds "DEPRECATED" or a "REMOVED" notice with no block).
static std::string type_table() {
    std::string out = "# type\tname\tid\tblck\tbytes\tstatus\treason\n";
    for (int id = 0; id < GGML_TYPE_COUNT; ++id) {
        const enum ggml_type type = (enum ggml_type)id;
        const struct ggml_type_traits * tr = ggml_get_type_traits(type);
        if (!tr || !tr->type_name || tr->blck_size <= 0) continue;
        const bool dumped = is_dumped(type);
        char line[256];
        const int n = std::snprintf(line, sizeof line, "type\t%s\t%d\t%lld\t%zu\t%s\t%s\n", tr->type_name, id, (long long)tr->blck_size,
                                    tr->type_size, dumped ? "dumped" : "skip", dumped ? "-" : skip_reason(type).c_str());
        if (n < 0 || (size_t)n >= sizeof line) fail(std::string(tr->type_name) + ": the type row does not fit its buffer");
        out += line;
    }
    for (const enum ggml_type type : kTypes) {
        const struct ggml_type_traits * tr = traits_of(type);
        if (tr->blck_size <= 0) fail(std::string(tr->type_name) + ": a type of kTypes has no block size in ggml's table");
    }
    return out;
}

// The commit the build states for itself against the pin: a tree rebuilt without a readable git says "unknown" and is refused.
static void check_commit(const std::string & pin) {
    const std::string have = ggml_commit();
    if (have.size() < 7 || have.size() > pin.size() || pin.compare(0, have.size(), have) != 0) {
        fail("ggml states commit '" + have + "', the pin is " + pin);
    }
}

static std::string cpu_features() {
    std::string s;
    const struct { const char * name; int on; } f[] = {
        {"avx", ggml_cpu_has_avx()},       {"avx_vnni", ggml_cpu_has_avx_vnni()}, {"avx2", ggml_cpu_has_avx2()},
        {"avx512", ggml_cpu_has_avx512()}, {"avx512_vbmi", ggml_cpu_has_avx512_vbmi()}, {"avx512_vnni", ggml_cpu_has_avx512_vnni()},
        {"avx512_bf16", ggml_cpu_has_avx512_bf16()}, {"amx_int8", ggml_cpu_has_amx_int8()}, {"fma", ggml_cpu_has_fma()},
        {"f16c", ggml_cpu_has_f16c()},     {"neon", ggml_cpu_has_neon()},         {"arm_fma", ggml_cpu_has_arm_fma()},
        {"fp16_va", ggml_cpu_has_fp16_va()}, {"dotprod", ggml_cpu_has_dotprod()}, {"matmul_int8", ggml_cpu_has_matmul_int8()},
        {"sve", ggml_cpu_has_sve()},
    };
    for (const auto & e : f) {
        if (e.on) s += (s.empty() ? "" : " ") + std::string(e.name);
    }
    return s.empty() ? "none" : s;
}

int main(int argc, char ** argv) {
    std::string pin, out;
    std::vector<std::string> only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pin" && i + 1 < argc) pin = argv[++i];
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a.rfind("--", 0) == 0) fail("unknown option " + a);
        else only.push_back(a);
    }
    if (pin.size() != 40 || pin.find_first_not_of("0123456789abcdef") != std::string::npos) fail("--pin wants the 40-hex commit of the mainline tree");
    if (out.empty()) fail("--out DIR is required");
    check_commit(pin);

    ggml_cpu_init();

    std::vector<enum ggml_type> types;
    for (const std::string & want : only) {
        const auto it = std::find_if(std::begin(kTypes), std::end(kTypes), [&](enum ggml_type t) { return want == ggml_type_name(t); });
        if (it == std::end(kTypes)) fail("type " + want + " is not one of the oracle's types");
        types.push_back(*it);
    }
    if (types.empty()) types.assign(std::begin(kTypes), std::end(kTypes));

    for (const enum ggml_type type : types) {
        for (const int64_t k : {kKLong, kKShort}) {
            const int64_t vb = ggml_blck_size(cpu_traits_of(type)->vec_dot_type);
            // K = 2688 is the short-block types' only; a block of 256 does not divide it.
            if (k % ggml_blck_size(type) != 0 || k % vb != 0) {
                if (k == kKLong) fail(std::string(ggml_type_name(type)) + ": K=4096 does not divide into its blocks");
                continue;
            }
            run_case(out, type, k);
        }
    }
    ggml_quantize_free();

    const std::string cpu = cpu_features();
    const std::string rows = type_table() + cases_tsv;
    write_file(out + "/cases.tsv", rows.data(), rows.size());
    write_file(out + "/files.txt", files_written.data(), files_written.size());
    const std::string info = "ggml_commit\t" + std::string(ggml_commit()) + "\ncpu\t" + cpu + "\n";
    write_file(out + "/info.tsv", info.data(), info.size());
    std::printf("wrote %zu types to %s (cpu: %s)\n", types.size(), out.c_str(), cpu.c_str());
    return 0;
}
