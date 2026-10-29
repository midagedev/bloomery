//! The probe's variant arms: the shipped GEMM source with one change edited
//! in, compiled into a library of its own and timed beside the shipped
//! kernel. A variant is how a phase of the kernel is priced (the staging
//! removed, the multiply removed, the activation check removed) and how a
//! change is tried before it goes into `gemm.metal`. Each edit's find text
//! must occur exactly once in the source; a text that moved is a named
//! error, never a silent no-op.

/// One variant: a name, what it does, the edits, whether the activation
/// tiles run fastest across the dispatch (the shipped kernels take the
/// weight tiles across x), and
/// the tile the edited source owns (the weight rows and the activation rows
/// of a threadgroup), which sizes the grid.
pub struct Variant {
    pub name: &'static str,
    pub what: &'static str,
    pub edits: &'static [(&'static str, &'static str)],
    pub acts_fast: bool,
    pub rows: usize,
    pub acts: usize,
}

/// The variants the probe knows.
pub const VARIANTS: &[Variant] = &[
    Variant {
        name: "nostage",
        what: "no staging: the K loop's two barriers and the multiply, over a tile nobody refills",
        edits: &[
            (
                "FOR_UNROLL(short a = 0; a < A_CHUNKS; ++a) {\n            const int pos",
                "FOR_UNROLL(short a = 0; a < 0; ++a) {\n            const int pos",
            ),
            (
                "FOR_UNROLL(short b = 0; b < B_CHUNKS; ++b) {\n            device const X * xr",
                "FOR_UNROLL(short b = 0; b < 0; ++b) {\n            device const X * xr",
            ),
            (
                "const uint nb = (uint)cols / (uint)Q::values_per_block();",
                "const uint nb = (uint)cols / (uint)Q::values_per_block();\n    sa[tiitg % 64] = T(1);",
            ),
        ],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "nomma",
        what: "no multiply: the staging and the barriers, each thread reading its own staged run back",
        edits: &[(
            "for (short kk = 0; kk < BK; kk += 8) {\n            simdgroup_barrier(mem_flags::mem_none);\n            vec<float, 2> af[GEMM_FRAGS_A];",
            "FOR_UNROLL(short q = 0; q < 16; ++q) {\n            c[0][0].x += float(sa[(tiitg / 2) * LDA + (tiitg % 2) * 16 + q]);\n            c[0][0].y += float(sb[(tiitg / 2) * LDA + (tiitg % 2) * 16 + q]);\n        }\n        for (short kk = 0; kk < 0; kk += 8) {\n            simdgroup_barrier(mem_flags::mem_none);\n            vec<float, 2> af[GEMM_FRAGS_A];",
        )],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "nocheck",
        what: "the f16 arm without its range check on the activations",
        edits: &[(
            "if (sizeof(T) == 2) {\n            dst[g] = vec<T, 4>(gemm_half4(",
            "if (false) {\n            dst[g] = vec<T, 4>(gemm_half4(",
        )],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "xvec",
        what: "the activations load as aligned float4s (the window and stride must be 16-byte aligned)",
        edits: &[(
            "const float4 v = float4(xr[4 * g], xr[4 * g + 1], xr[4 * g + 2], xr[4 * g + 3]);",
            "const float4 v = *(device const float4 *)(xr + 4 * g);",
        )],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "ff",
        what: "float fragments and the float multiply in the f16 arm (MLX's form)",
        edits: &[
            (
                "if (sizeof(T) == 2) {\n                FOR_UNROLL(short i = 0; i < GEMM_FRAGS_A; ++i) {\n                    ah[i]",
                "if (false) {\n                FOR_UNROLL(short i = 0; i < GEMM_FRAGS_A; ++i) {\n                    ah[i]",
            ),
            (
                "af[i] = *(threadgroup const vec<float, 2> *)(arow[i] + kk);",
                "af[i] = vec<float, 2>(float(arow[i][kk]), float(arow[i][kk + 1]));",
            ),
            (
                "GEMM_FRAGS_W - 1 - n : n;\n                    if (sizeof(T) == 2) {\n                        frag_mma_half(",
                "GEMM_FRAGS_W - 1 - n : n;\n                    if (false) {\n                        frag_mma_half(",
            ),
        ],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "bvec",
        what: "(timing only, wrong values) the weight fragments load as one 32-bit word instead of two 16-bit ones",
        edits: &[(
            "bh[j] = vec<half, 2>(brow[j][kk], (brow[j] + LDA)[kk]);",
            "bh[j] = *(threadgroup const vec<half, 2> *)((threadgroup const half *)((ulong)(brow[j] + kk) & ~(ulong)3));",
        )],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "noa",
        what: "no weight staging (the dequant and its loads gone); the activations still stage",
        edits: &[
            (
                "FOR_UNROLL(short a = 0; a < A_CHUNKS; ++a) {\n            const int pos",
                "FOR_UNROLL(short a = 0; a < 0; ++a) {\n            const int pos",
            ),
            (
                "const uint nb = (uint)cols / (uint)Q::values_per_block();",
                "const uint nb = (uint)cols / (uint)Q::values_per_block();\n    sa[tiitg % 64] = T(1);",
            ),
        ],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "nob",
        what: "no activation staging (the loads and the cast gone); the weights still stage",
        edits: &[
            (
                "FOR_UNROLL(short b = 0; b < B_CHUNKS; ++b) {\n            device const X * xr",
                "FOR_UNROLL(short b = 0; b < 0; ++b) {\n            device const X * xr",
            ),
            (
                "const uint nb = (uint)cols / (uint)Q::values_per_block();",
                "const uint nb = (uint)cols / (uint)Q::values_per_block();\n    sb[tiitg % 64] = T(1);",
            ),
        ],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "q4stub",
        what: "Q4_K's dequant16 without its code-word loads and masks (the scale unpack stays)",
        edits: &[
            ("dl * (float)pair.x - ml;", "dl - ml;"),
            ("dl * (float)pair.y - ml;", "dl - ml;"),
        ],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "q6stub",
        what: "Q6_K's dequant16 without its code-word loads and bit assembly",
        edits: &[
            ("fma((float)codes.x, dsc, offset);", "dsc;"),
            ("fma((float)codes.y, dsc, offset);", "offset;"),
        ],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "a4stub",
        what: "A4's dequant16 without its code-word reads",
        edits: &[("const uint word = (uint)w[h];", "const uint word = 0u;")],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "acts128",
        what: "a threadgroup owns 128 activation rows: the weights dequantize once for twice the products",
        edits: &[(
            "#define GEMM_ACTS 64     // activation rows a threadgroup owns",
            "#define GEMM_ACTS 128",
        )],
        acts_fast: false,
        rows: 64,
        acts: 128,
    },
    Variant {
        name: "rows128",
        what: "a threadgroup owns 128 weight rows: the activations stage once for twice the products",
        edits: &[(
            "#define GEMM_ROWS 64     // weight rows a threadgroup owns",
            "#define GEMM_ROWS 128",
        )],
        acts_fast: false,
        rows: 128,
        acts: 64,
    },
    Variant {
        name: "flip",
        what: "the staging order the other way round: the weights first in the f16 arm, the activations first in the f32 arm",
        edits: &[(
            "if (sizeof(T) == 2) {\n            stage_b(k0);",
            "if (sizeof(T) != 2) {\n            stage_b(k0);",
        )],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "nofence",
        what: "the multiply phase without its two simdgroup fences",
        edits: &[
            (
                "simdgroup_barrier(mem_flags::mem_none);\n            vec<float, 2> af[GEMM_FRAGS_A];",
                "vec<float, 2> af[GEMM_FRAGS_A];",
            ),
            (
                "simdgroup_barrier(mem_flags::mem_none);\n            FOR_UNROLL(short m = 0; m < GEMM_FRAGS_A; ++m) {",
                "FOR_UNROLL(short m = 0; m < GEMM_FRAGS_A; ++m) {",
            ),
        ],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "afast",
        what: "the activation tiles run fastest across the dispatch (the order before the weight tiles took x)",
        edits: &[(
            "(int)tgpig.x * GEMM_ROWS, (int)tgpig.y * GEMM_ACTS,",
            "(int)tgpig.y * GEMM_ROWS, (int)tgpig.x * GEMM_ACTS,",
        )],
        acts_fast: true,
        rows: 64,
        acts: 64,
    },
];

/// The routed mode's variants: edits of the grouped GEMM (`gemm_routed.metal`)
/// and its table (`route.metal`). `acts` is the table's tile height the
/// edited source cuts; `rows` the tile's weight rows.
pub const ROUTED: &[Variant] = &[
    Variant {
        name: "rnomma",
        what: "no multiply: the staging and the barriers, each thread reading its own staged run back",
        edits: &[(
            "        for (short kk = 0; kk < BK; kk += 8) {\n            simdgroup_barrier(mem_flags::mem_none);\n            vec<float, 2> af[NA];",
            "        FOR_UNROLL(short q = 0; q < 16; ++q) {\n            c[0][0].x += float(sa[(tiitg / 2) * LDA + (tiitg % 2) * 16 + q]);\n            c[0][0].y += float(sb[((tiitg / 2) % ROUTED_ACTS) * LDA + (tiitg % 2) * 16 + q]);\n        }\n        for (short kk = 0; kk < 0; kk += 8) {\n            simdgroup_barrier(mem_flags::mem_none);\n            vec<float, 2> af[NA];",
        )],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "rnostage",
        what: "no staging: the K loop's two barriers and the multiply, over a tile nobody refills",
        edits: &[
            (
                "        if (b_stage) {\n            device const float * xr = b_src + k0;",
                "        if (false) {\n            device const float * xr = b_src + k0;",
            ),
            (
                "        FOR_UNROLL(short i = 0; i < 4; ++i) {\n            ad[i]",
                "        FOR_UNROLL(short i = 0; i < 0; ++i) {\n            ad[i]",
            ),
        ],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "rnoa",
        what: "no weight staging (the dequant and its loads gone); the activations still stage",
        edits: &[(
            "        FOR_UNROLL(short i = 0; i < 4; ++i) {\n            ad[i]",
            "        FOR_UNROLL(short i = 0; i < 0; ++i) {\n            ad[i]",
        )],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "rnob",
        what: "no activation staging (the loads and the cast gone); the weights still stage",
        edits: &[(
            "        if (b_stage) {\n            device const float * xr = b_src + k0;",
            "        if (false) {\n            device const float * xr = b_src + k0;",
        )],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
    Variant {
        name: "rdb",
        what: "two staged tiles: step k+1 stages while step k multiplies, one barrier a K step (f16 arm only: the f32 arm's two tiles pass 32 KB)",
        edits: &[
            (
                "    return (ROUTED_ROWS + ROUTED_ACTS) * gemm_lda<T>() * (int)sizeof(T);",
                "    return 2 * (ROUTED_ROWS + ROUTED_ACTS) * gemm_lda<T>() * (int)sizeof(T);",
            ),
            (
                "    for (int k0 = 0; k0 < (int)args.cols; k0 += BK) {\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        stage(k0, 0);\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        multiply(0);\n    }",
                "    constexpr int TILE = (ROUTED_ROWS + ROUTED_ACTS) * LDA;\n    stage(0, 0);\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    int at = 0;\n    for (int k0 = 0; k0 < (int)args.cols; k0 += BK) {\n        if (k0 + BK < (int)args.cols) {\n            stage(k0 + BK, TILE - at);\n        }\n        multiply(at);\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n        at = TILE - at;\n    }",
            ),
        ],
        acts_fast: false,
        rows: 64,
        acts: 64,
    },
];

/// The routed variant named `name`.
pub fn find_routed(name: &str) -> Option<&'static Variant> {
    ROUTED.iter().find(|v| v.name == name)
}

/// The variant named `name`.
pub fn find(name: &str) -> Option<&'static Variant> {
    VARIANTS.iter().find(|v| v.name == name)
}

/// `source` with the variant's edits made.
pub fn apply(variant: &Variant, source: &str) -> Result<String, String> {
    let mut text = source.to_string();
    for (find, replace) in variant.edits {
        let hits = text.matches(find).count();
        if hits != 1 {
            return Err(format!(
                "variant {}: the find text {find:?} occurs {hits} times in the source, not once",
                variant.name
            ));
        }
        text = text.replacen(find, replace, 1);
    }
    Ok(text)
}
