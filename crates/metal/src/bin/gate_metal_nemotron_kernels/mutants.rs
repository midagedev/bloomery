//! The gate's kernel mutants: `--mutant NAME` compiles the kernels' source with
//! one defect edited in, through `Library::from_source_text`, and runs the
//! clause that must catch it. The shipped kernels never carry a mutant; the
//! edit lives here, as text, and a find text that occurs a number of times
//! other than the one named is an error by name, so a kernel edit cannot
//! silently turn a mutant into a no-op.
//!
//! Every kernel clause has a mutant. A family's table is one constant here and
//! one entry of [`TABLES`].

#[path = "../shared/mutants.rs"]
#[allow(
    dead_code,
    reason = "the gate edits the kernels' library only; the tensor-op program's walk is the other gate's"
)]
mod program;

pub use program::Mutant;

/// Every family's mutant table.
const TABLES: &[&[Mutant]] = &[B32_MUTANTS, super::clauses_mamba2::MUTANTS];

/// Every mutant of every table.
pub fn all() -> impl Iterator<Item = &'static Mutant> {
    TABLES.iter().flat_map(|table| table.iter())
}

/// The mutant of this name.
pub fn find(name: &str) -> Option<&'static Mutant> {
    all().find(|m| m.name == name)
}

/// The kernels' source with the mutant's edits applied.
pub fn source_of(mutant: &Mutant) -> Result<String, String> {
    program::source_of(mutant, program::Program::Main)
}

/// The row loop's tail guard dropped: a SIMD group past the last row stores
/// its rows too.
const TAIL_STORE: (&str, &str, usize) = (
    "    for (short row = 0; row < NR0 && first_row + row < rows; ++row) {\n        float sum_all",
    "    for (short row = 0; row < NR0; ++row) {\n        float sum_all",
    1,
);

/// The segmented mat-vec reads every segment from the first segment's weights.
const SEG_WEIGHTS: (&str, &str, usize) = ("W + (ulong)args.segs[s].wblocks,", "W,", 1);

/// The norm on load drops the gain of the high nibbles' first value.
const LOAD_GAIN: (&str, &str, usize) = ("a2 *= gain[base + i + 16];", "a2 *= 1.f;", 1);

/// A staged chunk takes the other half of its block.
const DEQUANT16_CHUNK: (&str, &str, usize) = (
    "FOR_UNROLL (short j = 0; j < 16; ++j) {\n            reg[j / 4][j % 4] = value(blk, (uint)(16 * il + j));",
    "FOR_UNROLL (short j = 0; j < 16; ++j) {\n            reg[j / 4][j % 4] = value(blk, (uint)(16 * (1 - il) + j));",
    2,
);

/// The fifth-bit word is read as one 32-bit load.
const QH_WORD: (&str, &str, usize) = (
    "return (uint)words[0] | ((uint)words[1] << 16);",
    "return *(device const uint *)words;",
    1,
);

/// The Q5_0 and Q5_1 owners' mutants (`quant_b32.metal`, and the row loop and
/// the segmented entry they instantiate). An edit both types share has a row
/// for each type's clause.
const B32_MUTANTS: &[Mutant] = &[
    Mutant {
        name: "b32_dot_plane",
        clause: "q5_0_matvec",
        why: "the high nibbles' fifth bit is read from the plane one place low",
        edits: &[(
            "(qh >> (i + 0 + il + 16) << 8) & 0x00100)",
            "(qh >> (i + 0 + il + 15) << 8) & 0x00100)",
            1,
        )],
    },
    Mutant {
        name: "b32_dot_offset",
        clause: "q5_0_matvec",
        why: "the Q5_0 dot drops the -16 offset of the codes",
        edits: &[(
            "(sumy * -16.f + acc[0] + acc[1])",
            "(sumy * 0.f + acc[0] + acc[1])",
            1,
        )],
    },
    Mutant {
        name: "b32_dot_scale",
        clause: "q5_0_matvec",
        why: "the nibble trick's divisor of the high nibbles of the even bytes is wrong",
        edits: &[(
            "lane.yl[i + 8] = a2 / 16.f;",
            "lane.yl[i + 8] = a2 / 15.f;",
            1,
        )],
    },
    Mutant {
        name: "b32_dot_sum",
        clause: "q5_0_matvec",
        why: "the high nibbles' activation sum drops a value",
        edits: &[("lane.sumy[1] += a2 + a3;", "lane.sumy[1] += a2;", 1)],
    },
    Mutant {
        name: "b32_dot_min",
        clause: "q5_1_matvec",
        why: "the Q5_1 dot drops the min's term",
        edits: &[(
            "(acc[0] + acc[1]) + sumy * (float)blk.m;",
            "(acc[0] + acc[1]) + sumy * 0.f;",
            1,
        )],
    },
    Mutant {
        name: "b32_load_gain",
        clause: "q5_0_norm",
        why: "the norm on load drops the gain of the high nibbles' first value",
        edits: &[LOAD_GAIN],
    },
    Mutant {
        name: "b32_value_plane",
        clause: "q5_0_embed",
        why: "the values 16 to 31 read their fifth bit from the plane one place low",
        edits: &[(
            "return nibble | (((qh >> j) & 1u) << 4);",
            "return nibble | (((qh >> (j < 16 ? j : j - 1)) & 1u) << 4);",
            1,
        )],
    },
    Mutant {
        name: "b32_value_nibble",
        clause: "q5_0_embed",
        why: "the values 16 to 31 take the low nibble of their byte, not the high",
        edits: &[("(qs[j - 16] >> 4)", "(qs[j - 16] & 0x0Fu)", 1)],
    },
    Mutant {
        name: "b32_value_offset",
        clause: "q5_0_embed",
        why: "the Q5_0 value drops the -16 offset of the code",
        edits: &[(
            "(float)((int)code - 16) * (float)blk.d;",
            "(float)((int)code) * (float)blk.d;",
            1,
        )],
    },
    Mutant {
        name: "b32_value_min",
        clause: "q5_1_embed",
        why: "the Q5_1 value drops the min",
        edits: &[(
            "(float)code * (float)blk.d + (float)blk.m;",
            "(float)code * (float)blk.d;",
            1,
        )],
    },
    Mutant {
        name: "b32_dequant16_chunk",
        clause: "q5_0_dequant16",
        why: "a chunk stages the other half of its block",
        edits: &[DEQUANT16_CHUNK],
    },
    Mutant {
        name: "b32_qh_word",
        clause: "q5_0_odd_blocks",
        why: "the fifth-bit word is read as one 32-bit load, which a block at a 2-byte alignment does not allow",
        edits: &[QH_WORD],
    },
    Mutant {
        name: "b32_load_gain_q5_1",
        clause: "q5_1_norm",
        why: "the norm on load drops the gain of the high nibbles' first value",
        edits: &[LOAD_GAIN],
    },
    Mutant {
        name: "b32_dequant16_chunk_q5_1",
        clause: "q5_1_dequant16",
        why: "a chunk stages the other half of its block",
        edits: &[DEQUANT16_CHUNK],
    },
    Mutant {
        name: "b32_row_tail",
        clause: "q5_0_row_tail",
        why: "a SIMD group past the last row stores its rows",
        edits: &[TAIL_STORE],
    },
    Mutant {
        name: "b32_row_tail_q5_1",
        clause: "q5_1_row_tail",
        why: "a SIMD group past the last row stores its rows",
        edits: &[TAIL_STORE],
    },
    Mutant {
        name: "b32_seg_weights",
        clause: "q5_0_seg",
        why: "every segment reads the first segment's weights",
        edits: &[SEG_WEIGHTS],
    },
    Mutant {
        name: "b32_seg_weights_q5_1",
        clause: "q5_1_seg",
        why: "every segment reads the first segment's weights",
        edits: &[SEG_WEIGHTS],
    },
    Mutant {
        name: "b32_qh_word_q5_1",
        clause: "q5_1_odd_blocks",
        why: "the fifth-bit word is read as one 32-bit load, which a block at a 2-byte alignment does not allow",
        edits: &[QH_WORD],
    },
];
