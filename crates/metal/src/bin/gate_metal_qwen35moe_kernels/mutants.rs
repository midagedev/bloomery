//! The gate's kernel mutants: `--mutant NAME` compiles the kernels' source with
//! one defect edited in, through `Library::from_source_text`, and runs the
//! clause that must catch it. The shipped kernels never carry a mutant; the
//! edit lives here, as text, and a find text that occurs a number of times
//! other than the one named is an error by name, so a kernel edit cannot
//! silently turn a mutant into a no-op.
//!
//! `matvec_refusals` has no mutant here: it checks the host module, whose
//! defect is a source edit of `src/kernels/matvec.rs`, not of a kernel. The
//! refusals of `seg_bounds` are the host's too, so that clause's mutant is
//! the kernel's: a dispatch of the most segments must write every row.

use bloomery_metal::library::SOURCE;

/// A defect and the clause that must be red on it.
pub struct Mutant {
    pub name: &'static str,
    pub clause: &'static str,
    pub why: &'static str,
    /// `(find, replace, occurrences of find)`, applied in order.
    pub edits: &'static [(&'static str, &'static str, usize)],
}

const TAIL_STORE_FIND: &str =
    "    for (short row = 0; row < NR0 && first_row + row < rows; ++row) {\n        float sum_all";
const TAIL_STORE_REPLACE: &str =
    "    for (short row = 0; row < NR0; ++row) {\n        float sum_all";
const ROW_POINTER: &str = "rp[r] = W + (ulong)min(first_row + r, rows - 1) * (ulong)nb;";
const ROW_POINTER_RAW: &str = "rp[r] = W + (ulong)(first_row + r) * (ulong)nb;";
const EARLY_RETURN: &str = "    if (first_row >= rows) {\n        return;\n    }\n";

pub const MUTANTS: &[Mutant] = &[
    Mutant {
        name: "q4k_dmin_sign",
        clause: "q4k_matvec",
        why: "the dmin correction is added, not subtracted",
        edits: &[("1.f/16.f) -\n", "1.f/16.f) +\n", 1)],
    },
    Mutant {
        name: "q6k_offset",
        clause: "q6k_matvec",
        why: "one of the four 6-bit values is offset by 31, not 32",
        edits: &[(
            "((qh[l] & kmask1) << 4)) - 32);",
            "((qh[l] & kmask1) << 4)) - 31);",
            1,
        )],
    },
    Mutant {
        name: "norm_gain",
        clause: "norm_on_load",
        why: "the Q4_K norm drops the gain of a value",
        edits: &[("a0 *= g4[i+  0];", "a0 *= 1.f;", 1)],
    },
    Mutant {
        name: "norm_no_raise",
        clause: "norm_on_load",
        why: "a norm that cannot take its input or its gain writes NaN rows and raises nothing",
        edits: &[(
            "raise(faults, layer, BLOOMERY_FAULT_SITE_NORM_QUANT);",
            "",
            4,
        )],
    },
    Mutant {
        name: "norm_eps",
        clause: "norm_on_load",
        why: "the scale ignores the epsilon",
        edits: &[(
            "sum_squares / float(count) + eps",
            "sum_squares / float(count)",
            1,
        )],
    },
    Mutant {
        name: "resid_overwrite",
        clause: "residual_epilogue",
        why: "the residual epilogue overwrites instead of adding",
        edits: &[(
            "y[first_row + row] += sum_all;",
            "y[first_row + row] = sum_all;",
            2,
        )],
    },
    Mutant {
        name: "tail_store",
        clause: "row_tail",
        why: "the store of a pair no longer stops at the last row",
        edits: &[(TAIL_STORE_FIND, TAIL_STORE_REPLACE, 2)],
    },
    Mutant {
        name: "embed_no_raise",
        clause: "embed",
        why: "an id past the vocabulary raises nothing",
        edits: &[(
            "raise(faults, args.layer, BLOOMERY_FAULT_SITE_TOKEN_ID);",
            "",
            1,
        )],
    },
    Mutant {
        name: "embed_plausible_row",
        clause: "embed",
        why: "an id past the vocabulary writes a zero row",
        edits: &[("value = NAN;", "value = 0.0f;", 1)],
    },
    // A read the output never shows: only Metal's shader validation reports
    // it, so `row_tail` stays green and `row_tail_reads` runs the launches in a
    // child under the layer. It is that clause's control as well.
    Mutant {
        name: "tail_read",
        clause: "row_tail_reads",
        why: "the rows past the last read past the weights window",
        edits: &[(ROW_POINTER, ROW_POINTER_RAW, 2), (EARLY_RETURN, "", 2)],
    },
    Mutant {
        name: "seg_weight_offset",
        clause: "seg_bit",
        why: "a segment reads the weights of the first segment",
        edits: &[("W + (ulong)args.segs[s].wblocks,", "W,", 2)],
    },
    Mutant {
        name: "seg_last_segment",
        clause: "seg_bounds",
        why: "the search for the segment stops before the last one",
        edits: &[(
            "for (uint i = 0; i < args.nseg; ++i) {",
            "for (uint i = 0; i < 3 && i < args.nseg; ++i) {",
            2,
        )],
    },
    Mutant {
        name: "headnorm_divisor",
        clause: "headnorm",
        why: "the head scale divides the sum by twice the head's columns",
        edits: &[(
            "sum / float(BLOOMERY_MATVEC_HEAD_COLS)",
            "sum / float(BLOOMERY_MATVEC_HEAD_COLS * 2)",
            1,
        )],
    },
    Mutant {
        name: "headnorm_no_raise",
        clause: "headnorm_nan",
        why: "a head the norm cannot take raises nothing",
        edits: &[
            (
                "raise(faults, args.layer, BLOOMERY_FAULT_SITE_LINEAR_GATE);",
                "",
                1,
            ),
            (
                "raise(faults, layer, BLOOMERY_FAULT_SITE_LINEAR_GATE);",
                "",
                1,
            ),
        ],
    },
    Mutant {
        name: "nsg_first_row",
        clause: "nsg",
        why: "the first row of a threadgroup ignores the SIMD group count",
        edits: &[("BLOOMERY_MATVEC_SIMD_GROUPS + sgitg", "2 + sgitg", 9)],
    },
    Mutant {
        // K2-REGISTER
        name: "router_logit",                                    // K2-REGISTER
        clause: "router_logit",                                  // K2-REGISTER
        why: "one component of every float4 product is dropped", // K2-REGISTER
        edits: &[("router_acc += w.x * h.x", "router_acc += 0.f", 1)], // K2-REGISTER
    }, // K2-REGISTER
    Mutant {
        // K2-REGISTER
        name: "router_ids",                           // K2-REGISTER
        clause: "router_ids",                         // K2-REGISTER
        why: "the selection keeps the smaller logit", // K2-REGISTER
        edits: &[(
            // K2-REGISTER
            "router_better = v > best_v", // K2-REGISTER
            "router_better = v < best_v", // K2-REGISTER
            1,                            // K2-REGISTER
        )], // K2-REGISTER
    }, // K2-REGISTER
    Mutant {
        // K2-REGISTER
        name: "router_weights",   // K2-REGISTER
        clause: "router_weights", // K2-REGISTER
        why: "a routed weight is the exponential, not its share of the eight", // K2-REGISTER
        edits: &[(
            // K2-REGISTER
            "router_weight = term[s] / sum_exp", // K2-REGISTER
            "router_weight = term[s]",           // K2-REGISTER
            1,                                   // K2-REGISTER
        )], // K2-REGISTER
    }, // K2-REGISTER
    Mutant {
        // K2-REGISTER
        name: "router_ticket",                              // K2-REGISTER
        clause: "router_ticket",                            // K2-REGISTER
        why: "a routed id is one past the selected expert", // K2-REGISTER
        edits: &[(
            // K2-REGISTER
            "ids[s] = router_chosen[s]",      // K2-REGISTER
            "ids[s] = router_chosen[s] + 1u", // K2-REGISTER
            1,                                // K2-REGISTER
        )], // K2-REGISTER
    }, // K2-REGISTER
    Mutant {
        // K2-REGISTER
        name: "router_nan",                       // K2-REGISTER
        clause: "router_nan",                     // K2-REGISTER
        why: "a non-finite logit raises nothing", // K2-REGISTER
        edits: &[(
            // K2-REGISTER
            "raise(faults, args.layer, BLOOMERY_FAULT_SITE_ROUTER);", // K2-REGISTER
            "",                                                       // K2-REGISTER
            1,                                                        // K2-REGISTER
        )], // K2-REGISTER
    }, // K2-REGISTER
    Mutant {
        // K2-REGISTER
        name: "router_bounds",                        // K2-REGISTER
        clause: "router_bounds",                      // K2-REGISTER
        why: "every thread returns before it writes", // K2-REGISTER
        edits: &[("router_bounds_gate == 0u", "router_bounds_gate == 1u", 1)], // K2-REGISTER
    }, // K2-REGISTER
];

/// The mutant of this name.
pub fn find(name: &str) -> Option<&'static Mutant> {
    MUTANTS.iter().find(|m| m.name == name)
}

/// The kernels' source with the mutant's edits applied.
pub fn source_of(mutant: &Mutant) -> Result<String, String> {
    let mut text = SOURCE.to_string();
    for (find, replace, count) in mutant.edits {
        let found = text.matches(find).count();
        if found != *count {
            return Err(format!(
                "mutant {}: {find:?} occurs {found} times, not {count}",
                mutant.name
            ));
        }
        text = text.replace(find, replace);
    }
    Ok(text)
}
