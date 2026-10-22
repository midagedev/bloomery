//! The gate's kernel mutants: `--mutant NAME` compiles the kernels' source
//! with one defect edited in, through `Library::from_source_text`, opens the
//! session on those kernels and runs the clause that must catch it. A find
//! text that occurs a number of times other than the one named is an error
//! by name, so a kernel edit cannot silently turn a mutant into a no-op.
//!
//! `d_count`'s defects are the host's (a barrier too many, an allocation in
//! the step, a second wait a step), as are `b_chain`'s (an embed reading
//! another record slot) and `c_resume`'s host half (a lane parity from a
//! count of calls, not from the position): source edits of the session, not
//! of a kernel, so they have no row here.

use bloomery_metal::library::SOURCE;

/// A defect and the clause that must be red on it.
pub struct Mutant {
    pub name: &'static str,
    pub clause: &'static str,
    pub why: &'static str,
    /// `(find, replace, occurrences of find)`, applied in order.
    pub edits: &'static [(&'static str, &'static str, usize)],
}

pub const MUTANTS: &[Mutant] = &[
    Mutant {
        name: "embed_no_raise",
        clause: "e_token_id",
        why: "an id past the table embeds NaN and raises no token-id site",
        edits: &[(
            "raise(faults, args.layer, BLOOMERY_FAULT_SITE_TOKEN_ID);",
            "",
            1,
        )],
    },
    Mutant {
        name: "delta_lane_in_place",
        clause: "c_resume",
        why: "the delta step writes the lane it read, so a dropped token ahead leaves its \
              update in the lane its replacement reads",
        edits: &[(
            "uint word_to = lane_buf[args.lane_to];",
            "uint word_to = lane_buf[args.lane_at];",
            1,
        )],
    },
    Mutant {
        name: "conv_tap",
        clause: "a_free",
        why: "the delta conv's oldest tap reads this token's input",
        edits: &[(
            "tap_x(c, pos, 3u, ch, x, ring)",
            "tap_x(c, pos, 0u, ch, x, ring)",
            1,
        )],
    },
    Mutant {
        name: "merge_first_segment",
        clause: "f_deep",
        why: "the merge folds the first segment's partial alone, so every live key \
              past one segment misses from the output",
        edits: &[(
            "for (uint seg = 0u; seg < args.segments; ++seg) {",
            "for (uint seg = 0u; seg < 1u; ++seg) {",
            1,
        )],
    },
];

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
