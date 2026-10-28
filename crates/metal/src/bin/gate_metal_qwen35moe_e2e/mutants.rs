//! The gate's kernel mutants: `--mutant NAME` compiles the kernels' source
//! with one defect edited in, through `Library::from_source_text`, opens the
//! session on those kernels and runs the clause that must catch it (a kept
//! mutant, below, has none and runs over every clause). A find
//! text that occurs a number of times other than the one named is an error
//! by name, so a kernel edit cannot silently turn a mutant into a no-op.
//!
//! `d_count`'s defects are the host's (a barrier too many, an allocation in
//! the step, a second wait a step), as are `b_chain`'s (an embed reading
//! another record slot) and `c_resume`'s host half (a lane parity from a
//! count of calls, not from the position): source edits of the session, not
//! of a kernel, so they have no row here. The ahead embed, the headless last
//! row and the mark's dirty lane are the same kind of host defect, and the
//! rows below stand in for them on the kernel side, where each defect's
//! observable lives: the embed reading one word past its id window (the
//! adjacent record slot for a device-fed token), the head storing no pick
//! (the last row's stale slot), and the delta step writing the lane it read
//! (the token a mark's take drops leaving its update in the lane the point
//! copies). `q_layers`'s defects — a pass that
//! commits a command buffer an op, one buffer for the whole pass, and a layer's
//! buffer that skips the token boundary's wait — are the host's too and have
//! no kernel side. The host edits themselves are one-off edits recorded in the
//! round that proves the clause.

#[path = "../shared/mutants.rs"]
mod program;

pub use program::{Mutant, Program, source_of};

/// Every mutant of both tables, with the program it edits.
pub fn all() -> impl Iterator<Item = (&'static Mutant, Program)> {
    program::all(MUTANTS, TENSOR_MUTANTS)
}

/// The mutant of this name, and the program it edits.
pub fn find(name: &str) -> Option<(&'static Mutant, Program)> {
    all().find(|(m, _)| m.name == name).or_else(|| {
        KEPT.iter()
            .find(|m| m.name == name)
            .map(|m| (m, Program::Main))
    })
}

/// Whether `name` is in [`KEPT`]: a mutant no staying clause is expected to
/// turn red, which `--mutant` runs over every clause.
pub fn kept(name: &str) -> bool {
    KEPT.iter().any(|m| m.name == name)
}

pub const MUTANTS: &[Mutant] = &[
    Mutant {
        name: "embed_no_raise",
        clause: "e_token_id",
        why: "an id past the table embeds NaN and raises no token-id site (both embeddings')",
        edits: &[(
            "raise(faults, args.layer, BLOOMERY_FAULT_SITE_TOKEN_ID);",
            "",
            2,
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
    // ahead_other_slot's catcher was h_prompt. b_chain turns red.
    // PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified
    Mutant {
        name: "ahead_other_slot",
        clause: "b_chain",
        why: "the embed reads one word past its id window: the token ahead takes the other \
              record slot's id, and a host-fed token reads past its slot",
        edits: &[("const uint id = ids[t];", "const uint id = ids[t + 1u];", 2)],
    },
    // ahead_mark_dirty_lane's catcher was h_prompt.
    // PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified
    Mutant {
        name: "ahead_mark_dirty_lane",
        clause: "c_resume",
        why: "the delta step writes the lane it read, so the token a checkpoint discards at \
              a mark leaves its update in the lane the point copies",
        edits: &[(
            "uint word_to = lane_buf[args.lane_to];",
            "uint word_to = lane_buf[args.lane_at];",
            1,
        )],
    },
    Mutant {
        name: "mark_row_late",
        clause: "q_mark",
        why: "a pass copies a delta layer's state lane one row after the row its copy names",
        edits: &[(
            "if (copy < copies && r == snaps->after[copy]) {",
            "if (copy < copies && r == snaps->after[copy] + 1u) {",
            1,
        )],
    },
    Mutant {
        name: "mark_ring_dropped",
        clause: "q_mark",
        why: "a pass leaves a copy's conv ring unwritten",
        edits: &[(
            "    snap[(ulong)k * snaps.ring_len + s * args.ch + c] = v;",
            "    (void)v;",
            1,
        )],
    },
    Mutant {
        name: "prefill_ring_last_row",
        clause: "q_state",
        why: "the ring commit of a pass writes the rows before the last: the last row's \
              conv input is missing from its slot",
        edits: &[(
            "uint r = args.rows - kept + i;",
            "uint r = args.rows - kept + i - 1u;",
            1,
        )],
    },
    // The next three read nothing at position 0: one key makes the softmax 1, the rope
    // angle is 0 and the delta state is zero. `m_logits` sees them at the pin's later steps.
    Mutant {
        name: "attn_score_unscaled",
        clause: "m_logits",
        why: "the attention score is the raw dot: the softmax over two or more keys is \
              sharper than the model's",
        edits: &[("float score = dot * args.scale;", "float score = dot;", 1)],
    },
    Mutant {
        name: "rope_key_row_zero",
        clause: "m_logits",
        why: "a key is roped with position 0's row whatever its position: the query-key \
              angle past position 0 is the query's alone",
        edits: &[(
            "device float const *rope_row = rope + args.pos * BLOOMERY_ATTN_ROT;",
            "device float const *rope_row = rope;",
            1,
        )],
    },
    Mutant {
        name: "delta_decay_one",
        clause: "m_logits",
        why: "the delta state never decays: a step above 0 reads the full state the step \
              before left",
        edits: &[("*decay = exp(sa * softplus(aa + dt));", "*decay = 1.0f;", 1)],
    },
    Mutant {
        name: "conv_tap",
        clause: "m_logits",
        why: "the delta conv's oldest tap reads this token's input",
        edits: &[(
            "tap_x(c, pos0, r, 3u, ch, x_stride, x, ring)",
            "tap_x(c, pos0, r, 0u, ch, x_stride, x, ring)",
            1,
        )],
    },
];

/// Mutants no staying clause is expected to turn red. `--mutant NAME` runs one over every
/// clause, and `--mutants-all` runs each like the others and prints the clauses that turned
/// red and `m_logits`' distance. A kept mutant that turns a clause red is no longer kept: the
/// run fails and says the list is stale, and the mutant moves to `MUTANTS` with that clause.
/// A row's `clause` here is the staying clause nearest its defect.
///
/// The healthy session is green on `m_logits` (worst step 1.92e-3). Observed in the last
/// `--mutants-all`, each red on none of the 19 clauses, `m_logits` unmoved:
/// `merge_first_segment` (`f_deep`) merges a pass's segments; the session holds one segment
/// of keys.
/// `ahead_headless_last_row` (`h_prompt`) stores no pick, and `m_logits` feeds the pin's ids.
/// `prefill_kv_offset` (`q_ref`) shifts the batched pass's append, and `m_logits` steps one
/// token at a time.
/// PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified
pub const KEPT: &[Mutant] = &[
    Mutant {
        name: "merge_first_segment",
        clause: "q_chunk",
        why: "the fold folds the first segment's values alone, so every live key \
              past one segment misses from the output",
        edits: &[(
            "if (!refused) {\n        for (uint seg = 0u; seg < args.segments; ++seg) {",
            "if (!refused) {\n        for (uint seg = 0u; seg < 1u; ++seg) {",
            1,
        )],
    },
    Mutant {
        name: "ahead_headless_last_row",
        clause: "b_chain",
        why: "the head stores no pick: the prompt's last row, the one whose pick is read, \
              reports its record slot's stale word",
        edits: &[("step_id[0] = ~uint(best_key);", "", 1)],
    },
    Mutant {
        name: "prefill_kv_offset",
        clause: "q_state",
        why: "a pass appends and attends its rows one position after their own: the cache \
              holds the prompt shifted by a row",
        edits: &[(
            "a.pos = args.from + row;",
            "a.pos = args.from + row + 1u;",
            1,
        )],
    },
];

/// The tensor-op GEMMs' defects `q_tensor` must catch: the program it forces
/// is the one its pass runs.
pub const TENSOR_MUTANTS: &[Mutant] = &[Mutant {
    name: "tensor_pass_pair_swapped",
    clause: "q_tensor",
    why: "the routed pair's gate and up accumulators swap places in the tensor-op SwiGLU",
    edits: &[("pair_swiglu(cg[i], cu[i])", "pair_swiglu(cu[i], cg[i])", 1)],
}];
