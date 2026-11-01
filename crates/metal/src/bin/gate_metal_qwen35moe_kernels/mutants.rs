//! The gate's kernel mutants: `--mutant NAME` compiles the kernels' source with
//! one defect edited in, through `Library::from_source_text`, and runs the
//! clause that must catch it. The shipped kernels never carry a mutant; the
//! edit lives here, as text, and a find text that occurs a number of times
//! other than the one named is an error by name, so a kernel edit cannot
//! silently turn a mutant into a no-op.
//!
//! Every kernel clause has a mutant. The refusal tables check the host launch
//! modules, whose defect is a source edit of `crates/metal/src/kernels/*.rs`:
//! the module is compiled into this gate's own binary, so each such defect is
//! a one-off edit recorded in the round that proves it, not a mutant here.
//! `matvec_refusals`, `delta_bounds`, `experts_bounds`, `fold_type` and
//! `handon_refusals` are wholly host-side; `seg_bounds`, `router_bounds`,
//! `head_bounds` and `attn_bounds` also have a kernel-side contract, and the
//! kernel mutants below pin that side. The gate's refusal loop turns red on a deleted launch
//! check; a table that stayed green on one would be wrong.
//!
//! The tensor-op GEMMs' mutants ([`TENSOR_MUTANTS`]) edit their own program's
//! text and are caught by the `tensor_*` clauses, which run those kernels on
//! any device that compiles them.

#[path = "../shared/mutants.rs"]
mod program;

pub use program::{Mutant, Program, source_at};

use super::Table;

/// Every mutant of an instance's run (`table`, its row of the gate), with the
/// program it edits: at an instance that runs the whole gate both tables; at
/// one that runs its shaped kernels' clauses, the mutants of either table
/// that name those clauses (`kept`) and the instance's own.
pub fn all(table: Table) -> Vec<(&'static Mutant, Program)> {
    match table {
        Table::Whole => program::all(MUTANTS, TENSOR_MUTANTS).collect(),
        Table::Shaped { kept, own }
        | Table::Delta { kept, own }
        | Table::Attention { kept, own }
        | Table::Window { kept, own } => {
            if let Some(name) = kept.iter().find(|name| {
                !MUTANTS
                    .iter()
                    .chain(TENSOR_MUTANTS)
                    .any(|m| m.name == **name)
            }) {
                panic!("the row keeps the mutant {name}, which neither table holds");
            }
            program::all(MUTANTS, TENSOR_MUTANTS)
                .filter(|(m, _)| kept.contains(&m.name))
                .chain(program::all(own, &[]))
                .collect()
        }
    }
}

/// The mutants of the clauses the Qwen3-30B-A3B row runs, less
/// those whose text its preprocessor removes or whose edit is the same
/// program there: the gated epilogues (`attn_gate_s1`, `attn_gate_s1_q8`, `attn_gate_merge`),
/// the shared slot (`pair_shared_stack`), the shared group of the grouped GEMM
/// (`routed_shared_base`, `routed_shared_group_refused`), the picks' stride (`route_rows_per_row`), the
/// logits' stride (`route_rows_stride`: the row's logits are dense, the stride
/// its row count) and the key head swap (`prefill_kv_head`: four key heads
/// make `1 - kv` wrap). The A4 mutants are kept but the head-normed scale's
/// (`a4_headnorm_scale`), a delta net's; so are those of the A4 GEMM clauses
/// (dense, segmented and tensor-op), which run at both rows.
pub const Q30_KEPT: &[&str] = &[
    "router_logit",
    "router_ids",
    "router_tie",
    "router_weights",
    "router_ticket",
    "router_nan",
    "router_bounds",
    "router_rounds",
    "router_bad",
    "router_norm_no_raise",
    "pair_no_silu",
    "pair_id_zero",
    "pair_norm_gain",
    "pair_norm_no_raise",
    "fold_tail",
    "fold_drop_weight",
    "fold_no_resid",
    "attn_score_scale",
    "attn_q8_id",
    "attn_key_count",
    "attn_no_fold",
    "attn_f16_round",
    "attn_s1_partials",
    "attn_half",
    "attn_bounds",
    "attn_span_tile",
    "attn_tile_dead_unmasked",
    "attn_prefetch_stale",
    "attn_fold_stage_l",
    "attn_fold_empty_unskipped",
    "attn_fold_underflow_skipped",
    "attn_one_dead_unmasked",
    "attn_one_chunk_rescale",
    "attn_one_merge_unscaled",
    "attn_one_store_owner",
    "attn_one_local_key_swap",
    "rows_rope_relative",
    "rows_causal_plus1",
    "rows_norm_shared",
    "rows_norm_no_raise",
    "rows_norm_plausible",
    "prefill_causal_plus1",
    "prefill_tile_relative",
    "prefill_no_rescale",
    "prefill_score_scale",
    "prefill_overflow_no_raise",
    "prefill_nan_plausible",
    "route_table_duplicate",
    "route_table_tile_cut",
    "routed_expert_base",
    "routed_no_weight",
    "routed_silu_up",
    "routed_no_gather",
    "routed_dead_fragment",
    "routed_range_off",
    "routed_upper_rows",
    "routed_last_row_unstaged",
    "combine_skip_shared",
    "gemm_f32w_rotate",
    "a4_lane_pair",
    "a4_word_offset",
    "a4_bias_sign",
    "a4_scale_bias",
    "a4_tail_store",
    "a4_norm_gain",
    "a4_seg_weights",
    "a4_pair_sum_sq",
    "a4_load_x",
    "a4_load_gain",
    "a4_fold_slot",
    "a4_head_logit",
    "a4_embed_code",
    "a4_routed_bias_sign",
    "gemm_row_edge",
    "gemm_resid_overwrite",
    "gemm_seg_weights",
    "gemm_a4_chunk_order",
    "gemm_a4_nibble_scale",
    "gemm_seg_grid_order",
    "gemm_h16_stage",
    "gemm_cast_stride",
    "gemm_cast_unchecked",
    "gemm_weight_edge",
    "tensor_resid_stores",
    "tensor_seg_out",
    "tensor_routed_block_rows",
    "tensor_tile_no_raise",
];

/// The defects only a head-128, ungated, unshared instance shows, and those of
/// the clauses only its row runs: each edit is the same program at Qwen3.6's
/// geometry, text its preprocessor removes there, or a defect of a clause
/// that row alone has.
pub const Q30_MUTANTS: &[Mutant] = &[
    Mutant {
        name: "attn_factors_tile_sized",
        clause: "attn_f16",
        why: "the shared block holds a head row for every two SIMD groups, so two query heads' prepares pair through one row",
        edits: &[(
            "threadgroup float *row = scratch + sg * BLOOMERY_ATTN_HEAD;",
            "threadgroup float *row = scratch + (sg / 2u) * BLOOMERY_ATTN_HEAD;",
            1,
        )],
    },
    Mutant {
        name: "attn_norm_pair_32",
        clause: "attn_f16",
        why: "the norm's pair tree pairs dim t with t + 32, not with t + rot/2",
        edits: &[(
            "            i1 = tid + BLOOMERY_ATTN_ROT / 2u;",
            "            i1 = tid + 32u;",
            1,
        )],
    },
    Mutant {
        name: "router_shared_write_ungated",
        clause: "router_ids",
        why: "a router with no shared row still writes the shared slot's id and weight after the picks",
        edits: &[(
            "#if BLOOMERY_ROUTER_SHARED\n        ids[top_k] = ROUTER_EXPERTS;",
            "#if 1\n        ids[top_k] = ROUTER_EXPERTS;",
            1,
        )],
    },
    Mutant {
        name: "route_rows_per_row_shared",
        clause: "route_rows_select",
        why: "a row's ids and weights are written at the picks' stride and a shared slot's, which the router has none of",
        edits: &[(
            "const uint per_row = args.top_k + BLOOMERY_ROUTER_SHARED;",
            "const uint per_row = args.top_k + 1u;",
            1,
        )],
    },
    Mutant {
        name: "route_table_shared_slot",
        clause: "route_table",
        why: "the last slot of every row is taken for a shared expert's, which the instance has none of",
        edits: &[(
            "const bool shared = BLOOMERY_ROUTER_SHARED && (s % args.per_row) == args.per_row - 1u;",
            "const bool shared = 1 && (s % args.per_row) == args.per_row - 1u;",
            2,
        )],
    },
    Mutant {
        name: "combine_shared_term",
        clause: "combine",
        why: "the combine sums one slot more than the picks, a shared expert's term the instance has none of",
        edits: &[(
            "for (uint s = 0u; s < args.per_row; ++s) {",
            "for (uint s = 0u; s < BLOOMERY_EXPERT_ROUTED + 1u; ++s) {",
            1,
        )],
    },
    Mutant {
        name: "routed_q6_offset",
        clause: "grouped_q6_down",
        why: "the Q6_K down of the grouped GEMM is centred on 31, not 32, so every weight is off by one scale step",
        edits: &[(
            "const float offset = -32.f * dsc;",
            "const float offset = -31.f * dsc;",
            1,
        )],
    },
    Mutant {
        name: "prefill_kv_head_reversed",
        clause: "attn_prefill",
        why: "the prefill reads the mirrored key head's rows for its query heads",
        edits: &[(
            "ulong at = ((ulong)kv * args.ctx + key) * PF_ROW_WORDS + c;",
            "ulong at = ((ulong)(BLOOMERY_ATTN_KV - 1u - kv) * args.ctx + key) * PF_ROW_WORDS + c;",
            1,
        )],
    },
    Mutant {
        name: "attn_gate_ungated",
        clause: "attn_q8",
        why: "a query with no gate still multiplies the q8_0 cache's one-segment attention by the sigmoid of the value past its head",
        edits: &[
            (
                "#if BLOOMERY_ATTN_GATED\n            float g = q[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim0 + d];",
                "#if 1\n            float g = q[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + dim0 + d];",
                1,
            ),
            (
                "#if ATTN_GATE\n            float gated = attn * attn_sigmoid(g);",
                "#if 1\n            float gated = attn * attn_sigmoid(g);",
                1,
            ),
        ],
    },
    Mutant {
        name: "attn_one_gate_ungated",
        clause: "attn_f16",
        why: "a query with no gate still multiplies the one-pass flash's attention by the sigmoid of the value past its head",
        edits: &[
            (
                "#if BLOOMERY_ATTN_GATED\n    float g = gate_src[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + tid];",
                "#if 1\n    float g = gate_src[head * BLOOMERY_ATTN_QG + BLOOMERY_ATTN_HEAD + tid];",
                1,
            ),
            (
                "#if ATTN_GATE\n    attn_commit(out, o, attn, g, attn * attn_sigmoid(g), faults, args.layer);",
                "#if 1\n    attn_commit(out, o, attn, g, attn * attn_sigmoid(g), faults, args.layer);",
                1,
            ),
        ],
    },
    Mutant {
        name: "routed_group_past_routed_open",
        clause: "grouped_q6_down",
        why: "a tile of the group one past the routed experts passes the expert-id check on an instance with no shared expert and reads one expert past the stack",
        edits: &[(
            "expert >= args.n_experts + BLOOMERY_ROUTER_SHARED",
            "expert > args.n_experts",
            1,
        )],
    },
    Mutant {
        name: "routed_group_past_routed_shared",
        clause: "grouped_q6_down",
        why: "a tile of the group one past the routed experts is taken for the shared expert's on an instance with none: it raises no expert-id site and reads the stack's first expert",
        edits: &[
            (
                "expert >= args.n_experts + BLOOMERY_ROUTER_SHARED",
                "expert > args.n_experts",
                1,
            ),
            (
                "const bool shared = BLOOMERY_ROUTER_SHARED != 0 && expert == args.n_experts;",
                "const bool shared = expert == args.n_experts;",
                1,
            ),
        ],
    },
];

/// The mutants of the clauses an attention row runs (the decode flash, the
/// rows flash and the prefill flash), every one a defect that row shows at a
/// group size of six or four as at eight.
pub const ATTN_KEPT: &[&str] = &[
    "attn_score_scale",
    "attn_q8_id",
    "attn_key_count",
    "attn_no_fold",
    "attn_f16_round",
    "attn_s1_partials",
    "attn_half",
    "attn_bounds",
    "attn_span_tile",
    "attn_gate_s1",
    "attn_gate_s1_q8",
    "attn_gate_merge",
    "attn_tile_dead_unmasked",
    "attn_prefetch_stale",
    "attn_fold_stage_l",
    "attn_fold_empty_unskipped",
    "attn_fold_underflow_skipped",
    "attn_one_dead_unmasked",
    "attn_one_chunk_rescale",
    "attn_one_merge_unscaled",
    "attn_one_store_owner",
    "attn_one_local_key_swap",
    "rows_rope_relative",
    "rows_causal_plus1",
    "prefill_causal_plus1",
    "prefill_kv_head",
    "prefill_tile_relative",
    "prefill_no_rescale",
    "prefill_score_scale",
    "prefill_overflow_no_raise",
    "prefill_nan_plausible",
];

/// The defects only a group size under eight or four key heads show: each
/// edit is the same program at Qwen3.6's eight query heads over two key heads.
pub const GQA_MUTANTS: &[Mutant] = &[
    Mutant {
        name: "attn_gqa_read_as_8",
        clause: "attn_f16",
        why: "a key head's query heads start at eight a key head, whatever the group",
        edits: &[(
            "uint head = kv * BLOOMERY_ATTN_GQA + sg * BLOOMERY_ATTN_HPT;",
            "uint head = kv * 8u + sg * BLOOMERY_ATTN_HPT;",
            2,
        )],
    },
    Mutant {
        name: "attn_one_gqa_read_as_8",
        clause: "attn_f16",
        why: "the one-pass flash's query head reads key head head / 8, whatever the group",
        edits: &[(
            "uint kv = head / BLOOMERY_ATTN_GQA;",
            "uint kv = head / 8u;",
            1,
        )],
    },
    Mutant {
        name: "fold_dim_stride_16",
        clause: "attn_f16",
        why: "the fold's lane owns the values from lane * 4, whatever the head divides the SIMD width into",
        edits: &[(
            "uint fold_dim0 = lane * BLOOMERY_ATTN_LANE_DIMS;",
            "uint fold_dim0 = lane * 4u;",
            1,
        )],
    },
    Mutant {
        name: "prefill_pad_rows_stored",
        clause: "attn_prefill",
        why: "the fragment's padded rows store their outputs over the next key head's heads",
        edits: &[(
            "if (!writes || fm >= BLOOMERY_ATTN_GQA) {",
            "if (!writes) {",
            1,
        )],
    },
];

/// The mutants of the other tables a window row keeps: none. The plain
/// launches of the other clause files refuse an instance that cuts a window
/// and gates per head, so the row runs only the clauses of `clauses_window`,
/// which [`WINDOW_MUTANTS`] holds the mutants of.
pub const WINDOW_KEPT: &[&str] = &[];

/// The defects of a layer's window, per-head gate and rope table, each red on
/// the clause that holds that axis. The table-side mutants (`rope_*`) edit no
/// kernel text: the clause reads its name from the gate's command line and
/// builds the defect into the launch's table, so the oracle's table stays
/// the recipe.
pub const WINDOW_MUTANTS: &[Mutant] = &[
    Mutant {
        name: "window_255",
        clause: "attn_window",
        why: "a window of w keys attends w - 1: the last `window - 1` keys",
        edits: &[(
            "return (args.window != 0u && live > args.window) ? live - args.window : 0u;",
            "return (args.window != 0u && live > args.window) ? live - args.window + 1u : 0u;",
            1,
        )],
    },
    Mutant {
        name: "window_257",
        clause: "attn_window",
        why: "a window of w keys attends w + 1: it reads the key before the window",
        edits: &[(
            "return (args.window != 0u && live > args.window) ? live - args.window : 0u;",
            "return (args.window != 0u && live > args.window) ? live - args.window - 1u : 0u;",
            1,
        )],
    },
    Mutant {
        name: "window_ignored",
        clause: "attn_window",
        why: "a sliding layer's launch names its window and the flash attends every live key",
        edits: &[(
            "return (args.window != 0u && live > args.window) ? live - args.window : 0u;",
            "return 0u;",
            1,
        )],
    },
    Mutant {
        name: "window_always_256",
        clause: "attn_window",
        why: "a full layer's launch names no window and the flash cuts one of 256 keys",
        edits: &[(
            "return (args.window != 0u && live > args.window) ? live - args.window : 0u;",
            "return live > 256u ? live - 256u : 0u;",
            1,
        )],
    },
    Mutant {
        name: "window_gqa_read_as_8",
        clause: "attn_window",
        why: "a key head's query heads start at eight a key head, whatever the group: the windowed flash at group 4 pairs a head with another key head's window",
        edits: &[(
            "uint head = kv * BLOOMERY_ATTN_GQA + sg * BLOOMERY_ATTN_HPT;",
            "uint head = kv * 8u + sg * BLOOMERY_ATTN_HPT;",
            2,
        )],
    },
    Mutant {
        name: "window_one_gqa_read_as_8",
        clause: "attn_window",
        why: "the one-pass flash's query head reads key head head / 8, whatever the group: at group 4 a head attends another key head's window and the last two key heads' rows are never stored",
        edits: &[(
            "uint kv = head / BLOOMERY_ATTN_GQA;",
            "uint kv = head / 8u;",
            1,
        )],
    },
    Mutant {
        name: "window_append_owner",
        clause: "attn_window",
        why: "the segment that appends the position's row is the one the unwindowed split gives it: a position past the window edge is stored by a segment that does not own it, or by none",
        edits: &[(
            "uint pos_seg = args.pos >= win_lo ? (args.pos - win_lo) / span : 0u;",
            "uint pos_seg = args.pos / span;",
            1,
        )],
    },
    Mutant {
        name: "window_prefill_mask_off",
        clause: "attn_window_prefill",
        why: "a prompt row attends every key its block's first row does: the keys before its own window are not masked",
        edits: &[(
            "live[j] = live[j] && key >= row_lo;",
            "live[j] = live[j];",
            1,
        )],
    },
    Mutant {
        name: "window_prefill_first_tile_late",
        clause: "attn_window_prefill",
        why: "the prompt pass starts one key tile after the block's first row's window: a key of the window in that tile is never read",
        edits: &[(
            "uint first_tile = attn_window_lo(first, first.keys) / BLOOMERY_ATTN_PF_KEYS;",
            "uint first_tile = attn_window_lo(first, first.keys) / BLOOMERY_ATTN_PF_KEYS + 1u;",
            1,
        )],
    },
    Mutant {
        name: "head_gate_per_channel",
        clause: "attn_head_gate",
        why: "the gate is read per channel, as the packed gate is, from a buffer that holds one value a head",
        edits: &[
            ("float g = q[head];", "float g = q[dim0 + d];", 1),
            ("float g = gate_src[head];", "float g = gate_src[tid];", 1),
            ("float gate = q[head];", "float gate = q[dim];", 1),
            (
                "float g = hgate[(ulong)row * BLOOMERY_ATTN_Q + head];",
                "float g = hgate[(ulong)row * BLOOMERY_ATTN_Q + dim];",
                1,
            ),
        ],
    },
    Mutant {
        name: "head_gate_next_head",
        clause: "attn_head_gate",
        why: "head h is gated by head h + 1's value",
        edits: &[
            ("float g = q[head];", "float g = q[head + 1u];", 1),
            (
                "float g = gate_src[head];",
                "float g = gate_src[head + 1u];",
                1,
            ),
            ("float gate = q[head];", "float gate = q[head + 1u];", 1),
            (
                "float g = hgate[(ulong)row * BLOOMERY_ATTN_Q + head];",
                "float g = hgate[(ulong)row * BLOOMERY_ATTN_Q + head + 1u];",
                1,
            ),
        ],
    },
    Mutant {
        name: "head_gate_no_sigmoid_s1",
        clause: "attn_head_gate",
        why: "the one-pass flash (one segment of the f16 cache) stores the raw gate's product with no sigmoid",
        edits: &[(
            "attn_commit(out, o, attn, g, attn * attn_sigmoid(g), faults, args.layer);",
            "attn_commit(out, o, attn, g, attn * g, faults, args.layer);",
            1,
        )],
    },
    Mutant {
        name: "head_gate_no_sigmoid_fold",
        clause: "attn_head_gate",
        why: "the fold stores the raw gate's product with no sigmoid",
        edits: &[(
            "float merged = attn * attn_sigmoid(gate);",
            "float merged = attn * gate;",
            1,
        )],
    },
    Mutant {
        name: "head_gate_no_sigmoid_prefill",
        clause: "attn_head_gate",
        why: "the prompt pass stores the raw gate's product with no sigmoid",
        edits: &[(
            "float product = attn * attn_sigmoid(g);",
            "float product = attn * g;",
            1,
        )],
    },
    Mutant {
        name: "head_gate_rows_row0",
        clause: "attn_head_gate",
        why: "every row of the rows launch is gated by the first row's values",
        edits: &[(
            "ATTN_HGATE_ACTUAL(row) partials + row * partials_stride,",
            "ATTN_HGATE_ACTUAL(0u) partials + row * partials_stride,",
            1,
        )],
    },
    Mutant {
        name: "head_gate_one_rows_row0",
        clause: "attn_head_gate",
        why: "every row of the one-pass rows launch is gated by the first row's values",
        edits: &[(
            "device float const *gate_src = hgate + (ulong)row * BLOOMERY_ATTN_Q;",
            "device float const *gate_src = hgate;",
            1,
        )],
    },
    Mutant {
        name: "head_gate_prefill_row0",
        clause: "attn_head_gate",
        why: "every row of the prompt pass is gated by the first row's values",
        edits: &[(
            "float g = hgate[(ulong)row * BLOOMERY_ATTN_Q + head];",
            "float g = hgate[head];",
            1,
        )],
    },
    Mutant {
        name: "rope_pairs_32",
        clause: "attn_rope_factors",
        why: "the appended key turns the pairs (i, i + 32), not the NeoX pairs (i, i + 64) of a 128-wide head",
        edits: &[(
            "uint pairs = BLOOMERY_ATTN_ROT / 2u;",
            "uint pairs = 32u;",
            1,
        )],
    },
    Mutant {
        name: "rope_factors_ignored",
        clause: "attn_rope_factors",
        why: "the table the launch reads is built with no factors: every pair rotates",
        edits: &[],
    },
    Mutant {
        name: "rope_exponent_64",
        clause: "attn_rope_factors",
        why: "the table's exponent runs over 64 dims, not the head's 128",
        edits: &[],
    },
];

/// The mutant of this name, and the program it edits.
pub fn find(name: &str, table: Table) -> Option<(&'static Mutant, Program)> {
    all(table).into_iter().find(|(m, _)| m.name == name)
}

/// The mutants of the clauses the 48-value-head row runs: the delta step's
/// and rows', and the head norm over rows. Each is a defect the 32-head row
/// shows too, so the table's clause is red on it at both.
pub const GDN48_KEPT: &[&str] = &[
    "delta_fma_sign",
    "delta_roll_keep",
    "delta_lane_in_place",
    "delta_ring_slot",
    "delta_partial_scale",
    "delta_nan_zero",
    "rows_ring_slot",
    "snap_row_late",
    "snap_ring_dropped",
    "rows_lane_parity",
    "rows_ring_inner",
    "rows_lane_unchecked",
    "rows_prep_prev_row",
    "rows_prep_value_head",
    "rows_prep_value_flag",
    "rows_recur_decay_row",
    "rows_post_partial_block",
    "rows_post_gate_row",
    "headnorm_rows_head",
    "headnorm_rows_no_raise",
];

/// The defects only a value-head count past the SIMD width or a key head
/// serving three value heads shows: each edit is the same program at
/// Qwen3.6's 32 value heads over 16 key heads (two to a key head).
pub const GDN48_MUTANTS: &[Mutant] = &[
    Mutant {
        name: "delta_per_two_step",
        clause: "delta_step",
        why: "the step's seat takes two value heads for each key head, whatever the layer's counts",
        edits: &[(
            "    uint per = n_v / n_k;\n    s.kh = ",
            "    uint per = 2u;\n    s.kh = ",
            1,
        )],
    },
    Mutant {
        name: "delta_per_two_prep",
        clause: "delta_rows",
        why: "the state-free half of a chunk prepares two value heads for each key head, whatever the layer's counts",
        edits: &[(
            "    uint per = n_v / n_k;\n    device float *row",
            "    uint per = 2u;\n    device float *row",
            1,
        )],
    },
    Mutant {
        name: "headnorm_scales_one_round",
        clause: "headnorm_q4k",
        why: "the head norm's scales stop at the SIMD width, so the value heads past it keep unset scales",
        edits: &[(
            "for (uint head = tiisg; head < BLOOMERY_DELTA_VALUE_HEADS; head += 32u) {",
            "for (uint head = tiisg; head < 32u; head += 32u) {",
            1,
        )],
    },
    Mutant {
        name: "a4_head_wrap_32",
        clause: "headnorm_a4",
        why: "an MLX 4-bit group reads its head's scale modulo 32, so the value heads past 32 read the first heads'",
        edits: &[(
            "const int head = ib >> 1;",
            "const int head = (ib >> 1) & 31;",
            1,
        )],
    },
];

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
        edits: &[("- dh[1] * (lane.sumy[0]", "+ dh[1] * (lane.sumy[0]", 1)],
    },
    Mutant {
        name: "q6k_offset",
        clause: "q6k_matvec",
        why: "one of the four 6-bit values is offset by 31, not 32",
        edits: &[(
            "((char)((q1[l] & 0xF) | h0) - 32);",
            "((char)((q1[l] & 0xF) | h0) - 31);",
            1,
        )],
    },
    Mutant {
        name: "norm_gain",
        clause: "norm_on_load",
        why: "the Q4_K norm drops the gain of a value",
        edits: &[("a0 *= g0[k];", "a0 *= 1.f;", 1)],
    },
    Mutant {
        name: "norm_no_raise",
        clause: "norm_on_load",
        why: "a norm that cannot take its input or its gain writes NaN rows and raises nothing",
        edits: &[(
            "raise(faults, layer, BLOOMERY_FAULT_SITE_NORM_QUANT);",
            "",
            2,
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
            1,
        )],
    },
    Mutant {
        name: "tail_store",
        clause: "row_tail",
        why: "the store of a pair no longer stops at the last row",
        edits: &[(TAIL_STORE_FIND, TAIL_STORE_REPLACE, 1)],
    },
    Mutant {
        name: "embed_no_raise",
        clause: "embed",
        why: "an id past the vocabulary raises nothing (both embeddings')",
        edits: &[(
            "raise(faults, args.layer, BLOOMERY_FAULT_SITE_TOKEN_ID);",
            "",
            2,
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
        edits: &[(ROW_POINTER, ROW_POINTER_RAW, 1), (EARLY_RETURN, "", 1)],
    },
    Mutant {
        name: "seg_weight_offset",
        clause: "seg_bit",
        why: "a segment reads the weights of the first segment",
        edits: &[("W + (ulong)args.segs[s].wblocks,", "W,", 1)],
    },
    Mutant {
        name: "seg_last_segment",
        clause: "seg_bounds",
        why: "the search for the segment stops before the last one",
        edits: &[(
            "for (uint i = 0; i < args.nseg; ++i) {",
            "for (uint i = 0; i < 3 && i < args.nseg; ++i) {",
            1,
        )],
    },
    Mutant {
        name: "headnorm_divisor",
        clause: "headnorm",
        why: "the head scale divides the sum by twice the head's columns",
        edits: &[(
            "sum / float(BLOOMERY_DELTA_HEAD_COLS)",
            "sum / float(BLOOMERY_DELTA_HEAD_COLS * 2)",
            1,
        )],
    },
    Mutant {
        name: "headnorm_no_raise",
        clause: "headnorm_nan",
        why: "a head the norm cannot take, and a group sum the row loop cannot take, raise nothing",
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
        name: "fold_tail",
        clause: "fold",
        why: "the fold's store no longer stops at the last row",
        edits: &[(
            "    for (short r = 0; r < NR0 && row0 + r < (int)args.rows; ++r) {\n        float d = simd_sum(sum[r]);",
            "    for (short r = 0; r < NR0; ++r) {\n        float d = simd_sum(sum[r]);",
            1,
        )],
    },
    Mutant {
        name: "nsg_first_row",
        clause: "nsg",
        why: "the first row of a threadgroup ignores the SIMD group count",
        edits: &[("BLOOMERY_MATVEC_SIMD_GROUPS + sg", "2 + sg", 1)],
    },
    Mutant {
        name: "router_logit",
        clause: "router_logit",
        why: "one component of every float4 product is dropped",
        edits: &[("acc += w.x * h.x", "acc += 0.f", 1)],
    },
    Mutant {
        name: "router_ids",
        clause: "router_ids",
        why: "the selection keeps the smaller logit",
        edits: &[("argmax_key(l, row)", "argmax_key(-l, row)", 1)],
    },
    Mutant {
        name: "router_tie",
        clause: "router_tie",
        why: "at an equal value the larger id is taken",
        edits: &[(
            "((ulong)o << 32) | (ulong)(~i)",
            "((ulong)o << 32) | (ulong)(i)",
            1,
        )],
    },
    Mutant {
        name: "router_weights",
        clause: "router_weights",
        why: "a routed weight is the exponential, not its share of the eight",
        edits: &[("weights[s] = term[s] / sum_exp", "weights[s] = term[s]", 1)],
    },
    Mutant {
        name: "router_ticket",
        clause: "router_ticket",
        why: "a routed id is one past the selected expert",
        edits: &[("ids[s] = best_i", "ids[s] = best_i + 1u", 1)],
    },
    Mutant {
        name: "router_nan",
        clause: "router_nan",
        why: "a non-finite logit raises nothing",
        edits: &[("raise(faults, layer, BLOOMERY_FAULT_SITE_ROUTER);", "", 1)],
    },
    Mutant {
        name: "router_bounds",
        clause: "router_bounds",
        why: "the last threadgroup does not run the epilogue",
        edits: &[(
            "if (last_flag == 1u) {\n        route_last(",
            "if (last_flag == 2u) {\n        route_last(",
            1,
        )],
    },
    Mutant {
        name: "router_rounds",
        clause: "router_ids",
        why: "the exclusion keys out every key but the winner, so every slot repeats it",
        edits: &[(
            "            if (held[j] == chosen_key) {\n                held[j] = 0u;\n            }",
            "            if (held[j] != chosen_key) {\n                held[j] = 0u;\n            }",
            1,
        )],
    },
    Mutant {
        name: "router_bad",
        clause: "router_nan",
        why: "a non-finite logit's word does not cross the lanes",
        edits: &[("mine_bad |= isfinite(l) ? 0u : 1u;", "mine_bad |= 0u;", 1)],
    },
    Mutant {
        name: "router_norm_no_raise",
        clause: "router_norm_nan",
        why: "a norm the threadgroup cannot take raises nothing",
        edits: &[(
            "raise(faults, args.layer, BLOOMERY_FAULT_SITE_NORM_QUANT);",
            "",
            1,
        )],
    },
    Mutant {
        name: "pair_no_silu",
        clause: "pair",
        why: "the pair multiplies gate by up and skips silu",
        edits: &[(
            "return gate / (1.0f + exp(-gate)) * up;",
            "return gate * up;",
            1,
        )],
    },
    Mutant {
        name: "pair_id_zero",
        clause: "pair_id",
        why: "an id past the stack writes zero",
        edits: &[(
            "y[(ulong)slot * stride + (uint)(row0 + o)] = NAN;",
            "y[(ulong)slot * stride + (uint)(row0 + o)] = 0.0f;",
            1,
        )],
    },
    Mutant {
        name: "pair_shared_stack",
        clause: "pair_shared",
        why: "slot 8 reads the expert stack",
        edits: &[(
            "pair_slot<QS, SCALED>(shexp_gate, shexp_up, x,",
            "pair_slot<QS, SCALED>((device const typename QS::Block *)gate, (device const typename QS::Block *)up, x,",
            1,
        )],
    },
    Mutant {
        name: "pair_norm_gain",
        clause: "pair",
        why: "the pair's Q4_K norm drops the gain of a value",
        edits: &[("a0 *= g0[k];", "a0 *= 1.f;", 1)],
    },
    Mutant {
        name: "pair_norm_no_raise",
        clause: "pair_norm",
        why: "a pair norm that cannot take its input writes NaN rows and raises nothing",
        edits: &[(
            "pair_write_bad(y, slot, args.stride, row0, (int)args.rows, faults, args.layer,\n                           BLOOMERY_FAULT_SITE_NORM_QUANT);",
            "for (short o = 0; o < PAIR_OUTS && row0 + o < (int)args.rows; ++o) {\n                y[(ulong)slot * args.stride + (uint)(row0 + o)] = NAN;\n            }",
            1,
        )],
    },
    Mutant {
        name: "fold_drop_weight",
        clause: "fold",
        why: "the fold loads the activation without the route weight",
        edits: &[(
            "float scale = w[Q::fold_slot(ib)];",
            "float scale = 1.0f;",
            1,
        )],
    },
    Mutant {
        name: "a4_lane_pair",
        clause: "a4_matvec",
        why: "one nibble of an A4 word pairs the lane value beside it",
        edits: &[(
            "acc[1] += lane.yl[4 * m + 1] * (words[m] & 0x00F0);",
            "acc[1] += lane.yl[4 * m + 2] * (words[m] & 0x00F0);",
            1,
        )],
    },
    Mutant {
        name: "a4_word_offset",
        clause: "a4_matvec",
        why: "an A4 lane reads the group's code words a quarter early",
        edits: &[(
            "device const uint32_t * q = blk.qs + 2 * lane.iq;",
            "device const uint32_t * q = blk.qs + lane.iq;",
            1,
        )],
    },
    Mutant {
        name: "a4_bias_sign",
        clause: "a4_matvec",
        why: "the A4 bias term is subtracted, not added",
        edits: &[(
            "+ as_type<float>(sb & 0xFFFF0000u) * lane.sumy;",
            "- as_type<float>(sb & 0xFFFF0000u) * lane.sumy;",
            1,
        )],
    },
    Mutant {
        name: "a4_scale_bias",
        clause: "a4_matvec",
        why: "the A4 code dot is scaled by the bias, not the scale",
        edits: &[(
            "return as_type<float>(sb << 16) * ((acc[0] + 1.f/16.f * acc[1]) +",
            "return as_type<float>(sb & 0xFFFF0000u) * ((acc[0] + 1.f/16.f * acc[1]) +",
            1,
        )],
    },
    Mutant {
        name: "a4_fold_slot",
        clause: "a4_fold",
        why: "the A4 fold maps every sixteen groups, not every eight, to a slot",
        edits: &[(
            "    static inline float2 head_scales(threadgroup const float * scales, int ib) {\n        const int head = ib >> 1;\n        return float2(scales[head], scales[head]);\n    }\n\n    static inline int fold_blocks() {\n        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());\n    }\n\n    static inline int fold_slot(int ib) {\n        return ib / (BLOOMERY_EXPERT_K / values_per_block());\n    }",
            "    static inline float2 head_scales(threadgroup const float * scales, int ib) {\n        const int head = ib >> 1;\n        return float2(scales[head], scales[head]);\n    }\n\n    static inline int fold_blocks() {\n        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());\n    }\n\n    static inline int fold_slot(int ib) {\n        return ib / 16;\n    }",
            1,
        )],
    },
    Mutant {
        name: "a4_tail_store",
        clause: "a4_row_tail",
        why: "the store of a pair no longer stops at the last row",
        edits: &[(TAIL_STORE_FIND, TAIL_STORE_REPLACE, 1)],
    },
    Mutant {
        name: "a4_norm_gain",
        clause: "a4_norm",
        why: "the A4 norm load drops the gain of a value",
        edits: &[("a0 *= gq.x;", "a0 *= 1.f;", 1)],
    },
    Mutant {
        name: "a4_seg_weights",
        clause: "a4_seg",
        why: "a segment reads the weights of the first segment",
        edits: &[("W + (ulong)args.segs[s].wblocks,", "W,", 1)],
    },
    Mutant {
        name: "a4_pair_sum_sq",
        clause: "a4_pair",
        why: "the A4 norm load's sum of squares drops a value",
        edits: &[(
            "*sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);\n                a0 *= gq.x;",
            "*sum_sq += (a0 * a0 + a1 * a1) + a2 * a2;\n                a0 *= gq.x;",
            1,
        )],
    },
    Mutant {
        name: "a4_load_x",
        clause: "a4_matvec",
        why: "an A4 lane's activation load takes the next quarter of its sixteen values, wrapped",
        edits: &[(
            "const float4 xq = act4(block + base + 4 * l);",
            "const float4 xq = act4(block + base + 4 * ((l + 1) & 3));",
            1,
        )],
    },
    Mutant {
        name: "a4_load_gain",
        clause: "a4_norm",
        why: "an A4 lane's gain load takes the next quarter of its sixteen values, wrapped",
        edits: &[(
            "gq = act4(gain + base + 4 * l);",
            "gq = act4(gain + base + 4 * ((l + 1) & 3));",
            1,
        )],
    },
    Mutant {
        name: "q4k_load_x",
        clause: "q4k_matvec",
        why: "a Q4_K lane's second activation group is read from sixty-four values on, not thirty-two",
        edits: &[(
            "const float4 x1 = act4(block + base + j +  32);",
            "const float4 x1 = act4(block + base + j +  64);",
            1,
        )],
    },
    Mutant {
        name: "q4k_load_gain",
        clause: "norm_on_load",
        why: "a Q4_K lane's third gain group is read from the fourth group's place",
        edits: &[(
            "g2 = act4(gain + base + j + 128);",
            "g2 = act4(gain + base + j + 160);",
            1,
        )],
    },
    Mutant {
        name: "q6k_load_x",
        clause: "q6k_matvec",
        why: "a Q6_K lane's third activation group is read from the fourth group's place",
        edits: &[(
            "const float4 x2 = act4(block + base + 64);",
            "const float4 x2 = act4(block + base + 96);",
            1,
        )],
    },
    Mutant {
        name: "q8_0_load_x",
        clause: "q8_0_matvec",
        why: "a Q8_0 lane's activation load takes the other half of its eight values",
        edits: &[(
            "const float4 x8 = act4(block + base + l);",
            "const float4 x8 = act4(block + base + (l ^ 4));",
            1,
        )],
    },
    Mutant {
        name: "q8_0_load_gain",
        clause: "q8_0_norm",
        why: "a Q8_0 lane's gain load takes the other half of its eight values",
        edits: &[(
            "g8 = act4(gain + base + l);",
            "g8 = act4(gain + base + (l ^ 4));",
            1,
        )],
    },
    Mutant {
        name: "a4_head_logit",
        clause: "a4_head",
        why: "the first stored logit of every SIMD group is doubled",
        edits: &[(
            "gain, nullptr,\n                                                         faults, first_row, (int)args.rows, tiisg);",
            "gain, nullptr,\n                                                         faults, first_row, (int)args.rows, tiisg);\n    if (tiisg == 0u && first_row < (int)args.rows) {\n        logits[first_row] *= 2.f;\n    }",
            1,
        )],
    },
    Mutant {
        name: "a4_routed_bias_sign",
        clause: "grouped_gemm",
        why: "the A4 values the routed GEMM stages (dequant16) subtract the bias, not add it",
        edits: &[(
            "const float b = bias_of(blk);",
            "const float b = -bias_of(blk);",
            1,
        )],
    },
    Mutant {
        name: "a4_headnorm_scale",
        clause: "a4_headnorm",
        why: "an A4 group takes the neighbouring head's scale",
        edits: &[(
            "const int head = ib >> 1;",
            "const int head = (ib >> 1) ^ 1;",
            1,
        )],
    },
    Mutant {
        name: "a4_embed_code",
        clause: "a4_embed",
        why: "the embed's code drops its top bit",
        edits: &[(
            "(word >> (4 * (j % 8))) & 0xF",
            "(word >> (4 * (j % 8))) & 0x7",
            1,
        )],
    },
    Mutant {
        name: "fold_no_resid",
        clause: "fold",
        why: "the fold overwrites the residual",
        edits: &[("y[(uint)(row0 + r)] += d;", "y[(uint)(row0 + r)] = d;", 1)],
    },
    Mutant {
        name: "delta_fma_sign",
        clause: "delta_step",
        why: "the state update subtracts k*u",
        edits: &[(
            "sp[i] = fma(kk[i], u, sp[i]);",
            "sp[i] = fma(-kk[i], u, sp[i]);",
            1,
        )],
    },
    Mutant {
        name: "delta_roll_keep",
        clause: "delta_roll",
        why: "the state store keeps the value it loaded",
        edits: &[("st[i] = sp[i];", "st[i] = st[i];", 1)],
    },
    Mutant {
        name: "delta_lane_in_place",
        clause: "delta_step",
        why: "the step writes the lane it read, not the lane its write word names",
        edits: &[(
            "uint word_to = lane_buf[args.lane_to];",
            "uint word_to = lane_buf[args.lane_at];",
            1,
        )],
    },
    Mutant {
        name: "delta_ring_slot",
        clause: "delta_ring",
        why: "the ring write uses the next slot",
        edits: &[(
            "uint slot = pos % BLOOMERY_DELTA_RING_ROWS;",
            "uint slot = (pos + 1u) % BLOOMERY_DELTA_RING_ROWS;",
            1,
        )],
    },
    Mutant {
        name: "delta_partial_scale",
        clause: "delta_partial",
        why: "the partial sum is stored twice",
        edits: &[(
            "partials_row[h * BLOOMERY_DELTA_COLUMN_BLOCKS + s.block] = p;",
            "partials_row[h * BLOOMERY_DELTA_COLUMN_BLOCKS + s.block] = p + p;",
            1,
        )],
    },
    Mutant {
        name: "delta_nan_zero",
        clause: "delta_nan",
        why: "a non-finite lane stores zero",
        edits: &[("st[i] = NAN;", "st[i] = 0.0f;", 1)],
    },
    Mutant {
        name: "shortconv_tap_order",
        clause: "shortconv_rows",
        why: "the taps run newest first: weight k pairs with the predecessor k back, not taps - 1 - k back",
        edits: &[("r, a.taps - 1u - k, a);", "r, k, a);", 1)],
    },
    Mutant {
        name: "shortconv_width_short",
        clause: "shortconv_rows",
        why: "the conv runs one tap short, a width-6 conv on seven taps' weights: the newest tap is dropped",
        edits: &[(
            "for (uint k = 0u; k < a.taps; ++k) {",
            "for (uint k = 0u; k + 1u < a.taps; ++k) {",
            1,
        )],
    },
    Mutant {
        name: "shortconv_no_b",
        clause: "shortconv_rows",
        why: "the B gate is skipped: u is x, not B times x",
        edits: &[(
            "return row[c] * row[2u * a.hidden + c];",
            "return row[2u * a.hidden + c];",
            1,
        )],
    },
    Mutant {
        name: "shortconv_no_c",
        clause: "shortconv_rows",
        why: "the C epilogue is skipped: the output is the conv sum alone",
        edits: &[(
            "float y = acc * bcx[r * a.x_stride + a.hidden + c];",
            "float y = acc;",
            1,
        )],
    },
    Mutant {
        name: "shortconv_start_stale",
        clause: "shortconv_start",
        why: "a predecessor before the sequence's first position reads the ring's stale slot, not 0",
        edits: &[(
            "    if (pos0 < before) {\n        return 0.0f;\n    }\n",
            "",
            1,
        )],
    },
    Mutant {
        name: "shortconv_chunk_ring",
        clause: "shortconv_rows",
        why: "every predecessor of a chunk's rows is read from the ring, so a row inside the chunk reads the slot of the position before the chunk instead of its in-chunk predecessor",
        edits: &[
            ("if (back <= r) {", "if (false) {", 1),
            ("uint before = back - r;", "uint before = back;", 1),
        ],
    },
    Mutant {
        name: "shortconv_ring_alias",
        clause: "shortconv_state",
        why: "the ring is cut to five rows, so a seven-tap conv's positions 1 and 6 back share a slot",
        edits: &[(
            "return pos % ring_rows;",
            "return pos % min(ring_rows, 5u);",
            1,
        )],
    },
    Mutant {
        name: "shortconv_no_raise",
        clause: "shortconv_nan",
        why: "an element that is not finite raises nothing",
        edits: &[(
            "inline void sc_raise(device atomic_uint *faults, uint layer) {\n    raise(faults, layer, BLOOMERY_FAULT_SITE_LINEAR_CONV);\n}",
            "inline void sc_raise(device atomic_uint *faults, uint layer) {\n}",
            1,
        )],
    },
    Mutant {
        name: "shortconv_nan_plausible",
        clause: "shortconv_nan",
        why: "an element that is not finite is written as zero, not NaN",
        edits: &[("return ok ? v : NAN;", "return ok ? v : 0.0f;", 1)],
    },
    Mutant {
        name: "attn_score_scale",
        clause: "attn_f16",
        why: "the score is twice the scaled dot, in the segmented and the one-pass flash",
        edits: &[(
            "float score = dot * args.scale;",
            "float score = dot * args.scale * 2.0f;",
            2,
        )],
    },
    Mutant {
        name: "attn_q8_id",
        clause: "attn_q8",
        why: "a q8_0 code uses 126 / amax instead of the host quantizer's 127 / amax",
        edits: &[("127.0f / amax", "126.0f / amax", 1)],
    },
    Mutant {
        name: "attn_key_count",
        clause: "attn_fault",
        why: "a live key count of zero or past the cache raises nothing",
        edits: &[(
            "raise(faults, args.layer, BLOOMERY_FAULT_SITE_KEY_COUNT);",
            "",
            1,
        )],
    },
    Mutant {
        name: "attn_gate_s1",
        clause: "attn_f16",
        why: "the one-pass flash (one segment of the f16 cache) stores the attention without the gate",
        edits: &[(
            "attn_commit(out, o, attn, g, attn * attn_sigmoid(g), faults, args.layer);",
            "attn_commit(out, o, attn, g, attn, faults, args.layer);",
            1,
        )],
    },
    Mutant {
        name: "attn_gate_s1_q8",
        clause: "attn_q8",
        why: "one segment of the q8_0 cache stores the attention without the gate",
        edits: &[(
            "float gated = attn * attn_sigmoid(g);",
            "float gated = attn;",
            1,
        )],
    },
    Mutant {
        name: "attn_gate_merge",
        clause: "attn_f16",
        why: "the fold stores the attention without the gate",
        edits: &[(
            "float merged = attn * attn_sigmoid(gate);",
            "float merged = attn;",
            1,
        )],
    },
    Mutant {
        name: "attn_no_fold",
        clause: "attn_f16",
        why: "the fold never runs, so past one segment the output is never written",
        edits: &[(
            "if (args.segments > 1u) {\n        ticket_is_last(ticket, groups, tid, last_flag);",
            "if (args.segments > 64u) {\n        ticket_is_last(ticket, groups, tid, last_flag);",
            1,
        )],
    },
    Mutant {
        name: "attn_f16_round",
        clause: "attn_f16",
        why: "the f16 conversion rounds one bit high, so the appended row is not the host quantizer's bits",
        edits: &[(
            "uint t = v + 0x0fffu + ((v >> 13) & 1u);",
            "uint t = v + 0x1fffu + ((v >> 13) & 1u);",
            1,
        )],
    },
    Mutant {
        name: "attn_s1_partials",
        clause: "attn_q8",
        why: "one segment of the q8_0 cache writes partials and leaves the output untouched",
        edits: &[("if (args.segments == 1u)", "if (args.segments == 0u)", 1)],
    },
    Mutant {
        name: "attn_half",
        clause: "attn_half",
        why: "the hardware half conversion flips the low bit",
        edits: &[(
            "float y = float(as_type<half>(bits));",
            "float y = float(as_type<half>(ushort(bits ^ 1u)));",
            1,
        )],
    },
    Mutant {
        name: "attn_bounds",
        clause: "attn_bounds",
        why: "the one-pass flash's chunk reads neighbouring keys, not keys a round of SIMD groups apart, so one segment's keys are partly attended",
        edits: &[(
            "uint j = base + c * BLOOMERY_ATTN_ONE_SG;",
            "uint j = base + c;",
            1,
        )],
    },
    Mutant {
        name: "attn_span_tile",
        clause: "attn_f16",
        why: "every segment of the split spans one tile, so a longer key count is only partly attended",
        edits: &[(
            "return per < BLOOMERY_ATTN_TILE ? BLOOMERY_ATTN_TILE : per;",
            "return BLOOMERY_ATTN_TILE;",
            1,
        )],
    },
    Mutant {
        name: "head_logit",
        clause: "head_logit",
        why: "the first stored logit of every SIMD group is doubled",
        edits: &[(
            "gain, nullptr,\n                                                         faults, first_row, (int)args.rows, tiisg);",
            "gain, nullptr,\n                                                         faults, first_row, (int)args.rows, tiisg);\n    if (tiisg == 0u && first_row < (int)args.rows) {\n        logits[first_row] *= 2.f;\n    }",
            1,
        )],
    },
    Mutant {
        name: "head_id",
        clause: "head_id",
        why: "a thread of the pick keeps the smaller logit",
        edits: &[("if (v > best_v) {", "if (v < best_v) {", 1)],
    },
    Mutant {
        name: "head_tie_higher",
        clause: "head_id",
        why: "a thread of the pick keeps the later row of an equal logit, so a tie is not \
              the lowest row's",
        edits: &[("if (v > best_v) {", "if (v >= best_v) {", 1)],
    },
    Mutant {
        name: "head_unroll_last",
        clause: "head_id",
        why: "the pick's unrolled run takes three of its four loads, so a winner in the \
              fourth is lost",
        edits: &[(
            "FOR_UNROLL (short u = 0; u < 4; ++u) {\n            const uint row",
            "FOR_UNROLL (short u = 0; u < 3; ++u) {\n            const uint row",
            1,
        )],
    },
    Mutant {
        name: "head_quad_last",
        clause: "head_id",
        why: "the pick's unrolled run takes three logits of each quad, so a winner in a \
              quad's last is lost",
        edits: &[(
            "FOR_UNROLL (short c = 0; c < 4; ++c) {\n                head_take(v[u][c]",
            "FOR_UNROLL (short c = 0; c < 3; ++c) {\n                head_take(v[u][c]",
            1,
        )],
    },
    Mutant {
        name: "head_quad_tail",
        clause: "head_id",
        why: "the pick's quad tail stops a quad short, so a winner in the last quad is lost",
        edits: &[(
            "for (; q < nq; q += HEAD_THREADS) {",
            "for (; q + 1u < nq; q += HEAD_THREADS) {",
            1,
        )],
    },
    Mutant {
        name: "head_last_row",
        clause: "head_partial",
        why: "the pick's row tail stops one row short, so the last row's logit is never read",
        edits: &[(
            "if (row < args.rows) {\n        head_take(logits[row]",
            "if (row + 1u < args.rows) {\n        head_take(logits[row]",
            1,
        )],
    },
    Mutant {
        name: "head_id_plus_one",
        clause: "head_repeat",
        why: "the step id is one past the selected row",
        edits: &[(
            "step_id[0] = ~uint(best_key)",
            "step_id[0] = ~uint(best_key) + 1u",
            1,
        )],
    },
    Mutant {
        name: "head_nan",
        clause: "head_nan",
        why: "a non-finite logit raises nothing",
        edits: &[(
            "raise(faults, args.layer, BLOOMERY_FAULT_SITE_LOGIT);",
            "",
            2,
        )],
    },
    Mutant {
        name: "head_nan_row",
        clause: "head_nan",
        why: "a non-finite logit raises nothing in the pick's scan, and the empty key's raise \
              stays, which a finite winner never reaches",
        edits: &[(
            "if (!isfinite(v)) {\n        raise(faults, args.layer, BLOOMERY_FAULT_SITE_LOGIT);\n    }",
            "if (!isfinite(v)) {\n    }",
            1,
        )],
    },
    Mutant {
        name: "head_bounds",
        clause: "head_bounds",
        why: "the mat-vec does not run, so the logits stay the sentinel",
        edits: &[(
            "matvec_rows<Q, !SCALED, false, false, false, SCALED>(weights, x, logits,",
            "if (false) matvec_rows<Q, !SCALED, false, false, false, SCALED>(weights, x, logits,",
            1,
        )],
    },
    Mutant {
        name: "head_all_nan",
        clause: "head_all_nan",
        why: "an all-NaN head stores the empty key's id, which is token 0",
        edits: &[("best_key == empty_key", "best_key == 0u", 1)],
    },
    Mutant {
        name: "head_stage",
        clause: "head_repeat",
        why: "the threadgroup's key reduction skips the last SIMD group's share",
        edits: &[(
            "    ulong best = stage[0];\n    for (uint g = 1u; g < sg_count; ++g) {\n        best = max(best, stage[g]);\n    }",
            "    ulong best = stage[0];\n    for (uint g = 1u; g < sg_count - 1u; ++g) {\n        best = max(best, stage[g]);\n    }",
            1,
        )],
    },
    Mutant {
        name: "rows_ring_slot",
        clause: "delta_rows",
        why: "the ring commit writes each position one slot late",
        edits: &[(
            "uint slot = (args.pos + r) % BLOOMERY_DELTA_RING_ROWS;",
            "uint slot = (args.pos + r + 1u) % BLOOMERY_DELTA_RING_ROWS;",
            1,
        )],
    },
    Mutant {
        name: "snap_row_late",
        clause: "delta_rows_snap",
        why: "the rows launch copies the state lane one row after the row its copy names",
        edits: &[(
            "if (copy < copies && r == snaps->after[copy]) {",
            "if (copy < copies && r == snaps->after[copy] + 1u) {",
            1,
        )],
    },
    Mutant {
        name: "snap_ring_dropped",
        clause: "delta_rows_snap",
        why: "the rows launch leaves a copy's conv ring unwritten",
        edits: &[(
            "    snap[(ulong)k * snaps.ring_len + s * args.ch + c] = v;",
            "    (void)v;",
            1,
        )],
    },
    Mutant {
        name: "rows_lane_parity",
        clause: "delta_rows",
        why: "the rows launch writes the state to the lane it read, not the lane the write word names",
        edits: &[(
            "    if (lane_ok) {\n        for (uint i = 0u; i < 16u; ++i) {\n            state[base_to + key_of(seat.j, i)] = st[i];",
            "    if (lane_ok) {\n        for (uint i = 0u; i < 16u; ++i) {\n            state[base + key_of(seat.j, i)] = st[i];",
            1,
        )],
    },
    Mutant {
        name: "rows_ring_inner",
        clause: "delta_rows",
        why: "a row from the fourth on reads its conv taps from the ring, not from the chunk",
        edits: &[
            (
                "    if (d <= r) {\n        return x[(r - d) * x_stride + c];",
                "    if (d == 0u || (d <= r && r < 3u)) {\n        return x[(r - d) * x_stride + c];",
                1,
            ),
            (
                "    uint back = d - r;",
                "    uint back = d > r ? d - r : d;",
                1,
            ),
        ],
    },
    Mutant {
        name: "rows_rope_relative",
        clause: "norm_rope_rows",
        why: "the chunk's keys are roped at their position in the chunk, not their absolute position",
        edits: &[(
            "v + row * args.v_stride, k_gain, rope, faults,",
            "v + row * args.v_stride, k_gain, rope - (ulong)args.from * BLOOMERY_ATTN_ROT, faults,",
            1,
        )],
    },
    Mutant {
        name: "rows_causal_plus1",
        clause: "attn_rows_v0",
        why: "every row of the chunk attends one key past its own position",
        edits: &[("a.keys = a.pos + 1u;", "a.keys = a.pos + 2u;", 1)],
    },
    Mutant {
        name: "rows_norm_shared",
        clause: "norm_rows",
        why: "every row of the chunk takes row 0's sum of squares, so row 0's scale",
        edits: &[(
            "const uint f4s = args.cols / 4u;\n\n    float sum_sq = 0.f;\n    for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {\n        float4 xv = norm_rows_load4(xr, f4);",
            "const uint f4s = args.cols / 4u;\n\n    float sum_sq = 0.f;\n    for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {\n        float4 xv = norm_rows_load4(x, f4);",
            1,
        )],
    },
    Mutant {
        name: "rows_lane_unchecked",
        clause: "delta_rows_nan",
        why: "the rows launch runs its rows as though the lane words were in range, so a word past the lanes raises nothing",
        edits: &[("bool row_ok = lane_ok;", "bool row_ok = true;", 1)],
    },
    Mutant {
        name: "rows_prep_prev_row",
        clause: "delta_rows",
        why: "the state-free half reads the previous row's conv window for the query channels",
        edits: &[(
            "kh * head + tid, args.step.pos, r, ch, args.x_stride, x, ring, conv_w,\n                        &q_ok);",
            "kh * head + tid, args.step.pos, r > 0u ? r - 1u : 0u, ch, args.x_stride, x, ring, conv_w,\n                        &q_ok);",
            1,
        )],
    },
    Mutant {
        name: "rows_prep_value_head",
        clause: "delta_rows_nan",
        why: "the state-free half gives a key head the value heads of the other head map, so a bad key head flags the wrong value heads",
        edits: &[(
            "uint h = args.step.grouped != 0u ? kh * per + i : kh + i * n_k;",
            "uint h = args.step.grouped == 0u ? kh * per + i : kh + i * n_k;",
            1,
        )],
    },
    Mutant {
        name: "rows_prep_value_flag",
        clause: "delta_rows_nan",
        why: "a value conv that is not finite is not in the recurrence group's status word",
        edits: &[(
            "bool unusable = group_bad != 0u || block_bad != 0u || !scalars_ok;",
            "bool unusable = group_bad != 0u || !scalars_ok;",
            1,
        )],
    },
    Mutant {
        name: "rows_recur_decay_row",
        clause: "delta_rows",
        why: "the recurrence takes the decay of the previous row",
        edits: &[(
            "float decay = row[args.prep.decay + seat.h];",
            "float decay = (r > 0u ? row - args.prep.stride : row)[args.prep.decay + seat.h];",
            1,
        )],
    },
    Mutant {
        name: "rows_post_partial_block",
        clause: "delta_rows",
        why: "the partial sum of a column block is taken over its neighbour's columns",
        edits: &[(
            "gdn_block_sum(o2 + tid * BLOOMERY_DELTA_COLS_PER_BLOCK)",
            "gdn_block_sum(o2 + (tid ^ 1u) * BLOOMERY_DELTA_COLS_PER_BLOCK)",
            1,
        )],
    },
    Mutant {
        name: "rows_post_gate_row",
        clause: "delta_rows",
        why: "every row's numerator takes row 0's gate",
        edits: &[(
            "prep[(ulong)r * args.prep.stride + args.prep.gate + h * head + tid]",
            "prep[args.prep.gate + h * head + tid]",
            1,
        )],
    },
    Mutant {
        name: "prefill_causal_plus1",
        clause: "attn_prefill",
        why: "every row of the prefill attends one key past its own position",
        edits: &[("live[j] = key <= a.pos;", "live[j] = key <= a.pos + 1u;", 1)],
    },
    Mutant {
        name: "prefill_kv_head",
        clause: "attn_prefill",
        why: "the prefill reads the other key head's rows for its query heads",
        edits: &[(
            "ulong at = ((ulong)kv * args.ctx + key) * PF_ROW_WORDS + c;",
            "ulong at = ((ulong)(1u - kv) * args.ctx + key) * PF_ROW_WORDS + c;",
            1,
        )],
    },
    Mutant {
        name: "prefill_tile_relative",
        clause: "attn_prefill",
        why: "the prefill's key tiles start at the chunk's first position, not at key 0",
        edits: &[(
            "uint k0 = t * BLOOMERY_ATTN_PF_KEYS;",
            "uint k0 = args.from + t * BLOOMERY_ATTN_PF_KEYS;",
            1,
        )],
    },
    Mutant {
        name: "prefill_no_rescale",
        clause: "attn_prefill",
        why: "the prefill never rescales its accumulators when a tile raises the running max",
        edits: &[(
            "if (t > 0u && simd_any(factor != 1.0f)) {",
            "if (false) {",
            1,
        )],
    },
    Mutant {
        name: "prefill_score_scale",
        clause: "attn_prefill",
        why: "the prefill's score is not multiplied by the scale",
        edits: &[("sv[j] = raw * a.scale;", "sv[j] = raw;", 1)],
    },
    Mutant {
        name: "prefill_overflow_no_raise",
        clause: "attn_prefill_nan",
        why: "a query past f16's range raises nothing",
        edits: &[(
            "raise(faults, a.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);",
            "",
            1,
        )],
    },
    Mutant {
        name: "prefill_nan_plausible",
        clause: "attn_prefill_nan",
        why: "a row that is not whole is written as zeros, not NaN",
        edits: &[(
            "out[at] = NAN;\n                continue;",
            "out[at] = 0.0f;\n                continue;",
            1,
        )],
    },
    Mutant {
        name: "rows_norm_no_raise",
        clause: "norm_rows_nan",
        why: "a row the norm cannot take raises nothing",
        edits: &[("raise(word, layer, BLOOMERY_FAULT_SITE_NORM_QUANT);", "", 1)],
    },
    Mutant {
        name: "rows_norm_plausible",
        clause: "norm_rows_nan",
        why: "a row the norm cannot take is written as zeros, not NaN",
        edits: &[(
            "norm_rows_raise(faults, args.layer);\n        }\n        for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {\n            *((device float4 *)(yr + f4 * 4u)) = float4(NAN);",
            "norm_rows_raise(faults, args.layer);\n        }\n        for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {\n            *((device float4 *)(yr + f4 * 4u)) = float4(0.f);",
            1,
        )],
    },
    Mutant {
        name: "sandwich_no_norm",
        clause: "sandwich",
        why: "the post norm is skipped: the stream takes the sub-layer's output as it is, a plain add",
        edits: &[("+ args.add * value;", "+ norm_rows_load4(xr, f4);", 1)],
    },
    Mutant {
        name: "sandwich_no_gain",
        clause: "sandwich",
        why: "the norm's gain is not applied to the value",
        edits: &[("return (x * gain) * scale;", "return x * scale;", 1)],
    },
    Mutant {
        name: "sandwich_add_ignored",
        clause: "sandwich",
        why: "the add scale is ignored: the normed value is always added once",
        edits: &[("+ args.add * value;", "+ value;", 1)],
    },
    Mutant {
        name: "sandwich_nan_stays",
        clause: "sandwich_nan",
        why: "a row the norm cannot take leaves the stream's row as it was, a finite value, not NaN",
        edits: &[(
            "norm_rows_raise(faults, args.norm.layer);\n        }\n        for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {\n            *((device float4 *)(yr + f4 * 4u)) = float4(NAN);",
            "norm_rows_raise(faults, args.norm.layer);\n        }\n        for (uint f4 = tid; f4 < f4s; f4 += NORM_ROWS_THREADS) {\n            *((device float4 *)(yr + f4 * 4u)) = norm_rows_load4(yr, f4);",
            1,
        )],
    },
    Mutant {
        name: "handon_xg_no_gain",
        clause: "handon_readers",
        why: "the join hands on the new stream without the reader's gain",
        edits: &[("= joined * norm_rows_load4(next_gain, f4);", "= joined;", 1)],
    },
    Mutant {
        name: "handon_scale_ulp",
        clause: "handon_readers",
        why: "the hand-on scale is a few roundings off the reader's own, as a sum in another order is",
        edits: &[(
            "rms_scale(sum_squares, (uint)nb * Q::values_per_block(), next.eps);",
            "rms_scale(sum_squares * 1.000001f, (uint)nb * Q::values_per_block(), next.eps);",
            1,
        )],
    },
    Mutant {
        name: "handon_seg_scale_ignored",
        clause: "handon_readers",
        why: "the scaled mat-vec rows skip the join's scale",
        edits: &[(
            "const float in_scale = SCALED ? gain[0] : 1.f;\n    // Each row's blocks",
            "const float in_scale = 1.f;\n    // Each row's blocks",
            1,
        )],
    },
    Mutant {
        name: "handon_pair_scale_ignored",
        clause: "handon_readers",
        why: "the scaled pair skips the join's scale",
        edits: &[(
            "const float in_scale = SCALED ? gain[0] : 1.f;\n    for (int ib",
            "const float in_scale = 1.f;\n    for (int ib",
            1,
        )],
    },
    Mutant {
        name: "handon_plain_no_gain",
        clause: "handon_readers",
        why: "the plain residual's hand-on writes the stream without the reader's gain",
        edits: &[(
            "= norm_rows_load4(x, f4) * norm_rows_load4(next_gain, f4);",
            "= norm_rows_load4(x, f4);",
            1,
        )],
    },
    Mutant {
        name: "handon_nan_scale_finite",
        clause: "handon_nan",
        why: "a next norm that cannot take the stream hands on its zero scale, a finite value",
        edits: &[("xn[COLS] = NAN;", "xn[COLS] = scale;", 1)],
    },
    Mutant {
        name: "handon_nan_xg_stays",
        clause: "handon_nan",
        why: "a join the norm cannot take leaves the hand-on values as they were",
        edits: &[(
            "if (xg != nullptr) {\n                *((device float4 *)(xg + f4 * 4u)) = float4(NAN);\n            }",
            "",
            1,
        )],
    },
    Mutant {
        name: "gemm_k_tail",
        clause: "gemm_q6k_store",
        why: "the tile loop's last K step is skipped, so a block pair of every row is missing",
        edits: &[(
            "for (int k0 = 0; k0 < cols; k0 += BK) {",
            "for (int k0 = 0; k0 + BK < cols; k0 += BK) {",
            1,
        )],
    },
    Mutant {
        name: "gemm_il_swap",
        clause: "gemm_q4k_store",
        why: "two 16-value halves of a Q4_K chunk are swapped in the dequant",
        edits: &[(
            "const uchar2 pair = as_type<uchar2>((ushort)(q[i] & mask));",
            "const uchar2 pair = as_type<uchar2>((ushort)(q[(i + 4) % 8] & mask));",
            1,
        )],
    },
    Mutant {
        name: "gemm_q4_mask",
        clause: "gemm_q4k_store",
        why: "a chunk's nibble side is the other one: the high-nibble chunks read the low nibbles and the reverse",
        edits: &[(
            "const ushort mask = (il & 2) == 0 ? 0x0F0F : 0xF0F0;",
            "const ushort mask = (il & 2) == 0 ? 0xF0F0 : 0x0F0F;",
            1,
        )],
    },
    Mutant {
        name: "gemm_q6_hi_shift",
        clause: "gemm_q6k_store",
        why: "the 6-bit codes take their two high bits from the wrong field of qh: the shift of the chunk's field is dropped",
        edits: &[(
            "const ushort hi = (ushort)((qh[i] >> shift) & 0x0303);",
            "const ushort hi = (ushort)((qh[i] >> 0) & 0x0303);",
            1,
        )],
    },
    Mutant {
        name: "gemm_q6_offset",
        clause: "gemm_q6k_store",
        why: "the 6-bit codes are centred on 31, not 32, so every weight is off by one scale step",
        edits: &[(
            "const float offset = -32.f * dsc;",
            "const float offset = -31.f * dsc;",
            1,
        )],
    },
    Mutant {
        name: "gemm_row_edge",
        clause: "gemm_a4_store",
        why: "the bounded store ignores the weight-column bound, so a tail tile's dead columns are written",
        edits: &[("if (n < nr0) {", "if (n < GEMM_ROWS) {", 1)],
    },
    Mutant {
        name: "gemm_resid_overwrite",
        clause: "gemm_a4_resid",
        why: "the residual epilogue stores the product over the output's even words instead of adding to them",
        edits: &[(
            "p[n] = RESID ? p[n] + c[i][j].x : c[i][j].x;",
            "p[n] = c[i][j].x;",
            1,
        )],
    },
    Mutant {
        name: "gemm_seg_weights",
        clause: "gemm_seg",
        why: "a segment of the segmented GEMM reads the weights of the first segment",
        edits: &[("W + wblocks,", "W,", 1)],
    },
    Mutant {
        name: "gemm_a4_chunk_order",
        clause: "gemm_a4_store",
        why: "an A4 chunk stages each code word's second value in its first value's place",
        edits: &[(
            "reg[h][0] = fma((float)(word & 0x000Fu), s0, b);",
            "reg[h][0] = fma((float)(word & 0x00F0u), s1, b);",
            1,
        )],
    },
    Mutant {
        name: "gemm_a4_nibble_scale",
        clause: "gemm_a4_store",
        why: "an A4 chunk's third nibble is scaled as if it sat one place lower, so those values are 16 times too large",
        edits: &[(
            "const float s2 = s0 * (1.f / 256.f);",
            "const float s2 = s0 * (1.f / 16.f);",
            1,
        )],
    },
    Mutant {
        name: "gemm_weight_edge",
        clause: "gemm_a4_store",
        why: "the weight rows past a tile's live rows are staged unclamped: a tile of 16 live rows reads 48 rows past its weights, which no output shows and only shader validation reports",
        edits: &[(
            "const short row = min((short)(c0 / CHUNKS), (short)(nr0 - 1));",
            "const short row = (short)(c0 / CHUNKS);",
            1,
        )],
    },
    Mutant {
        name: "gemm_act_edge",
        clause: "gemm_q4k_store",
        why: "the activation rows past the last read past their window",
        edits: &[(
            "const short row = min((short)(c0 / CHUNKS), (short)(nr1 - 1));",
            "const short row = (short)(c0 / CHUNKS);",
            1,
        )],
    },
    Mutant {
        name: "gemm_frag_swap",
        clause: "gemm_q6k_store",
        why: "a B fragment's two weight-row values swap places, so every product pairs w[r+1] with x's row r",
        edits: &[(
            "bh[j] = vec<half, 2>(brow[j][kk], (brow[j] + LDA)[kk]);",
            "bh[j] = vec<half, 2>((brow[j] + LDA)[kk], brow[j][kk]);",
            1,
        )],
    },
    Mutant {
        name: "gemm_store_interleave",
        clause: "gemm_q4k_store",
        why: "the bounded store drops its SIMD group's across offset, so half the tile's columns land 8 places off and past the window",
        edits: &[(
            "const short n = gn + 16 * j + fn;",
            "const short n = 16 * j + fn;",
            1,
        )],
    },
    Mutant {
        name: "gemm_grid_order",
        clause: "gemm_q4k_store",
        why: "the plain GEMM reads its weight tile from the grid's y and its activation tile from x, the launch's other order, so tiles are missed or read past the output",
        edits: &[(
            "(int)tgpig.x * GEMM_ROWS, (int)tgpig.y * GEMM_ACTS,",
            "(int)tgpig.y * GEMM_ROWS, (int)tgpig.x * GEMM_ACTS,",
            1,
        )],
    },
    Mutant {
        name: "gemm_seg_grid_order",
        clause: "gemm_seg",
        why: "the segmented GEMM picks its segment from the grid's y, the activation tiles' axis",
        edits: &[(
            "if (tgpig.x >= args.segs[i - 1u].first_group) {",
            "if (tgpig.y >= args.segs[i - 1u].first_group) {",
            1,
        )],
    },
    Mutant {
        name: "gemm_h16_stage",
        clause: "gemm_half_rows",
        why: "a half row's chunk stages its four runs pairwise swapped",
        edits: &[(
            "        dst[g] = *(device const half4 *)(xr + 4 * g);",
            "        dst[g] = *(device const half4 *)(xr + 4 * (g ^ 1));",
            1,
        )],
    },
    Mutant {
        name: "gemm_cast_stride",
        clause: "gemm_half_rows",
        why: "the cast reads the f32 rows at their width, not at the stride they are kept at",
        edits: &[(
            "device const float * xr = x + (ulong)tgpig.y * args.x_stride + c;",
            "device const float * xr = x + (ulong)tgpig.y * args.cols + c;",
            1,
        )],
    },
    Mutant {
        name: "gemm_cast_unchecked",
        clause: "gemm_half_rows",
        why: "the cast skips the staging's range check, so an activation past the half range casts to inf",
        edits: &[(
            "+ c) = gemm_half4(v, faults, args.layer);",
            "+ c) = half4(v);",
            1,
        )],
    },
    Mutant {
        name: "gemm_range_off",
        clause: "gemm_range",
        why: "the f16 operand's range check is gone, so an activation past the half range casts to inf",
        edits: &[(
            "const bool4 bad = !(fabs(v) <= 65504.f);",
            "const bool4 bad = bool4(false);",
            1,
        )],
    },
    Mutant {
        name: "route_rows_stride",
        clause: "route_rows_select",
        why: "a row's logits start a router's 257 rows on, ignoring the stride the pass keeps them at",
        edits: &[(
            "device const float *lr = logits + (ulong)row * args.stride;",
            "device const float *lr = logits + (ulong)row * ROUTER_ROWS;",
            1,
        )],
    },
    Mutant {
        name: "route_rows_per_row",
        clause: "route_rows_select",
        why: "a row's ids and weights are written at the picks' stride, not the picks and the shared slot's",
        edits: &[(
            "const uint per_row = args.top_k + BLOOMERY_ROUTER_SHARED;",
            "const uint per_row = args.top_k;",
            1,
        )],
    },
    Mutant {
        name: "route_table_duplicate",
        clause: "route_table",
        why: "every slot of a group takes the group's first place, so the later slots are lost",
        edits: &[(
            "start[g] + atomic_fetch_add_explicit(&count[g], 1u, memory_order_relaxed);",
            "start[g] + atomic_fetch_add_explicit(&count[g], 0u, memory_order_relaxed);",
            1,
        )],
    },
    Mutant {
        name: "route_table_tile_cut",
        clause: "route_table",
        why: "a group's last partial tile is not counted, so its slots are in no tile",
        edits: &[(
            "tile += (n + ROUTE_TILE - 1u) / ROUTE_TILE;",
            "tile += n / ROUTE_TILE;",
            1,
        )],
    },
    Mutant {
        name: "routed_expert_base",
        clause: "grouped_gemm",
        why: "a group reads the next expert's weights",
        edits: &[(
            "wg = W + (ulong)expert * args.stack_blocks;",
            "wg = W + (ulong)(expert + 1u) * args.stack_blocks;",
            1,
        )],
    },
    Mutant {
        name: "routed_shared_base",
        clause: "grouped_gemm",
        why: "the shared group reads the stack's first expert, not the shared tensors",
        edits: &[(
            "routed_run<QS, T, PAIR>(shmem, W_sh,",
            "routed_run<QS, T, PAIR>(shmem, (device const typename QS::Block *)W,",
            1,
        )],
    },
    Mutant {
        name: "routed_shared_group_refused",
        clause: "q8_0_grouped",
        why: "the group past the routed experts is refused as a tile the table cannot write, though the instance has a shared expert",
        edits: &[(
            "expert >= args.n_experts + BLOOMERY_ROUTER_SHARED",
            "expert >= args.n_experts",
            1,
        )],
    },
    Mutant {
        name: "routed_no_weight",
        clause: "grouped_gemm",
        why: "the pair's output is not scaled by the slot's route weight",
        edits: &[
            (
                "p[o] = pair_swiglu(c[i][0].x, c[i][1].x) * rw;",
                "p[o] = pair_swiglu(c[i][0].x, c[i][1].x);",
                1,
            ),
            (
                "p[o + 1] = pair_swiglu(c[i][0].y, c[i][1].y) * rw;",
                "p[o + 1] = pair_swiglu(c[i][0].y, c[i][1].y);",
                1,
            ),
        ],
    },
    Mutant {
        name: "routed_silu_up",
        clause: "grouped_gemm",
        why: "the pair applies silu to the up projection and multiplies by the gate",
        edits: &[
            (
                "p[o] = pair_swiglu(c[i][0].x, c[i][1].x) * rw;",
                "p[o] = pair_swiglu(c[i][1].x, c[i][0].x) * rw;",
                1,
            ),
            (
                "p[o + 1] = pair_swiglu(c[i][0].y, c[i][1].y) * rw;",
                "p[o + 1] = pair_swiglu(c[i][1].y, c[i][0].y) * rw;",
                1,
            ),
        ],
    },
    Mutant {
        name: "routed_no_gather",
        clause: "grouped_gemm",
        why: "the pair's activation rows are the places' own, not the slots' pass rows",
        edits: &[(
            "args.gather != 0u ? (ulong)(sorted[b_place] / BLOOMERY_EXPERT_SLOTS) : (ulong)b_place;",
            "(ulong)b_place;",
            1,
        )],
    },
    Mutant {
        name: "routed_dead_fragment",
        clause: "grouped_gemm",
        why: "a fragment row-block that holds live rows but fewer than a block is treated as dead",
        edits: &[("switch ((count + 7) / 8) {", "switch (count / 8) {", 1)],
    },
    Mutant {
        name: "routed_range_off",
        clause: "grouped_gemm",
        why: "the grouped GEMM's f16 range check is gone, so an activation past the half range casts to inf",
        edits: &[(
            "const bool4 bad = !(fabs(v) <= 65504.f);",
            "const bool4 bad = bool4(false);",
            1,
        )],
    },
    Mutant {
        name: "routed_upper_rows",
        clause: "grouped_oracle",
        why: "the tile's second half of activation rows multiplies the first half's",
        edits: &[(
            "arow[i] = sb + (8 * i + fm) * LDA + fn;",
            "arow[i] = sb + ((8 * i) % (ROUTED_ACTS / 2) + fm) * LDA + fn;",
            1,
        )],
    },
    Mutant {
        name: "routed_last_row_unstaged",
        clause: "grouped_oracle",
        why: "a run that fills its last fragment leaves that fragment's last row unstaged",
        edits: &[(
            "const bool b_stage = (short)(tiitg / 2) < 8 * NA;",
            "const bool b_stage = (short)(tiitg / 2) < 8 * NA - 1;",
            1,
        )],
    },
    Mutant {
        name: "combine_skip_shared",
        clause: "combine",
        why: "the combine sums the picks and leaves the shared expert's slot out",
        edits: &[(
            "for (uint s = 0u; s < args.per_row; ++s) {",
            "for (uint s = 0u; s + 1u < args.per_row; ++s) {",
            1,
        )],
    },
    Mutant {
        name: "headnorm_rows_head",
        clause: "headnorm_rows",
        why: "a head's scale is taken from the next head's partials",
        edits: &[(
            "head * BLOOMERY_DELTA_COLUMN_BLOCKS;",
            "((head + 1u) % BLOOMERY_DELTA_VALUE_HEADS) * BLOOMERY_DELTA_COLUMN_BLOCKS;",
            1,
        )],
    },
    Mutant {
        name: "headnorm_rows_no_raise",
        clause: "headnorm_rows",
        why: "a head the rows norm cannot take is NaN and raises nothing",
        edits: &[("raise(faults, args.layer, site);", "", 1)],
    },
    Mutant {
        name: "gemm_f32w_rotate",
        clause: "gemm_f32w",
        why: "an f32 weight block's sixteen values are staged rotated by one",
        edits: &[(
            "reg[i / 4][i % 4] = (T)blk.v[i];",
            "reg[i / 4][i % 4] = (T)blk.v[(i + 1) % 16];",
            1,
        )],
    },
    Mutant {
        name: "f32_seg_input_f16",
        clause: "f32_seg",
        why: "the F32 owner reads its first activation of a block through f16",
        edits: &[(
            "float a0 = block[base + 0];",
            "float a0 = (float)(half)block[base + 0];",
            1,
        )],
    },
    Mutant {
        name: "apart_pair_shared_blocks",
        clause: "apart_pair",
        why: "the shared slot of an Apart pair walks its rows in the stacks' block count",
        edits: &[(
            "const uint shared_blocks = args.nblocks * (uint)Q::values_per_block() / (uint)QS::values_per_block();",
            "const uint shared_blocks = args.nblocks;",
            1,
        )],
    },
    Mutant {
        name: "apart_fold_shared_hidden",
        clause: "apart_fold",
        why: "the shared slot of an Apart fold reads the first slot's hidden states",
        edits: &[(
            "device const float * hs = h + (ulong)BLOOMERY_EXPERT_SHARED * BLOOMERY_EXPERT_K;",
            "device const float * hs = h;",
            1,
        )],
    },
    Mutant {
        name: "apart_grouped_shared_group",
        clause: "apart_grouped",
        why: "the shared group's tiles of an Apart grouped GEMM read the stacks' type past the stack",
        edits: &[(
            "const bool shared = BLOOMERY_ROUTER_SHARED != 0 && expert == args.n_experts;",
            "const bool shared = false;",
            1,
        )],
    },
    Mutant {
        name: "q8_0_lane_quarter",
        clause: "q8_0_matvec",
        why: "a Q8_0 lane dots its activations with the other quarter of the block's codes",
        edits: &[(
            "device const int8_t * qs = blk.qs + 8 * lane.il;",
            "device const int8_t * qs = blk.qs + 8 * (lane.il ^ 1);",
            1,
        )],
    },
    Mutant {
        name: "q8_0_tail_store",
        clause: "q8_0_row_tail",
        why: "the store of a pair no longer stops at the last row",
        edits: &[(TAIL_STORE_FIND, TAIL_STORE_REPLACE, 1)],
    },
    Mutant {
        name: "q8_0_norm_gain",
        clause: "q8_0_norm",
        why: "the Q8_0 norm load drops the gain of a value",
        edits: &[("a0 *= g8.x;", "a0 *= 1.f;", 1)],
    },
    Mutant {
        name: "q8_0_seg_weights",
        clause: "q8_0_seg",
        why: "a segment reads the weights of the first segment",
        edits: &[("W + (ulong)args.segs[s].wblocks,", "W,", 1)],
    },
    Mutant {
        name: "q8_0_head_scale",
        clause: "q8_0_headnorm",
        why: "a Q8_0 block takes the neighbouring head's scale",
        edits: &[(
            "const int head = ib / (BLOOMERY_DELTA_HEAD_COLS / 32);",
            "const int head = (ib / (BLOOMERY_DELTA_HEAD_COLS / 32)) ^ 1;",
            1,
        )],
    },
    Mutant {
        name: "q8_0_sum_sq",
        clause: "q8_0_pair",
        why: "the Q8_0 norm load's sum of squares drops a value",
        edits: &[(
            "*sum_sq += (a0 * a0 + a1 * a1) + (a2 * a2 + a3 * a3);\n                a0 *= g8.x;",
            "*sum_sq += (a0 * a0 + a1 * a1) + a2 * a2;\n                a0 *= g8.x;",
            1,
        )],
    },
    Mutant {
        name: "q8_0_fold_slot",
        clause: "q8_0_fold",
        why: "the Q8_0 fold maps every eight blocks, not every sixteen, to a slot",
        edits: &[(
            "        return isfinite(sum);\n    }\n\n    static inline int fold_blocks() {\n        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());\n    }\n\n    static inline int fold_slot(int ib) {\n        return ib / (BLOOMERY_EXPERT_K / values_per_block());",
            "        return isfinite(sum);\n    }\n\n    static inline int fold_blocks() {\n        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());\n    }\n\n    static inline int fold_slot(int ib) {\n        return ib / 8;",
            1,
        )],
    },
    Mutant {
        name: "q8_0_dot_short",
        clause: "q8_0_head",
        why: "the Q8_0 code dot stops one value short of the lane's eight",
        edits: &[(
            "FOR_UNROLL (short i = 0; i < 8; ++i) {\n            sumq += qs[i] * lane.yl[i];",
            "FOR_UNROLL (short i = 0; i < 7; ++i) {\n            sumq += qs[i] * lane.yl[i];",
            1,
        )],
    },
    Mutant {
        name: "q8_0_value_unsigned",
        clause: "q8_0_embed",
        why: "the Q8_0 value reads its code as unsigned",
        edits: &[(
            "return (float)blk.qs[j] * (float)blk.d;",
            "return (float)(uint8_t)blk.qs[j] * (float)blk.d;",
            1,
        )],
    },
    Mutant {
        name: "q8_0_chunk_half",
        clause: "q8_0_gemm",
        why: "a GEMM chunk stages the other half of its Q8_0 block",
        edits: &[(
            "device const int8_t * q = blk.qs + 16 * il;",
            "device const int8_t * q = blk.qs + 16 * (il ^ 1);",
            1,
        )],
    },
    Mutant {
        name: "q8_0_gemm_seg_out",
        clause: "q8_0_gemm_seg",
        why: "every GEMM segment stores at the first segment's output offset",
        edits: &[(
            "                           x,\n                           y + args.segs[s].out,",
            "                           x,\n                           y,",
            1,
        )],
    },
    Mutant {
        name: "q8_0_stage_unsigned",
        clause: "q8_0_grouped",
        why: "the GEMM staging reads the Q8_0 codes as unsigned",
        edits: &[(
            "reg[j / 4][j % 4] = (float)q[j] * d;",
            "reg[j / 4][j % 4] = (float)(uint8_t)q[j] * d;",
            1,
        )],
    },
    Mutant {
        name: "q5k_high_bit",
        clause: "q5k_matvec",
        why: "the second group's fifth bit is read two bits up, not one",
        edits: &[(
            "const uint8_t hm2 = hm1 << 1;",
            "const uint8_t hm2 = hm1 << 2;",
            1,
        )],
    },
    Mutant {
        name: "q5k_tail_store",
        clause: "q5k_row_tail",
        why: "the store of a pair no longer stops at the last row",
        edits: &[(TAIL_STORE_FIND, TAIL_STORE_REPLACE, 1)],
    },
    Mutant {
        name: "q5k_plain_load",
        clause: "q5k_norm",
        why: "the Q5_K load takes the plain form: no gain, no sum of squares",
        edits: &[(
            "Q4K::load<LOAD>(lane, block, gain, scale, sum_sq);",
            "Q4K::load<LOAD_PLAIN>(lane, block, gain, scale, sum_sq);",
            1,
        )],
    },
    Mutant {
        name: "q5k_seg_weights",
        clause: "q5k_seg",
        why: "a segment reads the weights of the first segment",
        edits: &[("W + (ulong)args.segs[s].wblocks,", "W,", 1)],
    },
    Mutant {
        name: "q5k_head_halves",
        clause: "q5k_headnorm",
        why: "the first half of a Q5_K block takes the second head's scale",
        edits: &[("return lo * (d * (sc8[0]", "return hi * (d * (sc8[0]", 1)],
    },
    Mutant {
        name: "q5k_dmin_sign",
        clause: "q5k_pair",
        why: "the Q5_K dmin correction is added, not subtracted",
        edits: &[("- dmin * (lane.sumy[0]", "+ dmin * (lane.sumy[0]", 1)],
    },
    Mutant {
        name: "q5k_fold_slot",
        clause: "q5k_fold",
        why: "the Q5_K fold maps every four blocks, not every two, to a slot",
        edits: &[(
            "        return Q4K::sums_finite(lane);\n    }\n\n    static inline int fold_blocks() {\n        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());\n    }\n\n    static inline int fold_slot(int ib) {\n        return ib / (BLOOMERY_EXPERT_K / values_per_block());",
            "        return Q4K::sums_finite(lane);\n    }\n\n    static inline int fold_blocks() {\n        return BLOOMERY_EXPERT_SLOTS * (BLOOMERY_EXPERT_K / values_per_block());\n    }\n\n    static inline int fold_slot(int ib) {\n        return ib / 4;",
            1,
        )],
    },
    Mutant {
        name: "q5k_nibble_group",
        clause: "q5k_head",
        why: "the third group's nibbles are read from the first group's bytes",
        edits: &[(
            "acc1[2] += lane.yh[l + 0] * (q2[l] & 0x0F);",
            "acc1[2] += lane.yh[l + 0] * (q1[l] & 0x0F);",
            1,
        )],
    },
    Mutant {
        name: "q5k_fifth_bit",
        clause: "q5k_embed",
        why: "a value in a group's second half takes the first half's fifth bit",
        edits: &[(
            "const uint fifth = (blk.qh[l] >> (2 * group + high)) & 1;",
            "const uint fifth = (blk.qh[l] >> (2 * group)) & 1;",
            1,
        )],
    },
    Mutant {
        name: "q5k_chunk_group",
        clause: "q5k_gemm",
        why: "a GEMM chunk takes the scale and min of the group two chunks on",
        edits: &[(
            "const float2 sm = group_scale(blk, il >> 1);",
            "const float2 sm = group_scale(blk, il >> 2);",
            1,
        )],
    },
    Mutant {
        name: "q5k_gemm_seg_out",
        clause: "q5k_gemm_seg",
        why: "every GEMM segment stores at the first segment's output offset",
        edits: &[(
            "                           x,\n                           y + args.segs[s].out,",
            "                           x,\n                           y,",
            1,
        )],
    },
    Mutant {
        name: "q5k_scale_byte",
        clause: "q5k_grouped",
        why: "the second half's groups take the mins' bytes as their scales",
        edits: &[(
            "const short at = (s & 1) + 4 * (s >> 2);",
            "const short at = (s & 1) + 2 * (s >> 2);",
            1,
        )],
    },
    Mutant {
        name: "attn_tile_dead_unmasked",
        clause: "attn_f16",
        why: "a key past the segment's end scores its zero row as 0 and weighs in",
        edits: &[
            (
                "float scored = FULL || lo + t < hi ? score : -INFINITY;",
                "float scored = score;",
                1,
            ),
            (
                "float weight = lo + lane < hi ? exp(mine - new_m) : 0.0f;",
                "float weight = exp(mine - new_m);",
                1,
            ),
        ],
    },
    Mutant {
        name: "attn_prefetch_stale",
        clause: "attn_f16",
        why: "every tile reads the cache's first tile's rows, so every tile after the first is the first's rows again",
        edits: &[("uint key = lo + t0 + c;", "uint key = t0 + c;", 2)],
    },
    Mutant {
        name: "attn_tile_score_lane",
        clause: "attn_f16",
        why: "lane t keeps the score of the tile's key t - 1, so every key weighs at its neighbour's score",
        edits: &[(
            "mine = lane == t ? scored : mine;",
            "mine = lane == t + 1u ? scored : mine;",
            1,
        )],
    },
    Mutant {
        name: "attn_tile_weight_shuffle",
        clause: "attn_f16",
        why: "every value of a run of rows in flight takes the run's first key's weight",
        edits: &[(
            "float w = simd_shuffle(weight, ushort(t0 + c));",
            "float w = simd_shuffle(weight, ushort(t0));",
            1,
        )],
    },
    Mutant {
        name: "attn_fold_stage_l",
        clause: "attn_f16",
        why: "the fold reads a segment's first value in place of its weight sum",
        edits: &[(
            "lj = ticket_load(partials, base + 1u);",
            "lj = ticket_load(partials, base + 2u);",
            1,
        )],
    },
    Mutant {
        name: "attn_fold_empty_unskipped",
        clause: "attn_f16",
        why: "a fold slot past the head's segments is not left empty: it folds the next head's first segment as one of its own",
        edits: &[(
            "if (slot < args.segments) {",
            "if (slot < args.segments + 1u) {",
            1,
        )],
    },
    Mutant {
        name: "attn_fold_underflow_skipped",
        clause: "attn_fold_nan",
        why: "the fold skips a segment whose f32 factor is 0, so the NaN values of a live segment whose weight underflows never reach the output",
        edits: &[(
            "if (simd_shuffle(ls[r], ushort(c)) == 0.0f) {",
            "if (f == 0.0f) {",
            1,
        )],
    },
    Mutant {
        name: "attn_one_dead_unmasked",
        clause: "attn_f16",
        why: "a chunk slot past the live keys in the one-pass flash scores its zero row as 0 and weighs in",
        edits: &[
            (
                "s[c] = base + c * BLOOMERY_ATTN_ONE_SG < max_live ? score : -INFINITY;",
                "s[c] = score;",
                1,
            ),
            (
                "float weight = base + c * BLOOMERY_ATTN_ONE_SG < max_live ? exp(s[c] - new_m) : 0.0f;",
                "float weight = exp(s[c] - new_m);",
                1,
            ),
        ],
    },
    Mutant {
        name: "attn_one_chunk_rescale",
        clause: "attn_f16",
        why: "a SIMD group of the one-pass flash never rescales its sums when a later chunk raises its maximum",
        edits: &[(
            "float factor = m > -INFINITY ? exp(m - new_m) : 0.0f;",
            "float factor = 1.0f;",
            1,
        )],
    },
    Mutant {
        name: "attn_one_merge_unscaled",
        clause: "attn_f16",
        why: "the one-pass merge adds each SIMD group's sums at its own maximum, not rescaled to the largest",
        edits: &[("float f = exp(pm[g] - big);", "float f = 1.0f;", 1)],
    },
    Mutant {
        name: "attn_one_store_owner",
        clause: "attn_f16",
        why: "only the first query head of the first key head stores the appended rows: the other key heads' rows are never written",
        edits: &[(
            "bool stores = local && head % BLOOMERY_ATTN_GQA == 0u;",
            "bool stores = local && head == 0u;",
            1,
        )],
    },
    Mutant {
        name: "attn_one_local_key_swap",
        clause: "attn_f16",
        why: "the one-pass flash scores this position's key against the value's halves it holds in threadgroup memory",
        edits: &[("a = kp[lane];", "a = vp[lane];", 1)],
    },
];

/// The tensor-op GEMMs' defects, at least one for each `tensor_*` clause.
pub const TENSOR_MUTANTS: &[Mutant] = &[
    Mutant {
        name: "tensor_k_step_skipped",
        clause: "tensor_gemm_q4k",
        why: "the dense tile's K loop stops one step short: the last 32 columns are never multiplied",
        edits: &[(
            "    for (int k0 = 0; k0 < cols; k0 += BK) {",
            "    for (int k0 = 0; k0 + BK < cols; k0 += BK) {",
            1,
        )],
    },
    Mutant {
        name: "tensor_rows_offset",
        clause: "tensor_gemm_q6k",
        why: "the dense op's weight fragment starts one staged row late: every output takes the next weight row",
        edits: &[(
            "auto ta = tensor_rows<T, TENSOR_SM>(sb + tm * LDA);\n    auto tw = tensor_rows<T, TENSOR_SN>(sa + tn * LDA);",
            "auto ta = tensor_rows<T, TENSOR_SM>(sb + tm * LDA);\n    auto tw = tensor_rows<T, TENSOR_SN>(sa + (tn + 1) * LDA);",
            1,
        )],
    },
    Mutant {
        name: "tensor_resid_stores",
        clause: "tensor_gemm_a4",
        why: "the residual epilogue stores the product over the output instead of adding to it",
        edits: &[("*p = RESID ? *p + ct[i] : ct[i];", "*p = ct[i];", 1)],
    },
    Mutant {
        name: "tensor_seg_out",
        clause: "tensor_gemm_seg",
        why: "every segment stores at the first segment's output offset",
        edits: &[("y + args.segs[s].out", "y", 1)],
    },
    Mutant {
        name: "tensor_range_no_raise",
        clause: "tensor_gemm_range",
        why: "an activation past the half range casts as NaN but raises no quantizer site",
        edits: &[(
            "    if (any(bad)) {\n        raise(faults, layer, BLOOMERY_FAULT_SITE_QUANT_COLUMN);",
            "    if (any(bad)) {",
            1,
        )],
    },
    Mutant {
        name: "tensor_pair_swapped",
        clause: "tensor_grouped_q4k",
        why: "the pair's gate and up accumulators swap places in the SwiGLU",
        edits: &[("pair_swiglu(cg[i], cu[i])", "pair_swiglu(cu[i], cg[i])", 1)],
    },
    Mutant {
        name: "tensor_routed_short_block",
        clause: "tensor_grouped_q4k",
        why: "a SIMD group's second block of places runs only when the run passes it by more than 8: a run ending 1 to 8 places into the block leaves them unstored",
        edits: &[(
            "const bool live1 = tm + TENSOR_RM < count;",
            "const bool live1 = tm + TENSOR_RM + 8 < count;",
            1,
        )],
    },
    Mutant {
        name: "tensor_routed_block_rows",
        clause: "tensor_grouped_a4",
        why: "a SIMD group's second block of places reads the first block's staged rows",
        edits: &[(
            "auto ta1 = tensor_rows<T, TENSOR_RM>(sb + (tm + TENSOR_RM) * LDA);",
            "auto ta1 = tensor_rows<T, TENSOR_RM>(sb + tm * LDA);",
            1,
        )],
    },
    Mutant {
        name: "tensor_routed_k_step_skipped",
        clause: "tensor_grouped_q6k",
        why: "the routed tile's K loops (pair and plain) stop one step short",
        edits: &[(
            "        for (int k0 = 0; k0 < (int)args.cols; k0 += BK) {",
            "        for (int k0 = 0; k0 + BK < (int)args.cols; k0 += BK) {",
            2,
        )],
    },
    Mutant {
        name: "tensor_tile_no_raise",
        clause: "tensor_grouped_a4",
        why: "a tile the table cannot write stores nothing but raises no expert-id site",
        edits: &[(
            "            raise(faults, args.layer, BLOOMERY_FAULT_SITE_EXPERT_ID);",
            "",
            1,
        )],
    },
    Mutant {
        name: "tensor_q8_0_rotate",
        clause: "tensor_q8_0_gemm",
        why: "a Q8_0 chunk's values rotate inside each four as they are staged",
        edits: &[(
            "reg[j / 4][j % 4] = (float)q[j] * d;",
            "reg[j / 4][(j + 1) % 4] = (float)q[j] * d;",
            1,
        )],
    },
    Mutant {
        name: "tensor_q8_0_seg_out",
        clause: "tensor_q8_0_gemm_seg",
        why: "every segment stores at the first segment's output offset",
        edits: &[("y + args.segs[s].out", "y", 1)],
    },
    Mutant {
        name: "tensor_q8_0_chunk_half",
        clause: "tensor_q8_0_grouped",
        why: "a routed chunk stages the other half of its Q8_0 block",
        edits: &[(
            "device const int8_t * q = blk.qs + 16 * il;",
            "device const int8_t * q = blk.qs + 16 * (il ^ 1);",
            1,
        )],
    },
    Mutant {
        name: "tensor_q5k_min_sign",
        clause: "tensor_q5k_gemm",
        why: "the staged Q5_K value adds the group's min",
        edits: &[(
            "reg[j / 4][j % 4] = sm.x * (float)code(blk, (uint)(16 * il + j)) - sm.y;",
            "reg[j / 4][j % 4] = sm.x * (float)code(blk, (uint)(16 * il + j)) + sm.y;",
            1,
        )],
    },
    Mutant {
        name: "tensor_q5k_seg_out",
        clause: "tensor_q5k_gemm_seg",
        why: "every segment stores at the first segment's output offset",
        edits: &[("y + args.segs[s].out", "y", 1)],
    },
    Mutant {
        name: "tensor_q5k_scale_words",
        clause: "tensor_q5k_grouped",
        why: "a group's scale words come from the other half of the table",
        edits: &[(
            "blk.scales + ((s >> 1) & 1), sc16);",
            "blk.scales + ((s >> 2) & 1), sc16);",
            1,
        )],
    },
    Mutant {
        name: "tensor_routed_shared_group_refused",
        clause: "tensor_q8_0_grouped",
        why: "the group past the routed experts is refused as a tile the table cannot write, though the instance has a shared expert",
        edits: &[(
            "expert >= args.n_experts + BLOOMERY_ROUTER_SHARED",
            "expert >= args.n_experts",
            1,
        )],
    },
];
