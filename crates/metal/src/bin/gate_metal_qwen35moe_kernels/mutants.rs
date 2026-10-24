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
//! `matvec_refusals`, `delta_bounds`, `experts_bounds` and `fold_type` are
//! wholly host-side; `seg_bounds`, `router_bounds`, `head_bounds` and
//! `attn_bounds` also have a kernel-side contract, and the kernel mutants
//! below pin that side. The gate's refusal loop turns red on a deleted launch
//! check; a table that stayed green on one would be wrong.

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
        edits: &[("a0 *= gain[base + i +  0];", "a0 *= 1.f;", 1)],
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
        edits: &[("return shared_slot ? shexp : stack;", "return stack;", 1)],
    },
    Mutant {
        name: "pair_norm_gain",
        clause: "pair",
        why: "the pair's Q4_K norm drops the gain of a value",
        edits: &[("a0 *= gain[base + i +  0];", "a0 *= 1.f;", 1)],
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
            "acc[1] += lane.yl[4 * m + 1] * (q[m] & 0x00F0);",
            "acc[1] += lane.yl[4 * m + 2] * (q[m] & 0x00F0);",
            1,
        )],
    },
    Mutant {
        name: "a4_word_offset",
        clause: "a4_matvec",
        why: "an A4 lane reads the group's code words a quarter early",
        edits: &[(
            "device const uint16_t * q = (device const uint16_t *)blk.qs + 4 * lane.iq;",
            "device const uint16_t * q = (device const uint16_t *)blk.qs + 2 * lane.iq;",
            1,
        )],
    },
    Mutant {
        name: "a4_bias_sign",
        clause: "a4_matvec",
        why: "the A4 bias term is subtracted, not added",
        edits: &[(
            "+ bias_of(blk) * lane.sumy;",
            "- bias_of(blk) * lane.sumy;",
            1,
        )],
    },
    Mutant {
        name: "a4_scale_bias",
        clause: "a4_matvec",
        why: "the A4 code dot is scaled by the bias, not the scale",
        edits: &[(
            "return scale_of(blk) * ((acc[0] + 1.f/16.f * acc[1]) +",
            "return bias_of(blk) * ((acc[0] + 1.f/16.f * acc[1]) +",
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
        name: "attn_score_scale",
        clause: "attn_f16",
        why: "the score is twice the scaled dot",
        edits: &[(
            "float score = dot * args.scale;",
            "float score = dot * args.scale * 2.0f;",
            1,
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
        why: "one segment stores the attention without the gate",
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
        clause: "attn_f16",
        why: "one segment writes partials and leaves the output untouched",
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
        why: "every segment spans one tile, so a longer key count is only partly attended",
        edits: &[(
            "return per < BLOOMERY_ATTN_TILE ? BLOOMERY_ATTN_TILE : per;",
            "return BLOOMERY_ATTN_TILE;",
            1,
        )],
    },
    Mutant {
        name: "head_logit",
        clause: "head_logit",
        why: "the stored logit is doubled",
        edits: &[(
            "            if (!isfinite(logits[row])) {\n                raise(faults, args.layer, BLOOMERY_FAULT_SITE_LOGIT);\n            }",
            "            logits[row] *= 2.f;\n            if (!isfinite(logits[row])) {\n                raise(faults, args.layer, BLOOMERY_FAULT_SITE_LOGIT);\n            }",
            1,
        )],
    },
    Mutant {
        name: "head_id",
        clause: "head_id",
        why: "the threadgroup keeps the smaller logit",
        edits: &[(
            "v > best_v || (v == best_v && (uint)row < best_i)",
            "v < best_v || (v == best_v && (uint)row < best_i)",
            1,
        )],
    },
    Mutant {
        name: "head_ticket",
        clause: "head_ticket",
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
        name: "head_bounds",
        clause: "head_bounds",
        why: "the mat-vec does not run, so the logits stay the sentinel",
        edits: &[(
            "matvec_rows<Q, true, false, false, false>(weights, x, logits,",
            "if (false) matvec_rows<Q, true, false, false, false>(weights, x, logits,",
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
        clause: "head_ticket",
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
            "        float4 xv = norm_rows_load4(xr, f4);",
            "        float4 xv = norm_rows_load4(x, f4);",
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
            "ulong at = ((ulong)kv * args.ctx + key) * (BLOOMERY_ATTN_HEAD / 8u) + c;",
            "ulong at = ((ulong)(1u - kv) * args.ctx + key) * (BLOOMERY_ATTN_HEAD / 8u) + c;",
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
            "*((device float4 *)(yr + f4 * 4u)) = float4(NAN);",
            "*((device float4 *)(yr + f4 * 4u)) = float4(0.f);",
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
            "const uint per_row = args.top_k + 1u;",
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
            ": W + (ulong)expert * args.stack_blocks;",
            ": W + (ulong)(expert + 1u) * args.stack_blocks;",
            1,
        )],
    },
    Mutant {
        name: "routed_shared_base",
        clause: "grouped_gemm",
        why: "the shared group reads the stack's first expert, not the shared tensors",
        edits: &[("shared ? W_sh : W +", "shared ? W : W +", 1)],
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
            "const bool4 over = !(fabs(v) <= 65504.f);",
            "const bool4 over = bool4(false);",
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
