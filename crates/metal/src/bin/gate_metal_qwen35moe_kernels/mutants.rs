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
        edits: &[("1.f/16.f) -\n", "1.f/16.f) +\n", 3)],
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
        edits: &[(
            "raise(faults, args.layer, BLOOMERY_FAULT_SITE_ROUTER);",
            "",
            1,
        )],
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
            "y[(ulong)slot * stride + row] = NAN;",
            "y[(ulong)slot * stride + row] = 0.0f;",
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
        edits: &[("a0 *= gain[base + i + 0];", "a0 *= 1.f;", 1)],
    },
    Mutant {
        name: "pair_norm_no_raise",
        clause: "pair_norm",
        why: "a pair norm that cannot take its input writes NaN rows and raises nothing",
        edits: &[(
            "pair_write_bad(y, slot, args.stride, (uint)row0, faults, args.layer,\n                           BLOOMERY_FAULT_SITE_NORM_QUANT);",
            "y[(ulong)slot * args.stride + (uint)row0] = NAN;",
            2,
        )],
    },
    Mutant {
        name: "fold_drop_weight",
        clause: "fold",
        why: "the fold loads the activation without the route weight",
        edits: &[("float scale = w[ib >> 1];", "float scale = 1.0f;", 2)],
    },
    Mutant {
        name: "fold_no_resid",
        clause: "fold",
        why: "the fold overwrites the residual",
        edits: &[
            ("y[(uint)row0] += d0;", "y[(uint)row0] = d0;", 2),
            ("y[(uint)row1] += d1;", "y[(uint)row1] = d1;", 2),
        ],
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
        edits: &[(
            "state[base_to + key] = sp[i];",
            "state[base_to + key] = s0[i];",
            1,
        )],
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
            "partials[h * BLOOMERY_DELTA_COLUMN_BLOCKS + block] = p;",
            "partials[h * BLOOMERY_DELTA_COLUMN_BLOCKS + block] = p + p;",
            1,
        )],
    },
    Mutant {
        name: "delta_nan_zero",
        clause: "delta_nan",
        why: "a non-finite lane stores zero",
        edits: &[(
            "state[base_to + key] = NAN;",
            "state[base_to + key] = 0.0f;",
            1,
        )],
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
        why: "the merge stores the attention without the gate",
        edits: &[(
            "float merged = attn * attn_sigmoid(gate);",
            "float merged = attn;",
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
            "q6k_rows<true, false>(weights, x, logits,",
            "if (false) q6k_rows<true, false>(weights, x, logits,",
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
