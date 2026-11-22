//! The Inkling 2 kernels' defects, at least one for each clause of the
//! family's kernels: each edit is the one change that makes its clause
//! red (`mutants.rs` holds the gate's other tables and walks this one).

use super::Mutant;

/// The family's kernel defects.
pub const INK2_MUTANTS: &[Mutant] = &[
    Mutant {
        name: "res_mix_unnormed_sum",
        clause: "res_mix",
        why: "the weighted sum takes the raw vectors, not the normed ones",
        edits: &[(
            "const float normed = src[j] * shared.rstd[v];",
            "const float normed = src[j];",
            1,
        )],
    },
    Mutant {
        name: "res_mix_score_unscaled_by_rstd",
        clause: "res_mix",
        why: "a vector's score skips its RMS scale",
        edits: &[(
            "shared.rstd[v] * shared.prob[q][v] * args.scale;",
            "shared.prob[q][v] * args.scale;",
            1,
        )],
    },
    Mutant {
        name: "res_mix_stride_is_row",
        clause: "res_mix",
        why: "a bank's rows sit a row apart, the stride's padding read as data",
        edits: &[(
            "v < args.n_banks ? banks + v * args.stride + row : cur + row;",
            "v < args.n_banks ? banks + v * (args.m * n) + row : cur + row;",
            2,
        )],
    },
    Mutant {
        name: "res_mix_nan_unraised",
        clause: "res_mix_nan",
        why: "an unusable vector raises nothing",
        edits: &[(
            "usable = usable && rms_usable(shared.rstd[v], scale);",
            "usable = true;",
            1,
        )],
    },
    Mutant {
        name: "conv_res_taps_reversed",
        clause: "conv_res_rows",
        why: "the filter's taps read the window in the opposite order",
        edits: &[(
            "conv_res_tap(x, ring, d, args.ch, args.base, args.rows, t, args.pos, reach - k, c),",
            "conv_res_tap(x, ring, d, args.ch, args.base, args.rows, t, args.pos, k - 1u, c),",
            1,
        )],
    },
    Mutant {
        name: "conv_res_scale_dropped",
        clause: "conv_res_rows",
        why: "the residual sum skips the 1/sqrt scale",
        edits: &[(
            "const float out = (x[t * d + c] + acc) * args.scale;",
            "const float out = (x[t * d + c] + acc);",
            1,
        )],
    },
    Mutant {
        name: "conv_res_ring_slot_shifted",
        clause: "conv_res_rows",
        why: "the commit stores a position's input one slot late",
        edits: &[(
            "ring[sc_slot(args.pos + t, args.rows) * args.ch + args.base + c] = x[t * d + c];",
            "ring[sc_slot(args.pos + t + 1u, args.rows) * args.ch + args.base + c] = x[t * d + c];",
            1,
        )],
    },
    Mutant {
        name: "conv_res_nan_unraised",
        clause: "conv_res_nan",
        why: "a non-finite output raises nothing",
        edits: &[(
            "ok = ok && isfinite(out);\n        y[t * d + c] = isfinite(out)",
            "ok = true;\n        y[t * d + c] = isfinite(out)",
            1,
        )],
    },
    Mutant {
        name: "conv_res_start_reads_ring",
        clause: "conv_res_start",
        why: "a call at position 0 reads the ring slot before the start",
        edits: &[(
            "if (before > pos0) {\n        return 0.0f;\n    }\n    return ring[sc_slot(pos0 - before, rows) * ch",
            "if (before > pos0 + 1u) {\n        return 0.0f;\n    }\n    return ring[sc_slot(pos0 - before, rows) * ch",
            1,
        )],
    },
    Mutant {
        name: "qk_clip_factor_upscales",
        clause: "qk_clip_rows",
        why: "a row under its threshold is scaled up to it",
        edits: &[(
            "clamp(threshold / (root + 1e-6f), 0.0f, 1.0f)",
            "min(threshold / (root + 1e-6f), 4.0f)",
            1,
        )],
    },
    Mutant {
        name: "qk_clip_key_gained",
        clause: "qk_clip_rows",
        why: "the key rows take the query's per-channel gain",
        edits: &[(
            "if (is_q) {\n                x[i] *= gq[lane + 32u * i];",
            "if (true) {\n                x[i] *= gq[lane + 32u * i];",
            1,
        )],
    },
    Mutant {
        name: "qk_clip_tau_on_key",
        clause: "qk_clip_rows",
        why: "the position factor scales the key rows as it does the query's",
        edits: &[(
            "rowbuf[base + lane + 32u * i] = is_q ? x[i] * f * args.tau : x[i] * f;",
            "rowbuf[base + lane + 32u * i] = x[i] * f * args.tau;",
            1,
        )],
    },
    Mutant {
        name: "qk_clip_value_unnormed",
        clause: "qk_clip_rows",
        why: "the value rows are cached without their RMS norm",
        edits: &[("x[i] *= scale;", "x[i] *= 1.0f;", 1)],
    },
    Mutant {
        name: "qk_clip_key_pos_unchecked",
        clause: "qk_clip_faults",
        why: "a key append past the context is not refused",
        edits: &[("if (!is_q && args.pos >= args.ctx) {", "if (false) {", 1)],
    },
    Mutant {
        name: "qk_clip_value_pos_unchecked",
        clause: "qk_clip_faults",
        why: "a value append past the context is not refused",
        edits: &[("if (args.pos >= args.ctx) {", "if (false) {", 1)],
    },
    Mutant {
        name: "qk_clip_r_gain_dropped",
        clause: "qk_clip_rows",
        why: "the relative rows skip their gain",
        edits: &[(
            "float x = mine ? r[base + lane] * gr[lane] : 0.0f;",
            "float x = mine ? r[base + lane] : 0.0f;",
            1,
        )],
    },
    Mutant {
        name: "loop_control_msg_unnormed",
        clause: "loop_control",
        why: "the message half of the staged row skips its norm",
        edits: &[(
            "staged[n + j] = staged[n + j] * rstd[1];",
            "staged[n + j] = staged[n + j];",
            1,
        )],
    },
    Mutant {
        name: "loop_control_forget_sign",
        clause: "loop_control",
        why: "the forget gate adds to the stream instead of removing from it",
        edits: &[(
            "(1.0f - tanh(f)) * cur[row + j]",
            "(1.0f + tanh(f)) * cur[row + j]",
            1,
        )],
    },
    Mutant {
        name: "loop_control_down_cur_only",
        clause: "loop_control",
        why: "the down projection reads the stream half of the staged row alone",
        edits: &[(
            "for (uint k = lane; k < 2u * n; k += 32u) {",
            "for (uint k = lane; k < n; k += 32u) {",
            1,
        )],
    },
    Mutant {
        name: "loop_control_nan_row_kept",
        clause: "loop_control_nan",
        why: "an unusable row is passed through as it came",
        edits: &[(
            "            y[row + j] = NAN;",
            "            y[row + j] = cur[row + j];",
            1,
        )],
    },
    Mutant {
        name: "out_gate_sigmoid_flipped",
        clause: "out_gate",
        why: "the gate is the sigmoid of the negated channel",
        edits: &[(
            "const float gate = 1.0f / (1.0f + exp(-g[i]));",
            "const float gate = 1.0f / (1.0f + exp(g[i]));",
            1,
        )],
    },
    Mutant {
        name: "out_gate_nan_survives",
        clause: "out_gate_nan",
        why: "a non-finite product is stored as it came",
        edits: &[("y[i] = isfinite(out) ? out : NAN;", "y[i] = out;", 1)],
    },
    Mutant {
        name: "nvfp4_nibbles_swapped",
        clause: "nvfp4_rows",
        why: "a code byte's nibbles are read high first",
        edits: &[
            (
                "const uint c0 = blk.qs[lane] & 0x0fu;",
                "const uint c0 = blk.qs[lane] >> 4u;",
                1,
            ),
            (
                "const uint c1 = blk.qs[lane] >> 4u;",
                "const uint c1 = blk.qs[lane] & 0x0fu;",
                1,
            ),
        ],
    },
    Mutant {
        name: "nvfp4_scale_full",
        clause: "nvfp4_rows",
        why: "the UE4M3 scale is not halved",
        edits: &[("return raw * 0.5f;", "return raw;", 1)],
    },
    Mutant {
        name: "nvfp4_global_scale_dropped",
        clause: "nvfp4_rows",
        why: "the expert's global scale is not applied",
        edits: &[(
            "const float dot = simd_sum(acc) * args.w_scale;",
            "const float dot = simd_sum(acc);",
            2,
        )],
    },
    Mutant {
        name: "nvfp4_rows_group_bytes",
        clause: "nvfp4_rows",
        why: "a row-major group is eight bytes, not nine",
        edits: &[(
            "(ulong)r * (ulong)groups * 9ul",
            "(ulong)r * (ulong)groups * 8ul",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_negative_codes_positive",
        clause: "nvfp4_rows",
        why: "the negative half of the code table is positive",
        edits: &[(
            "0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f,",
            "0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f,",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_bad_scale_passes",
        clause: "nvfp4_nan",
        why: "a scale byte that is not a number is used as a number",
        edits: &[("usable = usable && isfinite(s);", "usable = usable;", 2)],
    },
    Mutant {
        name: "router_sigmoid_scale_dropped",
        clause: "router_sigmoid",
        why: "the route scale is not applied",
        edits: &[(
            "weights[s] = chosen[s] / divisor * args.scale;",
            "weights[s] = chosen[s] / divisor;",
            1,
        )],
    },
    Mutant {
        name: "router_sigmoid_bias_ignored",
        clause: "router_sigmoid",
        why: "the bias does not steer the choice",
        edits: &[(
            "tg_keys[row] = argmax_key(sel, row);",
            "tg_keys[row] = argmax_key(score, row);",
            1,
        )],
    },
    Mutant {
        name: "router_sigmoid_key_half_compare",
        clause: "router_sigmoid",
        why: "the round's low half is taken over the lanes whose low half equals the high maximum",
        edits: &[(
            "uint lo = simd_max(uint(local >> 32) == hi ? uint(local) : 0u);",
            "uint lo = simd_max(uint(local) == hi ? uint(local) : 0u);",
            1,
        )],
    },
    Mutant {
        name: "router_sigmoid_nan_unflagged",
        clause: "router_sigmoid_nan",
        why: "a score or selection value that is not finite is not flagged",
        edits: &[(
            "mine_bad |= isfinite(score) && isfinite(sel) ? 0u : 1u;",
            "mine_bad |= 0u;",
            1,
        )],
    },
    Mutant {
        name: "router_sigmoid_linear",
        clause: "router_sigmoid",
        why: "the score is the logit, not its sigmoid",
        edits: &[(
            "const float score = 1.0f / (1.0f + exp(-simd_sum(acc)));",
            "const float score = simd_sum(acc);",
            1,
        )],
    },
    Mutant {
        name: "situ_up_uncapped",
        clause: "situ",
        why: "the up term is not capped by its tanh",
        edits: &[(
            "const float u = value_beta * tanh(up / value_beta);",
            "const float u = up;",
            1,
        )],
    },
    Mutant {
        name: "situ_gate_cap_unscaled",
        clause: "situ",
        why: "the gate term's tanh takes the raw gate",
        edits: &[(
            "gate_beta * tanh(gate / gate_beta)",
            "gate_beta * tanh(gate)",
            1,
        )],
    },
    Mutant {
        name: "situ_nan_unraised",
        clause: "situ_nan",
        why: "a non-finite activation raises nothing",
        edits: &[("if (!betas || !isfinite(h)) {", "if (!betas) {", 1)],
    },
    Mutant {
        name: "head_f32_gain_dropped",
        clause: "head_f32",
        why: "the output norm's gain is not applied",
        edits: &[(
            "staged[j] = (x[j] * scale) * gain[j];",
            "staged[j] = x[j] * scale;",
            1,
        )],
    },
    Mutant {
        name: "head_f32_cap_dropped",
        clause: "head_f32",
        why: "the soft cap is not applied",
        edits: &[(
            "logits[row] = args.capped != 0u ? args.cap * tanh(dot / args.cap) : dot;",
            "logits[row] = dot;",
            1,
        )],
    },
    Mutant {
        name: "head_f32_last_float4_skipped",
        clause: "head_f32",
        why: "the dot stops one float4 short of the row",
        edits: &[(
            "for (uint i = 0u; i < HF_K / 128u; ++i) {",
            "for (uint i = 0u; i < HF_K / 128u - 1u; ++i) {",
            1,
        )],
    },
    Mutant {
        name: "head_f32_nan_unraised",
        clause: "head_f32_nan",
        why: "an unusable input row raises nothing",
        edits: &[("if (!usable && tid == 0u) {", "if (false) {", 1)],
    },
    Mutant {
        name: "rel_flash_lane_word_unindexed",
        clause: "rel_flash_rows",
        why: "every lane reads the first word of a key or value row",
        edits: &[("* REL_HEAD))[lane]", "* REL_HEAD))[0]", 2)],
    },
    Mutant {
        name: "rel_flash_bias_distance_forward",
        clause: "rel_flash_rows",
        why: "the bias column is the key's position, not its distance from the newest",
        edits: &[(
            "const uint dist = live - 1u - (lo + key);",
            "const uint dist = lo + key;",
            1,
        )],
    },
    Mutant {
        name: "rel_flash_extent_unpadded",
        clause: "rel_flash_rows",
        why: "a distance at or past the extent reads the table instead of the pad",
        edits: &[("if (dist < BLOOMERY_INK2_EXTENT) {", "if (true) {", 1)],
    },
    Mutant {
        name: "rel_flash_sink_dropped",
        clause: "rel_flash_rows",
        why: "the sink joins nothing in the denominator",
        edits: &[("(sink > -INFINITY ? exp(sink - m) : 0.f)", "0.f", 1)],
    },
    Mutant {
        name: "rel_flash_window_unclipped",
        clause: "rel_flash_rows",
        why: "a windowed visit attends every live key",
        edits: &[(
            "    return windowed ? live - args.window : 0u;",
            "    return false ? live - args.window : 0u;",
            1,
        )],
    },
    Mutant {
        name: "rel_flash_ring_unwrapped",
        clause: "rel_flash_rows",
        why: "a ring's key row is the position, not the position modulo the rows",
        edits: &[("pos % args.rows", "pos", 2)],
    },
    Mutant {
        name: "rel_flash_fold_first_segment",
        clause: "rel_flash_rows",
        why: "the fold takes the first segment's values alone",
        edits: &[(
            "for (uint s = 0u; s < a.segments; ++s) {",
            "for (uint s = 0u; s < 1u; ++s) {",
            1,
        )],
    },
    Mutant {
        name: "rel_flash_scale_dropped",
        clause: "rel_flash_rows",
        why: "the score's scale is not applied",
        edits: &[(
            "const float scaled =\n        (sixteen + simd_shuffle_xor(sixteen, ushort(16))) * args.scale;",
            "const float scaled =\n        (sixteen + simd_shuffle_xor(sixteen, ushort(16)));",
            1,
        )],
    },
    Mutant {
        name: "rel_flash_sink_nan_unraised",
        clause: "rel_flash_sink",
        why: "a sink that is not a number raises nothing",
        edits: &[(
            "if (lane == 0u && !refused && (isnan(sink) || sink == INFINITY)) {",
            "if (false) {",
            1,
        )],
    },
    Mutant {
        name: "rel_flash_keys_past_ctx_unraised",
        clause: "rel_flash_keys",
        why: "a live count past the context is attended as it is",
        edits: &[(
            "const bool refused = a.keys == 0u || a.keys > a.ctx;",
            "const bool refused = a.keys == 0u;",
            2,
        )],
    },
];
