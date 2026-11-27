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
            "const float normed = state[c][v] * shared.rstd[v];",
            "const float normed = state[c][v];",
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
        name: "res_mix_prefetch_pass_skips",
        clause: "res_mix",
        why: "a lane's next pass of loads starts a value late: the scores drop values",
        edits: &[(
            "j0 < n; j0 += 32u * MIX_PREFETCH",
            "j0 < n; j0 += 32u * MIX_PREFETCH + 32u",
            1,
        )],
    },
    Mutant {
        name: "res_mix_softmax_row_zero",
        clause: "res_mix",
        why: "every softmax thread takes query row 0: the other rows keep raw dots",
        edits: &[(
            "        const uint q = tid;\n",
            "        const uint q = 0u;\n",
            1,
        )],
    },
    Mutant {
        name: "res_mix_sum_pass_skips",
        clause: "res_mix",
        why: "a thread's next pass of outputs starts a row late: outputs go unwritten",
        edits: &[(
            "j0 < n; j0 += MIX_THREADS * MIX_COLS",
            "j0 < n; j0 += MIX_THREADS * MIX_COLS + MIX_THREADS",
            1,
        )],
    },
    Mutant {
        name: "res_mix_reads_row_zero",
        clause: "res_mix_rows",
        why: "every row's mix reads row 0's banks and stream",
        edits: &[(
            "    const uint row = t * n;\n    const uint sg = tid / 32u;",
            "    const uint row = 0u;\n    const uint sg = tid / 32u;",
            1,
        )],
    },
    Mutant {
        name: "res_mix_nan_unraised",
        clause: "res_mix_nan",
        why: "an unusable vector raises nothing",
        edits: &[(
            "if (tid == 0u && !usable) {",
            "if (tid == 0u && false) {",
            1,
        )],
    },
    Mutant {
        name: "res_mix_overflow_usable",
        clause: "res_mix_nan",
        why: "a vector whose sum of squares overflows is mixed at a zero scale instead of failing",
        edits: &[(
            "shared.rstd[v] = rms_usable(shared.rstd[v], scale) ? scale : NAN;",
            "shared.rstd[v] = scale;",
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
            "if (is_q) {\n                x[i] *= gq[base + lane + 32u * i];",
            "if (true) {\n                x[i] *= gq[base + lane + 32u * i];",
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
            "float x = mine ? r[base + lane] * gr[base + lane] : 0.0f;",
            "float x = mine ? r[base + lane] : 0.0f;",
            1,
        )],
    },
    Mutant {
        name: "qk_clip_q_scale_one_head",
        clause: "qk_clip_rows",
        why: "every query head takes head 0's scale row, not its own: the file's \
              `attn_q_clip` is one row a head",
        edits: &[(
            "x[i] *= gq[base + lane + 32u * i];",
            "x[i] *= gq[lane + 32u * i];",
            1,
        )],
    },
    Mutant {
        name: "qk_clip_r_scale_one_head",
        clause: "qk_clip_rows",
        why: "every relative head takes head 0's scale row, not its own: the file's \
              `attn_rel_clip` is one row a head",
        edits: &[(
            "float x = mine ? r[base + lane] * gr[base + lane] : 0.0f;",
            "float x = mine ? r[base + lane] * gr[lane] : 0.0f;",
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
            "(1.0f - tanh(f)) * c + tanh(g) * m;",
            "(1.0f + tanh(f)) * c + tanh(g) * m;",
            1,
        )],
    },
    Mutant {
        name: "loop_control_down_cur_only",
        clause: "loop_control",
        why: "the down projection reads the stream half of the staged row alone",
        edits: &[("const uint kn = 2u * n;", "const uint kn = n;", 1)],
    },
    Mutant {
        name: "loop_control_down_pass_skips",
        clause: "loop_control",
        why: "a lane's next pass of down weights starts a value late: the dots drop weights",
        edits: &[(
            "k0 < kn; k0 += 32u * LC_DOWN_PREFETCH",
            "k0 < kn; k0 += 32u * LC_DOWN_PREFETCH + 32u",
            1,
        )],
    },
    Mutant {
        name: "loop_control_scales_pass_first",
        clause: "loop_control",
        why: "every stream value a thread loads is its pass's first: the stream half's norm is wrong",
        edits: &[("c[i] = cur[row + j];", "c[i] = cur[row + j0];", 1)],
    },
    Mutant {
        name: "loop_control_merge_pass_first",
        clause: "loop_control",
        why: "every pass of a merge reads the row's first gain weights",
        edits: &[(
            "gv[u] = float4(gw[d0 / 4u + u]);",
            "gv[u] = float4(gw[u]);",
            1,
        )],
    },
    Mutant {
        name: "loop_control_down_reads_row_zero",
        clause: "loop_control_rows",
        why: "every row's down projection reads row 0's stream and message",
        edits: &[(
            "loop_control_scales(cur, msg, args, t * n, tid, true, staged, partials, rstd);",
            "loop_control_scales(cur, msg, args, 0u, tid, true, staged, partials, rstd);",
            1,
        )],
    },
    Mutant {
        name: "loop_control_merge_reads_row_zero",
        clause: "loop_control_rows",
        why: "every row's merge reads row 0's down projection",
        edits: &[("hs[d] = h[t * rank + d];", "hs[d] = h[d];", 1)],
    },
    Mutant {
        name: "loop_control_down_first_block",
        clause: "loop_control",
        why: "every down threadgroup computes the rank's first eight outputs: the rest of h is never written",
        edits: &[("const uint r = tg.x * LC_SG + sg;", "const uint r = sg;", 1)],
    },
    Mutant {
        name: "loop_control_merge_first_block",
        clause: "loop_control",
        why: "every merge threadgroup writes the row's first 256 outputs: the rest of the row is never written",
        edits: &[(
            "const uint j = tg.x * LC_THREADS + tid;",
            "const uint j = tid;",
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
        why: "no entry applies its slot's scale: the expert's global scale and route weight are lost",
        edits: &[(
            "const float dot = simd_sum(acc) * w_scale;",
            "const float dot = simd_sum(acc);",
            4,
        )],
    },
    Mutant {
        name: "nvfp4_planes_row_stride",
        clause: "nvfp4_planes",
        why: "a planes row's codes start one group short of their place: every row past the first reads its neighbour's tail",
        edits: &[(
            "cr[i] = (device const uint2 *)(base + r * code_row);",
            "cr[i] = (device const uint2 *)(base + r * (code_row - 8ul));",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_planes_scale_swapped",
        clause: "nvfp4_rows",
        why: "a planes row reads the scale bytes of its pair row in the set",
        edits: &[(
            "sr[i] = sbase + r * (ulong)groups;",
            "sr[i] = sbase + (ulong)min(rs + (i ^ 1u), args.rows - 1u) * (ulong)groups;",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_planes_scale_doubled",
        clause: "nvfp4_rows",
        why: "the planes' group scale is ue4m3, not ue4m3 / 2",
        edits: &[(
            "#define NVP_SCALE 4194304.0f",
            "#define NVP_SCALE 8388608.0f",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_planes_global_scale_dropped",
        clause: "nvfp4_rows",
        why: "the planes' rows skip their slot's scale: the expert's global scale and route weight are lost",
        edits: &[(
            "const float dot = sum * w_scale;",
            "const float dot = sum;",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_planes_sign_dropped",
        clause: "nvfp4_rows",
        why: "the planes' decode drops a code's sign bit: the negative codes are positive",
        edits: &[("(sgn & 0x80008000u)", "(sgn & 0u)", 1)],
    },
    Mutant {
        name: "nvfp4_planes_pairs_misread",
        clause: "nvfp4_rows",
        why: "the planes' decode reads a word's codes 2 and 6 where its codes 1 and 5 stand",
        edits: &[(
            "const float2 a1 = nvp_pair<1u>(c.x);",
            "const float2 a1 = nvp_pair<2u>(c.x);",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_planes_x_held",
        clause: "nvfp4_planes",
        why: "the inputs a lane holds in registers for its first group serve its rows' every later group",
        edits: &[("xv[i] = xq[4u * g + i];", "xv[i] = xq[4u * j + i];", 1)],
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
        edits: &[("usable = usable && isfinite(s);", "usable = usable;", 1)],
    },
    Mutant {
        name: "nvfp4_planes_bad_scale_passes",
        clause: "nvfp4_nan",
        why: "a planes scale byte that is not a number is used as a number",
        edits: &[("bad |= (b >= 0x7fu ? 1u : 0u) << i;", "bad |= 0u << i;", 1)],
    },
    Mutant {
        name: "q8_0_scaled_word_dropped",
        clause: "nvfp4_rows",
        why: "the Q8_0 entry's rows skip their slot's scale: a coda expert's route weight is lost",
        edits: &[(
            "const float v = float(as_type<char>(blk[2u + lane])) * d;\n        acc += v * x[i * 32u + lane];\n    }\n    const float dot = simd_sum(acc) * w_scale;",
            "const float v = float(as_type<char>(blk[2u + lane])) * d;\n        acc += v * x[i * 32u + lane];\n    }\n    const float dot = simd_sum(acc);",
            1,
        )],
    },
    Mutant {
        name: "mxfp8_scale_bias_off",
        clause: "nvfp4_rows",
        why: "the MXFP8 group scale is read 2^(s − 126), one octave high",
        edits: &[("exp2(float(sb) - 127.0f)", "exp2(float(sb) - 126.0f)", 1)],
    },
    Mutant {
        name: "mxfp8_nan_code_passes",
        clause: "nvfp4_nan",
        why: "an E4M3 NaN code decodes as the largest normal value: the row stays finite",
        edits: &[(
            "    if (mag == 0x7fu) {\n        return NAN;\n    }",
            "    if (mag == 0x7fu) {\n        return 448.0f;\n    }",
            1,
        )],
    },
    Mutant {
        name: "mxfp8_quads_scale_group_swapped",
        clause: "nvfp4_rows",
        why: "the quads' scale bytes are read in reverse order: groups 0 and 3, 1 and 2 swap scales",
        edits: &[(
            "const uint sb = (sw >> (8u * gg)) & 0xffu;",
            "const uint sb = (sw >> (8u * (3u - gg))) & 0xffu;",
            1,
        )],
    },
    Mutant {
        name: "mxfp8_quads_code_byte_rotated",
        clause: "nvfp4_rows",
        why: "a group takes its neighbour's code byte of the lane's word",
        edits: &[(
            "const uint c = (cw >> (8u * gg)) & 0xffu;",
            "const uint c = (cw >> (8u * ((gg + 1u) & 3u))) & 0xffu;",
            1,
        )],
    },
    Mutant {
        name: "mxfp8_quads_lane_pair_swapped",
        clause: "nvfp4_rows",
        why: "lane l reads lane l^1's code word but its own input value",
        edits: &[(
            "*(device const uint *)(blk + 4u * lane);",
            "*(device const uint *)(blk + 4u * (lane ^ 1u));",
            1,
        )],
    },
    Mutant {
        name: "mxfp8_quads_sign_in_exponent",
        clause: "nvfp4_rows",
        why: "a negative code's sign lands in the f32 exponent field, not the sign bit",
        edits: &[("((c & 0x80u) << 24)", "((c & 0x80u) << 23)", 1)],
    },
    Mutant {
        name: "mxfp8_quads_nan_scale_usable",
        clause: "nvfp4_nan",
        why: "the NaN scale byte 0xff is the number 2.0: its row stays finite",
        edits: &[(
            "sb == 0xffu ? 0x7fc00000u :",
            "sb == 0xffu ? 0x40000000u :",
            1,
        )],
    },
    Mutant {
        name: "mxfp8_quads_nan_code_passes",
        clause: "nvfp4_nan",
        why: "an E4M3 NaN code decodes as the largest normal value: the row stays finite",
        edits: &[(
            "const float m = mag == 0x7fu ? NAN : e;",
            "const float m = mag == 0x7fu ? 448.0f : e;",
            1,
        )],
    },
    Mutant {
        name: "mxfp8_quads_fault_untracked",
        clause: "nvfp4_nan",
        why: "the quads' finite tracking is dropped: the NaN row raises the product site, not the quant-column one",
        edits: &[(
            "bad |= (as_type<uint>(v) << 1) >= 0xff000000u;",
            "bad = bad;",
            1,
        )],
    },
    Mutant {
        name: "mxfp8_quads_scale_octave_high",
        clause: "mxfp8_quads_identity",
        why: "every quads scale reads one octave high: the quads differ from the rows on every pair",
        edits: &[("(sb << 23)", "((sb + 1u) << 23)", 1)],
    },
    Mutant {
        name: "mxfp8_quads_subnormal_code_flushed",
        clause: "mxfp8_quads_identity",
        why: "the quads' subnormal E4M3 codes (1 to 7, exponent field 0) read 0, as the rows' man x 2^-9 does not",
        edits: &[(
            "const float e = float(as_type<half>(ushort(mag << 7))) * 256.0f;",
            "const float e = mag < 8u ? 0.0f : float(as_type<half>(ushort(mag << 7))) * 256.0f;",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_slot_reads_slot_zero",
        clause: "nvfp4_route",
        why: "every slot reads the first slot's id: the route runs one expert eight times",
        edits: &[(
            "pair_expert(ids, slot, args.experts, e)",
            "pair_expert(ids, 0u, args.experts, e)",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_route_weight_dropped",
        clause: "nvfp4_route",
        why: "a routed down's slot keeps its expert's scale without its route weight",
        edits: &[(
            "args.weighted != 0u ? scales[e] * route_w[slot] : scales[e];",
            "scales[e];",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_planes_ignored",
        clause: "nvfp4_route",
        why: "a slot's rows land as one block: the gate·up's up rows sit inside the next slot's gate rows",
        edits: &[(
            "return ((r / args.plane) * args.slots + slot) * args.plane + r % args.plane;",
            "return slot * args.rows + r;",
            1,
        )],
    },
    Mutant {
        name: "nvfp4_slot_input_shared",
        clause: "nvfp4_route",
        why: "every slot of a down reads the first slot's activation",
        edits: &[("    x += slot * args.x_stride;\n", "", 5)],
    },
    Mutant {
        name: "nvfp4_bad_id_reads_expert_zero",
        clause: "nvfp4_route",
        why: "an id past the stack reads expert 0 and raises nothing",
        edits: &[(
            "if (slot >= BLOOMERY_EXPERT_ROUTED || pair_expert(ids, slot, args.experts, e)) {",
            "if (slot >= BLOOMERY_EXPERT_ROUTED || (pair_expert(ids, slot, args.experts, e) && false)) {",
            1,
        )],
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
