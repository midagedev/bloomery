# b5plan: the V4.1 decode step at m = 1, node by node, and its predicted A6000 time.
# Tags: M = measured (source), D = derived here, A = assumed.
# Run: python3 nodes.py [depth]
import sys, math

D = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
BW = 575e3            # bytes per us (575 GB/s), plan.md 「모델」 BW head value [A6000]
C_NODE = 0.852        # us per node, plan.md 「모델」 (B9 touch graph, A6000) [M]
GAP = 0.12            # us between back-to-back kernels in a replay [M: nsys-d6-graph-234345, 74.8 us / 621 kernels]

# M: B11b A6000 site table, graph_us_mean per launch (lead-b11b/v41-B1.log). Includes the node's own cost.
S = dict(engram_wkv=254.227, q_a=13.983, q_b=69.808, kv=7.863, comp_kv=6.081, comp_gate=5.898,
         idx_k=1.444, lid_q=10.677, lid_proj=5.481, wo_a_group=9.994, wo_a_dense=60.762,
         wo_b=72.002, router=13.321, gate_exps=48.440, up_exps=48.421, down_exps=69.694,
         sh_gate=22.662, sh_up=22.460, sh_down=21.501, head=776.008)
# M: V2-Lite small-kernel durations in a replay (nsys-d6-graph-234345, 3090, 09-22): card-insensitive
# latency-bound kernels [A: same on the A6000]. Plus GAP each.
K = dict(norm2048=4.004, rms2048=4.416, rope=1.601, kv_append=3.084, merge=2.587, quant=1.974,
         quant_pair=2.110, add=1.339, combine=2.132, embed=2.176, router_topk64=8.391,
         argmax102400=42.641, gather=1.527)

def norm(n):          # D: single-block RMS over n values; load phase scales with n/256 per thread
    return K['norm2048'] * (0.45 + 0.55 * n / 2048)

EXP_PER = 16_773_120  # M inventory: gate+up q3_K 2x5,068,800 + down q4_K 6,635,520 per expert
GU_PER, DN_PER = 2 * 5_068_800, 6_635_520
R_SEL = 3_824_271_360 / 6335.0   # bytes/us on the `_sel` path (measured.py r_sel, 604 GB/s) [D from M]

def layer_kind(l):
    return dict(engram=l in (1, 14), source=l in (2, 8, 14, 20), gate=l in (2, 8, 14),
                index=l in (2, 8, 14, 20, 24, 28, 32, 36), window_only=l in (0, 1),
                host_only=l in (0, 1), ratio=0 if l < 2 else (2 if l < 20 else 1))

def keys(l, D):      # visible keys at depth D (window + compressed, capped at 512 selected)
    k = layer_kind(l)
    win = min(D, 128)
    if k['ratio'] == 0: return win
    vis = (D + 1) // k['ratio'] if D > 0 else 0
    return win + min(vis, 512)

def nodes(n_l=62.5, D=4096, fused=True):
    """Every node of one step on the GPU, as (name, kernel/owner, us, critical)."""
    out = []
    add = lambda *a: out.append(a)
    gpu_share = n_l / 384.0      # expected GPU slots per layer under the prefix rule, uniform [A]
    add('embed_broadcast', 'new (B5): bf16 row -> 4 f32 streams', K['embed'] + GAP, True)
    for l in range(40):
        k = layer_kind(l)
        if k['engram']:
            add(f'L{l} engram_rows', 'new (B5): mapped q8_0 rows -> f32 6144', 3.0 + GAP, True)  # A: PCIe read latency
            # depends on the token id only: B5 places it in layer 0's host-leg shadow
            add(f'L{l} engram_wkv', 'q8f32.rs q8_0_gemv 6144->25600', S['engram_wkv'], False)
            add(f'L{l} engram_gate+fold', 'ds41 engram_gate (b4engram)', 2 * norm(5120) + GAP, True)
        # attention sub-layer
        # one capture stream: HC_PRE(attn) stays on the critical path (a side stream is lever K2)
        add(f'L{l} hc_pre_attn', 'ds41 hc_pre (b4hc, fused split-K)', 4.0 + GAP, True)
        add(f'L{l} attn_norm', 'fused.rs norm_quant / elem.rs rms_norm', norm(5120) + GAP, True)
        add(f'L{l} q_a', 'q8f32.rs q8_0_gemv 5120->1280', S['q_a'], True)
        add(f'L{l} q_a_norm', 'elem.rs rms_norm 1280', norm(1280) + GAP, True)
        add(f'L{l} q_b', 'q8f32.rs q8_0_gemv 1280->32768', S['q_b'], True)
        add(f'L{l} q_rope', 'ds41 rope (b4rope)', K['rope'] + GAP, True)
        add(f'L{l} kv', 'q8f32.rs q8_0_gemv 5120->512', S['kv'], True)
        add(f'L{l} kv_norm_rope_append', 'ds41 rope/append (b4rope C)', K['kv_append'] + GAP, True)
        if k['source']:
            add(f'L{l} comp_kv', 'lib.rs q3k_gemv 5120->512', S['comp_kv'], True)
            if k['gate']:
                add(f'L{l} comp_gate', 'lib.rs q3k_gemv 5120->512', S['comp_gate'], True)
            add(f'L{l} compress', 'ds41 compress (b4-comp)', K['kv_append'] + 1.0 + GAP, True)
            add(f'L{l} idx_key_quant', 'lib.rs quantize_q8_1 512', K['quant'] + GAP, True)
            add(f'L{l} idx_key_gemv', 'lib.rs q3k_gemv 512->128', S['idx_k'], True)
            add(f'L{l} idx_key_tail', 'ds41 index_key (b4-comp)', K['kv_append'] + GAP, True)
        if k['index']:
            n_vis = (D + 1) // k['ratio'] if k['ratio'] else 0
            add(f'L{l} lid_q', 'q8f32.rs q8_0_gemv 1280->4096', S['lid_q'], True)
            add(f'L{l} lid_q_rope_had', 'ds41 indexer (b4-index)', K['rope'] + 1.0 + GAP, True)
            add(f'L{l} lid_weights', 'lib.rs q3k_gemv 5120->32', S['lid_proj'], True)
            scan = n_vis * 256 / BW                    # D: 128 f16 index key per row
            add(f'L{l} lid_score', 'ds41 indexer (b4-index)', max(scan, 3.0) + GAP, True)   # A: 3 us floor
            add(f'L{l} lid_topk', 'ds41 indexer (b4-index)', 6.0 + n_vis * 0.0005 + GAP, True)  # A
        nk = keys(l, D)
        att_bytes = nk * 1024
        seg = max(att_bytes / BW, 4.0) + nk * 0.0093   # D: 4 us fixed + per-key compute/L2 term
        add(f'L{l} attn_seg', 'ds41 attn (b4attn, 512 MMA)', seg + GAP, True)
        add(f'L{l} attn_merge_sink_ropeback', 'ds41 attn merge (b4attn/b4rope)', K['merge'] + GAP, True)
        add(f'L{l} wo_a', 'model/kernels.rs q8_0_gemv_heads (b4woa)', S['wo_a_dense'], True)
        add(f'L{l} wo_b', 'q8f32.rs q8_0_gemv 8192->5120', S['wo_b'], True)
        add(f'L{l} hc_post_fold_attn', 'ds41 hc_post+fold (b4hc)', 2.0 + GAP, True)
        # FFN sub-layer
        add(f'L{l} hc_pre_ffn', 'ds41 hc_pre (b4hc)', 4.0 + GAP, not fused)
        add(f'L{l} ffn_norm_quant', 'fused.rs norm_quant', norm(5120) + GAP, True)
        add(f'L{l} router', 'ds41 router (b4moe: f32 gemv + fused top-6)', S['router'] + 3.0, True)
        add(f'L{l} go', 'hybrid.rs memop (b2b)', 1.5, True)                       # A
        if not k['host_only']:
            gu = 6 * gpu_share * GU_PER / R_SEL
            dn = 6 * gpu_share * DN_PER / R_SEL
            add(f'L{l} exp_gate_up_swiglu', 'ds41 experts (b4moe, _sel + clamp)', max(gu, 3.0) + GAP, False)
            add(f'L{l} exp_quant', 'q5.rs quantize (existing)', K['quant'] + GAP, False)
            add(f'L{l} exp_down', 'q4k_sel.rs q4k_gemv_sel', max(dn, 3.0) + GAP, False)
        sh = S['sh_gate'] + S['sh_up'] - (0.0 if not fused else 0.0)
        add(f'L{l} sh_gate_up_swiglu', 'ds41 experts (b4moe, q8_0 fused)', sh, False)
        add(f'L{l} sh_down', 'q8f32.rs q8_0_gemv 2304->5120', S['sh_down'], False)
        add(f'L{l} wait', 'hybrid.rs memop (b2b)', 1.5, True)                     # A
        add(f'L{l} combine', 'ds41 combine (b4moe) + mapped read', K['combine'] + 1.5 + GAP, True)
        add(f'L{l} hc_post_fold_ffn', 'ds41 hc_post+fold (b4hc)', 2.0 + GAP, True)
    add('head rms_norm', 'elem.rs rms_norm 5120', norm(5120) + GAP, True)
    add('head quant', 'lib.rs quantize_q8_1', K['quant'] + GAP, True)
    add('head q6k', 'lib.rs q6k_gemv 5120->129280', S['head'], True)
    add('head argmax', 'elem.rs argmax (1 block)', K['argmax102400'] * 129280 / 102400 + GAP, True)
    return out

def summary(n_l, D, label):
    ns = nodes(n_l, D)
    tot = sum(n[2] for n in ns)
    crit = sum(n[2] for n in ns if n[3])
    overl = sum(n[2] for n in ns if not n[3])
    groups = {}
    for n in ns:
        g = n[0].split(' ', 1)[1] if ' ' in n[0] else n[0]
        groups.setdefault(g, [0, 0.0])
        groups[g][0] += 1; groups[g][1] += n[2]
    return ns, tot, crit, overl, groups

if __name__ == '__main__':
    for n_l, lab in ((62.5, '(a) n_l 62-63'),):
        ns, tot, crit, overl, groups = summary(n_l, D, lab)
        print(f'# {lab}  depth {D}: nodes {len(ns)}  gpu_sum {tot/1000:.3f} ms  '
              f'(overlappable with the host leg {overl/1000:.3f} ms)')
        for g, (c, t) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
            print(f'{g:28s} n={c:4d}  {t/1000:8.3f} ms')
