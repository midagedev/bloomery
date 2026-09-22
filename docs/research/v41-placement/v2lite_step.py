# V2-Lite GPU bytes read per decode step, in the resident device formats (weights.rs), derived.
attn = 8192 + 2048*1320//1 if False else 0
per_layer_attn = (2048*4            # attn_norm f32
    + 3072*880                      # attn_q Q3_K, 8 SB x 110 B
    + 576*880                       # attn_kv_a_mqa Q3_K
    + 512*4                         # attn_kv_a_norm
    + 16*512*(128+16)               # derived q_nope2: rows 8192 x (k=128 codes + 4 f32 scales)
    + 2048*220                      # wv_b rows of attn_kv_b (16 heads x 128 v rows), k=512 -> 2 SB x 110
    + 2048*1152                     # attn_output Q4_K, 8 SB x 144
    + 2048*4)                       # ffn_norm
dense0 = 2*10944*880 + 2048*(10944 + 8*342)          # gate/up Q3_K + down Q5_1 (codes 1 B/value + d,m f32; padding not read)
moe = (2048*64*4                                     # router f32
       + 6*(2*1408*880 + 2048*(1408 + 4*44))         # 6 experts: gate/up Q3_K + down Q5_0 (codes + d f32)
       + 2*2816*880 + 2048*(11*144))                 # shexp gate/up Q3_K + down Q4_K
head = 102400*1680 + 2048*4
total = 27*per_layer_attn + dense0 + 26*moe + head + 880
print(f'per-layer attn {per_layer_attn:,}  dense0 {dense0:,}  moe {moe:,}  head {head:,}  TOTAL/step {total:,}')
step_ms = 1000/229.54      # A6000 depth table, depth 6, MMA default (plan.md:24)
nodes = 648*0.80e-3        # 648 nodes (plan.md:11) x c_node 0.80 us (plan.md:65, 3090-calibrated)
print(f'step {step_ms:.4f} ms, node term {nodes:.4f} ms, effective BW = {total/((step_ms-nodes)*1e-3)/1e9:.1f} GB/s (without node term: {total/(step_ms*1e-3)/1e9:.1f})')
