## shards

| shard | path | version | tensors | kv | header_end | data_base | alignment | file_bytes | tensor_bytes | unknown_size | pad | split.no | split.count | split.tensors.count |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | /models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8-exp8MXFP4/DeepSeek-V4.1-Flash-Q3_K_M-00001-of-00009.gguf | 3 | 37 | 68 | 5,770,702 | 5,770,720 | 32 | 9,403,133,664 | 9,397,362,904 | 0 | 40 | 0 | 9 |  |
| 2 | /models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8-exp8MXFP4/DeepSeek-V4.1-Flash-Q3_K_M-00002-of-00009.gguf | 3 | 6 | 3 | 466 | 480 | 32 | 104,616,879,968 | 104,616,879,488 | 0 | 0 | 1 | 9 |  |
| 3 | /models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8-exp8MXFP4/DeepSeek-V4.1-Flash-Q3_K_M-00003-of-00009.gguf | 3 | 163 | 3 | 10,221 | 10,240 | 32 | 49,225,812,736 | 49,225,802,256 | 0 | 240 | 2 | 9 |  |
| 4 | /models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8-exp8MXFP4/DeepSeek-V4.1-Flash-Q3_K_M-00004-of-00009.gguf | 3 | 177 | 3 | 11,172 | 11,200 | 32 | 42,268,673,728 | 42,268,662,248 | 0 | 280 | 3 | 9 |  |
| 5 | /models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8-exp8MXFP4/DeepSeek-V4.1-Flash-Q3_K_M-00005-of-00009.gguf | 3 | 8 | 3 | 615 | 640 | 32 | 107,180,313,376 | 107,180,312,736 | 0 | 0 | 4 | 9 |  |
| 6 | /models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8-exp8MXFP4/DeepSeek-V4.1-Flash-Q3_K_M-00006-of-00009.gguf | 3 | 183 | 3 | 11,591 | 11,616 | 32 | 43,773,808,480 | 43,773,796,584 | 0 | 280 | 5 | 9 |  |
| 7 | /models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8-exp8MXFP4/DeepSeek-V4.1-Flash-Q3_K_M-00007-of-00009.gguf | 3 | 158 | 3 | 10,044 | 10,048 | 32 | 44,233,617,984 | 44,233,607,696 | 0 | 240 | 6 | 9 |  |
| 8 | /models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8-exp8MXFP4/DeepSeek-V4.1-Flash-Q3_K_M-00008-of-00009.gguf | 3 | 175 | 3 | 11,077 | 11,104 | 32 | 44,370,290,016 | 44,370,278,632 | 0 | 280 | 7 | 9 |  |
| 9 | /models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8-exp8MXFP4/DeepSeek-V4.1-Flash-Q3_K_M-00009-of-00009.gguf | 3 | 139 | 3 | 8,833 | 8,864 | 32 | 37,015,004,320 | 37,014,995,216 | 0 | 240 | 8 | 9 |  |

Model: 1046 tensors across 9 shard(s), 0 duplicate name(s).

## types

| id | type | tensors | bytes | unknown_size | in engine GgmlType |
|---|---|---:|---:|---:|---|
| 0 | f32 | 449 | 2,097,600 | 0 | yes |
| 8 | q8_0 | 332 | 216,166,225,440 | 0 | no |
| 11 | q3_K | 163 | 124,596,285,440 | 0 | yes |
| 12 | q4_K | 32 | 81,537,269,760 | 0 | yes |
| 14 | q6_K | 1 | 542,976,000 | 0 | yes |
| 30 | bf16 | 45 | 1,481,277,440 | 0 | no |
| 39 | mxfp4 | 24 | 57,755,566,080 | 0 | no |

## tensors

Sorted by block index, then name; non-blk tensors last.

| block | name | dims | type | bytes | shard |
|---|---|---|---|---:|---|
| 0 | blk.0.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 0 |
| 0 | blk.0.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 0 |
| 0 | blk.0.attn_norm.weight | 5120 | f32 | 20,480 | 0 |
| 0 | blk.0.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 0 |
| 0 | blk.0.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 0 |
| 0 | blk.0.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 0 |
| 0 | blk.0.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 0 |
| 0 | blk.0.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 0 |
| 0 | blk.0.attn_sinks.weight | 64 | f32 | 256 | 0 |
| 0 | blk.0.exp_probs_b.bias | 384 | f32 | 1,536 | 0 |
| 0 | blk.0.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 0 |
| 0 | blk.0.ffn_down_exps.weight | 2304×5120×384 | mxfp4 | 2,406,481,920 | 0 |
| 0 | blk.0.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 0 |
| 0 | blk.0.ffn_gate_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 0 |
| 0 | blk.0.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 0 |
| 0 | blk.0.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 0 |
| 0 | blk.0.ffn_norm.weight | 5120 | f32 | 20,480 | 0 |
| 0 | blk.0.ffn_up_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 0 |
| 0 | blk.0.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 0 |
| 0 | blk.0.hc_attn_base.weight | 24 | f32 | 96 | 0 |
| 0 | blk.0.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 0 |
| 0 | blk.0.hc_attn_scale.weight | 3 | f32 | 12 | 0 |
| 0 | blk.0.hc_ffn_base.weight | 24 | f32 | 96 | 0 |
| 0 | blk.0.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 0 |
| 0 | blk.0.hc_ffn_scale.weight | 3 | f32 | 12 | 0 |
| 1 | blk.1.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 0 |
| 1 | blk.1.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 0 |
| 1 | blk.1.attn_norm.weight | 5120 | f32 | 20,480 | 0 |
| 1 | blk.1.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 0 |
| 1 | blk.1.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 0 |
| 1 | blk.1.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 0 |
| 1 | blk.1.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 0 |
| 1 | blk.1.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 0 |
| 1 | blk.1.attn_sinks.weight | 64 | f32 | 256 | 0 |
| 1 | blk.1.engram_embd.weight | 256×384006168 | q8_0 | 104,449,677,696 | 1 |
| 1 | blk.1.engram_k.weight | 5120×4 | bf16 | 40,960 | 1 |
| 1 | blk.1.engram_q.weight | 5120×4 | bf16 | 40,960 | 1 |
| 1 | blk.1.engram_wkv.weight | 6144×25600 | q8_0 | 167,116,800 | 1 |
| 1 | blk.1.exp_probs_b.bias | 384 | f32 | 1,536 | 1 |
| 1 | blk.1.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 1 |
| 1 | blk.1.ffn_down_exps.weight | 2304×5120×384 | mxfp4 | 2,406,481,920 | 2 |
| 1 | blk.1.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 2 |
| 1 | blk.1.ffn_gate_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 1 | blk.1.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 2 |
| 1 | blk.1.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 1 | blk.1.ffn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 1 | blk.1.ffn_up_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 1 | blk.1.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 1 | blk.1.hc_attn_base.weight | 24 | f32 | 96 | 2 |
| 1 | blk.1.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 1 | blk.1.hc_attn_scale.weight | 3 | f32 | 12 | 2 |
| 1 | blk.1.hc_ffn_base.weight | 24 | f32 | 96 | 2 |
| 1 | blk.1.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 1 | blk.1.hc_ffn_scale.weight | 3 | f32 | 12 | 2 |
| 2 | blk.2.attn_compressor_gate.weight | 5120×512 | q3_K | 1,126,400 | 2 |
| 2 | blk.2.attn_compressor_kv.weight | 5120×512 | q3_K | 1,126,400 | 2 |
| 2 | blk.2.attn_compressor_norm.weight | 512 | f32 | 2,048 | 2 |
| 2 | blk.2.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 2 |
| 2 | blk.2.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 2 |
| 2 | blk.2.attn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 2 | blk.2.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 2 |
| 2 | blk.2.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 2 |
| 2 | blk.2.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 2 |
| 2 | blk.2.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 2 |
| 2 | blk.2.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 2 |
| 2 | blk.2.attn_sinks.weight | 64 | f32 | 256 | 2 |
| 2 | blk.2.exp_probs_b.bias | 384 | f32 | 1,536 | 2 |
| 2 | blk.2.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 2 |
| 2 | blk.2.ffn_down_exps.weight | 2304×5120×384 | mxfp4 | 2,406,481,920 | 2 |
| 2 | blk.2.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 2 |
| 2 | blk.2.ffn_gate_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 2 | blk.2.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 2 |
| 2 | blk.2.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 2 | blk.2.ffn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 2 | blk.2.ffn_up_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 2 | blk.2.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 2 | blk.2.hc_attn_base.weight | 24 | f32 | 96 | 2 |
| 2 | blk.2.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 2 | blk.2.hc_attn_scale.weight | 3 | f32 | 12 | 2 |
| 2 | blk.2.hc_ffn_base.weight | 24 | f32 | 96 | 2 |
| 2 | blk.2.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 2 | blk.2.hc_ffn_scale.weight | 3 | f32 | 12 | 2 |
| 2 | blk.2.indexer.attn_k.weight | 512×128 | q3_K | 28,160 | 2 |
| 2 | blk.2.indexer.attn_q_b.weight | 1280×4096 | q8_0 | 5,570,560 | 2 |
| 2 | blk.2.indexer.k_norm.weight | 128 | f32 | 512 | 2 |
| 2 | blk.2.indexer.proj.weight | 5120×32 | q3_K | 70,400 | 2 |
| 3 | blk.3.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 2 |
| 3 | blk.3.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 2 |
| 3 | blk.3.attn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 3 | blk.3.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 2 |
| 3 | blk.3.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 2 |
| 3 | blk.3.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 2 |
| 3 | blk.3.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 2 |
| 3 | blk.3.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 2 |
| 3 | blk.3.attn_sinks.weight | 64 | f32 | 256 | 2 |
| 3 | blk.3.exp_probs_b.bias | 384 | f32 | 1,536 | 2 |
| 3 | blk.3.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 2 |
| 3 | blk.3.ffn_down_exps.weight | 2304×5120×384 | mxfp4 | 2,406,481,920 | 2 |
| 3 | blk.3.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 2 |
| 3 | blk.3.ffn_gate_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 3 | blk.3.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 2 |
| 3 | blk.3.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 3 | blk.3.ffn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 3 | blk.3.ffn_up_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 3 | blk.3.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 3 | blk.3.hc_attn_base.weight | 24 | f32 | 96 | 2 |
| 3 | blk.3.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 3 | blk.3.hc_attn_scale.weight | 3 | f32 | 12 | 2 |
| 3 | blk.3.hc_ffn_base.weight | 24 | f32 | 96 | 2 |
| 3 | blk.3.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 3 | blk.3.hc_ffn_scale.weight | 3 | f32 | 12 | 2 |
| 4 | blk.4.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 2 |
| 4 | blk.4.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 2 |
| 4 | blk.4.attn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 4 | blk.4.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 2 |
| 4 | blk.4.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 2 |
| 4 | blk.4.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 2 |
| 4 | blk.4.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 2 |
| 4 | blk.4.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 2 |
| 4 | blk.4.attn_sinks.weight | 64 | f32 | 256 | 2 |
| 4 | blk.4.exp_probs_b.bias | 384 | f32 | 1,536 | 2 |
| 4 | blk.4.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 2 |
| 4 | blk.4.ffn_down_exps.weight | 2304×5120×384 | mxfp4 | 2,406,481,920 | 2 |
| 4 | blk.4.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 2 |
| 4 | blk.4.ffn_gate_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 4 | blk.4.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 2 |
| 4 | blk.4.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 4 | blk.4.ffn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 4 | blk.4.ffn_up_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 4 | blk.4.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 4 | blk.4.hc_attn_base.weight | 24 | f32 | 96 | 2 |
| 4 | blk.4.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 4 | blk.4.hc_attn_scale.weight | 3 | f32 | 12 | 2 |
| 4 | blk.4.hc_ffn_base.weight | 24 | f32 | 96 | 2 |
| 4 | blk.4.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 4 | blk.4.hc_ffn_scale.weight | 3 | f32 | 12 | 2 |
| 5 | blk.5.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 2 |
| 5 | blk.5.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 2 |
| 5 | blk.5.attn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 5 | blk.5.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 2 |
| 5 | blk.5.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 2 |
| 5 | blk.5.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 2 |
| 5 | blk.5.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 2 |
| 5 | blk.5.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 2 |
| 5 | blk.5.attn_sinks.weight | 64 | f32 | 256 | 2 |
| 5 | blk.5.exp_probs_b.bias | 384 | f32 | 1,536 | 2 |
| 5 | blk.5.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 2 |
| 5 | blk.5.ffn_down_exps.weight | 2304×5120×384 | mxfp4 | 2,406,481,920 | 2 |
| 5 | blk.5.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 2 |
| 5 | blk.5.ffn_gate_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 5 | blk.5.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 2 |
| 5 | blk.5.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 5 | blk.5.ffn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 5 | blk.5.ffn_up_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 5 | blk.5.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 5 | blk.5.hc_attn_base.weight | 24 | f32 | 96 | 2 |
| 5 | blk.5.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 5 | blk.5.hc_attn_scale.weight | 3 | f32 | 12 | 2 |
| 5 | blk.5.hc_ffn_base.weight | 24 | f32 | 96 | 2 |
| 5 | blk.5.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 5 | blk.5.hc_ffn_scale.weight | 3 | f32 | 12 | 2 |
| 6 | blk.6.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 2 |
| 6 | blk.6.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 2 |
| 6 | blk.6.attn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 6 | blk.6.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 2 |
| 6 | blk.6.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 2 |
| 6 | blk.6.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 2 |
| 6 | blk.6.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 2 |
| 6 | blk.6.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 2 |
| 6 | blk.6.attn_sinks.weight | 64 | f32 | 256 | 2 |
| 6 | blk.6.exp_probs_b.bias | 384 | f32 | 1,536 | 2 |
| 6 | blk.6.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 2 |
| 6 | blk.6.ffn_down_exps.weight | 2304×5120×384 | mxfp4 | 2,406,481,920 | 2 |
| 6 | blk.6.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 2 |
| 6 | blk.6.ffn_gate_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 6 | blk.6.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 2 |
| 6 | blk.6.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 6 | blk.6.ffn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 6 | blk.6.ffn_up_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 6 | blk.6.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 6 | blk.6.hc_attn_base.weight | 24 | f32 | 96 | 2 |
| 6 | blk.6.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 6 | blk.6.hc_attn_scale.weight | 3 | f32 | 12 | 2 |
| 6 | blk.6.hc_ffn_base.weight | 24 | f32 | 96 | 2 |
| 6 | blk.6.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 2 |
| 6 | blk.6.hc_ffn_scale.weight | 3 | f32 | 12 | 2 |
| 7 | blk.7.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 2 |
| 7 | blk.7.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 2 |
| 7 | blk.7.attn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 7 | blk.7.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 2 |
| 7 | blk.7.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 2 |
| 7 | blk.7.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 2 |
| 7 | blk.7.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 2 |
| 7 | blk.7.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 2 |
| 7 | blk.7.attn_sinks.weight | 64 | f32 | 256 | 2 |
| 7 | blk.7.exp_probs_b.bias | 384 | f32 | 1,536 | 2 |
| 7 | blk.7.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 2 |
| 7 | blk.7.ffn_down_exps.weight | 2304×5120×384 | mxfp4 | 2,406,481,920 | 2 |
| 7 | blk.7.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 2 |
| 7 | blk.7.ffn_gate_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 2 |
| 7 | blk.7.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 2 |
| 7 | blk.7.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 2 |
| 7 | blk.7.ffn_norm.weight | 5120 | f32 | 20,480 | 2 |
| 7 | blk.7.ffn_up_exps.weight | 5120×2304×384 | mxfp4 | 2,406,481,920 | 3 |
| 7 | blk.7.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 7 | blk.7.hc_attn_base.weight | 24 | f32 | 96 | 3 |
| 7 | blk.7.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 7 | blk.7.hc_attn_scale.weight | 3 | f32 | 12 | 3 |
| 7 | blk.7.hc_ffn_base.weight | 24 | f32 | 96 | 3 |
| 7 | blk.7.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 7 | blk.7.hc_ffn_scale.weight | 3 | f32 | 12 | 3 |
| 8 | blk.8.attn_compressor_gate.weight | 5120×512 | q3_K | 1,126,400 | 3 |
| 8 | blk.8.attn_compressor_kv.weight | 5120×512 | q3_K | 1,126,400 | 3 |
| 8 | blk.8.attn_compressor_norm.weight | 512 | f32 | 2,048 | 3 |
| 8 | blk.8.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 3 |
| 8 | blk.8.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 3 |
| 8 | blk.8.attn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 8 | blk.8.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 3 |
| 8 | blk.8.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 3 |
| 8 | blk.8.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 3 |
| 8 | blk.8.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 3 |
| 8 | blk.8.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 3 |
| 8 | blk.8.attn_sinks.weight | 64 | f32 | 256 | 3 |
| 8 | blk.8.exp_probs_b.bias | 384 | f32 | 1,536 | 3 |
| 8 | blk.8.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 3 |
| 8 | blk.8.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 3 |
| 8 | blk.8.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 3 |
| 8 | blk.8.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 8 | blk.8.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 3 |
| 8 | blk.8.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 8 | blk.8.ffn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 8 | blk.8.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 8 | blk.8.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 8 | blk.8.hc_attn_base.weight | 24 | f32 | 96 | 3 |
| 8 | blk.8.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 8 | blk.8.hc_attn_scale.weight | 3 | f32 | 12 | 3 |
| 8 | blk.8.hc_ffn_base.weight | 24 | f32 | 96 | 3 |
| 8 | blk.8.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 8 | blk.8.hc_ffn_scale.weight | 3 | f32 | 12 | 3 |
| 8 | blk.8.indexer.attn_k.weight | 512×128 | q3_K | 28,160 | 3 |
| 8 | blk.8.indexer.attn_q_b.weight | 1280×4096 | q8_0 | 5,570,560 | 3 |
| 8 | blk.8.indexer.k_norm.weight | 128 | f32 | 512 | 3 |
| 8 | blk.8.indexer.proj.weight | 5120×32 | q3_K | 70,400 | 3 |
| 9 | blk.9.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 3 |
| 9 | blk.9.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 3 |
| 9 | blk.9.attn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 9 | blk.9.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 3 |
| 9 | blk.9.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 3 |
| 9 | blk.9.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 3 |
| 9 | blk.9.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 3 |
| 9 | blk.9.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 3 |
| 9 | blk.9.attn_sinks.weight | 64 | f32 | 256 | 3 |
| 9 | blk.9.exp_probs_b.bias | 384 | f32 | 1,536 | 3 |
| 9 | blk.9.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 3 |
| 9 | blk.9.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 3 |
| 9 | blk.9.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 3 |
| 9 | blk.9.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 9 | blk.9.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 3 |
| 9 | blk.9.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 9 | blk.9.ffn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 9 | blk.9.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 9 | blk.9.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 9 | blk.9.hc_attn_base.weight | 24 | f32 | 96 | 3 |
| 9 | blk.9.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 9 | blk.9.hc_attn_scale.weight | 3 | f32 | 12 | 3 |
| 9 | blk.9.hc_ffn_base.weight | 24 | f32 | 96 | 3 |
| 9 | blk.9.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 9 | blk.9.hc_ffn_scale.weight | 3 | f32 | 12 | 3 |
| 10 | blk.10.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 3 |
| 10 | blk.10.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 3 |
| 10 | blk.10.attn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 10 | blk.10.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 3 |
| 10 | blk.10.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 3 |
| 10 | blk.10.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 3 |
| 10 | blk.10.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 3 |
| 10 | blk.10.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 3 |
| 10 | blk.10.attn_sinks.weight | 64 | f32 | 256 | 3 |
| 10 | blk.10.exp_probs_b.bias | 384 | f32 | 1,536 | 3 |
| 10 | blk.10.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 3 |
| 10 | blk.10.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 3 |
| 10 | blk.10.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 3 |
| 10 | blk.10.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 10 | blk.10.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 3 |
| 10 | blk.10.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 10 | blk.10.ffn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 10 | blk.10.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 10 | blk.10.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 10 | blk.10.hc_attn_base.weight | 24 | f32 | 96 | 3 |
| 10 | blk.10.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 10 | blk.10.hc_attn_scale.weight | 3 | f32 | 12 | 3 |
| 10 | blk.10.hc_ffn_base.weight | 24 | f32 | 96 | 3 |
| 10 | blk.10.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 10 | blk.10.hc_ffn_scale.weight | 3 | f32 | 12 | 3 |
| 11 | blk.11.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 3 |
| 11 | blk.11.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 3 |
| 11 | blk.11.attn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 11 | blk.11.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 3 |
| 11 | blk.11.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 3 |
| 11 | blk.11.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 3 |
| 11 | blk.11.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 3 |
| 11 | blk.11.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 3 |
| 11 | blk.11.attn_sinks.weight | 64 | f32 | 256 | 3 |
| 11 | blk.11.exp_probs_b.bias | 384 | f32 | 1,536 | 3 |
| 11 | blk.11.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 3 |
| 11 | blk.11.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 3 |
| 11 | blk.11.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 3 |
| 11 | blk.11.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 11 | blk.11.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 3 |
| 11 | blk.11.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 11 | blk.11.ffn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 11 | blk.11.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 11 | blk.11.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 11 | blk.11.hc_attn_base.weight | 24 | f32 | 96 | 3 |
| 11 | blk.11.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 11 | blk.11.hc_attn_scale.weight | 3 | f32 | 12 | 3 |
| 11 | blk.11.hc_ffn_base.weight | 24 | f32 | 96 | 3 |
| 11 | blk.11.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 11 | blk.11.hc_ffn_scale.weight | 3 | f32 | 12 | 3 |
| 12 | blk.12.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 3 |
| 12 | blk.12.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 3 |
| 12 | blk.12.attn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 12 | blk.12.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 3 |
| 12 | blk.12.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 3 |
| 12 | blk.12.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 3 |
| 12 | blk.12.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 3 |
| 12 | blk.12.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 3 |
| 12 | blk.12.attn_sinks.weight | 64 | f32 | 256 | 3 |
| 12 | blk.12.exp_probs_b.bias | 384 | f32 | 1,536 | 3 |
| 12 | blk.12.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 3 |
| 12 | blk.12.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 3 |
| 12 | blk.12.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 3 |
| 12 | blk.12.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 12 | blk.12.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 3 |
| 12 | blk.12.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 12 | blk.12.ffn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 12 | blk.12.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 12 | blk.12.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 12 | blk.12.hc_attn_base.weight | 24 | f32 | 96 | 3 |
| 12 | blk.12.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 12 | blk.12.hc_attn_scale.weight | 3 | f32 | 12 | 3 |
| 12 | blk.12.hc_ffn_base.weight | 24 | f32 | 96 | 3 |
| 12 | blk.12.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 12 | blk.12.hc_ffn_scale.weight | 3 | f32 | 12 | 3 |
| 13 | blk.13.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 3 |
| 13 | blk.13.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 3 |
| 13 | blk.13.attn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 13 | blk.13.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 3 |
| 13 | blk.13.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 3 |
| 13 | blk.13.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 3 |
| 13 | blk.13.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 3 |
| 13 | blk.13.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 3 |
| 13 | blk.13.attn_sinks.weight | 64 | f32 | 256 | 3 |
| 13 | blk.13.exp_probs_b.bias | 384 | f32 | 1,536 | 3 |
| 13 | blk.13.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 3 |
| 13 | blk.13.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 3 |
| 13 | blk.13.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 3 |
| 13 | blk.13.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 13 | blk.13.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 3 |
| 13 | blk.13.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 13 | blk.13.ffn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 13 | blk.13.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 3 |
| 13 | blk.13.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 3 |
| 13 | blk.13.hc_attn_base.weight | 24 | f32 | 96 | 3 |
| 13 | blk.13.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 13 | blk.13.hc_attn_scale.weight | 3 | f32 | 12 | 3 |
| 13 | blk.13.hc_ffn_base.weight | 24 | f32 | 96 | 3 |
| 13 | blk.13.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 3 |
| 13 | blk.13.hc_ffn_scale.weight | 3 | f32 | 12 | 3 |
| 14 | blk.14.attn_compressor_gate.weight | 5120×512 | q3_K | 1,126,400 | 3 |
| 14 | blk.14.attn_compressor_kv.weight | 5120×512 | q3_K | 1,126,400 | 3 |
| 14 | blk.14.attn_compressor_norm.weight | 512 | f32 | 2,048 | 3 |
| 14 | blk.14.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 3 |
| 14 | blk.14.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 3 |
| 14 | blk.14.attn_norm.weight | 5120 | f32 | 20,480 | 3 |
| 14 | blk.14.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 3 |
| 14 | blk.14.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 3 |
| 14 | blk.14.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 3 |
| 14 | blk.14.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 3 |
| 14 | blk.14.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 3 |
| 14 | blk.14.attn_sinks.weight | 64 | f32 | 256 | 3 |
| 14 | blk.14.engram_embd.weight | 256×384016682 | q8_0 | 104,452,537,504 | 4 |
| 14 | blk.14.engram_k.weight | 5120×4 | bf16 | 40,960 | 4 |
| 14 | blk.14.engram_q.weight | 5120×4 | bf16 | 40,960 | 4 |
| 14 | blk.14.engram_wkv.weight | 6144×25600 | q8_0 | 167,116,800 | 4 |
| 14 | blk.14.exp_probs_b.bias | 384 | f32 | 1,536 | 4 |
| 14 | blk.14.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 4 |
| 14 | blk.14.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 4 |
| 14 | blk.14.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 4 |
| 14 | blk.14.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 14 | blk.14.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 5 |
| 14 | blk.14.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 14 | blk.14.ffn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 14 | blk.14.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 14 | blk.14.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 14 | blk.14.hc_attn_base.weight | 24 | f32 | 96 | 5 |
| 14 | blk.14.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 14 | blk.14.hc_attn_scale.weight | 3 | f32 | 12 | 5 |
| 14 | blk.14.hc_ffn_base.weight | 24 | f32 | 96 | 5 |
| 14 | blk.14.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 14 | blk.14.hc_ffn_scale.weight | 3 | f32 | 12 | 5 |
| 14 | blk.14.indexer.attn_k.weight | 512×128 | q3_K | 28,160 | 5 |
| 14 | blk.14.indexer.attn_q_b.weight | 1280×4096 | q8_0 | 5,570,560 | 5 |
| 14 | blk.14.indexer.k_norm.weight | 128 | f32 | 512 | 5 |
| 14 | blk.14.indexer.proj.weight | 5120×32 | q3_K | 70,400 | 5 |
| 15 | blk.15.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 5 |
| 15 | blk.15.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 5 |
| 15 | blk.15.attn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 15 | blk.15.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 5 |
| 15 | blk.15.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 5 |
| 15 | blk.15.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 5 |
| 15 | blk.15.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 5 |
| 15 | blk.15.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 5 |
| 15 | blk.15.attn_sinks.weight | 64 | f32 | 256 | 5 |
| 15 | blk.15.exp_probs_b.bias | 384 | f32 | 1,536 | 5 |
| 15 | blk.15.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 5 |
| 15 | blk.15.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 5 |
| 15 | blk.15.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 5 |
| 15 | blk.15.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 15 | blk.15.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 5 |
| 15 | blk.15.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 15 | blk.15.ffn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 15 | blk.15.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 15 | blk.15.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 15 | blk.15.hc_attn_base.weight | 24 | f32 | 96 | 5 |
| 15 | blk.15.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 15 | blk.15.hc_attn_scale.weight | 3 | f32 | 12 | 5 |
| 15 | blk.15.hc_ffn_base.weight | 24 | f32 | 96 | 5 |
| 15 | blk.15.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 15 | blk.15.hc_ffn_scale.weight | 3 | f32 | 12 | 5 |
| 16 | blk.16.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 5 |
| 16 | blk.16.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 5 |
| 16 | blk.16.attn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 16 | blk.16.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 5 |
| 16 | blk.16.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 5 |
| 16 | blk.16.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 5 |
| 16 | blk.16.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 5 |
| 16 | blk.16.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 5 |
| 16 | blk.16.attn_sinks.weight | 64 | f32 | 256 | 5 |
| 16 | blk.16.exp_probs_b.bias | 384 | f32 | 1,536 | 5 |
| 16 | blk.16.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 5 |
| 16 | blk.16.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 5 |
| 16 | blk.16.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 5 |
| 16 | blk.16.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 16 | blk.16.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 5 |
| 16 | blk.16.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 16 | blk.16.ffn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 16 | blk.16.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 16 | blk.16.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 16 | blk.16.hc_attn_base.weight | 24 | f32 | 96 | 5 |
| 16 | blk.16.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 16 | blk.16.hc_attn_scale.weight | 3 | f32 | 12 | 5 |
| 16 | blk.16.hc_ffn_base.weight | 24 | f32 | 96 | 5 |
| 16 | blk.16.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 16 | blk.16.hc_ffn_scale.weight | 3 | f32 | 12 | 5 |
| 17 | blk.17.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 5 |
| 17 | blk.17.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 5 |
| 17 | blk.17.attn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 17 | blk.17.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 5 |
| 17 | blk.17.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 5 |
| 17 | blk.17.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 5 |
| 17 | blk.17.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 5 |
| 17 | blk.17.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 5 |
| 17 | blk.17.attn_sinks.weight | 64 | f32 | 256 | 5 |
| 17 | blk.17.exp_probs_b.bias | 384 | f32 | 1,536 | 5 |
| 17 | blk.17.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 5 |
| 17 | blk.17.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 5 |
| 17 | blk.17.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 5 |
| 17 | blk.17.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 17 | blk.17.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 5 |
| 17 | blk.17.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 17 | blk.17.ffn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 17 | blk.17.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 17 | blk.17.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 17 | blk.17.hc_attn_base.weight | 24 | f32 | 96 | 5 |
| 17 | blk.17.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 17 | blk.17.hc_attn_scale.weight | 3 | f32 | 12 | 5 |
| 17 | blk.17.hc_ffn_base.weight | 24 | f32 | 96 | 5 |
| 17 | blk.17.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 17 | blk.17.hc_ffn_scale.weight | 3 | f32 | 12 | 5 |
| 18 | blk.18.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 5 |
| 18 | blk.18.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 5 |
| 18 | blk.18.attn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 18 | blk.18.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 5 |
| 18 | blk.18.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 5 |
| 18 | blk.18.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 5 |
| 18 | blk.18.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 5 |
| 18 | blk.18.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 5 |
| 18 | blk.18.attn_sinks.weight | 64 | f32 | 256 | 5 |
| 18 | blk.18.exp_probs_b.bias | 384 | f32 | 1,536 | 5 |
| 18 | blk.18.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 5 |
| 18 | blk.18.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 5 |
| 18 | blk.18.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 5 |
| 18 | blk.18.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 18 | blk.18.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 5 |
| 18 | blk.18.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 18 | blk.18.ffn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 18 | blk.18.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 18 | blk.18.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 18 | blk.18.hc_attn_base.weight | 24 | f32 | 96 | 5 |
| 18 | blk.18.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 18 | blk.18.hc_attn_scale.weight | 3 | f32 | 12 | 5 |
| 18 | blk.18.hc_ffn_base.weight | 24 | f32 | 96 | 5 |
| 18 | blk.18.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 18 | blk.18.hc_ffn_scale.weight | 3 | f32 | 12 | 5 |
| 19 | blk.19.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 5 |
| 19 | blk.19.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 5 |
| 19 | blk.19.attn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 19 | blk.19.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 5 |
| 19 | blk.19.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 5 |
| 19 | blk.19.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 5 |
| 19 | blk.19.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 5 |
| 19 | blk.19.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 5 |
| 19 | blk.19.attn_sinks.weight | 64 | f32 | 256 | 5 |
| 19 | blk.19.exp_probs_b.bias | 384 | f32 | 1,536 | 5 |
| 19 | blk.19.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 5 |
| 19 | blk.19.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 5 |
| 19 | blk.19.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 5 |
| 19 | blk.19.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 19 | blk.19.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 5 |
| 19 | blk.19.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 19 | blk.19.ffn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 19 | blk.19.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 19 | blk.19.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 19 | blk.19.hc_attn_base.weight | 24 | f32 | 96 | 5 |
| 19 | blk.19.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 19 | blk.19.hc_attn_scale.weight | 3 | f32 | 12 | 5 |
| 19 | blk.19.hc_ffn_base.weight | 24 | f32 | 96 | 5 |
| 19 | blk.19.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 19 | blk.19.hc_ffn_scale.weight | 3 | f32 | 12 | 5 |
| 20 | blk.20.attn_compressor_kv.weight | 5120×512 | q3_K | 1,126,400 | 5 |
| 20 | blk.20.attn_compressor_norm.weight | 512 | f32 | 2,048 | 5 |
| 20 | blk.20.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 5 |
| 20 | blk.20.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 5 |
| 20 | blk.20.attn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 20 | blk.20.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 5 |
| 20 | blk.20.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 5 |
| 20 | blk.20.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 5 |
| 20 | blk.20.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 5 |
| 20 | blk.20.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 5 |
| 20 | blk.20.attn_sinks.weight | 64 | f32 | 256 | 5 |
| 20 | blk.20.exp_probs_b.bias | 384 | f32 | 1,536 | 5 |
| 20 | blk.20.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 5 |
| 20 | blk.20.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 5 |
| 20 | blk.20.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 5 |
| 20 | blk.20.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 20 | blk.20.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 5 |
| 20 | blk.20.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 20 | blk.20.ffn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 20 | blk.20.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 5 |
| 20 | blk.20.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 5 |
| 20 | blk.20.hc_attn_base.weight | 24 | f32 | 96 | 5 |
| 20 | blk.20.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 20 | blk.20.hc_attn_scale.weight | 3 | f32 | 12 | 5 |
| 20 | blk.20.hc_ffn_base.weight | 24 | f32 | 96 | 5 |
| 20 | blk.20.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 5 |
| 20 | blk.20.hc_ffn_scale.weight | 3 | f32 | 12 | 5 |
| 20 | blk.20.indexer.attn_k.weight | 512×128 | q3_K | 28,160 | 5 |
| 20 | blk.20.indexer.attn_q_b.weight | 1280×4096 | q8_0 | 5,570,560 | 5 |
| 20 | blk.20.indexer.k_norm.weight | 128 | f32 | 512 | 5 |
| 20 | blk.20.indexer.proj.weight | 5120×32 | q3_K | 70,400 | 5 |
| 21 | blk.21.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 5 |
| 21 | blk.21.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 5 |
| 21 | blk.21.attn_norm.weight | 5120 | f32 | 20,480 | 5 |
| 21 | blk.21.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 5 |
| 21 | blk.21.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 5 |
| 21 | blk.21.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 5 |
| 21 | blk.21.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 5 |
| 21 | blk.21.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 5 |
| 21 | blk.21.attn_sinks.weight | 64 | f32 | 256 | 5 |
| 21 | blk.21.exp_probs_b.bias | 384 | f32 | 1,536 | 5 |
| 21 | blk.21.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 5 |
| 21 | blk.21.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 6 |
| 21 | blk.21.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 6 |
| 21 | blk.21.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 21 | blk.21.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 6 |
| 21 | blk.21.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 21 | blk.21.ffn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 21 | blk.21.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 21 | blk.21.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 21 | blk.21.hc_attn_base.weight | 24 | f32 | 96 | 6 |
| 21 | blk.21.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 21 | blk.21.hc_attn_scale.weight | 3 | f32 | 12 | 6 |
| 21 | blk.21.hc_ffn_base.weight | 24 | f32 | 96 | 6 |
| 21 | blk.21.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 21 | blk.21.hc_ffn_scale.weight | 3 | f32 | 12 | 6 |
| 22 | blk.22.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 6 |
| 22 | blk.22.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 6 |
| 22 | blk.22.attn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 22 | blk.22.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 6 |
| 22 | blk.22.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 6 |
| 22 | blk.22.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 6 |
| 22 | blk.22.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 6 |
| 22 | blk.22.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 6 |
| 22 | blk.22.attn_sinks.weight | 64 | f32 | 256 | 6 |
| 22 | blk.22.exp_probs_b.bias | 384 | f32 | 1,536 | 6 |
| 22 | blk.22.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 6 |
| 22 | blk.22.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 6 |
| 22 | blk.22.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 6 |
| 22 | blk.22.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 22 | blk.22.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 6 |
| 22 | blk.22.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 22 | blk.22.ffn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 22 | blk.22.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 22 | blk.22.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 22 | blk.22.hc_attn_base.weight | 24 | f32 | 96 | 6 |
| 22 | blk.22.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 22 | blk.22.hc_attn_scale.weight | 3 | f32 | 12 | 6 |
| 22 | blk.22.hc_ffn_base.weight | 24 | f32 | 96 | 6 |
| 22 | blk.22.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 22 | blk.22.hc_ffn_scale.weight | 3 | f32 | 12 | 6 |
| 23 | blk.23.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 6 |
| 23 | blk.23.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 6 |
| 23 | blk.23.attn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 23 | blk.23.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 6 |
| 23 | blk.23.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 6 |
| 23 | blk.23.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 6 |
| 23 | blk.23.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 6 |
| 23 | blk.23.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 6 |
| 23 | blk.23.attn_sinks.weight | 64 | f32 | 256 | 6 |
| 23 | blk.23.exp_probs_b.bias | 384 | f32 | 1,536 | 6 |
| 23 | blk.23.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 6 |
| 23 | blk.23.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 6 |
| 23 | blk.23.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 6 |
| 23 | blk.23.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 23 | blk.23.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 6 |
| 23 | blk.23.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 23 | blk.23.ffn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 23 | blk.23.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 23 | blk.23.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 23 | blk.23.hc_attn_base.weight | 24 | f32 | 96 | 6 |
| 23 | blk.23.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 23 | blk.23.hc_attn_scale.weight | 3 | f32 | 12 | 6 |
| 23 | blk.23.hc_ffn_base.weight | 24 | f32 | 96 | 6 |
| 23 | blk.23.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 23 | blk.23.hc_ffn_scale.weight | 3 | f32 | 12 | 6 |
| 24 | blk.24.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 6 |
| 24 | blk.24.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 6 |
| 24 | blk.24.attn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 24 | blk.24.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 6 |
| 24 | blk.24.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 6 |
| 24 | blk.24.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 6 |
| 24 | blk.24.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 6 |
| 24 | blk.24.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 6 |
| 24 | blk.24.attn_sinks.weight | 64 | f32 | 256 | 6 |
| 24 | blk.24.exp_probs_b.bias | 384 | f32 | 1,536 | 6 |
| 24 | blk.24.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 6 |
| 24 | blk.24.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 6 |
| 24 | blk.24.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 6 |
| 24 | blk.24.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 24 | blk.24.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 6 |
| 24 | blk.24.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 24 | blk.24.ffn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 24 | blk.24.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 24 | blk.24.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 24 | blk.24.hc_attn_base.weight | 24 | f32 | 96 | 6 |
| 24 | blk.24.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 24 | blk.24.hc_attn_scale.weight | 3 | f32 | 12 | 6 |
| 24 | blk.24.hc_ffn_base.weight | 24 | f32 | 96 | 6 |
| 24 | blk.24.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 24 | blk.24.hc_ffn_scale.weight | 3 | f32 | 12 | 6 |
| 24 | blk.24.indexer.attn_q_b.weight | 1280×4096 | q8_0 | 5,570,560 | 6 |
| 24 | blk.24.indexer.proj.weight | 5120×32 | q3_K | 70,400 | 6 |
| 25 | blk.25.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 6 |
| 25 | blk.25.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 6 |
| 25 | blk.25.attn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 25 | blk.25.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 6 |
| 25 | blk.25.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 6 |
| 25 | blk.25.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 6 |
| 25 | blk.25.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 6 |
| 25 | blk.25.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 6 |
| 25 | blk.25.attn_sinks.weight | 64 | f32 | 256 | 6 |
| 25 | blk.25.exp_probs_b.bias | 384 | f32 | 1,536 | 6 |
| 25 | blk.25.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 6 |
| 25 | blk.25.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 6 |
| 25 | blk.25.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 6 |
| 25 | blk.25.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 25 | blk.25.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 6 |
| 25 | blk.25.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 25 | blk.25.ffn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 25 | blk.25.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 25 | blk.25.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 25 | blk.25.hc_attn_base.weight | 24 | f32 | 96 | 6 |
| 25 | blk.25.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 25 | blk.25.hc_attn_scale.weight | 3 | f32 | 12 | 6 |
| 25 | blk.25.hc_ffn_base.weight | 24 | f32 | 96 | 6 |
| 25 | blk.25.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 25 | blk.25.hc_ffn_scale.weight | 3 | f32 | 12 | 6 |
| 26 | blk.26.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 6 |
| 26 | blk.26.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 6 |
| 26 | blk.26.attn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 26 | blk.26.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 6 |
| 26 | blk.26.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 6 |
| 26 | blk.26.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 6 |
| 26 | blk.26.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 6 |
| 26 | blk.26.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 6 |
| 26 | blk.26.attn_sinks.weight | 64 | f32 | 256 | 6 |
| 26 | blk.26.exp_probs_b.bias | 384 | f32 | 1,536 | 6 |
| 26 | blk.26.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 6 |
| 26 | blk.26.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 6 |
| 26 | blk.26.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 6 |
| 26 | blk.26.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 26 | blk.26.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 6 |
| 26 | blk.26.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 26 | blk.26.ffn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 26 | blk.26.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 26 | blk.26.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 26 | blk.26.hc_attn_base.weight | 24 | f32 | 96 | 6 |
| 26 | blk.26.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 26 | blk.26.hc_attn_scale.weight | 3 | f32 | 12 | 6 |
| 26 | blk.26.hc_ffn_base.weight | 24 | f32 | 96 | 6 |
| 26 | blk.26.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 6 |
| 26 | blk.26.hc_ffn_scale.weight | 3 | f32 | 12 | 6 |
| 27 | blk.27.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 6 |
| 27 | blk.27.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 6 |
| 27 | blk.27.attn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 27 | blk.27.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 6 |
| 27 | blk.27.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 6 |
| 27 | blk.27.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 6 |
| 27 | blk.27.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 6 |
| 27 | blk.27.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 6 |
| 27 | blk.27.attn_sinks.weight | 64 | f32 | 256 | 6 |
| 27 | blk.27.exp_probs_b.bias | 384 | f32 | 1,536 | 6 |
| 27 | blk.27.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 6 |
| 27 | blk.27.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 6 |
| 27 | blk.27.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 6 |
| 27 | blk.27.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 6 |
| 27 | blk.27.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 6 |
| 27 | blk.27.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 6 |
| 27 | blk.27.ffn_norm.weight | 5120 | f32 | 20,480 | 6 |
| 27 | blk.27.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 27 | blk.27.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 27 | blk.27.hc_attn_base.weight | 24 | f32 | 96 | 7 |
| 27 | blk.27.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 27 | blk.27.hc_attn_scale.weight | 3 | f32 | 12 | 7 |
| 27 | blk.27.hc_ffn_base.weight | 24 | f32 | 96 | 7 |
| 27 | blk.27.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 27 | blk.27.hc_ffn_scale.weight | 3 | f32 | 12 | 7 |
| 28 | blk.28.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 7 |
| 28 | blk.28.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 7 |
| 28 | blk.28.attn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 28 | blk.28.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 7 |
| 28 | blk.28.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 7 |
| 28 | blk.28.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 7 |
| 28 | blk.28.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 7 |
| 28 | blk.28.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 7 |
| 28 | blk.28.attn_sinks.weight | 64 | f32 | 256 | 7 |
| 28 | blk.28.exp_probs_b.bias | 384 | f32 | 1,536 | 7 |
| 28 | blk.28.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 7 |
| 28 | blk.28.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 7 |
| 28 | blk.28.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 7 |
| 28 | blk.28.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 28 | blk.28.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 7 |
| 28 | blk.28.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 28 | blk.28.ffn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 28 | blk.28.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 28 | blk.28.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 28 | blk.28.hc_attn_base.weight | 24 | f32 | 96 | 7 |
| 28 | blk.28.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 28 | blk.28.hc_attn_scale.weight | 3 | f32 | 12 | 7 |
| 28 | blk.28.hc_ffn_base.weight | 24 | f32 | 96 | 7 |
| 28 | blk.28.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 28 | blk.28.hc_ffn_scale.weight | 3 | f32 | 12 | 7 |
| 28 | blk.28.indexer.attn_q_b.weight | 1280×4096 | q8_0 | 5,570,560 | 7 |
| 28 | blk.28.indexer.proj.weight | 5120×32 | q3_K | 70,400 | 7 |
| 29 | blk.29.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 7 |
| 29 | blk.29.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 7 |
| 29 | blk.29.attn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 29 | blk.29.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 7 |
| 29 | blk.29.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 7 |
| 29 | blk.29.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 7 |
| 29 | blk.29.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 7 |
| 29 | blk.29.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 7 |
| 29 | blk.29.attn_sinks.weight | 64 | f32 | 256 | 7 |
| 29 | blk.29.exp_probs_b.bias | 384 | f32 | 1,536 | 7 |
| 29 | blk.29.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 7 |
| 29 | blk.29.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 7 |
| 29 | blk.29.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 7 |
| 29 | blk.29.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 29 | blk.29.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 7 |
| 29 | blk.29.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 29 | blk.29.ffn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 29 | blk.29.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 29 | blk.29.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 29 | blk.29.hc_attn_base.weight | 24 | f32 | 96 | 7 |
| 29 | blk.29.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 29 | blk.29.hc_attn_scale.weight | 3 | f32 | 12 | 7 |
| 29 | blk.29.hc_ffn_base.weight | 24 | f32 | 96 | 7 |
| 29 | blk.29.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 29 | blk.29.hc_ffn_scale.weight | 3 | f32 | 12 | 7 |
| 30 | blk.30.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 7 |
| 30 | blk.30.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 7 |
| 30 | blk.30.attn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 30 | blk.30.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 7 |
| 30 | blk.30.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 7 |
| 30 | blk.30.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 7 |
| 30 | blk.30.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 7 |
| 30 | blk.30.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 7 |
| 30 | blk.30.attn_sinks.weight | 64 | f32 | 256 | 7 |
| 30 | blk.30.exp_probs_b.bias | 384 | f32 | 1,536 | 7 |
| 30 | blk.30.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 7 |
| 30 | blk.30.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 7 |
| 30 | blk.30.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 7 |
| 30 | blk.30.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 30 | blk.30.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 7 |
| 30 | blk.30.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 30 | blk.30.ffn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 30 | blk.30.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 30 | blk.30.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 30 | blk.30.hc_attn_base.weight | 24 | f32 | 96 | 7 |
| 30 | blk.30.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 30 | blk.30.hc_attn_scale.weight | 3 | f32 | 12 | 7 |
| 30 | blk.30.hc_ffn_base.weight | 24 | f32 | 96 | 7 |
| 30 | blk.30.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 30 | blk.30.hc_ffn_scale.weight | 3 | f32 | 12 | 7 |
| 31 | blk.31.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 7 |
| 31 | blk.31.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 7 |
| 31 | blk.31.attn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 31 | blk.31.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 7 |
| 31 | blk.31.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 7 |
| 31 | blk.31.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 7 |
| 31 | blk.31.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 7 |
| 31 | blk.31.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 7 |
| 31 | blk.31.attn_sinks.weight | 64 | f32 | 256 | 7 |
| 31 | blk.31.exp_probs_b.bias | 384 | f32 | 1,536 | 7 |
| 31 | blk.31.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 7 |
| 31 | blk.31.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 7 |
| 31 | blk.31.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 7 |
| 31 | blk.31.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 31 | blk.31.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 7 |
| 31 | blk.31.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 31 | blk.31.ffn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 31 | blk.31.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 31 | blk.31.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 31 | blk.31.hc_attn_base.weight | 24 | f32 | 96 | 7 |
| 31 | blk.31.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 31 | blk.31.hc_attn_scale.weight | 3 | f32 | 12 | 7 |
| 31 | blk.31.hc_ffn_base.weight | 24 | f32 | 96 | 7 |
| 31 | blk.31.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 31 | blk.31.hc_ffn_scale.weight | 3 | f32 | 12 | 7 |
| 32 | blk.32.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 7 |
| 32 | blk.32.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 7 |
| 32 | blk.32.attn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 32 | blk.32.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 7 |
| 32 | blk.32.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 7 |
| 32 | blk.32.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 7 |
| 32 | blk.32.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 7 |
| 32 | blk.32.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 7 |
| 32 | blk.32.attn_sinks.weight | 64 | f32 | 256 | 7 |
| 32 | blk.32.exp_probs_b.bias | 384 | f32 | 1,536 | 7 |
| 32 | blk.32.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 7 |
| 32 | blk.32.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 7 |
| 32 | blk.32.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 7 |
| 32 | blk.32.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 32 | blk.32.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 7 |
| 32 | blk.32.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 32 | blk.32.ffn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 32 | blk.32.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 32 | blk.32.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 32 | blk.32.hc_attn_base.weight | 24 | f32 | 96 | 7 |
| 32 | blk.32.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 32 | blk.32.hc_attn_scale.weight | 3 | f32 | 12 | 7 |
| 32 | blk.32.hc_ffn_base.weight | 24 | f32 | 96 | 7 |
| 32 | blk.32.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 32 | blk.32.hc_ffn_scale.weight | 3 | f32 | 12 | 7 |
| 32 | blk.32.indexer.attn_q_b.weight | 1280×4096 | q8_0 | 5,570,560 | 7 |
| 32 | blk.32.indexer.proj.weight | 5120×32 | q3_K | 70,400 | 7 |
| 33 | blk.33.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 7 |
| 33 | blk.33.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 7 |
| 33 | blk.33.attn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 33 | blk.33.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 7 |
| 33 | blk.33.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 7 |
| 33 | blk.33.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 7 |
| 33 | blk.33.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 7 |
| 33 | blk.33.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 7 |
| 33 | blk.33.attn_sinks.weight | 64 | f32 | 256 | 7 |
| 33 | blk.33.exp_probs_b.bias | 384 | f32 | 1,536 | 7 |
| 33 | blk.33.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 7 |
| 33 | blk.33.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 7 |
| 33 | blk.33.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 7 |
| 33 | blk.33.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 33 | blk.33.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 7 |
| 33 | blk.33.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 33 | blk.33.ffn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 33 | blk.33.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 7 |
| 33 | blk.33.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 7 |
| 33 | blk.33.hc_attn_base.weight | 24 | f32 | 96 | 7 |
| 33 | blk.33.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 33 | blk.33.hc_attn_scale.weight | 3 | f32 | 12 | 7 |
| 33 | blk.33.hc_ffn_base.weight | 24 | f32 | 96 | 7 |
| 33 | blk.33.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 7 |
| 33 | blk.33.hc_ffn_scale.weight | 3 | f32 | 12 | 7 |
| 34 | blk.34.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 7 |
| 34 | blk.34.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 7 |
| 34 | blk.34.attn_norm.weight | 5120 | f32 | 20,480 | 7 |
| 34 | blk.34.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 7 |
| 34 | blk.34.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 7 |
| 34 | blk.34.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 7 |
| 34 | blk.34.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 7 |
| 34 | blk.34.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 7 |
| 34 | blk.34.attn_sinks.weight | 64 | f32 | 256 | 7 |
| 34 | blk.34.exp_probs_b.bias | 384 | f32 | 1,536 | 7 |
| 34 | blk.34.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 7 |
| 34 | blk.34.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 7 |
| 34 | blk.34.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 7 |
| 34 | blk.34.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 34 | blk.34.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 8 |
| 34 | blk.34.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 34 | blk.34.ffn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 34 | blk.34.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 34 | blk.34.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 34 | blk.34.hc_attn_base.weight | 24 | f32 | 96 | 8 |
| 34 | blk.34.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 34 | blk.34.hc_attn_scale.weight | 3 | f32 | 12 | 8 |
| 34 | blk.34.hc_ffn_base.weight | 24 | f32 | 96 | 8 |
| 34 | blk.34.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 34 | blk.34.hc_ffn_scale.weight | 3 | f32 | 12 | 8 |
| 35 | blk.35.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 8 |
| 35 | blk.35.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 8 |
| 35 | blk.35.attn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 35 | blk.35.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 8 |
| 35 | blk.35.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 8 |
| 35 | blk.35.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 8 |
| 35 | blk.35.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 8 |
| 35 | blk.35.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 8 |
| 35 | blk.35.attn_sinks.weight | 64 | f32 | 256 | 8 |
| 35 | blk.35.exp_probs_b.bias | 384 | f32 | 1,536 | 8 |
| 35 | blk.35.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 8 |
| 35 | blk.35.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 8 |
| 35 | blk.35.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 8 |
| 35 | blk.35.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 35 | blk.35.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 8 |
| 35 | blk.35.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 35 | blk.35.ffn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 35 | blk.35.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 35 | blk.35.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 35 | blk.35.hc_attn_base.weight | 24 | f32 | 96 | 8 |
| 35 | blk.35.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 35 | blk.35.hc_attn_scale.weight | 3 | f32 | 12 | 8 |
| 35 | blk.35.hc_ffn_base.weight | 24 | f32 | 96 | 8 |
| 35 | blk.35.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 35 | blk.35.hc_ffn_scale.weight | 3 | f32 | 12 | 8 |
| 36 | blk.36.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 8 |
| 36 | blk.36.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 8 |
| 36 | blk.36.attn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 36 | blk.36.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 8 |
| 36 | blk.36.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 8 |
| 36 | blk.36.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 8 |
| 36 | blk.36.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 8 |
| 36 | blk.36.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 8 |
| 36 | blk.36.attn_sinks.weight | 64 | f32 | 256 | 8 |
| 36 | blk.36.exp_probs_b.bias | 384 | f32 | 1,536 | 8 |
| 36 | blk.36.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 8 |
| 36 | blk.36.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 8 |
| 36 | blk.36.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 8 |
| 36 | blk.36.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 36 | blk.36.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 8 |
| 36 | blk.36.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 36 | blk.36.ffn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 36 | blk.36.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 36 | blk.36.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 36 | blk.36.hc_attn_base.weight | 24 | f32 | 96 | 8 |
| 36 | blk.36.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 36 | blk.36.hc_attn_scale.weight | 3 | f32 | 12 | 8 |
| 36 | blk.36.hc_ffn_base.weight | 24 | f32 | 96 | 8 |
| 36 | blk.36.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 36 | blk.36.hc_ffn_scale.weight | 3 | f32 | 12 | 8 |
| 36 | blk.36.indexer.attn_q_b.weight | 1280×4096 | q8_0 | 5,570,560 | 8 |
| 36 | blk.36.indexer.proj.weight | 5120×32 | q3_K | 70,400 | 8 |
| 37 | blk.37.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 8 |
| 37 | blk.37.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 8 |
| 37 | blk.37.attn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 37 | blk.37.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 8 |
| 37 | blk.37.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 8 |
| 37 | blk.37.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 8 |
| 37 | blk.37.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 8 |
| 37 | blk.37.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 8 |
| 37 | blk.37.attn_sinks.weight | 64 | f32 | 256 | 8 |
| 37 | blk.37.exp_probs_b.bias | 384 | f32 | 1,536 | 8 |
| 37 | blk.37.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 8 |
| 37 | blk.37.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 8 |
| 37 | blk.37.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 8 |
| 37 | blk.37.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 37 | blk.37.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 8 |
| 37 | blk.37.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 37 | blk.37.ffn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 37 | blk.37.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 37 | blk.37.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 37 | blk.37.hc_attn_base.weight | 24 | f32 | 96 | 8 |
| 37 | blk.37.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 37 | blk.37.hc_attn_scale.weight | 3 | f32 | 12 | 8 |
| 37 | blk.37.hc_ffn_base.weight | 24 | f32 | 96 | 8 |
| 37 | blk.37.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 37 | blk.37.hc_ffn_scale.weight | 3 | f32 | 12 | 8 |
| 38 | blk.38.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 8 |
| 38 | blk.38.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 8 |
| 38 | blk.38.attn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 38 | blk.38.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 8 |
| 38 | blk.38.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 8 |
| 38 | blk.38.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 8 |
| 38 | blk.38.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 8 |
| 38 | blk.38.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 8 |
| 38 | blk.38.attn_sinks.weight | 64 | f32 | 256 | 8 |
| 38 | blk.38.exp_probs_b.bias | 384 | f32 | 1,536 | 8 |
| 38 | blk.38.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 8 |
| 38 | blk.38.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 8 |
| 38 | blk.38.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 8 |
| 38 | blk.38.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 38 | blk.38.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 8 |
| 38 | blk.38.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 38 | blk.38.ffn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 38 | blk.38.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 38 | blk.38.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 38 | blk.38.hc_attn_base.weight | 24 | f32 | 96 | 8 |
| 38 | blk.38.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 38 | blk.38.hc_attn_scale.weight | 3 | f32 | 12 | 8 |
| 38 | blk.38.hc_ffn_base.weight | 24 | f32 | 96 | 8 |
| 38 | blk.38.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 38 | blk.38.hc_ffn_scale.weight | 3 | f32 | 12 | 8 |
| 39 | blk.39.attn_kv.weight | 5120×512 | q8_0 | 2,785,280 | 8 |
| 39 | blk.39.attn_kv_a_norm.weight | 512 | f32 | 2,048 | 8 |
| 39 | blk.39.attn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 39 | blk.39.attn_output_a.weight | 4096×8192 | q8_0 | 35,651,584 | 8 |
| 39 | blk.39.attn_output_b.weight | 8192×5120 | q8_0 | 44,564,480 | 8 |
| 39 | blk.39.attn_q_a.weight | 5120×1280 | q8_0 | 6,963,200 | 8 |
| 39 | blk.39.attn_q_a_norm.weight | 1280 | f32 | 5,120 | 8 |
| 39 | blk.39.attn_q_b.weight | 1280×32768 | q8_0 | 44,564,480 | 8 |
| 39 | blk.39.attn_sinks.weight | 64 | f32 | 256 | 8 |
| 39 | blk.39.exp_probs_b.bias | 384 | f32 | 1,536 | 8 |
| 39 | blk.39.exp_probs_b_vl.bias | 384 | f32 | 1,536 | 8 |
| 39 | blk.39.ffn_down_exps.weight | 2304×5120×384 | q4_K | 2,548,039,680 | 8 |
| 39 | blk.39.ffn_down_shexp.weight | 2304×5120 | q8_0 | 12,533,760 | 8 |
| 39 | blk.39.ffn_gate_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 39 | blk.39.ffn_gate_inp.weight | 5120×384 | bf16 | 3,932,160 | 8 |
| 39 | blk.39.ffn_gate_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 39 | blk.39.ffn_norm.weight | 5120 | f32 | 20,480 | 8 |
| 39 | blk.39.ffn_up_exps.weight | 5120×2304×384 | q3_K | 1,946,419,200 | 8 |
| 39 | blk.39.ffn_up_shexp.weight | 5120×2304 | q8_0 | 12,533,760 | 8 |
| 39 | blk.39.hc_attn_base.weight | 24 | f32 | 96 | 8 |
| 39 | blk.39.hc_attn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 39 | blk.39.hc_attn_scale.weight | 3 | f32 | 12 | 8 |
| 39 | blk.39.hc_ffn_base.weight | 24 | f32 | 96 | 8 |
| 39 | blk.39.hc_ffn_fn.weight | 20480×24 | q3_K | 211,200 | 8 |
| 39 | blk.39.hc_ffn_scale.weight | 3 | f32 | 12 | 8 |
| - | output.weight | 5120×129280 | q6_K | 542,976,000 | 0 |
| - | output_norm.weight | 5120 | f32 | 20,480 | 0 |
| - | token_embd.weight | 5120×129280 | bf16 | 1,323,827,200 | 0 |

## groups

| group | rule | tensors | bytes | GiB | unknown_size |
|---|---|---:|---:|---:|---:|
| token_embd | token_embd* | 1 | 1,323,827,200 | 1.2329 | 0 |
| output | output*/output_norm* | 2 | 542,996,480 | 0.5057 | 0 |
| attn | blk.*.attn* | 371 | 5,390,170,112 | 5.0200 | 0 |
| router | blk.MoE.ffn_gate* | 40 | 157,286,400 | 0.1465 | 0 |
| ffn_dense | ffn_* dense-path | 40 | 819,200 | 0.0008 | 0 |
| ffn_shexp | *_shexp* | 120 | 1,504,051,200 | 1.4008 | 0 |
| ffn_exps_routed | *_exps* | 120 | 263,863,664,640 | 245.7422 | 0 |
| engram | *engram* | 8 | 209,236,612,640 | 194.8668 | 0 |
| mtp_nextn | *mtp*/*nextn*/*spark* | 0 | 0 | 0.0000 | 0 |
| other | unmatched | 344 | 62,269,888 | 0.0580 | 0 |

### engram tensor names

| name | dims |
|---|---|
| blk.1.engram_embd.weight | 256×384006168 |
| blk.1.engram_k.weight | 5120×4 |
| blk.1.engram_q.weight | 5120×4 |
| blk.1.engram_wkv.weight | 6144×25600 |
| blk.14.engram_embd.weight | 256×384016682 |
| blk.14.engram_k.weight | 5120×4 |
| blk.14.engram_q.weight | 5120×4 |
| blk.14.engram_wkv.weight | 6144×25600 |

### other tensor names

| name | dims |
|---|---|
| blk.0.exp_probs_b.bias | 384 |
| blk.0.exp_probs_b_vl.bias | 384 |
| blk.0.hc_attn_base.weight | 24 |
| blk.0.hc_attn_fn.weight | 20480×24 |
| blk.0.hc_attn_scale.weight | 3 |
| blk.0.hc_ffn_base.weight | 24 |
| blk.0.hc_ffn_fn.weight | 20480×24 |
| blk.0.hc_ffn_scale.weight | 3 |
| blk.1.exp_probs_b.bias | 384 |
| blk.1.exp_probs_b_vl.bias | 384 |
| blk.1.hc_attn_base.weight | 24 |
| blk.1.hc_attn_fn.weight | 20480×24 |
| blk.1.hc_attn_scale.weight | 3 |
| blk.1.hc_ffn_base.weight | 24 |
| blk.1.hc_ffn_fn.weight | 20480×24 |
| blk.1.hc_ffn_scale.weight | 3 |
| blk.2.exp_probs_b.bias | 384 |
| blk.2.exp_probs_b_vl.bias | 384 |
| blk.2.hc_attn_base.weight | 24 |
| blk.2.hc_attn_fn.weight | 20480×24 |
| blk.2.hc_attn_scale.weight | 3 |
| blk.2.hc_ffn_base.weight | 24 |
| blk.2.hc_ffn_fn.weight | 20480×24 |
| blk.2.hc_ffn_scale.weight | 3 |
| blk.2.indexer.attn_k.weight | 512×128 |
| blk.2.indexer.attn_q_b.weight | 1280×4096 |
| blk.2.indexer.k_norm.weight | 128 |
| blk.2.indexer.proj.weight | 5120×32 |
| blk.3.exp_probs_b.bias | 384 |
| blk.3.exp_probs_b_vl.bias | 384 |
| blk.3.hc_attn_base.weight | 24 |
| blk.3.hc_attn_fn.weight | 20480×24 |
| blk.3.hc_attn_scale.weight | 3 |
| blk.3.hc_ffn_base.weight | 24 |
| blk.3.hc_ffn_fn.weight | 20480×24 |
| blk.3.hc_ffn_scale.weight | 3 |
| blk.4.exp_probs_b.bias | 384 |
| blk.4.exp_probs_b_vl.bias | 384 |
| blk.4.hc_attn_base.weight | 24 |
| blk.4.hc_attn_fn.weight | 20480×24 |
| blk.4.hc_attn_scale.weight | 3 |
| blk.4.hc_ffn_base.weight | 24 |
| blk.4.hc_ffn_fn.weight | 20480×24 |
| blk.4.hc_ffn_scale.weight | 3 |
| blk.5.exp_probs_b.bias | 384 |
| blk.5.exp_probs_b_vl.bias | 384 |
| blk.5.hc_attn_base.weight | 24 |
| blk.5.hc_attn_fn.weight | 20480×24 |
| blk.5.hc_attn_scale.weight | 3 |
| blk.5.hc_ffn_base.weight | 24 |
| blk.5.hc_ffn_fn.weight | 20480×24 |
| blk.5.hc_ffn_scale.weight | 3 |
| blk.6.exp_probs_b.bias | 384 |
| blk.6.exp_probs_b_vl.bias | 384 |
| blk.6.hc_attn_base.weight | 24 |
| blk.6.hc_attn_fn.weight | 20480×24 |
| blk.6.hc_attn_scale.weight | 3 |
| blk.6.hc_ffn_base.weight | 24 |
| blk.6.hc_ffn_fn.weight | 20480×24 |
| blk.6.hc_ffn_scale.weight | 3 |
| blk.7.exp_probs_b.bias | 384 |
| blk.7.exp_probs_b_vl.bias | 384 |
| blk.7.hc_attn_base.weight | 24 |
| blk.7.hc_attn_fn.weight | 20480×24 |
| blk.7.hc_attn_scale.weight | 3 |
| blk.7.hc_ffn_base.weight | 24 |
| blk.7.hc_ffn_fn.weight | 20480×24 |
| blk.7.hc_ffn_scale.weight | 3 |
| blk.8.exp_probs_b.bias | 384 |
| blk.8.exp_probs_b_vl.bias | 384 |
| blk.8.hc_attn_base.weight | 24 |
| blk.8.hc_attn_fn.weight | 20480×24 |
| blk.8.hc_attn_scale.weight | 3 |
| blk.8.hc_ffn_base.weight | 24 |
| blk.8.hc_ffn_fn.weight | 20480×24 |
| blk.8.hc_ffn_scale.weight | 3 |
| blk.8.indexer.attn_k.weight | 512×128 |
| blk.8.indexer.attn_q_b.weight | 1280×4096 |
| blk.8.indexer.k_norm.weight | 128 |
| blk.8.indexer.proj.weight | 5120×32 |
| blk.9.exp_probs_b.bias | 384 |
| blk.9.exp_probs_b_vl.bias | 384 |
| blk.9.hc_attn_base.weight | 24 |
| blk.9.hc_attn_fn.weight | 20480×24 |
| blk.9.hc_attn_scale.weight | 3 |
| blk.9.hc_ffn_base.weight | 24 |
| blk.9.hc_ffn_fn.weight | 20480×24 |
| blk.9.hc_ffn_scale.weight | 3 |
| blk.10.exp_probs_b.bias | 384 |
| blk.10.exp_probs_b_vl.bias | 384 |
| blk.10.hc_attn_base.weight | 24 |
| blk.10.hc_attn_fn.weight | 20480×24 |
| blk.10.hc_attn_scale.weight | 3 |
| blk.10.hc_ffn_base.weight | 24 |
| blk.10.hc_ffn_fn.weight | 20480×24 |
| blk.10.hc_ffn_scale.weight | 3 |
| blk.11.exp_probs_b.bias | 384 |
| blk.11.exp_probs_b_vl.bias | 384 |
| blk.11.hc_attn_base.weight | 24 |
| blk.11.hc_attn_fn.weight | 20480×24 |
| blk.11.hc_attn_scale.weight | 3 |
| blk.11.hc_ffn_base.weight | 24 |
| blk.11.hc_ffn_fn.weight | 20480×24 |
| blk.11.hc_ffn_scale.weight | 3 |
| blk.12.exp_probs_b.bias | 384 |
| blk.12.exp_probs_b_vl.bias | 384 |
| blk.12.hc_attn_base.weight | 24 |
| blk.12.hc_attn_fn.weight | 20480×24 |
| blk.12.hc_attn_scale.weight | 3 |
| blk.12.hc_ffn_base.weight | 24 |
| blk.12.hc_ffn_fn.weight | 20480×24 |
| blk.12.hc_ffn_scale.weight | 3 |
| blk.13.exp_probs_b.bias | 384 |
| blk.13.exp_probs_b_vl.bias | 384 |
| blk.13.hc_attn_base.weight | 24 |
| blk.13.hc_attn_fn.weight | 20480×24 |
| blk.13.hc_attn_scale.weight | 3 |
| blk.13.hc_ffn_base.weight | 24 |
| blk.13.hc_ffn_fn.weight | 20480×24 |
| blk.13.hc_ffn_scale.weight | 3 |
| blk.14.exp_probs_b.bias | 384 |
| blk.14.exp_probs_b_vl.bias | 384 |
| blk.14.hc_attn_base.weight | 24 |
| blk.14.hc_attn_fn.weight | 20480×24 |
| blk.14.hc_attn_scale.weight | 3 |
| blk.14.hc_ffn_base.weight | 24 |
| blk.14.hc_ffn_fn.weight | 20480×24 |
| blk.14.hc_ffn_scale.weight | 3 |
| blk.14.indexer.attn_k.weight | 512×128 |
| blk.14.indexer.attn_q_b.weight | 1280×4096 |
| blk.14.indexer.k_norm.weight | 128 |
| blk.14.indexer.proj.weight | 5120×32 |
| blk.15.exp_probs_b.bias | 384 |
| blk.15.exp_probs_b_vl.bias | 384 |
| blk.15.hc_attn_base.weight | 24 |
| blk.15.hc_attn_fn.weight | 20480×24 |
| blk.15.hc_attn_scale.weight | 3 |
| blk.15.hc_ffn_base.weight | 24 |
| blk.15.hc_ffn_fn.weight | 20480×24 |
| blk.15.hc_ffn_scale.weight | 3 |
| blk.16.exp_probs_b.bias | 384 |
| blk.16.exp_probs_b_vl.bias | 384 |
| blk.16.hc_attn_base.weight | 24 |
| blk.16.hc_attn_fn.weight | 20480×24 |
| blk.16.hc_attn_scale.weight | 3 |
| blk.16.hc_ffn_base.weight | 24 |
| blk.16.hc_ffn_fn.weight | 20480×24 |
| blk.16.hc_ffn_scale.weight | 3 |
| blk.17.exp_probs_b.bias | 384 |
| blk.17.exp_probs_b_vl.bias | 384 |
| blk.17.hc_attn_base.weight | 24 |
| blk.17.hc_attn_fn.weight | 20480×24 |
| blk.17.hc_attn_scale.weight | 3 |
| blk.17.hc_ffn_base.weight | 24 |
| blk.17.hc_ffn_fn.weight | 20480×24 |
| blk.17.hc_ffn_scale.weight | 3 |
| blk.18.exp_probs_b.bias | 384 |
| blk.18.exp_probs_b_vl.bias | 384 |
| blk.18.hc_attn_base.weight | 24 |
| blk.18.hc_attn_fn.weight | 20480×24 |
| blk.18.hc_attn_scale.weight | 3 |
| blk.18.hc_ffn_base.weight | 24 |
| blk.18.hc_ffn_fn.weight | 20480×24 |
| blk.18.hc_ffn_scale.weight | 3 |
| blk.19.exp_probs_b.bias | 384 |
| blk.19.exp_probs_b_vl.bias | 384 |
| blk.19.hc_attn_base.weight | 24 |
| blk.19.hc_attn_fn.weight | 20480×24 |
| blk.19.hc_attn_scale.weight | 3 |
| blk.19.hc_ffn_base.weight | 24 |
| blk.19.hc_ffn_fn.weight | 20480×24 |
| blk.19.hc_ffn_scale.weight | 3 |
| blk.20.exp_probs_b.bias | 384 |
| blk.20.exp_probs_b_vl.bias | 384 |
| blk.20.hc_attn_base.weight | 24 |
| blk.20.hc_attn_fn.weight | 20480×24 |
| blk.20.hc_attn_scale.weight | 3 |
| blk.20.hc_ffn_base.weight | 24 |
| blk.20.hc_ffn_fn.weight | 20480×24 |
| blk.20.hc_ffn_scale.weight | 3 |
| blk.20.indexer.attn_k.weight | 512×128 |
| blk.20.indexer.attn_q_b.weight | 1280×4096 |
| blk.20.indexer.k_norm.weight | 128 |
| blk.20.indexer.proj.weight | 5120×32 |
| blk.21.exp_probs_b.bias | 384 |
| blk.21.exp_probs_b_vl.bias | 384 |
| blk.21.hc_attn_base.weight | 24 |
| blk.21.hc_attn_fn.weight | 20480×24 |
| blk.21.hc_attn_scale.weight | 3 |
| blk.21.hc_ffn_base.weight | 24 |
| blk.21.hc_ffn_fn.weight | 20480×24 |
| blk.21.hc_ffn_scale.weight | 3 |
| blk.22.exp_probs_b.bias | 384 |
| blk.22.exp_probs_b_vl.bias | 384 |
| blk.22.hc_attn_base.weight | 24 |
| blk.22.hc_attn_fn.weight | 20480×24 |
| blk.22.hc_attn_scale.weight | 3 |
| blk.22.hc_ffn_base.weight | 24 |
| blk.22.hc_ffn_fn.weight | 20480×24 |
| blk.22.hc_ffn_scale.weight | 3 |
| blk.23.exp_probs_b.bias | 384 |
| blk.23.exp_probs_b_vl.bias | 384 |
| blk.23.hc_attn_base.weight | 24 |
| blk.23.hc_attn_fn.weight | 20480×24 |
| blk.23.hc_attn_scale.weight | 3 |
| blk.23.hc_ffn_base.weight | 24 |
| blk.23.hc_ffn_fn.weight | 20480×24 |
| blk.23.hc_ffn_scale.weight | 3 |
| blk.24.exp_probs_b.bias | 384 |
| blk.24.exp_probs_b_vl.bias | 384 |
| blk.24.hc_attn_base.weight | 24 |
| blk.24.hc_attn_fn.weight | 20480×24 |
| blk.24.hc_attn_scale.weight | 3 |
| blk.24.hc_ffn_base.weight | 24 |
| blk.24.hc_ffn_fn.weight | 20480×24 |
| blk.24.hc_ffn_scale.weight | 3 |
| blk.24.indexer.attn_q_b.weight | 1280×4096 |
| blk.24.indexer.proj.weight | 5120×32 |
| blk.25.exp_probs_b.bias | 384 |
| blk.25.exp_probs_b_vl.bias | 384 |
| blk.25.hc_attn_base.weight | 24 |
| blk.25.hc_attn_fn.weight | 20480×24 |
| blk.25.hc_attn_scale.weight | 3 |
| blk.25.hc_ffn_base.weight | 24 |
| blk.25.hc_ffn_fn.weight | 20480×24 |
| blk.25.hc_ffn_scale.weight | 3 |
| blk.26.exp_probs_b.bias | 384 |
| blk.26.exp_probs_b_vl.bias | 384 |
| blk.26.hc_attn_base.weight | 24 |
| blk.26.hc_attn_fn.weight | 20480×24 |
| blk.26.hc_attn_scale.weight | 3 |
| blk.26.hc_ffn_base.weight | 24 |
| blk.26.hc_ffn_fn.weight | 20480×24 |
| blk.26.hc_ffn_scale.weight | 3 |
| blk.27.exp_probs_b.bias | 384 |
| blk.27.exp_probs_b_vl.bias | 384 |
| blk.27.hc_attn_base.weight | 24 |
| blk.27.hc_attn_fn.weight | 20480×24 |
| blk.27.hc_attn_scale.weight | 3 |
| blk.27.hc_ffn_base.weight | 24 |
| blk.27.hc_ffn_fn.weight | 20480×24 |
| blk.27.hc_ffn_scale.weight | 3 |
| blk.28.exp_probs_b.bias | 384 |
| blk.28.exp_probs_b_vl.bias | 384 |
| blk.28.hc_attn_base.weight | 24 |
| blk.28.hc_attn_fn.weight | 20480×24 |
| blk.28.hc_attn_scale.weight | 3 |
| blk.28.hc_ffn_base.weight | 24 |
| blk.28.hc_ffn_fn.weight | 20480×24 |
| blk.28.hc_ffn_scale.weight | 3 |
| blk.28.indexer.attn_q_b.weight | 1280×4096 |
| blk.28.indexer.proj.weight | 5120×32 |
| blk.29.exp_probs_b.bias | 384 |
| blk.29.exp_probs_b_vl.bias | 384 |
| blk.29.hc_attn_base.weight | 24 |
| blk.29.hc_attn_fn.weight | 20480×24 |
| blk.29.hc_attn_scale.weight | 3 |
| blk.29.hc_ffn_base.weight | 24 |
| blk.29.hc_ffn_fn.weight | 20480×24 |
| blk.29.hc_ffn_scale.weight | 3 |
| blk.30.exp_probs_b.bias | 384 |
| blk.30.exp_probs_b_vl.bias | 384 |
| blk.30.hc_attn_base.weight | 24 |
| blk.30.hc_attn_fn.weight | 20480×24 |
| blk.30.hc_attn_scale.weight | 3 |
| blk.30.hc_ffn_base.weight | 24 |
| blk.30.hc_ffn_fn.weight | 20480×24 |
| blk.30.hc_ffn_scale.weight | 3 |
| blk.31.exp_probs_b.bias | 384 |
| blk.31.exp_probs_b_vl.bias | 384 |
| blk.31.hc_attn_base.weight | 24 |
| blk.31.hc_attn_fn.weight | 20480×24 |
| blk.31.hc_attn_scale.weight | 3 |
| blk.31.hc_ffn_base.weight | 24 |
| blk.31.hc_ffn_fn.weight | 20480×24 |
| blk.31.hc_ffn_scale.weight | 3 |
| blk.32.exp_probs_b.bias | 384 |
| blk.32.exp_probs_b_vl.bias | 384 |
| blk.32.hc_attn_base.weight | 24 |
| blk.32.hc_attn_fn.weight | 20480×24 |
| blk.32.hc_attn_scale.weight | 3 |
| blk.32.hc_ffn_base.weight | 24 |
| blk.32.hc_ffn_fn.weight | 20480×24 |
| blk.32.hc_ffn_scale.weight | 3 |
| blk.32.indexer.attn_q_b.weight | 1280×4096 |
| blk.32.indexer.proj.weight | 5120×32 |
| blk.33.exp_probs_b.bias | 384 |
| blk.33.exp_probs_b_vl.bias | 384 |
| blk.33.hc_attn_base.weight | 24 |
| blk.33.hc_attn_fn.weight | 20480×24 |
| blk.33.hc_attn_scale.weight | 3 |
| blk.33.hc_ffn_base.weight | 24 |
| blk.33.hc_ffn_fn.weight | 20480×24 |
| blk.33.hc_ffn_scale.weight | 3 |
| blk.34.exp_probs_b.bias | 384 |
| blk.34.exp_probs_b_vl.bias | 384 |
| blk.34.hc_attn_base.weight | 24 |
| blk.34.hc_attn_fn.weight | 20480×24 |
| blk.34.hc_attn_scale.weight | 3 |
| blk.34.hc_ffn_base.weight | 24 |
| blk.34.hc_ffn_fn.weight | 20480×24 |
| blk.34.hc_ffn_scale.weight | 3 |
| blk.35.exp_probs_b.bias | 384 |
| blk.35.exp_probs_b_vl.bias | 384 |
| blk.35.hc_attn_base.weight | 24 |
| blk.35.hc_attn_fn.weight | 20480×24 |
| blk.35.hc_attn_scale.weight | 3 |
| blk.35.hc_ffn_base.weight | 24 |
| blk.35.hc_ffn_fn.weight | 20480×24 |
| blk.35.hc_ffn_scale.weight | 3 |
| blk.36.exp_probs_b.bias | 384 |
| blk.36.exp_probs_b_vl.bias | 384 |
| blk.36.hc_attn_base.weight | 24 |
| blk.36.hc_attn_fn.weight | 20480×24 |
| blk.36.hc_attn_scale.weight | 3 |
| blk.36.hc_ffn_base.weight | 24 |
| blk.36.hc_ffn_fn.weight | 20480×24 |
| blk.36.hc_ffn_scale.weight | 3 |
| blk.36.indexer.attn_q_b.weight | 1280×4096 |
| blk.36.indexer.proj.weight | 5120×32 |
| blk.37.exp_probs_b.bias | 384 |
| blk.37.exp_probs_b_vl.bias | 384 |
| blk.37.hc_attn_base.weight | 24 |
| blk.37.hc_attn_fn.weight | 20480×24 |
| blk.37.hc_attn_scale.weight | 3 |
| blk.37.hc_ffn_base.weight | 24 |
| blk.37.hc_ffn_fn.weight | 20480×24 |
| blk.37.hc_ffn_scale.weight | 3 |
| blk.38.exp_probs_b.bias | 384 |
| blk.38.exp_probs_b_vl.bias | 384 |
| blk.38.hc_attn_base.weight | 24 |
| blk.38.hc_attn_fn.weight | 20480×24 |
| blk.38.hc_attn_scale.weight | 3 |
| blk.38.hc_ffn_base.weight | 24 |
| blk.38.hc_ffn_fn.weight | 20480×24 |
| blk.38.hc_ffn_scale.weight | 3 |
| blk.39.exp_probs_b.bias | 384 |
| blk.39.exp_probs_b_vl.bias | 384 |
| blk.39.hc_attn_base.weight | 24 |
| blk.39.hc_attn_fn.weight | 20480×24 |
| blk.39.hc_attn_scale.weight | 3 |
| blk.39.hc_ffn_base.weight | 24 |
| blk.39.hc_ffn_fn.weight | 20480×24 |
| blk.39.hc_ffn_scale.weight | 3 |

## per-token active bytes

| key | value |
|---|---|
| architecture | deepseek41 |
| expert_count | Some(384) |
| expert_used_count (top-k) | Some(6) |
| block_count | Some(40) |
| first_k_dense_replace | None |
| moe blocks | 40 |

| quantity | bytes/token | GiB/token | roofline (engramQ8 split) | delta GiB |
|---|---:|---:|---:|---:|
| routed experts | 4,122,869,760 | 3.8397 | 3.7656 | +0.0741 |
| dense | 8,981,420,480 | 8.3646 | 7.132 | +1.2326 |
| dense excl token_embd | 7,657,593,280 | 7.1317 | 7.132 | -0.0003 |
| totals: file | 482,087,534,272 | 448.9790 | 444.23 | — |
| totals: engram table | 209,236,612,640 | 194.8668 | 194.867 | — |
