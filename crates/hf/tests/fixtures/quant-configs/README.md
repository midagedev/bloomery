# Quantization configs

`config.json` and, for ModelOpt exports that ship one, `hf_quant_config.json` of twelve quantized checkpoints, gzipped, for `crates/hf/src/quant_config.rs`. Each file is named `<owner>__<repo>__<file>.gz` and is the file at the revision below, unchanged except where noted.

| checkpoint | revision | producer, scheme |
|---|---|---|
| `nvidia/GLM-5.3-Flash-NVFP4` | `da920bb0b9f4a06727223a349e55468e38352348` | ModelOpt NVFP4 |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | `fc694b54fb0174e0913e6adf86691ef85a4ead47` | ModelOpt MIXED_PRECISION: NVFP4, FP8, FP8_BLOCK_SCALES |
| `nvidia/GLM-5.3-Flash-FP4-Local` | `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` | ModelOpt MIXED_PRECISION: NVFP4 with csf_32x2_v1 scales, W4A16_NVFP4, MXFP8, FP8_PB_WO |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | `b9ce18a6887e18c089fdea8509a33e63429a3300` | ModelOpt MIXED_PRECISION in `config.json` alone: NVFP4 with csf scales, W4A16_NVFP4, MXFP8 |
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` | `c6cc2e8ffa353403213d6b9b18aaf8eb684d705a` | ModelOpt MIXED_PRECISION with Q8_0 and IQ2_XXS (refused); cut, see below |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8` | `3fe6dab75665a93884214ad4b1b95cf02717d081` | ModelOpt FP8 |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` | `bee7596271d1495f6992ae224aefde4410e816b8` | ModelOpt MIXED_PRECISION: FP8 per tensor for the Mamba-2 projections (46 modules), W4A16_NVFP4 for the routed and shared experts and the head (5,935), `kv_cache_quant_algo` FP8; `hf_quant_config.json` unchanged, `config.json` with its `quantization_config` cut to `quant_method`, `quant_algo`, `kv_cache_scheme` and `producer` (the rest is a verbatim copy of `hf_quant_config.json`) |
| `zai-org/GLM-5.3-Flash` | `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` | fp8, 128x128 blocks |
| `Qwen/Qwen3.8-Flash-Next-FP8` | `236dfdf285828023ca3bcd3f37366c58a3469b13` | fp8, 128x128 blocks |
| `deepseek-ai/DeepSeek-V4.1-Flash` | `314113f8a7dadb388215396ba5fbee2c610a04d6` | fp8, 32x32 blocks, MXFP4 experts |
| `openbmb/MiniCPM-V-4.7-35B-A3B-FP8` | `25f6715c91b8e528d45532d4a3cbcb206c59fb47` | compressed-tensors FP8 per channel |
| `openbmb/MiniCPM-V-4.7-35B-A3B-int4` | `1fe0fee8ba1e05a3180fb306b05ccc4a75cc4c86` | compressed-tensors pack-quantized int4, group 128 |
| `MiniMaxAI/MiniMax-M3.1-Flash` | `76faee1b4ac0f412e547781172e94548a6526a88` | `mxfp8` method: MXFP8 dense, `nvfp4-pack-quantized` experts (compressed-tensors tensor names) |

Bolt's two files list 27,921 modules (15 MB); they are cut to the first module of each algorithm (`quantized_layers`) and the first group (`config_groups`), the rest unchanged.

`../checkpoint-headers/glm-5.3-flash-nvfp4-layers-0-3.tensors.json.gz` is the dtype and shape of every tensor of `nvidia/GLM-5.3-Flash-NVFP4` layers 0 and 3 and every unlayered `model.language_model.` tensor and `lm_head.weight`, gathered from all 33 shard headers.

| `mlx-community/Qwen3.6-35B-A3B-4bit` | (the folder on the M4 Pro; no revision recorded) | MLX affine 4-bit g64, 8-bit router and shared-expert gates |
| `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` | `e9675aa3ca5f900ccef55267914466d55ab325fa` | MLX affine 4-bit g64, the `quantization` block names no `mode`, 8-bit routers |
| `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | `6e302ea604ad9ab206367e2c501d1571023e7b6d` | MLX affine 4-bit g64, the `quantization` block names no `mode`, 8-bit routers; its config names `qkv_bias`, `use_qk_norm` and `shared_expert_intermediate_size` |
| `mlx-community/Qwen3.8-27B-4bit` | `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` | MLX affine 4-bit g64, one `quantization` block with no entry of a module; `lmstudio-community/Qwen3.8-27B-MLX-4bit` at `6067b15cf581666a4aecf6af3afaba4bb5efc20c` has the same `config.json` byte for byte |
| `mlx-community/Qwen3.6-27B-4bit` | `c000ac2c2057d94be3fa931000c31723aac53282` | the same quantization; its config lacks the top level's sampling keys (`do_sample`, `temperature`, `top_k`, `top_p`, `generation_config`) |
| `LiquidAI/LFM3-1.7B-MLX-4bit` | `d0ca73dcfa70ce18733754a492cdc1fb0123262d` | MLX affine 4-bit g64, one `quantization` block with no entry of a module, the embedding quantized with the linears; the text model's own `config.json` (`Lfm3ForCausalLM`, `tie_word_embeddings` true, `layer_types` of 13 `conv` and 11 attention layers) |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | `90215b22ec18e72f623dde2ea7af4097025160e2` | MLX affine 4-bit g64, one `quantization` block with no entry of a module, the embedding and the head quantized with the linears; the text model's own `config.json` (`LlamaForCausalLM`, no `head_dim`, `rope_scaling` of type `llama3`, `tie_word_embeddings` false, `eos_token_id` a list of three) |
| `mlx-community/Qwen3-8B-4bit` | `545dc4251c05440727734bcd94334791f6ab0192` | MLX affine 4-bit g64, one `quantization` block with no entry of a module, the embedding and the head quantized with the linears (`Qwen3ForCausalLM`, `head_dim` 128, `rope_scaling` null, `tie_word_embeddings` false, the window keys of full attention) |
| `mlx-community/Qwen3-32B-4bit` | `bcaaf7f538adf166c1080a2befdb4f6019f66639` | the same quantization and keys at the 32B's counts (64 query heads over 8 key heads, 64 layers, `intermediate_size` 25600); `mlx-community/Qwen3-14B-4bit` at `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4` holds the same keys at the 14B's (40 over 8, 40 layers) and is not committed |

`bloomery__qwen3moe-mlx-fixture__config.json.gz` is the `config.json` of the MLX-format fixture folder `tools/mlx-fixture.py make` writes from the qwen3moe GGUF fixture (the Qwen3-30B-A3B config above cut to four layers and their quantization entries, no `mode` key), unchanged.

`bloomery__qwen35moe-mlx-fixture__config.json.gz` is the `config.json` of the MLX-format fixture folder (`tools/mlx-fixture.py make`: the config above cut to four layers, `layer_types` of the fixture's layers and the quantization entries of those layers), unchanged.

**Class-name targets.** `quant_config::plan` takes an explicit module-class resolver; the checkpoint-module
classifier in `model_hf::arch::module_class` supplies the projection and embedding roles read by the GLM and
Qwen3.8 maps. `Linear` and `Embedding` select only their own classes, in both `targets` and `ignore`.
Norms, convolutions and unknown roles do not match a class merely because their stored weights look like matrices.
Module-name and `re:` targets still match their names; an unsupported bare capitalized class name is refused by
name while reading the config. BF16 and F32 tensors remain placed as stored, independently of quantization targets.

