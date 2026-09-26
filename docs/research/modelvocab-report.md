# modelvocab 보고 — 다음 모델들이 요구하는 연산 어휘

2026-09-26, 읽기 전용 조사 라운드 `modelvocab`(opus)의 보고다. 재구성 설계 `docs/rebuild.md` §2-2가 이 보고를 쓴다. 본문은 라운드가 쓴 그대로이고, 머리말과 「증거 원본」 단락만 리드가 고쳤다.

리드가 원본에서 다시 확인한 인용: GLM-5.3-Flash 카드 L25("With 320B total parameters and just 18B active parameters"), config의 층 수 45·`layer_types`·288/8·`swiglu_limit 10.0`·`qk_rope_head_dim 0`·`index_kpool 4`·`hc_mult 4`·vocab 154880, llama.cpp PR #27754(open, 미머지), `crates/gpu-deepseek41/src/router.rs:88,90`의 단언, `crates/gpu/src/flash_gqa.rs`의 `HEAD 128`·`GROUP 8`과 `flash_gqa_prefill.rs:178`, `route_core.rs`의 `Sigmoid` 문장, ik `ggml-cuda/unary.cu`의 SwiGLU limit 규칙과 우리 `experts.rs:19`, `placement.rs`의 `CardFormat::of`·`of_routed`, `roles.rs:39-41`의 nextn `Role::Unused`.

표기: [유도] = 계산한 값, 미확인 = 원문을 보지 못한 사실.

## 요약

1. **"GLM-5.3 Flash"는 실재합니다.** `zai-org/GLM-5.3-Flash`(2026-08-25)이고, 카드에 "With 320B total parameters and just 18B active parameters"라고 적혀 있습니다.
   - 45층 구성은 KDA 선형 어텐션 34층 + DSA k-pool 인덱서가 붙은 NoPE MLA 11층입니다. 모든 블록이 mHC(4 스트림, Sinkhorn 20)로 싸여 있고, expert는 288개 중 8개를 sigmoid 라우터로 고릅니다.
   - **llama.cpp 메인라인에는 머지된 지원이 없습니다.** 열린 PR은 #27752, #27754(unsloth), #27773이고 MTP는 드래프트 PR #27917입니다. unsloth GGUF는 #27754 브랜치에서 나온 파일입니다.
   - ik(`glm5next`, `src/graphs/build_glm5next.cpp`)와 exllamav3(`architecture/glm5_next.py`)는 이미 지원합니다.
2. **이 장비에 들어갑니다.** UD-Q4_K_XL 파일이 199.7 GB입니다. routed expert 약 189.4 GB[유도]는 호스트 214 GB 안에, 비-expert 약 10.3 GB[유도]는 A6000에 올라갑니다. 토큰당 호스트 바이트는 V4.1의 1.27배(Q4_XL), 0.92배(Q3_XL)입니다[유도].
3. **V4.1 스택과 겹치는 부분이 큽니다.** 겹치는 것은 mHC(ik가 V4와 같은 `ggml_hc_pre`를 씀), SwiGLU limit(ik의 규칙이 우리 `experts.rs:19`와 같음), K=V latent 512 어텐션, histogram top-k, 호스트 티어입니다. 새로 필요한 것은 KDA(선형 어텐션 계열 전체), k-pool 인덱서 점수, LayerNorm, NoPE 흡수 MLA의 조합, 라우터 인스턴스 288/8입니다.
4. **Qwen 계열은 세 가지 프로그램 모양으로 나뉩니다.**
   - Qwen3: 전 층 GQA입니다.
   - Qwen3-Next/3.5/3.6/3.8: GDN 3층마다 게이트 GQA 1층입니다.
   - `qwen4exp`(Qwen3.8-Flash-Next): 여기에 QSA 인덱서, gated-residual HC, PLE n-gram이 더해집니다.
   - plan의 `qwen35moe: GQA 16/2 × 256, GDN 30층`은 config로 **확인했습니다**.
   - GROUP 값이 2·4·5·6·8·12·16으로 퍼져 있는데, 오늘 커널은 8만 받습니다.
5. **층 어휘는 벤더 단위가 아니라 연산 단위로 공유됩니다.** delta rule은 Qwen·GLM·Kimi가, MLA+DSA는 DeepSeek·GLM·Hy4가, mHC는 DeepSeek·GLM이, 해시 n-gram 임베딩은 V4.1 engram과 Qwen PLE가 함께 씁니다. 그래서 모델 서술 하나(층별 종류 + 매개변수)와 연산 라이브러리를 단위로 삼는 것이 맞습니다. exllamav3가 바로 이 모양입니다(`glm5_next.py` 469줄이 모듈을 조립).
6. **추천 순위:** ① GLM-5.3-Flash ② Qwen3.6-35B-A3B(Qwen3.5 계열의 운반체) ③ Qwen3.8-Flash-Next ④ gpt-oss-120b. 사용자 순위와 제가 권하는 제작 순서가 다른 이유는 §6에 적었습니다.

---

## 0. 읽은 자료와 찾지 못한 것

**가져온 URL** (2026-09-26, `curl`/`urllib`/`gh api`)
- Hugging Face 원본 config.json(`https://huggingface.co/<repo>/raw/main/config.json`) 약 60개. 대상은 GLM 10종, Qwen 30종, DeepSeek 10종, Kimi 3종, MiniMax 3종, gpt-oss 2종, Mistral 3종, Nemotron 3종, Gemma 2종, Granite 2종, MiMo·Step·Hy4입니다.
- HF API `https://huggingface.co/api/models/<repo>`의 `safetensors.total`과 파일 목록, 조직별 최신 모델 목록(`?author=…&sort=createdAt`).
- 모델 카드 README를 줄 번호로 grep했습니다. GLM-5.3-Flash, GLM-5.3, Qwen3-Next, Qwen3.5-35B-A3B, Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash/V4-Flash/V4-Pro, Kimi-K3/K2.6/Linear, MiniMax-M3입니다.
- GGUF 파일 크기는 HF tree API로 받아 양자화별로 합산했습니다(unsloth, bartowski, ggml-org, antirez 저장소).
- `tokenizer_config.json`과 `chat_template.jinja`에서는 특징 키워드만 뽑았습니다.
- transformers `main`의 원본 소스: `models/glm5_next/modeling_glm5_next.py`(2444줄), `glm_moe_dsa/*`, `qwen4_exp/modeling_qwen4_exp.py`(2746), `qwen3_next/modeling_qwen3_next.py`, `qwen3_5_moe/modeling_qwen3_5_moe.py`.
- `gh api`로 llama.cpp PR을 검색하고 상세를 읽었습니다(#27752, #27754, #27773, #27917).

**로컬 트리** (읽기만 했습니다)
- `~/repo/llama.cpp-fork`, `9e47962ef`(2026-09-14, 메인라인 기반): `src/models/*`, `models.h`, `llama-hparams.h`, `ggml-cuda/fattn.cu`, `gated_delta_net.cu`, 그리고 git 이력의 arch 도입 커밋.
- `~/repo/ik_llama.cpp`, `41b17995`(2026-09-25): `src/graphs/build_glm5next.cpp`, `src/llama-model.h`, `ggml/src/ggml-cuda/unary.cu`.
- `~/repo/mistral.rs`, `b0f26d5cd`(2026-09-17).
- 우리 트리 `a229bfa`: `crates/*`, `docs/research/models-survey.md`(09-24), `docs/research/linear-attn.md`(09-24), `docs/arch-split.md`, 감사 `rebuild.md`.

**박스:** `/home/user/exllamav3-src`, `0740edc`(2026-09-13). `architecture/`, `modules/`, `exllamav3_ext/`를 읽었습니다.

**증거 원본:** 인용한 줄을 모은 작업 파일(`facts.txt`, 100줄)과 적합성 산술(`fit.txt`)을 이 문서 끝 부록 A·B에 원문 그대로 옮겼습니다. config 원문, GGUF 크기 목록, transformers 소스 사본 같은 나머지 작업 파일은 저장소에 싣지 않았습니다.

**출처 약칭**
- `[C]` 해당 저장소의 config.json
- `[A]` HF API total
- `[K Ln]` 카드 README 줄
- `[G]` GGUF tree 합
- `[tf]` transformers 소스
- `[lc f:n]` llama.cpp 포크의 파일:줄
- `[ik f:n]` ik 트리의 파일:줄
- `[S]` `docs/research/models-survey.md`(09-24)

**선행 문서:** `models-survey.md`와 `linear-attn.md`가 GLM-5.3-Flash, GLM-4.7-Flash, V4-Flash, Qwen3/3.5/3.8, GDN/KDA를 이미 원문 근거로 다뤘습니다. 이 보고서는 그 위에 다음 넷을 더합니다.
- 새 모델: Qwen3.8 계열, MiniMax-M3, Kimi-K3/Linear, Nemotron 3.5, Gemma 4, Mistral Small 4, gpt-oss
- 연산 합집합 표
- 참조 엔진의 다중 모델 구조
- 재구성 설계 함의

**찾지 못한 것 (미확인)**
- llama.cpp 도입 PR 번호: `qwen3`/`qwen3moe`/`deepseek32`의 번호, `gemma4`의 최초 번호(처음 보이는 커밋은 수정 PR #21309입니다). `deepseek41`은 vcruz305의 커밋 `bea3b8cd3`이고 PR 번호가 없습니다.
- Qwen3-Omni의 llama.cpp 텍스트 arch 이름. 목록에 없습니다.
- `mistralai/Mistral-Large-3-675B`의 config.json. 저장소에 `params.json`만 있습니다.
- Kimi-K3, Qwen3.8-2.4T, K-EXAONE-2.0의 config 본문. 크기만으로 제외했습니다.
- 라우터 점수 함수: Mistral-Small-4와 Nemotron 3.5는 config에 키가 없습니다.
- 활성 파라미터: GLM-5.x, GLM-4.x, MiniMax-M2.7.
- 텐서별 양자화 타입: MiniMax-M3, Qwen3.5-397B, GLM-5.3 GGUF. 그래서 이 셋의 적합성은 범위로 적었습니다.

---

## 1. 후보군과 적합성

**기준.** 호스트 expert 예산 214 GB(plan (a)), 카드는 A6000 48 GB 한 장입니다. 두 카드 배치는 오늘 엔진이 거부합니다(`body.rs:127-135`, rebuild.md 인용). 호스트 대역은 스펙의 132–136 GB/s 중 136 GB/s로 나눕니다. facts.md:64는 147.7 GB/s를 하한 나눗수로 쓰지만, 이 보고서는 136으로 통일합니다. 파일 크기는 `[G]` 측정값이고 나머지는 [유도]입니다.

| 판정 | 모델 | 근거 한 줄 |
|---|---|---|
| **호스트 티어** | GLM-5.3-Flash | UD-Q4_K_XL 199.7 GB. expert 189.4 GB ≤ 214이고, 비-expert 10.3 GB(`[S]` 표 C: "attn and shexp Q8_0")는 A6000에[유도] |
| 호스트 티어 | Qwen3.8-Flash-Next | UD-Q4_K_XL 111.3 GB = expert 약 78.0 + n-gram 표 약 28.8(IQ4_NL `[S]`) + 나머지 약 4.5 GB[유도]. Q8_0 195.1 GB도 들어감 |
| 호스트 티어 | MiniMax-M3 | UD-Q3_K_XL 194.9 GB에서 expert 180.1–187.1 GB[유도, 카드 몫 7.8–14.8 GB 미확인]. UD-Q4_K_XL 264.9 GB는 안 들어감 |
| 호스트 티어 | Qwen3.5-397B-A17B | UD-IQ4_XS 189.7 GB에서 expert 171.7–180.2 GB[유도, 카드 몫 미확인]. Q4_K_M 244.1 GB는 카드에 expert 12–21 GB를 얹어야 함(경계) |
| 호스트 티어 | MiniMax-M2.7(= M2/M2.1/M2.5 모양) | UD-Q4_K_XL 140.8 GB |
| 호스트 티어 | Step-3.7-Flash / MiMo-V2.6-Flash | UD-Q4_K_XL 122.2 GB / Q2_K 126.2 GB, MXFP4 167.4 GB |
| 호스트 티어 | Qwen3-235B-A22B-2507, GLM-4.7/4.6/4.5, DeepSeek-V4-Flash | `[S]`: Q4_K_M 142.15 / Q3_K_M 171.27(Q4_K_M 216.46은 경계) / UD-Q4_K_XL 155.10. V4-Flash는 plan의 v4port 사슬에 이미 있음 |
| 호스트 티어 | Qwen3-Coder-480B-A35B | Q2_K급이면 약 157 GB[유도, 파일은 안 읽음]. 활성 35B라 느림 |
| A6000 + 호스트 | gpt-oss-120b | MXFP4 65.4 GB. BF16 비-expert 약 4.3 GB와 expert 약 40 GB는 카드에, 약 21 GB는 호스트에[유도] |
| A6000 + 호스트 | Qwen3-Next-80B / Coder-Next, Qwen3.5-122B-A10B, Mistral-Small-4-119B, Nemotron-3-Super-120B-A12B | Q4_K_M 48.5 / 76.5 GB, UD-Q4_K_XL 74.2 GB, BF16 123.6B → Q4급 약 70 GB[유도] |
| 카드 한 장 | Qwen3.6-35B-A3B(= 3.5 모양), Qwen3-30B-A3B(우리)·Coder-30B·VL-30B·Omni-30B, Qwen3 dense 0.6–32B, Qwen3.5/3.6/3.8 dense 0.8–27B | 22.4 GB(UD-Q4_K_XL), 18.56(`[S]`), 27B UD-Q4_K_M 16.46(`[S]`) |
| 카드 한 장 | GLM-4.7-Flash 18.3 GB, Kimi-Linear-48B-A3B 30.1 GB, Nemotron-3.5-Lightning 25.5 GB, Gemma-4-26B-A4B 14.2 GB / 31B, Granite-4.2-30b(dense) | A6000 기준. 3090 24 GB에는 18–22 GB급만 |
| **양자화 한 단계 밖** | GLM-5.3 / 5.2 / 5.1 / 5 (753B) | UD-IQ1_S 216.7 GB, UD-IQ2_M 238.6 GB(expert 218.6–228 GB, 카드에 4.6–14 GB 얹음[유도]), UD-Q2_K_XL 253.9 GB(카드 30–52 GB, 넘칠 수 있음). 2비트 품질과 토큰당 호스트 50–52 ms[유도] |
| 양자화 한 단계 밖 | Hy4-preview 780B(DSA), DeepSeek-V3.1/V3.2 685B | Q2_K 약 256 / 225 GB[유도]. Hy4는 preview, V3.x는 V4에 밀린 세대 |
| 제외 | Qwen3.8-2.4T-A95B(2.45T), Kimi-K3(2.8T), DeepSeek-V4-Pro(1.6T), MiMo-V2.6-Pro(1.02T), Kimi-K2.x(1.03T, INT4) | Q2급에서도 286 GB를 넘음[유도] |
| 제외 | Nemotron-3-Ultra-550B-A55B | Q2 약 184 GB로 RAM에는 들어가지만 활성 55B라 토큰당 호스트 약 133 ms, 약 7 tok/s[유도] |
| 제외 | Mistral-Medium-3.5-128B(dense), Mistral-Large-3-675B(config 없음) | dense 128B는 토큰마다 약 72 GB를 읽어 호스트에서 약 0.5 s/token[유도] |
| 제외 | Llama 4 Scout/Maverick | 2025-04가 마지막이고, meta-llama 조직에 2026년 공개작이 없음 |

---

## 2. 모델별 표

### 2A. 모양, 어텐션, 위치, 정규화

| 모델 | 총 / 활성 | 층(종류) | hidden | 어텐션 | RoPE | 정규화 | 출처 |
|---|---|---|---|---|---|---|---|
| **GLM-5.3-Flash** | 321.3B[A] / 18B("320B total … 18B active" [K L25]) | 45: `layer_types`는 `linear_attention`×3 + `deepseek_sparse_attention`×1 반복. KDA 34, DSA 11(`full_attn_layers [3,7,…,43]`). MTP 1 | 4096 | KDA: `num_heads 64`, `head_dim 128`, `short_conv_kernel_size 4`, `gate_lower_bound -5.0`. MLA: `q_lora_rank 1536`, `kv_lora_rank 512`, `qk_nope_head_dim 256`, `qk_rope_head_dim 0`, `v_head_dim 256`, 64 heads(GGUF는 흡수형 64/1, key_length 512 `[S]`). 인덱서: `index_n_heads 32`, `index_head_dim 128`, `index_topk 2048`, `index_kpool 4`, `index_kpool_always_select_tail true` | 없음. `mla_use_nope true`, "Key change using NoPE" [tf `Glm5NextTextModel.forward`] | RMS `1e-05`, 인덱서 k는 `nn.LayerNorm`, KDA 출력은 gated RMSNorm(sigmoid), q/k는 L2 | C, K, tf |
| GLM-5.3 / 5.2 | 753.3B[A] / 미확인 | 78(dense 3 + MoE 75), 전 층 MLA+DSA, MTP 1 | 6144 | MLA `q_lora 2048`, `kv_lora 512`, nope 192, rope 64, v 256, 64 heads. 인덱서 32×128 top 2048. `indexer_types`는 full×3 뒤 (shared×3, full×1) 반복(`index_topk_freq 4`) | NORM 64(`rope_interleave true`), θ 8e6 | RMS 1e-5 | C |
| GLM-5.1 / 5 | 753.9B[A] | 78 | 6144 | 5.2와 같음. 층마다 자기 인덱서 | θ 1e6, ctx 202752 | 같음 | C |
| GLM-4.7-Flash | 31.2B[A] / 3B(`[S]` 카드 "30B-A3B") | 47(dense 1), MTP 1 | 2048 | MLA `q_lora 768`, `kv_lora 512`, nope 192, rope 64, v 256, 20 heads | NORM 64, θ 1e6 | RMS 1e-5 | C |
| GLM-4.7 / 4.6 / 4.5 | 358.3 / 356.8 / 358.3B[A] | 92(dense 3), MTP 1 | 5120 | GQA 96/8 × 128(GROUP 12), `attention_bias true` | NeoX partial 0.5, θ 1e6 | `use_qk_norm true` | C |
| GLM-4.5-Air | 110.5B[A] | 46(dense 1), MTP 1 | 4096 | GQA 96/8 × 128, bias 있음 | NeoX partial 0.5 | `use_qk_norm false` | C |
| DeepSeek-V4.1-Flash(우리) | 763.2B[A](engram 표 포함). "# Backbone Params 552B", "Activated 8B / 16B" [K L77-78] | 40, MTP 3 | 5120 | latent MQA 64/1 × 512, `q_lora 1280`, `o_lora_rank 1024`/`o_groups 8`, `sliding_window 128`, `compress_ratios` [0×2, 2×18, 1×20, 0×3], `kv_source_layer_ids [2,8,14,20]`, 인덱서 32×128 top 512, candidate blocks | NORM tail 64, YaRN ×16, `compress_rope_theta 160000` | RMS 1e-20 | C, K |
| DeepSeek-V4-Flash / -0731 | 290.9 / 304.2B[A]. "284B parameters (13B activated)" [K L43] | 43 | 4096 | 64/1 × 512. ratio 4(CSA)와 128(HCA)가 교대. 인덱서 64 heads. `num_hash_layers 3` | 같음 | 1e-6 | C |
| DeepSeek-V4-Pro | 1.60T[A]. "1.6T (49B activated)" [K L43] | 61 | 7168 | 128/1, top 1024, `o_groups 16` | 같음 | | C |
| DeepSeek-V3.2 / V3.1 | 685.4 / 684.5B[A] | 61 | 7168 | MLA 128 heads, `q_lora 1536`, `kv_lora 512`, nope 128, rope 64, v 128. V3.2는 DSA 64×128 top 2048 | NORM 64, YaRN ×40 | | C |
| Qwen 계열 | 2C 표 참고 | | | | | | |
| Kimi-Linear-48B-A3B | 49.1B[A]. 카드 "48B / 3B" | 27 = KDA 20 + MLA 7 | 2304 | KDA 32×128, conv 4. MLA 32 heads, `kv_lora 512`, `mla_use_nope true` | 없음 | | C, K |
| Kimi-K2.6 / K2-Thinking | "1T / 32B" [K L56-57] | 61 | 7168 | MLA 64 heads(V3 모양) | YaRN ×64 | | C |
| Kimi-K3 | "2.8T … KDA and Attention Residuals", 활성 104B, "69 KDA + 24 Gated MLA", expert 896 top-16 [K] | 93 | 7168 | | | | K |
| MiniMax-M2.7 | 228.7B[A] | 62, 전부 full(`attn_type_list` 1×62) | 3072 | GQA 48/8 × 128(GROUP 6) | NeoX partial(`rotary_dim 64`), θ 5e6 | q 전체 폭 RMSNorm(`qk_norm_type "per_layer"`, [lc minimax-m2.cpp:30-31]) | C, lc |
| MiniMax-M3 | "~428B … ~23B activated" [K L34] | 60(dense 3) | 6144 | GQA 64/4 × 128(16) + MSA 블록 희소(`sparse_num_index_heads 4`, `sparse_index_dim 128`, `sparse_block_size 128`, `sparse_topk_blocks 16`, `sparse_local_block 1`) | partial 0.5(64), θ 5e6 | `use_gemma_norm true`, QK-norm `per_head` | C |
| gpt-oss-120b / 20b | 116.8 / 20.9B[A] | 36 / 24, `sliding_attention`(128)와 `full_attention` 교대 | 2880 | GQA 64/8 × 64, `attention_bias true`, sinks [lc openai-moe.cpp:44,115] | YaRN ×32, θ 150000 | RMS 1e-5 | C, lc |
| Mistral-Small-4-119B | 119.4B[A] | 36 | 4096 | MLA 32 heads, `q_lora 1024`, **`kv_lora 256`**, nope 64, rope 64, v 128 | YaRN ×128 + `llama_4_scaling_beta 0.1`(어텐션 온도 [lc deepseek2.cpp:40-44]) | | C, lc |
| Nemotron-3.5-Lightning-30B-A3B | 31.6B[A] | 52블록 `layers_block_type`: mamba 23 / moe 23 / attention 6 | 2688 | GQA 32/2 × 128. Mamba2 `mamba_num_heads 64`×64, `ssm_state_size 128`, `n_groups 8` | θ 1e4 | | C |
| Nemotron-3-Super-120B-A12B | 123.6B[A] | 88블록 `hybrid_override_pattern "MEMEMEM*E…"` | 4096 | 같은 종류, Mamba2 128 heads | | | C |
| Gemma-4-26B-A4B / 31B | 25.8 / 31.3B[A] | 30 / 60, sliding(1024)×5 : full×1 | 2816 / 5376 | sliding 16/8 × 256. global은 `global_head_dim 512`, kv 2/4, `attention_k_eq_v true` | full: proportional partial 0.25, θ 1e6 / sliding: θ 1e4 | `final_logit_softcapping 30.0` | C, lc |

### 2B. FFN, MoE, 특수층, 어휘, 지원 상태

| 모델 | MLP | expert 수 / 선택 / 공유, expert ff | 라우터 | dense 선두 | MTP | 특수 | vocab / ctx | 토크나이저·템플릿 | GGUF arch(llama.cpp) | 원 정밀도 / GGUF |
|---|---|---|---|---|---|---|---|---|---|---|
| **GLM-5.3-Flash** | SiLU, `swiglu_limit 10.0` | 288 / 8 / 1, 2048 | sigmoid + bias(`noaux_tc`), `n_group 1`, norm, ×2.5 | 3 | 1(`index_share_for_mtp_iteration`) | mHC 4×20. 최종 합류는 가중치 없는 평균("Unlike DeepSeek-V4, this is an unweighted mean" [tf]) | 154880 / 1048576 | pre `glm4` `[S]`. 템플릿: `<tool_call>`+`<arg_key>`, `<think>`, `reasoning_effort` | **`glm5next`: 메인라인 없음**(#27752, #27754, #27773 열림, #27917 MTP 드래프트). ik는 있음 | FP8 e4m3 128×128. UD-IQ2_XXS 101.8, UD-Q3_K_XL 147.5, UD-Q4_K_XL 199.7, Q8_0 341.0 GB |
| GLM-5.3 / 5.2 | SiLU | 256 / 8 / 1, 2048 | 같음, ×2.5 | 3 | 1 | 층 사이 top-k 공유 | 154880 / 1M | glm4 | `glm-dsa` #19460(인덱서 #25407, MTP 스펙 디코딩 #25980) | FP8(5.3) / BF16(5.2) |
| GLM-4.7-Flash | SiLU | 64 / 4 / 1, 1536 | sigmoid + bias, ×1.8 | 1 | 1(GGUF에는 없음 `[S]`) | | 154880 / 202752 | glm4, `enable_thinking` | `deepseek2`(변환기가 매핑 `[S]`) | Q4_K_M 18.3 GB |
| GLM-4.x | SiLU | 160 / 8 / 1, 1536 | sigmoid(기본값 [lc glm4-moe.cpp:15-17]) + `exp_probs_b`(:82), ×2.5 | 3 | 1 | | 151552 | glm4 | `glm4moe` #14939 | BF16 |
| DeepSeek-V4.1-Flash | SiLU, limit 10 | 384 / 6 / 1, 2304 | sqrt-softplus + bias, ×1.5 | 0 | 3(우리는 `Role::Unused`) | mHC, engram(층 1·14), DSpark | 129280 / 1M | `joyai-llm` | `deepseek41`(vcruz305 커밋, PR 없음) | FP8 32×32 + FP4 expert. antirez Q2 365.7 GB |
| DeepSeek-V4-Flash | SiLU, limit | 256 / 6 / 1 | sqrt-softplus, 해시 3층 | 0 | 1 | mHC | 129280 | joyai-llm | `deepseek4` #24162 | FP8 + FP4 |
| Kimi-Linear | SiLU | 256 / 8 / 1, 1024 | sigmoid, renorm, ×2.446 | 1 | 0 | | 163840 / 1M | TikToken | `kimi-linear` #18755 | Q4_K_M 30.1 GB |
| MiniMax-M2.7 | SiLU | 256 / 8 / 0, 1536 | sigmoid + `use_routing_bias` | 0 | 3 modules | | 200064 / 204800 | `<minimax:tool_call>`, `<think>` | `minimax-m2` #16831 | FP8. UD-Q4_K_XL 140.8 GB |
| MiniMax-M3 | `swigluoai`(`swiglu_alpha 1.702`, limit 7) | 128 / 4 / 1(3072) | sigmoid + bias, ×2.0 | 3 | 7 modules / nextn 1 | 비전 | 200064 / 1M | minimax 도구 태그 | `minimax-m3` #24908 | BF16. UD-Q3_K_XL 194.9 GB |
| gpt-oss-120b | SWIGLU_OAI(limit 7) [lc openai-moe.cpp:141] | 128 / 4 / 0, 2880 | `SOFTMAX_WEIGHT`(선택된 것만 softmax, :143) | 0 | | sinks | 201088 / 131072 | harmony `<\|channel\|>`, `reasoning_effort` | `gpt-oss` #15091 | MXFP4 expert. 62.6–65.4 GB |
| Mistral-Small-4 | SiLU | 128 / 4 / 1, 2048 | 점수 함수 미확인, norm, ×1.0 | 0 | (EAGLE 저장소가 따로 있음) | 비전 | 131072 / 1M | `reasoning_effort` | `mistral4` #20649(deepseek2 그래프 재사용, 6줄) | FP8. UD-Q4_K_XL 74.2 GB |
| Nemotron-3.5-Lightning | relu² [lc nemotron-h-moe.cpp:111] | 128 / 6 / 1(3712), 1856 | 점수 함수 미확인, norm, ×2.5 | | 1 | Mamba2 | 131072 / 262144 | `<think>`, xml 도구 | `nemotron_h_moe` #18058 | UD-Q4_K_XL 25.5 GB |
| Nemotron-3-Super | relu² | 512 / 22 / 1, `moe_latent_size 1024` | ×5.0 | | `mtp_hybrid_override_pattern "*E"` | latent MoE | 131072 | | nemotron_h_moe | BF16 |
| Gemma-4-26B-A4B | GELU-tanh(gated) | 128 / 8 + dense 병렬 2112 [lc gemma4.cpp:304,323] | 라우터 입력 스케일 `ffn_gate_inp_s`(:319) | | assistant GGUF | softcap 30 | 262144 | GemmaTokenizer | `gemma4`(최초 PR 미확인) | UD-Q4_K_XL 14.2 GB |

### 2C. Qwen 계열, 변형별 (우리 Qwen3-30B-A3B 대비)

우리 Qwen3-30B-A3B의 기준값은 `HEAD 128`, `GROUP 8`, `N_EXPERT 128`/`N_USED 8`, `NORM_K 2048`(= hidden), 머리별 QK-norm, NeoX full 128(θ 1e6)입니다. 아래 표의 "매개변수"는 값만 다른 것이고, "새 연산·층 종류"는 커널이나 층 종류가 새로 필요한 것입니다.

| 변형 [C] | GGUF arch | 매개변수 차이 | 새 연산·층 종류 |
|---|---|---|---|
| Qwen3 dense 0.6/1.7/4/8/14/32B | `qwen3` | hidden 1024/2048/2560/4096/5120/5120. GROUP 16:8=**2**, 2, 32:8=**4**, 4, 40:8=**5**, 64:8=8. ff 3072…25600. 0.6–4B는 임베딩 tied. ctx 40960 | dense FFN(MoE 없음). **GROUP 2/4/5**(오늘 커널은 8 고정) |
| Qwen3-30B-A3B-2507 / Coder-30B-A3B | `qwen3moe` | θ 1e7, ctx 262144 | 없음 |
| Qwen3-235B-A22B-2507 | `qwen3moe` | hidden 4096, 94층, 64/4 → **GROUP 16**, ff 1536, θ 5e6 | 없음(GROUP만 다름) |
| Qwen3-Coder-480B-A35B | `qwen3moe` | 6144, 62층, 96/8 → **GROUP 12**, expert 160/8, ff 2560 | 없음 |
| Qwen3-VL-30B-A3B / Omni-30B-A3B | `qwen3vlmoe` #16780 / Omni는 미확인 | 30B-A3B와 같은 모양. θ 5e6 / 1e6, Omni vocab 152064 | M-RoPE 인터리브 섹션 [24,20,20]. **deepstack**: 비전 특징을 앞쪽 N개 층의 잔차에 더함 [lc qwen3vlmoe.cpp:164-167]. 텍스트만 넣으면 섹션이 한 값으로 접힘(`[S]` 범례). 새 mixer가 아니라 층별 가산 연산 둘 |
| Qwen3-Next-80B-A3B / Coder-Next | `qwen3next` #16095 | 48층 = 12 × (GDN×3 + gated attn×1) [K L45]. GQA 16/2 × **256**. expert 512/**10**, ff 512 + 공유 512. θ 1e7(Coder 5e6) | **GDN**(k16/v32 × 128, conv 4), 어텐션 **출력 sigmoid 게이트** [tf qwen3_next:308], **head 256**, NeoX partial 0.25, **공유 expert sigmoid 게이트** [tf :853-862]. GDN k-head 매핑은 grouped(`linear-attn.md` 발견 5) |
| **Qwen3.5 / 3.6-35B-A3B**(AgentWorld-35B-A3B도 같은 모양) | `qwen35moe` #19468 | hidden 2048, 40층(**GDN 30 + GQA 10**), GQA **16/2 × 256**, GDN v32, 256/8 + 공유 512, vocab **248320**, MTP 1 | GDN, 출력 게이트 [tf qwen3_5_moe:822], **IMROPE [11,11,10], 64/256 회전**, 공유 게이트 [:910-919]. k-head 매핑은 tiled. **plan의 qwen35moe 서술 확인** |
| Qwen3.5-122B-A10B / 397B-A17B | `qwen35moe` | 3072·48층·32/2(**16**)·v64·256/8 / 4096·60층·32/2·v64·**512/10**. ff 1024 + 공유 1024 | 위와 같음 |
| Qwen3.5 dense 27B(= 3.6/3.8-27B) / 9 / 4 / 2 / 0.8B | `qwen35` | 27B: 5120·64층·24/4 → **GROUP 6**·v48·ff 17408. 9B: 16/4(4). 4B·2B·0.8B는 tied. 3.6/3.8-27B의 `output_gate_type "swish"`는 SiLU로 Qwen3.5 기본과 같음(`linear-attn.md:108`) | dense 변형의 GDN + gated GQA. GROUP 6 |
| **Qwen3.8-Flash-Next** | `qwen4exp` #27742 | 2560, 48층 = 12 × (GDN×3 + QSA×1), GQA 24/2 → **GROUP 12**, GDN v48, **512/10** + 공유 640 | + **QSA 블록 인덱서**(4q+1k × 128, 4토큰 mean-pool, RMSNorm, 블록 시작 위치로 rope, budget 2048) [tf qwen4exp:673-781]. + **gated-residual HC**(4 branches, low-rank 320, 원소별 sigmoid 읽기 게이트, Sinkhorn 없음) [:1003-1040]. + **PLE n-gram**(층 1, 2·3-gram 해시, 헤드 16개에 소수 크기 표, 기본 20M, dilated conv) [:1080-1257]. GDN 게이트 sigmoid. 카드 L26: "the architecture that will underpin Qwen4" |
| Qwen3.8-2.4T-A95B | 미확인 | 2.45T | 제외 |

---

## 3. 연산 어휘 표 (연산 → 쓰는 모델 → 우리 트리 상태)

상태는 셋으로 나눴습니다. **있음**은 쓸 수 있는 커널이 있다는 뜻입니다. **고정**은 커널은 있지만 한 모델의 상수가 박혀 있다는 뜻입니다. **없음**은 트리를 grep해서 0건이었다는 뜻입니다. 예를 들어 `softcap`, `conv1d`, `delta_rule`, `mamba`, `gelu`, `relu2`, `fp8`, `nvfp4`, `e4m3`는 엔진 크레이트에서 0건이었습니다.

**A. 어텐션과 KV 배치**

| 연산 | 모델 | 우리 상태 |
|---|---|---|
| GQA flash, head 128, GROUP 8(디코드 split-K + 프리필) | Qwen3 MoE 30B·Coder-30B·VL·Omni, Qwen3-32B | **고정**: `crates/gpu/src/flash_gqa.rs:67,69` `HEAD 128`/`GROUP 8`, `:79` `THREADS = GROUP*32`, `flash_gqa_prefill.rs:178` `assert!(GROUP == 8 …)` (`gqa_flash_seg(_mma)`, `gqa_flash_merge`, `gqa_prefill_flash`) |
| GQA, GROUP 2/4/5/6/12/16 | Qwen3 dense, 235B, Coder-480B, Qwen3.5 dense, Flash-Next, GLM-4.x, MiniMax, Nemotron, Mistral-Medium | 없음(상수 때문) |
| GQA head 256 / 64 / 512(K=V) | Qwen3-Next·3.5·3.8 full, Gemma sliding / gpt-oss / Gemma global | 없음 |
| 어텐션 출력 sigmoid 게이트 | Qwen3-Next, 3.5/3.6/3.8, Flash-Next | 없음 |
| QKV bias | GLM-4.x, gpt-oss | 없음 |
| GQA 위 sliding window(층 패턴 포함) | gpt-oss 128(1:1), Gemma 1024(5:1), Step-3.7 512, MiMo 128 | 없음. V4.1 latent 경로의 window ring만 있음(`gpu-deepseek41/src/attn.rs`) |
| attention sinks | gpt-oss, V4/V4.1 | V4.1 latent 경로만: `ds41_attn_merge`("Each head's learned sink joins the softmax denominator", attn.rs 머리말) |
| 흡수형 MLA + rope(행 576) | V2-Lite, GLM-4.7-Flash(20 heads), V3.x, Kimi-K2, Mistral-4(**LATENT 256**) | LATENT 512는 있음: `crates/gpu/src/flash.rs` `flash_latent*`, 흡수 gemv는 `model/kernels.rs` `q8_0_gemv_heads`/`q3k_gemv_heads`. LATENT 256은 없음. 20 heads와 MMA 16행 타일 문제는 `[S]` |
| 흡수형 NoPE MLA(K=V latent 512) | GLM-5.3-Flash 11층, Kimi-Linear 7층 | 변형으로 가능[유도]: V4.1의 K=V latent 어텐션 `ds41_attn_seg`/`_sel`에서 window와 sink를 끔. 흡수 gemv는 V2-Lite 것. 흡수 산술로 유도한 대응이라 게이트로 확인해야 함 |
| V4 latent MQA + window ring ⧺ 압축 행 + grouped o_lora | V4-Flash/Pro, V4.1 | V4.1은 있음(`gpu-deepseek41` attn/compress/chain). V4 자체 압축기(APE, HCA 128)는 부분(plan v4port) |
| 층 사이 KV / 인덱스 / top-k 공유 | V4.1(`kv_source`/`index_source`), GLM-5.x(`indexer_types` shared) | V4.1은 있음(공유 압축 스트림, `arch-split.md:41`) |
| DSA 토큰 인덱서(LayerNorm k, top 2048) | V3.2, GLM-5.x, Hy4 | 부분: `ds41_indexer_score`/`ds41_indexer_topk`(relu 가중 헤드 합, histogram 정확 top-k, 개수는 디바이스 워드). Hadamard와 rope가 몸체에 박혀 있음. `indexer.rs:65` `HEADS 32`(V3.2는 64). LayerNorm은 없음 |
| DSA k-pool 인덱서(softmax(gate+APE)로 4키 풀) | GLM-5.3-Flash | 점수와 풀링은 없음. top-k는 재사용 가능 |
| QSA 블록 인덱서 / MSA 블록 희소 | Qwen3.8-Flash-Next / MiniMax-M3 | 없음 |
| GDN(머리당 스칼라 감쇠) | Qwen3-Next, 3.5/3.6/3.8, Flash-Next | 없음 |
| KDA(채널별 감쇠, 하한 −5) | GLM-5.3-Flash, Kimi-Linear, Kimi-K3 | 없음. 메인라인은 GDN과 KDA를 커널 하나(`template<int S_v, bool KDA, …>` [lc gated_delta_net.cu:4])로 처리. exllamav3도 한 모듈에 "KDA mode"(`modules/gated_delta_net.py:380`) |
| Mamba2 SSM | Nemotron-3/3.5 | 없음 |
| depthwise causal conv1d(k 4) + SiLU | GDN, KDA, Mamba2, PLE(dilated) | 없음 |

**B. 위치 부호화**

| 연산 | 모델 | 상태 |
|---|---|---|
| NeoX full 128 | Qwen3 | **고정**: `crates/gpu/src/rope_neox.rs:46` `HEAD 128`, 머리별 RMS와 융합(`head_norm_neox_append`) |
| NeoX partial(0.5 / 64 of 128 / 0.25) | GLM-4.x, MiniMax, Qwen3-Next | 없음 |
| IMROPE partial 64/256 [11,11,10], M-RoPE [24,20,20] | Qwen3.5+, VL/Omni | 없음. 텍스트만이면 부분 NeoX와 같은 값으로 접힌다는 것이 `[S]` 범례의 가설이고, 비트 동일 여부는 게이트가 판정 |
| NORM tail 64 + YaRN | V2-Lite, V3.x, V4/V4.1, GLM-4.7-Flash, GLM-5.x, Kimi-K2, Mistral-4 | 있음: `gpu-deepseek41/src/rope.rs`(`ds41_rope_tail`, YaRN 표), `crates/gpu/src/fused.rs` `kv_norm_rope_append` |
| NoPE | GLM-5.3-Flash, Kimi-Linear | 있음(rope 생략) |
| 이중 θ / 층별 θ / proportional / llama3 스케일링 / 어텐션 온도 | Gemma 4, Step-3.7, Mistral 4 | 없음 |

**C. 정규화**

| 연산 | 모델 | 상태 |
|---|---|---|
| RMSNorm | 전부 | 있음: `crates/gpu/src/elem.rs` `rms_norm`, 융합판 `norm_quant` |
| 머리별 QK RMSNorm | Qwen3 전 세대, GLM-4.5/4.6/4.7, MiniMax-M3, Gemma | head 128 한정으로 있음(`head_norm_neox_append`) |
| q 전체 폭 RMSNorm | MiniMax-M2.x | 없음 |
| LayerNorm(bias 포함) | DSA 인덱서 k(GLM-5.x, V3.2) | 없음 |
| gated RMSNorm(sigmoid·SiLU), L2 norm | GDN, KDA | 없음 |
| gemma norm (1+w), grouped RMSNorm | MiniMax-M3·Gemma / Flash-Next HC | 없음 |

**D. MLP와 활성함수**

| 연산 | 모델 | 상태 |
|---|---|---|
| SwiGLU | 대부분 | 있음: `elem.rs` `swiglu`, `gate_up_swiglu_q3k`, `qwen3moe_gate_up_swiglu_q4k`, `gemm_swiglu_quant` |
| **SwiGLU limit, SiLU 뒤 클램프**: `min(silu(g), L) · clamp(u, −L, L)` | V4/V4.1, **ik 아래의 GLM-5.3-Flash** | 있음: `gpu-deepseek41/src/experts.rs:19`, `dense.rs:603`. ik도 같은 규칙입니다: `ggml-cuda/unary.cu:80-82` "g = min(g, limit); dst = g * max(-limit, min(limit, y))". 이 규칙이 dsv4 계열과 GLM5NEXT에 함께 적용됩니다(`src/llama-model.h:671-675`). **transformers `Glm5NextTextMLP`는 SiLU 앞에서 클램프**합니다(`gate.clamp(max=limit)`). 원소당 최대 차이는 L·σ(−L) = 10 × 4.54e-5 = 4.5e-4입니다[유도]. 그래서 ik 오라클 게이트는 통과하고, transformers 기준 검사는 통과하지 못할 수 있습니다. AGENTS의 Performance-first 규칙대로 엔진은 ik 규칙 하나만 싣습니다 |
| swigluoai(α 1.702, limit 7, up+1) | gpt-oss, MiniMax-M3 | 없음 |
| relu²(게이트 없음) / GELU-tanh gated | Nemotron-H / Gemma 4 | 없음 |

**E. MoE 라우터와 합산**

| 연산 | 모델 | 상태 |
|---|---|---|
| softmax → top-k → renorm | Qwen3 전 세대(3.5/3.8 포함: [tf qwen3_5_moe:896], qwen4exp `Qwen4ExpTextTopKRouter`), V2-Lite | **고정**: 128/8 `crates/gpu/src/arch/qwen3moe/router.rs:72,75`(+`MAX_TOKENS 8` :79, `NORM_K 2048` :135), 64/6 `crates/gpu/src/router.rs:38,41` |
| 선택된 것만 softmax | gpt-oss | 변형. 위 규칙과 수학적으로 같음[유도]: exp(lᵢ)/Σ_topk exp(lⱼ) |
| sigmoid + bias(noaux) + norm + scale | GLM 전 세대, V3.x, Kimi, MiniMax, Hy4, MiMo | 점수 함수만 있음: `crates/gpu/src/route_core.rs:14-16`("[`Sigmoid`] is compiled but no kernel routes with it yet") |
| sqrt-softplus + bias | V4/V4.1 | **고정**: 384/6 `gpu-deepseek41/src/router.rs:60,63`, `PER_LANE = N_EXPERT/32`(:72) |
| group 제한(8/4) | V3.x | 없음 |
| 해시 라우팅(`tid2eid`) | V4-Flash/Pro 앞 3층 | 부분: 배치 역할만 있음(`crates/model/src/placement.rs:58`), 커널은 없음 |
| 공유 expert(일반) / sigmoid 게이트 달린 공유 expert | GLM·V3/V4·Kimi·MiniMax-M3·Mistral4·Nemotron / Qwen3-Next·3.5·3.8 | `ds41_shexp_gate_up(_q3k)`(`dense.rs:621`)는 있음. 게이트형은 없음 |
| latent MoE / dense와 MoE 병렬 | Nemotron-3-Super·Kimi-K3 / Gemma-4-26B | 없음 |
| (E, K) 인스턴스 | 필요한 조합: 32/4, 64/4, 64/6, 128/4, 128/6, 128/8, 160/8, 256/6, 256/8, **288/8**, 384/6, 384/8, 512/10, 512/22 | 있는 조합: 64/6, 128/8, 384/6, 128/3(드래프트 `experts_mxfp4.rs:63,66`) |

**F. 특수층**

| 연산 | 모델 | 상태 |
|---|---|---|
| MTP / nextn | GLM 전 세대(1), Qwen3.5+(1), V4.1(3), MiniMax(3/7), Nemotron(1), MiMo·Step(3) | 없음. V4.1의 nextn은 `Role::Unused`(`crates/model/src/arch/deepseek41/roles.rs:39-41`) |
| 외부 블록 드래프트(DSpark/DFlash) | V4.1. 공개된 것: Qwen3.6(RedHatAI dspark, z-lab DFlash), GLM-5.3-Flash(`RedHatAI/GLM-5.3-Flash-speculator.dspark-preview`, DFlash2 GGUF), MiniMax-M3(`nvidia/MiniMax-M3-DSpark`), Nemotron 3.5, Qwen3 dense(`deepseek-ai/dspark_qwen3_*`) | V4.1 DSpark만: `gpu-deepseek41/src/draft/`, `dflash_*` 128/3 |
| n-gram lookup 드래프트 | 모델 무관 | 있음: `crates/gpu-gates/src/draft.rs:14` `Lookup` |
| mHC Sinkhorn | V4/V4.1, GLM-5.3-Flash | 있음: `gpu-deepseek41/src/hc.rs`. `ds41_hc_pre`는 Q3_K gemv와 융합(`HC_PIECE 512`, :71), f32 판은 `hc_f32.rs`, 평균 합류는 `ds41_hc_mean`. GLM의 hc fn은 Q8_0(`[S]`) |
| gated-residual HC | Flash-Next | 없음 |
| engram / PLE n-gram | V4.1 / Flash-Next(Gemma E 계열의 층별 임베딩도 비슷) | engram은 있음(`crates/engram`, `engram_gate.rs`는 **`ROW 5120` 고정** :57). PLE는 없음 |
| 최종 logit softcap / deepstack | Gemma 4 / Qwen3-VL·Omni | 없음 |
| 비전 타워 | V4.1, GLM-5.3-Flash, Qwen3.5+, MiniMax-M3, Gemma 4 | V4.1만(`crates/vision`, `crates/gpu-vision`) |

**G. 포맷, 토크나이저, 템플릿**

| 항목 | 상태 |
|---|---|
| Q3_K·Q4_K·Q6_K·Q5_0·Q5_1·Q8_0 | 카드: routed와 dense 모두 가능. Q5_K는 dense만(`placement.rs:363-368` `of_routed`). 호스트 qdot은 전부(`crates/qdot/src/lib.rs:1-3`) |
| Q2_K, IQ2_XS, IQ3_XXS, IQ4_XS | 카드 커널은 있음(`crates/gpu/src/iq.rs` `q2_k_rows`, `iq2_xs_rows`, `iq3_xxs_rows`, `iq4_xs_rows`). 그런데 **배치가 거부**합니다(`placement.rs:282-290`). 호스트는 IQ3_XXS만 |
| IQ1_S/M, IQ2_XXS/S/M, IQ3_S, IQ4_NL(Flash-Next PLE 표) | 없음 |
| MXFP4 | 카드 `crates/gpu/src/mxfp4.rs`, 드래프트 전용 `experts_mxfp4.rs`, 호스트 qdot MXFP4×Q8_2_X4. 배치는 거부(`placement.rs:281`) |
| FP8 e4m3, NVFP4 | 없음 |
| pre-tokenizer | `deepseek-v3`/`hunyuan-dense`/`joyai-llm`, `qwen2`만(`crates/tokenizer/src/pretok.rs:202-214`). `glm4`와 `qwen35`가 없음 |
| 도구 호출 / 추론 분리 | DeepSeek DSML(`crates/serve/src/dsml.rs`)과 think 분리(`reasoning.rs`)만. GLM `<arg_key>`, Qwen xml, harmony, MiniMax 형식은 없음 |

---

## 4. 참조 엔진의 다중 모델 구조 (실측)

| 엔진 | 모델 하나를 더하는 단위 | 실측 크기(`wc -l` / `git show --stat`) | 공유되는 것 | 모델 상수가 커널에 들어가는 길 |
|---|---|---|---|---|
| llama.cpp(포크 `9e47962ef`) | `src/models/<arch>.cpp` 클래스 하나. `load_arch_hparams`, `load_arch_tensors`, `build_arch_graph`를 가짐([lc qwen35moe.cpp:4,36,149], MTP 그래프 :551). 여기에 변환기, gguf 상수, 새 위상이면 memory/kv-cache가 붙음 | qwen3moe 179, qwen35moe 741, qwen3next 822, qwen4exp 1283, glm-dsa 769, deepseek41 948(+deepseek4 1521), kimi-linear 562, minimax-m3 603, openai-moe 177. PR diffstat: #27742 qwen4exp는 28파일 +2881(`llama-memory-hybrid-idx` +621), #19468 +2096, #18755 +1518, #24908 +1044. GLM-5.3-Flash PR은 +2395(`glm5next.cpp` +1261) | ggml 연산 전부, `llm_graph_context` 헬퍼, `delta-net-base.cpp`(qwen3next·qwen35·qwen35moe·qwen4exp·kimi-linear·kimi-k3·bailingmoe3), `mamba-base.cpp`. 상속: `deepseek41 : deepseek4`(models.h:1363), `mistral4 : deepseek2`(:1492, 6줄) | 런타임 인자(텐서 모양)와 **템플릿 인스턴스 표**. FA는 (DKQ, DV) ∈ {40…576/512} × (ncols1, ncols2 = GQA 묶음 1…32)이고 `gqa_ratio % k`로 고름(`fattn.cu:201-306`). `gated_delta_net.cu`는 `template<int S_v, bool KDA, bool keep_rs_t>` + S_v 16/32/64/128 switch. mmq는 타입별 22개. hparams는 층별 배열(`llama-hparams.h:98-105`, `is_swa_impl` :183, `is_recr_impl` :186, `is_indexer_full_impl` :285)인데 모델 전용 필드가 쌓였음(`dsv4_compress_ratios` :295, `dsv41_*_source` :313-315, `swiglu_clamp_*` :378-379) |
| mistral.rs(`b0f26d5cd`) | `models/<m>.rs` 또는 `vision_models/<m>/` + `pipeline/loaders/normal_loaders.rs`의 로더 impl 셋(Normal·Isq·DeviceMapped, 약 240줄; `Qwen3NextLoader` :5693-5934) | qwen3_moe 835, qwen3_next 1380, glm4_moe_lite 1136, gpt_oss 1324. `vision_models/qwen3_5/`는 5,155(text 3137, mtp 210, speculative 1298), `qwen3_5_moe/`는 1,832 | candle 연산, `mistralrs-quant`의 `QuantMethod` 트레이트(lib.rs:1786: gguf·fp8·mxfp4·nvfp4·gptq…), CUDA `gdn.cu` 6143, moe gemm/gemv | 런타임 인자. 템플릿은 상태 타입과 타일(`template<StateT, BK, BV>`)로 잡고 모델로 잡지 않음 |
| exllamav3(박스 `0740edc`) | `architecture/<m>.py` 선언형 조립 파일 하나(+ `_mtp.py`) | qwen3_moe 204, qwen3 194, **qwen3_5 575**(dense·MoE·VL·VL-MoE 네 클래스가 한 파일), qwen4_exp 328, **glm5_next 469**(+mtp 238), deepseek_v4 322(+mtp 368), gpt_oss 229. modules 16,628(gated_delta_net 1336, mla_attn 1128, dsv4 1779, hyperconnections 591, qsa_indexer 671, ngram_embedding 556, ple 389, mamba2 721, block_sparse_mlp 1513 + cpu 711) | 모듈 라이브러리 전부. 아키텍처는 config를 읽고 층별로 모듈을 고름: `layer_types[idx]`에 따라 `GatedDeltaNet(key_f_a…, gate_lower_bound)` 또는 `MLAttention(indexer_mode, index_kpool…)`. 여기에 `HyperConnection`, `GatedMLP`/`BlockSparseMLP(act_limit, routed_scaling_factor, n_group, topk_group)` | 개념별 CUDA ext(`routing.cu` `template<int ACT>`, `gdn.cu`, `hc_mix.cu`, `dsa_topk.cu`, `ngram.cu`, `ple.cu`, `softcap.cu`), Triton JIT(dsa/qsa/mla 어텐션, conv1d), GEMM은 `select_gemm_shape(cc, m, k, n, bits)` 표(`exl3_kernel_map.cu:23`) |
| ik(`41b17995`) | `src/graphs/build_<m>.cpp` + hparams/load-tensors의 case | `build_glm5next.cpp` 550 | V4 hc 연산(`ggml_hc_pre`, `build_mhc_post`), `llama-delta-net.cpp`/`llama-kda.cpp`, swiglu limit 규칙 하나 | ggml와 같음 |
| **우리(`a229bfa`)** | 크레이트나 디렉터리 + 전용 커널 + 게이트 bin | V4.1: `gpu-deepseek41` 29,742 + `model/arch/deepseek41` 5,438 = **35,180**(게이트 bin 28). Qwen3 MoE: `gpu/arch/qwen3moe` 5,376 + `model/arch/qwen3moe` 941 + Qwen 모양 공용(`flash_gqa*`, `rope_neox`, `gemm`) 4,480 = **약 10,797**(bin 8) | 행 gemv/GEMM, elem, fault, graph, 호스트 티어 | Rust `const`가 몸체와 **커널 이름**에 박힘(`N_EXPERT 384`, `HEAD 128`/`GROUP 8`, `ROW 5120`, `qwen3moe_*`). AOT PTX(cuda-oxide) |

**비교할 때의 단서.** llama.cpp와 exllamav3의 줄 수에는 커널이 빠져 있습니다(generic ggml·ext). 반대로 우리 줄 수에는 게이트 전용 표면이 들어 있습니다(rebuild.md DS3: `Body`의 `pub fn` 48개 중 약 20개가 게이트 전용). 두 방향으로 모두 보정해도 모델 하나에 드는 코드가 수십 배라는 비는 남습니다. V4.1은 35k줄 대 948(+1521) / 322(+1779 모듈), Qwen3 MoE는 약 10.8k 대 179 / 204입니다. 연산 라이브러리를 만들어야 하는 근거가 이 비입니다.

---

## 5. 설계 함의

### (0) 계열 문제: `qwen` 서술 하나가 변형을 덮는 방법

- **참조 엔진이 선을 긋는 곳.**
  - llama.cpp는 프로그램 모양마다 arch 이름을 따로 둡니다(`qwen3`, `qwen3moe`, `qwen3vl`, `qwen3vlmoe`, `qwen3next`, `qwen35`, `qwen35moe`, `qwen4exp`). 클래스도 모두 `llama_model_base`에서 따로 나옵니다(models.h:605, 618, 644, 2369, 2414, 2460, 2565). 공유는 `delta-net-base.cpp` 같은 층 부품으로만 합니다.
  - mistral.rs도 모양마다 파일이 하나씩입니다.
  - exllamav3는 `qwen3_5.py` 한 파일(575줄)에 dense, MoE, VL, VL-MoE를 매개변수로 담습니다. 사용자가 원하는 모양에 이것이 가장 가깝습니다.
- **권고.** 모든 모델이 쓰는 `ModelSpec` 타입을 하나 둡니다(b). 모델 계열이 따로 갖는 것은 GGUF 메타데이터 방언을 `ModelSpec`으로 옮기는 **리더**뿐입니다.
  - `qwen` 리더가 arch 문자열 여덟 개를 받고, 층 종류는 메타데이터와 텐서 존재로 정합니다. `arch-split.md`의 원칙("LayerKind는 텐서 존재로", "커널은 모델을 모른다")과 같습니다.
  - 같은 식으로 `glm` 리더는 `glm4moe`, `glm-dsa`, `glm5next`를 받습니다. GLM-4.7-Flash는 파일이 `deepseek2`라서 `deepseek` 리더로 갑니다.
  - `deepseek` 리더는 `deepseek2`, `deepseek32`, `deepseek4`, `deepseek41`을 받고, `mistral4`와 Kimi-K2도 deepseek2 그래프를 쓰므로 여기로 옵니다.
  - 변형 사이의 차이는 2C 표대로 매개변수 쪽과 종류 쪽으로 가릅니다.
  - 모델 차원의 사실 하나를 강조합니다. **층 종류는 벤더를 가로질러 공유됩니다.**
    - MLA+DSA: V3.2, GLM-5.x, Hy4
    - delta rule: Qwen3-Next+, GLM-5.3-Flash, Kimi
    - mHC: V4.x, GLM-5.3-Flash
    - 해시 n-gram 임베딩: V4.1 engram, Qwen PLE
    - 블록 선택: V4.1 candidate blocks, GLM kpool, Qwen QSA, MiniMax MSA
  - 그래서 리더는 얇게 둡니다. exllamav3의 아키텍처 파일이 194–575줄이니 목표는 약 300줄 이하입니다. 공유 단위는 연산 종류입니다.

### (a) 연산 라이브러리 이름과 상수 전달

- **이름에는 연산과 변형만 넣고 모델은 넣지 않습니다.** 예: `route<Score, Tie, E, K>`, `gqa_flash<HEAD, PACK>`, `latent_attn<LATENT, ROPE>`, `delta_rule<D, KDA>`, `index_score<HEADS, D, Pool>`, `hc_pre<STREAMS, Fmt>`. 오늘의 `qwen3moe_*`, `ds41_*`, `dflash_router` 사본(감사 보고)을 이 이름으로 옮깁니다.
- **컴파일 시간 크기는 레지스터·smem 크기나 루프 전개를 정하는 곳에만 씁니다.**
  - **라우터.** `PER_LANE = N_EXPERT/32`를 레지스터 배열로 들고, `N_USED`만큼 선택 루프를 펼칩니다. 필요한 (E, K)는 §3-E의 14개입니다. **지금의 assert가 이미 전부 받아들입니다.** `router.rs:88`의 `N_EXPERT.is_multiple_of(32) && …ROWS_PER_BLOCK`와 `:90`의 `PER_LANE <= 32`에서 288/32 = 9, 512/32 = 16이 모두 통과합니다. 그래서 const-generic 코어는 재설계가 아니라 이름 바꾸기에 가깝습니다.
  - **GQA flash.** HEAD ∈ {64, 128, 256, 512}이 smem 행과 MMA k-스텝을 정합니다(`flash_gqa.rs:84-97`). GROUP은 스레드 수와 프리필 Q_ROWS를 정합니다(`:79`, `flash_gqa_prefill.rs:94`). 그러니 **PACK은 GROUP을 나누는 가장 큰 2의 거듭제곱**으로 둡니다. GROUP 5는 1, 6은 2, 12는 4, 16은 16이 됩니다. 홀수 GROUP(Qwen3-14B)은 PACK 1 경로로 돌고 효율이 떨어집니다. 선례는 llama.cpp의 `gqa_ratio % k` 디스패치입니다.
  - **latent 어텐션.** LATENT ∈ {256(Mistral-4가 증거), 512}, ROPE ∈ {0, 64}입니다. 블록당 16 heads(MMA M)이므로 20 heads는 패딩이 필요합니다.
  - **delta rule.** D(128×128 상태)와 KDA bool을 컴파일 시간 인자로 둡니다(메인라인 템플릿이 선례). head 수는 런타임 인자입니다.
  - **인덱서.** HEADS ∈ {4, 32, 64}, D 128, 풀 종류.
  - **hc.** STREAMS 4는 세 계열이 모두 씁니다. fn 포맷은 변형입니다. 오늘은 Q3_K와 융합돼 있고 GLM의 fn은 Q8_0입니다.
  - **GEMM·gemv.** 타일은 (m, k, n, format) 선택표로 고릅니다(exllamav3 `select_gemm_shape`, 우리 `gemm.rs` 매크로).
- **나머지는 런타임 인자입니다.** 층 수, K·N, head 수(grid), top-k 폭(오늘 `indexer.rs`처럼 디바이스 워드), eps·scale·limit·θ 표, 위치가 여기에 듭니다.
- **인스턴스 표.** 컴파일 시간 목록(매크로)으로 만들고 적재 때 조회합니다(d). 인스턴스가 늘면 PTX도 늘지만 `ptx-shapes.tsv` 래칫이 이미 모든 진입점을 고정합니다.

### (b) `ModelSpec`의 모양

```
ModelSpec { vocab, hidden, eps, embed{tied}, head{norm, softcap?}, pre, template,
            layers: Vec<LayerSpec>, mtp: Vec<LayerSpec>, drafts: Vec<DraftSpec> }
LayerSpec { mixer, ffn, residual, extras: Vec<Extra>, sources{kv_from, topk_from, index_key_from} }
mixer    = Gqa{heads, kv_heads, head_dim, qk_norm: None|PerHead|WholeQ, rope, bias, out_gate: None|Sigmoid,
               window: Option, sinks, select: Option<Selector>}
         | Latent{heads, latent, nope, rope_dim, v_dim, q_lora, o: Plain|Grouped{groups, rank},
                  window, sinks, compress: Option, select: Option<Selector>}
         | DeltaRule{kind: Gdn{khead_map: Grouped|Tiled} | Kda{lower_bound}, k_heads, v_heads, d, conv, out_gate_act}
         | Mamba2{heads, head_dim, state, groups, conv}
Selector = TokenTopK{heads, d, k, key_norm: Rms|LayerNorm, rope, hadamard}
         | Pool{kind: SoftmaxApe|Mean, width, heads, budget, tail} | Block{block, top, local}
ffn      = Dense{ff, act: SwiGlu{limit}|SwiGluOai{alpha, limit}|GeGlu|Relu2}
         | Moe{experts, top_k, expert_ff, act, router{score: Softmax|SoftmaxSelected|Sigmoid|SqrtSoftplus,
               bias, groups, norm, scale, hash}, shared: Option<{ff, gate: None|Sigmoid}>, latent: Option,
               parallel_dense: Option}
residual = Plain | Hc{streams, kind: Mhc{sinkhorn, eps, collapse: Weighted|Mean} | GatedResidual{rank}}
Extra    = Engram{..} | Ple{..} | Deepstack{..}
```

- **선례.** HF config가 이미 층별 목록을 들고 있습니다: `layer_types`, `mlp_layer_types`, `indexer_types`, `compress_ratios`, `kv_source_layer_ids`, `engram_layer_ids`, `ple_layer_ids`, `layers_block_type`, `moe_layer_freq`, `sparse_attention_freq`.
- **반면교사.** llama.cpp의 hparams는 층별 배열을 쓰지만 공용 구조체에 모델 전용 필드가 쌓였습니다. 층별 열거형으로 가면 이것을 피할 수 있습니다.
- **MTP도 `LayerSpec`입니다.** 드래프트 역시 층 프로그램입니다.

### (c) 층 프로그램

- **새 모델이 쓰는 것.** 리더(메타데이터 → `ModelSpec`), 텐서 역할 표(`roles.rs` 모양), 그리고 정말 새 연산 종류일 때만 그 연산입니다.
- **런타임이 갖는 것.**
  - 스케줄 셋: decode m=1 캡처, verify m ≤ 8 캡처, prompt m ≤ 512/4096 eager(rebuild.md DS6)
  - (스케줄, m)별 CUDA 그래프 캡처
  - 호스트 티어: MoE 연산의 성질이고 배치가 정함
  - 종류별 KV와 상태 저장소: GQA 평면, latent 행, 압축 스트림, 순환 상태 슬롯
  - 되감기(`SeqState`)
- 층 프로그램은 모든 스케줄에 공통인 토큰 블록 위의 함수 하나입니다. 8열 gemv와 GEMM 사이의 컷은 (스케줄, m) 선택자가 정합니다.
- **순환 상태는 런타임이 소유합니다.** `linear-attn.md` 발견 4가 말하듯, delta rule은 상태를 제자리에서 덮어씁니다. 그래서 오늘 V4.1이 쓰는 "위치 색인 쓰기 → 되감기" 규칙이 서지 않습니다.

### (d) 적재 시 커버리지 검사

- 업로드 전에 `ModelSpec`의 모든 칸을 하나씩 확인합니다. 연산 인스턴스(a), 가중치 포맷, pre-tokenizer, 도구 호출 파서가 각각 있는지 봅니다.
- 가중치 포맷은 카드 커널, qdot, `CardFormat`을 하나의 주인으로 합쳐서 봅니다. 오늘은 커널이 있는데 배치가 거부하는 경우가 있습니다(MXFP4, IQ, Q5_K routed).
- 빠진 것은 모두 모아서 **이름 붙은 거부 하나**로 냅니다(`roles.rs:5`의 "fails the file with every …" 모양).
- 오늘 GLM-5.3-Flash를 넣으면 이렇게 나열될 것입니다: `DeltaRule{Kda}` ×34, `Pool{SoftmaxApe, 4}` ×11, router `(Sigmoid, 288, 8)`, Q5_K routed(카드), pre `glm4`, GLM 도구 파서.

### (e) 게이트

- **연산별 적합성 게이트는 모델 파일 없이 돕니다.** 인스턴스마다 매개변수 점에서 합성 가중치를 호스트 기준과 대조합니다. 대상은 모든 (E, K), HEAD×PACK, LATENT×ROPE, D×KDA입니다. 오라클 덤프는 모델 파일이 있는 연산에만 붙입니다. 선례는 `gate_iq`, mistral.rs `grouped_mmq_packed_cuda_tests.rs`(rebuild GG1)입니다.
- **모델별 e2e는 조립만 봅니다.** (모델, 배치)마다 하나씩, greedy 토큰과 `l_out-N`을 대조합니다.
  - GLM-5.3-Flash: ik `glm5next`가 오라클이고, 공개 숫자는 llama.cpp #27754 브랜치로 냅니다.
  - Qwen3.5/3.6: 메인라인 `qwen35moe`로 대조합니다.
- 결과적으로 새 모델의 연산 종류가 이미 있으면 새 연산 게이트는 0개이고, e2e 수는 모델 수 × 배치 수입니다.

### (f) 스펙 디코딩 소스

- **인터페이스 하나.** `DraftSource { propose(seq, k) -> tokens(tree 선택), observe(accepted), rollback }`
  - `NGram`: 있음(`draft.rs:14` `Lookup`)
  - `Mtp{layers}`: 모델 자신의 nextn 층이 embed와 head를 공유하며 도는 층 프로그램. GLM-5.3-Flash 1, Qwen3.5+ 1, V4.1 3(오늘은 Unused), MiniMax 3/7, Nemotron 1
  - `Block{file}`: DSpark/DFlash 외부 드래프트. §3-F의 공개 목록대로 이제 여러 대상에 대해 나와 있습니다
  - EAGLE3와 DSpark가 읽는 hidden 탭(`ds41_hc_mean`이 그런 탭입니다)은 `ModelSpec`의 층별 탭 이름으로 둡니다.
- **검증은 verify 스케줄(m ≤ 8 캡처)이 합니다.**
- **순환 층의 되감기.** 채택된 길이의 상태가 필요하므로 verify 안에서 위치별 **순환 상태 스냅숏**을 씁니다. 메인라인의 `n_rs_seq` 슬롯이 선례이고, `linear-attn.md`의 2슬롯 핑퐁은 k=1만 덮습니다. k=4일 때 카드 비용은 순환 항만으로 계산합니다[유도].

  | 모델 | 순환 상태 / 시퀀스 | k=4 스냅숏 | conv 상태(별도 항, 3토큰 창이라 재생이 쉬움) |
  |---|---|---|---|
  | GLM-5.3-Flash | 34 × 64 × 128 × 128 × 4 B = 142.6 MB | **570 MB** | 34 × 24,576 채널 × 3 = 5.0 MB(bf16) |
  | Qwen3.6-35B-A3B | 30 × 32 × 128² × 4 B = 62.9 MB | **252 MB** | 30 × 8,192 × 3 = 1.5 MB |
  | Qwen3.8-Flash-Next | 36 × 48 × 128² × 4 B = 113.2 MB | **453 MB** | 36 × 10,240 × 3 = 2.2 MB, + PLE conv 1층 |

  GLM의 스냅숏을 verify마다 쓰는 비용은 570 MB ÷ 약 700 GB/s ≈ 0.8 ms입니다[유도]. 메모리가 모자라면 채택된 prefix 재생으로 물러섭니다.

---

## 6. 추천 대상

| 순위 | 모델 | 이유 | 적합성 [유도] | 참조 덤프 | 새로 강제하는 연산 |
|---|---|---|---|---|---|
| **1** | **GLM-5.3-Flash** | 사용자가 지명한 모델. 다운로드 424만. V4.1 스택과 겹침이 최대 | 아래 GLM 산술 참고 | **ik `glm5next`**가 오라클. 공개 숫자는 **llama.cpp #27754 브랜치**(unsloth GGUF가 여기서 나옴). 메인라인에는 없음 | KDA(+conv1d, gated RMSNorm, L2), k-pool DSA 점수(+LayerNorm), NoPE 흡수 MLA 조합, router 288/8 sigmoid, Q5_K routed 카드, hc fn Q8_0, pre `glm4`, GLM 도구 파서, MTP 1 |
| **2** | **Qwen3.6-35B-A3B**(Qwen3.5 계열의 운반체. 같은 프로그램으로 27B dense, 122B, 397B, Qwen3-Next까지 감) | 사용자의 Qwen 변형 요구. 다운로드 319만 + FP8 786만. plan의 `q35` 사슬 | UD-Q4_K_XL 22.4 GB + KV 20.5 KB/tok(32k에서 0.67 GB) + 상태 63 MB. 3090에는 빠듯하고 A6000에는 넉넉함. 하한 324 tok/s(3090) / 243(A6000) (`plan-triage.md:286`, 거기서의 유도) | 메인라인 `qwen35moe` #19468 + ik | GDN(KDA와 같은 delta-rule 커널의 const 변형), 게이트 어텐션, GQA HEAD 256 PACK 8, IMROPE 64/256, 공유 게이트, router 256/8, pre `qwen35`, Qwen xml 도구 파서, MTP 1 |
| **3** | **Qwen3.8-Flash-Next**(`qwen4exp`) | 가장 새 Qwen이고 "the architecture that will underpin Qwen4" [K L26] | UD-Q4_K_XL 111.3 GB = expert 약 78.0 + n-gram 약 28.8 + 나머지 약 4.5 GB. A6000에 나머지 + expert 약 41 GB(53 %), 호스트에 나머지. 전부 호스트라면 routed 2.36B params → 1.52 GB/token → 11.2 ms @136. n-gram 조회는 토큰당 약 1.4 KB. Q8_0 195.1 GB도 들어감 | 메인라인 `qwen4exp` #27742 + ik | 2번에 더해: QSA 블록 인덱서, gated-residual HC(4×320), PLE n-gram(splitmix·XOR 해시, 소수 크기 표, dilated conv; engram 행 경로가 틀이 됨), GROUP 12 → PACK 4, router 512/10, GDN 게이트 sigmoid |
| **4** | **gpt-oss-120b** | 다운로드 456만. GQA 위 sinks와 SWA, 네이티브 MXFP4(커널이 이미 있음)를 가장 싸게 덮는 한 걸음. SWA는 MiMo·Step·Gemma도 씀 | MXFP4 65.4 GB = expert 60.9 GB(114.66B × 4.25 bpw) + BF16 비-expert 4.3 GB(파일 크기와 맞음). A6000에 약 40 GB(66 %), 호스트에 약 21 GB. 호스트 몫 0.65 GB/token → 4.8 ms(균등 라우팅 가정) | 메인라인 `gpt-oss` #15091 | GQA HEAD 64 PACK 8, GQA flash의 sinks, SWA 128(1:1), swigluoai, 선택-softmax 라우터(Qwen 규칙과 같음), QKV bias, MXFP4 배치 라우팅, o200k pre(이름 미확인), harmony 파서 |

**GLM-5.3-Flash 적합성 산술 [유도]**
- **expert 파라미터.** 43층(MTP 포함) × 288 × 3 × 4096 × 2048 = 311.7B입니다. 비-expert는 321.3 − 311.7 = 9.7B입니다.
- **UD-Q4_K_XL(199.7 GB).**
  - 비-expert를 Q8_0 8.5 bpw로 보면(`[S]` 표 C) 10.3 GB이고, expert는 189.4 GB = 4.86 bpw로 **214 GB 안**입니다.
  - 카드 쪽은 KV 16.9 KB/tok(MLA 11 × 512 × 2 + 인덱서 [key; gate] 11 × 256 × 2)과 KDA 상태 143 MB를 더해도 A6000에 hot expert용으로 약 35 GB가 남습니다.
- **토큰당 바이트.**
  - routed 8.46B params → 5.14 GB → 전부 호스트면 37.8 ms @136
  - 카드 쪽 비-expert 8.29B params(KDA 4.69B + MLA·인덱서 1.37B + dense 0.45B + 공유 1.06B + lm_head 0.63B) → 8.8 GB → 12.6 ms @약 700 GB/s
  - V4.1은 16.77 MB × 6 × 40 = 4.04 GB/token(facts.md:64)이므로 GLM Q4_XL은 V4.1의 1.27배입니다.
- **UD-Q3_K_XL(147.5 GB).** expert 137.2 GB = 3.52 bpw → 3.72 GB/token → 27.4 ms(V4.1의 0.92배)입니다. 다만 이 파일의 expert는 IQ3_XXS와 IQ4_XS인데(`[S]`), qdot에 IQ4_XS가 없습니다.

**다음 후보.** MiniMax-M3(MSA 블록 희소, 계열 전용), Nemotron-3.5(Mamba2), Gemma-4-26B-A4B(softcap, GeGLU, K=V 512), Mistral-Small-4(LATENT 256)입니다.

**사용자 순위와 제작 순서가 갈리는 이유.**
- 순위는 사용자 관심 순서라 GLM이 먼저입니다.
- 제작은 ②에서 먼저 **delta-rule 커널을 세우는 편이 쌉니다[추천]**.
  - 메인라인 오라클이 있고 카드 한 장에서 빠른 게이트를 돌릴 수 있습니다.
  - 그 뒤 ①의 KDA가 같은 커널의 `KDA` const 변형으로 올라가고, 호스트 티어와 mHC는 V4.1 것을 씁니다. 메인라인 커널도 GDN과 KDA를 bool 하나로 가릅니다.
- **GLM을 먼저 한다면** KDA는 ik(`llama-kda.cpp`)와만 대조됩니다. 메인라인 대조가 필요하면 Kimi-Linear-48B-A3B(메인라인 #18755, Q4_K_M 30.1 GB, A6000 한 장)가 KDA 확인 수단입니다. 이것은 다섯째 대상이 아니라 확인 수단일 뿐입니다.
- 두 길 모두 앞에 공통 단계가 있습니다. 재구성으로 `ModelSpec`과 연산 라이브러리를 만들고, 오늘의 두 모델(V4.1, Qwen3)을 그 위로 옮기는 것입니다.

---

## 7. 지정 밖에서 본 개선 여지 (보고만 했고 손대지 않았습니다)

1. **`crates/gpu-deepseek41/src/engram_gate.rs:57`.** `pub const ROW: usize = 5120;`로 V4.1의 hidden이 커널 기하에 박혀 있습니다(:60 assert). Qwen3.8 PLE(2560)나 다른 n-gram 모델이 재사용할 수 없습니다. 크기 S.
2. **`crates/model/src/placement.rs:270-299`와 `:363-368`.** `CardFormat::of`와 `of_routed`가 MXFP4, Q2_K, IQ2_XS/IQ3_XXS/IQ4_XS를 거부합니다. 그런데 카드 커널은 있습니다(`crates/gpu/src/mxfp4.rs`, `crates/gpu/src/iq.rs`). "카드가 무엇을 돌리나"의 주인이 둘인 셈입니다. GLM-5.3-Flash UD-Q4_K_XL의 routed down은 Q5_K라서 hot list가 카드로 못 갑니다. 크기 S–M.
3. **`crates/gpu/src/route_core.rs:14-16`.** `Sigmoid` 점수는 컴파일되지만 아무 라우터도 쓰지 않습니다. 라우터 넷이 각자 (E, K) 상수를 가집니다: `crates/gpu/src/router.rs:38-41`, `crates/gpu-deepseek41/src/router.rs:60-63`, `crates/gpu/src/arch/qwen3moe/router.rs:72-75`, `crates/gpu-deepseek41/src/experts_mxfp4.rs:63-66`. `[S]` §6의 const-generic 코어로 모을 수 있습니다(감사의 `dflash_router` 발견과 겹칩니다). 크기 M.
4. **`crates/model/src/arch/deepseek41/roles.rs:39-41`.** V4.1 파일에 든 nextn 3층이 `Role::Unused`입니다. DSpark와 n-gram 옆에 셋째 드래프트 소스가 놀고 있습니다. 크기 M.
5. **`crates/gpu/src/flash_gqa.rs:67-69`, `flash_gqa_prefill.rs:178`, `rope_neox.rs:46`.** HEAD 128, GROUP 8, 전면 회전이 assert로 고정돼 있습니다. Qwen3 dense(GROUP 2/4/5), 235B(16), Qwen3.5+ 전부(HEAD 256, partial 64)가 이 경로를 못 씁니다. 크기 M.
6. **`crates/gpu-deepseek41/src/hc.rs:66-71`.** HC_PRE가 Q3_K gemv와 융합돼 있습니다(`HC_PIECE 512` = q3_K 행 걷기 한 번). 연산이 가중치 포맷에 묶인 형태라서 GLM의 Q8_0 hc fn은 `hc_f32.rs` 경로나 새 융합판이 필요합니다. 크기 S.
7. **`crates/tokenizer/src/pretok.rs:202-214`와 `crates/serve/src/dsml.rs`.** pre 이름이 넷뿐이고 도구 파서는 DSML 하나입니다. 1–3순위에는 `glm4`, `qwen35` pre와 GLM(`<arg_key>`)·Qwen(xml `<function=…>`) 파서가 필요합니다. 각각 크기 S.
8. **`docs/research/models-survey.md:108, 127, 129`.** "unverified" 칸 셋을 이제 채울 수 있습니다.
   - V4.1 활성 파라미터: "8B / 16B"(카드 L78)
   - Qwen3.6과 Flash-Next 라우터: softmax → top-k → renorm + sigmoid 공유 게이트(`modeling_qwen3_5_moe.py:896, 910-919`, `modeling_qwen4_exp.py` `Qwen4ExpTextTopKRouter`)
   
   크기 XS(문서).

---

## 부록 A. 인용 줄 (`facts.txt`, 작업 파일 원문)

원문 그대로이되 한 곳만 고쳤다. Mistral-Small-4 줄의 아키텍처 클래스 이름(32자)을 GitHub 푸시 보호가 Mistral API 키로 읽어 푸시를 막았으므로, 이름을 두 조각으로 적었다.

```text
## GLM-5.3-Flash (zai-org/GLM-5.3-Flash, config.json + transformers main models/glm5_next/modeling_glm5_next.py fetched 2026-09-26)
- card L25 "With 320B total parameters and just 18B active parameters"; L27 "hybrid architecture combining sparse and linear attention"; "Manifold-Constrained Hyper-Connections (mHC)"; first natively multimodal GLM-5
- HF api total=321,323,031,390 (BF16 6.93B, F8_E4M3 314.4B); FP8 e4m3 block [128,128] quant; license mit
- text_config: hidden 4096, layers 45, layer_types period 4 = linear_attention x3 + deepseek_sparse_attention x1 (full_attn_layers [3,7,...,43] = 11 DSA, kda_layers 34)
- linear_attn_config: num_heads 64, head_dim 128, short_conv_kernel_size 4, gate_lower_bound -5.0
- MLA: q_lora_rank 1536, kv_lora_rank 512, qk_nope_head_dim 256, qk_rope_head_dim 0, v_head_dim 256, mla_use_nope true, num_attention_heads 64, num_key_value_heads 64
- indexer: index_n_heads 32, index_head_dim 128, index_topk 2048, index_kpool 4, index_kpool_compress true, index_kpool_always_select_tail true, indexer_types all "full" (45), index_share_for_mtp_iteration true, indexer_rope_interleave true (but NoPE: modeling L~1500 "Key change using NoPE" position_embeddings=None)
- mHC: hc_mult 4, hc_sinkhorn_iters 20, hc_eps 1e-6, "mhc": true; pre=sigmoid+eps, post=2*sigmoid, comb=softmax+Sinkhorn; final hc_head = unweighted mean ("Unlike DeepSeek-V4, this is an unweighted mean")
- MoE: n_routed_experts 288, num_experts_per_tok 8, n_shared_experts 1, moe_intermediate_size 2048, intermediate_size 12288 (dense), first_k_dense_replace 3 (mlp_layer_types dense x3, sparse x42), scoring_func sigmoid, topk_method noaux_tc, n_group 1, topk_group 1, norm_topk_prob true, routed_scaling_factor 2.5, moe_router_dtype float32
- swiglu_limit 10.0: gate clamp(max=10), up clamp(-10,10), silu(gate)*up (dense MLP, shared, routed experts)
- rms_norm_eps 1e-5, vocab 154880, max_position_embeddings 1048576, num_nextn_predict_layers 1, tie_word_embeddings false
- KDA layer (modeling L622-773): q/k/v proj hidden->64*128 each; depthwise causal conv1d k=4 + SiLU over concat qkv; forget gate per-channel = lower_bound*sigmoid(exp(A_log)*(f_b(f_a(x))+dt_bias)) (f_a: hidden->128, f_b: 128->8192); beta=sigmoid(b_proj(x)) per head; q,k l2norm in kernel; state 64 x 128x128 fp32; decode = recurrent_kimi_delta_attention, prefill = chunk_kimi_delta_attention(chunk 64); output = RMSNormGated(sigmoid gate from g_b(g_a(x))) then o_proj
- DSA indexer (L774-1101): wq_b from q_resid (q_lora), wk + LayerNorm k_norm, weights_proj (hidden->32 heads), ReLU scores * scale, kpool: softmax(gate_scores+APE) weighted avg of 4 consecutive keys, select topk//kpool=512 pools, expand to tokens (+tail up to 3); cross-layer top-k sharing supported ("shared" indexer_types) but unused in 5.3-Flash
- vision tower: depth 24, hidden 1024, patch 14, axial 2D rope, spatial_merge 2
## GLM-5.3 / GLM-5.2 (GlmMoeDsaForCausalLM, model_type glm_moe_dsa; configs fetched) — identical text arch
- hidden 6144, layers 78 (dense x3 then sparse x75), MLA q_lora 2048 kv_lora 512 qk_nope 192 qk_rope 64 v 256, heads 64 (kv 64), rope_theta 8,000,000 rope_interleave true, max_pos 1048576
- DSA indexer: index_n_heads 32, index_head_dim 128, index_topk 2048, index_topk_freq 4, index_skip_topk_offset 3, indexer_types full x3 then (shared x3, full x1) repeating => cross-layer top-k sharing
- MoE: 256 routed + 1 shared, top-8, moe_inter 2048, dense inter 12288, sigmoid, noaux_tc, n_group 1, scaling 2.5, norm_topk true; MTP 1; vocab 154880
- GLM-5.3 FP8 e4m3 [128,128] 753.3B (api); GLM-5.2 BF16 753.3B; license 5.3 "other", 5.2 mit; card GLM-5.3 transformers doc glm_moe_dsa
## GLM-5.1 / GLM-5 (glm_moe_dsa): same as 5.2 except rope_theta 1,000,000, max_pos 202752, no index_topk_freq/indexer_types (every layer own indexer); 753.9B BF16 (api)
## GLM-4.7-Flash (Glm4MoeLiteForCausalLM, glm4_moe_lite): hidden 2048, 47 layers, MLA q_lora 768 kv_lora 512 qk_nope 192 qk_rope 64 v 256, 20 heads, 64 routed top-4 + 1 shared, moe_inter 1536, first_k_dense 1, scaling 1.8, noaux_tc, rope_theta 1e6, max_pos 202752, MTP 1, vocab 154880; 31.2B BF16 (api)
## GLM-4.7 / 4.6 / 4.5 (Glm4MoeForCausalLM, glm4_moe): hidden 5120, 92 layers, GQA 96 q / 8 kv, head_dim 128, partial_rotary_factor 0.5, attention_bias true, use_qk_norm true, 160 routed top-8 + 1 shared, moe_inter 1536, dense inter 12288, first_k_dense 3, scaling 2.5, rope_theta 1e6, max_pos 202752 (4.5: 131072), MTP 1, vocab 151552; ~357-358B BF16 (api)
## GLM-4.5-Air: hidden 4096, 46 layers, 96 q / 8 kv x 128, partial rotary 0.5, bias true, use_qk_norm false, 128 routed top-8 + 1 shared, moe_inter 1408, first_k_dense 1, scaling 1.0, MTP 1, vocab 151552; 110.5B (api)
## Qwen3-30B-A3B (Qwen3MoeForCausalLM, qwen3_moe; config fetched): hidden 2048, 48 layers, 32 q / 4 kv x head_dim 128 (GROUP 8), 128 experts top-8, moe_inter 768, norm_topk true, rms eps 1e-6, rope_theta 1e6, max_pos 40960, vocab 151936, no shared expert; 30.5B (api)
## Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration, model_type qwen4_exp; config + transformers models/qwen4_exp/modeling_qwen4_exp.py fetched)
- api total 179,999,981,459 BF16; license other; image-text-to-text (vision depth 27, hidden 1152, patch 16)
- text: hidden 2560, 48 layers, layer_types linear_attention x3 + full_attention x1 (full_attention_interval 4) => 12 full, 36 GDN
- full attn: 24 q / 2 kv x head_dim 256, q_proj x2 => sigmoid output gate ("attn_output * torch.sigmoid(gate)"), q_norm/k_norm RMSNorm, partial_rotary 0.25, rope_parameters mrope_interleaved true mrope_section [11,11,10] rope_theta 1e7
- QSA indexer on every full layer (Qwen4ExpTextQSAIndexer): index_qk_proj hidden->(4+1)*128, q RMSNorm + rope, keys mean-pooled in blocks of indexer_compress_ratio 4, RMSNorm, rope at block start, relu(q.k) summed over 4 heads /sqrt(128), top indexer_budget/4=512 blocks + tail
- GDN (Qwen4ExpTextGatedDeltaNet): linear_num_key_heads 16, linear_num_value_heads 48, key/value head dim 128, conv 4, in_proj_qkv/z/b/a, g=-exp(A_log)*softplus(a+dt_bias) (per-head scalar), beta=sigmoid(b), q repeat_interleave v/k=3, RMSNormGated (output_gate_type sigmoid), mamba_ssm_dtype float32
- MoE every layer: 512 experts top-10, moe_inter 640, softmax router + norm_topk, shared expert (640) with sigmoid shared_expert_gate
- hyper-connections: hc_count 4, hc_lowrank 320; Qwen4ExpTextGatedResidual = grouped RMSNorm, low-rank silu->sigmoid elementwise mix weights, mean over streams; injection 2*sigmoid(Linear(hc*h -> hc)); final hyper_connection_mixer (NOT Sinkhorn mHC)
- PLE: ple_layer_ids [2] (layer_idx 1), hashed n-gram embedding (ngram_size 3, heads_per_ngram 8 => 16 heads, per-head prime-sized tables base ngram_vocab_size_base 20,000,000, splitmix64 multipliers, XOR), ple_embed_dim 2560 -> head_dim 160; key per stream / shared value, sigmoid gate, dilated depthwise conv (kernel 4, dilation 3) + SiLU; n-gram table ~16 x 20M x 160 = 51.2B params [derived]
- MTP: mtp_num_hidden_layers 1 (full_attention); vocab 248320; max_pos 262144
## Qwen3.5/3.6 (configs fetched; keys-qwen35.txt). All: attn_output_gate true, head_dim 256, full_attention_interval 4 (linear x3 + full x1), GDN key heads 16 x128, conv 4, mrope interleaved [11,11,10], partial 0.25, theta 1e7, vocab 248320, max_pos 262144, MTP 1, rms 1e-6
- Qwen3.5-35B-A3B / Qwen3.6-35B-A3B (qwen3_5_moe): hidden 2048, 40 layers (30 GDN + 10 full), GQA 16/2, GDN v-heads 32, 256 experts top-8, moe_inter 512 + shared 512; 35.95B (api)  => plan's "qwen35moe: GQA 16/2 x 256, GDN 30 layers" CONFIRMED
- Qwen3.5-122B-A10B: hidden 3072, 48 layers, GQA 32/2, v-heads 64, 256 top-8, moe 1024 + shared 1024; 125.1B
- Qwen3.5-397B-A17B: hidden 4096, 60 layers, GQA 32/2, v-heads 64, 512 top-10, moe 1024 + shared 1024; 403.4B
- Qwen3.5 dense (qwen3_5): 27B hidden 5120/64L/24:4/v48/ff 17408; 9B 4096/32L/16:4/v32/ff 12288; 4B 2560/32L/16:4/v32/ff 9216 tied; 2B 2048/24L/8:2/v16/ff 6144 tied; 0.8B 1024/24L/8:2/v16/ff 3584 tied
- Qwen3.6-27B and Qwen3.8-27B (qwen3_5): = Qwen3.5-27B + output_gate_type "swish" (GDN gated-norm activation)
- Qwen-AgentWorld-35B-A3B = qwen3_5_moe 35B-A3B shape + output_gate_type swish
## Qwen3-Next-80B-A3B / Qwen3-Coder-Next (qwen3_next): hidden 2048, 48 layers, full_attention_interval 4, GQA 16/2 x 256, GDN k16/v32 x128, 512 experts top-10, moe 512 + shared 512, partial 0.25, theta 1e7 (Coder-Next 5e6), plain rope_scaling null (no M-RoPE), vocab 151936, max_pos 262144; 81.3B / 79.7B; no mtp key in config
## Qwen3 MoE: 235B-A22B-2507: hidden 4096, 94L, 64/4 x128 (GROUP 16), 128 top-8, moe 1536, theta 5e6, max_pos 262144; Coder-480B-A35B: hidden 6144, 62L, 96/8 x128 (GROUP 12), 160 top-8, moe 2560, theta 1e7; Coder-30B-A3B: = 30B-A3B shape, theta 1e7, max_pos 262144
## Qwen3-VL-30B-A3B (qwen3_vl_moe_text): = 30B-A3B text shape, theta 5e6, mrope interleaved section [24,20,20], deepstack_visual_indexes present; Qwen3-Omni-30B-A3B thinker: = 30B-A3B shape, mrope [24,20,20], theta 1e6, vocab 152064, max_pos 65536
## Qwen3 dense (qwen3): 0.6B 1024/28L/16:8 ff3072 tied; 1.7B 2048/28L/16:8 ff6144 tied; 4B 2560/36L/32:8 ff9728 tied; 8B 4096/36L/32:8 ff12288; 14B 5120/40L/40:8 (GROUP 5) ff17408; 32B 5120/64L/64:8 ff25600; all head_dim 128, theta 1e6, max_pos 40960, vocab 151936
- Qwen3.8-Flash-Next card L26 "experimental preview of the architecture that will underpin Qwen4"; L46 "125B with 6B activated, plus 51B n-gram embedding and 4B MTP"; "N-gram Embedding: 20,000,000 (bigrams/trigrams at layer 2)"; QSA "Indexer Structure: MQA with 4 Query Heads and 1 Shared Key Head", "Budget: 512 blocks or 2048 tokens", "Rotary Position Embedding Dimension: 64"; Gated Residual "Number of Branches: 4, Bottleneck Rank: 320"; context 262,144 native, YaRN to 1M
- Qwen3-Next card: "80B in total and 3B activated", "Hybrid Layout: 12 * (3 * (Gated DeltaNet -> MoE) -> 1 * (Gated Attention -> MoE))", MTP listed in highlights
- Qwen3.5-35B-A3B card: "35B in total and 3B activated", "10 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Gated Attention -> MoE))", "8 Routed + 1 Shared", "MTP: trained with multi-steps"
## DeepSeek-V4.1-Flash (ours; DeepseekV41ForCausalLM deepseek_v41; config fetched cfg-v41flash.txt): hidden 5120, 40 layers, 64 heads / 1 kv, head_dim 512, qk_rope 64, q_lora 1280, o_lora 1024, o_groups 8, swiglu_limit 10, rms eps 1e-20, rope_theta 1e4 yarn factor 16 orig 65536, 384 routed top-6 + 1 shared, moe_inter 2304, scoring_func "sqrtsoftplus", noaux_tc, scaling 1.5, sliding_window 128, compress_ratios len43 [0x2,2x18,1x20,0x3], compress_rope_theta 160000, kv_source_layer_ids [2,8,14,20], index_source_layer_ids [2,8,14,20,24,28,32,36], index 32 heads x128 topk 512, candidate_source_layer_id 20 candidate_topk_blocks 2048 block 8, hc_mult 4 sinkhorn 20, engram_layer_ids [1,14] max_ngram 4 heads 8 x256 vocab 16M, MTP 3, dspark block 5 markov rank 256 128 experts top-3 target layers [37,38,39]; quant fp8 [32,32] ue8m0 + expert_dtype fp4; vocab 129280; max_pos 1048576; vision tower 32L
## DeepSeek-V4-Flash (deepseek_v4): hidden 4096, 43 layers, 256 routed top-6, moe 2048, q_lora 1024, index 64 heads, compress_ratios [0,0 then 4/128 alternating], num_hash_layers 3, MTP 1, rms 1e-6, fp8 [128,128] + fp4 experts; 0731 variant 46 entries, dspark targets [40,41,42]; api 290.9B / 304.2B
## DeepSeek-V4-Pro (deepseek_v4): hidden 7168, 61 layers, 128 heads, 384 routed top-6, moe 3072, q_lora 1536, o_groups 16, index 64 heads topk 1024, scaling 2.5, num_hash_layers 3; api 1.60T (0813: 1.65T)
## DeepSeek-V3.2 (deepseek_v32) / V3.1 (deepseek_v3): hidden 7168, 61L, MLA 128 heads q_lora 1536 kv_lora 512 nope 128 rope 64 v 128, 256 routed top-8 + 1 shared, n_group 8 topk_group 4 (group-limited), sigmoid noaux_tc, scaling 2.5, first_k_dense 3, yarn factor 40 orig 4096, MTP 1, vocab 129280, max_pos 163840; V3.2 DSA index 64 heads x128 topk 2048; ~685B (api)
## Kimi-Linear-48B-A3B (kimi_linear; config fetched): hidden 2304, 27 layers, KDA 20 layers (32 heads x128, conv 4) + MLA 7 layers (full_attn_layers [4,8,12,16,20,24,27] 1-indexed), mla_use_nope true, 32 heads, kv_lora 512, q_lora null, nope 128 rope 64 v 128, 256 routed top-8 + 1 shared, moe 1024, sigmoid, scaling 2.446, first_k_dense 1, vocab 163840, 1M ctx; card "48B total 3B activated"; api 49.1B
## Kimi K3 (card): "2.8T-parameter model built on Kimi Delta Attention (KDA) and Attention Residuals (AttnRes)", 104B active, 93 layers "69 KDA + 24 Gated MLA", 896 experts top-16 + 2 shared, LatentMoE dim 3584, activation "SiTU-GLU", 160K vocab, 1M ctx
## Kimi K2.6 / K2-Thinking (kimi_k2, DeepSeek-V3 arch): hidden 7168, 61L, MLA 64 heads, 384 routed top-8 + 1 shared, sigmoid noaux_tc n_group 1, scaling 2.827, first_k_dense 1, yarn factor 64, vocab 163840, MTP 0, compressed-tensors pack-quantized INT4; card "1T total, 32B activated"
## MiniMax-M2.7 (= M2/M2.1/M2.5 shape; minimax_m2): hidden 3072, 62L all full attention, GQA 48/8 x128, rotary_dim 64 (partial 0.5), use_qk_norm true qk_norm_type "per_layer", 256 experts top-8 (num_local_experts), sigmoid + use_routing_bias, no shared, moe inter 1536, theta 5e6, MTP 3 modules, vocab 200064, max_pos 204800, FP8 [128,128]; api 228.7B
## MiniMax-M3 (minimax_m3_vl): card "~428B parameters and ~23B activated", native multimodal 1M ctx; hidden 6144, 60L, GQA 64/4 x128, partial 0.5 (rotary_dim 64), use_gemma_norm true, qk_norm per_head, hidden_act swigluoai (swiglu_alpha 1.702, swiglu_limit 7.0), 128 experts top-4 + 1 shared (3072), dense 12288 first 3 layers (moe_layer_freq 0x3,1x57), sigmoid + routing bias, scaling 2.0, block sparse attention (4 index heads x128, topk 16 blocks x 128, score max, local block 1) on layers 3..59, MTP modules 7 / nextn 1, vocab 200064; api 427.0B BF16
## gpt-oss-120b (gpt_oss): hidden 2880, 36L alternating sliding_attention(128)/full, GQA 64/8 x 64, attention_bias true, 128 experts top-4, intermediate 2880, swiglu_limit 7.0, yarn factor 32 orig 4096 theta 150000, vocab 201088, max_pos 131072, quant mxfp4 (experts; attn/router/embed not converted); api 116.8B
## gpt-oss-20b: hidden 2880, 24L alt sliding(128)/full, 64/8 x64, bias, 32 experts top-4, mxfp4; api 20.9B
## Mistral-Small-4-119B-2603 (`Mistral3` + `ForConditionalGeneration`, model_type mistral4): hidden 4096, 36L, MLA q_lora 1024 kv_lora 256 nope 64 rope 64 v 128, 32 heads, 128 routed top-4 + 1 shared, moe 2048, first_k_dense 0, scaling 1.0, yarn factor 128 orig 8192 + llama_4_scaling_beta 0.1, vocab 131072, max_pos 1048576, fp8; api 119.4B
## Mistral-Medium-3.5-128B (ministral3): dense hidden 12288, 88L, GQA 96/8 x128, ff 28672, yarn 64, vocab 131072, fp8; api 127.7B
## Nemotron-3.5-Lightning-30B-A3B (nemotron_h): hidden 2688, 52 blocks layers_block_type = mamba x23 / moe x23 / attention x6 (one mixer per block), attention GQA 32/2 x128, Mamba2 64 heads x64 ssm_state 128 n_groups 8 conv 4 expand 2 chunk 128, MoE 128 routed top-6 + shared 3712, moe inter 1856, mlp_hidden_act relu2, scaling 2.5, MTP 1 (mtp_layers_block_type [attention, moe]), vocab 131072, max_pos 262144; api 31.6B
## Nemotron-3-Super-120B-A12B (nemotron_h): hidden 4096, 88 blocks hybrid_override_pattern "MEMEMEM*E…" (M=Mamba2, E=MoE, *=attention), Mamba2 128 heads x64, 512 routed top-22, moe_latent_size 1024 (latent MoE), scaling 5.0, relu2, MTP "*E"; api (not fetched yet)
## Gemma-4-26B-A4B (gemma4_text): hidden 2816, 30L sliding(1024) x5 : full x1, 16 q / 8 kv x 256 (sliding), global_head_dim 512 num_global_key_value_heads 2 attention_k_eq_v true, full rope proportional partial 0.25 theta 1e6 / sliding theta 1e4, 128 experts top_k_experts 8 moe 704 + enable_moe_block (dense intermediate 2112), gelu_pytorch_tanh, final_logit_softcapping 30.0, tied, vocab 262144; api 25.8B
## Gemma-4-31B (gemma4_text): dense hidden 5376, 60L, 32/16 x256, global head 512 kv 4, same sliding/softcap; api 31.3B
## Granite-4.2-30b (granite): dense hidden 4096, 64L, 32/8, ff 32768, theta 5e7, vocab 100352; api 29.3B; granite-swash-3b-a600m (granitemoe_swa): 48 experts top-4, SWA 128 pattern
## GGUF (HF api tree sizes, summed per quant, 1e9 B; gguf-a/b/c.txt, gguf-glm53.txt)
- unsloth/GLM-5.3-Flash-GGUF: UD-IQ1_S 93.1, UD-IQ2_XXS 101.8, UD-Q2_K_XL 108.7, UD-IQ3_XXS 120.4, UD-Q3_K_XL 147.5, UD-IQ4_XS 156.8, UD-Q4_K_XL 199.7, UD-Q5_K_XL 240.3, Q8_0 341.0, BF16 641.6, mmproj 2.3
- unsloth/GLM-5.3-GGUF: UD-IQ1_S 216.7, UD-IQ2_M 238.6, UD-Q2_K_XL 253.9, UD-IQ3_XXS 281.7, UD-Q4_K_XL 467.3; GLM-5.2 same sizes
- unsloth/Qwen3.8-Flash-Next-GGUF: UD-IQ1_S 72.5, UD-Q2_K_XL 78.9, UD-Q3_K_XL 90.0, UD-IQ4_XS 93.7, UD-Q4_K_XL 111.3, UD-Q5_K_XL 158.3, Q8_0 195.1, BF16 367.0
- unsloth/Qwen3.6-35B-A3B-GGUF: UD-Q4_K_XL 22.4, MXFP4_MOE 21.7, Q8_0 36.9, BF16 69.4 (+ unsloth/Qwen3.6-35B-A3B-MTP-GGUF exists)
- unsloth/Qwen3.5-122B-A10B-GGUF: Q4_K_M 76.5, Q8_0 129.9; Qwen3.5-397B-A17B: UD-IQ3_XXS 140.3, Q3_K_M 177.4, UD-IQ4_XS 189.7, Q4_K_M 244.1
- unsloth/MiniMax-M3-GGUF: UD-Q2_K_XL 143.0, UD-Q3_K_XL 194.9, UD-IQ4_XS 207.6, UD-Q4_K_XL 264.9; MiniMax-M2.7: UD-Q3_K_XL 101.9, UD-Q4_K_XL 140.8, Q8_0 243.1
- unsloth/GLM-4.7-Flash-GGUF: Q4_K_M 18.3, Q8_0 31.8; bartowski Kimi-Linear-48B-A3B: Q4_K_M 30.1, Q8_0 52.2; unsloth/gpt-oss-120b-GGUF ~62.6-65.4 (MXFP4 native, all quants ~same); Qwen3-Next-80B: Q4_K_M 48.5, Q8_0 84.8; Qwen3-Coder-Next Q4_K_M 48.5
- antirez/deepseek-v4.1-flash-gguf: Q2 365.7 GB (single file) + Vision 1.0; Nemotron-3.5-Lightning UD-Q4_K_XL 25.5, Q8_0 35.0; gemma-4-26B-A4B UD-Q4_K_XL 14.2; Mistral-Small-4 UD-Q4_K_XL 74.2; Step-3.7-Flash UD-Q4_K_XL 122.2; MiMo-V2.6-Flash ggml-org Q2_K 126.2, MXFP4 167.4
## llama.cpp (local fork ~/repo/llama.cpp-fork @9e47962ef 2026-09-14): arch names include qwen3 qwen3moe qwen3next qwen3vl qwen3vlmoe qwen35 qwen35moe qwen4exp glm4 glm4moe glm-dsa deepseek2 deepseek32 deepseek4 deepseek41 kimi-linear kimi-k3 minimax-m2 minimax-m3 gpt-oss mistral3 mistral4 nemotron_h nemotron_h_moe gemma4 mimo2 step35 hy_v4 dflash eagle3; NO glm5next
- GLM-5.3-Flash PRs (gh api, 2026-09-26): #27752 open (eauchs, +2395, src/models/glm5next.cpp +1261, llama-memory-hybrid-idx), #27754 open (danielhanchen/unslothai:glm5next/upstream, +2558, upd 09-17), #27773 open (timkhronos, 74 review comments), #27917 draft "Add MTP for GLM-5.3-Flash"; none merged
## Reference engine structure (measured)
- llama.cpp fork @9e47962ef (2026-09-14): per-arch class llama_model_<arch> in src/models/<arch>.cpp with load_arch_hparams / load_arch_tensors / build_arch_graph (qwen35moe.cpp:4,36,149; graph_mtp :551); models.h 2740 lines; llama-hparams.h per-layer arrays n_head_arr/n_head_kv_arr/n_ff_arr/n_ff_exp_arr/n_expert_used_arr(:98-105), rope_pattern(:168), is_swa_impl(:183), is_recr_impl(:186), is_indexer_full_impl(:285), dsv4_compress_ratios(:295), engram_layer_ids(:307), dsv41_kv_source/index_key_source/topk_source(:313-315), is_ple_impl(:330), deepstack_mapping_arr(:354), swiglu_clamp_exp/shexp(:378-379); LLAMA_MAX_LAYERS 512
- wc -l src/models: qwen3 159, qwen3moe 179, qwen35 644, qwen35moe 741, qwen3next 822, qwen3vl 196, qwen3vlmoe 190, qwen4exp 1283, delta-net-base 606, glm4-moe 444, glm-dsa 769, deepseek2 713, deepseek32 726, deepseek4 1521, deepseek41 948, kimi-linear 562, kimi-k3 618, minimax-m2 168, minimax-m3 603, openai-moe 177, gemma4 498, nemotron-h 330, nemotron-h-moe 156, mamba-base 302, mistral4 6 (reuses deepseek2 graph), mimo2 396, step35 560, hy-v4 601, dflash 1001, eagle3 338
- model-add PR diffstats (git show --stat): qwen4exp #27742 28 files +2881/-38 (qwen4exp.cpp +1199, llama-memory-hybrid-idx.cpp +465/.h +156, kv-cache +155, conversion/qwen4exp.py +195, gguf constants +101); qwen3.5 #19468 17 files +2096; Kimi-Linear #18755 18 files +1518; MiniMax-M3 #24908 21 files +1044 (minimax-m3.cpp +562, kv-cache +292); GLM-5.3-Flash PRs +2395/+2558/+1983 open
- llama.cpp CUDA constants: FA MMA instances by (DKQ,DV) = 40,64,72,80,96,112,128,192/128,256,320/256,512,576/512 and ncols1 x ncols2 (GQA pack 1..32) chosen by gqa_ratio % k switch (fattn.cu:201-306); gated_delta_net.cu:4 template<int S_v, bool KDA, bool keep_rs_t>, switch S_v 16/32/64/128 (:190-210); 22 mmq type instances
- mistral.rs @b0f26d5cd (2026-09-17): 27 files in mistralrs-core/src/models (qwen3 766, qwen3_moe 835, qwen3_next 1380, glm4_moe 989, glm4_moe_lite 1136, deepseek2 1121, deepseek3 1217, gpt_oss 1324); Qwen3.5 under vision_models/qwen3_5 (text.rs 3137, mtp.rs 210, speculative.rs 1298, packed_gdn.rs 237) and qwen3_5_moe (text 1038); loader block per model in pipeline/loaders/normal_loaders.rs ~240 lines (Qwen3NextLoader :5693-5934: NormalModelLoader + IsqModelLoader + DeviceMappedModelLoader); NormalLoaderType enum :178; CUDA kernels mistralrs-core/src/cuda gdn.cu 6143 (templates on StateT, BK, BV, MAX_K), moe_gemm/gemv, ssm.cu; mistralrs-quant QuantMethod trait (lib.rs:1786, forward/gather_forward) with gguf, fp8, mxfp4, nvfp4, gptq, afq, hqq...
- exllamav3 @0740edc (2026-09-13, box): architecture/*.py 17,355 lines total — glm5_next.py 469 (+ glm5_next_mtp 238), qwen4_exp 328 (+mtp 185), qwen3_5 575 (Qwen3_5Model, Qwen3_5MoeModel, Qwen3_5VLModel, Qwen3_5VLMoeModel in one file), qwen3_moe 204, qwen3 194, qwen3_next 261, deepseek_v4 322 (+mtp 368), glm_moe_dsa 266, gpt_oss 229, minimax_m2 188, nemotronh 311, gemma4 845, architectures.py registry 137; modules/*.py 16,628 (gated_delta_net 1336 with "KDA mode (GLM5.3/Kimi linear attention): per-k-channel decay" :380, mla_attn 1128, dsv4 1779, hyperconnections 591, qsa_indexer 671, ngram_embedding 556, ple 389, mamba2 721, block_sparse_mlp 1513 + _cpu 711, sliding_attn 1258, attn 1351); glm5_next.py composes per layer: layer_types[idx]=="linear_attention" -> GatedDeltaNet(kda keys f_a/f_b/g_a/g_b, gate_lower_bound) else MLAttention(indexer_mode, index_kpool...), HyperConnection(hc_mult, sinkhorn_iters), GatedMLP vs BlockSparseMLP(act_limit, router_type "dots", routed_scaling_factor, n_group, topk_group, shared_experts)
- exllamav3_ext: per-concept CUDA files routing.cu (template<int ACT>), gdn.cu, hc_mix.cu, dsa_topk.cu, dsv4_compress.cu, ngram.cu, ple.cu, softcap.cu, rope.cu, norm.cu; 137 templates
- ik_llama.cpp @41b17995 (2026-09-25): arch names incl glm5next, qwen4exp, qwen35moe, qwen35, qwen3next, minimax-m2/m3, gpt-oss, gemma4, mistral4, glm-dsa, deepseek4/41; src/llama-delta-net.cpp, llama-kda.cpp; no kimi-linear / nemotron_h in grep
- op facts from llama.cpp builders: gpt-oss attn_sinks (openai-moe.cpp:44,115), LLM_FFN_SWIGLU_OAI_MOE (:141), GATING_FUNC_TYPE_SOFTMAX_WEIGHT (:143); MiniMax-M2 q_norm over n_embd_head_k*n_head (minimax-m2.cpp:30-31, whole-q RMSNorm), n_rot 64 of head 128 (:51); Gemma4 final_logit_softcapping (gemma4.cpp:20), proportional rope via rope_freqs (:94-95), per-layer embeddings (:57-60), MoE layer = dense GELU FFN (:304) + MoE GELU (:323-330) with router input scale ffn_gate_inp_s (:319); Nemotron-H-MoE LLM_FFN_RELU_SQR (nemotron-h-moe.cpp:111,126), mamba2 ssm_conv1d (nemotron-h.cpp:87); Qwen3-VL deepstack (qwen3vlmoe.cpp:164-167); GLM4-MoE gating default sigmoid (glm4-moe.cpp:15-17), exp_probs_b (:82); Mistral Small 4 via deepseek2 attention temperature (deepseek2.cpp:40-44); Kimi-Linear KDA layers as recurrent via n_head_kv==0 (kimi-linear.cpp:15-18); MiniMax-M3 MSA block sparse via indexer block_size/top_k/local_blocks (minimax-m3.cpp:10-26)
## Draft models on HF (search 2026-09-26): GLM-5.3-Flash: RedHatAI/GLM-5.3-Flash-speculator.dspark-preview, incoai/GLM-5.3-Flash-DFlash2 (+GGUF by Anbeeld, vcruz305); Qwen3.6-35B-A3B: RedHatAI/Qwen3.6-35B-A3B-speculator.dspark, z-lab/Qwen3.6-35B-A3B-DFlash; MiniMax-M3: nvidia/MiniMax-M3-DSpark; Nemotron-3.5-Lightning: nvidia ...-NVFP4-DSpark/-DFlash; deepseek-ai dspark_qwen3_{4,8,14}b_block7, eagle3_qwen3_*, dflash_qwen3_*, dspark_gemma4_12b; nvidia/Kimi-K2.6-DFlash; XiaomiMiMo/MiMo-V2.5-DFlash; Qwen3.8-Flash-Next: PixelML DFlash only
## MTP inside models: GLM-5.3-Flash nextn 1 (index_share_for_mtp_iteration), GLM-5.x 1, GLM-4.x 1, Qwen3.5/3.6/3.8 mtp_num_hidden_layers 1 (unsloth ...-MTP-GGUF repos), Qwen3.8-Flash-Next MTP 1 (4B, separate GGUF), V4.1 nextn 3 (our roles.rs:39-41 Role::Unused), V4-Flash 1, MiniMax-M2.x num_mtp_modules 3, MiniMax-M3 7, Nemotron-3.5 1, MiMo-V2.6 3, Step-3.7 3; mistral.rs vision_models/qwen3_5/mtp.rs 210 + speculative.rs 1298; llama.cpp qwen35moe.cpp graph_mtp :551; llama.cpp PR #27917 MTP for GLM-5.3-Flash (draft)
## ik glm5next (src/graphs/build_glm5next.cpp, 550 lines): reuses dsv4_hc_mult/ggml_hc_pre (:18-58), build_mhc_post, delta.build_layer_attn_kda (:461), "NoPE absorbed MLA ... When --dsa is active, the indexer builds a sparse top-k mask; else dense" (:463-466), hc_collapse = sum of streams * 1/hc (:520-545); ik swiglu_limit applies to STEP35, BAILINGMOE3, dsv4 archs and GLM5NEXT alike (src/llama-model.h:671-675)
## ChainBody (crates/gpu/src/model.rs:55) ~20 methods: arch derive load decode_input refresh enqueue_chain reset seed_depth set_probe head_eps resident_bytes hybrid_weights load_hybrid load_placed serve_replay decode_pair enqueue_pair take_host_refusal serve_replay_pair rollback
## Our tree constants: gpu/router.rs:38,41 N_EXPERT 64 N_USED 6 (V2-Lite); gpu-deepseek41/router.rs:60,63 N_EXPERT 384 N_USED 6, PER_LANE = N_EXPERT/32 (:72); qwen3moe/router.rs:72,75,79,135 N_EXPERT 128 N_USED 8 MAX_TOKENS 8 NORM_K 2048; flash_gqa.rs:67,69 HEAD 128 GROUP 8; flash_gqa_prefill.rs:178 assert GROUP==8; rope_neox.rs:46 HEAD 128; experts_mxfp4.rs:63,66 N_EXPERT 128 N_USED 3; indexer.rs:65 HEADS 32, HEAD_DIM = index_key WIDTH 128 (index_key.rs:40-41); hc.rs:66-83 HC_STREAMS 4 HC_MIX 24 HC_PIECE 512 HC_MAX_TOKENS 8; attn.rs:70 LATENT 512; compress.rs:62-63 WIDTH 512; engram_gate.rs:57 ROW 5120 (= V4.1 hidden)
## Our tree ops: route_core.rs Score trait SqrtSoftplus + Sigmoid ("compiled but no kernel routes with it yet" :16), softmax in wrappers; experts.rs:19 V4.1 SwiGLU "min(silu(g), L) · clamp(u, -L, L)"; attn.rs: K=V latent attention over window ring ⧺ (compressed prefix | indexer-selected rows), per-head sinks in merge; compress.rs DS4_COMP pool (ratio>=2) and ratio-1 identity rows; index_key.rs 512->128 proj, norm, rope tail, Hadamard; indexer.rs relu-weighted head sum on MMA (hi/lo f16 split) + histogram exact top-k (counts in device words); hc.rs HC_PRE (q3_K fused gemv) / HC_POST / fold / mean; hc_f32.rs ds41_hc_pre_f32; rope.rs NORM-mode tail rope + YaRN tables; rope_neox.rs head_norm_neox_append (per-head RMS + NeoX full 128 + append); flash.rs V2-Lite latent flash (LATENT 512, row 576); flash_gqa*.rs GQA decode/prefill HEAD 128 GROUP 8; gemm.rs int8 MMA GEMM q3k/q4k/q5k/q6k + swiglu quant; iq.rs IQ2_XS/IQ3_XXS/IQ4_XS/Q2_K rows; mxfp4.rs MXFP4 x q8_1; qdot host: Q3_K x Q8_K, Q4_K/Q5_K/Q6_K/Q5_0/Q5_1 x Q8_2_X4, IQ3_XXS x Q8_K, MXFP4 x Q8_2_X4, Q8_0; placement CardFormat (placement.rs:270-299) accepts Q3_K Q4_K Q5_K Q6_K Q5_0 Q5_1 Q8_0 F32 BF16->F32, refuses F16 MXFP4 Q2_K IQ* I* TQ*; tokenizer Pre = deepseek-v3/hunyuan-dense/joyai-llm, qwen2 only (pretok.rs:209-214)
## Absent in tree (grep 0 hits in engine crates): softcap/tanh, conv1d/delta-rule/ssm/mamba/KDA, gelu/relu2, fp8/nvfp4/e4m3, M-RoPE (only vision rope2d)
## SwiGLU limit rule: ik ggml-cuda/unary.cu:74-83 fused_mul_silu_f32: "g = x/(1+expf(-x)); g = min(g, limit); dst = g * max(-limit, min(limit, y))" (clamp AFTER SiLU), used for dsv4 archs and GLM5NEXT alike (src/llama-model.h:671-675, llama-build-context.cpp:1210,1613); our experts.rs:19 same rule; transformers glm5_next clamps gate BEFORE SiLU (modeling_glm5_next.py Glm5NextTextMLP: gate.clamp(max=limit), up.clamp(-limit, limit), act(gate)*up) -> max |diff| per clamped element = L*sigmoid(-L) = 10*4.54e-5 = 4.5e-4 [derived]
## exllamav3 JIT/selector: modules/attention_fn/{dsa_triton,qsa_triton,mla_triton,triton_paged,bc_attn,bc_mla}.py + gated_delta_net_fn/conv1d.py import triton (JIT specialized at runtime); exllamav3_ext/quant/exl3_kernel_map.cu:23 select_gemm_shape(cc, size_m, size_k, size_n, K, multi, ...) picks a precompiled GEMM shape (:147)
## llama.cpp arch intro commits (git log -S in llama-arch.h, fork 9e47962ef): qwen35moe #19468 (2026-02-10); qwen3next #16095 (2025-11-28); qwen4exp #27742 (2026-08-27); glm-dsa #19460 (2026-02-13, "indexer is not yet supported"; indexer #25407 2026-07-24; NextN/MTP spec decode #25980 2026-07-29); deepseek4 #24162 (2026-06-29); deepseek41 commit bea3b8cd3 by vcruz305 (no PR number); kimi-linear #18755; minimax-m2 #16831; minimax-m3 #24908; gpt-oss #15091; mistral4 #20649; nemotron_h_moe #18058; glm4moe #14939; qwen3vl(moe) #16780; gemma4 first seen in #21309 (a fix; initial PR 미확인)
## Qwen3-Next/3.5-MoE HF modeling: attention output sigmoid gate (modeling_qwen3_next.py:308, modeling_qwen3_5_moe.py:822), softmax router + norm_topk (:838 / :896), sigmoid shared_expert_gate (:853-862 / :910-919)
## GLM-5.3-Flash non-expert params read per token [derived]: KDA 34 x 137.8M = 4.69B; MLA+indexer 11 x 125M = 1.37B; dense FFN 3 layers 0.45B; shared experts 42 x 25.2M = 1.06B; routers 0.05B; hc 0.04B; lm_head 0.63B => 8.29B; at Q8_0 8.5 bpw (survey Table C: UD-Q4_K_XL attn and shexp Q8_0) = 8.81 GB/token -> 12.6 ms @ ~700 GB/s (AGENTS kernel ceiling figure); experts in UD-Q4_K_XL = 199.7 - 10.2 = 189.5 GB for 311.7B params = 4.86 bpw; per-token routed 8.46B params -> 5.14 GB -> 37.8 ms @136 GB/s all-host; UD-Q3_K_XL experts 137.3 GB (3.52 bpw) -> 3.72 GB/token -> 27.4 ms @136
## V4.1 host set 214.0 GB for 12,692 experts (AGENTS), facts.md:64 V4.1 16.77 MB x 6 per layer = 0.684 ms/layer @147.7 GB/s; qdot-rate-mt kernel MT ceiling 122.6-136.3 GB/s (facts.md:108); spec says ~132-136 GB/s -> report uses 136 GB/s
```

## 부록 B. 적합성 산술 (`fit.txt`, 작업 파일 원문)

```text
model                            expert params B per-tok expert params B
GLM-5.3-Flash                              311.7                    8.46   kv 16896 B/tok (0.55 GB @32k), state 143 MB/seq  42 sparse + 1 MTP layer
      UD-Q3_K_XL         file  147.5 GB
      UD-Q4_K_XL         file  199.7 GB
      Q8_0               file  341.0 GB
Qwen3.6-35B-A3B                             32.2                    1.01   kv 20480 B/tok (0.67 GB @32k), state 63 MB/seq  all layers MoE
      UD-Q4_K_XL         file   22.4 GB
      Q8_0               file   36.9 GB
Qwen3.8-Flash-Next                         120.8                    2.36   kv 27648 B/tok (0.91 GB @32k), state 113 MB/seq  +51B n-gram table (card: host lookups)
      UD-Q4_K_XL         file  111.3 GB
      Q8_0               file  195.1 GB
MiniMax-M3                                 413.1                   12.91   kv 137472 B/tok (4.50 GB @32k), state 0 MB/seq  3 dense layers
      UD-Q3_K_XL         file  194.9 GB
      UD-IQ4_XS          file  207.6 GB
      UD-Q4_K_XL         file  264.9 GB
MiniMax-M2.7                               224.7                    7.02   kv 253952 B/tok (8.32 GB @32k), state 0 MB/seq  
      UD-Q4_K_XL         file  140.8 GB
      Q8_0               file  243.1 GB
gpt-oss-120b                               114.7                    3.58   kv 73728 B/tok (2.42 GB @32k), state 0 MB/seq  SWA half the layers
      MXFP4(F16 file)    file   65.4 GB
      Q4_K_M             file   62.8 GB
Qwen3.5-397B-A17B                          386.5                    7.55   kv 30720 B/tok (1.01 GB @32k), state 189 MB/seq  
      UD-IQ4_XS          file  189.7 GB
      Q4_K_M             file  244.1 GB
Qwen3.5-122B-A10B                          116.0                    3.62   kv 24576 B/tok (0.81 GB @32k), state 151 MB/seq  
      Q4_K_M             file   76.5 GB
      Q8_0               file  129.9 GB
GLM-5.3 (non-Flash)                        734.4                   22.65   kv 89856 B/tok (2.94 GB @32k), state 0 MB/seq  75 sparse + 1 MTP
      UD-IQ1_S           file  216.7 GB
      UD-IQ2_M           file  238.6 GB
      UD-Q2_K_XL         file  253.9 GB
Kimi-Linear-48B-A3B                         47.1                    1.47   kv 7168 B/tok (0.23 GB @32k), state 42 MB/seq  
      Q4_K_M             file   30.1 GB
      Q8_0               file   52.2 GB
Nemotron-3.5-Lightning-30B-A3B              29.4                    1.38   kv 6144 B/tok (0.20 GB @32k), state 48 MB/seq  relu2 experts are 2 matrices
      UD-Q4_K_XL         file   25.5 GB
      Q8_0               file   35.0 GB
DeepSeek-V4.1-Flash (ours)                 543.6                    8.49   kv 0 B/tok (0.00 GB @32k), state 0 MB/seq  file includes engram tables
      antirez Q2         file  365.7 GB
```
