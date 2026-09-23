DSpark draft 설계 메모입니다. 읽기 전용으로 진행했고 파일 편집, 게이트, 빌드는 하지 않았습니다. 박스에서는 `cat`/`sed`/`grep`/`awk`/`ls` 읽기만 했습니다. 숫자 가운데 [유도] 표시는 파생값이고 측정값이 아닙니다. 나머지 숫자는 파일 헤더이거나 rig-log와 plan.md가 적은 측정값입니다.

## 먼저 알아야 할 것

- **ik의 draft 그래프는 V4 모양이다.** `ik-idxkey`와 `ik_llama.cpp`의 `build_dflash_dsv4`(`build_deepseek4.cpp:1727`)에는 세 가지가 남아 있습니다.
  - q의 헤드별 `ggml_rms_norm`이 무조건 걸립니다.
  - HC가 lag 없이 계산됩니다.
  - 학습된 `hc_head`로 collapse합니다.
  - 로더(`llama-load-tensors.cpp:2818`)는 `output_hc_*`를 필수로 요구하는데 tl37 파일에는 이 텐서가 없어서 적재가 실패합니다.
- **고치는 패치는 스크립트로만 있다.** V4.1 draft 반쪽 패치는 `/home/user/ik-dsv41-draft.py`에만 남아 있고, 박스의 어느 ik 트리에도 적용돼 있지 않습니다. 트리 11개를 awk로 확인했습니다. 따라서 ik 오라클은 이 스크립트를 적용한 트리를 새로 빌드해야 생깁니다.
- **ik 결함은 rig-log가 적은 둘 말고 둘이 더 있다.** 우리 포트가 따라 하면 안 됩니다.
  - 과거 창 마스크가 `q_pos − k_pos < 128`입니다(`llama-dflash.cpp:707-720`). 블록 행 j가 링에서 가장 오래된 1+j칸을 못 봅니다. `model.py`에서는 블록의 모든 행이 링 128칸 전체를 봅니다(`get_dspark_topk_idxs`의 `arange(min(win, start_pos+1))`).
  - 블록 길이가 `n_max`입니다(`speculative-dflash-impl.h:334,372`). `[id_last, mask×(k−1)]`만 돌립니다. `model.py`가 학습한 블록은 늘 5토큰이고 블록 안 어텐션이 양방향이라, k토큰 블록은 5토큰 블록을 자른 것과 다른 분포입니다. rig-log의 60.1 %는 3토큰 블록에서 잰 값입니다.

## 1. draft의 수학 (`model.py` 1021–1163, 1273–1282)

w는 블록 폭입니다(`model.py`는 5, ik는 k). r+1은 이번 패스에 확정된 target 위치 수입니다. 체크포인트 원본에서는 target 특징(3×5120)이 블록 임베딩에 더해지지도, 게이트되지도, 이어 붙지도 않습니다. 각 draft 층의 과거 KV 링으로 들어갑니다.

| # | 연산과 모양 | 우리 커널 |
|---|---|---|
| 1 | 특징 = 4스트림의 **평균**. target 37–39층 attention 입력(=36–38층 출력)이고 `h.mean(dim=2)`라서 `pre` 가중 접기가 아님 → [r+1, 15360] | **신규** 탭: 36–38층 ffn HC_POST 뒤의 평균 |
| 2 | `main_x = rms(enc.output_norm)(fc·feat)`, fc Q8_0 [15360→5120] | `q8_0_gemv` m=r+1, `elem::rms_norm` (fc는 **신규 가중치**) |
| 3 | 층마다 `main_kv = kv_norm(wkv·main_x)`, 위치 p에서 tail rope 64, 링 칸 p%128 | `q8_0_gemv` + `ds41_kv_norm_rope_append` (`RopeSpec::window`) |
| 4 | 블록 입력 ids `[id_last, 128799×(w−1)]`, target `token_embd`(bf16), 4스트림으로 복제, `pre`=one-hot(0) | glue의 임베딩 브로드캐스트 |
| 5 | 층×3, attention: `q = wq_b·rms(wq_a·x)` (헤드별 norm 없음), rope 위치 p+1…p+w | `q8_0_gemv`, `rms_norm`, `ds41_rope_tail` |
| 6 | 블록 kv = `kv_norm(wkv·x)` + rope, w행 | `ds41_kv_norm_rope_append`(w칸짜리 버퍼) |
| 7 | 키 = 링 min(p+1,128) ⧺ 블록 w행. 모든 쿼리가 둘 다 전부 봄(과거 쪽 인과, 블록 안 양방향). sink 포함, scale 512^−½, 출력 역 rope | `ds41_attn_seg` + `ds41_attn_merge`. 블록 행을 둘째 출처(압축 행 자리)로 두고 `vis[2t+1]=w` |
| 8 | `wo_a` 8그룹 [4096→1024] Q8_0, `wo_b` [8192→5120] | `q8_0_gemv_heads`, `q8_0_gemv` |
| 9 | HC 4스트림, **lag**, Sinkhorn 20, `hc_*_fn`가 **F32** [20480,24] | `ds41_hc_post`는 그대로. `ds41_hc_pre`는 q3_K 가중치 전용이라 **f32 변형이 신규** |
| 10 | MoE: sqrt-softplus, bias 선택, 128개 중 top-3, 정규화 후 ×1.5, gate_inp bf16 | `ds41_router`. `N_EXPERT=384`/`N_USED=6` 상수라 **일반화 필요** |
| 11 | routed experts SwiGLU(±10) MXFP4 [5120↔2304] | **신규**(아래 3절) |
| 12 | shared expert Q8_0 | `ds41_shexp_gate_up` + `q8_0_gemv` |
| 13 | 헤드: 마지막 ffn `pre`로 접기, draft 자기 `output_norm`, **target** 헤드(Q6_K) m=w | `ds41_hc_fold` + `rms_norm` + q6k gemv. `Head`는 m=1 전용이라 m=w가 필요 |
| 14 | Markov 루프 i=0…w−1: `logits_i += markov_w2·markov_w1[out_i]`(bf16 [256×129280]) → argmax → out_{i+1}. out_0 = id_last. 위치마다 직렬 | **신규**: bf16 gemv(K=256, 행 w1[id] 모으기와 logits 더하기를 한 커널로) + `argmax` |
| 15 | 신뢰도 `conf_proj·[x_hc_pre ⧺ markov_e]`(5376→1) | **신규 XS**. ik는 쓰지 않음. 1차에서는 끄고 적응형 k 연구용으로 둠 |

`model.py`에는 있고 ik와 우리에게는 없는 반올림류 차이가 둘 있습니다.
- `model.py`는 kv를 fp8로 양자화했다가 되돌립니다(`act_quant`). ik와 우리는 f16 링입니다.
- SwiGLU clamp 위치가 다릅니다. `model.py`는 `silu(min(g,10))`, 우리 커널(ik 규칙)은 `min(silu(g),10)`이라 g>10일 때만 약 5e-5 차이가 납니다.

rope는 plain입니다(yarn 없음, `compress_ratio 0`). GGUF 헤더의 yarn ×16은 무시해야 합니다. ik도 무시하고, rig-log도 "rope는 맞다"고 정정해 두었습니다.

## 2. ik가 draft를 모는 방법

- **특징 캡처.**
  - 캡처는 이름이 `l_out-<il>`인 텐서입니다(`build_deepseek4.cpp:1646`).
  - 값은 층 출력 4스트림의 평균이고 f32입니다(`dsv4_hc_mean_for_capture`, `:12`).
  - eval 콜백이 `tensor_get_async`로 호스트에 읽어 냅니다(`llama-spec-features-dflash.cpp:499-590`). 호스트를 거치는 방식이고 캡처 지점마다 스케줄러가 그래프를 끊습니다.
  - 특징 행은 호스트 링 `target_window_ring`(cross_ctx×15360 f32)에 쌓입니다. 확정된 행만 append하고 전환 계획에 따라 rebuild합니다(`speculative-dflash-impl.h:476-600`).
- **draft KV.**
  - 별도 그래프 `build_dflash_kv_cache`(`build_dflash.cpp:114`)가 fc → hidden_norm → 층마다 wkv → norm → rope(target 위치)를 거쳐 `set_rows`로 링 칸 `(write+i)%cross_ctx`에 씁니다.
  - 가시 범위는 `llama_set_dflash_visible_cross_ctx`로 정하고, 실제 창은 SWA 마스크(128)가 자릅니다.
- **마스크.**
  - 블록 안은 `!cparams.causal_attn || block_k ≤ j`입니다(`llama-dflash.cpp:718,746`). 포크는 이 자리를 draft 전용 플래그로 고쳤습니다(`515a94a3`).
  - 과거 쪽의 1+j 결함은 위에 적었습니다.
- **헤드.**
  - `build_dspark_logits`(`build_dflash.cpp:66`)가 Markov 루프를 그래프 안에서 돌립니다.
  - `conf_proj`는 적재만 하고 쓰지 않습니다.
- **수락.**
  - `common_sampler_sample_and_accept_n`(`sampling.cpp:790`)을 씁니다.
  - target이 뽑은 토큰이 draft와 같은 동안 받고, 첫 불일치에서 target 토큰을 붙인 뒤 멈춥니다.
  - 전부 맞으면 보너스 토큰 하나를 더 받습니다.
  - 거절은 서버의 `llama_kv_cache_seq_rm`(`server-context.cpp:3327`)이 처리합니다.
- **rig-log가 적은 결함 둘**(09-13)은 target_layers off-by-one(tl37로 고침)과 블록 causal 마스크입니다.
- **우리가 피할 것.** 위 둘과 함께 이 메모가 찾은 과거 창 1+j 결함, V4 모양 그래프(패치 없는 트리)를 따르지 않습니다.

## 3. 우리 설계

**배치.** draft는 3090에 둡니다. (a)에서 3090은 놀고 있고, A6000은 (a) 적재 뒤 여유가 약 1.5 GB뿐입니다.
- draft 전용 `Gpu::with_device`와 캡처 그래프를 w마다 하나씩 둡니다. 추가된 행 수는 장치 파라미터로 읽습니다.
- 링은 3층×128×512 f16이고, 스테이징 k+1행과 확정 복사를 둡니다. 거절된 위치를 링에 바로 쓰면 아직 필요한 p−128+i 칸을 덮으므로 스테이징이 필요합니다.
- **호스트를 거치는 이동.** 확정 행마다 61,440 B를 D2H한 뒤 H2D하고, draft id를 되돌려 받습니다. 동기화 두 번에 약 0.05–0.1 ms로 가정했습니다[유도, 지연은 가정].
- **임베딩은 복사하지 않는다.** 우리 glue는 이미 스텝마다 bf16 행을 파일에서 이미지로 올립니다. draft 블록에 필요한 행은 id_last 한 줄과, 적재 때 캐시하는 mask 행 한 줄뿐입니다. 1.32 GB를 복사할 필요도, 테이블을 오갈 필요도 없습니다.
- **헤드는 3090으로 복사한다.** Q6_K 543 MB를 적재 때 한 번 복사합니다. 반대로 A6000에서 헤드를 돌리면 위치마다 hidden을 두 번 오가야 하고, draft 중에도 A6000을 붙잡아 겹치기를 막습니다. 바이트 시간은 두 경우가 같습니다(약 0.78 ms[유도]).

**expert 형식.** 권고는 (iii) MXFP4 네이티브 커널입니다. rig-log 09-13에 따르면 체크포인트의 draft expert는 원래 fp4이고, 32개마다 E8M0 스케일 하나를 가집니다. 따라서 이 GGUF는 체크포인트와 비트가 같습니다.

| 안 | 3090 상주 | 패스당 routed 바이트 | 수락 | 새 작업 |
|---|---|---|---|---|
| (i) Q8_0 | 14.44 GB + 1.30 GB | ×2 | 무손실 | q8_0 `_sel` 커널(우리에게 없음) + 재변환 |
| (ii) Q4_K | 7.64 GB + 1.30 GB | ×1.06 | **손실**. 균등 16준위로는 {0,.5,1,1.5,2,3,4,6}을 못 담고 미측정 | q3_K gate·up 경로 대신 q4_K `_sel`을 두 번 |
| (iii) MXFP4 | 7.22 GB + 1.30 GB | ×1 | 비트 동일 | 새 `_sel` 커널 둘 |

(i)에 대한 보충입니다. 재변환기 `convert_hf_to_gguf.py --dspark`는 존재합니다(`/home/user/llama.cpp-v41`, 129행). 그런데 fp4 값을 Q8_0으로 옮기면 {0,±1,±2,±3,±4,±6,±8,±12}×2^(E−128)입니다. 적재 때 정확하게 넓힐 수 있어 변환기는 필요 없습니다(E가 f16 범위 안인지 검사). 그래서 (i)는 대체 경로로만 둡니다.

(iii)의 디코드입니다.
- 블록 하나는 17 B입니다(e8m0 바이트 E, 니블 16 B). 코드 c를 `kvalues={0,1,2,3,4,6,8,12,0,−1,−2,−3,−4,−6,−8,−12}` int8 표로 바꾸고, 스케일 d_w=2^(E−128)(ggml의 `e8m0_to_fp32_half`)을 곱합니다.
- q8_1 활성값과는 dp4a로 int32 합을 낸 뒤 d_w·d_x를 곱합니다. 표 조회는 u32 두 워드에서 byte-permute하거나 공유 메모리 LUT로 합니다.
- 호스트 참조 dequant는 이 규칙 그대로의 10줄입니다.

(iii)을 권하는 이유는 셋입니다.
- (i)과 (iii) 모두 새 `_sel` 커널이 필요하고 크기도 같습니다.
- (iii)의 바이트가 절반입니다.
- 8.5 GB라서 (b′)에서 3090에 target expert 자리가 남습니다.

draft를 3090에 두는 것은 (b)(3090이 target 스테이지를 가짐)와 양립하지 않습니다.

**draft 비용 D.** 3090, BW 700 GB/s(plan.md의 3090 행)로 셉니다. 패스마다 고정 1,162.5 MB가 듭니다. 층 셋의 dense(attn 134.5 + hc 3.9 + gate 2.6 + shexp 37.6 MB), fc 83.6, 헤드 543입니다. 여기에 Markov 66.2 MB×k와 routed가 붙습니다. routed는 서로 다른 expert 수 n_d(w) = 128(1−(125/128)^w)로 셌는데, 독립 가정이라 상한입니다.

| k (=w) | n_d/층 | 바이트 MXFP4 / Q8_0 | 바이트 시간 | D, 노드·t_floor 약 0.4–0.6 ms 포함 |
|---|---|---|---|---|
| 1 | 3 | 1,398 / 1,567 MB | 2.00 / 2.24 ms | **2.4–2.8 ms** [유도] |
| 2 | 5.93 | 1,629 / 1,964 | 2.33 / 2.81 | 2.8–3.4 |
| 3 | 8.79 | 1,857 / 2,353 | 2.65 / 3.36 | 3.1–4.0 |

ik의 10.32 ms와 비교하면 3–4배 낮은 쪽입니다. 전제가 하나 있습니다. m>1 q8_0 몸통과 m>1 Q6_K 헤드가 가중치를 한 번만 읽어야 합니다. 지금 커널로는 `wo_a`/`wo_b`/헤드를 토큰마다 다시 읽어서 k=3에서 약 5 ms 이상이 됩니다[유도]. 이 커널은 ktok 검증 단계와 같은 필요입니다(plan.md 521, 524행).

**값이 나는 곳.** 1스텝 T1은 39.3 ms입니다(b5time, 깊이 400, 측정). 검증 토큰 하나를 더하는 비용 Δ는 routed 몫 약 67 %로 보아 26.3 ms입니다[유도]. 패스당 기대 토큰 E(k)는 ik 실측으로 1.702, 2.140, 2.434입니다.

| 조건 | k=1 손익분기 D | D=2.6일 때 |
|---|---|---|
| 지금 (a) | 1.3 ms | −2 %, 손해 |
| B12 뒤(T1 약 30.8, Δ 약 15.8, plan.md의 −7…−10 ms 중앙) | 5.8 ms | **+6.5 %**. k=2는 +1 %, k=3은 −8 % [유도] |

k*는 1입니다.

**둘째 스트림.** 특징→KV 몫(fc와 wkv, 약 0.2 ms)만 target의 39층과 헤드 뒤에 숨길 수 있습니다. 블록 패스는 id_last를 기다려야 하므로 직렬입니다. 나머지는 "링이 한 패스 늦은 draft-ahead" 안입니다.
- 검증과 동시에 다음 블록을 draft합니다. "k개 전부 수락"과 옛 링을 가정합니다.
- 드러나는 D는 (1−p_all)·D이고, 옛 링 때문에 잃는 수락률이 붙습니다. 그 손실은 미측정이라 ik 쪽 실험 하나가 필요합니다.

## 4. 검증

- **오라클 A (ik draft 덤프).**
  - `tools/ref/dump_ref.cpp`는 dflash 컨텍스트에 닿지 않습니다(`dflash` 문자열 없음).
  - 필요한 것은 둘입니다. ① `ik-idxkey` 사본에 `ik-dsv41-draft.py`를 적용하고(스크립트의 ROOT 수정) 마스크 플래그를 확인한 트리 ② 새 `dump_draft.cpp`로 `ctx_dft`에 eval 콜백을 걸어 `dsv4_dflash_q/kv/attn`, `dflash_kv_fused_target`, `dflash_base_result_output`, `result_output`, `draft_argmax`를 받는 것.
  - 과거 창 1+j 결함은 깊이 < 123에서 두 규칙이 같아지므로, 게이트는 그 깊이에서 겁니다.
- **오라클 B (`model.py`).**
  - `kernel.py`는 tilelang(fp8 gemm, sparse_attn)이 필요하고 박스의 python에는 torch가 없습니다(`ModuleNotFoundError`).
  - 순수 torch로 다시 짜고 가중치를 쓸 때마다 dequant하면, 팩 그대로(mtp 약 8 GB + embd·head bf16 2.6 GB) 약 11 GB라 3090에도 들어갑니다[유도].
  - bf16 전체 전개는 28 GB라 A6000에서, 측정과 측정 사이에만 가능합니다.
  - 입력은 우리 엔진이 덤프한 특징으로 두는 순수 함수 오라클입니다. 그러면 target 차이와 분리됩니다.
- **게이트.**
  - G-d1: 고정 입력으로 층별 포락선, draft id 일치, 편향 logit top-1 마진.
  - G-d2: 프롬프트 세트에서 draft id가 ik와 일치하는지. rig-log 09-13의 스무 개가 어느 것인지는 리드가 고정해야 합니다(`prompts.tsv`는 32개).
  - **G-d3 무손실**: draft를 켠 greedy 연속이 끈 것과 토큰 단위로 같아야 합니다. ktok 게이트("k토큰 스텝 + 롤백 = r+1 순차 스텝, 비트 동일")가 서면 이 게이트는 밴드가 아니라 등식입니다.
  - G-d4: 위치별 수락률을 ik 0.702/0.570/0.478과 비교합니다. 이항 SE는 약 3000 draft에서 ±0.9포인트[유도]이고, mainline 대 ik 1.5포인트가 알려진 폭입니다.
  - D는 리드가 한 번 재고 유도 밴드와 대조합니다.

## 5. 라운드와 사용자 결정

| id | 내용 | 파일 | 게이트 | ktok에서 받을 것 | 크기 |
|---|---|---|---|---|---|
| `dsread` | `GgmlType::MXFP4`(39)와 호스트 dequant, dspark hparams·이름, 3090 적재 계획(헤드 복사, bf16→f32 넓힘) | `gguf/quant.rs`, `model::arch` 새 모듈, `weights.rs` | 호스트 dequant가 ik dequant와 비트 동일, 적재 바이트 = 위 표 | 없음 | S |
| `dsref` (리드, 박스) | ik draft 트리 빌드와 `dump_draft.cpp` | `tools/ref` | 덤프 MANIFEST | 없음 | S–M |
| `dsmx` | MXFP4 `_sel` gate·up·SwiGLU와 down(m ≤ 8), 라우터 128/3 일반화, 3슬롯 combine | `experts.rs`, `router.rs`, 새 `mxfp4.rs` | 호스트 dequant+f32 참조와 밴드 비교 | 다토큰 expert id 배치 규약 | M |
| `dsm` | m ≤ 8 q8_0 몸통(heads 포함), Q6_K 헤드 m>1 | `q8f32.rs`, `head.rs` | 기존 m=1과 비트 동일, m>1은 열 순서 규칙 | **ktok과 공유**. 먼저 끝나는 쪽이 소유 | M |
| `dshc` | f32 가중치 HC_PRE(K=20480), bf16 Markov gemv+add+argmax | `hc.rs` 옆, 새 커널 | `hc_pre_lane` 호스트 규칙과 비트 동일 | 없음 | S |
| `dsgraph` | draft 본체(링, 스테이징, 블록 패스, 헤드, Markov), w마다 그래프 하나 | 새 `gpu-deepseek41/src/draft/` | G-d1(깊이 < 123), 노드 수 핀 | 없음 | M–L |
| `dsloop` | target 36–38 탭, 호스트 거친 이동, 검증, 확정 | `body.rs`, `gpu/model.rs`(B5 뒤 점유) | G-d3, G-d4 | k토큰 스텝과 롤백 | M |

순서는 `dsread` ‖ `dsref` ‖ `dsm`(ktok과 함께) → `dsmx` ‖ `dshc` → `dsgraph` → `dsloop` → 리드가 D와 tok/s를 잽니다. 박스 작업 추정은 ik 빌드 10–25분, 프롬프트 세트 수락률 한 회차 약 15분입니다[유도]. 둘 다 30분 안이지만 한 번에 묶기를 권합니다.

**사용자에게 가져갈 결정 둘.**
1. **expert 형식.** MXFP4 커널(권고)과 적재 때 Q8_0으로 정확히 넓히기(+7.2 GB, 패스당 +0.24–0.7 ms), Q4_K(손실, 미측정) 가운데 고릅니다. 이 선택은 3090을 (b′)에 쓸 여지도 함께 정합니다.
2. **fp8 오라클을 어디서 돌릴지.** ik 패치 트리 덤프(박스 빌드 한 번)와 torch venv를 설치하는 `model.py` 순수 torch 참조(3090의 게이트 락 아래, 또는 측정 사이의 A6000) 가운데 고릅니다. 리드는 A를 먼저, B는 편차 심판으로 쓰는 것을 권합니다.

---

**변경한 파일.** 없습니다(읽기 전용).

**증명·게이트.** 실행하지 않았습니다(스펙상 없음). 판정의 근거가 된 읽기는 다음과 같습니다.
- 트리 11개에서 `build_dflash_dsv4` 안의 `hc_lag`/`pre_in`/`q_head_norm`이 0건이었습니다. 전 트리에서 `hc_head_fn` assert가 살아 있습니다.
- `grep dflash tools/ref/dump_ref.cpp`는 0건이었습니다.
- 박스 `python3 -c "import torch"`는 `ModuleNotFoundError: No module named 'torch'`였습니다.

**예측.** D(k=1) 2.4–2.8 ms, B12 뒤 k=1은 +6.5 %, (a)의 손익분기 D는 1.3 ms입니다. 모두 [유도]입니다.

**못 한 것.**
- 3090 BW는 plan.md의 700 GB/s를 썼고, 250 W 캡 아래의 값은 확인하지 못했습니다.
- q6k gemv의 m>1 경로는 존재만 확인했고 비용 규칙은 읽지 못했습니다.
- A6000과 3090 사이 P2P 여부는 auditor ③의 몫이라 호스트를 거친다고 가정했습니다.
- advisor 호출은 시간 초과로 실패했습니다.

**스펙 밖 개선 지점**(보고만 했고 손대지 않았습니다).
- `crates/gpu-deepseek41/src/router.rs:41,44`: `N_EXPERT`/`N_USED`가 컴파일 상수라 128/3 draft에 못 씁니다. const generic으로 바꾸면 됩니다(S).
- `crates/gpu-deepseek41/src/experts.rs:49`: combine 슬롯 수가 리터럴 6입니다. 같은 문제입니다(S).
- `crates/gpu/src/head.rs:7,141`: m=1 assert 때문에 draft와 검증 모두 막힙니다(S–M).
- `crates/gguf/src/quant.rs:33`: MXFP4가 없어 strict 리더가 draft 파일을 거부합니다(XS–S).
- `crates/gpu-deepseek41/src/experts.rs:23`: SwiGLU clamp가 ik 규칙이고, 참조(`model.py`)는 g를 먼저 자른다는 사실이 모듈 문서에 없습니다(XS, 문서).
- `crates/gpu-deepseek41/src/attn.rs:23`: "창이 링을 감는 배치는 계약 밖"이라, pos ≥ 128의 모든 k토큰 검증이 여기에 걸립니다. ktok 항목입니다(M).

**모델.** opus로 스폰됐고 Claude Opus 5.5로 실행했습니다.
