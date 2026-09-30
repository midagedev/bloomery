# GLM-5.3-Flash — 새 연산, 세 엔진 대조, 라운드 분해

라운드 `glmops`(재구성 파동 3, 읽기 전용 설계, 2026-09-27)의 메모를 보고한 그대로 두고, 위에 리드의 처분을 적는다.
메모의 수치는 출처를 이름으로 대는 것 말고는 전부 [유도]다(코드를 돌리지 않았다; 박스 명령은 읽기만). 헤더 값은
`glm5next` 라운드가 스크래치에 남긴 UD-Q4_K_XL 샤드 1 덤프에서 읽었다.

## 리드의 처분 (2026-09-27)

- **G1 정정: mHC epsilon은 1e-6이다.** 샤드 1의 `hyper_connection.epsilon`은 9.999999974752427e-07이다. 두 스펙(`glmops.md`,
  `glm5next.md`)의 1e-7은 리드의 오독이었다. 레포 문서에는 1e-7이 없다(grep). glm5next 라운드에는 "헤더에서 읽은 값을 핀하라,
  스펙 수치가 아니라"고 보냈다(턴 끝에 읽힌다; 착륙 diff에서 핀 값을 확인한다).
- **G2 첫 프로그램은 밀집 MLA + 인덱서 키 캐시 쓰기, 한계 너머는 이름 붙은 거부.** 받는다. ik의 k-pool 인덱서는 기본 꺼짐
  (`dsa = false`, `--dsa` 옵트인·EXPERIMENTAL)이고, 켜도 한 질의가 보는 온전한 풀이 512개 이하(위치 약 2,051 이하)이면 모든
  풀을 고르므로 밀집과 같다. qwen4exp의 QSA와 같은 모양(`research/qwen4arch-design.md` Q2)이라 두 모델의 선택기 라운드가 같은
  `TokenPool` 계열 하나를 나눈다. **사용자 결정 둘**(트리아지에 올린다): ① 한계 너머에서 학습 의미(희소)를 따를지 ik 기본
  (밀집)을 따를지 — 리드 권고는 희소(모델이 그렇게 학습됐고, ik의 밀집은 실험 플래그의 기본값일 뿐) ② 그 구간의 오라클 세트를
  `--dsa`로 뜰지 — 권고는 예.
- **G3 카드 트렁크는 토큰당 8.97 GB, 벽시계 (a) 41–43 ms(23–24 tok/s)[유도].** 실제 헤더로 센 비-expert 0–44층 8.292 GB +
  Q8_0 output 0.674 GB가 q5kexpdesign의 대리 하한 6.48 GB를 대체한다. routed가 아닌 텐서는 전부 Q8_0 또는 F32(token_embd도
  Q8_0)라 카드 쪽 형식 항목은 Q8_0 gemv 하나로 모인다. 호스트 Σ브리지 29.6 ms가 지배하고 카드 커널 개선은 합의 약 30 %만
  움직인다 — 카드 최대 항은 KDA 투영(Q8_0 m=1 gemv 4.97 GB/토큰)이고 delta 커널이 아니다. `q8_0_gemv`의 m=1 GB/s는 실측이
  없다(§4-2): GLM 시팅의 nsys 한 번에 얹는다. 가장 큰 레버는 검증 m열(MTP 1층 또는 DSpark 초안) — e2e 뒤.
- **G4 KDA는 03의 `delta_body::<DECAY_KEY>` 위의 엔트리 셋(라운드 `glmkda`, 03, S–M, q35gdn 착륙 뒤).** `kda_delta`, KDA 준비
  (키 채널별 감쇠 `exp(lb·σ(−ssm_a·(f+dt)))`), `norm_gate::<GATE_SIGMOID>`. conv 가중치 셋과 `W_q|k|v`는 적재 때 이어 붙인다.
  ik 오라클과 다른 세 곳(내적 순서, libm `expf` 대 `expf_ik`, **ik의 매 스텝 ±1e6 상태 클램프** 대 03의 fault)은 이름 붙은
  행동 차이로 계약에 적는다 — "자기 호스트 규칙과 비트 동일 + ik와 밴드". 클램프 건은 03 DONE 릴레이 때 한 줄로 전한다
  (조용한 값 보정 대 fault: AGENTS 조용한 실패 금지 조항으로 03의 fault가 맞다). 청크 KDA(fla `chunk_kda`)는 호스트 union
  그늘 아래라 지금 가치 0.
- **G5 MLA는 V4.1 `ds41_attn_seg`·`merge`와 `q8_0_gemv_heads`를 런타임 인자만으로 재사용(라운드 `glmmla`, aa, M).** window 0,
  sinks −∞(`online_fold`에서 정확한 no-op[코드 유도] — 첫 게이트에서 한 번 확인), scale 1/√256. 새 것은 rope 없는 잠재 RMS +
  f16 추가, LayerNorm(128, bias, ε 1e-6) + [k; gate] 캐시 추가 둘뿐. opslib의 attn 계열 이동 뒤. 32k에서 밀집 대 희소의 디코드
  차이는 약 0.5 ms/토큰(1 %)이라 선택기의 디코드 가치는 속도가 아니라 학습 의미다(프롬프트에서는 8배 FLOP 레버).
- **G6 라우터 288/8은 `ds41_router` 인스턴스 행(라운드 `route288`, aa, S).** E 288(PER_LANE 9, SCORE_TILES 36), K 8, Score Sigmoid,
  scale 2.5; 편향은 선택에만. 우리 `+1e-20` 가드는 ik GLM에 없지만 σ 8개 합이 2^-42 아래(선택 로짓 8개 모두 −29 아래)일 때만
  비트를 바꾼다 — 이름 붙은 편차로 둔다. opslib의 router 계열 이동 뒤(모듈 상수 384·계약 리터럴 셋이 그때 인스턴스 키가 된다).
- **G7 mHC는 Own 규칙 — 부층당 런치 3(라운드 `hcq8`, aa, S).** `hc_post`(fold 출력 버림) → `hc_pre` → `hc_fold`. 새 것은
  `hc_pre`의 Q8_0 fn 인스턴스(V4.1 split-K 골격에 Q8_0 조각 내적). 붕괴 `ds41_hc_mean`은 비트 동일. `hc_post`의 버려지는 fold
  쓰기를 접는 레버는 GLM 시팅 뒤.
- **G8 풀 키는 exllamav3의 불변 풀 평면(라운드 `glmsel`, aa, M, e2e 뒤).** 완성된 풀마다 한 번 계산해 두면 ik의 매 스텝
  O(n) 재풀링과 연산 순서가 같아 비트 동일이고 캐시가 25 % 준다. f32 점수는 SIMT 새 커널(V4.1 `ds41_indexer_score`는 rope·
  Hadamard·텐서코어가 박혀 있다), top-k는 `ds41_indexer_topk` 재사용. 동점 규칙(ik `ggml_argsort` DESC를 iqk가 덮는지)은 그
  라운드의 첫 grep. ik의 O(n) 재풀링은 업스트림 후보지만 EXPERIMENTAL 경로 + 메인테이너 부하 규칙 → 트리아지 업스트림 절에만.
- **G9 클램프 SwiGLU Q8_0 gate|up + act 카드 커널(라운드 `ffnq8act`, S).** q5kexp의 `kq_gate_up_act` 계열(act 런타임 인자)에
  Q8_0 워크 행을 더하는 것이 1안; e2e 최소안은 `q8_0_gemv` ×2 + 원소별 act 런치(+1 런치, 바이트 같음). 규칙은 ik
  `min(silu(g), L)·clamp(u, ±L)`(exllamav3 동일, transformers만 SiLU 앞 클램프) — 이미 `experts.rs` `swiglu_clamp`에 있다.
- **G10 오라클은 `/home/user/ik-idxkey` `db517b69` 그대로(라운드 `glmref`, aa, S, `/root/glmdl.rc` ok 즉시).** `libllama.so`에
  glm5next가 있고 V4.1·qwen35moe와 같은 `IK_BUILD`를 핀할 수 있다. 세트 셋(`ref_glm5next` 32토큰 CPU, `_step4_every_node`,
  `_d1k`) + 선택기용 `--dsa` d3k 세트. 덤프 벽시계 차가우면 1.5–3분[유도] — 30분 안이라 승인 없이 리드 러너로. GPU 덤프면
  `NVIDIA_TF32_OVERRIDE=0`(rig-log 09-15: 없을 때 top-1 일치 0.896). Kimi-Linear는 받지 않는다(ik에 없고 감쇠 준비가 softplus).
- **G11 서빙 셋은 지금 가능(라운드 `glmserve`, aa, S+S, 박스는 `gate-tokenizer`·`gate-serve`만).** `glm4` pre = qwen2 분할기의
  숫자 가지를 `\p{N}{1,3}`로, BOS 없음 → `split_qwen2`에 매개변수 하나 + `Pre::NAMES` 행; 템플릿은 Jinja `macro`·`break`·
  `capitalize`가 필요(우리 부분집합에 셋 다 없음); 도구 파서는 `<tool_call>{name}<arg_key>…` 형식으로 `dsml.rs` 옆 새 파서;
  정지 토큰 셋(154820/154827/154829 — rig-log 09-15의 exl3 결함이 이 셋 중 하나만 넘긴 것); MTP 45층은 ik처럼 싣고 돌리지
  않는다(호스트 집합에서 expert 4.38 GB를 뺀다). 함대 라운드로 바로 발사한다(경계가 비행 중 어느 라운드와도 겹치지 않는다).
- **G12 사슬.** opslib(계열 이동) → {`hcq8` ‖ `route288` ‖ `glmmla` ‖ `ffnq8act`} → `glmprog`(aa, L: 새 `arch/glm5next` 프로그램,
  배치 288, e2e ik 토큰 대조; layerprog R1·session·hostone·q5kexp 뒤) → `glmsel` → `glmmtp`(시팅). `glmkda`(03 레인)·`glmref`·
  `glmserve`는 opslib와 병렬로 지금 시작할 수 있다. 트리아지 「glm 사슬」 행.
- **§5 범위 밖 지점의 처분.** `indexer.rs`의 `Hparams` 의존과 `router.rs`의 384 리터럴은 opslib 범위(알려진 항목); `hc_post`
  fold는 G7; ik `--dsa` 재풀링은 G8; ±1e6 클램프는 G4(03에 전달); 스펙 ε는 G1.

## 메모 원문 (라운드 `glmops`, 2026-09-27)

# glmops 설계 메모: GLM-5.3-Flash에 필요한 새 연산, 세 엔진 대조, 라운드 분해

**변경한 파일은 없습니다.** git 상태도 바꾸지 않았습니다. 박스에서는 `ssh ws`로 `sed`·`grep`·`ls`·`git log` 같은 읽기 명령만 썼습니다. 빌드, 바이너리, 임대는 쓰지 않았습니다. 숫자는 표시가 없으면 [유도]이고, 실측은 출처를 붙였습니다.

참조 트리는 다음과 같습니다.
- **ik:** Mac 클론 `~/repo/ik_llama.cpp` `41b17995`의 줄 번호로 인용합니다. GLM 파일은 `42a9a2fd`(#2376)만 만졌습니다. 박스 `db517b69`는 `c10fbbcc` 위에 V4.1 커밋 둘뿐이고, 그 차이는 `iqk_flash_attn.cpp`와 `build_deepseek4.cpp` 두 파일입니다(`git diff --stat`). 그래서 인용 줄은 오라클 트리와 같습니다.
- **exllamav3:** 박스 `0740edc`입니다.
- **mistral.rs:** `b0f26d5cd`입니다. GLM 계열은 `glm4_moe_lite`까지만 있고, KDA·mHC·glm5next는 없습니다. grep 결과 0건입니다.
- **GLM 파일 헤더:** glm5next 라운드가 스크래치에 남긴 UD-Q4_K_XL 샤드 1 덤프를 읽었습니다(`scratchpad/glm5next/glm-shard1.txt`, `glm-tensors.txt`).

## §0 한 줄 권고

1. **정정: hc epsilon은 1e-7이 아니라 1e-6입니다.** 샤드 1은 `hyper_connection.epsilon t6 9.999999974752427e-07`입니다. glmops.md와 glm5next.md의 1e-7은 오독입니다. glm5next 라운드가 1e-7을 핀했다면 틀린 핀입니다.
2. **ik의 k-pool 인덱서는 기본 꺼짐입니다.** `llama-cparams.h:46`에 `dsa = false`가 있고, `--dsa` 옵트인이며 EXPERIMENTAL입니다.
   - 켜도 한 질의가 보는 온전한 풀이 512개 이하이면 모든 풀을 고릅니다(`build_glm5next.cpp:153-155`, 풀 편향 `llama.cpp:5625-5632`). 위치 약 2,051 이하, 즉 2,048과 꼬리 3입니다.
   - 그래서 e2e까지의 최단 경로는 점수, top-k, 풀링 없이 **밀집 MLA와 인덱서 키 캐시 쓰기만** 하는 것입니다. 한계를 넘는 위치는 V4.1의 16,384 거부처럼 이름으로 거부합니다. 선택기는 e2e 뒤 라운드입니다.
   - 리드와 사용자가 정할 것은 둘입니다. 하나는 한계 너머에서 학습 의미(희소)를 따를지 ik 기본(밀집)을 따를지입니다. 다른 하나는 그 구간 오라클 세트를 `--dsa`로 뜰지입니다. 권고는 희소와 `--dsa`입니다.
3. **카드 트렁크는 토큰당 8.97 GB입니다.** 실제 헤더 기준으로 비-expert 0–44층이 8.292 GB이고 Q8_0 output이 0.674 GB입니다. q5kexpdesign의 "6.48 GB 이상(대리 하한)"을 대체합니다.
   - routed가 아닌 텐서는 전부 Q8_0이거나 F32입니다. `token_embd`도 Q8_0입니다. Q2_K_XL 목록의 Q5_K는 이 파일에 해당하지 않습니다.
   - 그래서 카드 쪽 형식 항목은 Q8_0 gemv 하나로 모입니다.
4. **디코드 벽시계 예측은 (a) 41–43 ms/토큰(23–24 tok/s)입니다.** 카드 직렬분 11.6–13.4 ms에 Σ브리지 29.6 ms를 더한 값입니다.
   - 호스트가 지배합니다. 카드 쪽 최대 항은 KDA 투영(Q8_0 m=1 gemv, 토큰당 4.97 GB)이고, delta 커널이 아닙니다.
5. **KDA는 03의 `delta_body::<DECAY_KEY>` 위의 엔트리 셋입니다.** bool 하나를 뒤집어서는 되지 않습니다.
   - `kda_delta`, KDA 준비(키 채널별 감쇠), `norm_gate::<GATE_SIGMOID>`가 필요합니다.
   - conv 가중치 셋은 적재 때 이어 붙이면 conv 커널을 바꿀 필요가 없습니다.
   - 파일이 03의 것이므로 03 라운드입니다.
6. **MLA는 V4.1 `ds41_attn_seg`와 merge를 런타임 인자만으로 재사용합니다.**
   - 인자는 window 0, sinks −∞, scale 1/√256입니다.
   - 흡수 `wk_b`/`wv_b`는 `q8_0_gemv_heads`를 그대로 씁니다.
   - 새 것은 LayerNorm(bias)과 [key; gate] 캐시 추가뿐입니다.
7. **라우터 288/8은 `ds41_router` 모듈의 새 인스턴스 행입니다.** E=288(PER_LANE 9), Score=Sigmoid이고, bias·norm·scale은 이미 런타임입니다.
8. **mHC의 차이는 셋입니다.** `ds41_hc_pre`의 Q8_0 fn 인스턴스가 새로 필요합니다. 믹스 규칙 Own은 기존 `ds41_hc_fold` 한 번을 더 부르면 되고, 부층당 런치가 2에서 3이 됩니다. 붕괴는 `ds41_hc_mean`이 비트 동일입니다.
9. **재사용할 커널 넷이 모델 크레이트 `gpu-deepseek41`에 있습니다.** attention, hc, router, indexer top-k입니다. opslib의 계열 이동이 GLM 프로그램의 직렬 선행입니다.
10. **풀 키는 exllamav3 설계를 따르자고 제안합니다.** 완성된 풀을 한 번만 계산해 불변 평면에 둡니다. ik는 매 스텝, 매 층 모든 풀을 다시 만듭니다. 연산 순서만 같으면 비트 동일이고, 스텝당 풀링 비용이 O(n)에서 O(1)이 됩니다.
11. **오라클은 `db517b69` 빌드(`/home/user/ik-idxkey`)가 그대로 됩니다.** `libllama.so`에 `glm5next`가 9회 있고, `GLM5NEXT: hyper_connection.count is required` 문자열도 있습니다. V4.1과 qwen35moe가 핀하는 같은 빌드입니다.
   - Kimi-Linear는 지금 받지 않습니다. ik에 없는 아키텍처이고, 감쇠 준비가 softplus라 GLM의 경계 sigmoid 준비를 검증하지 못합니다.
12. **서빙 항목은 셋입니다.**
    - `glm4` pre는 qwen2 분할기의 숫자 가지를 `\p{N}{1,3}`로 바꾼 것이고, BOS가 없습니다.
    - 템플릿에는 Jinja `macro`·`break`·`capitalize`가 필요합니다.
    - 정지 토큰은 셋(154820/154827/154829)입니다.
    - MTP 층은 ik처럼 싣고 돌리지 않습니다. 그 expert 4.38 GB는 호스트 집합에서 뺍니다.

## §1 층 종류별 연산 순서 (ik `build_glm5next.cpp` 기준, exllamav3 대조)

층 종류는 `head_count_kv[l]`가 정합니다. **0이 KDA, 1이 MLA**입니다(ik `llama-hparams.cpp:2477-2479`, 헤더 배열 `[0,0,0,1,…,0,1]`, 45번이 1).
- KDA 헤드는 `head_count` 64입니다(`ssm_dt_rank = n_head` `:2469`, `ssm_n_group = n_head` `:2473-2475`). 64 × 128 = 8192가 `attn_q`의 폭입니다.
- `key_length`/`value_length` 512는 KDA가 아니라 MLA 흡수 행(kv_lora 512 + rope 0)입니다. "4 heads"가 아닙니다.
- exllamav3는 같은 구분을 `layer_types[idx]`로 합니다(`glm5_next.py:168-224`).

**공통 뼈대(0–44층 트렁크, 부층마다 한 번씩, `build_glm5next.cpp:446-527`).**

| 단계 | 연산 | 텐서(UD-Q4_K_XL) | ε |
|---|---|---|---|
| hc_pre | 흐름 [4096×4] 평탄화, RMS(16384), `mixes = fn·normed`(24), `ggml_hc_pre`(Sinkhorn 20), 자기 `pre`로 가중합 (`:18-53`) | `hc_{attn,ffn}_fn` Q8_0 [16384,24], `_scale` F32[3], `_base` F32[24] | RMS 1e-5(`:451`), Sinkhorn 1e-6(`:43`) |
| 믹서 | KDA 또는 MLA. 각자 `attn_norm`을 안에서 적용 | 아래 | 1e-5 |
| hc_post | `build_mhc_post(cur, post, residual, comb)` (`:469`) | — | — |
| hc_pre | FFN용, 위와 같음 (`:476-482`) | `hc_ffn_*` | 1e-5 / 1e-6 |
| ffn_norm | RMS (`:484`) | `ffn_norm` F32[4096] | 1e-5 |
| FFN | dense(0–2) 또는 MoE+공유(3–44) | 아래 | — |
| hc_post | (`:524`) | — | — |
| 머리 | 흐름 합 `((s0+s1)+s2)+s3`에 ×¼(`:531-543`), `output_norm` RMS, output | `output` Q8_0 [4096,154880] 674 MB | 1e-5 |

**KDA 믹서(34층: 0–2, 4–6, …, 44; `llama-kda.cpp:200-235`).**

| # | 연산 | 텐서 / 모양 |
|---|---|---|
| 1 | `attn_norm` RMS | F32[4096] |
| 2 | q, k, v = W·x (`:12-14`) | `attn_{q,k,v}` Q8_0 [4096,8192] ×3 |
| 3 | z = g_b(g_a(x)) (`:15-19`) | `ssm_g_a` Q8_0 [4096,128], `ssm_g_b` Q8_0 [128,8192] |
| 4 | β_raw = W_β·x (`:39`) | `ssm_beta` Q8_0 [4096,64] |
| 5 | f = f_b(f_a(x)); g = lb·σ(−ssm_a·(f + dt)) (`:43-62`). ssm_a는 −exp(A_log)라 σ(exp(A_log)(f+dt))이고 lb = −5 | `ssm_f_a` [4096,128], `ssm_f_b` [128,8192] Q8_0, `ssm_dt.bias` F32[8192], `ssm_a` F32[64] |
| 6 | conv: q\|k\|v 세 conv를 이어 붙인 depthwise causal 4탭 (`:70-78`), SiLU (`llama-delta-net.cpp:336-341`) | `ssm_conv1d_{q,k,v}` F32 [4,1,8192] |
| 7 | q, k L2 정규화, ε = rms eps 1e-5 (`delta-net.cpp:364-374`, eps_norm = `f_norm_rms_eps` `llama-kda.cpp:225`) | — |
| 8 | delta 재귀. 상태 [64][128 v][128 k]이고 키 채널별 감쇠 exp(g), β = σ(β_raw), q ×1/√128 | 상태 f32 4,194,304 B/층 |
| 9 | 출력: 머리별 RMS(128), ×σ(z) (`:80-89`) | `ssm_norm` F32[128] |
| 10 | o = W_o·y | `attn_output` Q8_0 [8192,4096] |

**MLA 믹서(11층: 3, 7, …, 43; `build_glm5next.cpp:232-379`).**

| # | 연산 | 텐서 / 모양 |
|---|---|---|
| 1 | `attn_norm` RMS → cur (`:256`) | F32[4096] |
| 2 | qr = RMS(W_qa·cur) (`:260-263`) | `attn_q_a` Q8_0 [4096,1536] + norm F32[1536] |
| 3 | 인덱서 키: k = LayerNorm(W_k·cur)(bias, ε 1e-6), gate = W_g·cur, [k; gate] 256 값을 kr 캐시에 쓰기 (`:88-107`) | `indexer.attn_k` Q8_0 [4096,128], `k_norm.{weight,bias}` F32[128], `indexer_compressor_gate` Q8_0 [4096,128] |
| 4 | (한계 너머, `--dsa`만) 풀링, 점수, top-k, 목록 (`:113-217`) | `indexer.attn_q_b` Q8_0 [1536,4096], `indexer.proj` F32 [4096,32], `indexer_compressor_ape` F32 [128,4] |
| 5 | q = W_qb·qr [64×256] (`:272`), Qcur = wk_b·q 머리별 [256→512] (`:276-278`) | `attn_q_b` Q8_0 [1536,16384], `attn_k_b` Q8_0 [256,512,64] |
| 6 | kv = RMS(W_kva·cur) [512], k_l 캐시에 f16 행 (`:282-293`) | `attn_kv_a_mqa` Q8_0 [4096,512] + norm F32[512] |
| 7 | 흡수 MQA: K=V=잠재 512, scale 1/√256 (`:387`), rope 없음 | — |
| 8 | wv_b 머리별 [512→256] (`:360-366`), `attn_output` [16384→4096] | `attn_v_b` Q8_0 [512,256,64], `attn_output` Q8_0 [16384,4096] |

**FFN.**
- **dense(0–2):** `ffn_{gate,up}` Q8_0 [4096,12288]과 `ffn_down` Q8_0 [12288,4096]입니다. 클램프 SwiGLU 한계 10에 **공유 한계**(`swiglu_clamp_shexp`)를 씁니다(`llama-build-context.cpp:1269`).
- **MoE(3–44)의 라우터**(`llama-build-context.cpp:1482-1580`) 순서는 다음과 같습니다.
  - `ffn_gate_inp` F32 [4096,288]로 로짓을 만들고 σ를 씌웁니다.
  - `exp_probs_b`를 더하는 것은 **선택에만** 씁니다.
  - top-8을 고르고, 가중치는 편향 없는 σ에서 가져옵니다.
  - 합으로 나누는데, GLM은 epsilon이 없습니다. 1e-20은 BAILINGMOE2/3·STEP35에만 붙습니다(`:1564-1570`).
  - 마지막에 ×2.5를 합니다.
- **routed expert:** Q4_K gate/up [4096,2048,288]과 Q5_K down(11층은 Q5_K gate/up과 Q6_K down, 12·44층은 Q6_K down)입니다. 한계는 routed용 `swiglu_limit(il, false)`입니다(`:1613`).
- **공유 expert:** `ffn_*_shexp` Q8_0 [4096↔2048]이고, 한계는 shared용입니다. `moe + shexp`는 더하기 한 번입니다(`build_glm5next.cpp:511-519`).
- **클램프 규칙:** `min(silu(g), L)·clamp(u, ±L)`입니다. ik `unary.cu:80-82`와 exllamav3 `activation_kernels.cuh:170-183`(SiLU 뒤 클램프)가 같습니다. transformers만 SiLU 앞에서 자릅니다. 엔진은 ik 규칙을 따르고, 이미 `experts.rs:94-102` `swiglu_clamp`가 있습니다.

**MTP(45층).** MLA + 인덱서 + MoE + `nextn.{eh_proj Q8_0 [8192,4096], enorm, hnorm, shared_head_norm}`로 이뤄지고, hc는 없습니다.
- ik는 싣기만 하고 돌리지 않습니다. `:14`에 "loaded but not wired for generation"이 있고, 트렁크는 `n_layer − nextn`입니다(`llama-hparams.cpp:2486`, `build_glm5next.cpp:444-446`).
- MTP 그래프는 PR #2399 트리 `/home/user/ik-glm53-mtp` `425a2c1d`에 있습니다. exllamav3는 `glm5_next_mtp.py`(238줄)입니다.

**ε 정리.**
- `layer_norm_epsilon` 1e-6은 **인덱서 `k_norm` LayerNorm 하나에만** 쓰입니다(`llama-hparams.cpp:2379-2387`, `build_glm5next.cpp:89`).
- `hyper_connection.epsilon` 1e-6은 Sinkhorn과 pre의 하한입니다.
- 나머지 RMS·L2는 모두 1e-5입니다.

**ik와 exllamav3가 갈리는 곳.**
- exllamav3의 k-pool은 불변 풀 평면(`mla_attn.py:548`)입니다.
- exllamav3의 KDA 프리필은 fla `chunk_kda`입니다(`gated_delta_rule.py:110-137`). ik는 모든 길이에 순차 재귀를 씁니다(`kda.cu:207-229`).
- 수치 계약은 ik를 따릅니다(오라클). 풀 평면은 비트를 바꾸지 않는 흐름 변경이라 exllamav3를 따르자고 제안합니다.

## §2 새 연산별: 재사용, 결손, 종류, 비용, 증명

디코드 바이트를 604–700 GB/s로 나눠 µs를 냈습니다. 브리지 적합식은 `host-tier-design.md` §4.3입니다.

### 2.1 KDA (34층)

**있는 것(03, `~/repo/bloomery-q35gdn`, `61979df`).**
- `crates/gpu/src/linear/delta.rs`의 `delta_body<const DECAY>`는 DECAY_KEY 경로가 이미 몸체에 있습니다(`:225-253`). 엔트리는 `gdn_delta`(DECAY_HEAD)뿐입니다(`:308-330`).
- `linear/mod.rs:94-103`에 `DECAY_KEY`와 `GATE_SIGMOID` 상수가 있지만 "no entry is built"입니다.
- `norm_gate.rs:41-47`에 `act::<GATE_SIGMOID>`가 있습니다.
- conv는 채널 무관 depthwise입니다(`conv.rs:1-23`). 상태 배치는 [n_v][v][k]이고, 감쇠는 k 축입니다. ik의 감쇠 축(`iqk_kda.cpp:80`, `col` = k)과 같습니다.

**없는 것(세 엔트리, 모두 03 파일).**
1. **`kda_delta`:** `delta_body::<DECAY_KEY>`에 계약 `decay.len() ≥ m·n_v·128`을 붙입니다. 새 엔트리이고 몸체는 공유합니다.
2. **KDA 준비 엔트리:**
   - conv, SiLU, L2(ε 1e-5), β=σ(b)는 GDN과 같습니다.
   - 감쇠만 머리별이 아니라 키 채널별입니다. `exp(lb·σ(−ssm_a_h·(f_{h,k} + dt_{h,k})))`이고 입력은 `f_b` 출력 [m][8192]입니다.
   - GDN의 `a_raw[m][n_v]`와 인자 모양이 달라 새 엔트리입니다. conv와 L2 부분은 공유합니다.
3. **`norm_gate` 엔트리:** `GATE_SIGMOID`를 인스턴스화하는 일만 남았습니다.

적재 때 할 일이 둘 있습니다. `ssm_conv1d_{q,k,v}` [4,1,8192] 셋을 채널 순서 q|k|v로 이어 붙입니다(ik `build_kda_conv` `llama-kda.cpp:70-78`와 같음). `W_q|W_k|W_v`도 한 gemv 행으로 쌓습니다.

**참조 설계 대조.**

| 엔진 | GDN/KDA 분기 |
|---|---|
| mainline | `gated_delta_net.cu:4` `template <int S_v, bool KDA, bool keep_rs_t>`, KDA 가지 `:110-132` |
| exllamav3 | `gdn.cu:660` `template <bool save_history, int V_SPLIT, bool CHANNELWISE>`, 갱신 `:827` |
| ik | 별도 파일 `kda.cu`(delta-net.cu의 사본). 감쇠 줄만 다름. 게이트 모양으로 분기(`ggml-cuda.cu:4253`) |
| 03 | mainline과 같은 모양(const 하나에 엔트리 여럿). 그대로 따름 |

**ik 오라클과의 수치 차이(명명).** 박스 CPU는 AVX2이므로 오라클 경로는 `iqk_kda.cpp`의 AVX2 가지입니다. 03 몸체와 세 곳이 다릅니다.
- **내적 순서:** ik는 `Σ_col S·(k·d)`를 순차로 더합니다(`:273-277`). 03은 S'=d·S를 반올림한 뒤 레인 분할 내적입니다.
- **exp 구현:** ik는 β와 감쇠에 libm `expf`를 씁니다(`:225`, `:227`). 03은 `expf_ik`입니다.
- **상태 클램프:** ik는 매 스텝 상태를 ±1e6으로 자릅니다(`:288-296`, `kda.cu:153`, `delta-net.cu:158`). 03은 자르지 않고 비유한 값에서 fault만 올립니다.
- 그래서 계약은 "자기 호스트 규칙과 비트 동일 + ik와 밴드"입니다. 클램프 차이는 |S| > 1e6일 때만 드러나는 명명된 행동 차이입니다.

**디코드 비용(층당, m=1).**

| 항 | 바이트 | 런치 |
|---|---|---|
| W_q\|k\|v gemv (4096→24576) | 106.95 MB | 1 |
| f_a\|g_a\|β gemv (4096→320) | 1.39 MB | 1 |
| f_b, g_b gemv (128→8192 ×2) | 2.23 MB | 1–2 |
| conv+prep | 1.28 MB | 1 |
| delta (상태 r+w) | 8.39 MB | 1 |
| norm_gate | ≈0.1 MB | 1 |
| W_o gemv (8192→4096) | 35.65 MB | 1 |
| **합** | **155.9 MB → 223–258 µs** | 7–8 (+attn_norm) |

토큰당 34층 합은 5.30 GB이고, 그중 투영이 4.97 GB, 상태가 285 MB입니다. 결론은 둘입니다.
- KDA의 카드 레버는 delta 커널이 아니라 **Q8_0 m=1 gemv의 품질**입니다. `q8_0_gemv`(`q8f32.rs:1157`)의 GB/s는 실측이 없습니다.
- 시퀀스당 상태는 142.6 MB, conv 상태는 10.0 MB입니다.

**프롬프트 비용.** 03의 GDN 유도(512 ubatch당 4.3–6.0 ms, 30층 × 32머리)를 34/30 × 64/32로 늘리면 **9.7–13.6 ms/512**입니다. 호스트 유니온(층-배치당 약 120 ms급, §2.5 끝) 그늘 아래라서, chunked KDA(fla `chunk_kda`) 라운드는 지금 가치가 0입니다.

**게이트.** 03의 `gate_linear`에 KDA 절을 붙입니다.
- n_k = n_v = 64이고, m = 1/512/깊이 4096입니다.
- 절마다 FAIL-first를 둡니다. 증명 급은 "add"로, 기존 행과 md5는 동일하고 새 행만 늘어납니다.
- 오라클 밴드는 ik 노드 덤프 `attn_output`, `new_state`, `final_output`입니다.

### 2.2 MLA-256/512 (NoPE 흡수, 11층)

**재사용(런타임 인자만).**
- **attention:**
  - `ds41_attn_seg`(접두) + `ds41_attn_merge`(`gpu-deepseek41/src/attn.rs:385`, `:684`)를 씁니다.
  - K=V 잠재 512 f16 행(`win: &[u16]`, `comp: &[u16]`)은 ik `k_l`의 기본 f16과 같은 폭·타입입니다.
  - 컴파일 상수는 LATENT 512 = 블록 폭(`:68-70`)과 MMA_ROWS 16(헤드 64 = 4묶음)입니다.
  - 창 0행이면 `source_segments(0)=0`이고, scale은 런타임(`:390`)이므로 1/√256을 넘깁니다.
  - sinks −∞는 `online_fold`(`gpu/src/flash.rs:329-343`)의 `mj ≤ mx` 가지에서 `s + 0·…`, `acc + 0·0`이 되어 정확한 no-op입니다[코드에서 유도]. 첫 게이트에서 한 번 확인합니다.
- **흡수 gemv:** `q8_0_gemv_heads`(`gpu/src/model/kernels.rs:97`)는 k, rows_per_head, n_heads, 보폭이 모두 런타임입니다. wk_b(k 256 → 512행 ×64)와 wv_b(k 512 → 256행 ×64) 둘 다 그대로 됩니다.
- **나머지 gemv:** `q8_0_gemv`입니다. `attn_q_a|attn_kv_a_mqa|indexer.attn_k|compressor_gate`는 4096 → 1536+512+128+128 = 2304행 한 번으로 쌓습니다.

**새 것.**
- **(a) 잠재 RMS + f16 추가(rope 없음):** V2-Lite `kv_norm_rope_append`(`fused.rs`)는 rope가 박혀 있습니다. rope 0 인스턴스이거나 새 작은 커널입니다(S).
- **(b) 인덱서 키 행 추가:** LayerNorm(128, bias, ε 1e-6)을 k에 적용하고 gate를 붙여 256 값을 캐시에 씁니다. LayerNorm은 새 연산입니다(§2.7).

**KV 캐시(위치당).**

| 방식 | 층당 | 11층 합 |
|---|---|---|
| ik | 잠재 512 f16 1,024 B + [k; gate] 256 f16 512 B = 1,536 B | 16.9 KB |
| 풀 평면(§2.3, f32 풀 키) | 1,024 + 128 = 1,152 B | 12.7 KB (−25 %) |

**디코드 비용(층당).**

| 항 | 바이트 |
|---|---|
| 합(선택기 없음) | 126.4 MB → 181–209 µs |
| 그중 attn_output | 71.3 MB |
| 그중 q_b | 26.7 MB |
| 그중 wk_b + wv_b | 17.8 MB |
| attention 키(n=2,051) | 2.1 MB → 3.0–3.5 µs |
| attention 키(밀집 n=32,768) | 33.6 MB → 48–56 µs |

- 32k에서 밀집과 희소의 디코드 차이는 11층 × 약 50 µs ≈ 0.5 ms/토큰입니다. 41 ms 스텝의 약 1 %입니다.
- **선택기의 디코드 가치는 속도가 아니라 학습 의미**입니다.
- 프롬프트에서는 다릅니다. P = 32k 밀집은 11층 × 7.0e13 FLOP이고, 선택하면 8.8e12 FLOP이므로 긴 프롬프트에서는 속도 레버입니다.

런치는 약 10개입니다(쌓은 gemv, proj F32, 두 norm, q_b, wk_b, 두 append, seg, merge, wv_b, o).

**게이트.**
- 호스트 규칙 대비 비트: LN, append.
- ik 밴드: `kqv_out`, `kv_compressed`, `dsa_indexer_k`.
- 구조: 헤드 64에서 창 0, sinks −∞가 sinks 없는 호스트 규칙과 비트 동일해야 합니다.

### 2.3 k-pool 인덱서 (11층, e2e 뒤)

**재사용.**
- `ds41_indexer_topk`(`gpu-deepseek41/src/indexer.rs:597`)는 히스토그램 정확 top-k이고, 개수가 디바이스 워드입니다.
- `n_vis ≤ k`이면 항등 목록입니다(`:32-37`). GLM 한계 아래 동작과 정확히 같습니다.
- `ds41_attn_seg_sel`(`attn.rs:453`)은 목록을 통해 행을 읽습니다.

**새 것.**
- **(a) 풀 평면 갱신:** 완성된 풀마다 한 번, 멤버 4개의 `softmax_r(gate + ape)·k`를 채널별로 계산합니다(ik 수식 `build_glm5next.cpp:127-136`, exllamav3 설계 `mla_attn.py:548`). 풀 키는 멤버의 순수 함수이므로 ik와 같은 연산 순서면 비트 동일입니다.
- **(b) f32 점수:** `Σ_h w_h·relu(q_h·pk)`, w = proj(cur)/√(32·128)입니다(`:145-146`, `:168-173`).
  - V4.1 `ds41_indexer_score`는 rope와 Hadamard가 몸체에 박혀 있고, f16 hi/lo 텐서코어 경로입니다(`indexer.rs:9-26`).
  - GLM은 풀이 n/4개뿐이라(32k에서 8k 풀, 토큰당 층당 67 MFLOP) SIMT f32 작은 커널이 ik 값과 맞추기 쉽습니다. 인스턴스가 아니라 새 커널입니다.
- **(c) 목록 전개:** 선택 풀을 셀 4개로 펴고, 꼬리 최대 3셀을 붙입니다(`:203-214`, 꼬리 정의 `llama.cpp:5635-5660`).

**비용(한계 너머, 층당).**
- 인덱서 q_b 6.7 MB에 풀 키 n/4 × 512 B(f32)를 더합니다. 32k면 4.2 MB로 약 15–18 µs입니다.
- top-k와 attention_sel 2,051 키 2.1 MB가 붙습니다.

**게이트.** ik `--dsa`로 뜬 d3k 이상 세트와 목록을 비교합니다. top-k 동점 규칙은 §4에 적었습니다.

### 2.4 mHC (부층 90개: 45층 × 2)

**재사용.**
- `hc_pre_lane`(Sinkhorn)의 eps·iters·rms_eps가 모두 런타임입니다(`hc.rs:126-180`, `:454-456`).
- `ds41_hc_post`(`:1034`)는 흐름을 쓰고, **같은** `pre`로 fold도 씁니다. 이것은 V4.1의 Lagged 규칙입니다.
- `ds41_hc_fold`(`:1068`)가 있습니다.
- `ds41_hc_mean`(`:1108`)은 `((s0+s1)+s2)+s3` ×0.25로, ik 붕괴(`build_glm5next.cpp:536-542`)와 비트 동일입니다(modelspec K5에서 aa가 대조했습니다).

**GLM은 Own 규칙입니다.** 다음 부층 입력은 그 부층 자신의 pre로 접습니다(`:52`). 그래서 순서는 `hc_post`(fold 출력은 버림) → `hc_pre` → `hc_fold`이고, 부층당 런치가 3입니다. V4.1은 2이므로 토큰당 90개가 늘어납니다.

**새 것.** `hc_pre`의 Q8_0 fn 인스턴스(modelspec K4, `hc_pre<STREAMS, Fmt=Q8_0>`)입니다.
- V4.1은 Q3_K 워크와 융합돼 있습니다(`HC_PIECE 512` `:73`).
- `hc_f32.rs`는 F32 fn으로 토큰당 블록 하나입니다. 디코드에서 418 KB를 블록 하나가 읽어 지연에 묶입니다.
- 그래서 V4.1의 split-K 골격에 Q8_0 조각 내적을 넣는 인스턴스로 만듭니다. 컴파일 인스턴스 행입니다.

**비용.** 층당 fn 0.84 MB와 흐름 약 0.4 MB로 1.8–2.0 µs입니다. 바이트는 무시할 만하고, 항은 런치 수(층당 6)입니다.

**게이트.** 호스트 규칙 대비 비트, ik `hc_pre_mixes`와 `hc_attn_pre` 밴드입니다.

### 2.5 라우터 288/8

**있는 것.** `ds41_router`(`gpu-deepseek41/src/router.rs`)입니다.
- 선택 편향(`:21-35`), 합 f64 뒤 f32 나눗셈, `renorm_divisor`, 런타임 scale(`:143`)이 있습니다.
- 동점은 큰 id로 갑니다. ik 쌍 정렬 `std::greater`와 같습니다(`:27-29`).
- Score 트레이트에 `Sigmoid`가 있습니다(`gpu/src/route_core.rs:57-80`).

**모델 상수(모듈 수준).**
- `N_EXPERT 384`, `N_USED 6`이 있습니다(`:57-60`).
- 계약 리터럴 `n_expert == 384`가 세 곳에 있습니다(`:126`, `:260`, `:365`).

**GLM 인스턴스는 E=288, K=8, Score=Sigmoid입니다.**
- PER_LANE 9 ≤ 32이고, SCORE_TILES 36입니다. 288 = 9·32 = 36·8이라 모든 const 단언이 성립합니다.
- 03 kernelshape K1은 top-k를 런타임으로 보자고 했습니다. 이 모듈은 N_USED가 공유 배열과 계약에 박혀 있어 인스턴스 키에 넣습니다.

**편차 하나(명명).** ik GLM은 맨 합으로 나눕니다. 우리 가드 +1e-20은 합 ≥ 2^-42면 비트를 바꾸지 않습니다(`route_core.rs:40-50`). σ 8개의 합이 그 아래가 되려면 선택된 로짓 8개가 모두 −29 아래여야 합니다.

mistral.rs `glm4_moe_lite.rs:595-600`은 1e-20을 더합니다. exllamav3는 `routing.cu`(sigmoid·편향 `:366`)입니다.

**비용.** F32 [4096,288] 4.72 MB로 6.7–7.8 µs이고, 런치 1입니다.

**게이트.** 인스턴스 셋 (288, 8, Sigmoid, 2.5)에 대한 호스트 규칙 비트와, ik `ffn_moe_topk` id 정확 일치입니다.

### 2.6 클램프 SwiGLU (dense 0–2, 공유 3–44, routed)

**있는 것.**
- 규칙 `swiglu_clamp`(`experts.rs:94-102`).
- 호스트 쪽 `GroupInput::SwigluClamp`(`model/src/moe.rs:905-930`).
- Q4_K/Q5_K 호스트 타일(`qdot/src/lib.rs:435-445`, `:522-524`). Q6_K 타일은 없어서 11·12·44층 프롬프트만 느립니다(host-tier-design §3.1).

**없는 것.** Q8_0 gate|up + act 카드 커널입니다.
- `ds41_shexp_gate_up*`는 Q3_K이고(`dense.rs:621`), `qwen3moe_gate_up_swiglu_q4k`는 한계가 없습니다.
- q5kexp의 `kq_gate_up_act`(act는 런타임 인자) 계열에 Q8_0 워크 행을 추가하는 것이 1안입니다.
- e2e용 최소안은 `q8_0_gemv` ×2(gate|up 쌓기) + 원소별 act 런치 + down입니다. 바이트는 같고 런치가 +1입니다.

**비용.**

| 항 | 바이트 | 시간 |
|---|---|---|
| dense 층 | 160.4 MB | 229–266 µs |
| 공유 | 26.7 MB | 38–44 µs |

공유 expert는 브리지 그늘 안에 넣을 수 있는 몇 안 되는 항입니다. 토큰당 1.12 GB, 1.6–1.9 ms입니다.

**호스트.** 288과 2의 거듭제곱 가정을 grep했습니다(`moe.rs`, `placement*.rs`, `hybrid.rs`, `deepseek41/host.rs`). 테스트 상수(`moe.rs:1716` 384, `:1869-1968`) 외에는 없습니다.
- `SlotMap::prefix`가 n_expert를 런타임으로 받습니다(`hybrid.rs:334`).
- `EXPERTS_INTO_MAX = 8`(`moe.rs:753`)에 딱 맞습니다.
- 유니온 열 수는 디코드에서 expert 8개 × 1열입니다. 프롬프트 512 토큰에서는 4,096 슬롯을 288 expert가 나눠 평균 14.2열입니다.
- 토큰·층당 호스트 바이트는 8 × 15.2 MB = 121.6 MB이고, 브리지는 약 903 µs입니다.

### 2.7 LayerNorm (인덱서 k, 11층)

없는 연산입니다. 128 값에 대해 평균, 분산, ε 1e-6, ×w + b를 합니다(ggml `LLM_NORM`, `build_glm5next.cpp:89`). §2.2(b) 추가 커널 안에 한 워프로 융합하는 새 작은 커널이고, 비용은 무시할 만합니다. 게이트는 호스트 규칙 비트와 ik `dsa_indexer_k` 밴드입니다.

### 2.8 디코드 자원 시간선 (토큰당, 배치 (a), A6000 604–700 GB/s)

| 자원 | 항 | 값 | 벽시계 기여 |
|---|---|---|---|
| 카드 | KDA 34 + MLA 11 + hc 90 + dense 3 + 라우터 42 + output | 8.10 GB → 11.6–13.4 ms | 직렬(라우트 전·조인 뒤) |
| 카드 | 공유 expert 42층 | 1.12 GB → 1.6–1.9 ms | 브리지 그늘(0) |
| 카드 | routed 카드 expert(q5kexp) | 층당 46 µs(q5kexpdesign §4) | 그늘(0) |
| 호스트 DRAM | Σ브리지 | 29.6 ms (k_host 6.17), 전부 호스트면 37.9 ms | 직렬 |
| PCIe | 이미지와 hsum | 층당 1–2 µs | 무시 |
| NVMe | 호스트 집합 고정 | 0 | 0 |
| **벽시계** | 카드 직렬 + Σ브리지 | **(a) 41.2–43.0 ms(23.3–24.3 tok/s), 전부 호스트 49.5–51.3 ms** | 합 |

흐름 레버의 판정은 다음과 같습니다.
- 브리지와 카드 믹서는 데이터 의존이라 겹칠 수 없습니다.
- 카드 쪽 커널 개선은 합의 약 30 %만 움직입니다.
- 가장 큰 레버는 **검증 m열**(MTP 1층 또는 DSpark 초안)입니다. 호스트 바이트를 m 토큰에 나눠 읽기 때문이고, e2e 뒤 항목입니다.

예측과 따로 두는 참고 실측이 있습니다. rig-log 2026-09-15(ik `7434a014`, 09-22 이전 카드 구성이라 같은 표에 넣지 않음):
- ik 디코드 17.0 tok/s(expert 전부 CPU).
- 카드 상주 가중치 읽기 15.1 ms, 토큰당 8.5 GB(nsys).
- MTP +12 %.
- exl3 4.05 bpw 22.4 tok/s(2026-09-16).

### 2.9 서빙: MTP, `glm4`, 템플릿·도구

- **MTP.** ik는 싣고 돌리지 않습니다(§1). 우리도 같습니다.
  - 45층의 비-expert 200.3 MB와 expert 4.38 GB는 적재하지도 잠그지도 않습니다. 호스트 예산 214 GB에서 4.38 GB를 돌려받습니다.
  - 초안 사용은 `Mtp{1}` 스케줄 라운드에서 합니다.
- **`glm4` pre.**
  - ik는 `glm4`를 CHATGLM4로 매핑하고 `special_bos_id = NULL`입니다(`llama-vocab.cpp:2074-2077`).
  - 정규식(`:408-411`)은 사용자 정의 llama3 분할기로 갑니다(`unicode.cpp:1076-1079`).
  - llama3 분할기(`:333`)와 qwen2 분할기(`:624`)는 숫자 가지 하나만 다릅니다. `\p{N}{1,3}`(`:406-414`) 대 `\p{N}`(`:697-698`)이고, 나머지는 diff 0입니다.
  - 따라서 `crates/tokenizer/src/pretok.rs`의 `split_qwen2`(`:259`) 숫자 가지에 매개변수 하나를 두고, `Pre::NAMES`(`:209-214`)에 `("glm4", …)`를 추가합니다(S).
- **템플릿.** GLM 템플릿(10,648자)은 `macro` 12회, `break` 3회, `| capitalize`를 씁니다. `crates/serve/src/template.rs:1-14`의 부분집합에는 셋 다 없습니다(S–M).
  - 생성 프롬프트는 `<|assistant|><think>`로 끝납니다. `reasoning.rs`의 "span 안에서 시작" 규칙과 `<think>`/`</think>` 문자열이 그대로 맞습니다.
- **도구 파서.** 형식은 `<tool_call>{name}<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`이고, 문자열이 아닌 v는 JSON입니다(템플릿 50행, 163행). `dsml.rs` 옆에 새 파서를 둡니다(S).
- **정지 토큰.** eos 154820, eot 154827, eom 154829 셋입니다. rig-log 09-15가 exl3 서빙에서 "종료 토큰 셋 중 하나만 넘긴" 결함을 적었습니다.

### 2.10 오라클

- **트리.** `/home/user/ik-idxkey` `db517b69`를 씁니다. 빌드된 `build/src/libllama.so`(09-23 17:10)에 glm5next가 들어 있고, `llama-eval-callback`이 있습니다. refset 행은 V4.1·qwen35moe와 같은 `IK_BUILD`를 핀할 수 있습니다.
- **refset 가족 행.** `crates/refset/src/arch/qwen35moe/mod.rs`(q35 트리)의 모양을 그대로 씁니다.
  - MODEL은 `/models/GLM-5.3-Flash-UD-Q4_K_XL/…-00001-of-00006.gguf`(첫 샤드 전체 경로가 정체성), ARCH는 `glm5next`입니다.
  - 세트는 셋입니다. `ref_glm5next`(32토큰 배치, CPU), `…_step4_every_node`, `…_d1k`입니다.
  - 여기에 `tools/ref/models/glm5next.sh`와 `dump-ref-glm5next` 레시피를 더합니다.
  - 선택기 라운드에는 `--dsa` d3k 세트를 따로 둡니다.
- **덤프 벽시계 [유도].** 32토큰 CPU 한 세트 기준입니다.
  - 적재: 199.7 GB를 차갑게 읽으면 약 80 s입니다. exl3 GLM 적재 66 s(rig-log 09-16)에서 얻은 유효 2.5 GB/s로 계산했습니다. 박스 RAM이 263.7 GB라 파일이 페이지 캐시에 들어가지만, 지금 캐시는 다른 모델로 차 있습니다.
  - expert 읽기: 32토큰이 층당 평균 171/288 expert를 건드리므로 109 GB이고, 135 GB/s로 0.8 s입니다.
  - 연산: 1.15 TFLOP로 1–2 s입니다.
  - 노드 쓰기: 수 GB로 수 초입니다.
  - 합: **차가우면 1.5–3분, 따뜻하면 1분 미만**입니다. d1k는 +25–40 s입니다. 모두 30분 안이고 `DUMP_BOUND` 1800 s(`tools/ref/dump.sh:118`) 안입니다.
  - GPU 덤프를 한다면 `NVIDIA_TF32_OVERRIDE=0`이 필요합니다. rig-log 09-15에 따르면 없을 때 top-1 일치가 0.896이었습니다.
- **Kimi-Linear(#18755)은 지금 받지 않습니다.**
  - ik `llama-arch.cpp`에 kimi 계열이 없어서 mainline 전용 덤퍼가 따로 필요합니다(`dump.sh`의 foreign-lib 거부).
  - 감쇠 준비가 softplus라(mainline `kimi-linear.cpp:313-318`) GLM의 경계 sigmoid 준비를 검증하지 못합니다.
  - 공유 delta 핵심과 채널 감쇠를 두 번째 코드베이스로 판별하는 수단일 뿐입니다. ik 대비 KDA가 밴드 밖으로 나갈 때만 꺼냅니다.

## §3 라운드 (e2e까지 직렬 단계를 최소로, 파일 경계 분리)

전제로 비행 중이거나 계획된 것이 여섯입니다. opslib(3파동, 계열 이동), layerprog R1(트레이트·Overlap), session(순환 상태 슬롯), hostone StepPort(n_used·hidden 런타임), q5kexp(카드 routed Q5_K; 없으면 전부 호스트로 약 50 ms), glm5next(리더와 커버리지 목록)입니다.

박스 분은 gate-times 원장(`~/.cache/bloomery/gate-times.tsv`)의 실측 중앙값으로 유도했습니다. `gate-gpu-linear` 38 s, `ptx-scan generate_ds41` 약 138 s, `gate_e2e` 약 7 s, `gate-ptx-spill` 약 19 s, `gate-serve` 6 s, `gate-tokenizer` 84–114 s입니다. 차가운 oxide 빌드는 "a few minutes"(AGENTS)입니다.

| # | 라운드 | 크기 | 파일 경계 | 선행 | 증명 급 | 박스 분 [유도] |
|---|---|---|---|---|---|---|
| 1 | `glmkda`(03) | S–M | `crates/gpu/src/linear/{delta,conv,norm_gate}.rs`, `gate_linear` 절 | q35gdn 착륙 | add + 호스트 규칙 비트 + 절별 FAIL-first | 15–25 |
| 2 | `hcq8`(aa) | S | opslib의 hc 계열(없으면 `gpu-deepseek41/src/hc.rs` 임시) | opslib hc 이동 | add + 비트 | 10–20 |
| 3 | `route288`(aa) | S | opslib의 router 계열(K2 매크로 인스턴스) | opslib router 이동 | add + 비트 + id 정확 | 10–20 |
| 4 | `glmmla`(aa) | M | 새 `latent` 추가 커널, LayerNorm, 적재 때 행 쌓기 | opslib attn 이동, glm5next | add + 비트(LN·append) + sinks −∞ 동일성 | 15–30 |
| 5 | `ffnq8act`(aa 또는 q5kexp 계열 주인) | S | kquant/act 계열의 Q8_0 행 | q5kexp | add + 비트 | 10–20 |
| 6 | `glmref`(aa) | S | `tools/ref/models/glm5next.sh`, `crates/refset/src/arch/glm5next/`, justfile 레시피 1 | 다운로드 ok, glm5next | refset 가족 게이트 | 세트당 1.5–3 (+검사 1) |
| 7 | `glmserve`(aa) | S+S | `tokenizer/src/pretok.rs` / `serve/src/{template.rs, glmxml.rs(신설), stop}` | 없음(지금 가능) | FAIL-first 단위 게이트 | 3–5 |
| 8 | `glmprog`(aa) | L | 새 `arch/glm5next` 프로그램, 배치 288, Q8_0 임베드 행, e2e 게이트 | 1–6, layerprog R1, session, hostone, q5kexp | 새 모델: ik 토큰 대조 e2e | 30–60 (착륙 배치 별도) |
| 9 | `glmsel`(aa) | M | 풀 평면, f32 점수, 목록, `_sel` 배선 | 8 | 변경: `--dsa` 세트 밴드와 목록 | 20–40 |
| 10 | `glmmtp` | M–L | 스케줄러의 `Mtp{1}` | 8, session verify | 변경 + A/B 시팅 | 시팅 |

직렬 사슬은 **opslib → {2, 3, 4, 5} 병렬 → 8 → e2e**로 세 단계입니다. 1(03 레인), 6, 7은 opslib와 병렬로 지금 시작할 수 있습니다. 파일 경계는 모두 서로 겹치지 않습니다.

## §4 파일을 읽어야 알 것, glm5next가 정할 것

**이미 정해진 것.** 비routed 형식 전부(Q8_0/F32), routed 형식, 라우터 F32, 인덱서 proj F32, 헤더 키 72개가 샤드 1 덤프로 확정됐습니다.

**남은 것.**
1. **풀 top-k 동점 규칙.** ReLU 때문에 점수가 정확히 0인 풀이 흔할 수 있어서, 약 2,051 위치 너머에서는 경계 동점이 비트를 가릅니다. V4.1 `ds41_indexer_topk`는 낮은 행으로 보냅니다. ik CPU `ggml_top_k`는 `ggml_argsort` DESC(`ggml.c:10765-10781`)인데, iqk가 이것을 덮는지 확인해야 합니다.
   ```
   grep -n 'argsort' ~/repo/ik_llama.cpp/ggml/src/iqk/*.cpp
   ```
   e2e를 막지 않습니다.
2. **`q8_0_gemv`의 m=1 GB/s.** 실측이 없고, KDA 투영 4.97 GB/토큰의 속도가 여기에 달립니다. nsys 한 번으로 답이 나오며, 기존 러너로 가능한지는 리드가 판단합니다.
3. **ik 덤프 경로의 풀 키 정밀도.** `kr_l` 캐시 타입(`idx_type_k`)의 기본값과, 풀링이 f32인지 확인해야 합니다(`llama.cpp:1089`, `:1461`).
4. **glm5next 라운드가 정할 것.**
   - hc ε 핀(1e-6).
   - 커버리지 목록 항목. Q8_0 hc fn, Q8_0 `token_embd`/output, `DeltaRule{Kda}` ×34, `TokenPool` ×11, `(Sigmoid, 288, 8)`, routed Q5_K(카드), Q6_K 11·12·44, pre `glm4`, 도구 파서가 들어가야 합니다.
   - nextn 층을 "싣고 돌리지 않음"으로 둔 줄.
   - `layer_norm_epsilon` 필수 여부. ik는 없으면 RMS ε로 대체하고(`llama-hparams.cpp:2382-2387`), PR은 필수입니다.
5. **다운로드 완료.** `/root/glmdl.rc`가 ok가 되어야 합니다. 디렉터리는 11:07에 갱신 중이었습니다.

**하지 않은 것.** Kimi-Linear GGUF 헤더는 박스에 파일이 없어 읽지 않았습니다. 박스 실행은 없습니다.

## §5 범위 밖 개선 지점 (보고만)

- **`crates/gpu-deepseek41/src/indexer.rs:58`과 `:992`:** 커널 모듈이 `model::arch::deepseek41::hparams::Hparams`에 의존합니다. `ds41_indexer_topk`를 모델 없는 계열로 옮길 때 걸립니다(S).
- **`crates/gpu-deepseek41/src/router.rs:126`, `:260`, `:365`:** 계약 리터럴 `n_expert == 384`와 모듈 상수가 있습니다. 인스턴스 매크로 전에는 288을 받을 수 없습니다. opslib 범위이고 이미 알려진 항목입니다(S).
- **`crates/gpu-deepseek41/src/hc.rs:1034-1060`:** `ds41_hc_post`는 항상 fold를 씁니다. Own 규칙 모델에서는 버려지는 n·m f32 쓰기이고 런치는 2+1입니다. fold를 다음 hc_pre의 마무리 블록이나 소비자 RMS에 접는 레버는 GLM 시팅 뒤 판단합니다(S).
- **ik `build_glm5next.cpp:113-137`:** `--dsa`가 매 스텝, 매 층 모든 풀 키를 다시 만듭니다(O(n)). exllamav3 `mla_attn.py:548`의 불변 풀 평면이 선례입니다. 업스트림 후보이지만 EXPERIMENTAL 경로이고 메인테이너 부하 규칙에 걸리므로 트리아지감입니다(M).
- **ik 세 구현의 ±1e6 상태 클램프:** `kda.cu:153`, `delta-net.cu:158`, `iqk_kda.cpp:288-296`의 클램프는 03 커널에 없습니다. 조용한 값 보정 대 fault의 선택이므로 AGENTS의 조용한 실패 금지 조항으로 판단해 03 보고에 한 줄 둘 가치가 있습니다(S).
- **glmops.md와 glm5next.md 스펙의 hc ε 1e-7(§0.1):** 리드 문서 정정이 필요합니다(S).

## §6 모델

opus(Opus 5.5)로 돌았습니다.
