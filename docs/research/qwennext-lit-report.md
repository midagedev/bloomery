# qwennext-lit — 최신 Qwen 모델이 이 엔진에 요구하는 것, 종이 위에서 (라운드 보고, 2026-09-25)

리드 메모: 파동 25 대기 창의 박스 없는 조사(opus). 스펙 `specs/wave-m6/spec-qwennext-lit.md`. 결론은 Gated DeltaNet(GDN) 프로그램 하나를
Qwen3.6-35B-A3B로 세우고 같은 사슬 뒤에 Qwen3.8-27B(가장 많이 받는 최신 Qwen, +GROUP 6 flash·dense FFN·MTP)를 붙이는 것이다. 가장 큰 새 사실:
lmstudio-community의 두 Q4_K_M 파일은 Q4_K/Q6_K뿐이라 우리 Qwen3-30B-A3B와 같은 타입 집합이고(Q5_K·IQ 경로는 unsloth UD 파일에만), 3090에
통째로 든다[유도]. 처분은 `docs/plan-triage.md`(q35 사슬 카드, 사용자 결정: 순서·다운로드 둘)와 plan M4 행이다. 아래는 라운드의 보고 원문이다.

모든 조사를 마쳤습니다. 보고서입니다.

# qwennext-lit 보고 (조사 라운드, 리드 bloomery-69 앞)

**결론.** 추천은 한 프로그램입니다. `qwen35` 계열의 Gated DeltaNet(GDN) 프로그램을 Qwen3.6-35B-A3B로 먼저 세우고, 같은 사슬 뒤쪽에 Qwen3.8-27B를 붙입니다. 두 모델은 GDN 커널을 그대로 나눠 씁니다. 다른 점은 V 헤드 수 32 대 48뿐이고 이것은 launch 인자입니다.

순서에는 표면화할 충돌이 하나 있습니다. 사람들이 실제로 돌리는 "최신" Qwen은 Qwen3.8-27B입니다. unsloth GGUF가 약 6.9M 다운로드로, Qwen3.6-35B-A3B의 약 1.3M을 크게 앞섭니다. 그런데 27B에는 추가 비용이 셋 붙습니다.
- GROUP 6 flash 타일이 새로 필요합니다.
- dense FFN 역할이 필요합니다.
- 공개 비교선이 llama.cpp + MTP라서 MTP 링크가 필수입니다.

그래서 추천 순서는 35B-A3B 먼저입니다. 순서는 리드와 사용자가 정할 일입니다.

**가장 큰 새 사실.** lmstudio-community가 올린 두 모델의 Q4_K_M 파일은 헤더상 Q4_K와 Q6_K뿐입니다. 이것은 우리 Qwen3-30B-A3B 파일과 같은 타입 집합입니다. 그러니 linear-attn.md의 a5 단계인 Q5_K 카드 형식 + `_sel`은 임계 경로에서 빠집니다. 그 요구는 unsloth UD 파일에만 있습니다.

## 1. 순위표, 추천, 라운드 사슬

| 순위 | 모델 · 아키텍처 · 공개일 | 왜 중요한가 | 재사용 vs 신규 | 크기 | 가장 틀리기 쉬운 항 |
|---|---|---|---|---|---|
| 1 | Qwen3.6-35B-A3B · `qwen35moe` · 2026-04-15 | 3090 목표 문장(PR #83 252 tok/s)의 모델. 공개 3090 기준이 있음: llama.cpp 비투기 115.7 tok/s(UD-Q4_K_XL), pp2048 4,436 → 5,490(PR #29353). 3090에 통째로 들어감 | **재사용**: 라우터 규칙(softmax·renorm, 폭만 256/8), `q4k_sel`·`q6k_sel`, `gemm.rs`의 Q4_K/Q6_K, Q6_K 헤드, GROUP 8. **신규**: GDN(conv+prep·delta·norm_gate), HEAD 256 flash와 prefill flash, 출력 게이트, 부분 rope 64/256, 공유 전문가의 sigmoid 게이트 | L (M 넷 + S 셋) | 3090 무드래프트 디코드가 252를 넘는가. 유도 219–248이라 넘지 못할 쪽 |
| 2 | Qwen3.8-27B · `qwen35` · 2026-08-05 | 가장 많이 돌리는 최신 Qwen. 파생 파인튜닝도 전부 `qwen35`. 3090 llama.cpp 41.55 → MTP 66.39 tok/s(비손실 아님) | 1번의 GDN·IMROPE·게이트 전부 재사용. **신규**: GROUP 6 flash(assert가 막음), dense FFN 역할, H_v 48 k-head 매핑(인자), MTP 초안 | 1번 위에 M + M | dense 디코드는 대역폭 싸움. 우리 dense Q4_K gemv가 A6000에서 약 560 GB/s 이상 나와야 동률 |
| 3 | Qwen3.8-Flash-Next · `qwen4exp` · 2026-08-24 | 가장 새것. 모양이 호스트 티어 MoE + PLE(engram 닮음) + hc라 V4.1 기반과 가장 닮음 | GDN 재사용. **신규**: QSA 인덱서, PLE(IQ4_NL), 저랭크 hc, 512/10 전문가, Q5_1 `_sel` | L+ | PLE·QSA 오라클 크기, 111 GB 파일 |
| 4 | Qwen3-Coder-Next · `qwen3next` · 2026-02 | 코더 수요. 박스의 파일은 IQ4_XS 42.68 GB | GDN 재사용(k-head `h / ratio`, 합쳐진 `ssm_ba`). **신규**: IQ4_XS 호스트 dot(qdot에 없음), MXFP4 shexp 배선, 512/10 | L | IQ4_XS 경로 전체 |
| 5 | Qwen3 dense 8/14/32B · `qwen3` · 2025-04 | 싸지만 "최신"이 아님 | GDN 없음. dense 역할만 신규 | S–M | 없음(보류 권장) |

**파일 선택이 사슬을 바꿉니다.** 파일별 타입과 우리 커널 보유 여부:

| 파일 | 타입 | 우리 쪽 공백 |
|---|---|---|
| lmstudio Qwen3.6-35B-A3B-Q4_K_M, 21.17 GB | Q4_K / Q6_K / F32. 전문가 down은 Q6_K 20층, Q4_K 20층 | 형식 공백 없음. MTP 층 없음(block_count 40) |
| lmstudio Qwen3.8-27B-Q4_K_M, 16.81 GB | Q4_K / Q6_K / F32. `nextn` MTP 층 포함(block_count 65) | 형식 공백 없음 |
| unsloth Qwen3.6 UD-Q4_K_XL, 박스에 있음 | Q8_0 투영, down Q5_K 36층 | Q5_K 라우팅 `_sel`(`placement.rs:326-328`가 Q5_K를 뺌), Q8_0 GEMM(`gemm.rs:121-131`) |
| unsloth Qwen3.6 UD-Q3_K_M / Q3_K_XL | gate·up IQ3_XXS, down IQ4_XS, 투영·shexp Q8_0 | IQ 배선 전체. 토큰당 활성 2.46 GB로 Q4_K_M(1.98 GB)보다 **더 무거움** |
| unsloth Qwen3.8-27B UD-Q4_K_M / Q4_K_XL / Q5_K_M | IQ4_XS · IQ4_NL · IQ3_S · Q3–Q6_K · Q8_0 혼합 | IQ 경로 전체 |

- 카드의 IQ4_XS·IQ3_XXS 행 코어는 `iq.rs`에 있고 `gate-gpu-iq`로 게이트됩니다. 하지만 `CardFormat::of`(`placement.rs:241-256`)가 IQ를 거부해서 어느 파일도 그 코어를 못 씁니다.
- qdot에는 IQ3_XXS와 MXFP4 호스트 dot이 있고 IQ4_XS는 없습니다.
- lmstudio 두 파일은 박스에 없습니다. 다운로드 21.17 GB와 16.81 GB는 plan.md M4의 "사용자 결정(다운로드)" 칸에 해당합니다.

**맞는지(fit), [유도]:**
- Qwen3.6 Q4_K_M는 3090에 들어갑니다. 파일 21.17 GB + KV(20,480 B/키, 32k에서 0.67 GB) + 상태 2 × 65.8 MB + 여유 약 1.6 GB = 약 24.2 GB로, 가용 25.77 GB 안입니다.
- Qwen3.8-27B Q4_K_M도 들어갑니다. 16.8 GB + KV(65,536 B/키, 32k에서 2.1 GB) + 상태 2 × 157 MB + 여유 = 약 20.9 GB입니다.
- 같은 모델의 Q6_K(22.43 GB)는 32k에서 넘치고 8k에서는 들어갑니다.
- Flash-Next(111 GB)와 Coder-Next Q4_K_M(48.5 GB)은 호스트 티어가 필요합니다.

**추천 사슬 (Qwen3.6-35B-A3B → Qwen3.8-27B):**

`q35meta → q35oracle → (q35gdn ‖ q35attn ‖ q35moe) → q35body → q35time → q35chunk → q38dense → q38mtp`

| 링크 | 내용 | 크기 | 게이트 |
|---|---|---|---|
| q35meta | `arch/qwen35moe/{hparams,names,roles,kv,place,host}.rs`, `Arch::Qwen35moe`, 이름 붙은 거절. UD 파일의 Q5_K 라우팅 스택은 호스트로 조용히 넘기지 않고 거절 | S | 두 파일 인벤토리 게이트 + 거절 문구 핀 |
| q35oracle | `tools/ref/models/qwen35moe.sh` step 변형 ik 덤프. linear-attn §7의 Q1(클램프가 발동하는가), Q3(`new_state` 탭이 유효한가), Q5(텍스트 t=h=w)를 닫음 | S, 임대, 약 10–20분[유도] | MANIFEST, t 단계 `new_state` = t+1 단계 `state_in` 비트 일치 |
| q35gdn | `crates/gpu/src/linear/{conv,delta,norm_gate}.rs`. 재귀 커널, `DECAY=Scalar`, SLOTS, k-head 매핑을 인자로 | M | 호스트 규칙 비트 동일 + ik 탭 대비 밴드, 깊이 6/1024/4096 |
| q35attn | `flash_gqa`·`flash_gqa_prefill`의 HEAD를 const generic으로(128 기존 + 256), q/gate 분할과 sigmoid 게이트, `rope_neox` 부분 64/256 | M | 기존 HEAD 128 인스턴스의 `just ptx-scan` 표가 동일(이동 클래스) + `kqv_out` 밴드 |
| q35moe | 라우터 폭 256/8, 공유 전문가와 sigmoid 게이트 | S | 라우터 탭: id 동일, 가중치 비트 |
| q35body | `ChainBody`, 상태 슬롯 인덱스를 장치 스칼라로(그래프 하나), `gemm.rs` ubatch 프리필 + 재귀 GDN | M | e2e 토큰 개수 핀, `l_out-N` 밴드, 프리필 밴드 조항(qwen3prefill 형태) |
| q35time | A6000 한 임대에서 깊이 6/1024/4096 디코드와 pp512/pp4096. 대상 우리 / llama.cpp / mistral.rs | 시팅 ≤ 30분 | — |
| q35chunk | 청크 GDN 프리필(C = 64, 텐서코어) | M | 매 청크 끝 상태를 우리 재귀 커널 대비 밴드 + e2e 프리필 밴드. 비트 동일이 아님: PR #29353 상위 토큰 일치 98.8 % |
| q38dense | Qwen3.8-27B: dense FFN 역할, GROUP 6 타일, 64층 | M | 27B e2e 핀 |
| q38mtp | `nextn` 초안을 `step_pair`와 k슬롯 상태로 | M | `tokens` 줄 = 초안 없는 실행(`gate-gpu-ds41-draft` 형태) |

**MTP는 두 가지 이유로 표면화합니다.**
- 사용자는 이전에 "드래프트 없이 3090에서 252 초과"(plan.md:131)라고 목표를 정했습니다. 27B 경쟁선은 MTP 없이는 넘을 수 없습니다.
- 공개 llama.cpp MTP는 탐욕 출력을 바꿉니다: "23 to 25 of 25 prompts diverged". 우리 초안 경로는 무손실이 게이트로 핀되어 있어 차별점이 됩니다.

## 2. 새 커널 계산서: GDN delta 규칙 (Qwen3.6-35B-A3B 기준)

**설계 선택.** mainline의 워프-값열 방식을 따릅니다.
- 상태를 레지스터에 두고 `[v][k]` 배치로, 토큰마다 워프 셔플 두 번만 합니다(`gated_delta_net.cu:63,93,107@53ed051ce`).
- ik는 토큰마다 블록 배리어 둘과 블록 reduce를 합니다(`delta-net.cu:125,140,176@db517b69`). 그래서 프리필 토큰 지연이 더 깁니다.
- 되감기는 exllamav3의 히스토리 슬롯(`[slots][max_history+1][H][128][128]`, `gated_delta_net.py:188@0740edc`)과 우리 핑퐁 설계를 합친 k+1 슬롯으로 합니다.
- 청크 전환 기준은 셋이 갈립니다. exllamav3는 `seqlen >= num_v_heads`이고 히스토리가 없을 때(`gated_delta_rule.py:166`), mistral.rs는 64(`backend.rs:15@d5ae0f18f`), mainline은 PR에서 128 이상입니다.

**층당 분해 (디코드 토큰 하나, [유도], 헤더 차원과 타입에서):**

| 항 | 값 |
|---|---|
| GDN 투영 가중치 (lmstudio Q4_K_M) | 21.1 MB/층 (30층 합 633.0 MB) |
| 투영 MAC | 33.7 M |
| conv MAC | 32.8 K |
| delta 연산 | 3.67 MFLOP (헤드당 7·d², 32헤드) |
| 상태 | 2,097,152 B + conv 창 98,304 B. 읽기와 쓰기 합 4.39 MB |
| 새 launch | 3 (conv+prep, delta, norm_gate). norm_gate는 delta 저장에 접을 수 있어 2 |
| delta 격자 (mainline 방식) | 블록 (32헤드 × 32) = 1,024개, 스레드 128. 3090(82 SM)과 A6000(84 SM)에서 한 wave |

**스텝 자원 시간선 (디코드, A6000, 깊이 6, [유도]).** 벽시계는 카드 커널의 직렬 합입니다. 다른 자원의 그늘은 없습니다.

| 자원 | 바쁜 시간 | 근거 |
|---|---:|---|
| 카드, 가중치 | 4.34–4.64 ms | 1,975.7 MB를 426–455 GB/s로. Qwen3-30B-A3B 실측 4.728 ms에서 노드 604개 × c_node 0.37–0.85 µs를 뺀 실효 대역 |
| 카드, 노드 고정비 | 0.22–0.51 ms | 약 600노드 |
| 카드, GDN 상태 | 0.22 ms | 131.7 MB를 600 GB/s로 |
| 카드, conv | 0.09 ms | 30층 × 약 3 µs 하한 |
| 호스트 | µs 단위 | cuGraphLaunch |
| PCIe | ≈ 0 | 토큰과 폴트 워드 |
| **벽시계** | **4.88–5.46 ms** | **183–205 tok/s**. 깊이 4096에서 GQA +0.29 ms로 174–194 |

- GDN 항은 약 0.30 ms로 스텝의 6 %입니다. 새 커널은 디코드 선을 정하지 않습니다.
- 3090으로 옮기면 219–248 tok/s입니다[유도: 실효 대역을 936/768로 확대, 고정비는 그대로. 다른 카드에서 옮긴 값이지 측정이 아닙니다]. PR #83의 252 아래입니다.
- UD-Q3_K_M은 활성 2.46 GB라 더 느립니다. 이것이 표의 위험 항입니다.

**512 ubatch 자원 시간선 (프리필, A6000, [유도]).** Qwen3-30B-A3B의 ubatch 82.1 ms(pp512 6,236)에서 출발합니다.

| 자원 / 항 | ms | 근거 |
|---|---:|---|
| 카드, 전문가 GEMM | ≈ 43 | Qwen3 43 ms × MAC비 0.75 × 효율 벌점 1.6(전문가당 열 16개, a5gemm T=256 대 512) × 40/48 |
| 카드, dense 투영 GEMM | 19.8 | 1,284 M MAC/토큰, 66.4 TOPS (a5gemm dense T=512 실측) |
| 카드, 나머지 | ≈ 20.7 | Qwen3의 (82.1 − 43 − 14.0) × 40/48 |
| 카드, GDN 재귀 | **22.1** | 30층 × 512 × 1.44 µs. 1.44는 PR #29353의 3090 수치에서 낸 하한: pp2048이 4,436 → 5,490, 벽시계 0.462 s 중 0.089 s 절약 ÷ (30 × 2048) |
| 카드, conv/prep | 0.6 | — |
| PCIe | ≈ 0.04 | id 2 KB 입력, logits 한 행 1 MB 출력 |
| **벽시계** | **≈ 106 ms** | **pp512 ≈ 4,800 (±20 %)** |

- 재귀 항은 지연 바운드 직렬 항이고 어떤 그늘 아래에도 없습니다.
- SIMT 레버는 청크 커널입니다. ubatch당 90.4 GFLOP을 25–50 TFLOPS로 1.8–3.6 ms에 끝내고, 청크 간 상태 경로를 포함해 3–5 ms입니다. 그러면 **pp512 ≈ 5,700–5,900 (+19–22 %)**입니다.
- 비동기 레버는 두 반쪽 ubatch의 GDN을 서로의 GEMM과 다른 스트림에서 어긋내는 것입니다. 청크 커널을 만들지 않을 때만 값이 있습니다.
- 27B의 같은 항: 재귀 48층 × 512 × 2.10 µs = 51 ms. dense GEMM(24.9 TOP)은 227–376 ms라 재귀 항은 12–18 %입니다.
- 1.44 µs와 2.10 µs의 비 1.46은 블록 수 비 1,536/1,024 = 1.5와 맞습니다. 두 wave 해석과 일관됩니다.
- **linear-attn.md 정정:** §2.4는 토큰당 0.2–0.4 µs를 가정해 "프리필의 몇 %"라고 했습니다. 공개 수치에서 나온 하한은 5–7배 큽니다. §0 발견 2의 "청크 CUDA 커널 없음"도 이제 틀립니다. mistral.rs에 CUDA 청크 커널이 있고(`gdn.cu:909`), exllamav3는 FLA 청크를 쓰며, mainline에는 PR #26001과 #29353이 열려 있습니다.

**비트 오라클 계획.**
- 오라클은 ik입니다. 우리 하니스가 ik를 링크합니다.
- ik 규칙에는 우리와 다른 점이 둘 있습니다. 하나는 출력을 옛 상태에서 `decay·(Sᵀq) + u·(k·q)`로 내는 것입니다. 다른 하나는 g ≤ 50 클램프와 상태 ±1e6 클램프(`delta-net.cu:128,158`)입니다.
- 그래서 경로는 셋입니다.
  1. 우리 f32 순서를 호스트에서 비트 그대로 흉내 낸 규칙으로 커널을 비트 동일로 게이트합니다(`exact_ref --act ours`와 같은 형태).
  2. 같은 입력의 f64 재귀 시뮬레이션으로 깊이별 상태 표류 σ를 진단으로 냅니다. 감쇠가 수축적이라 깊이 6/1024/4096 밴드가 맞는 모양입니다.
  3. ik 탭 `new_state`, `final_output`, `l_out-N`에 대한 밴드와 e2e 개수 핀을 둡니다.
- q35oracle이 Q1에서 클램프 발동을 보면 ik는 규칙의 오라클이 아닙니다. 그때는 mainline 덤프를 전치해서 두 번째 기준으로 씁니다.
- 우리 쪽은 클램프하지 않습니다. 비유한 상태는 폴트 워드 사이트로 올립니다(no silent failure).
- IMROPE는 종이에서 닫혔습니다. `rope.cu:253-291@53ed051ce`에서 sections [11,11,10,0]의 32개 쌍 전부가 t/h/w로 가고, 쌍은 `(i0/2, i0/2 + n_dims/2)`입니다. 그러니 t=h=w=pos이면 첫 64차원 위의 NEOX와 같습니다. 텍스트 경로가 네 스트림을 같게 채우는지는 그래프 코드를 읽지 않았으니 q35oracle의 rope 탭으로 확인합니다.

**MTP 계산서 (27B).**
- 초안 한 토큰 = MTP 층 263.3 MB + 공유 헤드 1,043.0 MB = 트렁크 15,821 MB의 **8.3 %**[헤더에서 유도].
- 3위치 검증은 가중치를 한 번 읽으므로 약 1.0–1.1 스텝입니다.
- 공개 수락률은 3090 n-max 2에서 0.78이고, llama.cpp는 +59.8 %였습니다.
- 이음매는 `step_pair`와 k+1 상태 슬롯입니다(linear-attn §3.3의 일반화).

**Qwen3.8-27B 디코드 예측, A6000 [유도]: 27–42 tok/s.** 밴드가 넓은 이유는 dense gemv 실효 대역이 426–701 GB/s 중 어디인지 모르기 때문입니다. 이것이 측정 하나로만 닫히는 항입니다.

## 3. 출처, 스펙 밖 개선 여지, 공통 보고 항목

**웹 출처 (전부 이번 라운드에 가져온 것):**
- https://huggingface.co/api/models?author=Qwen&limit=60&sort=createdAt&direction=-1 : `Qwen/Qwen3.8-27B` createdAt 2026-08-05, downloads 6,579,319.
- https://huggingface.co/api/models?search=Qwen3.8&filter=gguf&sort=downloads&direction=-1 : `unsloth/Qwen3.8-27B-GGUF` 6,938,321. Qwen3.6 검색: `unsloth/Qwen3.6-35B-A3B-GGUF` 1,262,498.
- Qwen3.8-27B 카드 @1d4bf0f2ff60 (https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/README.md): "Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))".
- Qwen3.6-27B 카드 @6a9e13bd6fc8: "Number of Linear Attention Heads: 48 for V and 16 for QK". config.json이 3.8-27B와 같습니다.
- Qwen3.6-35B-A3B 카드 @995ad96eacd9: "Number of Parameters: 35B in total and 3B activated".
- Qwen3.8-Flash-Next 카드 @de4b8e4d43b9: "Number of Parameters: 125B with 6B activated, plus 51B n-gram embedding and 4B MTP".
- Qwen3-Coder-Next 카드 @a7fbcb5c0e12: "Number of Parameters: 80B in total and 3B activated".
- GGUF 헤더 (HTTP range로 읽음):
  - lmstudio-community/Qwen3.8-27B-GGUF @5a7da681f605 `Qwen3.8-27B-Q4_K_M.gguf`
  - lmstudio-community/Qwen3.6-35B-A3B-GGUF @68a34855558a `Qwen3.6-35B-A3B-Q4_K_M.gguf`
  - unsloth/Qwen3.8-27B-GGUF @4ca720788d1e: UD-Q4_K_M, UD-Q4_K_XL, UD-Q5_K_M, Q4_1
  - unsloth/Qwen3.6-35B-A3B-GGUF @a483e9e6cbd5: UD-Q3_K_M, UD-Q3_K_XL
  - 크기 목록은 각 repo의 `/api/models/<repo>/tree/main?recursive=1`
- llama.cpp PR #29353 (https://github.com/ggml-org/llama.cpp/pull/29353, 열림): "The RTX 3090 reported thermal throttling and varying clocks during these measurements". 행: "RTX 3090 | Qwen3.8-27B-Q4_K_M | 2048 | 1,150.37 | 1,301.14". 조건은 `-b 2048 -ub 2048`.
- PR #26001 (https://github.com/ggml-org/llama.cpp/pull/26001): "This PR adds a chunked mode to the Gated Delta Net (GDN) CUDA operator".
- 이슈 #22967: "still has the TODO (from #19504) for a chunked prefill kernel".
- 이슈 #29172: "throughput collapses once the KV position passed in via `-d` exceeds roughly 150K tokens".
- thc1006/qwen3.8-speculative-decoding-rtx3090 @705dc2242248: "MTP `n-max 2` is **+59.8 % [+57.0, +62.8]** on server-reported decode". 비투기 기준 41.55 tok/s, master `c060ca974c77`, UD-Q4_K_XL, `-c 8192`, 탐욕.
- thc1006/qwen3.6-speculative-decoding-rtx3090 @689fc3af83a8: 행 "no speculation | 115.7". llama.cpp `97895129e5f2`, UD-Q4_K_XL.
- sudoingX/qwen38-mtp @51187204b553: "llama.cpp added draft-mtp speculative decoding in PR #22673 (July 2026)". 같은 문서 규칙 6: "a current build was +10-15% on every quant". 기준선이 빌드마다 움직이니 커밋 없는 수치는 쓰지 않습니다.
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/37 : "Qwen3.6-35B-A3B with Unsloth's Q3 quant takes 23GB of VRAM, runs at 120 tok/s". WebSearch 요약에만 나온 157.66은 인용하지 않습니다.

**코드 출처 (박스, 읽기 전용):**
- mainline `/home/user/llama.cpp-mainline` @53ed051ce5e8:
  - `ggml/src/ggml-cuda/gated_delta_net.cu:4,37,63,93,107,145,180,183`
  - `src/models/qwen35.cpp:160-198,254-329,323,335-461,470,486-489`
  - `ggml/src/ggml-cuda/rope.cu:253-291`
- ik `/home/user/ik-idxkey` @db517b690a8c:
  - `src/graphs/build_qwen35.cpp:88,110-127`
  - `ggml/src/ggml-cuda/delta-net.cu:125,128,140,158,163,176,216`
  - `src/llama-build-context.cpp:2569`
- exllamav3 @0740edc2da56:
  - `exllamav3_ext/gdn.cu:426-435`
  - `modules/gated_delta_net.py:188,1026`
  - `modules/gated_delta_net_fn/gated_delta_rule.py:166`
- mistral.rs @d5ae0f18f217:
  - `mistralrs-core/src/gdn/backend.rs:15,525-535`
  - `mistralrs-core/src/cuda/gdn.cu:277-288,909-926,3590-3600`
- ik 로그 `/home/user/ikbatch-qwen36.log`: UD-Q6_K, ik c10fbbcc, B=1에서 TG 133.84 tok/s. 카드가 기록되지 않았고 09-22 이전이라 **공개 표에 넣을 수 없는** 참고치입니다.
- 우리 트리 @1475df3: `flash_gqa.rs:61-63,82,87`, `flash_gqa_prefill.rs:99`, `rope_neox.rs:45`, `arch/qwen3moe/router.rs:68-71`, `gemm.rs:121-131`, `placement.rs:230-256,326-328`, `iq.rs`, `qdot/src/lib.rs:1-3`.
- rig-log: 09-25#q3pflash-ab (pp512 6,236, pp4096 5,557), 09-25 q3unfold 절 (211.49 tok/s, 4.728 ms, 604노드), 09-25#a5gemm-a6000 (dense T=512 66.4 TOPS).

**스펙 밖 개선 여지 (보고만, 손대지 않음):**
- `crates/gpu/src/flash_gqa.rs:61-63,82,87`와 `flash_gqa_prefill.rs:99`: HEAD 128과 GROUP 8이 assert로 고정돼 있어 HEAD 256·GROUP 6을 못 받습니다. const generic으로 바꾸는 일, M.
- `crates/gpu/src/rope_neox.rs:45`: 머리 전체 128 회전만 있고 부분 rope가 없습니다. S.
- `crates/gpu/src/arch/qwen3moe/router.rs:68-71`: 폭이 상수입니다. `route_core.rs`는 score와 tie만 일반화합니다. S.
- `crates/gpu/src/gemm.rs:121-131`: Q8_0 가중치가 없어 UD 파일의 Q8_0 투영이 GEMM 프리필을 못 탑니다. S–M.
- `crates/model/src/placement.rs:326-328`: `of_routed`가 Q5_K를 "호스트에 남김"으로 돌립니다. 호스트 티어가 없는 아키텍처에서 이것이 이름 붙은 거절인지 확인이 필요합니다. XS.
- `crates/gpu/src/iq.rs` 대 `placement.rs:241-256`: 게이트된 IQ 코어를 쓰는 배치가 없습니다. `_sel` 배선, M.
- 문서 (리드가 씀, 각 XS):
  - `docs/research/linear-attn.md` §0 발견 2와 §2.4: 위 정정.
  - `docs/research/models-survey.md` 표 C의 Qwen3.6 행: "Q5_K"는 unsloth UD 파일에만 해당합니다.
  - `docs/plan.md:131`: "Q3_K급 파일"로 적혀 있지만 공개 Q3 파일은 IQ이고 Q4_K_M보다 더 무겁습니다.

**변경 파일:** 없음.

**게이트와 증명:** 조사 라운드라 실행한 게이트가 없습니다.

**예측 대 결과:** 모든 예측이 [유도]이고 측정되지 않았습니다. 다음 측정 둘이 가장 큰 불확실성을 닫습니다.
- dense Q4_K gemv 실효 대역. 예측 426–701 GB/s.
- q35time의 A6000 같은 임대 llama.cpp 대비 비율.

**못 한 것과 이탈:**
- 참조 트리에서 읽기 전용 `git log`를 한 번 실행했습니다. 상태는 바뀌지 않았지만 스펙의 "never git there"를 어겼습니다. 그 뒤로는 `.git/HEAD` 파일만 읽었습니다.
- Coder-Next 파일은 스펙의 `/home/user/models`가 아니라 `/models/Qwen3-Coder-Next/Qwen3-Coder-Next-IQ4_XS.gguf`(42,676,135,968 B)에 있습니다. 헤더를 확인했습니다: `qwen3next`, 512/10, 대부분 IQ4_XS, shexp MXFP4, `pre qwen2`.
- 스펙 괄호의 사실 정정: IQ4_XS 호스트 dot은 없지만 카드 행 코어는 있고, 배치가 그것을 쓰지 않습니다.
- 박스의 임시 스크립트 `/tmp/qwennext_gguf_hdr.py`는 실행 직후 지웠습니다.
- 3.8-27B의 A6000 공개 절대값은 찾지 못했습니다.

**모델:** opus(Opus 5.5)로 실행했습니다. 지시대로 은퇴한 에이전트에게는 SendMessage를 보내지 않았습니다.
