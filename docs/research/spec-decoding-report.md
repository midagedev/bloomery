# specresearch 보고: 호스트 대역폭에 묶인 MoE 엔진에서 DSpark를 이기는 투기적 디코딩

모델: Opus 5.5 (1M context). 코드 편집, 게이트, 박스는 쓰지 않았다. 외부 출처는 12개이고 전부 이번 라운드에서 직접 받아 문장을 대조했다. arXiv는 `export.arxiv.org/abs`의 초록 원문, PDF는 `pdftotext` 결과로 확인했다.

## 요약표

조건은 전부 n=1 단일 스트림, A6000 + 호스트, 깊이 무관이다. 수치는 [유도]이고, 규칙은 스펙 본문의 `D + dense + m·(k+1)`을 따른다.

| 방법 | 옮기는 항 | tok/s 오늘 | B12 뒤 | 확인 측정(≤30분) | 출처 |
|---|---|---|---|---|---|
| 기준(투기 없음) | — | 25.3 | 30.7–33.6 | — | 스펙 |
| 평범한 검증 + DSpark, k=1 | dense 상각 | 22.4 (D=10.3) / 24.7 (D=3) | 27.4–30.1 / 30.9–34.3 | — | rig-log 09-19 |
| **층 어긋남(skew) 2행 검증 + DSpark, k=1** | dense를 호스트 그늘로 | **26.8 (D=10.3) / 30.2 (D=3)** | **34.4–38.7 / 40.0–46.1** | 가짜 2행 skew 패스 시간 대 2m+fill | 선행 사례 없음(§1) |
| skew + n-gram 초안 | D≈0.35 ms | 수용률 a에 달림, 손익분기 a>0.36 | 손익분기 a>0.15–0.23 | 편집 워크로드 수용률 | llama.cpp 문서 |
| skew + Markov 헤드 단독 | D≈0.15 ms, 앞서 달릴 수 있음 | 손익분기 a>0.36 | a>0.15–0.22 | 오프라인 argmax 일치율 | — |
| 상주 expert만 쓰는 자기 초안 | D≈G+ce≈14 ms | a=1이어도 25.2 이하 | 30.6–34.1 | 제한 라우팅 argmax 일치 | DraftExpert(비관 사전) |
| 두 번째 스트림(서빙, 합계) | 수용률 1.0, skew 가능 | 합계 30.3 / skew 37.5 (n=1 이득 0) | 합계 38.3–43.0 / skew 50.6–59.0 | ik `-np 2` 한 번 | KTransformers·llama.cpp |

**전제.**
- 모든 계산은 스펙의 값을 쓴다: G 12.95 ms, m 26.54 ms(3.409 GB ÷ 128 GB/s), 카드 expert ce 1.10 ms(0.634 GB ÷ 575 GB/s).
- B12 뒤 m은 plan의 2.15–2.51 GB를 같은 128 GB/s로 나눈 16.8–19.6 ms이고, ce는 2.7–3.3 ms다.
- plan B12 행의 호스트 항 23.08 ms는 147.7 GB/s 기준이라 여기서는 섞지 않는다.
- 수용률은 llama.cpp 정의(수락 ÷ 초안)를 따른다. 그래서 패스당 토큰은 E = 1 + k·a_k이고, k=1이면 1.702, k=2이면 2.14다.
- **교정.** 이 규칙은 ik를 재현한다. 50.21 + 10.32 + 17.28 = 77.81 ms로 실측 77.63 ms와 맞고, 1.702 ÷ 77.63 ms = 21.9 tok/s로 실측 21.83과 맞는다(rig-log 2026-09-19:128).

## §1 dense 항을 상각하지 않고 숨긴다

**산술.** skew된 k=1 패스는 D + max(2m, 2(G+ce)) + fill이다. fill은 한 층 dense로 G/40 = 0.32 ms다. 오늘은 호스트 쪽 53.1 ms가 카드 쪽 28.1 ms보다 길어서 dense가 통째로 숨는다.

| D | tok/s 오늘 |
|---|---|
| 0 | 31.9 |
| 3 | 30.2 |
| 10.3 | 26.8 |

- **스펙의 "≈30"은 D=3을 가정한 값이다.** 이 하드웨어에서 잰 DSpark 초안 비용은 ik의 10.3 ms뿐이다. DSpark 초안은 검증된 위치의 37–39층 은닉 상태를 읽으므로, PipeSpec식으로 앞서 달릴 수 없다.
- **k=2는 손해다.** skew에서도 25.8 tok/s(D=3)로 k=1보다 낮다.
- **수용률이 1이어도 상한은 1000/m = 37.7 tok/s다.** 평범한 검증이 위치마다 m을 읽는 한 넘을 수 없다.
- **B12 뒤에는 병목이 바뀐다.** B12 하단에서 2m = 33.6 ms이고 2(G+ce) = 32.5 ms다. skew에서는 dense를 행마다 따로 읽으므로 카드도 임계 경로에 함께 올라온다. 호스트 바이트를 더 줄여도 dense를 2행 배치로 다시 묶기 전까지는 이득이 멈춘다. "≈51"은 이 경계를 무시한 값이고, D=3에서 46.1이 상한이다.

**선행 사례.** 각 시스템이 무엇을 겹치는지 정리했다.

- **KTransformers의 Expert Deferral이 가장 가깝다.** 한 디코드 행 안에서 CPU expert(층 k)와 GPU attention(층 k+1)을 겹친다. 대신 모델을 바꾼다. 일부 expert 출력을 k+2층으로 미루고, 정확도 변화를 0.5 % 이하로 보고한다. 하드웨어는 Xeon과 A100이다. 리드의 skew는 둘째 행을 이용해 같은 겹침을 무손실로 얻는다.
- **MoE-Lightning(CGOPipe)은 독립 시퀀스의 마이크로배치 사이를 겹친다.** 배치 처리량이 목적이고 CPU가 attention을 맡는다. 하드웨어는 T4다. 한 시퀀스 안의 겹침은 아니다.
- **Fiddler는 겹침을 하지 않는다.** 논문 본문에 "overlap"이라는 단어가 0회 나온다. 입력 수에 따라 expert를 CPU에서 계산할지 GPU로 옮길지만 정한다.
- **PipeSpec은 초안 모델과 검증 모델 사이를 여러 장치에 걸쳐 비동기로 겹친다.** 즉 D 항을 겹치고, 층을 겹치지는 않는다.
- **MoE-SpeQ와 DraftExpert의 prefetch는 PCIe 전송을 겹친다.** 둘 다 expert를 GPU로 옮겨 계산한다는 전제다.
- **확인하지 않은 것.** HOBBIT, ProMoE, SiDA, Pre-gated MoE는 이번 라운드에서 받지 않았다. 성격을 규정하지 않는다.

**발견: 확인한 문헌 중 검증 행에 층 skew를 거는 것은 없다.**

## §2 초안 비용이 0에 가까운 경우

skew를 걸면 D가 차지하는 몫이 작아진다. D를 3 ms에서 0으로 줄이면 30.2 → 31.9 tok/s이지만, 수용률을 0.70에서 0.85로 올리면 34.6 tok/s다. 판정 축은 skew k=1의 손익분기 수용률이다.

- **n-gram / PLD.** D≈0.35 ms이고 host-surplus H3가 이미 다뤘다. 손익분기는 a > 0.36(오늘), 0.15–0.23(B12 뒤)이다. 평범한 검증이면 a > 0.68이어야 한다. 수용률은 워크로드에 달려 있다. 13k 프리픽스 편집 워크로드의 수용률을 재야 한다. 은닉 상태가 필요 없으므로 3090이나 호스트에서 앞서 달릴 수 있다.
- **Markov 헤드 단독(66 MB).** 66 MB ÷ 575 GB/s = 0.11 ms에 런치 몇 개를 더해 D≈0.15 ms다 [유도]. 스펙 설명대로 직전 토큰만 입력으로 받는다면 앞서 달릴 수 있다. 수용률은 모른다. 측정은 오프라인 CPU 작업 하나다. 타깃이 생성한 토큰 열에서 argmax(markov(이전 토큰)) == 다음 토큰의 비율을 몇 분이면 잰다. 0.36을 넘으면 오늘도 이득이다.
- **상주 expert만 쓰는 자기 초안.** D≈G+ce = 14.05 ms이고, B12 뒤에는 15.7–16.3 ms다. 손익분기는 a > 0.71(skew)이고, 평범한 검증이면 a=1이어도 진다. 받아들인 토큰이 있어야 돌 수 있으므로 앞서 달릴 수도 없다. DraftExpert가 비관적 사전 근거다. 순진한 shared+top-r 자기 초안은 22 %, 31 %, 42 %이고 "top-3도 50 % 아래"다. hot-64가 선택의 61–74 %를 덮는다는 것은 argmax 일치와 다른 척도다. 측정은 두 가지다. 상주 집합만으로 top-k를 제한한 teacher-forced 실행을 1만 토큰(약 2–3분) 돌려 위치별 argmax 일치와 KLD를 잰다. 다만 제한 라우팅 레버는 새 코드다.
- **층 건너뛰기 자기 초안(SWIFT, Kangaroo, LayerSkip, Draft&Verify).** 층 비율 f인 초안의 비용은 f·(G+m)이다. 층의 절반이면 D≈20 ms라 비용만으로 기각된다 [유도]. 출처는 받지 않았다.
- **EAGLE-3.** k번 순차 실행하므로 D≈k×(한 층 + 헤드 0.8 ms)이다 [유도: q6_K 헤드를 약 0.54 GB로 봤다]. V4.1용 헤드를 새로 학습해야 한다. DSpark는 이미 있으므로 우선순위가 낮다.

## §3 수락 토큰당 라우팅 바이트를 무손실로 줄이는 방법

- **트리 검증은 더 나쁘다.** 노드 N개면 비용이 N·m이다. 폭 2, 깊이 1 트리는 D + G + 3m이 드는데 토큰은 1.702에 둘째 후보 몫만 더해진다. 둘째 후보가 +0.1(가정)이어도 1.8 ÷ 95.6 ms로 k=1보다 낮다. MoE-Spec과 EVICT의 초안도 같은 물리를 말한다: "large draft trees activate many unique experts".
- **무손실 계열 둘.** EVICT는 트리를 잘라 비용 대비 이득이 있는 접두만 검증한다. EcoSpec은 expert를 공유하는 초안을 고르고, "without modifying the target-model verification rule"이다. 여기서 k=1이면 고를 후보가 사실상 하나라 이득 폭이 작다. 초안이 정해진 한, 두 위치의 기대 공유는 층당 0.094개다.
- **prefetch는 여기서 숨길 것이 없다.** MoE-SpeQ와 DraftExpert의 prefetch는 expert를 PCIe로 GPU에 옮겨 계산한다는 전제다. 우리 호스트 티어는 DRAM에서 제자리 계산을 하므로 DRAM 바이트가 곧 계산이고, 이미 대역폭이 포화돼 있다. 바이트를 줄이는 유일한 무손실 경로는 토큰 사이의 재사용, 즉 카드 캐시다. 그것은 B12이고, 온라인 갱신이 그 연장이다.
- **손실 계열.** MoE-Spec은 검증 시점 expert 예산으로 긴 꼬리를 버린다. 스스로 "trade accuracy for further latency"라고 쓴다. KTransformers의 Expert Deferral은 계산 그래프 자체를 바꾼다.
- **스펙의 추정 정정.** MoE-SpeQ는 손실 계열이 아니다. 초록상 prefetch와 지연 은닉 시스템이다.

## §4 두 번째 스트림

두 번째 스트림의 합계 처리량은 다음과 같다. n=1 사용자에게는 이득이 0이다.

| 조건 | 오늘 | B12 뒤 |
|---|---|---|
| skew 없음, 2 ÷ (G + 2m) | 30.3 | 38.3–43.0 |
| skew, 2 ÷ (2m + fill) | 37.5 | 50.6–59.0 |

skew k=1 검증의 상한(a=1)도 1000/m = 37.7이다. 같은 바이트를 읽으면서 수용률 0.70을 1.0으로 바꾼 것이 두 번째 스트림이다. **단일 사용자면 skew + 검증, 서빙이면 skew + 스트림이고 서빙에는 투기가 필요 없다.**

- **KTransformers + SGLang.** 8중 동시 요청에서 요청당 약 20 tok/s이고, 이득은 "placing more experts on GPUs, which reduces CPU memory accesses"에서 온다고 쓴다. 쓰는 수단은 동시 요청 배치다.
- **llama.cpp.** `-np`(슬롯), 연속 배치, `--n-cpu-moe`가 같은 서버에서 공존한다.
- **Fiddler.** CPU 지연이 입력 수에 "almost linearly"로 늘어난다고 본다. 연산 바운드인 bf16 기준이다.
- **ik로 가장 싸게 확인하는 법.** 09-19 적합식으로 두 실 토큰은 50.21 + 17.28 = 67.5 ms, 즉 합계 29.6 tok/s(대 19.92)가 예측된다. 위험은 메모리 기록의 "ik MoE decode does not batch"다. GPU 상주 expert 쪽에서 배치 2가 더 느렸던 전례가 있다.

## §5 ik가 하지 않는 것, 이득 × 확신 순

1. **skew 2행 검증 + DSpark k=1.** dense 항 G를 숨긴다. 오늘 26.8(D=10.3)–30.2(D=3), B12 뒤 34.4–46.1 tok/s다. 측정은 lease 하나 안에서 한다. 수락과 무관하게 둘째 행에 임의 토큰을 넣은 skew 패스를 캡처하고, 패스 시간이 2m + 0.32 + D에 오는지 본다. 위험은 셋이다. 두 행이 같은 층 대기를 교대하는 그래프를 새로 짜야 한다. 거부 시 KV 롤백이 필요하다. B12 뒤에는 카드가 공동 임계가 된다.
2. **skew + 앞서 달리는 싼 초안(Markov 헤드 단독 또는 n-gram, DSpark 폴백).** D를 0에 가깝게 만든다. a=0.5면 오늘 1.5 ÷ 53.6 ms = 28.0, a=0.7이면 31.7 tok/s다. 측정은 오프라인 argmax 일치율로, 박스 CPU에서 수 분이다. 위험은 수용률 미지수다.
3. **서빙용 skew 2스트림(C3 형제).** n=1 이득은 0이고 합계 37.5 tok/s다. 측정은 ik `-np 2` 한 번과 우리 skew 패스 한 번(1번 측정과 같은 것)이다. 위험은 배치 2의 GPU expert 경로다.
4. **k를 1로 고정하는 것.** 규칙상 k≥2는 항상 손해다(skew에서도 25.8 < 30.2). 비용 0의 설정이다. ik는 1–3을 쓴다.
5. **상주 expert 자기 초안.** a > 0.71이 필요한데 사전 근거가 50 % 아래라 확신이 낮다. 제한 라우팅 1만 토큰 argmax 일치로 기각 여부를 정할 수 있다.

**보고만 하는 개선 여지.**
- `docs/plan.md`의 B12 행은 호스트 대역을 147.7 GB/s(23.08 ms)로, 이 스펙은 128 GB/s(26.54 ms)로 쓴다. 한 문서 안에서 단위가 갈린다. 크기는 S다.
- `docs/research/host-surplus.md` H3는 드래프트 비용이 "잰 적이 없다"고 쓰지만, rig-log 09-19가 ik에서 D = 10.32 ms(k에 평평)로 쟀다. 줄을 긋고 링크를 달면 된다. 크기는 XS다.

## 출처(12개, 전부 이번 라운드에서 받아 문장을 대조했다)

1. KTransformers SOSP'25 — https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf — "Expert Deferral strategically delays some routed experts' outputs to layer 𝑘 + 2" / "at the cost of only slight changes in model behavior" / "Expert Deferral is applied exclusively during the decode phase"
2. LMSYS KTransformers 블로그 — https://www.lmsys.org/blog/2025-10-22-KTransformers/ — "under 8-way concurrency … each request achieves nearly 20 tokens per second on average. The improvement mainly comes from placing more experts on GPUs, which reduces CPU memory accesses under bandwidth bottlenecks." / "accuracy variation below 0.5%"
3. MoE-Lightning — https://arxiv.org/abs/2411.11217 (PDF §4.1) — "The GPU sequentially processes the post-attention tasks … for the current micro-batch, followed by the pre-attention tasks … for the next micro-batch. Concurrently, the CPU handles attention … for the next batch"
4. Fiddler — https://arxiv.org/abs/2402.07033 (PDF) — "for smaller input sizes, it is more efficient to execute expert layers on CPUs, avoiding the overhead of weight transfer." / "When expert layers are executed on the CPU, latency increases almost linearly with the input size"
5. PipeSpec — https://arxiv.org/abs/2505.01572 — "generalizes speculative decoding to $k$ models arranged in a hierarchical pipeline, enabling asynchronous execution with lightweight coordination for prediction verification and rollback"
6. DraftExpert — https://arxiv.org/abs/2607.24434 (PDF) — "verifying a multi-token block activates the union of target experts and is no longer close to one target step" / "increasing r raises acceptance from 22% to 31% and 42%" / "even the much more expensive top-3 path remains below 50% acceptance" / 하드웨어 "NVIDIA GeForce RTX 4090 … CPU-resident routed experts loaded on demand to GPU"
7. MoE-SpeQ — https://arxiv.org/abs/2511.14102 — "enables a runtime orchestrator to prefetch these experts from host memory, effectively overlapping the expensive I/O with useful computation and hiding the latency from the critical path"
8. EcoSpec — https://arxiv.org/abs/2607.12696 — "favors draft paths that preserve high acceptance likelihood while reusing experts already covered by the current verification set, without modifying the target-model verification rule"
9. MoE-Spec — https://arxiv.org/abs/2602.16052 — "large draft trees activate many unique experts" / "with flexibility to trade accuracy for further latency reductions through tighter budgets"
10. EVICT — https://arxiv.org/abs/2605.00342 — "a training-free, hyperparameter-free, and lossless adaptive verification method … truncating the draft tree before target verification"
11. The Limits of Speculation — https://arxiv.org/abs/2609.22156 — "Speculative decoding in Mixture-of-Experts (MoE) models faces the problem of unstable verification cost caused by input-dependent expert loading."
12. llama.cpp server README — https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md — "`-ncmoe, --n-cpu-moe N` | keep the Mixture of Experts (MoE) weights of the first N layers in the CPU" / "`-np, --parallel N` | number of server slots" / "`-cb, --cont-batching` … (default: enabled)"

내부 근거는 두 곳이다. rig-log `log/2026-09-19.md:128`(D=10.32, m=17.28, 수용률 0.702/0.570/0.478)과 `docs/plan.md`의 B12 행이다. host-surplus H3는 반복하지 않고 n-gram 비용 0.35 ms만 참조했다.
