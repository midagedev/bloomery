# 양자화 디코드의 효율을 근본적으로 올리는 방법

> **리드 서문 (2026-09-19).** 이 보고서는 위임 라운드가 썼고 리드가 읽었다. 판정 논리는
> 강하다 — 특히 MoE-Infinity·ProMoE·Pre-gated 계열을 "우리 expert는 이미 DDR4에 상주하며
> 그 자리에서 연산되니 숨길 전송이 없다"로 기각한 것과, W4A4·회전 기법을 "M=1에서 활성값은
> 수 KB라 대역폭 항을 안 움직인다"로 기각한 것. 그 두 판별이 이 보고서의 값이다.
>
> **두 가지를 리드가 고친다.**
>
> **① 1위 레버가 틀렸다.** 표는 "DSpark/MTP 검증 −19 ms, 채택 후보"라고 적는다. 우리는
> 그것을 **이미 켜 두고 있다**(`configs/v41-serve.sh:90`, `--spec-draft-n-max 3`). 기준선
> 25.05 tok/s가 드래프트를 켠 값이다. 우리 자신의 쌍(`configs/v41-serve.sh:63`)은 드래프트
> 없이 17.7, 드래프트로 22.8 — **1.29배**지 1.93배가 아니다. 진짜 레버는 투기 디코딩을
> 도입하는 것이 아니라 **검증 패스가 expert를 패스당 한 번만 읽게 하는 것**이고, 그 값은
> 우리 숫자로 22.8 → 38.6 tok/s다. 계산은 `docs/roofline.md`의 마지막 절에 있다. 표의
> "expert 버킷 + grouped GEMM" 행과 "batch 2 ≥ batch 1" 행이 실제로는 1위 레버이고,
> DSpark 행은 그 둘의 전제가 아니라 **이미 지불한 비용**이다.
>
> **② 출처 등급.** 서로 다른 URL 77개 중 상당수가 2차 출처이고, 개중에는 내용 농장으로
> 보이는 GitHub 저장소와 arxiv 프록시 도메인이 섞여 있다. 보고서가 `2차`로 표시한 것은
> 정직하지만, **2차 표시가 붙은 수치는 설계 판단의 근거로 쓰지 않는다.** 1차 출처가 붙은
> 행(V3 MTP 리포트, Medusa, QTIP, AQLM, MegaBlocks, Fiddler, MoE-Infinity, ik discussion #8)만
> 판정에 쓴다.
>
> 나머지는 라운드가 쓴 그대로다.

날짜 2026-09-19. `mulle` 설계 입력용 조사 보고서. 읽기만 했고, 포팅하지 않았다.
이 문서의 모든 실측치는 출처 URL을 달고, 우리 박스에 대한 단언은 스펙과
[roofline](../roofline.md)에 있는 것만 쓴다. 거기 없는 숫자가 필요하면 마지막 절에
"재야 하는 측정"으로 적는다.

전제(스펙·roofline에서 가져옴, 단위 고정): V4.1 토큰당 11.701 GB =
호스트 expert 3.336 GB(DDR4, 실측 147.7 GB/s → **22.6 ms**) + GPU expert 0.708 GB +
dense 7.658 GB(VRAM, 가정 500 GB/s → 16.7 ms). ik 실측 39.92 ms/토큰(25.05 tok/s).
KV는 토큰당 ~23.6 MB로 0.2 %라 항이 아니다. V2-Lite 토큰당 1.307 GB.
"토큰당 바이트를 줄이거나, 한 번 읽은 바이트를 여러 토큰에 나누지 않는 기법은 큰 이득을
낼 수 없다"가 이 보고서의 판정축이다.

## 결정표

판정은 네 값 중 하나: **채택 후보** / **조건부**(조건 명시) / **기각**(사유 한 절) /
**측정 필요**(판정을 가르는 단일 측정 명시). V2-Lite·V4.1열은 그 모델에서의 전기대 효과다.

| 기법 | 루프라인의 어느 항 | 출처의 실측치 (URL) | 정확도 비용 | 우리 엔진에 필요한 구조 | V2-Lite | V4.1 | 판정 |
|---|---|---|---|---|---|---|---|
| DSpark/MTP 스타일 검증 (k토큰/패스) | 전부 ÷ k | DSpark k=5에서 50.8→98.1 tok/s(1.93×), 평균 수락 3.02~3.48 ([field](https://github.com/chinaboy0618/deepseek-v4-cmp170hx/blob/HEAD/DSPARK_PAPER.md)); V3 MTP 2번째 토큰 수락 85~90 %, 1.8× ([report](https://arxiv.org/pdf/2412.19437v1)) | 무손실(검증 통과분만 수락) | 순전파 API가 토큰 배치를 1급으로; expert별 버킷 디스패치; draft 헤드 가중치 확보 | 유효(전량 VRAM이라 검증이 쌈) | **최대 레버, −19 ms급** | **채택 후보** |
| EAGLE-3 | 전부 ÷ k | dense 3.0~6.5×, EAGLE-2 대비 +20~40 % ([paper](https://papers.nips.cc/paper_files/paper/2025/file/c7b5a35ea98b62512a869c19ea7b03cb-Paper-Conference.pdf)); MoE 실전형은 수락 길이 ~2.5 ([2차](https://github.com/aeonmindai/arc/blob/HEAD/research/moe_speculative_decoding.md)) | 무손실 | 학습된 draft 헤드 + 위와 같은 검증 구조 | draft 헤드 없음 | draft 헤드 없음 | **조건부**(V4.1용 학습 헤드가 있을 때만) |
| Medusa heads | 전부 ÷ k | Zephyr-7B AlpacaEval 34.21→99.50 tok/s, 수락률 3.08, 2.91× ([paper](https://arxiv.org/pdf/2401.10774)); Spec-Bench 1.48→2.42× ([survey](https://arxiv.org/pdf/2401.07851)) | 경미(MT-Bench ±0.1 내) | Medusa 헤드 학습 + 검증 구조 | 헤드 없음 | 헤드 없음 | **조건부**(EAGLE-3과 동일 조건, 우선순위 낮음) |
| Expert 버킷 + grouped GEMM 디스패치 | 검증 패스의 expert 중복 읽기 제거 (구조) | MegaBlocks가 패딩 대비 학습 1.38/2.0/4.35× ([paper](https://arxiv.org/pdf/2211.15841)); 라우팅+순열은 층 시간의 ~9 % ([측정](https://arxiv.org/pdf/2603.07685v1)) | 없음 | 토큰을 expert별로 정렬하는 순열 버퍼 + grouped GEMM (아래 Q3 구조) | batch≥2 수정과 함께 | 검증 패스의 전제조건 | **채택 후보** |
| ik IQ2_K/IQ3_K 계열로 expert 재양자화 | 호스트 3.336 GB → ~2.3 GB, GPU expert도 동율 | IQ2_K 2.375 bpw·IQ3_K 3.4375 bpw; AVX2 tg가 bpw 비례, IQ3_K가 llama.cpp 대비 tg 2.37× ([ik #8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)); IQ4_K 오차가 Q4_0의 1/2.7 ([동일](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)) | IQ4_K가 Q4_K_S 대비 오차 −40 % (동일 출처; 2비트대 PPL은 아래 Q2) | `quant` 크레이트에 IQ 디코드 경로 추가(CPU AVX2·GPU 둘 다 실증됨) | 효과 작음(VRAM에 여유) | 호스트 −7 ms급 | **채택 후보** |
| QTIP 2비트 (trellis) | expert 바이트/토큰 | Llama2-7B 2비트 6.89 (no-finetune) ([paper](https://arxiv.org/pdf/2406.11235v4)); FT 시 2b 5.86·3b 5.28·4b 5.17 ([2차](https://github.com/azimml/trellis-webgpu/blob/HEAD/notes/qtip-exl3-spec.md)); QuIP# 속도와 동등, 피크 대역폭 80 % 이상 ([2차](https://github.com/cksac/turboquant-model/blob/HEAD/research/recent-quantization-papers.md)) | 2비트에서 FP16 대비 +1.8 PPL (7B) | GPU trellis 디코드 커널; CPU 측은 순차 디코드라 별도 설계 필요 | 해당 | GPU expert만 | **조건부**(GPU 상주 expert에만; 호스트 티어는 불가 판정) |
| EXL3 (trellis, exllamav3) | expert 바이트/토큰 | EXL3 4비트 PPL 7.42·3비트 7.89 ([2차](https://github.com/resmp-dev/metal-marlin/blob/HEAD/docs/formats/mr_gptq.md), 모델 미상이라 주의); Qwen3.6 2.0 bpw 9.441 ([카드](https://huggingface.co/peasantsmith/Qwen3.6-35B-A3B-Escha-W2-EXL3-2.0bpw-MTP)); GLM-5.3 3.0 bpw가 BF16 대비 +9.3 % ([카드](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw)) | 위 행 참조 | CUDA 전용 커널이 이미 있음; CPU 커널은 없음 | 해당 | GPU expert만 | **조건부**(QTIP과 동일; 포맷 종속이라 후순위) |
| AQLM 2비트 (additive) | expert 바이트/토큰 | Llama2 2비트: 7B 6.93·13B 5.70·70B 3.94(+0.11); 파레토 최적 ≈2.5 bpw ([paper](https://arxiv.org/pdf/2401.06118.pdf)) | 70B는 +2.9 % | 코드북 LUT 디코드; AVX2에 코드북 상주형 커널이 없음 | 해당 | GPU expert만 | **조건부**(QTIP과 동일 사유) |
| QuIP# 2비트 (E8 lattice) | expert 바이트/토큰 | Llama2-7B 8.22·70B 3.91 ([인용](https://openreview.net/pdf?id=m6nBgFSMTL)); A6000에서 QuIP 대비 동 비트율 2× 처리량 ([paper](https://arxiv.org/pdf/2402.04396)) | QTIP보다 나쁨(7B +1.4) | E8 lattice 디코드; CPU 커널 없음 | 해당 | GPU expert만 | **조건부**(QTIP이 품질·속도 둘 다 위) |
| VPTQ 2비트 (VQ) | expert 바이트/토큰 | 2비트에서 SOTA 대비 L2 −0.01~0.34·L3 −4.41~7.34 PPL, 추론 처리량 1.6~1.8× ([paper](https://arxiv.org/pdf/2409.17066)) | 위 행 참조 | VQ LUT 디코드(CUDA); CPU 커널 없음 | 해당 | GPU expert만 | **조건부**(AQLM과 동일 사유) |
| HIGGS (비트 배분) | 비트 배분 최적화 (포맷 아님) | Llama2-7B ≈2비트 비교에서 QTIP 6.82가 최상 ([paper](https://export.arxiv.org/pdf/2411.17525)); 4비트 Llama-3.1-8B에서 GPTQ 대비 −0.015 ([poster](https://assets.underline.io/lecture/117218/poster_document/adc11000c91e7860a499e1381616b1d8.pdf)) | 배분 기법이라 단독 PPL 없음 | 디코드 커널을 새로 요구하지 않음(GPTQ형 커널 재사용) | 재양자화 시 | 재양자화 시 | **조건부**(재양자화 파이프라인이 생기면 공짜로 얹음) |
| MXFP4 weight-only 저장 (E2M1, sm_86은 디퀀트 실행) | expert 바이트/토큰 (~4.25 bpw 실효) | Blackwell 네이티브가 FP8 대비 ~2× ([paper](https://arxiv.org/pdf/2505.14669v4)); H100도 네이티브 없이 구동 ([기사](https://www.theregister.com/software/2025/08/10/openai-gpt-oss-llms-use-mxfp4-smaller-faster-cheaper/1460892)) | gpt-oss가 이 포맷으로 출하 (정확도 수치는 별도 미인용) | 16-entry LUT 디퀀트(AVX2 `vpshufb`형·GPU 모두 가능) | 변형이 이미 박스에 있음 | Q3_K(3.44 bpw)보다 큼 | **기각**(바이트가 Q3_K보다 크고, 디퀀트 비용만 듦) |
| W4A16 (GPTQ+Marlin) | FP16 대비 가중치 바이트 ÷4 | Marlin이 vLLM FP16 대비 최대 2.8×(bs16), A10 Llama2-7B ~3× ([paper](https://arXiv.org/pdf/2408.11743)); 배치 16~32까지 ~4× 이상적 | Llama2-7B 5.47→5.55 (+1.5 %) ([2차](https://github.com/totnormal/skills-3/blob/HEAD/AI-Research-SKILLs/10-optimization/gptq/SKILL.md)) | INT4 IMMA/GEMV 커널 | Q3_K가 이미 4비트 아래 | Q3_K가 이미 4비트 아래 | **기각**(우리 가중치는 이미 3.4 bpw라 줄이는 게 아니라 늘림) |
| AWQ 4비트 | 위와 동일 | HF FP16 대비 4090에서 3.9×·Orin에서 3.5× ([paper](http://arxiv.org/pdf/2306.00978v2)) | GPTQ 대비 동등 이상 (동일 출처) | AWQ형 GEMV 커널 | 위와 동일 | 위와 동일 | **기각**(W4A16과 동일 사유) |
| W8A8 (SmoothQuant) | M=1에서 없음(활성화는 KB) | 1.56×·메모리 1/2 ([paper](https://huggingface.co/papers/2211.10438)) | 무시 가능 | INT8 IMMA 경로 | 연산만 빨라짐 | 연산만 빨라짐 | **기각**(M=1 대역폭 항을 안 움직임; 검증 배치에서만 재검토) |
| QuaRot/SpinQuant/FlatQuant/PrefixQuant (회전 W4A4) | M=1에서 없음(활성화는 KB) | QuaRot Llama2-70B prefill 3.33×·디코드 메모리 3.89×·PPL ≤+0.47 ([paper](http://papers.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf)); FlatQuant 디코드 1.7×·오버헤드 0.26→0.07× ([paper](https://arxiv.org/pdf/2410.09426v4.pdf)); PrefixQuant 1.60~2.81× ([paper](https://huggingface.co/papers/2410.05265)) | W4A4에서 +0.5~1.0 PPL급 | Hadamard/affine 융합 + INT4 `mma` 경로 | 연산만 | 연산만 | **기각**(M=1 항을 안 움직임; sm_86 INT4는 되지만 쓸 이유가 없음) |
| QServe W4A8KV4 | M=1에서 없음 | TRT-LLM 대비 1.2~1.4×(8B)·2.4~3.5×(72B) ([repo](https://github.com/mit-han-lab/omniserve)) | W4A4보다 작음 (동일 출처) | W4A8 grouped GEMM 코디자인 | 연산만 | 연산만 | **기각**(위와 동일 사유) |
| Atom W4A4 | M=1에서 없음 | 서빙 처리량 FP16 대비 7.7×·INT8 대비 2.5× ([paper](https://arxiv.org/abs/2310.19102v3)) | W4A4 표 참조 | INT4 서빙 커널 | 연산만 | 연산만 | **기각**(위와 동일 사유) |
| MoE-Infinity식 expert 캐시 | 우리 형상에는 없음 (PCIe 전송이 경로에 없음) | V2-Lite TPOT 51/155/169 ms로 3.1~16.7× ([paper](http://arxiv.org/pdf/2401.14361v3)); 적중률 46 %(ORACLE −10 %p, 베이스라인 32 %) ([paper](http://arxiv.org/pdf/2401.14361v1)) | 없음 | 캐시·프리페치 스케줄러 | 해당 없음 | 해당 없음 | **기각**(우리 expert는 이미 DDR4에 상주하며 그 자리에서 연산 — 옮길 전송이 없음) |
| ProMoE/SP-MoE식 선행 프리페치 | 위와 동일 | ProMoE 디코드 평균 2.07×(최대 5.02×) ([paper](http://arxiv.org/pdf/2410.22134v1)); SP-MoE가 Mixtral-Offloading 대비 최대 3.5× ([paper](https://arxiv.org/pdf/2510.10302.pdf)) | 없음 | 중간 활성화 기반 예측기 | 해당 없음 | NVMe engram에만 | **기각**(DRAM 상주 expert에는 숨길 전송 지연이 없음; engram NVMe 읽기에만 재검토) |
| Pre-gated MoE | 전송 회피 (우리 경로에 없음) | 처리량 평균 1.5×·지연 최대 1.9×·GPU 메모리 1/4.2 ([paper](https://arxiv.baiduqq.workers.dev/pdf/2308.12066)) | 재학습 후 동등 이상 | **모델 재학습**(pre-gate) | 불가 | 불가 | **기각**(재학습이 필요하고, 옮길 전송도 없음) |
| Fiddler식 CPU 실행 | 호스트 expert를 그 자리에서 연산 (이미 우리 설계) | 단일 배치 1.26×·긴 prefill 1.30×·빔서치 11.57× ([paper](https://arxiv.org/pdf/2402.07033)) | 없음 | CPU expert 커널(보유) | 해당 | 해당 | **채택 후보**(단, 그 AVX512_BF16 커널은 이식 불가 — 아래 Q4·Q5) |
| KTransformers식 AMX CPU expert | 호스트 3.336 GB의 연산 | AMX INT4 디코드 12.4·INT8 6.0·BF16 3.2 tok/s (MoE-only, [재현](https://github.com/sylvanus4/ktransformers-moe-offload-bench)); V3+4090 종단 418 tok/s ([doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)) | INT4 양자화 수준 | AMX (Xeon 전용) | 불가 | 불가 | **기각**(Threadripper에 AMX가 없음 — 구조(상주 연산)만 차용) |
| batch 2 ≥ batch 1 수정 | 검증 패스의 전제조건 (직접 절감 아님) | 구조 분석 + [ik PR #1377](https://github.com/ikawrakow/ik_llama.cpp/pull/1377)의 직렬화 실례 | 없음 | expert별 버킷 디스패치 (위 grouped GEMM 행과 동일) | 2단계 게이트에 이미 있음 | DSpark의 전제조건 | **채택 후보** |
| Dense 재양자화 (Q4_K/Q6_K → IQ4_K급) | dense 7.658 GB의 ~15 % | IQ4_K 오차가 Q4_K_S 대비 −40 % ([ik #8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)) | 측정 필요 | 위 IQ 행과 동일 | 작음 | −2 ms급 (추정) | **측정 필요**(dense PPL 민감도 한 번이 판정을 가름) |

상위 3개 레버의 산술은 마지막 절에 있다.

## Q1 — 양자화 도메인에서 전파하기

결론부터: M=1 디코드에서 활성화 양자화는 대역폭 레버가 아니다. 활성화 벡터는
수 KB고 가중치는 V4.1에서 토큰당 11.7 GB다. W8A8·W4A4·회전 기법이 재는
"고속화"는 전부 연산·지연 쪽이고, 우리 루프라인의 ~5 ms 비 대역폭 여백 안에서만
논다. 이 절의 기법들은 그래서 전부 기각이고, 살아남는 것은 "W4A16이 아니라 더 낮은
bpw"라는 Q2의 질문뿐이다.

### Weight-only (W4A16 / W3A16): 기준선이지만 우리보다 굵다

GPTQ 4비트(g=128)는 Llama2-7B에서 5.47→5.55 PPL(+1.5 %) 수준으로 사실상 무손실이고
([2차 정리](https://github.com/totnormal/skills-3/blob/HEAD/AI-Research-SKILLs/10-optimization/gptq/SKILL.md),
1차는 [GPTQ 논문](https://arxiv.org/abs/2210.17323)), Marlin 커널은 vLLM FP16 대비
배치 16에서 최대 2.8×, A10·Llama2-7B 생성에서 ~3×, 배치 16~32까지 이상적인 4×에
가깝게 나온다 ([Marlin](https://arXiv.org/pdf/2408.11743)). Sparse-MARLIN은 그 위에
1.2×를 얹는다 (동일 출처). AWQ는 HF FP16 대비 4090에서 3.9×·Orin에서 3.5×
([AWQ](http://arxiv.org/pdf/2306.00978v2)).

움직이는 항: FP16 가중치 바이트 ÷4. 우리 가중치는 이미 Q3_K ≈3.44 bpw·Q4_K·Q6_K라서
이 기법들은 바이트를 줄이는 게 아니라 **늘린다**(4비트 > 3.44비트). 기각 사유가 이것
한 줄이다. Marlin이 우리에게 주는 것은 커널이 아니라 교훈 하나다: 배치 16~32까지
이상적 고속화를 유지하는 GEMV/GEMM 설계가 검증 패스(Q3)의 목표 형태다.

### W8A8: M=1에서는 연산 레버

SmoothQuant는 W8A8로 최대 1.56×·메모리 1/2 ([논문](https://huggingface.co/papers/2211.10438)).
움직이는 것은 연산 처리량과 (배치·컨텍스트가 클 때의) 활성화·KV 메모리다.
M=1 디코드의 토큰당 바이트에는 손도 안 댄다. 기각. 단 Q3의 검증 배치에서는
M=k GEMM이 연산 바운드에 가까워지므로, 그때 INT8 IMMA 경로는 재검토 대상이다 —
지금이 아니다.

### W4A4와 회전 기법: 어디가 빨라지는지 정확히

- QuaRot: Llama2-70B에서 prefill 3.33×(배치 64×2048), 디코드 메모리 3.89×,
  WikiText-2 PPL 손실 ≤0.47 ([NeurIPS](http://papers.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf)).
  7B에서는 prefill 2.16×·메모리 3.39×·PPL ≤+0.63 ([arXiv](https://arxiv.org/pdf/2404.00456v1.pdf)).
- SpinQuant(학습된 회전): W4A4KV4 Llama2-7B zero-shot이 FP 대비 −2.9점,
  LLM-QAT 대비 +19.1점·SmoothQuant 대비 +25.0점 ([논문](http://arxiv.org/pdf/2405.16406v3)).
- FlatQuant(학습된 affine): W4A4 Llama-3-70B에서 정확도 하락 ≤1 %,
  SpinQuant 대비 +7.5 %, prefill 2.3×·디코드 1.7×, 변환 오버헤드 0.26→0.07×
  ([논문](https://arxiv.org/pdf/2410.09426v4.pdf)).
- PrefixQuant: W4A4KV4 Llama-3-8B PPL 7.43으로 QuaRot 대비 −0.98·정확도 +5.98점,
  FP16 대비 1.60~2.81×·QuaRot 대비 1.2~1.3× ([논문](https://huggingface.co/papers/2410.05265)).
- Atom: 서빙 처리량 FP16 대비 7.7×·INT8 대비 2.5× (동일 지연 목표,
  [논문](https://arxiv.org/abs/2310.19102v3)).
- QServe W4A8KV4: TensorRT-LLM 대비 Llama-3-8B 1.2~1.4×·Qwen1.5-72B 2.4~3.5×
  ([repo](https://github.com/mit-han-lab/omniserve)). 참고로 QServe 논문은
  QuaRot의 per-group→per-channel 전환 비용을 PPL +0.2로 잰다
  ([논문](http://arxiv.org/pdf/2405.04532v2)).

이 숫자들이 크다고 우리 항이 움직이지 않는다. W4A4의 디코드 이득은 (a) 가중치
4비트 — 우리는 이미 3.44비트, (b) 활성화 4비트 — 토큰당 수 KB, (c) INT4 연산기
가동률 — M=1 GEMV는 연산기가 아니라 대역폭이 논다. FlatQuant의 "디코드 1.7×"도
FP16 베이스라인 대비지 Q3_K 베이스라인 대비가 아니다. 전부 기각.

### sm_86에서 무엇이 실행되는가

INT8 IMMA는 된다(전제와 동일). INT4(`mma.*.s4`)도 된다: Ampere의 INT4 `mma`는
`IMMA.16832.S4.S4`로 컴파일되는 실재 명령이고, PTX의 `.s4`/`.u4` 타입으로
오늘 쓸 수 있다. 근거는 Hopper 해부 논문의 아키텍처 비교표
([arXiv](https://arxiv.org/pdf/2501.12084v1)) — Ampere·Ada는 네이티브, Hopper는
`IMAD` 에뮬레이션으로 떨어진다. 주의점 하나: Blackwell sm_120에서는 `.s4`가
컴파일은 되지만 에뮬레이션으로 ~4.6× 느리다는 커뮤니티 실측이 있다
([2차](https://github.com/vanities/matador-miner/blob/HEAD/research/matmul-v4/FINDINGS-nvidia-docs-sweep-2026-08-02.md)).
그래서 sm_86용 INT4 경로는 이 박스 전용 분기로만 의미가 있고, 이식성은 없다.
그리고 위에서 봤듯 쓸 이유가 없다 — M=1 항을 안 움직인다.

Q1이 Q3에 넘기는 것: 활성화 양자화가 대역폭 레버가 되는 지점은 정확히
"검증 배치 M=k가 연산 바운드에 들어가는 지점"이다. 그 지점에 가면 INT8 IMMA
검증 커널을 재검토한다.

## Q2 — 가중치당 더 적은 바이트

3비트 아래에서 품질을 붙잡는 것은 지금 trellis·VQ· lattice 삼파전이고,
2비트 Llama2-7B 무파인튜닝 기준 순위는 QTIP 6.89 < AQLM 6.93 < QuIP# 8.22다
(아래 출처). 그러나 우리에게 순위를 가르는 것은 PPL이 아니라 **AVX2 호스트
티어에서 디코드되는가**다. 3.336 GB/토큰을 매 토큰 디코드하는 CPU 커널이
없으면 그 포맷은 GPU 상주 7블록에만 쓸 수 있고, 레버는 1/6 토막이 난다.

### 포맷별 실측

- **QuIP#** (Hadamard + E8 lattice): 2비트 Llama2-7B 8.22~8.23, 70B 3.91
  ([ICQuant 인용](https://openreview.net/pdf?id=m6nBgFSMTL),
  [CALDERA 인용](https://github.com/maximiliankhan/openbeast/blob/HEAD/research/lowrank/prior-art/survey-lowrank-compression.md)).
  A6000에서 QuIP 대비 동 비트율 처리량 ~2× ([논문](https://arxiv.org/pdf/2402.04396)).
  블록당 연산: E8 lattice 디코드(8차원 격자 양자화) + Hadamard 역변환.
- **QTIP** (trellis + incoherence): 2비트 Llama2-7B 6.89 무파인튜닝
  ([논문](https://arxiv.org/pdf/2406.11235v4)), 파인튜닝 시 2b 5.86·3b 5.28·4b 5.17
  ([2차](https://github.com/azimml/trellis-webgpu/blob/HEAD/notes/qtip-exl3-spec.md)).
  Llama2-70B 2비트 C4 5.48로 QuIP# 5.71·AQLM 5.62를 이긴다
  ([2차](https://www.alphaxiv.org/abs/2406.11235v4)). 추론 속도는 QuIP#과 동등,
  피크 대역폭의 80 % 이상 ([2차](https://github.com/cksac/turboquant-model/blob/HEAD/research/recent-quantization-papers.md)).
  256차원 trellis는 VQ(d≤8) 대비 32× 차원이다 (논문). 블록당 연산: 비트 윈도우의
  의사난수 가우시안 해시로 코드워드를 그때그때 합성 — 저장된 코드북이 없고,
  상태 기계라 순차 디코드 성격이다.
- **EXL3** (exllamav3, QTIP 변형 trellis): 2~8 bpw, LDLQ 분해 + trellis 인코딩 +
  전용 CUDA GEMM/GEMV 커널 ([2차](https://deepwiki.com/turboderp-org/exllamav3)).
  PPL 실측은 HF 모델 카드들에 흩어져 있다: EXL3 4비트 7.42·3비트 7.89
  ([2차](https://github.com/resmp-dev/metal-marlin/blob/HEAD/docs/formats/mr_gptq.md),
  모델명 미상이라 약하게 인용), Qwen3.6 2.0 bpw 9.441(동일 프로토콜 비교,
  [카드](https://huggingface.co/peasantsmith/Qwen3.6-35B-A3B-Escha-W2-EXL3-2.0bpw-MTP)),
  GLM-5.3 3.0 bpw가 BF16 대비 PPL +9.3 %·순방향 KL 0.15251
  ([카드](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw)).
- **AQLM** (additive): 2비트 Llama2 7B 6.93·13B 5.70·70B 3.94(+0.11),
  파레토 최적 ≈2.5 bpw ([논문](https://arxiv.org/pdf/2401.06118.pdf)).
  블록당 연산: 코드북들의 가산 결합 — 가중치마다 여러 코드북 LUT 조회.
- **VPTQ** (VQ): 2비트에서 SOTA 대비 Llama2 −0.01~0.34·Mistral −0.38~0.68·
  Llama3 −4.41~7.34 PPL, 추론 처리량 1.6~1.8×, 양자화 시간 10.4~18.6 %
  ([논문](https://arxiv.org/pdf/2409.17066)). 블록당 연산: 벡터 코드북 LUT.
- **GLVQ** (2025 이후): 2비트 Llama2-70B 3.36으로 QTIP 3.78·QuIP# 3.91을
  큰 폭으로 이긴다 ([논문](http://openreview.net/pdf?id=Ynwl0V1YH0)).
  학습된 격자 VQ — 디코드 커널 실측은 논문에 없다.
- **HIGGS** (선형 정리 기반 비트 배분): Llama2-7B ≈2비트 비교에서 QTIP 6.82가
  최상이고 HIGGS는 배분 기법이다 ([논문](https://export.arxiv.org/pdf/2411.17525)).
  4비트 Llama-3.1-8B에서 HIGGS(p=2) 5.908 대 GPTQ 5.923 — 차이는 −0.015
  ([포스터](https://assets.underline.io/lecture/117218/poster_document/adc11000c91e7860a499e1381616b1d8.pdf)).
  HIGGS는 디코드 포맷이 아니라 "어디에 비트를 쓸지" 정하는 방법이라
  커널을 새로 요구하지 않는다. 재양자화 파이프라인이 생기면 공짜로 얹는다.
- **ik `IQ*_K`** (우리와 같은 GGUF 세계): IQ2_KS 2.1875·IQ2_K 2.375·IQ3_K 3.4375·
  IQ4_KS 4.25·IQ4_K 4.5·IQ5_KS 5.25·IQ5_K 5.5·IQ6_K 6.5 bpw, LUT는 2×N entry
  ([ik #8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)).
  Ryzen 7950X AVX2에서 8B IQ2_XS pp512이 46.45(llama.cpp)→125.46(ik)→194.64(iqk)
  t/s, IQ3_K는 pp 6.45×·tg 2.37×. 토큰 생성은 메모리 바운드라 속도가 bpw에만
  비례하고, IQ4_KS는 tg에서 Q4_0보다 빠르다 (동일 출처). 오차는 IQ4_K가 Q4_0의
  1/2.7·Q4_K_S의 0.6배, IQ5_K 1.4 %로 Q5_0의 1/2.1 (동일 출처).

### CPU 커널 판정 — Q2에서 가장 결정적인 부분

`vpmaddubsw`→`vpmaddwd` 사슬과 어울리는 디코드는 "블록당 한 번 풀고 내적은
정수로 도는" 것뿐이다. 포맷을 셋으로 나눈다:

1. **AVX2 가능 (실증됨): `IQ*_K`.** LUT가 2×4~2×64 entry라 레지스터·L1에
   상주하고, 디코드 뒤 내적은 기존 정수 사슬 그대로다. 위 AVX2 실측이 증거다.
   우리 `q3k-cpu`의 다음 포맷은 이것이다.
2. **AVX2 가능 (미실증, 구조상): MXFP4 weight-only.** E2M1 값은 16-entry LUT
   (`vpshufb`형) 한 번에 풀리고, E8M0 스케일은 2의 거듭제곱이라 지수 덧셈이다.
   그러나 실효 ~4.25 bpw로 Q3_K(3.44 bpw)보다 바이트가 크다. 가능하지만 쓸 이유가
   없어 기각 — 박스의 MXFP4 변형이 있어도 마찬가지다. sm_86에 FP4 하드웨어가
   없다는 전제는 맞고, 디퀀트 실행이면 그 전제와 무관하게 바이트 비교에서 진다.
3. **AVX2 불가 (GPU 전용): QTIP·EXL3·AQLM·VPTQ·QuIP#.** QTIP/EXL3의 trellis는
   상태 기계 순차 디코드(GPU에서 피크 대역폭의 80 %로 도는 것은 수천 스레드의
   병렬성이지 디코드 자체가 싼 게 아니다), AQLM/VPTQ는 가중치마다 큰 코드북
   다중 조회(캐시 미스·직렬 의존성), QuIP# E8 lattice는 8차원 격자 연산이다.
   셋 다 공개된 CPU 커널이 없고, 매 토큰 3.3 GB를 AVX2로 풀 구조가 아니다.
   GPU 상주 7블록에만 조건부로 쓴다 — 그 항은 0.708 GB라 이득 상한이 ~1.2 ms다.

그래서 Q2의 답은 둘이다: 호스트 티어는 `IQ*_K`로 간다(채택 후보). 그 아래
bpw가 필요하면 GPU 상주분만 trellis/VQ로 간다(조건부). "2비트면 바이트가
4할 준다"는 말은 CPU 디코드가 있을 때만 참이다.

## Q3 — 한 번 읽은 가중치를 여러 토큰에 나누기

이 절이 레버 1위다. 한 패스에 k 토큰을 내면 루프라인이 k로 나뉜다.
V4.1은 draft 헤드를 **체크포인트 안에** 갖고 있고, GGUF 변환이 그것을
떨군 것이 현재 상태다.

### V4.1-Flash는 MTP/DSpark 헤드를 싣고 나온다

- DeepSeek-V3의 MTP: 2번째 토큰 수락률 85~90 %, 1.8× TPS
  ([V3 리포트](https://arxiv.org/pdf/2412.19437v1)). 1차 출처의 MoE 실측이다.
- V4.1-Flash의 DSpark: 논문 "DSpark: Confidence-Scheduled Speculative Decoding
  with Semi-Autoregressive Generation"(DeepSeek+북경대). Qwen3-4B에서 EAGLE-3
  대비 평균 수락 길이 +30.9 %
  ([보도](http://www.techtimes.com/articles/319236/20260628/deepseek-releases-dspark-speculative-decoding-makes-v4-85-percent-faster.htm)).
  체크포인트 안에 20B draft 헤드가 들어 있고(284B 본체 + 20B, 304B)
  ([실측기](https://medium.com/@ukaszrewicz/running-deepseek-v4-flash-0731-on-a-single-dgx-spark-18e80ae15d3c)),
  `num_nextn_predict_layers = 3`으로 번들된다 ([HF 카드](https://viralpique.com/dealignai-deepseek-v4-1-flash-uncensored-fp8-·-hugging-face/)).
  vLLM은 `{"method":"dspark","num_speculative_tokens":7}` 한 플래그로 켠다
  ([보도](https://www.marktechpost.com/2026/07/31/deepseek-upgrades-deepseek-v4-flash-0731-with-major-agentic-and-coding-gains/)).
- 필드 실측(서드파티, 동일 요청·동일 카드 비교):
  CMP 170HX k=5에서 위치별 수락 0.833/0.637/0.462/0.323/0.228,
  평균 수락 3.02~3.48, 디코드 50.8→98.1 tok/s (**1.93×**)
  ([기록](https://github.com/chinaboy0618/deepseek-v4-cmp170hx/blob/HEAD/DSPARK_PAPER.md)).
  DGX Spark k=2에서 ~9.8 tok/s·수락 52 %·평균 2.04
  ([기록](https://medium.com/@ukaszrewicz/running-deepseek-v4-flash-0731-on-a-single-dgx-spark-18e80ae15d3c)),
  2× Spark에서 62.48 tok/s·수락 0.673·수락/드래프트 3.36
  ([기록](https://github.com/tonyd2wild/deepseek-v4-flash-dspark-60-tok-s-900k-ctx-2x-dgx-spark/blob/HEAD/README.md)).
- 우리 GGUF에 `mtp`/`nextn` 키가 없는 것은 변환이 떨군 것이다. 선례가 있다:
  DeepSeek-V3 GGUF도 "no MTP support"로 올라왔다
  ([카드](https://huggingface.co/bullerwins/DeepSeek-V3-GGUF)).
  llama.cpp는 2026-05 본선 MTP 지원(PR #22673, `--spec-type draft-mtp`)을
  합쳤고 1.4~2.2×를 잰다 — Qwen3.6-27B 38→65 tok/s(1.71×)
  ([2차](https://github.com/jamesburton/dotllm/issues/253)).

답: 업스트림 V4.1-Flash는 draft 헤드를 싣고 나온다. 우리 파일에는 없다.
DSpark를 쓰려면 nextn 텐서를 살린 재변환(또는 draft 가중치 별도 확보)이
선행 조건이다. SGLang 쪽은 `SGLANG_RAGGED_VERIFY_MODE=cap-accept` +
프로파일된 SPS 테이블이 실전 조건으로 붙는다 (위 HF 카드).

### MoE에서의 EAGLE·Medusa 실측

- EAGLE-3: dense에서 3.0~6.5×, EAGLE-2 대비 +20~40 %
  ([NeurIPS](https://papers.nips.cc/paper_files/paper/2025/file/c7b5a35ea98b62512a869c19ea7b03cb-Paper-Conference.pdf)).
  TensorRT-LLM 실전형은 2~3×
  ([2차](https://github.com/ozp3/speculative-decoding-vs-mtp)),
  vLLM 서버 로그는 평균 수락 길이 2.77·평균 수락률 58.9 %
  ([2차](https://github.com/czyszka/nanoserve-mini/blob/HEAD/docs/writeups/w1/t6-eagle3-speculative-decoding.md)).
  MoE 대형 실전형(SGLang V4 배포)에서는 수락 길이 ~2.5로 dense 천장보다 낮다
  ([2차](https://github.com/aeonmindai/arc/blob/HEAD/research/moe_speculative_decoding.md)).
- Medusa: Zephyr-7B AlpacaEval에서 34.21→99.50 tok/s, 수락률 3.08, 2.91×
  ([논문](https://arxiv.org/pdf/2401.10774)). 하드웨어가 좋아질수록 이득이
  커진다: 1.48→2.42× ([서베이](https://arxiv.org/pdf/2401.07851)).
  토큰 수락률은 ~0.6으로 EAGLE(~0.8)보다 낮다
  ([2차](https://github.com/krrish777/ideenkasten/blob/HEAD/02%20-%20Deep%20Dives/Harness-Engineering-Internals/Harness-Internals-Speculative-Decoding.md)).
- 주의(온도): MTP류 이득은 greedy·저온도에서 잰 것이다. 프로덕션 샘플링
  온도에서는 MTP가 중립(−1.6 %)으로 수렴한 실전형 기록이 있다
  ([2차](https://github.com/pestopoppa/epyc-root/blob/HEAD/wiki/speculative-decoding.md)).
  수락률은 온도·태스크 의존이라 우리 서빙 프로파일에서 재야 한다.

MoE 한정으로 1차 출처 수락률·고속화를 둘 다 갖춘 것은 V3 MTP(85~90 %·1.8×)와
DSpark 필드 기록들(위)뿐이다. EAGLE-3의 MoE 대형 실측은 2차뿐이라 조건부로 둔다.

### 구조 문제: k토큰 검증이 싸려면 expert별 버킷이 필요하다

k개 토큰이 서로 다른 expert로 라우팅되면, 토큰별로 expert를 도는 순진한
구현은 expert를 k번 읽는다 — 1번이 아니라 k번이다. 잘하는 엔진들은 전부
같은 구조로 푼다: **토큰을 expert별로 정렬한 뒤 grouped GEMM 한 번**이다.

- 원형은 MegaBlocks의 block-sparse: 패딩 대비 학습 종단 1.38/2.0/4.35×,
  dense 대비 1.8~2.4× ([논문](https://arxiv.org/pdf/2211.15841)).
  학습 수치라 디코드에 직접 못 쓰지만, 구조(버킷+grouped)가 원형이다.
- vLLM `fused_moe`의 실제 자료구조: `expert_num_tokens`에서
  `expert_offsets`/`problem_sizes`를 만들고, `shuffle_rows`로 순열/역순열한 뒤
  CUTLASS grouped GEMM을 한 번 호출한다
  ([코드 리딩 2차](https://github.com/jieen1/blackforge/blob/HEAD/notes/2026-07-23-vllm-fused-moe-cutlass-flashinfer-analysis.md)).
  즉 `permute → grouped GEMM → unpermute` 세 단계가 고정이다.
- 순열 비용 실측: Megatron Core 측정에서 라우팅+순열은 최적화 뒤에도 층
  실행 시간의 ~9 %다. 같은 측정에서 GEMM 비중은 dense 405B 70 % 대
  MoE(V3) 50 % 미만 ([리포트](https://arxiv.org/pdf/2603.07685v1)).
  순열은 공짜가 아니지만 k분할 앞에서는 한 자릿수 비용이다.
- SGLang/DeepEP·TRT-LLM도 같은 집합(grouped GEMM + 토큰 버킷)이다.
  multi-GPU all-to-all이 추론 시간의 60 %를 넘는다는 측정
  ([논문](http://arXiv.org/pdf/2410.17043))이 있지만, 우리 박스는 EP 분산이
  아니라 층 분할+호스트 티어라 이 항은 해당 없다.

우리 엔진이 갖춰야 할 구조: 라우터 출력 → expert별 토큰 인덱스 버킷(오프셋
테이블) → 버킷 단위 GEMM → 역순열 가중합. CPU 티어도 동일하다: M=k GEMM으로
expert를 한 번만 읽는다.

### ik가 batch 2에서 batch 1보다 느린 구조적 이유

측정 힌트(dense는 정상, 그래프·fmoe 아님)가 가리키는 것은 MoE 디스패치
자체다. batch 2의 두 토큰이 서로소 expert 집합으로 라우팅되면(384개 중
6개씩이라 거의 항상), expert별 디스패치 엔진은 expert 바이트를 ~2배 읽으면서
토큰은 2개 낸다 — 토큰당 바이트가 안 줄고, expert op마다 배리어·런치
오버헤드만 곱절이 된다. dense 층은 가중치를 한 번 읽고 M=2로 재사용하니
정상이다. 즉 "느려지는 것"이 아니라 "빨라질 이유가 없는 구조에서 오버헤드만
더 붙는 것"이다.

같은 계열의 실례가 ik 트리에 있다: PR #1377의 타일 선택 버그는 배치를
병렬 한 번이 아니라 토큰별 순차 런치(`ne12`번 커널 런치, 개당 ~10 µs)로
떨어뜨렸다 ([PR](https://github.com/ikawrakow/ik_llama.cpp/pull/1377)).
우리가 재야 할 것은 이 버그가 아니라 구조다: 안 느린 엔진들은 위 grouped
디스패치로 expert 읽기를 토큰 수와 무관하게 한 번으로 묶는다. 2단계 게이트의
"batch 2가 batch 1보다 느리지 않음"은 그래서 커널이 아니라 디스패치 구조의
게이트다.

### 호스트 티어 특례: 3.336 GB도 k로 나뉜다

33블록 expert가 DDR4에 있어도 나눗셈은 성립한다 — 조건 하나: CPU expert
연산이 M=k GEMM으로 expert를 한 번만 읽을 것. 우리 `q3k-cpu`는 이미 M≤8로
"행당 한 번 디코드, 열마다 maddubs" 구조다(인-트리
`crates/q3k-cpu/RESULTS-r4-lead.md`: big M=8이 ggml의 0.87~0.89배로 돌긴
돈다). 즉 모양은 맞고 남은 것은 M>1 튜닝이지 구조 변경이 아니다.
깨지는 경우는 디스패치를 토큰별로 돌리는 경우뿐이다 — 그때는 3.336 GB가
k배가 된다. Q6의 "1-2 전 결정"이 바로 이것이다.

## Q4 — 차가운 가중치를 안 읽기

이 절의 기법들은 전부 "VRAM에 안 들어가는 expert를 어떻게 덜 옮기나"다.
우리 형상은 다르다: expert가 이미 DDR4 256 GB에 상주하고 **그 자리에서
연산**된다. PCIe 전송이 경로에 없으므로, 전송을 숨기는 기법은 전부 기각이다.
살아남는 것은 "상주 연산" 아키텍처 자체와, 라우팅 예측 가능성이라는 지식이다.

### 시스템별 실측 (CPU 오프로드 티어 기준)

- **MoE-Infinity** (요청 단위 추적 + 희소성 인지 캐시): V2-Lite TPOT
  51/155/169 ms로 기존 오프로드 대비 3.1~16.7× 지연 감소
  ([논문](http://arxiv.org/pdf/2401.14361v3), A5000·PCIe4 단일 GPU).
  적중률은 Switch-128·15 GB 캐시(3072개 중 535개)에서 46 %,
  ORACLE보다 10 %p 낮고 최선 베이스라인 32 %보다 높다
  ([논문](http://arxiv.org/pdf/2401.14361v1)). 전제: GPU 캐시 미스가 PCIe
  전송을 탄다. 우리는 전송이 없으니 기각 — 단 V2-Lite 수치는 디딤돌 비교용으로
  보관한다.
- **Pre-gated MoE** (pre-gate 함수 학습 + 알고리즘·시스템 코디자인):
  처리량 평균 1.5×(42 tok/s, OnDemand 대비 1.6×·Prefetch 대비 52×),
  지연 최대 1.9× 감소, 피크 GPU 메모리 1/4.2·GPU-only의 23 %
  ([논문](https://arxiv.baiduqq.workers.dev/pdf/2308.12066)). 전제: 모델을
  고쳐(pre-gate) 재학습한다. 재학습 불가 + 전송 없음이라 기각.
- **Fiddler** (CPU·GPU 오케스트레이션): 단일 배치 1.26×·긴 prefill 1.30×·
  빔서치 11.57× ([ICLR25](https://arxiv.org/pdf/2402.07033)). 프리프린트의
  8.2×/10.1×는 베이스라인이 다르다 (동일 논문 v1). 핵심 통찰이 우리 설계와
  같다: 작은 배치에서는 expert를 CPU→GPU로 복사하는 것보다 CPU에서
  실행하는 게 싸다. 단 그 CPU 커널은 AVX512_BF16 전용이라 이식 불가 —
  구조(상주 연산)는 채택 후보, 커널은 기각이다.
- **KTransformers** (AMX CPU expert): AMX 커널이 Xeon4에서 BF16 21 TFLOPS·
  INT8 35 TOPS(PyTorch AMX 대비 ~4×), V3+4090 종단 418 tok/s
  ([doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)).
  V0.3 듀얼 소켓 8-expert에서 prefill 255.26·디코드 ~12 tok/s
  ([2차](https://github.com/dirty13itch/athanor/blob/HEAD/docs/research/2026-02-25-ram-utilization-strategies.md)),
  독립 재현에서 MoE-only 디코드 AMX INT4 12.4·INT8 6.0·BF16 3.2 tok/s
  ([재현](https://github.com/sylvanus4/ktransformers-moe-offload-bench)).
  전제: AMX(Xeon Sapphire Rapids+). Threadripper에 없으니 수치 이전 불가,
  기각 — 구조(상주 연산 + NUMA 인지 배치 + 태스크 스틸링)만 차용한다.
- **Mixtral-offloading** (LRU + 계층별 예측 적재): LRU 캐시(층당 16 expert)+
  미스 시 동기 폴백이 기본형이다 ([2차](https://github.com/caiovicentino/polarengine-vllm/blob/HEAD/docs/expert_offloading_design.md)).
  FineMoE 비교에서 LRU는 LFU보다도 나쁘다
  ([논문](https://arxiv.org/pdf/2502.05370)). SP-MoE가 이것 대비 DeepSeek-Lite·
  HumanEval·A100에서 최대 3.5× (아래). 전송 전제라 기각.
- **ProMoE** (중간 활성화 기반 선행 캐시): prefill 평균 2.20×(최대 3.21×)·
  디코드 평균 2.07×(최대 5.02×) ([논문](http://arxiv.org/pdf/2410.22134v1)).
  캐시 50 %에서도 미스 적재가 추론 시간의 60 %를 먹는다는 측정이 이 계열의
  출발점이다 (동일 논문). 전송 전제라 기각.
- **그 이후**: SP-MoE 1.07~3.5× ([논문](https://arxiv.org/pdf/2510.10302.pdf)),
  SPICE가 V2-Lite에서 TPOT 최대 3.12× ([논문](https://arxiv.org/abs/2608.21240)),
  ExpertFlow가 예측 정확도 +30 %·대기 지연 최대 99.9 % 제거
  ([논문](https://arxiv.org/pdf/2510.26730)),
  SpecPrefetch 1.14×(프리페치 기여 +16 %, 3.76 tok/s — 적재가 크리티컬
  패스일 때만 유효하다고 논문이 못박는다,
  [논문](https://arxiv.org/pdf/2607.24787)),
  PROBE가 SGLang 대비 최대 1.32× ([논문](https://arxiv.org/pdf/2602.00509)).
  전부 "옮기는 비용을 숨긴다" 계열이라 우리 형상에는 해당 없다.

### 라우팅은 예측 가능한가

예측 가능하다는 쪽에 실측이 셋이다: (1) 한 층 앞 expert를 선행 적재하면
정확한 expert가 실려 있을 확률 ~80 % 이상
([강의노트 2차](https://aipapersacademy.com/moe-offloading/)),
(2) ExpertFlow의 교차 층 예측(+30 %, 위), (3) pre-gate가 "이전 층 출력으로
다음 층 라우팅을 예측"한다는 전제 위에서 정확도를 유지한 Pre-gated MoE
자체. 그러나 우리에게는 예측할 전송이 없다 — 이 지식의 용도는 Q6의
engram NVMe 프리페치뿐이다. 33블록 DDR4 expert에 선행 적재를 붙이는 것은
측정 대상도 아니다(숨길 지연이 없다).

### 무엇이 이전되고 무엇이 안 되는가

이전된다: 상주 연산 아키텍처(Fiddler·KTransformers의 결론), 라우팅 예측
가능성(engram용). 이전 안 된다: 캐시 적중률 숫자(전송이 없으니 분모가 없다),
AMX·AVX512 커널 수치(ISA가 없다), pre-gate(재학습 불가). 한 줄로: Q4는
우리 설계를 정당화하는 문헌이지, 얹을 기법 목록이 아니다.

## Q5 — 호스트 티어의 커널, VNNI 없는 AVX2

정수 내적 사슬은 풀렸으니(우리 `q3k-cpu`가 ggml의 1.11~1.40배,
인-트리 `crates/q3k-cpu/RESULTS-r4-lead.md`), 남은 질문은 주변부다:
블로킹·프리페치·스레드 수·페이지. 답은 "우리가 이미 잰 것과 남들이 하는 것이
같은 방향"이다.

### 실구현들이 하는 것

- **ik `iqk_mul_mat`** (`ggml/src/iqk/iqk_mul_mat.cpp`): 양자 타입별 특화
  커널 + R4 패밀리의 행/스레드 정렬(4의 배수 강제,
  [커밋](https://github.com/ikawrakow/ik_llama.cpp/commit/7d107ee10ec62316da7532537e7a570a6eb89020)).
  AVX2 실측은 Q2에 인용했다(pp 6.45×·tg 2.37×,
  [ik #8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)).
  우리가 안 하는 것: 타입별 GEMM 특화(Q4_0 블록-32 경로 같은 것)와
  R4 정렬. 우리 big M=8이 ggml의 0.87배에 머무는 이유 후보다.
- **llamafile tinyBLAS** (`ggml/src/ggml-cpu/llamafile/sgemm.cpp`):
  레지스터 타일 + K를 타일당 한 번만 도는 구조
  ([코드 리딩 2차](https://github.com/timtoole02/camelid/blob/HEAD/docs/perf-deep-dive/LLAMA_CPP_ARCHAEOLOGY.md)).
  효과 실측: llama 7B pp512에서 Q4_0 119.75 t/s 대 Q4_1 63.80·Q5_0 59.50 —
  tinyBLAS 경로가 ~2배 ([llama.cpp #6840](https://gitmemories.com/ggml-org/llama.cpp/issues/6840)).
  AVX2 마이크로커널 정석은 16개 YMM에 3×4 타일
  ([2차](https://github.com/chayprabs/edgelm/blob/HEAD/research/02-avx2-vnni-simd-optimization.md),
  1차는 justine.lol/matmul). 교훈: prefill/GEMM 쪽은 가중치 repack+타일이
  정답이고, 우리 디코드 GEMV와는 다른 커널이다. 검증 배치(M=k)가 커지면
  이쪽 커널이 필요하다.
- **KTransformers CPU 경로**: NUMA 인지 expert 배치 + 스레드 태스크 스틸링 +
  AMX 전용 커널 ([doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md)).
  AMX를 빼면 남는 것은 배치와 스틸링 — 우리 CCD 연속 행 범위 핀 고정과
  같은 집합이다.

### 블로킹·프리페치·스레드·페이지

- **32 MB L3 블로킹**: CCD당 expert 행 블록 연속 배정이 정석이고 우리 구현도
  이미 그렇다(인-트리 `src/main.rs` 주석). 남들이 더 하는 것은 없다.
- **프리페치 거리**: 우리 인-트리 실측에서 소프트웨어 프리페치는 −9~−11 %,
  2배 언롤은 −25 %(하드웨어 스트리머가 이미 덮는다,
  인-트리 `RESULTS-r4-lead.md`). 이 형상에서 SW 프리페치를 얹은 공개 실측을
  찾지 못했다. 손대지 않는다.
- **스레드 수**: 대역폭 바운드 커널이라 SMT 64가 32보다 나을 이유가 없고,
  실측도 그렇다 — ggml은 64스레드에서 97.4/79.2로 무너지고 우리 팔은
  202.7로 유지된다(인-트리 `RESULTS-r4-lead.md`). ik의 26.7코어도 같은
  방향이다(스펙). 스윕 한 번이면 닫힌다.
- **Non-temporal·huge page**: 이 규모 MoE CPU 추론에서 NT 로드·huge page가
  유의미하다는 공개 실측을 찾지 못했다. 우리 인-트리 실측에서는
  `MADV_HUGEPAGE`가 초 단위 실행에서 `AnonHugePages` 0 kB로 미발동
  (인-트리 `RESULTS-r4-lead.md`, plan.md도 "미확인"으로 기록). NT 로드는
  캐시 오염을 줄이지만 우리 작업 집합(토큰당 3.3 GB 스트림)은 어차피
  L3를 흘러넘치므로, 효과를 주장하려면 측정이 먼저다. 둘 다 측정 필요 —
  단 우선순위는 디스패치 구조(Q3)보다 한참 아래다.

Q5의 한 줄: 커널 내부는 끝났고, 남은 CPU 과제는 (a) M>1 GEMM 튜닝(검증용),
(b) expert 버킷 디스패치, (c) 스레드 수 스윕이다. (b)가 없으면 (a)는 무의미하다.

## Q6 — 우리 엔진이 갖춰야 할 것, 그리고 되돌릴 수 없는 것

문헌이 아니라 설계에 답한다. 1단계 라운드(1-1 로더/디퀀트 완료, 1-2 임베딩+
블록 0, 1-3 MoE+라우터, 1-4 단일 토큰 순전파, 1-5 GPU 오프로드+KV)는
[plan](../plan.md)에서 가져왔다.

1. **순전파 API는 토큰 배치를 1급으로.** — **1-2 전 결정.**
   k토큰 검증이 레버 1위(Q3)인데, M=1로 쓴 엔진은 나중에 못 바꾼다:
   라우터·디스패치·KV 인덱싱·가중합이 전부 "토큰 하나"를 가정하고 굳기
   때문이다. 1-2의 블록 0부터 `forward(tokens: &[T])` 모양으로 쓰고 M=1은
   그 특수 경우로 둔다. 되돌릴 수 없는 이유: 호출 규약이 굳은 뒤의 개조는
   전면 재작성이다.
2. **MoE 디스패치는 expert별 버킷 + grouped 연산으로.** — **1-3 전 결정.**
   토큰별 디스패치는 batch 2부터 손해(Q3)고 검증 패스를 k배로 만든다.
   버킷(오프셋 테이블) → 버킷 단위 GEMM → 역순열 가중합을 1-3의 첫 설계로
   못박는다. CPU·GPU 공통 구조다.
3. **KV 캐시 키는 (시퀀스, 위치) 2차원으로.** — **1-2 전 결정.**
   검증 패스는 후보 k개의 KV를 쓰고 버린다. 위치 차원이 append-only 단일
   시퀀스를 가정하면 검증용 분기·롤백을 나중에 못 얹는다. 1-5의 KV보다
   먼저 모양이 굳어야 한다.
4. **Draft 가중치 경로를 로더에 남겨 둔다.** — **개조 가능.**
   1-1은 끝났지만 텐서 맵에 `nextn`/`mtp` 네임스페이스를 추가하는 것은
   개조다. 단 GGUF 재변환(헤드 살리기)은 외부 의존성이라 일찍 착수한다.
5. **`quant` 크레이트가 IQ 디코드를 소유한다.** — **1-3 전 결정.**
   Q2의 결론(`IQ*_K`가 다음 포맷)은 CPU·GPU 커널이 같은 디코드 정의를
   쓰게 만든다. plan이 이미 "양자 타입은 `quant` 하나가 소유"로 못박았으니
   확인만 하면 된다. trellis/VQ를 GPU 상주분에 얹을지는 포맷별 게이트로
   뒤에 결정해도 된다(개조 가능).
6. **CPU/GPU 경계는 층 단위 + expert 단위 두 축으로.** — **1-3 전 결정.**
   지금 배치는 층 단위(33블록 호스트)지만, 검증·캐시·헤드 구조가 굳으면
   expert 단위 이동(뜨거운 expert를 GPU로)이 필요해진다. 경계 인터페이스가
   층 ID만 알면 나중에 못 바꾼다. (Q4 기법들을 안 쓴다는 결정과 모순되지
   않는다 — 경계는 남기되 전송 최적화는 안 한다.)
7. **서빙 프로파일(온도·태스크)별 수락률 측정을 게이트에 넣는다.** —
   **개조 가능.** Q3의 온도 주의가 근거다. 4단계(DSpark+스케줄러) 게이트에
   "greedy가 아니라 서빙 온도에서 수락률·tok/s"를 못박는다.

## 주장됐으나 측정 없음

- **GLVQ의 디코드 처리량.** PPL은 1차(2비트 70B 3.36)지만 커널·tok/s 실측이
  논문에 없다. 있으면 순위가 바뀔 수 있다.
- **DSpark 논문의 60~85 % 대 MTP-1.** 2차 보도 인용이라 1차 논문 확인 전에는
  표에 넣지 않았다. 대신 필드 기록(1.93×)을 썼다.
- **EAGLE-3의 MoE 대형 수락 길이 ~2.5.** 2차뿐이다. 1차가 나오면 조건부→채택
  후보로 바뀔 수 있다.
- **EXL3 4비트 7.42·3비트 7.89.** 모델명 미상의 2차라 약하게 인용했다.
  HF 카드 수치(위 Q2)는 모델명이 박혀 있어 더 믿는다.
- **NT 로드·huge page의 MoE CPU 효과.** 긍정 실측을 찾지 못했다.
  우리 인-트리 결과도 미발동·역효과다.
- **INT4 `mma`의 sm_86 실효 처리량.** 명령 존재는 1차(해부 논문)로 확인,
  처리량 수치는 이 박스에서 재야 한다 — 쓸 이유가 생기면(Q1).

## 이 표가 틀릴 수 있는 지점

roofline의 가정이 깨지면 판정이 바뀐다. 구체적으로:

1. **호스트 대역폭 147.7 대 215.6.** 4라운드에서 두 엔진이 317 MB 작업
   집합으로 147.7을 넘었다(L3 잔류 의심, 인-트리 `RESULTS-r4-lead.md`).
   215.6이 참이면 호스트 항은 22.6→15.5 ms, 토큰 39.92 ms 중 미설명이
   24 ms로 늘고 "대역폭이 병목"이라는 이 보고서의 대전제가 흔들린다.
   WKS-35 판별 실험(≥1.5 GB 작업 집합 또는 캐시 플러시)이 닫는다.
   이 한 줄이 바뀌면 Q1 기각들이 재심 대상이다.
2. **GPU 유효 대역폭 500 GB/s는 고른 값.** roofline이 스스로 밝히듯 잰 값이
   아니다. A6000·3090 스트림 실측 한 번이 자유 파라미터를 하나 줄인다.
3. **수락률은 greedy·저온도 수치.** 서빙 온도에서 DSpark/MTP가 중립으로
   수렴하면(Q3 온도 주의) 레버 1위의 산술이 무너진다. 서빙 프로파일 실측이
   선행 조건이다.
4. **V4.1 아키텍처 공개 정보와 우리 파싱의 불일치.** 공개 보도에는
   "20+20 인코더-디코더"라는 말이 있다
   ([예](https://www.neoteo.com/en/deepseek-v41-flash-bets-on-efficiency-at-massive-scale)).
   우리 GGUF 파싱(40블록 MoE 디코더 가정)이 맞다면 무시해도 되지만,
   파싱이 블록 종류를 확정했는지 한 번 확인한다. 인코더 층이 디코드 경로에
   없으면 토큰당 바이트가 준다 — 좋은 쪽으로 틀리는 경우다.
5. **KV 0.2 %는 top_k 512·MLA 576차원 전제.** 인덱서·차원이 바뀌면 다시 잰다.
   항이 커져도 1 %를 넘기 어려우니 판정은 안 바뀐다.
6. **Draft 헤드 확보 전제.** nextn 텐서 없는 GGUF로는 DSpark를 못 켠다.
   재변환이 막히면 레버 1위는 EAGLE-3(학습 헤드 필요)으로 밀리고, 그것도
   막히면 Q3 전체가 "구조만 잡고 대기"가 된다.

재야 하는 측정 목록(우선순위순): (a) WKS-35 호스트 대역폭 판별,
(b) A6000·3090 스트림 실측, (c) 서빙 온도 수락률(재변환 뒤),
(d) dense PPL 민감도(IQ4_K급 재양자화 전), (e) V4.1 블록 종류 확인.

## 상위 3개 레버 (V4.1 토큰당 절감, 산술 포함)

베이스라인 39.92 ms = 호스트 22.6 ms + GPU 16.7 ms + 미설명 ~0.6 ms.

1. **DSpark/MTP 검증 (필드 1.93× 적용): 39.92 → ~20.7 ms, −19.2 ms.**
   산술: 39.92 / 1.93 = 20.68. 근거 필드 기록(CMP 170HX k=5, 평균 수락
   3.02~3.48, 50.8→98.1 tok/s). 호스트·GPU 양 항이 함께 나뉜다.
   전제: nextn 재변환 + expert 버킷 디스패치 + 서빙 온도 수락률 유지.
2. **Expert `IQ2_K`급 재양자화 (3.44 → 2.375 bpw, −31 %): −7.4 ms.**
   산술: 호스트 22.6 × (1 − 2.375/3.4375) = 22.6 × 0.309 = 6.98;
   GPU expert 0.708 GB 항 ≈1.4 ms × 0.309 = 0.43. 합 −7.4 ms.
   근거: bpw·AVX2 실증([ik #8](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)).
   2비트대 PPL은 Q2 수치로 별도 게이트가 필요하다.
3. **Dense `IQ4_K`급 재양자화 (dense 바이트 −15 % 추정): −2.3 ms.**
   산술: 7.658 GB × 0.15 = 1.15 GB, 500 GB/s에서 2.30 ms.
   근거: IQ4_K 오차 −40 % (위 ik #8). −15 %는 추정이라 "측정 필요" 판정 —
   dense PPL 민감도 한 번이 가른다.

합이 39.92를 넘는다고 다 더하지 않는다: 레버 1은 다른 둘의 분모를 나눈다.
1+2 적용 시 (22.6×0.691 + 16.7) / 1.93 ≈ (15.6+16.7)/1.93 ≈ 16.7 ms,
즉 25.05 → ~60 tok/s가 이 보고서가 그리는 도착점이다. Dense까지 되면
~18 ms대 → 60 tok/s 중반이다.

## 이 스펙의 틀린 전제

**확정적으로 틀린 전제는 찾지 못했다.** 검증한 것: sm_86 INT8 IMMA·dp4a
존재(참), FP8/FP4 하드웨어 부재(참 — 단 INT4 `mma.*.s4`는 존재하고 PTX에서
오늘 쓸 수 있다, Q1), CPU ISA 상한 AVX2+FMA3+F16C+BMI2(참, plan의 박스
읽기와 일치), 호스트 147.7 잠정치(참, 단 WKS-35 미결), 토큰당 바이트 표의
내부 정합성(합산 일치), 22.6 ms = 39.92 ms의 57 %(56.6 %, 반올림 일치),
KV 23.6 MB·0.2 %(40×512×576×2 B = 23,592,960 B, 일치), V3 MTP 탑재(참),
GGUF MTP 키 부재(선례와 일치), q3k-cpu 1.1×·q3k-gemv 1.05×(인-트리 결과와
일치). 주의로 남기는 것은 "틀린 전제"가 아니라 위 "틀릴 수 있는 지점"
1·4번(호스트 대역폭, V4.1 블록 종류 확인)이다.

