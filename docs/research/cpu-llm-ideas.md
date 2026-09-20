# CPU LLM 조사 종합 — Intel·AMD·광역 생태계 (2026-09-20)

세 조사 트랙(병렬 서브에이전트, 웹 전용)의 교차 인덱스다. 세부 근거·출처 URL은 각 편에 있다:
[Intel](cpu-llm-intel.md) · [AMD](cpu-llm-amd.md) · [광역 생태계](cpu-llm-wider.md).
판정 기준은 [MUL-35 원장](../RESULTS-mul35-saturation.md): Zen 3 AVX2+FMA+F16C(VAES/VPCLMULQDQ 미사용, AVX-512/VNNI/AMX/GFNI 없음), 32코어 단일 NUMA, STREAM 147.7 GB/s, 디코드는 1.31 GB/step을 읽으며 풀 평균 9.1/32 가동.

## 세 편이 동시에 가리킨 결론

**커널은 대역폭 근처까지 끝났고, 남은 격차는 '런치/디스패치 오버헤드 + 고정비 상각'이다.** Intel(LMSYS Xeon 6: "TPOT 이득이 2~4×에 그치는 이유는 디코드가 대역폭 바운드"), AMD(ZenDNN vLLM 대비 1.60×의 잔여를 "커널 런치 오버헤드"로 진단), ik 실측(같은 5975WX에서 108 GB/s = STREAM의 73%)이 MUL-35 원장의 판정과 독립적으로 일치한다.

## ik_llama.cpp 2.64배(82.78 대 31.79, 같은 기계)의 기제 4가지 → 우리 대응

| ik 기제 | 출처 | 우리 대응 |
|---|---|---|
| CPU FA를 GEMV가 아닌 GEMM으로 + 분할 축 재설계, DS-Lite latent FA 특화 | 광역 PR #332/#410 | MUL-36 flash SIMD(착륙) — GEMM화·분할축은 후속 재료 |
| MLA TG-최적 그래프를 배치 128까지 유지 | 광역 PR #282 | 이미 동일 설계(확인 완료) |
| -fmoe: gate/up 양자화 공유, 전문가 탐색 1회화, 장벽 제거 | 광역 PR #229 | MUL-37(착륙) + Intel 1순위와 동일 |
| 텐서당 1노드·행-방향 연속 청크·노드별 장벽만 | 광역(ggml 구조) | **미개척 — '패킹' 라운드가 이것** |

## 통합 아이디어 순위 (이식성 필터 적용 후)

| 순위 | 아이디어 | 출처 | 공격하는 병목(원장) | 이식성 |
|---|---|---|---|---|
| 1 | **패킹/스케줄링 재설계** — 텐서-1노드 연속 청크, MoE topk_ids 정렬→전문가별 블록 배치 GEMM, 게더의 GEMM 흡수 | 광역(ggml/ik) + Intel 3위 + AMD 2위(PACE)·3위(ZenDNN group_matmul) | 사이트 내 오케스트레이션 잔여 ~10.2 ms/step | Zen 3 순수 소프트웨어 |
| 2 | **gate·up 결합 GEMM + SiLU·mul 에필로그**(A×[B1,B2] 단일 GEMM) | Intel 1위(실측 +12% TPOT 계열) | swiglu 직렬 1.37 + 디스패치 52→26 + (MUL-37이 못 삼킨 단일 사이트 quant) | Zen 3 가능 |
| 3 | **스펙 디코딩**(PARD식 k토큰 검증) | AMD 1위(실측 3-4×, 우리 추정 1.5-2.5×) | lm_head 대역폭 벽 + 직렬 quant + 오케스트레이션 **통째로 상각** — 원장 §7과 직교하는 유일 대형 레버 | Zen 3 순수 알고리즘(초안 모델 필요 — 후보: 자기 모델 Q3_K or n-gram) |
| 4 | **KV q8_0 양자화** — flash SIMD 이후 ctx 기울기 절감 | 광역(ik) | flash KV 읽기 = 스프레드의 전부 | Zen 3(정밀도 라운드 필요) |
| 5 | **quant의 GEMM 인출 융합**(동적 quant를 소비 GEMM 입력 단계에) | Intel 2위 | 직렬 quant 잔여(MUL-37 후에도 단일 사이트 인라인 분기) | Zen 3 가능 |
| 6 | **워커 수 스윕/CCD 친화도** | AMD 4위(ZenDNN "128 < 2×64" 공식 인정, -t 스윕 20-30%) | 풀 9.1/32 가동과 같은 현상 — 거의 공짜 실험 | 즉시 가능 |
| 7 | 디스패처 전용 코어 예약 + 스텝 정적 캡처 | Intel 5위(+10%) | 파킹/깨움 캐스케이드 | 저위험 |
| 8 | KV 쓰기·패킹 융합 "Load Once Pack Twice" | Intel 4위 | KV 관련 고정비 — flash SIMD와 독립 축 | Zen 3 가능 |
| 9 | kq_dot 부분합 4→8(FMA 지연 포화) | MUL-36 보고의 후속 | flash SIMD 잔여 | 게이트 보호하 저비용 |
| 10 | THP/2MB 페이지 A/B(1회 노벨) | 광역(정량 근거 부재 명시) | 스트리밍 TLB | 1회 실험 |

## 배제 확정 (근거까지)

- **T-MAC(LUT 디코드)**·**KTransformers**(AMX·NUMA 특화): 우리 병목 구도와 불일치(광역).
- **FairyFuse**(AVX-512 구현), **QUOKKA**(프리필 KV 스파스 — 소 ctx 부적합), **NoMAD-Attention**(2× 가능하나 품질 리스크 — 보류).
- **AOCL "BF16 on AVX2"**: GitHub 이슈 #2(2025-07)로 반박 — BF16/INT8/VNNI 전부 Zen 4+ 게이트(AMD).
- **cutile-rs**: GEMV·weight-only 양자화 미지원 + block-scaled MMA는 Blackwell/Hopper 전용 → 3090(sm_86) 무의미([MUL-30 정찰](../RESULTS-mul30-gpu-scout.md)).
- ipex-llm은 아카이브(2026-01) — AVX2 자산은 neural-speed/BesTLA 구 저장소에 남음(VNNI 없는 AVX2의 int8 오버플로 주석 포함).

## 하드웨어 교체 판단 (2026-09-20 문의에 대한 기록)

3995WX(Zen 2 64C, 같은 WRX80 8채널 DDR4)로의 교체는 **이득 없음~역행**: 대역폭 천장이 플랫폼에 묶여 있어(1.31 GB ÷ 147.7 GB/s = 8.87 ms ≈ 113 tok/s가 플랫폼 천장) 코어 수 배가가 무의미하고, 코어당 ~0.65배가 스칼라·직렬·지연 구간을 직격. L3 128→256 MB는 ctx ≳ 2,000의 KV 상주에서만 의미. 돈의 우선 용도는 패킹 라운드(순위 1)이고, 플랫폼을 바꾼다면 방향은 DDR5 대역폭(천장 자체를 올림).
