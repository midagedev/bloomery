# GPU 전용 가중치 포맷 — GGUF를 떠나면 어디가 빨라지는가 (2026-09-21)

질문(사용자): "exllama처럼 별개의 모델 포맷을 쓰면 성능을 확 높일 지점도 존재할까?" 조사 라운드 둘(agy, gemini-3.8-flash-high)이 소스를 클론해 읽었고, 리드가 핵심 주장을 원본에서 대조했다. 아래에서 **확인**은 리드가 그 파일을 직접 열어 본 것, **보고**는 조사 라운드의 독해를 그대로 옮긴 것, **미검증**은 출처 대조가 안 된 것이다. 원 보고서 둘은 세션 스크래치에 있고(`fmt-a/report.md`, `fmt-b/report.md`) 클론은 tarball이라 커밋 해시는 검증하지 못했다.

## 답의 뼈대

디코드 한 스텝은 대역폭에 묶여 있다(1.33 GB, 바닥 1.42 ms — gpu-design.md). 그래서 포맷이 속도를 바꾸는 길은 셋뿐이다.

1. **같은 품질에서 비트를 줄인다** — 읽는 바이트가 1차 항이다. 트렐리스 계열(QTIP, EXL3, ik의 `IQ*_KT`)이 이 길이다.
2. **같은 비트를 GPU가 읽기 좋게 놓는다** — 행 인터리브(ik `_R4`/`_R8`), Marlin의 순열, gate·up의 물리적 결합. 값은 그대로라 오라클이 유지된다.
3. **디퀀트 자체를 싸게 만든다** — EXL3의 계산식 코드북.

그리고 흔히 "포맷 덕"으로 불리지만 **포맷과 무관한 것**이 가장 크다: MoE 층을 런치 몇 번으로 도는가.

## ExLlamaV3에서 읽은 것

- **MoE 층 전체가 런치 2번이다 — 확인**(`exllamav3_ext/quant/exl3_moe_coop_kernel.cuh` 머리 주석, `exl3_moe_coop.cu`). A 커널 = gate+up+활성화+down 입력 회전, B 커널 = down+가중합+공유 전문가 병합. 블록 하나가 (전문가 run, 32열 그룹, 투영) 하나를 맡고, 128열 청크를 덮는 마지막 도착 블록이 **완료 카운터**로 다음 단계를 마무리한다. 전문가 가중치는 디바이스 포인터 테이블로 찾는다. 런치는 `cudaLaunchCooperativeKernel`이고 그리드는 공동 상주 블록 수로 제한된다(`exl3_gemv.cu:131-172`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`). 같은 파일에 "split-k는 MoE 디코드에서 효과 없음으로 측정"이라는 주석이 있다(`exl3_moe_coop.cu:74-75`).
- **코드북은 계산식이고 테이블이 없다 — 확인**(`codebook.cuh`). `mul1`: `x·0x83DCD12D`의 네 바이트 합(`__dp4a(x, 0x01010101, 0x6400)`)을 half FMA 한 번으로 스케일·바이어스. 활성값이 int8이면 `dp4a(x·M, splat(a))` 한 명령이 디퀀트와 곱셈누산을 같이 하고, 두 번째 dp4a로 반올림 잔차를 보정하는 모드가 있다(`exl3_gemv_int8_kernel.cuh:23-30`).
- **sm_86 분기가 코드에 있다 — 확인.** 3 bpw는 Ampere에서 레지스터 디퀀트가 블록 파이프라인 커널에 져서 staged 경로에 남는다(`exl3_moe_coop_kernel.cuh`의 `tile_reg`). 정수 MAD 강제 워크어라운드는 CUDA 13.2에서 역전됐다는 주석(`codebook.cuh:3-5`).
- 레이아웃 — 보고: 16×16 타일이 정확히 K bpw(반정수 비트레이트는 +0.5), 스케일 텐서 없음, 부호 벡터 `su`/`sv`와 128점 Hadamard가 GEMV의 프롤로그·에필로그에 융합. Q·K·V는 `SlicedMultiLinear`로 한 커널(확인: `modules/multilinear.py:46`), gate/up은 변환 시 `--tie`로 비트레이트를 묶어 융합 커널이 성립(확인: `doc/optimize.md:204`).
- DeepSeek V3/V4·MLA 지원이 있다 — 확인(`architecture/deepseek_v4.py`, `modules/mla_attn.py`).

ExLlamaV2 쪽은 배울 것이 하나다. act-order 순열을 상류 층의 출력 열에 로드 시 미리 적용해 런타임 gather를 없앤다 — 확인(`exllamav2/mlp.py:162-164`). 우리는 GPTQ 계열이 아니라 해당 없다. V2의 MoE는 전문가마다 런치를 돌고 128전문가에서는 호스트 동기화까지 한다(보고) — V3가 고친 바로 그 부분이다.

## 그 밖의 포맷

- **ik_llama.cpp에 트렐리스 타입이 이미 있다 — 확인.** `GGML_TYPE_IQ2_KT/IQ3_KT/IQ4_KT`(`ggml.h:454-456`), CUDA mmvq 경로(`iqk_mmvq.cu:45`, q8_1 활성), 박스의 `llama-quantize`가 `IQ3_KT 3.125 bpw`, `IQ2_KT 2.125 bpw`를 받는다. Hadamard 변환이 없다(보고). **이것이 우리에게 가장 중요한 발견이다**: 길 1을 가더라도 GGUF 컨테이너·ik 변환기·ik 오라클을 전부 유지한다. EXL3로 가면 변환기(Hessian 캘리브레이션 + LDLQ)를 새로 짓거나 기존 EXL3 파일의 로더를 써야 하고, 오라클을 잃는다.
- `_R4`/`_R8` 행 인터리브의 CUDA 인스턴스가 있다 — 확인(`template-instances/mmq-instance-iq*_r4.cu`). 조사 B는 이것을 1순위("메모리 컨트롤러 효율 60–70% → 90%", 확신 100%)로 올렸으나 **그 수치는 출처가 없다 — 미검증**. CPU에서 우리가 잰 `-rtr`의 값은 +1.3–1.8%였다(cpu-dispatch-plan.md). GPU 값은 재야 안다.
- Marlin(보고): 4비트 전용 설계, 커널 접근 패턴에 맞춘 가중치·스케일 순열, M=1을 GEMM 타일로 패딩한다. 3비트급과 M=1 MoE에는 그대로 맞지 않는다는 것이 조사의 결론.
- AQLM 1x16은 1 MB 코드북이 공유 메모리를 넘쳐 배치 1에서 불리하다(보고). QTIP/QuIP#는 Hadamard 비용이 붙는다 — 조사 B의 "WHT 한 번 2.5 µs, 스텝당 350–450 µs"는 **미검증 수치**이고, EXL3는 그것을 GEMV 안에 융합해 런치를 따로 쓰지 않는다(위).

## 버린 것, 믿지 않는 것

- **두 보고서의 perplexity 표는 서로 모순된다**(Q3_K_M 7B가 한쪽 5.88, 다른 쪽 6.18; QTIP 3비트가 5.88 대 5.62; 한 표에는 13B 4비트가 FP16보다 낮다). 어느 쪽도 인용하지 않는다. "트렐리스가 같은 품질에서 0.3–1 bpw 적다"는 방향만 저자들의 주장으로 남기고, 크기는 우리 모델에서 KL/argmax로 재야 한다.
- 조사 A의 Hadamard 나노초 계산(FLOPs ÷ TFLOPS)과 우리 수치의 오독("런치 오버헤드 2.3 µs")은 버린다.

## 함의

1. **가장 큰 것은 포맷이 아니다.** MoE 2런치 구조(완료 카운터를 쓰는 협동 런치)는 GGUF 위에서 그대로 성립한다. P0b(블록 0 FFN 융합 스파이크)의 목표 모양을 이것으로 잡고, 선결 하나는 확인됐다: cuda-core 0.3.1에 `launch_kernel_cooperative`·`launch_kernel_cooperative_on_stream`이 있고 런치 계약에 `cooperative` 필드가 있다(`src/lib.rs:31`, `src/simt/launch.rs:241`). `#[cuda_module]`이 생성하는 런처에서 그것을 고를 수 있는지, 협동 런치가 그래프에 캡처되는지는 미확인 — P0b의 첫 걸음이고, 안 되면 NVlabs 장부 후보다.
2. **길 1은 추측 없이 잴 수 있다.** 박스에서 V2-Lite를 `IQ3_KT`·`IQ2_KT`로 재양자화(`--allow-requantize`, 품질은 버리는 속도 전용 파일, `/models/scratch-kt/`)해 ik의 `llama-bench -ngl 99`로 Q3_K_M과 한 임대에서 번갈아 잰다. 바이트가 줄어든 만큼 tok/s가 오르는지, 트렐리스 디퀀트가 그 이득을 먹는지가 ik라는 같은 엔진 위에서 한 번에 나온다. 오르면 `IQ3_KT` gemv가 P2 다음의 커널 꾸러미가 되고(오라클 유지), 안 오르면 길 1은 3090에서 닫는다.
3. **길 2는 결정 4가 이미 허용한다**(형식 변환은 로드 시점의 일). gate·up 인터리브는 P0b의 융합 커널이 요구하는 만큼만 한다 — 값이 안 바뀌니 밴드도 안 바뀐다.
4. 호스트 티어(V4.1): 전문가를 PCIe로 실어 나르는 설계라면 bpw가 버스 바이트에 그대로 곱해지므로 길 1의 값이 VRAM에서보다 크다. 다만 지금 설계는 전문가를 CPU에서 계산한다(offload-is-cpu-compute) — 그 경우 포맷은 CPU 커널의 문제다.
