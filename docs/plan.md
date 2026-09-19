# bloomery — 단계와 게이트

이 문서는 계획이고, 숫자는 [rig-log](https://github.com/midagedev/rig-log)에 측정된 뒤에만 여기 옮겨 적는다. 통과 기준은 전부 측정으로 쓴다.

## 목표

DeepSeek-V4.1-Flash를 이 워크스테이션(RTX A6000 48 GB + RTX 3090 24 GB, 둘 다 sm_86, Threadripper 5975WX 32코어, 256 GB)에서 우리 엔진으로 서빙한다. 호스트는 Rust, GPU 커널은 CUDA Rust(지금은 cuda-oxide, 툴킷 13.3 라운드 뒤 cutile-rs), CPU expert 티어는 Rust SIMD. 기준선은 같은 자리에서 잰 것이다. ~~ik_llama.cpp: 서빙 프로파일 PPL 2.2355, 디코드 25.05 tok/s(2026-09-16).~~ **선 그음 2026-09-19 — 그 한 줄이 엔진 둘을 합쳐 놨다.** PPL 2.2355는 **ik 포트**(ik_llama.cpp #2455)가 plain Q3_K_M 324 GB에서 낸 값이고, 디코드 25.05 tok/s는 **mainline 포크**(vcruz305/llama.cpp — `iqk` 디렉터리가 없다)가 grafted 445 GB 파일에서 낸 값이다. 이 박스의 V4.1 서빙은 처음부터 mainline이었다(rig-log `log/2026-09-12-deepseek-v41-first-run.md`가 "mainline llama.cpp지 ik 아님"이라고 적어 뒀고, 여기 옮겨 적으면서 틀렸다). `llm.service`가 띄우는 것은 또 다른 구성이다 — ik + DeepSeek-V4-Flash-**0731** Q4_K_XL 155 GB, 포트 8000.

같은 배치에서 둘을 나란히 잰 표가 #2455 본문에 있다: PPL 2.2355(ik) 대 2.2556(mainline), 디코드 20.4–20.7(ik) 대 21.2(mainline). **두 엔진이 3 % 안에 있다.** 이것이 이 프로젝트에 주는 것은 둘이다 — (1) 넘어야 할 선은 사실상 하나다, (2) ik의 존재 이유인 `iqk_mul_mat`이 이 배치의 호스트 expert 항에서 값을 못 받고 있다. 아주 다른 두 CPU 커널이 3 % 안에 떨어진다는 것은 그 항이 커널이 아니라 **대역폭**에 묶여 있다는 방증이고, `docs/roofline.md`가 그렇게 예측했다.

왜 직접 만드는가: 이 박스가 실제로 서빙하는 모델은 mistral.rs가 로드하지 못하고, ik의 MoE 디코드는 배치를 못 하며, 어느 쪽이든 고치려면 업스트림 머지를 기다려야 한다(2026-09-19 기준 mistral.rs는 09-08 이후 정지, 외부 PR 40건 대기). 발견은 계속 업스트림에 코멘트로 보내되, 엔진은 머지에 의존하지 않는다.

## 디딤돌

V4.1 Flash는 engram 테이블 NVMe 지연 읽기, 공유 압축 KV, 지연 하이퍼커넥션, 저랭크 query norm, CPU expert 티어, DSpark 드래프트를 한꺼번에 요구한다. 먼저 **DeepSeek-V2-Lite-Chat Q3_K_M**(8.1 GB, `deepseek2`, MLA, MoE 64 expert, 3090에 들어감)으로 로더·MLA·MoE·게이트를 세우고, V4.1 고유 부분은 ik 포트(ik_llama.cpp #2455)를 참조로 얹는다.

## 단계

| 단계 | 만드는 것 | 통과 기준 |
|---|---|---|
| 0 | cuda-oxide로 쓴 Q3_K gemv를 같은 텐서의 ggml `mmvq`와 대조 (MUL-1, rig-log WKS-35) | ~~최대 상대 오차 ≤ 1e-4~~ f32 활성값이면 1e-4, q8_1 활성값이면 1e-2(설계를 명시) — 2026-09-19 선 그음: 2라운드가 ggml처럼 q8_1로 갔고 오차 4e-3은 그 설계의 것이다. 대역폭 `stack` M=1에서 ggml의 0.9배 이상. **통과 2026-09-19: 1.05배**(348 대 333 GB/s, 3090). M=8 0.83배는 MUL-8, Q4_K·Q6_K·A6000 행은 MUL-9 |
| 1 | V2-Lite 순전파. 세로로 다섯 라운드, 각 라운드가 끝에서 끝까지 도는 것 하나를 내고 자기 게이트를 가진다(아래 표) | 고정 프롬프트에서 ik greedy와 토큰 동일, wikitext-2 PPL이 ik 오차 안 |
| 2 | 호스트 expert 티어: 일부 층 expert를 CPU에, 양자화 상태 sparse gemv를 코어 전체에. 커널은 이미 있다(MUL-3, ggml의 1.1배) — 남은 것은 호출자다 | 층·토큰당 비용을 같은 런의 ik와 대조(참조 급: Qwen3.6에서 ik 0.20 ms), batch 2가 batch 1보다 느리지 않음 |
| 3 | V4.1 아키: engram mmap + WILLNEED 프리페치, 하이퍼커넥션, 공유 KV, query norm | PPL 2.2355 행 패리티, 같은 배치에서 25.05 tok/s 대비 |
| 4 | DSpark 드래프트 + 스케줄러, 빠른/느린 모델 라우팅 | 수락률·tok/s가 ik 서빙 프로파일 이상 |

## 1단계를 세로로 자른다 (2026-09-19)

가로로 자르면("로더를 쓴다", "어텐션을 쓴다") 마지막 조각이 들어올 때까지 잴 것이 없다. 세로로 자르면
라운드마다 ik와 대조할 수치가 하나 나온다. 각 라운드는 `just` 타깃 하나로 실행되는 게이트를 갖는다.

| 라운드 | 만드는 것 | 게이트 | 의존 |
|---|---|---|---|
| 1-1 | GGUF 로더와 텐서 맵, 양자 타입별 디퀀트 | ~~≤ 1e-6~~ **통과 2026-09-19: 여섯 타입 전부 차이 0(비트 정확)**, 텐서 377개 전부 해석. `just gate-1-1` | 없음. GPU 불필요 |
| 1-2 | 임베딩 + 블록 0(MLA 어텐션 + dense FFN), CPU, f32, M=1 | 블록 0 출력이 ik의 중간 텐서와 ≤ 1e-3 | 1-1 |
| 1-2 ffn | dense FFN(블록 0)과 shared expert | **통과 2026-09-19**: `ffn_up_gate-0` 3.6e-7, `ffn_out-0` 2.2e-5, `ffn_shexp-1` 5.4e-7 (게이트 1e-4). `just gate-ffn` | ops |
| 1-2 attn | MLA 어텐션(블록 0) | 진행 중 | ops |
| 1-3 | MoE 블록: 라우터, top-6, expert 디스패치 | **통과 2026-09-19**: 라우팅 id 36/36 정확, `ffn_moe_out-1` 2.0e-5, `ffn_out-1` 2.0e-5 (게이트 1e-4), `down` 2.8e-4 (게이트 4e-4, 유도는 호출부 주석). 구조 게이트: 라우팅된 25개만 디퀀트. `just gate-moe` | 1-2 |
| 1-4 head | 출력 헤드 | **통과 2026-09-19**: `result_norm` 9.5e-7, `result_output` 4.0e-5, argmax·top-5 정확 일치. `just gate-head` | 1-3 |
| 1-4 | 토큰 하나의 전체 순전파, CPU, M=1 | 프롬프트 32개에서 logits argmax가 ik와 일치. 첫 tok/s(느릴 것이다) | 1-3 |
| 1-5 | dense 경로 GPU 오프로드 + 라우팅 expert에 `q3k-gemv`, KV 캐시 | 같은 logits, 3090에서 ik 대비 tok/s | 1-4 |

~~1-1과 1-2는 파일이 겹치지 않으므로 병렬로 돌린다. 나머지는 직렬이다.~~ 2026-09-19 선 그음:
**오라클이 직렬 사슬을 팬아웃으로 바꾼다.** 1-2 → 1-3 → 1-4가 직렬이었던 이유는 각 라운드의
입력이 앞 라운드의 출력이었기 때문인데, 오라클이 그 입력을 이미 파일로 갖고 있다. 그래서
attn·ffn·moe·head·gemv 다섯을 한 번에 띄웠고 넷이 같은 오후에 들어왔다. 남은 직렬 의존은
1-4 조립(모듈을 `forward`로 엮는 것)과 1-5뿐이다.

그 팬아웃이 즉시 값을 낸 지점도 기록해 둔다. **같은 사실이 세 라운드에서 따로 발견됐다** —
활성값 양자화 포맷이 가중치 타입마다 다르다(Q3_K만 Q8_K, 나머지는 Q8_2_X4). ffn이 Q5_1에서
먼저 찍었고, head가 Q6_K에서 409× 초과로 독립 재현했고, attn이 지금 같은 계열의 타이 플립을
쫓고 있다. 직렬이었으면 세 번 따로 나타났을 것이다.

라운드 전에 리드가 정해 두는 것(정하지 않으면 세 군데서 따로 발명된다): 텐서 레이아웃은 ggml의 `ne[]`
순서 그대로 — 비교가 전부 같은 모양이 된다. 양자 타입은 `quant` 크레이트 하나가 소유하고 CPU·GPU 커널이
같이 쓴다. 오라클 경계는 `$BLOOMERY_DATA/ref/`에 ik가 떨군 중간 텐서이고 게이트는 거기서 읽는다.
오류 타입은 라이브러리 `thiserror`, 바이너리 경계 `anyhow`.

**mistral.rs를 읽고 추가된 것** (2026-09-19, `docs/research/mistralrs-prior-art.md`). 이것들은
나중에 못 고치는 부류라 라운드 전에 박는다.

- **텐서마다 (디바이스, dtype) 주소를 로드 시점에 준다.** 층 단위 배치 + 전역 dtype 조합을 쓰지
  않는다. mistral.rs의 `DeviceMapper`는 층 인덱스만 받아서 expert 단위 배치를 아예 표현하지
  못하고, 그것이 2단계에서 우리가 필요한 바로 그것이다.
- **MoE CPU 경로는 첫날부터 라우팅된 expert만 도는 sparse dots다.** 폴백이 라우팅 안 된
  expert를 디퀀트하면 죽는 게이트를 1-3에 건다. 그 폴백이 mistral.rs의 442 ms다.
- 모델 하나는 파일 하나. 이름 표는 표로, arch마다 여섯 파일에 흩뿌리지 않는다.
- GGUF인지 아닌지는 가중치 소스 경계에서 한 번 푼다. 모델 코드가 컨테이너 형식으로 분기하지 않는다.
- 1단계 디스패치는 monomorphic. 토큰 경로에 `Mutex<dyn Trait>`·`Box<dyn Any>`를 두지 않고,
  넣어야 할 때는 비용을 먼저 잰다.
- KV는 미리 잡은 버퍼에 제자리 append. 디코드 경로의 clone-in/clone-out을 금지한다.
- MLA는 CPU에서 latent 위의 weight-absorbed 형태로 먼저. 융합 커널은 나중의 최적화지 설계가 아니다.
- **활성값은 첫 줄부터 `[ne0=embd, ne1=n_tokens]`다.** 라운드 표가 "M=1"이라고 적은 것은 그
  라운드가 재는 것이 토큰 하나라는 뜻이지, API가 토큰 하나만 받는다는 뜻이 아니다. 루프라인이
  말하는 가장 큰 지렛대는 한 번 읽은 가중치로 여러 토큰을 내는 것(`docs/roofline.md`)이고,
  M=1로 쓴 순전파는 거기로 싸게 못 간다. 블록 0은 dense라 지금은 배치 차원이 공짜다.
  1-3(expert 디스패치)은 `docs/research/quant-decode-efficiency.md`의 Q6가 도착한 뒤에 연다 —
  `ne1 > 1`에서 expert 디스패치가 어떻게 생겨야 하는지가 거기서 정해진다.

**리서치를 해가며 진행한다**(사용자 지시 2026-09-19). 라운드 스펙마다 "선행 조사:" 한 줄로
근거가 된 `docs/research/` 문서를 가리킨다. 루프라인은 `docs/roofline.md`가 소유하고,
어떤 기법이든 "루프라인의 어느 항을 얼마나 줄이는가"로 환원되지 않으면 설계 입력이 아니다.

## 이 모델에 실제로 들어 있는 것 (2026-09-19 GGUF 헤더 직접 파싱)

DeepSeek-V2-Lite-Chat Q3_K_M, GGUF v3, 텐서 377개, 블록 27개(블록 0은 dense, 1~26이 MoE),
expert 64개 중 6개 사용, 임베딩 2048, 헤드 16.

| ggml 타입 | 개수 | 어디 |
|---|---:|---|
| F32 (0) | 108 | 모든 norm, 라우터 `ffn_gate_inp` |
| **Q5_0 (6)** | 26 | `blk.N.ffn_down_exps` — expert down 투영 |
| **Q5_1 (7)** | 1 | `blk.0.ffn_down` — dense 블록의 down |
| Q3_K (11) | 188 | `token_embd`, `attn_q`, `attn_kv_a_mqa`, `attn_kv_b`, gate/up 전부 |
| Q4_K (12) | 53 | `attn_output`, `ffn_down_shexp` |
| Q6_K (14) | 1 | `output.weight` |

**선 그음:** 앞선 기록이 expert down을 Q5_K라 적었는데 실제는 **Q5_0**이다. K-quant가 아니라 레거시
계열이라 블록 기하가 완전히 다르다(32값, f16 d + 4비트 qs + 5번째 비트를 담은 u32). 1-3 스펙이 이걸
전제로 쓰여야 한다. 전체 순전파에 필요한 디퀀트 경로는 다섯이다: Q3_K, Q4_K, Q5_0, Q5_1, Q6_K.

## 이 기계에 맞춘다는 것 (사용자 요구, 2026-09-19)

목표는 범용 엔진이 아니라 **이 장비에서의 최선**이다. CPU 쪽 사실은 박스에서 읽었다(2026-09-19 `lscpu`, `/proc/cpuinfo`): Threadripper PRO 5975WX, **Zen 3**, 32코어 64스레드, CCD 4개에 L3 32 MB씩 128 MB, L2 512 KB/코어, NUMA 노드 1개(NPS1), SMT 켜짐, THP `madvise`. ISA는 **AVX2·FMA3·F16C·BMI2**까지이고 **AVX-512도 VNNI도 없다.** 메모리 읽기 실측 147.7 GB/s(잠정 — 4라운드에서 두 엔진이 317 MB 작업 집합으로 이 값을 넘었다, 재측정 WKS-35)(rig-log, 32스레드).

이 사실이 2단계(호스트 expert 티어)의 설계를 정한다.

- **정수 내적은 AVX2 `vpmaddubsw`/`vpmaddwd` 사슬**이다. VNNI가 없으니 ik의 `iqk_mul_mat`가 AVX2에서 하는 그대로, 활성값을 Q8로 양자화해 int8×int8→int16→int32로 누산한다. Rust는 `std::arch::x86_64` AVX2 intrinsics가 stable이고, 박스 빌드는 `-C target-cpu=znver3`, 런타임 감지는 `is_x86_feature_detected!`. f32 FMA 경로는 대조군으로만 둔다.
- **스레드 수는 재서 정한다.** 이 티어는 대역폭 바운드라 SMT 64스레드가 32코어보다 나을 이유가 없다 — ik는 26.7코어를 썼다. 8·16·32·64스레드 스윕 한 번이 답이고, 이후 스케줄러가 그 값을 고정한다. GPU 스트림 스레드·서버 스레드 몫을 뺀 코어 예산도 같은 스윕에서 정한다.
- **CCD 친화성.** 코어 8개씩 L3를 공유하니 expert 행 블록을 CCD 단위로 배정하고 스레드를 `sched_setaffinity`로 고정한다. 라우팅된 expert 8개를 CCD 4개에 나누면 L3에서 서로 밀어내지 않는다. NPS1이라 메모리 채널은 인터리브 하나다 — NPS2/NPS4가 대역폭을 바꾸는지는 BIOS 라운드(rig-log WKS-22)의 질문이고 여기서 가정하지 않는다.
- **페이지.** 가중치 mmap에 `madvise(MADV_HUGEPAGE — 4라운드(2026-09-19) 초 단위 벤치에서 THP 미발동(`AnonHugePages` 0 kB), 효과 미확인)`(THP가 `madvise` 모드라 이게 유일한 경로), 행 블록 선행 프리페치, 그리고 engram처럼 안 읽는 바이트는 안 올린다.
- **천장을 먼저 계산한다.** 토큰당 이 티어가 읽는 바이트(라우팅 expert 수 × 층 × Q3_K 행 크기)를 147.7 GB/s로 나눈 값이 하한이고, 게이트는 그 하한 대비 비율로 쓴다. ik의 0.20 ms/층(Qwen3.6, 2026-09-17)이 참조 급이다.

GPU 쪽도 같은 원칙이다. sm_86은 `dp4a`(int8 내적)와 int8 텐서 코어(IMMA)를 갖고 있고 FP8은 없다. M=1 gemv는 대역폭에, M=8 이상 배치는 IMMA에 거는 것이 이 카드의 형태다. 0단계 2라운드의 q8_1 선택지가 그 첫 걸음이다.

## 규칙

- 개발은 3090에서. A6000은 서빙과 야간 학습이 쥐고 있고, 0~2단계는 그것을 건드리지 않는다. 툴킷 13.3 승격만이 기계 변경이고, 그때는 rig-log의 machine-changes에 기록한다.
- 첨부 프로토콜(`/props` + 청크별 `timings`)은 1단계부터. toktape가 어느 단계든 녹화할 수 있어야 한다.
- GGUF 파서·토크나이저는 기존 crate. 이 엔진의 가치는 배치·스케줄러·커널이다.
- 측정은 조용한 기계 프로토콜(rig-log `docs/quiet-machine.md`)로, 행마다 증인을 남긴다.

## 툴체인 (2026-09-19 박스에서 확인)

nightly-2026-08-28(각 crate의 `rust-toolchain.toml`이 고정), LLVM 21.1.8은 apt가 아니라 릴리스 타르볼(`~/opt`), CUDA 13.0, 드라이버 615.71.09. `cargo oxide doctor` 전 항목 통과, `vecadd`가 `.target sm_86` PTX로 3090에서 정답. 빌드·실행은 `tools/box.sh`가 트리를 박스로 rsync한 뒤 돈다.
