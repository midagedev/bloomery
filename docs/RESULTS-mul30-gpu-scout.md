# MUL-30 GPU 단계 정찰 — cutile-rs vs cuda-oxide, Q4_K GEMV 스파이크의 재료

2026-09-20, 읽기 전용 정찰(빌드·컴파일·설치·푸시 없음). 조사 수단: `tools/box.sh` 가벼운 읽기 명령, Mac 워크트리 파일 읽기, GitHub/crates.io 웹 조회. 모든 주장에 출처를 붙였다.

---

## 1. 박스 인벤토리 — 설치됨/누락

| 항목 | 상태 | 값 (명령 출력 인용) | 출처 |
|---|---|---|---|
| 개발 GPU (index 0) | 설치됨 | `NVIDIA GeForce RTX 3090`, 24576 MiB, 유휴(1 MiB / 0% / 30 W) | box.sh `nvidia-smi` (2026-09-20 15:08 UTC) |
| 금지 GPU (index 1) | 존재(사용 금지) | `NVIDIA RTX A6000`, 49140 MiB, 유휴 | 同上 |
| 드라이버 / CUDA UMD | 설치됨 | `NVIDIA-SMI 615.71.09`, `CUDA UMD Version: 13.4` — cuda-oxide 요구 "R580+" 충족 | nvidia-smi; cuda-oxide README |
| nvcc (툴킷) | 설치됨 — 단 **13.0** | `release 13.0, V13.0.88` (Built 2025-08-20), `/usr/local/cuda-13.0` | box.sh `nvcc --version`, `ls /usr/local/cuda*` |
| CUDA 13.2/13.3 툴킷 | **누락** | `/usr/local/`에 `cuda`, `cuda-13`, `cuda-13.0`만 — cutile-rs sm_8x 요건(13.2) 미충족 | box.sh `ls /usr/local/` |
| cargo-oxide | 설치됨 | `/root/.cargo/bin/cargo-oxide`, `cargo-oxide 0.2.1` | box.sh `which cargo-oxide && cargo-oxide --version` |
| LLVM llc (cuda-oxide 백엔드) | 설치됨 | `LLVM version 21.1.8`, `/root/opt/LLVM-21.1.8-Linux-X64/bin/llc` | box.sh llc --version |
| Rust 툴체인 | 설치됨 | `rustc 1.100.0-nightly (e457a7b0d 2026-08-27)` = 핀 `nightly-2026-08-28` | box.sh rustc --version; `rust-toolchain.toml` |
| 디스크 | 충분 | `/` 1.8T 중 991G 가용 (44% 사용) | box.sh `df -h /root` |
| 메모리 | 충분 | 251G total, 245G available | box.sh `free -g` |
| ik_llama.cpp (참조) | 설치됨 | `/home/user/ik_llama.cpp` @ `c10fbbcc` (2026-09-15), `build/ggml/src/libggml.so` 빌들됨 | box.sh git log, ls |
| 참조 하네스 바이너리 | 설치됨 | `/root/bloomery-data/bin`: `q3k_ref`, `q4k_ref`, `q4k_x4_ref`, `q4k_x4_rate`, `q6k_x4_*`, `q5f0_*`, `q5f1_*` 등 14개 | box.sh ls |
| rsync 트리 | 존재 | `/root/repo/bloomery` (crates/q3k-gemv/src/main.rs 있음) + 워크트리 10개(-q6k, -q5f0, -qdot 등) | box.sh ls |
| cutile 크레이트 캐시 | **없음** | `~/.cargo/registry`에서 cutile 0건 — cutile 첫 빌드는 crates.io 접근(또는 벤더링) 필요 | box.sh ls registry (추정 아님, 확인된 부재) |

`/root/bloomery-env.sh` 전문 (box.sh cat, 핀 파일 — cuda-oxide 핀을 소유):

```bash
export PATH=$HOME/.cargo/bin:$HOME/opt/LLVM-21.1.8-Linux-X64/bin:/usr/local/cuda-13.0/bin:$PATH
export LIBCLANG_PATH=$HOME/opt/LLVM-21.1.8-Linux-X64/lib
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0
export CUDA_OXIDE_LLC=$HOME/opt/LLVM-21.1.8-Linux-X64/bin/llc
export CUDA_VISIBLE_DEVICES=GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1
```

마지막 줄이 3090을 UUID로 고정한다 — A6000(index 1)은 이 환경에서 보이지 않는다. 배경 메모의 "CUDA 13.3 툴킷 추가가 필요할 것"은 여전히 유효하되 **아직 실행되지 않았다**(13.0만 설치됨).

---

## 2. Mac 쪽 프로젝트 상태와 핀

### 2.1 핀 상태 — 드리프트 0

`Cargo.toml` [workspace.dependencies] (Mac 읽기):

```toml
cuda-device = { version = "0.2.1", git = "https://github.com/NVlabs/cuda-oxide.git", rev = "b9847e9515ed3a23096f22567d3eaf0a6e3e440c" }
cuda-host   = { version = "0.2.1", git = ".../cuda-oxide.git", rev = "b9847e9515ed3a23096f22567d3eaf0a6e3e440c" }
cuda-core   = "0.3.1"
```

- 핀 rev `b9847e9` = 업스트림 커밋 "fix(codegen): preserve assertions during loop unrolling (#1293)", 2026-09-18T07:12:34Z. (GitHub API /repos/NVlabs/cuda-oxide/commits/b9847e95…)
- 2026-09-20 기준 이 rev가 cuda-oxide main의 **최신 HEAD 그 자체**다 — 최근 커밋 10개가 전부 2026-09-18 07:09–07:12 배치 머지. (GitHub API commits?per_page=10) → **업스트리프트 드리프트 없음.**
- `cuda-core 0.3.1`은 crates.io 발행이며 Cargo.toml 주석대로 "NVlabs/cutile-rs 저장소에서 publish"된 공유 호스트 런타임. Cargo.lock에서 cuda-artifact-finalizer 0.2.1(같은 git rev), cuda-bindings 0.3.1(registry) 확인.
- 박스의 `cargo-oxide 0.2.1`이 핀과 같은 버전 대 — 도구와 핀이 정합.
- `rust-toolchain.toml` = `nightly-2026-08-28` + rust-src/rustc-dev/llvm-tools — cuda-oxide README가 명시하는 핀 툴체인과 동일.
- 참고: 핀 HEAD 커밋 자체가 루프 언롤 코드젠 수정이다. 우리가 실제로 맞은 `#[unroll]` ICE(MUL-7, `const_fold.rs:200` / `unroll.rs:387`, `APInt::shl` 폭 불일치)는 업스트림에 미보고 — 초안이 `docs/upstream/cuda-oxide-unroll-ice.md`에 있다. 언롤 패스가 계속 움직이는 영역이라 재검 가치 있음.

### 2.2 GPU 단계는 어디까지 갔나 (rig-log + 크레이트 결과 파일)

- **0단계(MUL-1), 2026-09-19**: Q3_K GEMV를 cuda-oxide로 4라운드. 1라운드 0.40배 → 2라운드(q8_1+dp4a) 1.05배 → 3라운드 결합 커널 **1.86배**(620 GB/s, 이론치 936의 66%; ggml mmvq 333 GB/s). 오차 4e-3대(q8_1 설계). `log/2026-09-19-a-first-rust-kernel-two-arms.md`. 이 기록이 명시하기를: **"cuTile 트랙(cutile-rs)은 툴킷 13.3 라운드(MUL-6) 전이라 재지 않았다"** — cutile은 한 번도 재지 않았다.
- **MUL-9 스테이지 0 확장**, 2026-09-19: Q4_K(`q4k_gemv`)·Q6_K(`q6k_gemv`) 커널이 `crates/q3k-gemv/src/main.rs`(1911줄)에 이미 존재. `RESULTS-q4k-q6k.md`: Q4_K attn-stack M=1 **809.8 GB/s = ggml mmvq의 1.06배**(게이트 0.9 통과), Q6_K head M=1 836.7 GB/s = 1.12배. M=8은 Q4_K 0.67배(구조적: min-offset B 체인의 dp4a 밀도). 하네스는 `tools/ref/measure.sh` — 3090 UUID 고정, `wait_gpu` 유휴 대기, 증인 블록, ggml 참조와 같은 호출에서 대조.
- **현재 위치**: `docs/plan.md`의 1-5 GPU 행("dense 경로 GPU 오프로드 + 라우팅 expert에 q3k-gemv", 게이트 "같은 logits, 3090에서 ik 대비 tok/s")은 아직 착수 안 함 — CPU 쪽 1-5가 MUL-34까지 마무리되어 스테이지 표가 평탄해진 지금이 GPU 차례. plan.md 목표 문장: "GPU 커널은 CUDA Rust(지금은 cuda-oxide, 툴킷 13.3 라운드 뒤 cutile-rs)".
- 크레이트 README는 스캐폴드 시절 텍스트(vector-add 템플릿 안내)로 낡았다 — main.rs 실체와 불일치(정찰 소견).

---

## 3. 웹 리서치

### 3.1 NVlabs/cutile-rs (github.com/NVlabs/cutile-rs, 2026-09-20 조회)

- **정체**: 타일(tile) 기반 DSL. `#[cutile::module]` 매크로가 커널 Rust AST를 임베드하고 **런타임에 JIT**로 "CUDA Tile IR을 거쳐 cubin"을 만든다. 소유권 모델을 런치 경계 너머로 확장(가변 텐서 분할/불변 공유). cuda-oxide의 SIMT 트랙과 형제 프로젝트 — 호스트 런타임(cuda-core, cuda-async)을 공유하고 SIMT 인터롭(같은 스트림에 cutile 타일 커널 + cuda-oxide SIMT 커널 체이닝)이 있다. README 자신을 "research-stage, bugs·incomplete features·API breakage 예고"라 적는다.
- **요구 툴체인**: Rust **stable 1.89+ (nightly 불필요)**; CUDA — sm_8x(Ampere/Ada, 우리 3090의 sm_86 포함)에 **13.2**, sm_90에 13.3, "CUDA 13.3 is recommended"; sm_80 미만(sm_70/75) 지원 안 함. 툴킷은 `CUDA_TOOLKIT_PATH`로 지정. 호스트 컴파일러 명시 없음(Ubuntu 24.04 테스트). → **박스의 13.0 툴킷으로는 sm_86 빌드가 안 된다** — MUL-6(13.2/13.3 설치)이 선행 조건.
- **빌드 모델**: 크레이트는 crates.io 배포(`cutile` 최신 **0.3.1**, 2026-09-04 발행) — cargo 첫 빌드 시 네트워크에서 당겨온다(박스 레지스트리 캐시 없음, 1절). 커널 컴파일은 빌드 시가 아니라 **첫 런치에 JIT**이고 0.3.1에서 캐시 히트 런치가 3.9→1.6 µs로 빨라졌으며 커널 캐시 제거 API가 추가됐다(CHANGELOG). 외부 LLVM/MLIR JIT 의존은 0.0.2에서 제거(자체 cutile-ir).
- **라이선스**: Apache-2.0 (전 크레이트).
- **활동**: 활발 — 마지막 커밋 2026-09-18(DGX Spark sm_121 bring-up 등), 181 커밋, 992 stars.
- **제공 연산**: element-wise, GEMM(B200 f16 2.07 PFlop/s, cuBLAS 급), 비동기 GEMM, NVFP4 pack·block-scaled MMA(13.3 필요), MXFP8 예시. **GEMV는 언급 없음. INT8/weight-only 양자화(Q4_K 등 gguf 형식) 지원 없음.** 정수 지원은 u8/u16 부호성 수정(0.3.1), i32 범위 추적 수정(0.3.0)이 전부 — dp4a급 정수 SIMD 원시는 미확인(README/CHANGELOG 무언급).

### 3.2 NVlabs/cuda-oxide 업스트림 (github.com/NVlabs/cuda-oxide)

- **정체**: rustc 커스텀 백엔드 — "Rust → MIR → Pliron IR → LLVM IR → PTX", `cargo oxide build`로 단일 소스 호스트+디바이스 컴파일. early alpha. ~3.5k stars, main 1,172 커밋.
- **요구**: CUDA 툴킷 13.0+ (cuRAND 헤더 포함), 드라이버 R580+, Rust nightly(rust-src·rustc-dev·llvm-tools) 핀 `nightly-2026-08-28`, llc 23/22/21 자동 탐색, Clang 21(bindgen). **→ 박스는 전부 이미 충족**(1절 표). 라이선스 Apache-2.0.
- **양자화**: Q4_K·INT8·dp4a 등 양자화 포맷 지원은 업스트림에 없다 — 커널은 우리가 직접 짰고(2.2), dp4a/warp 원시는 `cuda_device`가 제공(main.rs import 참조).
- **핀 대비 현재**: 2.1 — 드리프트 0.

### 3.3 ik_llama.cpp의 Q4_K×q8 GEMV 구조 (박스 /home/user/ik_llama.cpp 직독)

- 진입: `ggml/src/ggml-cuda/mmvq.cu`의 `ggml_cuda_op_mul_mat_vec_q_impl`이 타입 디스패치 — `case GGML_TYPE_Q4_K: mul_mat_vec_q4_K_q8_1_cuda(args, stream)` (73–75행). 인자 구조체 `mmvq_args`가 포인터·열 수·행 범위·bias·unary_op를 한 번에 넘긴다.
- 커널: `mmvq-templates.cuh`의 `template<ggml_type type, int ncols_y, int nwarps>` — CUDA 블록이 행 묶음을 맡고, `tid = WARP_SIZE*threadIdx.y + threadIdx.x`인 워프가 K 방향 슈퍼블록을 `blocks_per_iter = vdr*nwarps*WARP_SIZE/qi` 스트라이드로 순회하며 `vec_dot_q_cuda`(dp4a 기반 q4_K×q8_1 블록 내적)를 열(j)·행(i) 이중으로 누산, 워프 간 합은 `__shared__ float tmp_shared` 리덕션, bias는 블록 첫 워프가 로드. (mmvq-templates.cuh 68–148행)
- 즉 구조는 "워프 협력 K-스트라이드 + shared 리덕션"의 고전적 mmvq 형태 — bloomery의 q4k_gemv(워프당 행 1, 슈퍼블록 4/이터레이션, B 체인 lane-local)가 이미 이것과 같은 무게급에서 1.06배를 냈다(2.2). ik가 V2-Lite 디코드에서 쓰는 저수준 경로는 이 mmvq보다 `iqk_mul_mat` 계열이지만 GPU GEMV 비교 기준은 이 mmvq다(measure.sh의 q3k_ref.cpp가 ggml을 링크해 재는 근거).

---

## 4. 종합

### 4.1 Q4_K GEMV 스파이크 — 권고 (한 단락)

**판정 근거는 이미 한쪽으로 기울어 있다: 1-5 GPU 라운드는 cuda-oxide로 가고, cutile-rs는 MUL-6(툴킷 13.2/13.3) 뒤에 붙이는 별도 스파이크로만 남긴다.** 이유 셋. 첫째, "Q4_K GEMV 스파이크가 판정한다"의 절반은 이미 치러졌다 — cuda-oxide 트랙의 q4k_gemv가 같은 카드·같은 하네스에서 ggml mmvq의 1.06배(809.8 GB/s)로 게이트를 통과했고(MUL-9), 툴체인(13.0 툴킷, cargo-oxide 0.2.1, LLVM 21.1.8 llc, nightly-2026-08-28, 핀=업스트림 HEAD)가 오늘 박스에서 완비돼 있어 추가 설치 0으로 시작한다. 둘째, cutile-rs는 Q4_K 스파이크를 하려면 툴킷 13.2+ 설치가 먼저고(sm_8x 요건, 현재 13.0), weight-only 양자화 포맷이 전무해 디퀀트+dp4a 산술을 전부 손으로 짜야 하며, 제공 추상이 GEMM·elementwise 중심이라 GEMV(대역폭 바운드, M=1)에 대한 차별화가 없다 — 어디까지나 3090(sm_86)에서는. 셋째, cutile의 진짜 지렛대는 block-scaled MMA(NVFP4 등, 13.3 필요)인데 그 티어는 Blackwell/Hopper 이야기이고 3090엔 해당 텐서코어가 없다. cutile을 지금 채택할 근거는 "미래 카드를 위한 학습"뿐이고, 그 목적이라면 cuda-oxide와 호스트 런타임(cuda-core)을 공유하므로 나중에 붙여도 버리는 것이 없다(동일 cuda-core 0.3.1 위에서 SIMT 인터롭 공식 지원).

### 4.2 리스크

- **cuda-oxide**: early-alpha — `#[unroll]` ICE를 실제로 만났다(MUL-7, 업스트림 미보고, 초안 `docs/upstream/cuda-oxide-unroll-ice.md`). 핀 HEAD가 언롤 코드젠 수정(#1293)이라 회귀 가능성과 개선 가능성이 함께 있다. nightly 전용·rev 핀 필수(Cargo.toml 주석의 이유 그대로: "떠 있으면 어제의 수치와 오늘의 수치가 다른 컴파일러에서 나온다"). M=8이 Q4_K에서 0.67배인 구조적 한계(B 체인)는 트랙 선택과 무관.
- **cutile-rs**: research-stage(API breakage 예고, 0.4.0 breaking already slated), 런타임 JIT — 첫 런치에 컴파일이 끼므로 측정 규약(wait_gpu·증인·20 warmup)에서 첫 워밍업이 커널 캐시를 채우는 것을 명시적으로 분리해야 한다. 레지스트리 캐시 부재 → 첫 빌드 네트워크 필요. dp4a급 정수 산술의 DSL 표현 가능성 미확인 — 이것이 스파이크의 실제 판정 문제가 될 것.
- **공통**: 둘 다 Apache-2.0, 둘 다 Q4_K를 내장하지 않는다 — 양자화 커널은 어느 트랙이든 우리 코드다(이미 있다: cuda-oxide 쪽).

### 4.3 첫 GPU 라운드 체크리스트 (3090만, A6000 금지 — `CUDA_VISIBLE_DEVICES`·measure.sh가 이미 3090 UUID로 고정)

1. **설치: 없음.** 드라이버 615.71.09(≥R580)·툴킷 13.0·cargo-oxide 0.2.1·llc 21.1.8·nightly-2026-08-28 전부 확인됨. 설치가 필요한 유일한 시나리오는 cutile 병행 시 툴킷 13.2+/13.3(디스크 991G 가용, `bloomery-env.sh`의 툴킷 경로 갱신 필요 — 지금은 13.0 하드코딩).
2. **빌드**: `just build-gpu` (box.sh `cargo oxide build --arch sm_86 -- -p bloomery-q3k-gemv`) — rsync 트리 /root/repo/bloomery에서. 빌드 게이트는 이미 `just gate`에 포함(fmt-check lint build-gpu build-cpu gate-1-1).
3. **참조 세팅**: `tools/ref/build.sh`(ik c10fbbcc의 libggml 링크, 산출물은 $BLOOMERY_DATA/bin — 이미 14개 바이너리 있음). ik 빌드가 바뀌면 오라클 재생성(oracle 규약).
4. **측정 하네스**: `tools/ref/measure.sh` 그대로 — 3090 UUID `GPU-307fa0f6…` 고정, `wait_gpu` 유휴 대기, 전후 증인 블록, ggml 먼저 Rust 나중 같은 호출. 오차 게이트 ≤ 1e-2(q8_1 설계, MUL-9 전례), 대역폭 게이트 stack M=1 ≥ 0.9배 ggml.
5. **본라운드(1-5 GPU) 범위**: q3k-gemv(이제 q4k·q6k 포함)를 엔진 배선에 붙이는 것 — plan.md 게이트 "같은 logits, 3090에서 ik 대비 tok/s". CPU 1-5 종료 시점 스테이지 표가 스텝의 머리를 더 이상 양자화 사이트가 아니라 배칭 사이트 대역폭으로 지목했으므로(2026-09-20 MUL-34), GPU 라운드의 첫 질문은 "q3k-gemv 런치를 디코드 루프에 넣어 ik 대비 tok/s가 몇인가"다.
6. **cutile 스파이크(선택, MUL-6 뒤)**: 툴킷 13.3 설치 → 별도 워크트리 + `CUDA_TOOLKIT_PATH`만 바꾼 env → cutile 0.3.1로 Q4_K 타일 커널 시도 → 같은 measure.sh 하네스로 q4k_gemv(1.06배)와 대조. 벤더링 또는 네트워크 필요(레지스트리 캐시 없음). 판정 기준: 같은 게이트 통과 여부 + 코드량/복잡도.
7. **위생**: 핀 rev는 그대로(=HEAD, 드리프트 0). cutile 도입 시에도 cuda-core는 공유되므로 Cargo.toml 핀 구조는 유지. A6000은 어떤 명령으로도 건드리지 않는다(UUID 고정이 기계적으로 방어).

---

### 부록: 이 문서의 출처 목록

- 박스: `tools/box.sh`로 실행한 `nvidia-smi`, `nvcc --version`, `ls /usr/local/`, `which cargo-oxide && cargo-oxide --version`, `df -h /root`, `free -g`, `cat /root/bloomery-env.sh`, `ls /root/bloomery-data/bin`, `find /root -maxdepth 2 -name bloomery-env.sh`, `llc --version`, `rustc --version`, `cat /root/repo/bloomery/...`, `/home/user/ik_llama.cpp`의 mmvq.cu·mmvq-templates.cuh·git log
- Mac: `/Users/hckim/repo/bloomery/Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `justfile`, `crates/q3k-gemv/{Cargo.toml,README.md,src/main.rs,RESULTS-q4k-q6k.md}`, `tools/ref/{measure.sh,build.sh}`, `docs/plan.md`, `git log`
- rig-log: `log/2026-09-19-a-first-rust-kernel-two-arms.md`, `log/2026-09-19-g-the-fused-kernel-runs.md`, `log/2026-09-17-a-rust-engine-on-the-same-card.md` (grep 'gpu|sm_86|3090')
- 웹: github.com/NVlabs/cutile-rs (README·CHANGELOG), api.github.com/repos/NVlabs/cutile-rs/commits, crates.io/api/v1/crates/cutile, github.com/NVlabs/cuda-oxide (README), api.github.com/repos/NVlabs/cuda-oxide/commits(+핀 rev 조회)
