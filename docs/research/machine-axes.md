# Machine axes — GPU generation, AVX-512, DGX Spark, and the machine description

The memo of round `machineaxes` (rebuild wave 3, research only, 2026-09-27), kept as it reported, under the lead's
dispositions. It answers `docs/rebuild.md` §2-3's two "check first" questions and sizes the three axes the user asked
about. Every number in the memo is derived unless it names its measured source; the NVIDIA figures were read through
WebFetch summaries (the lead did not re-read the pages; the values match what the lead knows of GB10 — cc 12.1, 128 GB
LPDDR5x at 273 GB/s, 20 Arm cores — and are to be re-read from the pages before any of them is written into `docs/facts.md`).

## Lead's dispositions (2026-09-27)

- **M1 GPU generation is a key, not a branch (S).** Accepted: one owner for the build arch (the 81 justfile lines and
  `tools/gate.sh:24` fold into the box env's `CUDA_OXIDE_TARGET` or `.cargo/cuda-oxide.toml`'s `default-arch`);
  `tools/ref/ptx-shapes.tsv` gets `*` rows plus per-arch exception rows keyed `(arch)` for spill and `(card arch, driver)`
  for jit_local, `ptx-spill-check.sh` reading `PTX_SCAN_ARCH`; occupancy constants leave `ptx-scan.sh` and are read from
  the driver attributes where the JIT already runs. Round `archkey` (S, tools + one `.tsv` migration; after the Qwen3.6
  chain — nothing in the release needs it). The two refuting checks (§5-2 spill at `sm_121`/`sm_89`/`sm_120` by ptxas
  alone; §5-3 the `.rn`-less mul→add pairs counted in the extracted PTX) need no card and go into that round's spec.
- **M2 AVX-512: skip; the ISA refusal is not optional (S, before the public release).** The memo's derived gain is 0 on
  this box and −0…−16 % on the prefill union on a VNNI host — not a round. But §2's finding that `is_x86_feature_detected!`
  folds to `true` under `target-cpu=znver3`, so the scalar fallback and the named refusal are dead code and a non-AVX2 CPU
  gets SIGILL, is a silent failure: round `isarefuse` (S) puts one attribute-free CPUID check beside `at_main` in every
  binary (AVX2, FMA, F16C, BMI2 against the build's `cfg!(target_feature)` set) and turns `ops.rs:3571`'s
  `UnsupportedType` into an ISA variant. It touches every bin's first lines, so it goes after train 2 (levers2) lands,
  bundled with `fixup7`.
- **M3 DGX Spark is a different machine, M (S×4 + M×3), most of it keys.** Accepted as the memo sizes it. The four S
  rounds (S1 cfg guards on the x86 bodies + a named refusal of host-tier types on non-x86; S2 the `extra-rustflags`
  per-triple key as a fork patch + `check-rustflags` generalized; S3 the arch owner = M1's round; S4 N-card tools) need
  no hardware and can go in any wave-4/5 gap as `sparkprep`; the three M rounds (aarch64 toolchain, the unified-pool
  placement invariant + page-cache drop + Spark instance, the `sm_121` gate pass + ptx-shapes exception rows) wait for
  the machine. The memo's timeline stands: on unified memory the wall is bytes / 273 GB/s, host-card asynchrony is worth
  0, and Spark's value is the 128 GB, not bandwidth — V4.1 Q3 does not fit.
- **M4 the machine description extends `placement::Machine` and `workstation.rs`; no parallel type.** Accepted: the
  `MachineSpec { cards: Vec<CardSpec>, host: HostSpec, unified: Option<UnifiedPool> }` draft becomes the shape when
  `sparkprep` S4 or `archkey` first needs a second card or arch row; `workstation.rs` is this box's instance; the table
  in §4 (decision → field → today's source) is the migration list.
- **M5 the fork defect (§5-9) goes into `docs/upstream/nvlabs-ledger.md` now** (row 32): `extra-rustflags` is a flat list
  with no per-triple key, and `CARGO_ENCODED_RUSTFLAGS` masks cargo's own `target.<triple>.rustflags`.
- **§6 spots** are disposed in `docs/plan-triage.md` (train 2's landing record): the qdot dead-code check and `ops.rs:3571`
  → `isarefuse`; `check-rustflags.sh`'s `znver3`, `ptx-scan.sh`'s `blocks()` and "both GA102", `gate.sh:24`, `gate_p8`'s
  82-SM literal → `archkey`; `constants.tsv:8`'s "measured" label → XS docs; the heterogeneous-core split → held until a
  machine needs it.

## Round memo

메모를 작성합니다. 박스에서는 읽기 명령만 썼고 빌드·실행·임대는 없었습니다.

# machineaxes 메모: 기계 축(GPU 세대·AVX-512·DGX Spark)과 기계 서술

코드를 읽고 NVIDIA 문서 몇 개를 fetch해서 답했습니다. 박스에서는 읽기 명령만 썼습니다(`nvidia-smi --query-gpu`, ptxas 바이너리 문자열 `grep`, `/proc/cpuinfo`·env 파일 `grep`). 빌드, 바이너리 실행, 임대, 게이트는 하나도 하지 않았습니다. 트리는 `/Users/hckim/repo/bloomery-machineaxes`(`e2ecc4a`), 포크는 핀 `e589793a`를 `git grep`/`git show`로 읽었고 체크아웃은 하지 않았습니다.

## §0 요약

- **확인 1: 우리 백엔드가 낼 수 있는 sm.** 포크 핀의 대상 표가 `sm_70`부터 `sm_121`까지 담고 있습니다. GB10의 `sm_121`은 PTX 8.8 바닥으로 들어 있습니다(`crates/cuda-target-spec/src/lib.rs:173-330`, `capability: 121`은 245행).
  - `sm_86`을 박는 곳은 justfile의 `cargo oxide build --arch sm_86` 81줄과 `tools/gate.sh:24`입니다. 도구 기본값도 둘 있습니다(`tools/ptx-scan.sh:103`, `tools/sass-scan.sh:59`).
  - 오늘 바이너리는 `.target sm_86` PTX를 싣고 드라이버가 적재 때 JIT합니다. 그래서 다시 빌드하지 않아도 cc 8.6 이상 카드에서 돕니다(`cuda-host/src/embedded.rs:54`).
- **확인 2: 세대에 묶인 핀.** 비트·밴드 게이트는 구조상 세대와 무관합니다. 예외 후보는 둘이고, 둘 다 새 카드에서 한 번 돌려 봐야 닫힙니다(§5).
  - ptxas가 `.rn` 없는 `mul`/`add`를 합칠 가능성.
  - SFU 근사 명령(`ex2.approx` 등).
  - 반면 `ptx-shapes.tsv`의 spill·jit_local과 ptx-scan의 점유율 상수는 세대마다 값이 다릅니다. 이것들은 **표의 키**로 처리하면 됩니다.

| 축 | 판정 | 크기 | 키인가 분기인가 |
|---|---|---|---|
| GPU 세대(Ada·Hopper·Blackwell) | 새 코드 분기 없음. arch 주인 하나, ptx-shapes `arch` 키, 점유율 상수를 드라이버에서 읽기 | S | 키. 분기는 wgmma·TMA·FP4 같은 성능 선택을 할 때만 생기고, 그것도 인스턴스 표의 행입니다 |
| AVX-512 | 이 박스에서 이득은 정확히 0입니다(명령이 없음). VNNI 호스트에서도 프리필 union −0…−16 %가 상한입니다[유도]. **건너뛰고 거부만 넣습니다** | S(거부만) | 거부는 키. 커널 변형은 분기라 하지 않습니다 |
| DGX Spark(GB10) | 다른 기계입니다. x86 import 하나가 aarch64 빌드 전체를 막습니다. 통합 메모리 배치 불변식 하나, 툴체인 한 번이 필요합니다. 새 커널 계열은 없습니다 | M(S 넷 + M 셋) | 대부분 키. 코드는 cfg 가드, 배치 불변식, 도구의 N-카드 일반화 |

## 종이 위 분해와 자원 시간선

**분해.** 질문을 코드에서 셀 수 있는 항으로 나눴습니다.

- 빌드가 arch를 받는 자리: 81 + 1줄. 도구 env 기본값 2곳.
- `ptx-shapes.tsv`: 217행 중 0이 아닌 핀은 1행(`ds41_attn_seg_stage` 8/8).
- `launch_bounds` 225곳. 전부 레지스터 64K/SM 기준 상한이라 8.x, 9.0, 12.x에서 같은 캡입니다.
- 동적 smem 5개 런치. 크기는 56,064 B와 49,920 B이고 둘 다 블록당 99 KB 안입니다.
- mma·ldmatrix·cp.async는 sm_80 이상 명령뿐입니다.
- 호스트 ISA: qdot의 `#[target_feature]` 본문 22개, `deepseek2/attn.rs` 8개, `markov-accept` bin 1개.
- aarch64 빌드를 막는 무조건 import 1줄.

읽기로 닫히지 않는 항은 "`sm_121`에서 비트와 spill이 같은가" 하나입니다. spill 열은 박스의 x86 ptxas로 카드 없이 나오고, 비트는 카드가 있어야 나옵니다.

**Spark 시간선(토큰 하나 디코드, 카드 바운드 모델).**
- 호스트와 카드가 같은 LPDDR5x 한 자원(273 GB/s)을 씁니다. 그래서 벽시계는 바이트/273에 런치 오버헤드를 더한 값이고, 호스트와 카드를 겹치는 비동기 레버는 0입니다.
- 같은 바이트를 A6000(674 GB/s, `tools/flow/constants.tsv:10`) 대비 **2.47배 이상 오래** 읽습니다[유도, 대역폭 비].
- Spark의 가치는 대역폭이 아니라 128 GB 용량입니다. V4.1 Q3(약 300 GB)는 들어가지 않습니다(rebuild.md §2-3의 유도).

**AVX-512 시간선(V4.1 P = 512, 층-배치 하나).**
- 호스트 union 66.7 ms(uniondispatch), 카드 경로 19.6 ms(B1, AGENTS 기록)입니다. 벽시계는 max이고 호스트가 그 max입니다. 그래서 union에서 얻는 이득은 프리필에 그대로 전달됩니다.
- 디코드의 호스트 티어는 DRAM 바운드라 명령을 줄여도 0입니다(`docs/facts.md:64`의 바닥, rebuild.md §2-3의 h3tile Δ 0).

## §1 GPU 세대

**`--arch`가 흐르는 길.** 우선순위는 `--arch` > env `CUDA_OXIDE_TARGET` > `.cargo/cuda-oxide.toml`의 `default-arch` 순입니다(`cargo-oxide/src/commands/build_run.rs:48-50`, `codegen_env.rs:330-371`, `context.rs:275`).

- 값은 `CUDA_OXIDE_TARGET`으로 백엔드에 가고, 백엔드는 그 값으로 `.target` PTX만 냅니다(`cuda-oxide-codegen/src/options.rs:102`).
- SASS는 우리 트리에서 드라이버 JIT가 만듭니다. `ptx-scan`은 그 PTX를 ptxas `-arch=$ARCH`에 따로 넣어 열을 읽습니다(`tools/ptx-scan.sh:159`).
- `CUDA_OXIDE_TARGET`은 cargo-oxide 지문에 들어가서 arch를 바꾸면 다시 빌드합니다(`fingerprint.rs:104-107`).
- 백엔드 `.so` 하나가 표의 모든 sm을 냅니다. rev 디렉터리(`~/.cargo/cuda-oxide-bloomery/<rev>/`)에 arch 축은 없습니다.

**한 바이너리에 sm 이미지 둘?** 안 됩니다. `load_embedded_module`은 이름으로 번들 하나를 찾고, 번들은 `target` 하나에 payload를 종류별로 하나씩만 가집니다(`cuda-host/src/embedded.rs:56-65`, `load_bundle` 161행).
- 필요도 없습니다. sm_86 PTX는 더 높은 cc에서 JIT로 돕니다.
- 네이티브 빌드는 spill·성능·명령 선택을 위한 것이고, arch마다 `target/` 디렉터리가 따로 있어야 합니다. 러너는 `target/release`만 읽고 폴백하지 않습니다.

**세대별 사실 목록**

| 자리 | 무엇 | 분류 |
|---|---|---|
| `justfile`(81줄), `tools/gate.sh:24` | `--arch sm_86` | 키. 주인 하나로 접습니다. 박스 env의 `CUDA_OXIDE_TARGET` 또는 `default-arch` |
| `tools/ptx-scan.sh:103`, `tools/sass-scan.sh:59` | `ARCH` 기본 sm_86 | 키(이미 env 덮어쓰기가 있음) |
| `tools/ptx-scan.sh:81-86`, `221-236` | `blocks()` 상수: 65536 레지스터, 102400 B, 48 warp, 16 블록 | 키. 그런데 `ARCH`를 바꿔도 이 상수는 sm_86 그대로라서 blk/SM 열이 틀린 값을 냅니다. 드라이버 속성에서 읽게 제안합니다(아래) |
| `tools/ptx-scan.sh:192-194` | JIT 카드 `any` 가정("둘 다 GA102") | 키. 세대가 섞인 박스에서는 JIT 카드가 arch로 정해져야 합니다 |
| `tools/ref/ptx-shapes.tsv:6`(헤더), 217행 | spill(ptxas 13.3.73, sm_86), jit_local(3090 JIT, CUDA 13.4) | 열 추가(키), 아래 제안 |
| 225개 `#[launch_bounds]`, 예: `gpu/src/gemm/kernels.rs:354`(256,2), `gpu/src/q4k_sel.rs:390`와 `gpu-deepseek41/src/chain/ffn/batch.rs:313`(256,5) | 레지스터 캡 = 64K / (스레드 × 최소 블록) | 세대 무관. 64K 레지스터는 8.x/9.0/12.x 공통입니다(8.6은 ptx-scan 주석과 교차 확인). 5블록 × 8 warp = 40 ≤ 48 |
| `gpu/src/flash.rs:1078`(56,064 B), `gpu-deepseek41/src/attn.rs:371,437,515,585`(49,920 B) | 48 KB를 넘는 동적 smem | 세대 무관. 블록당 99 KB 안입니다. 12.x의 한도는 §5 |
| `gpu/src/gemm/{kernels,grouped}.rs`, `flash*.rs`, `gpu-deepseek41/src/{attn,indexer}.rs`, `gpu-vision/src/*` | `mma_m16n8k16_{f16,bf16,s8}`, `m16n8k32_s8`, `ldmatrix_x4`, `cp_async_cg_16` | 세대 무관. 백엔드 `arch_satisfies_feature`에서 Sm80/Ldmatrix는 cc ≥ 80/75로 앞으로 호환입니다(`cuda-oxide-codegen/src/target/arch.rs:53-57`) |
| `gpu-deepseek41/src/indexer.rs:1024` | 격자 = `multiprocessor_count()` | 이미 런타임에 기계를 읽음 |
| `gpu-gates/src/bin/gate_p8.rs:1026-1030` | 격자 목록 `[16, 22, 82, 164]`(3090의 82 SM) | 키(탐침 전용이고 비트 게이트 아님). §6 |
| `tools/flow/constants.tsv:8,10,11-12` | `n_sm` 84, `bw_card` 674, PCIe 26.28/21.16 | 키. 흐름 모델 상수를 기계별 행으로 |
| `docs/plan.md:65-80`(비용 모델), `:141-147`(타겟 프로필), `docs/facts.md:38,66,68` | c_node, BW 567–701, BW_host, PCIe, "sm_86 IMMA·FP8 없음" | 기계별 산문. 서술의 인스턴스로 옮길 사실들 |
| `tools/ref/timing-card.sh:21`, `tools/ref/cards.sh:13-14`, `tools/gpu-gate.sh:39-95`, `tools/box.sh:87-101` | 카드 둘의 UUID, 타이밍 카드, 카드마다 게이트 락 | 키. 카드 목록을 N개로 일반화(도구 코드 S) |
| 부동소수 원자 연산 | 장치 크레이트에 없음(`grep` 0) | 세대 무관 |

**다시 핀할 게이트.**
- 비트·밴드 게이트는 오라클 파일과 대조하므로 핀이 세대에 묶이지 않습니다.
- 명령 흐름이 같으면(JIT 경로) 바뀔 수 있는 것은 두 부류뿐입니다.
  - ptxas가 `.rn` 없는 곱셈·덧셈을 수축하는 경우. LLVM은 `-fp-contract=fast`로 이미 `fma.rn.f32`를 PTX에 적고 나머지 `mul.f32`를 남깁니다(`docs/upstream/nvlabs-ledger.md` 17행). 적재 경로가 무게를 싣는 곳은 `_rn` 내장이나 명시 fma로 고정돼 있습니다(`gpu/src/rope_neox.rs:54`, `gpu-deepseek41/src/hc.rs:24-29`).
  - SFU 근사. 장치 크레이트에 `.exp()` 21곳, `.ln()` 5곳, `rsqrt` 2곳, `ex2_approx` 2곳이 있습니다.
- 네이티브 `--arch sm_121` 빌드는 LLVM이 다른 명령을 고를 수 있으니, 비트 게이트 전체를 한 번 도는 것이 증명입니다.
- spill·regs·jit_local은 반드시 세대마다 다시 핀합니다.

**`ptx-shapes.tsv` 제안.** 키는 `arch` 하나로 부족합니다. spill은 (arch, ptxas 버전)에, jit_local은 (카드 arch, 드라이버)에 묶입니다(헤더 6행, `ptx-scan.sh:99-101`의 "toolkits differ here by a register").

```
# ptxas=13.3.73  jit[sm_86]=RTX 3090 / CUDA 13.4
# binary        arch    entry                 spill  jit_local
generate_ds41   *       add                       0          0
# PIN(…): …
generate_ds41   sm_86   ds41_attn_seg_stage       8          8
generate_ds41   *       ds41_attn_seg_stage       0          0
```

- `*` 행 217개(엔트리당 하나)와 arch별 예외 행으로 둡니다. 표가 N배로 늘지 않고, "새 엔트리는 행이 없으면 실패"라는 의미도 그대로입니다.
- `ptx-spill-check.sh`는 `PTX_SCAN_ARCH`를 받아 그 arch 행이 있으면 그것을, 없으면 `*`를 씁니다.
- 새 카드가 없는 arch는 `jit_local`을 읽을 수 없습니다(JIT는 박스 카드의 것). 그 경우 검사는 그 열을 `n/a`로 찍고 spill만 판정해야 합니다.
- 점유율 상수는 표로 박지 말고, `oxart_jit`가 이미 카드 위에서 돌 때 드라이버 속성을 읽어 banner에 싣게 합니다. 대상은 `MAX_BLOCKS_PER_MULTIPROCESSOR`, `MAX_THREADS_PER_MULTIPROCESSOR`, `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`, `MAX_REGISTERS_PER_MULTIPROCESSOR`입니다. 그러면 세대는 표의 키가 아니라 측정값이 됩니다.

## §2 AVX-512와 다른 x86 ISA 수준

**목록.**
- `crates/qdot/src/lib.rs`: `#[target_feature]` 본문 22개(avx2+fma+f16c, Q3_K는 avx2+f16c, MXFP4는 avx2+fma). 유형별 표는 `has_features`(166-188행)입니다.
- `crates/model/src/arch/deepseek2/attn.rs`: 8개(V2-Lite CPU 경로).
- `crates/model/src/bin/markov-accept.rs`: 1개.
- 빌드: `.cargo/config.toml:7-8`(`[target.x86_64-unknown-linux-gnu]` 키가 있음), `.cargo/cuda-oxide.toml:5`(키 없는 평면 목록).
- `tools/check-rustflags.sh:24`가 `znver3`를 반드시 요구합니다.

**중요한 발견: 런타임 검사는 박스 빌드에서 죽은 코드입니다.**
- 고정 툴체인의 `std_detect/src/detect/macros.rs:8-10`이 `is_x86_feature_detected!`를 `cfg!(target_feature = …) || __is_feature_detected::…()`로 펼칩니다.
- 그래서 `target-cpu=znver3` 아래에서는 다음이 전부 컴파일 시간 `true`입니다(cargo oxide 쪽도 `extra-rustflags`로 같음).
  - `qdot::has_features`(`lib.rs:170-172`)
  - `quantize_col`의 분기(`lib.rs:241`)
  - `deepseek2/attn.rs:960-962,1093-1095`
- 결과: AVX2가 없는 CPU에서 이름 붙은 거부(`model/src/ops.rs:3571`)는 발동하지 않습니다. 스칼라 폴백(`ops.rs:751`)도 죽은 코드이고, 그 스칼라 미러 자체가 znver3로 컴파일돼 AVX2 명령을 담을 수 있습니다. **첫 결과는 SIGILL**이고, "조용한 실패 금지" 조항 위반입니다.

**최소 조치(거부만, S).** `main` 첫 줄(`bloomery_levers::at_main` 옆)에 검사 하나를 둡니다.
- 속성 없는 함수가 CPUID leaf 1 ECX(FMA·F16C)와 leaf 7 EBX(AVX2·BMI2)를 읽습니다. `/proc/cpuinfo` flags를 읽어도 됩니다.
- 그 결과를 빌드의 `cfg!(target_feature)` 집합과 대조하고, 부족하면 "이 바이너리는 znver3(avx2,fma,f16c,bmi2)용이고 이 CPU에는 X가 없다"로 멈춥니다.
- std는 기본 x86-64로 미리 컴파일돼 있어 `main` 이전 코드는 AVX2를 쓰지 않습니다[유도].
- `ops.rs:3571`이 ISA 부족을 `UnsupportedType`으로 부르는 것은 §6에 넣었습니다.

**AVX-512를 넣는다면 자리.** qdot 커널마다 두 번째 `#[target_feature]` 본문이 필요합니다. 본문을 헬퍼로 쪼개지 말라는 규칙 때문에 커널 × ISA 곱셈이 됩니다. 디스패치는 이미 있는 유형별 표(`has_features`)를 ISA 수준 키로 바꾸면 되므로 키입니다.

**종이 위 이득.**
- 이 박스(Zen 3)에는 AVX-512도 VNNI도 없으니 **정확히 0**입니다(`docs/facts.md:38`, 박스 `/proc/cpuinfo`에 `avx512*` 0건).
- 다른 호스트에서도 폭(512비트)은 핵심이 아닙니다. 핵심은 `vpdpbusd`(VNNI)이고, AVX-VNNI(VEX-256)로 AVX-512 없이도 있습니다.
- 디코드: DRAM 바운드라 0입니다.
- 프리필 union: 호스트 union이 벽시계를 정하므로 이득이 그대로 전달됩니다. 명령 수 × IPC로 유도합니다.
  - `cpugemm-lit-report.md` §0의 h1fold Q3_K는 열·슈퍼블록당 약 30.6 + 99/8 ≈ 43 명령이고 IPC ≈ 3입니다.
  - `vpdpbusd`는 maddubs+madd 쌍 8개를 대신합니다. 하지만 16값마다 붙는 스케일이 정수 곱(`vpmulld`)으로 남습니다. 그래서 순 절감은 0…8 명령(0…19 %)입니다[유도, 상한].
  - c_union 22.2 µs(`constants.tsv:50`)는 ≥ 18.0 µs가 됩니다. m̄ = 7.6에서 t = a + c·m은 198.9에서 ≥ 167.0 µs가 되고(a 30.2, `:49`), 여전히 W 126.4보다 큽니다. 곧 **union −0…−16 %**[유도]입니다.
  - 비트 동일은 AVX2 경로가 i16을 포화시키지 않는 한도(`cpugemm-lit-report.md` Q4 표)에서 나옵니다. 곧 "integer-path reorder" 증명 부류입니다.
- **권고: 건너뛰고 거부만 넣습니다.** VNNI 호스트로 기계를 바꿀 때 Q3_K r8 타일과 Q4_K down 둘만 다시 유도합니다.

## §3 DGX Spark(GB10)

**fetch한 사양**
- https://www.nvidia.com/en-us/products/workstations/dgx-spark/ :
  - "Blackwell Architecture"
  - "20-core Arm, 10 Cortex-X925 + 10 Cortex-A725 Arm"
  - "128 GB LPDDR5x, coherent unified system memory"
  - "273 GB/s"
  - "NVIDIA DGX™ OS"
- https://docs.nvidia.com/dgx/dgx-spark/hardware.html : "6,144" CUDA 코어.
  - SM 48 = 6,144 / 128은 [유도]이고, SM당 FP32 코어 128을 가정합니다.
- https://developer.nvidia.com/cuda-gpus : "12.1 NVIDIA GB10 (DGX Spark)", "8.6 NVIDIA RTX A6000".
- 인용 신뢰도 한 줄: 위 인용은 WebFetch의 요약 모델이 돌려준 문장이라, 원문 대조는 리드가 한 번 하는 것이 좋습니다. 같은 도구로 읽은 compute-capabilities 표는 두 번 다 열이 어긋나서 인용하지 않았습니다(§5).
- 툴체인 근거: `sm_121`은 포크 대상 표에 있고, 박스 ptxas 13.0/13.3 바이너리에 `sm_121` 문자열이 있습니다(`grep -c` 2/1건, 약한 증거).

**(a) x86 가정**

| 자리 | 사실 | Spark에서 |
|---|---|---|
| `crates/qdot/src/lib.rs:28` | 무조건 `use std::arch::x86_64::*;` | **aarch64에서 컴파일되지 않습니다.** `gpu`, `gpu-deepseek41`, `gpu-gates`가 모두 `bloomery-model` → `bloomery-qdot`에 의존하므로(`crates/*/Cargo.toml`) 엔진 전체가 막힙니다 |
| `model/src/arch/deepseek2/attn.rs:779-3139`(x86 8곳), `model/src/bin/markov-accept.rs` | x86 intrinsics | `cfg(target_arch)` 가드 |
| `.cargo/cuda-oxide.toml:5` | 트리플 키 없는 `znver3` | aarch64 rustc에 그대로 넘어갑니다. 포크 결함 후보(§5의 ledger 줄) |
| `.cargo/config.toml:7` | 트리플 키가 있음 | aarch64에는 적용되지 않음(이미 키) |
| `threads/src/lib.rs:599-601`(affinity), `model/src/placement/host_lock.rs`(mlock·madvise·fadvise·mincore), `model/src/r8file.rs:607-637`(sync_file_range·fadvise), `model/src/fileio.rs:21-42`, `engram/src/prefetch.rs` | Linux libc | DGX OS도 Linux라 그대로 됩니다[유도] |
| `threads/src/lib.rs:531-560` | sysfs L3로 토폴로지 감지, 동일 코어 가정 | X925/A725 이종 코어에서는 같은 크기 청크가 느린 코어를 기다립니다. 호스트 계산이 없으면 무관 |
| 박스 env `CUDA_OXIDE_LLC=…/LLVM-21.1.8-Linux-X64/bin/llc` | x86 LLVM | aarch64 LLVM 21 필요 |

**(b) 호스트 티어가 카드와 따로 있다는 가정**
- `placement::Machine { cards, host }`(`placement.rs:199`)는 카드와 호스트 바이트를 따로 검사합니다. 통합 메모리에서는 **Σ 카드 + 호스트 ≤ 풀** 불변식 하나가 필요합니다.
- 파일 mmap의 page cache와 장치 사본이 같은 DRAM에 이중으로 올라갑니다. 적재 뒤에 `host_lock.rs:629-652`의 `MADV_DONTNEED`/`POSIX_FADV_DONTNEED`를 재사용해 버립니다.
- `workstation.rs:33-44,59-61`의 상수, `plan_a/plan_b/plan_gate`(105-136행), `generate.rs:49-50`과 `generate_ds41.rs:235-236`의 `Place`는 박스 이름입니다. Spark에서는 "한 카드에 전부"인 인스턴스 하나가 됩니다.
- `hybrid.rs`(host-mapped 페이지, system-scope 배리어), `HostResidency`(`hybrid.rs:247`), r8 사이드카, 브리지 상수(`plan.md` 브리지 행): 호스트 티어가 비면 쓰이지 않습니다. 코히어런트 메모리에서도 동작 자체는 된다고 봅니다[유도].
- Qwen 경로는 `Gpu::new()` = 디바이스 0이라(`gpu/src/lib.rs:1751`) 이미 기계와 무관합니다.
- 도구 쪽(`BLOOMERY_CARD`, 게이트 락 둘, 타이밍 카드)은 카드 목록 N개로 일반화하면 됩니다.

**(c) 툴체인**
- 포크 설치 문서는 "Linux (x86\_64 or aarch64)"라고 적습니다(`cuda-oxide-book/getting-started/installation.md:77`, `flake.nix:50`). 우리는 aarch64에서 빌드해 본 적이 없습니다.
- 백엔드 `.so`는 호스트마다 rev 디렉터리에 한 번씩 빌드해야 합니다.
- 오늘 sm_86 PTX가 GB10에서 JIT로 도는 경로는 `embedded.rs:54`의 주석이 말하는 그 경로입니다.

**(d) 기계 서술 초안.** 평행 타입을 만들지 않고 기존 주인 `placement::Machine`과 `workstation.rs`를 확장합니다. `workstation.rs`가 이 박스의 인스턴스가 됩니다.

```rust
/// One machine: what placement, kernel-instance selection, the gate runner
/// and the timing runner read. `workstation::this_box()` is the box's
/// instance; `placement::Machine` is derived from it plus a layer map.
pub struct MachineSpec {
    pub cards: Vec<CardSpec>,
    pub host: HostSpec,
    /// Card and host allocate from one DRAM pool (GB10): placement checks
    /// sum(card) + host <= pool.bytes instead of each side alone.
    pub unified: Option<UnifiedPool>,
}
pub struct CardSpec {
    pub name: &'static str,          // matched in the CUDA device name (Gpu::for_card)
    pub uuid: &'static str,          // the tools' pin (cards.sh)
    pub arch: CudaArch,              // build target and instance-table key
    pub sms: u32,                    // checked against the driver at open
    pub total_bytes: u64,
    pub driver_reserve_bytes: u64,
    pub granule_bytes: NonZeroU64,
    pub bw_gbs: f64,                 // measured; cost model only
    pub link: Link,                  // Pcie { gen, lanes, h2d_pinned_gbs } | Unified
    pub roles: CardRoles,            // gate / timing / serve
}
pub struct HostSpec {
    pub isa: HostIsa,                // X86 { avx2, fma, f16c, avx512, vnni } | Aarch64 { neon, sve2, i8mm }
    pub core_classes: Vec<(u32, &'static str)>, // (count, class): uniform on the box, 10+10 on GB10
    pub usable_bytes: u64,
    pub reserves: Vec<(&'static str, u64)>,
    pub dram_gbs: f64,
    pub numa_nodes: u32,
}
pub struct UnifiedPool { pub bytes: u64, pub bw_gbs: f64 }
```

**Spark 이식 라운드**

| 라운드 | 크기 | 표 항목인가 코드인가 |
|---|---|---|
| S1 `qdot`·`deepseek2/attn`·bin의 x86 본문 cfg 가드, 비x86에서 호스트 티어 유형은 이름 붙은 거부 | S | 코드 |
| S2 `extra-rustflags` 트리플 키(포크 패치) + `check-rustflags` 일반화 | S | 포크 코드 + 설정 표 |
| S3 arch 주인 하나(81+1줄 → env/`default-arch`), arch별 `target/` | S | 설정(원장 키 한 번 무효화) |
| S4 도구 N-카드(`cards.sh`, `timing-card.sh`, `gpu-gate.sh` 락) | S | 도구 코드 |
| M1 aarch64 툴체인: 백엔드 `.so`, LLVM 21 llc, CUDA 13 sbsa, `cargo oxide doctor`, `sm_121` vecadd | M | 운영, 하드웨어 필요 |
| M2 배치 통합 풀 불변식 + page cache 버리기 + Spark 인스턴스 | M | 불변식 하나는 코드, 인스턴스는 표 |
| M3 `sm_121` 게이트 전체 한 번 + ptx-shapes 예외 행 | M | 측정과 표 |

## §4 기계 서술과 그것을 읽을 결정

| 결정 | 필드 | 오늘의 출처 | 오늘 이미 런타임에 기계를 읽나 |
|---|---|---|---|
| 장치 코드 빌드 대상 | `card.arch` | `justfile` 81줄, `tools/gate.sh:24` | 아니오 |
| 점유율 열 | `card.arch`(드라이버 속성) | `tools/ptx-scan.sh:81-86,221-236` | 아니오 |
| spill/jit_local 핀 | `card.arch` + ptxas/드라이버 | `tools/ref/ptx-shapes.tsv:6` | 아니오 |
| 카드 바이트 예산 | `total_bytes`, `driver_reserve_bytes`, `granule_bytes` | `model/src/placement/workstation.rs:33-44,48-55` | 아니오(`BLOOMERY_CARD_BUDGET`로 줄이기만 가능) |
| 호스트 바이트 | `host.usable_bytes`, `reserves` | `workstation.rs:59-61` | 아니오 |
| 배치 선택 | `cards`, `roles`, `unified` | `gpu-gates/src/generate.rs:49-50`, `gpu-gates/src/bin/generate_ds41.rs:235-236` | 아니오 |
| 카드를 장치에 잇기 | `card.name` | `gpu/src/lib.rs:1907`(`for_card`), `workstation.rs:140-142` | 예(이름 대조) |
| 타이밍 카드 | `roles.timing`, `uuid` | `tools/ref/timing-card.sh:21`, `tools/ref/cards.sh:13-14` | 아니오 |
| 게이트 락 | `cards[].uuid` | `tools/gpu-gate.sh:39-95`, `tools/box.sh:87-101` | 아니오 |
| 호스트 커널 ISA | `host.isa` | `qdot/src/lib.rs:166-188`, `.cargo/config.toml:8`, `.cargo/cuda-oxide.toml:5` | 사실상 아니오(§2: 컴파일 시간 상수) |
| 스레드 수·토폴로지 | `host.core_classes` | `threads/src/lib.rs:341,531-560` | **예**(sysfs) |
| 격자 크기 | `card.sms` | `gpu-deepseek41/src/indexer.rs:1024` | **예**(`multiprocessor_count`) |
| 탐침 격자 | `card.sms` | `gpu-gates/src/bin/gate_p8.rs:1030` | 아니오(82 고정) |
| 흐름·비용 모델 상수 | `bw_gbs`, `link`, `sms`, `dram_gbs` | `tools/flow/constants.tsv:8,10-12`, `docs/plan.md:65-80` | 아니오 |
| 적재할 장치 | 장치 0 | `gpu/src/lib.rs:1751`(Qwen) | **예** |

## §5 열린 질문과 반박 검사

1. **ptxas가 `sm_121`을 실제로 조립하는가.** 문자열 grep은 약한 증거입니다. 검사는 박스에서 `ptxas -arch=sm_121` 모듈 하나입니다. 예측: 조립됩니다.
2. **다른 세대의 spill.** `PTX_SCAN_ARCH=sm_121`(그리고 sm_89, sm_120)로 `just ptx-scan generate_ds41 --features gpu,deepseek41`를 돌립니다.
   - 예측: 216 엔트리 spill 0, `ds41_attn_seg_stage` 0~16 B.
   - jit_local 열은 3090의 값이라 판정하지 않습니다.
   - 게이트 락을 잡습니다. 반박 조건: spill이 0이 아닌 엔트리가 2개 넘게 나오면 "S"가 틀립니다.
3. **ptxas 수축 노출.** 추출한 PTX에서 `.rn` 없는 `mul.f32` 결과가 `.rn` 없는 `add.f32`로 들어가는 쌍을 텍스트로 셉니다. 카드가 필요 없습니다. 0이면 이 위험은 닫힙니다. PTX ISA 절 fetch는 실패했으므로 "ptxas가 수축할 수 있다"는 인용 없는 의미론입니다.
4. **SFU 근사의 세대 의존.** `ex2/lg2/rsqrt.approx`를 쓰는 엔트리 목록을 만들고, 그 엔트리의 비트 게이트를 새 카드에서 한 번 돕니다.
5. **12.x 점유율 한도.** fetch한 표의 열이 어긋났습니다(9.0이 48 warp·16 블록으로 읽힘). 새 카드 위에서 드라이버 속성으로 읽습니다(§1 제안).
6. **GB10의 `nvidia-smi memory.total`과 `cuMemGetInfo`.** 통합 메모리에서 숫자를 주는지 확인해야 합니다. `CardSpec`이 "nvidia-smi로 잰 total"을 전제하기 때문입니다.
7. **네이티브 `--arch sm_121` 빌드의 PTX md5.** sm_86 빌드와 같은지 봅니다. 박스에서 빌드만 하면 되고 카드는 필요 없습니다. 다르면 JIT 경로와 네이티브 경로 중 하나를 고르는 결정이 생깁니다.
8. **aarch64에서 포크 백엔드 빌드.** 문서는 된다고 적습니다. 하드웨어가 필요합니다.
9. **nvlabs-ledger에 넣을 줄(포크 결함).** "cargo-oxide `extra-rustflags`(`crates/cargo-oxide/src/commands/context.rs:310`) is one flat list with no per-target-triple key; `CARGO_ENCODED_RUSTFLAGS` masks cargo's own `target.<triple>.rustflags`, so a workspace that pins `-C target-cpu` for its x86-64 host passes it to an aarch64 build too. Candidate: `[target.<triple>] extra-rustflags`."

## §6 범위 밖에서 본 개선 지점(보고만)

- `crates/qdot/src/lib.rs:170-172,241`: znver3 빌드에서 런타임 ISA 검사가 상수 참입니다. 폴백과 거부가 죽은 코드이고, AVX2가 없는 CPU에서는 SIGILL이 납니다. 크기 S.
- `crates/model/src/ops.rs:3571`: ISA 부족을 `QdotError::UnsupportedType`으로 부릅니다. ISA 변형을 따로 두는 것이 맞습니다. 크기 S.
- `tools/check-rustflags.sh:24`: 검사가 기계 상수 `znver3`를 하드코딩합니다. 크기 S.
- `tools/ptx-scan.sh:221-236`: `ARCH`는 덮어쓸 수 있는데 `blocks()`는 sm_86 상수라서 blk/SM 열이 틀린 값을 냅니다. 크기 S.
- `tools/ptx-scan.sh:192-194`: "두 카드 모두 GA102"를 가정합니다. 세대가 섞인 박스에서는 JIT 표가 카드에 따라 달라집니다. 크기 S.
- `tools/gate.sh:24`: justfile과 별개인 두 번째 `--arch sm_86` 주인입니다. 크기 S.
- `crates/gpu-gates/src/bin/gate_p8.rs:1030`: 82 SM 격자 리터럴. `multiprocessor_count`를 읽으면 됩니다. 크기 S.
- `tools/flow/constants.tsv:8`: `n_sm`이 "measured"로 적혀 있지만 출처는 설계 보고서이고 값 자체는 사양입니다. 분류를 바로잡는 정도, 크기 XS.
- `crates/threads/src/lib.rs:387-393`: 동일 코어 분배입니다. 이종 코어 기계에서는 가중이 필요합니다. 필요해질 때까지 보류, 크기 M.

## §7 모델

Opus 5.5(opus로 스폰됨)로 돌았습니다.
