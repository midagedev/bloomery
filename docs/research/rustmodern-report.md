# rustmodern — 이 트리가 아직 안 쓰는 모던 Rust 레버, 트리에 대고 확인 (라운드 보고, 2026-09-25)

리드 메모: 파동 25 대기 창의 박스 없는 조사 라운드(opus). 스펙 `specs/wave-m6/spec-rustmodern.md`. 첫 발견(oxide 빌드에
`-C target-cpu=znver3`가 안 걸린다)은 리드가 박스에서 대조했다: `objdump -d target/release/generate_ds41`의 레거시 스칼라 SSE
(mulss·addss·movss·cvtsi2ss) 1,688개 대 `bench_v41_host` 4개(2026-09-25, main 트리). 처분은 `docs/plan-triage.md`(oxcpu·asmscan·
mutpilot·expect·overflow-checks)와 `docs/upstream/nvlabs-ledger.md` §6·§7이다. 아래는 라운드의 보고 원문이다.

`rustmodern` 라운드 보고 (조사만 했고 저장소 파일은 바꾸지 않았다)

---

## 0. 요약

손볼 곳을 찾기 전에 결함이 하나 나왔다. **`cargo oxide build`/`cargo oxide test`로 지은 바이너리는 전부 `-C target-cpu=znver3` 없이 기준 x86-64로 컴파일된다.** V4.1 헤드라인 엔진 `generate_ds41`과 `gate_e2e` 같은 oxide 게이트가 모두 여기에 든다. 경로는 이렇다. cargo-oxide가 `CARGO_ENCODED_RUSTFLAGS`를 설정하고, Cargo 규칙상 그 값이 `.cargo/config.toml`의 `target.<triple>.rustflags`를 가린다. 그래서 `.cargo/config.toml:1-3`의 "비트 게이트 전부가 이 플래그 아래서 돈다"와 `AGENTS.md:414`의 "every box build compiles with AVX2 and FMA on"은 oxide 빌드에 대해 사실이 아니다. 조용한 실패의 한 형태다. 핫 커널은 `#[target_feature]`로 지켜져 있어서 속도 영향은 작다[유도]. 값은 계측의 충실성에 있다. 지금은 벤치(`bench_v41_host`, plain cargo)와 엔진(`generate_ds41`, oxide)이 같은 호스트 코드를 서로 다르게 컴파일한다.

나머지 레버는 대부분 ① 이미 적용돼 있거나(에디션 2024, 캐시라인 패딩), ② 툴체인이 막고 있거나(`assert_unchecked`), ③ 유도한 이득이 자(ruler)의 오차 ±1 % 아래다(PGO, LTO, `std::simd`).

## 1. 순위표

| # | 레버 | 트리의 현재 상태 | 바꾸는 것 | 증명 클래스 | 크기 | 가치/비용 한 줄 |
|---|---|---|---|---|---|---|
| 1 | **oxide 빌드의 호스트 `target-cpu`** | 없음. `codegen_env.rs:288-289`(핀 `b9847e95`)가 `CARGO_ENCODED_RUSTFLAGS`를 설정하고 `RUSTFLAGS` env만 합친다(:93). 트리에 `.cargo/cuda-oxide.toml`이 없다. 실측: oxide 유닛 5/5에 znver3 없음 | `.cargo/cuda-oxide.toml`에 `extra-rustflags = ["-C","target-cpu=znver3"]`를 넣고 두 파일의 동일성을 검사한다 | 이동 클래스: `ptx-scan` 표 동일(`generate_ds41`, `gate_e2e`) + `gate-gpu-e2e` 토큰 동일 + 같은 임대 A/B 1회(pp512 union, tg96 깊이 6), 예측 0…+1 %[유도] | XS | 벤치와 엔진이 같은 코드를 재게 된다. 틀린 계약 문장 두 곳도 바로잡힌다. 비용은 거의 없다 |
| 2 | **호스트 asm 스캔을 상설 도구로**(`ptx-scan`의 호스트 쌍둥이) | 없음. `ops.rs:2345` `tile_units`가 "the disassembly"를 손으로 읽은 값이다. 박스에 `cargo-show-asm`은 없고 `llvm-mca`는 `~/opt/LLVM-21.1.8…/bin`에 있다 | `tools/asm-scan.sh <bin> [fn]`(objdump만 쓴다: 함수별 명령 수, legacy-SSE/VEX, 호출 수) + 선택적으로 `cargo asm --mca` 레시피. oxide 바이너리의 legacy-SSE 수를 래칫 파일에 핀한다 | 빌드 시점 래칫. FAIL-first: 오늘 `generate_ds41` 1778이 실패한다 | S | #1의 재발 방지 층이면서 "명령 수" 클래스의 호스트 계기. 새 의존성이 없다 |
| 3 | **`#[expect(lint, reason)]`** | expect 0. allow 209 = 사유 있음 183(`#!` 2 포함) + **사유 없음 26**(R16 위반: gpu `too_many_arguments` 20 — flash.rs:960 이후 — model 3, gpu-gates `type_complexity` 1, gguf `non_camel_case_types` 1, q3k-cpu `absurd_extreme_comparisons` 1). 195/207이 `too_many_arguments`이고 그중 67은 `#[kernel]`에 붙어 있다 | 반드시 계속 발화해야 하는 것(커널 ABI 면제, R8 7′)은 `expect`로 바꾼다. 26곳에 사유를 단다. R16 규칙을 "allow는 cfg/피처에 따라 발화가 갈리는 자리에만"으로 개정 | lint 수가 167 이하로 유지되는지. 이동 클래스라 ptx-scan은 필요 없다(속성만) | S | 낡은 allow가 스스로 경고가 된다. `*Args` 라운드가 allow를 걷었는지 컴파일러가 알려 준다 |
| 4 | **`cargo mutants`**(FAIL-first 자동화) | 없음. 박스에 미설치 | threads + sampler 파일럿, 그다음 qdot `check_row`/quantize 경로만 `--file`/`--re`로 좁힌다 | 계기(게이트가 무엇을 못 박지 않았는가). 코드 변경 없음 | M | 한 시팅(30분)에 들어가는 것은 작은 크레이트뿐이다[유도, §3] |
| 5 | **게이트 빌드의 `overflow-checks`**(조용한 실패) | release 기본값은 off. 게이트 5개(`gate-ops/attn/ffn/moe/head`, `justfile:455-467`)는 dev라서 켜져 있고, 나머지 release 게이트와 GPU 게이트는 꺼져 있다 | `[profile.gate] inherits="release"`, `overflow-checks=true`, 디바이스 크레이트는 패키지별로 off. 게이트만 `--profile gate`로 돌린다 | 게이트 녹색 + 타이밍 러너 불변(다른 target 디렉터리) | S–M | 정수 wrap이 이름 붙은 패닉이 된다. 디바이스 쪽은 반드시 끈다. cargo-oxide 주석: "overflow-check MIR changes code shape enough to break pattern-sensitive device lowerings" |
| 6 | **Release 프로필**(`lto`/`codegen-units`/`panic`/`debug`) | `[profile.*]` 0개. 기본값은 `lto=false`(thin local), `codegen-units=16`, `panic='unwind'`, `debug=false` | ① plain 호스트 바이너리만 thin LTO/cgu=1: 이득 0…1 %[유도]. oxide 바이너리의 LTO는 미검증이고 위험하다(아래). ② `debug="line-tables-only"`는 **디바이스에도 보인다**: `device_codegen.rs:1324-1328`이 `LineTablesOnly → DebugKind::LineTables`로 옮겨 PTX에 라인 테이블이 붙는다. 그러니 `[profile.perf]` 커스텀 프로필로 perf 전용에만 쓴다. ③ `panic="abort"`는 **기각**: `threads` `catch_unwind`(lib.rs:281, :447)의 패닉 재생 계약이 무너진다 | ptx-scan 동일 + 같은 임대 A/B(A/A 팔 필수) | S | 기대 이득이 자의 오차 아래다. `debug`는 perf 귀속에만 값이 있다 |
| 7 | **loom / kani**(`crates/threads`) | 없음 | loom: 잃어버린 wakeup과 교착(`parked`/`seq` 핸드셰이크 lib.rs:108-116, :270, :427-432, :463). loom은 SeqCst를 AcqRel로 다루므로 지금 코드로는 **거짓 경보가 확정**이다. `fence(SeqCst)`로 바꿔야 하는데 그것 자체가 디스패치 경로 변경이다. kani: 동시성이 범위 밖이라 `chunk_bounds`(lib.rs:499) 같은 순수 함수만 된다 | loom 셈 + `gate-threads` + 디스패치 A/B(펜스 변경) | M–L | 2026-09-20 행(hang) 부류를 증명으로 닫을 유일한 길이지만 비용이 정직하게 크다 |
| 8 | **PGO/BOLT** | 없음. 박스 llvm-tools에 `llvm-profdata`(LLVM 23, 핀 nightly와 일치), LLVM 21 타르볼에 `llvm-bolt`/`merge-fdata`/`perf2bolt`가 있다 | 호스트 바이너리 프로파일 수집 → 재빌드. oxide 바이너리는 `RUSTFLAGS` env를 통해 가능하다(cargo-oxide가 합친다) | 같은 임대 A/B | M | 핫 코드가 분기가 적은 intrinsic 루프라서 0…1 %[유도]. 문헌 수치는 분기가 많은 컴파일러의 값이다 |
| 9 | **스크럽 도구** | `just deny`만 있다. 박스 설치는 deny, miri뿐이다. **`cargo-nextest`는 미설치**인데 `.config/nextest.toml`이 있다 | machete·typos·`RUSTDOCFLAGS=-D warnings cargo doc --document-private-items`를 각각 한 줄 레시피로. semver-checks는 **해당 없음**(17개 중 16개가 `publish = false`, 나머지 하나는 제외된 재현기 `oxide-ice-unroll`) | 도구 출력 0 | XS 각각 | 싸다. machete는 조사 보고서 §6이 이미 "yes"라 했는데 채택되지 않았다 |
| 10 | **단위 뉴타입(R5)/타입스테이트** | `gpu-deepseek41/src`의 `usize` 파라미터 중 단위 이름(`rows`/`cols`/`bytes`/`pos` 계열)이 43개, 모호한 `n`/`len`이 42개. `*Args` 47개. `NonZero` 7, `derive_more` 0 | 조사 보고서 §5가 무비용을 증명해 닫았고, 7″ ③이 커널 ABI 뉴타입을 기각했다. 남는 것은 모호한 `n`/`len` 42개의 이름에 단위를 붙이는 일이다. 타입스테이트는 **안 한다**: 필드가 전부 필수인 구조체 리터럴은 이미 컴파일 타임에 모든 필드를 강제한다 | 이동 클래스 | S | 낮다 |
| 11 | **캐시라인 패딩** | 이미 적용: `DoneMark`(threads lib.rs:475), `Lane`(ops.rs:2373), `RowMark`(attn.rs:2500). `UNION_SUM_BUF`(moe.rs:1161)는 thread-local(각자 할당) | `Pool`의 `seq`/`dispatches`/`parked`/`dispatcher_parked`는 `repr(Rust)`라서 배치를 소스로 알 수 없다. 디스패처가 `dispatches.fetch_add` → `seq.fetch_add`를 연달아 하므로 같은 줄이면 오히려 전송이 한 번이다 | occupancy가 아니라 레이아웃이라 `-Zprint-type-sizes`로 본다 | XS | ≈0[유도] |
| 12 | **`std::simd`** | 0. 트리 전체의 `#![feature]`가 **0**이다 | 호스트 오케스트레이션 루프는 znver3에서 자동 벡터화되거나(`rms_norm` ops.rs:275) libm `exp`에 묶여 있다(moe.rs:251-253, `StdFloat`는 math.h 호출일 수 있다). 라우터 top-k(moe.rs:271)는 SIMD가 아니라 알고리즘 문제(`select_nth_unstable`)이고 V2-Lite CPU 경로에서 스텝당 수십 µs 규모[유도] | — | — | 0. 첫 feature gate를 들일 이유가 없다 |
| 13 | **`std::hint::assert_unchecked`** | 0. `get_unchecked`는 **호스트 크레이트에 0개**이고 디바이스 크레이트에만 있다: gpu 743, gpu-deepseek41 317, q3k-gemv 159, gpu-vision 19 | 툴체인이 막는다: cuda-oxide MIR importer가 `Assume`을 버린다(`mir-importer/src/translator/statement.rs:860-861`). 대안은 R2: 검사한 부분 슬라이스를 배열로 한 번 보는 것(검사 1회 → trap) | ptx-scan 명령 수 | — | 0. nvlabs-ledger 후보 |

## 2. 먼저 할 세 가지 (라운드 스펙 모양)

**A. `oxcpu` — oxide 빌드에 호스트 CPU 플래그 복원 (XS, 리드 트랙 V4.1/fixups).** 한 줄 파일 `.cargo/cuda-oxide.toml`(`extra-rustflags = ["-C", "target-cpu=znver3"]`)을 추가한다. cargo-oxide는 standalone 프로젝트에서 cwd를 `workspace_root`로 잡고(context.rs `is_workspace=false` 분기) `.cargo/cuda-oxide.toml`을 읽는다(context.rs:222). box.sh가 `$REMOTE`로 cd하므로 레포 루트가 맞다.

세 층은 이렇게 짠다.
- **구조**: 호스트 rustflags의 소유자가 둘이 된다. `tools/check-arch.sh` 류에 "`config.toml` target rustflags == `cuda-oxide.toml` extra-rustflags" 검사를 넣어 한 사실로 묶는다.
- **재발 방지**: 스펙 B의 asm-scan으로 `generate_ds41`·`gate_e2e`의 legacy-SSE 수를 핀한다. 오늘 소스에서 1778/1592로 FAIL-first다.
- **디버깅**: 같은 도구.

증명은 다섯 가지다.
1. ptx-scan 표 동일(`generate_ds41`, `gate_e2e`). `-C target-cpu`가 같은 rustc 세션의 `cfg(target_feature)`를 통해 디바이스 MIR로 새지 않는지를 이 표가 확인한다.
2. `gate-gpu-e2e`와 ds41 비트 게이트 동일. rustc는 mul+add를 FMA로 합치지 않고, 스칼라 폴백은 런타임에 죽은 경로다.
3. 빌드 로그에 `Compiling bloomery-model` 재빌드가 보일 것.
4. 같은 임대 A/B 1회. 예측은 pp512 0…+1 %, tg96 ≈0[유도]. 근거: union 런당 glue가 수십 명령인 데 비해 tile 커널은 수천 명령이다. `is_x86_feature_detected!` 분기가 상수 폴딩되고 스칼라 루프가 AVX 폭으로 넓어지는 것이 전부다. 따라서 ds41batch 모형 잔차 17 %의 유력 후보는 아니다.
5. 문서 정정(리드): `.cargo/config.toml:1-3` 주석, `AGENTS.md:414` 문단.

업스트림: README는 `extra-rustflags`를 "project defaults"라고 문서화하지만, 패스스루가 `.cargo/config.toml`의 `build.rustflags`/`target.*.rustflags`를 **조용히 가린다**는 말은 없다. 중복 검색 결과는 §6. 경고를 내거나 합치자는 제안 이슈 후보다. ledger 한 줄은 리드가 쓴다.

**B. `asmscan` — 호스트 쌍둥이 계기 (S).** `tools/asm-scan.sh <bin> [fn-substr]`를 만든다. 이번 라운드에서 손으로 친 objdump/awk를 승격하는 것이고 새 의존성은 없다. 함수별 총 명령·legacy-SSE·VEX·호출·outlined `core_arch` 심볼을 출력한다. `just asm-scan BIN`(빌드 + 스캔, `ptx-scan` 레시피와 같은 모양)과 래칫 `tools/ref/asm-shapes.tsv`를 둔다. 선택으로 `cargo asm --mca`(llvm-mca 21 PATH 추가)를 붙이면 "이 스칼라 루프가 벡터화되나"에 한 줄로 답하고, `tile_units` 같은 손 계수를 대체한다. 증명은 FAIL-first(A 착륙 전 트리에서 빨강) + 착륙 후 녹색.

**C. `mutpilot` — cargo-mutants 파일럿 (M, 박스 시팅, 승인 필요).**
1. 박스에 `cargo install cargo-mutants --locked`(네트워크 필요).
2. `cargo mutants -p bloomery-threads -p bloomery-sampler --jobs 4 -- --release -- --include-ignored`. 인자 패스스루는 §5 인용.
3. 예상 벽시계[유도]: 뮤턴트는 이항연산자 수 + 함수당 1–2개로 잡으면 threads ≈ 62+27×1.5 ≈ 100, sampler ≈ 63+19×1.5 ≈ 90. 뮤턴트당 증분 빌드 + 게이트를 ~10–15 s로 보면(AGENTS "warm runs take seconds") 190 × 12 s / 4 jobs ≈ **10 분**, jobs별 콜드 사본 빌드 4 × ~1–2 분까지 합쳐 **~15–20 분**. 30분 안에 든다.
4. qdot(binops ~1355 + fns 101 → ~1500 뮤턴트)는 같은 계산으로 jobs 8에서도 ~45–60 분이라 **안 들어간다**. `--file crates/qdot/src/lib.rs --re 'check_row|quantize_col|non_finite'`로 좁힌다. model(binops ~2437)은 대상 밖이다.
5. `--in-place`는 `--jobs`와 함께 쓸 수 없으므로 기본(임시 사본)을 쓴다. 원격 디렉터리에 `.git`이 없을 때 target/을 복사에서 빼는지는 **미확인**이다.
6. 박스 포화 전례가 있으니 측정처럼 파동 사이에 배치한다.

산출물은 "잡히지 않은 뮤턴트" 목록이고, 각각이 비어 있는 게이트 축이다(§9 2층).

## 3. 검증 명령과 실제 출력 (common.md §2)

변경한 파일은 없다. 박스는 box.sh(rsync)를 쓰지 않고 `ssh ws`로 **읽기만** 했다(objdump, nm, fingerprint JSON, `ls`). 빌드는 없었다.

```
$ ssh ws '… objdump -d … | grep -cE "\s(mulss|addss|movss|cvtsi2ss|mulsd|addsd)\s"' (per binary)
bloomery-decode  legacy_sse(mulss|addss|movss|cvtsi2ss)=4 vex(v-prefixed same)=1395 popcnt=0 tzcnt=33
bench_v41_host   legacy_sse(mulss|addss|movss|cvtsi2ss)=6 vex(v-prefixed same)=697 popcnt=0 tzcnt=47
generate_ds41    legacy_sse(mulss|addss|movss|cvtsi2ss)=1778 vex(v-prefixed same)=103 popcnt=0 tzcnt=108
gate_e2e         legacy_sse(mulss|addss|movss|cvtsi2ss)=1592 vex(v-prefixed same)=103 popcnt=0 tzcnt=80
```
`target/release/build/bloomery-model/*/fingerprint/lib-*.json`의 `rustflags` 집계(열: 개수, znver3 유무, 빌드 경로, 플래그 수):
```
      1 - oxide 6
      4 - oxide 8
      3 - plain 0
      3 znver3 plain 2
```
(`plain 0` 3개는 설명하지 못했다. 설정 파일이 들어온 `48fc38c`(09-20) 이전의 낡은 유닛일 수 있고, 대조 결론은 바뀌지 않는다.)

함수 단위로 보면 `generate_ds41` 안의 union 호스트 glue가 전부 legacy다: `model::ops::run_group` 2165 insn legacy=116 vex=0, `group_core` 1510/42/0, `PairWork::compute_rows` 633/28/0, `model::moe::serve` 554/42/0. 같은 이름 공간의 `bench_v41_host` 함수들은 legacy=0이다.

intrinsic이 인라인되지 못한 흔적(AGENTS의 "tens of times slower" 함정)을 찾았다:
```
bloomery-decode  outlined core_arch fns=0  calls to them=0
bench_v41_host   outlined core_arch fns=0  calls to them=0
generate_ds41    outlined core_arch fns=2  calls to them=2     (_xgetbv only — feature detection)
gate_e2e         outlined core_arch fns=2  calls to them=2
```
그 함정은 오늘은 살아 있지 않다. 다만 oxide 빌드에서는 설정이 그것을 덮어 주지 않는다.

박스 툴체인: `rustc 1.100.0-nightly`, `LLVM version: 23.1.0`. `~/.cargo/bin`에는 cargo-deny, cargo-miri, cargo-oxide만 있고 nextest, mutants, cargo-asm은 없다. LLVM 21 타르볼에 `llvm-bolt llvm-mca llvm-profdata merge-fdata perf2bolt`가 있다.

## 4. 예측 대 결과

스펙은 A/B 실행을 금지했다. 표의 밴드는 전부 [유도]이고 측정한 것은 §3뿐이다. 레버 1(release 프로필)의 유도는 이렇다. `dot_row_cols` 호출마다 `check_row` × 열 + `tile_kind` + `array::from_fn`으로 ~50–100 명령이 든다. V4.1 k = 4096에서 한 런의 tile 작업은 16 SB × (fixed + 8 × per_col)/2(`tile_units` Q4_K 96/97)로 ≈ 7,000 명령이다. 그러니 LTO로 없앨 수 있는 몫은 ≤ 1–1.5 %이고, 디코드는 대역폭 경계라 0이다.

LTO를 oxide 바이너리에 쓰는 것은 미검증이다. `join_codegen`(lib.rs:806-838)이 호스트를 LLVM에 위임한 뒤 디바이스 아티팩트 객체를 `bytecode: None`으로 덧붙인다. cross-crate LTO에서 이 객체가 어떻게 되는지(비트코드 누락 오류인지, `.oxart` 소실인지)는 코드를 읽어서는 판정할 수 없다. 켠다면 ptx-scan이 섹션을 못 읽는 순간 실패하므로 그것이 FAIL 탐지기다. 게다가 `lto`는 패키지별로 줄 수 없어서 `[profile.release]`에 넣는 순간 oxide 바이너리에도 닿는다(`justfile:983` `--release`).

## 5. 출처 (가져온 것만, 한 문장씩 인용)

- Cargo config, build.rustflags — https://doc.rust-lang.org/cargo/reference/config.html : "There are four mutually exclusive sources of extra flags. They are checked in order, with the first one being used: 1. `CARGO_ENCODED_RUSTFLAGS` … 3. All matching `target.<triple>.rustflags` …"
- Cargo profiles — https://doc.rust-lang.org/cargo/reference/profiles.html : "Performs 'thin local LTO' which performs 'thin' LTO on the local crate only across its codegen units." / "Tests, benchmarks, build scripts, and proc macros ignore the `panic` setting." / line-tables-only: "Generates the minimal amount of debug info for backtraces with filename/line number info".
- cargo-oxide README(핀 `b9847e95`, 로컬 `git show`) : "Do not put `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS` in the `[env]` table; use `extra-rustflags` for project defaults." (로컬 git 객체이고 URL 페치가 아니다)
- Reference, diagnostics — https://doc.rust-lang.org/reference/attributes/diagnostics.html : "If the expectation is unfulfilled, because lint `C` would not be emitted, the `unfulfilled_lint_expectations` lint will be emitted at the attribute." / "Tool lints only get checked when the associated tool is active."
- Rust 1.88 — https://blog.rust-lang.org/2025/06/26/Rust-1.88.0/ : "Let chains are only available in the Rust 2024 edition…" (트리 17개 크레이트 모두 2024이고 let-chain 사용은 0)
- `assert_unchecked` — https://doc.rust-lang.org/std/hint/fn.assert_unchecked.html : "You may know this from other places as `llvm.assume`…" (1.81.0)
- `std::simd` — https://doc.rust-lang.org/nightly/std/simd/index.html : "This is a nightly-only experimental API. (`portable_simd` #86656)"
- `StdFloat` — https://doc.rust-lang.org/nightly/std/simd/trait.StdFloat.html : "…may, in the absence of hardware support, canonicalize to calling an operating system's `math.h` dynamically-loaded library."
- cargo-pgo — https://raw.githubusercontent.com/Kobzol/cargo-pgo/main/README.md : "The binary will be optimized with respect to the profiled workloads." (수치 없음)
- rustc PGO — https://doc.rust-lang.org/rustc/profile-guided-optimization.html : "Alternatively, an `llvm-profdata` coming with a recent LLVM or Clang version usually works too."
- Kobzol 2023 — https://kobzol.github.io/rust/cargo/2023/07/28/rust-cargo-pgo.html : "It's hard to say whether this will generalize to other programs."
- Kobzol 2022 — https://kobzol.github.io/rust/rustc/2022/10/27/speeding-rustc-without-changing-its-code.html : rustc PGO 갱신 "Mean improvement: -0.75%", LLVM PGO "Mean improvement: -2.48%".
- cargo-show-asm — https://raw.githubusercontent.com/pacak/cargo-show-asm/master/README.md : "A cargo subcommand that displays the Assembly, LLVM-IR, MIR and WASM generated for Rust source code."
- cargo-mutants — https://mutants.rs/nextest.html : "nextest currently does not run doctests…"; https://mutants.rs/timeouts.html : "The default test timeout is 5 times the baseline test time, with a minimum of 20 seconds."; https://mutants.rs/in-place.html : "currently incompatible with the `--jobs` option"; https://mutants.rs/cargo-args.html : "You can use a second double-dash to pass options through to the test targets".
- loom — https://raw.githubusercontent.com/tokio-rs/loom/master/README.md : "SeqCst accesses (e.g. `load`, `store`, ..) are regarded as `AcqRel`"; https://docs.rs/loom/latest/loom/ : "setting the thread pre-emption bound to 2 or 3 is enough to catch most bugs".
- Kani — https://model-checking.github.io/kani/rust-feature-support.html : "Concurrent features are currently out of scope for Kani."; https://model-checking.github.io/kani/install-guide.html : "Rust 1.58 or newer installed via `rustup`."
- cargo-semver-checks — https://raw.githubusercontent.com/obi1kenobi/cargo-semver-checks/main/README.md : "Lint your crate API changes for semver violations." (crates.io 기준선이 필요한데 우리 크레이트는 게시하지 않는다)
- typos — https://raw.githubusercontent.com/crate-ci/typos/master/README.md : "Low false positives so you can run on PRs"
- cargo-machete — https://raw.githubusercontent.com/bnjbvr/cargo-machete/main/README.md : "detects unused dependencies in Rust projects, in a fast (yet imprecise) way."
- rustdoc lints — https://doc.rust-lang.org/rustdoc/lints.html : `broken_intra_doc_links` "Warns by default".

cuda-oxide 소스 근거(핀 `b9847e95`): `cargo-oxide/src/commands/codegen_env.rs:84-140, 288-289`, `context.rs:222, 310`, `rustc-codegen-cuda/src/device_codegen.rs:1004, 1323-1328`, `mir-importer/src/translator/statement.rs:860-861`, `cuda-macros/src/kernel/codegen.rs:241-271`, `cuda_module/mod.rs:903-909`, `rustc-codegen-cuda/src/lib.rs:806-851`.

## 6. 하지 못한 것과 그 이유

- A/B, 빌드, 뮤턴트 실행은 스펙이 금지해서 하지 않았다. 밴드는 전부 [유도]다.
- `expect`가 `#[kernel]` 확장 뒤에서 충족되는지는 소스로 판정할 수 없었다. 제네릭 커널은 lint 속성이 구현 fn에만 가고, 생성되는 런치 메서드에는 `doc`만 복사된다. 커널 하나에 clippy 한 번을 돌리면 결정된다.
- 업스트림 중복 검색 쿼리(`gh search issues/prs --repo NVlabs/cuda-oxide`): "rustflags config.toml" 0/0, "build.rustflags" 0/0, "target-cpu" 관련 없음(#1243 "refuse host-CPU code reachable from kernels"는 다른 주제), "CARGO_ENCODED_RUSTFLAGS" → #531/#538(doctor 검증뿐). 중복은 찾지 못했다. cargo-miri가 같은 가림을 어떻게 다루는지(형제 선례)는 보지 않았다.
- advisor의 제안대로 `Pool` 필드가 실제로 어느 캐시라인을 공유하는지는 `-Zprint-type-sizes`가 필요해 확인하지 않았다.

## 7. 스펙 밖에서 본 개선 지점 (보고만, 손대지 않음)

- `.cargo/config.toml:1-3`과 `AGENTS.md:414`: oxide 빌드에 대해 사실이 아닌 문장. 리드가 정정. XS
- `crates/model/src/moe.rs:287` `*offsets.last().unwrap()`: 라이브러리 크레이트의 R10(원소 하나를 넣고 시작하니 실패가 불가능하다 → `expect("…")` 또는 fold). XS
- `justfile:455-467`: 게이트 5개(`ops/attn/ffn/moe/head`)가 dev 프로필이고 나머지는 `--release`다. 게이트 벽시계와, 배포 바이너리와 다른 최적화 수준으로 커널을 시험한다는 점이 걸린다. 비트는 같다. S
- `justfile:287` `build-cpu`의 `RUSTFLAGS="-C target-cpu=znver3"`는 config와 중복이다. 둘째 소유자다. XS
- `.config/nextest.toml`: 박스에 `cargo-nextest`가 없어 쓰이지 않는 설정이다(게이트는 `tools/gate.sh` → `cargo test`). 설치하거나 지운다. XS
- 사유 없는 allow 26곳(R16): `crates/gpu/src/flash.rs:960, 1072, 1135, 1256, 1392, 1467, 1537, 1607, 1677, 1753, 1825, 1915, 2002, 2121, 2225` 외 11곳. S (스펙 C3과 한 축)
- `crates/gpu/src/q8f32.rs:65` `f32_lane_partials`: 호출자 계약에 기대는 `pub(crate)` 안전 함수다. 크레이트 안이라 R28 허용 범위지만, 같은 모양의 `pub unsafe fn` 이웃(:161)과 표기가 섞여 있다. XS
- `crates/gpu-deepseek41/src/hc.rs:295-310`: 24개 `get_unchecked` 묶음은 `&hc[h..h + 24]`를 `&[f32; 24]`로 한 번 보면 검사 1회(trap)로 안전하게 된다(R2). 실패하면 조용하지 않고 trap이다. 증명은 ptx-scan 명령 수. 디바이스에서 범위 슬라이스와 배열 변환이 지원되는지는 미확인. S
- nvlabs-ledger 후보 둘(리드 몫): ① 패스스루가 `.cargo/config.toml` rustflags를 가림 ② `NonDivergingIntrinsic::Assume`을 버려서 `assert_unchecked`가 디바이스에서 무효.

## 8. 모델

opus(Opus 5.5)로 스폰되어 돌았다.
