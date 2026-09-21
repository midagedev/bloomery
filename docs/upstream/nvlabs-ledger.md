# NVlabs 툴체인 업스트림 장부 — cuda-oxide · cutile-rs(cuda-core/cuda-bindings)

GPU 단계에서 만나는 이슈·PR 후보를 발견 즉시 여기 적는다(사용자 지시 2026-09-21: "cuda oxide에 이슈나 PR 만들 거리 있으면 캐치해 놔줘"). 한 줄이면 된다 — 정식 초안은 `docs/upstream/<slug>.md`, 제출 파이프라인은 rig-log `docs/upstream-contributions.md`(중복 검색 → FAIL-first 실측 → 최소 패치 → 적대적 리뷰). 여기 적힌 것은 **후보**다. 코드 독해로 확정한 결함은 가설이고, 실측 재현이 붙은 것만 상태 열이 `재현`으로 오른다.

| # | 대상 | 무엇 | 상태 | 근거 |
|---|---|---|---|---|
| 1 | cuda-oxide | `#[unroll]`이 `usize` 카운터 `while` 루프에서 `APInt::shl: bitwidth mismatch`로 컴파일러 패닉(전체·부분 언롤 모두) | 재현, 초안 있음(`cuda-oxide-unroll-ice.md`), **미제출** | MUL-7, 재현기 `crates/oxide-ice-unroll` |
| 2 | cutile-rs `cuda-bindings` 0.3.1 | ~~`cuGraphInstantiate`·`cuGraphLaunch`·`cuGraphExecDestroy`·`cuGraphDestroy` 바인딩 부재~~ **철회(2026-09-21 낮)**: 있다. 바인딩은 build.rs의 bindgen이 `^cu.*` 전부를 `OUT_DIR/cuda_driver_shims.rs`에 생성하고 `dlopen("libcuda.so.1")`로 푼다 — 처음의 `grep -rn "fn cuGraph"`는 크레이트 소스만 봤고 생성물을 안 봤다(박스 생성물에서 `cuGraph` 550건, `cuGraphInstantiateWithFlags`·`cuGraphLaunch`·`cuGraphExecDestroy`·`cuGraphDestroy`·`cuGraphGetNodes` 시그니처 확인). `cuGraphInstantiate`라는 이름은 없고 3인자 `cuGraphInstantiateWithFlags`만 있다(cuda.h 12+의 이름 매핑) — `link_name` 함정은 없던 일 | 철회. 이슈 없음 | 박스 `target/debug/build/cuda-bindings-*/out/cuda_driver_shims.rs` |
| 3 | cutile-rs `cuda-core` 0.3.1 | 그래프 API의 안전 래퍼가 없다: `begin_capture`/`end_capture`는 구식 `runtime::Stream`에만 있고 `#[cuda_module]` 런처가 받는 `simt::CudaStream`에는 없다; 인스턴스화·런치·Drop-destroy 래퍼는 어느 층에도 없다. bloomery `crates/gpu/src/graph.rs`(`Graph::capture`/`launch`/`node_count`, Drop)가 그 모양이다 | 후보 — P0 게이트(즉시 실행 = 재생, 3090)가 녹색이면 실측 근거가 생긴다. 제출 전 cutile-rs HEAD 재확인 | `crates/gpu/src/graph.rs` |

발견 규칙: 우회로를 쓰기로 했더라도 여기 먼저 한 줄 적고 우회한다. 우회가 증거를 지운다.
