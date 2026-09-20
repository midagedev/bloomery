# NVlabs 툴체인 업스트림 장부 — cuda-oxide · cutile-rs(cuda-core/cuda-bindings)

GPU 단계에서 만나는 이슈·PR 후보를 발견 즉시 여기 적는다(사용자 지시 2026-09-21: "cuda oxide에 이슈나 PR 만들 거리 있으면 캐치해 놔줘"). 한 줄이면 된다 — 정식 초안은 `docs/upstream/<slug>.md`, 제출 파이프라인은 rig-log `docs/upstream-contributions.md`(중복 검색 → FAIL-first 실측 → 최소 패치 → 적대적 리뷰). 여기 적힌 것은 **후보**다. 코드 독해로 확정한 결함은 가설이고, 실측 재현이 붙은 것만 상태 열이 `재현`으로 오른다.

| # | 대상 | 무엇 | 상태 | 근거 |
|---|---|---|---|---|
| 1 | cuda-oxide | `#[unroll]`이 `usize` 카운터 `while` 루프에서 `APInt::shl: bitwidth mismatch`로 컴파일러 패닉(전체·부분 언롤 모두) | 재현, 초안 있음(`cuda-oxide-unroll-ice.md`), **미제출** | MUL-7, 재현기 `crates/oxide-ice-unroll` |
| 2 | cutile-rs `cuda-bindings` 0.3.1 | `cuGraphInstantiate`·`cuGraphLaunch`·`cuGraphExecDestroy`·`cuGraphDestroy` 바인딩 부재. `cuda-core`의 `CudaStream::begin_capture`/`end_capture`는 `CUgraph`를 돌려주지만 그것으로 할 수 있는 일이 없다 — 캡처는 있고 재생이 없다 | 확인(`grep -rn "fn cuGraph"` 0건, crates.io 최신 0.3.1도 같음), 미제출. bloomery가 `extern "C"`로 직접 선언해 쓰면 그 코드가 PR 본체가 된다 | `docs/gpu-design.md` §미정 |
| 3 | cuda-core | `cuda-core`에 그래프 실행 래퍼(`CudaGraph`/`CudaGraphExec`, Drop에서 destroy)가 없다 — 2번이 들어가면 그 위의 안전 래퍼 | 후보(2번 뒤) | 같은 자리 |

발견 규칙: 우회로를 쓰기로 했더라도 여기 먼저 한 줄 적고 우회한다. 우회가 증거를 지운다.
