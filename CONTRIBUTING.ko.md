# 기여 안내

관심 가져 주셔서 고맙습니다. 이슈와 PR 모두 환영합니다. 먼저 지금 상태를 솔직히 적어 둡니다. 모든 게이트(카드에서
도는 검증 테스트)는 메인테이너의 워크스테이션 한 대에서만 돕니다. 그래서 PR의 최종 검증은 메인테이너가 그 머신에서
합니다. 내 머신에서 확인할 수 있는 것과 없는 것을 아래에 정리했습니다. 영어판은 [`CONTRIBUTING.md`](CONTRIBUTING.md)입니다.

## 빌드

툴체인(고정된 nightly, rev로 고정한 cuda-oxide 포크, CUDA 13.3)과 모델별 명령은 [`docs/BUILD.md`](docs/BUILD.md)에
있습니다. 디바이스 크레이트는 그냥 `cargo`로 빌드하면 안 되고 `cargo oxide`를 써야 합니다(`AGENTS.md`의 "Never").

## 우리 하드웨어 없이 할 수 있는 일

- **순수 크레이트 (OS 무관):** `python3 tools/recipes.py pure-crates`가 목록을 보여 줍니다. `cargo test -p <crate>`로
  하나씩 돌립니다. `bloomery-serve`에는 mock 엔진이 있어서 모델 없이 HTTP API를 테스트할 수 있습니다.
- **호스트 tier (x86_64 Linux, 카드 불필요):** `python3 tools/recipes.py host-nightly`가 보여 주는 단위들입니다.
  Zen 3 대상으로 빌드되므로 다른 CPU에서는 `RUSTFLAGS='-C target-cpu=native'`를 주세요.
- **Python·셸 도구:** `tools/` 아래 도구들은 self-test가 있습니다(`python3 tools/recipes.py --self-test` 등).
- **Mac:** `just mac-static`이 정적 검사(포맷, x86_64-linux 대상 clippy, 모든 check 스크립트)를 돌립니다.
- **입문 작업:** [`good first issue` 라벨](https://github.com/midagedev/bloomery/labels/good%20first%20issue)이 붙은
  이슈들입니다. `no-box` 라벨은 우리 머신 없이 끝까지 검증할 수 있다는 뜻입니다.

## PR은 이렇게 검증됩니다

1. 내 머신에서 돌릴 수 있는 것을 초록으로 만들고, PR에 **실제로 돌린 명령과 출력**을 붙여 주세요.
2. 메인테이너가 `just affected`가 고른 게이트를 워크스테이션에서 돌리고, 초록이면 착지시킵니다.
3. PR 라벨 하나로 이 검증이 자동으로 도는 경로는 계획 중이고 아직 없습니다.

## 특히 바라는 두 가지

- **새 모델 아키텍처 추가:** [`docs/contrib/new-model.md`](docs/contrib/new-model.md). 최근 세 모델을 붙인 순서와,
  단계마다 혼자 검증할 수 있는지가 적혀 있습니다. 시작 전에 "New model support" 이슈를 열어 주세요. 같은 모델을 두
  사람이 잡지 않게 하려는 것입니다.
- **NVIDIA DGX Spark 포팅 (aarch64, GB10):** [`docs/contrib/dgx-spark.md`](docs/contrib/dgx-spark.md). 단계별
  계획(P0 컴파일 → P1 픽스처 게이트 → P2 실제 모델 → P3 속도)과 메인테이너의 결정이 §5에 있습니다.

## 코드 품질 (리뷰에서 가장 먼저 봅니다)

- **두 번째 복사본을 만들지 마세요.** 두 모델이나 두 호출처가 같이 쓰는 로직은 공통 소유자 하나로 올립니다. 모델
  고유 코드에는 그 아키텍처가 강제하는 것(텐서 모양, 어텐션 종류, 양자화)만 둡니다. 공통 코드에 모델별 분기나
  플래그를 추가하지 마세요.
- **안 쓰게 된 코드는 지웁니다.** 함수, 플래그, 게이트 조항, 문서까지.
- **조용히 실패하지 않습니다.** 정의되지 않은 입력은 이름 붙은 에러나 panic으로 거부합니다.
- **게이트를 완화하지 않습니다.** 기준을 옮겨야 하면 `PIN(YYYY-MM-DD):` 주석과 이유를 붙이고, 새 게이트는 먼저
  결함을 잡아 실패하는 것을 보여 주세요.
- **숫자에는 조건을 붙입니다.** `29 tok/s`가 아니라 `tok/s @ n=96, depth 4096, RTX A6000`. 내 머신에서 잰 숫자는
  내 표로 따로 적고, 우리 표와 섞지 않습니다.
- 규칙 전체는 [`docs/rust-quality.md`](docs/rust-quality.md)(R1–R29)이고, 리뷰는 규칙 번호로 말합니다.
  [`AGENTS.md`](AGENTS.md)는 메인테이너와 AI 라운드를 위한 전체 계약입니다. 첫 변경 전에 "Never" 목록은 꼭 읽어 주세요.

## AI 도움

AI로 작성한 PR도 환영합니다. bloomery의 상당 부분이 AI 도움으로 만들어졌습니다. 에이전트에게는 `AGENTS.md`를 먼저
읽히세요. PR에는 AI가 도왔는지, 어느 부분인지 적어 주세요. diff는 직접 읽고 설명할 수 있어야 하고, 출력은 실제로 돌린
것만 붙여 주세요.

## 문의

질문이나 아이디어는 [Discussions](https://github.com/midagedev/bloomery/discussions)에, 버그는 이슈 템플릿으로
남겨 주세요. 한국어로 쓰셔도 됩니다.
