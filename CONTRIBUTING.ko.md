# 기여 안내

관심 가져 주셔서 고맙습니다. 이슈와 PR 모두 환영합니다. 먼저 지금 상태를 솔직히 적어 둡니다. 게이트(카드에서 도는
검증 테스트)는 NVIDIA 카드와 모델 파일이 있어야 돌고, 전체 세트는 메인테이너의 워크스테이션에서 돕니다. 내 카드에서
돌린 결과도 근거로 인정합니다. 내 머신에서 확인할 수 있는 것과 없는 것을 아래에 정리했습니다. 영어판은
[`CONTRIBUTING.md`](CONTRIBUTING.md)입니다.

## 빌드

툴체인(고정된 nightly, rev로 고정한 cuda-oxide 포크, CUDA 13.3)과 모델별 명령은 [`docs/BUILD.md`](docs/BUILD.md)에
있습니다. 디바이스 크레이트는 그냥 `cargo`로 빌드하면 안 되고 `cargo oxide`를 써야 합니다(`AGENTS.md`의 "Never").

## 우리 하드웨어 없이 할 수 있는 일

- **순수 크레이트 (OS 무관):** `python3 tools/recipes.py pure-crates`가 목록을 보여 줍니다. `cargo test -p <crate>`로
  하나씩 돌립니다. `bloomery-serve`에는 mock 엔진이 있어서 모델 없이 HTTP API를 테스트할 수 있습니다.
- **호스트 tier (x86_64 또는 aarch64 Linux, 카드 불필요):** `python3 tools/recipes.py host-nightly`가 보여 주는 단위들입니다.
  Zen 3 대상으로 빌드되므로 다른 CPU에서는 `RUSTFLAGS='-C target-cpu=native'`를 주세요(`just contrib-check`는 알아서 설정합니다).
- **Python·셸 도구:** `tools/` 아래 도구들은 self-test가 있습니다(`python3 tools/recipes.py --self-test` 등).
- **Mac:** `just mac-static`이 정적 검사(포맷, x86_64-linux 대상 clippy, 모든 check 스크립트)를 돌립니다.
- **한 번에:** `just contrib-check`가 위 항목을 모두 돌리고, 항목마다 한 줄씩 결과를 낸 뒤 마지막에 `DONE` 줄을 찍습니다.
  `--tier t0`은 OS와 관계없이, `--tier t1`(x86_64 Linux의 기본값)은 호스트 tier까지 돌립니다. PR에는 이 출력을 붙여
  주세요. PR마다 GitHub CI가 `--tier t0`을 돌리지만, 이것은 변경을 빠르게 훑는 검사이지 착지 근거가 아닙니다.
- **입문 작업:** [`good first issue` 라벨](https://github.com/midagedev/bloomery/labels/good%20first%20issue)이 붙은
  이슈들입니다. `no-box` 라벨은 우리 머신 없이 끝까지 검증할 수 있다는 뜻입니다.

## PR은 이렇게 검증되고 머지됩니다

- **CI.** 모든 PR에서 `just contrib-check --tier t0`가 돕니다(`.github/workflows/pr.yml`). `main`에 들어가려면 이게
  초록이어야 합니다.
- **리뷰.** 준비된 PR마다 Claude가 참고용 리뷰를 한 번 남깁니다(`.github/workflows/review.yml`, 동작 방식은
  [`.github/review/README.md`](.github/review/README.md)). 댓글만 달고 머지를 막지 않습니다. draft, `no-review` 라벨,
  자동 리뷰 5회에서 멈춥니다. 변경 밖에서 입증된 결함을 찾으면 `review-bot` 이슈로 올립니다.
- **게이트.** `just affected`가 고른 `just gate-*`를 가진 카드에서 돌릴 수 있으면 돌리고(카드가 있는 Linux 머신에서
  `BLOOMERY_BOX_LOCAL=1 just gate-<이름>`, 준비는 `tools/box.sh` 머리말에 있습니다), **실제로 돌린 명령과 출력**을
  PR에 붙여 주세요. 내 카드에서 돌린 결과는 그대로 믿습니다. 돌리지 못한 것은 메인테이너가 워크스테이션에서 돌립니다.
- **머지.** `main`은 PR로만, rebase로만 받습니다(이력은 일직선). 쓰기 권한이 있는 기여자는 CI가 초록이고 근거가 PR에
  있으면 자기 PR을 직접 머지합니다.

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
- **숫자에는 조건을 붙입니다.** `29 tok/s`가 아니라 `tok/s @ n=96, depth 4096, RTX A6000`. A6000이 없어도 됩니다. 내 NVIDIA
  카드에서 잰 숫자는 그 카드에 대한 PR의 근거로 인정합니다. `nvidia-smi` 줄(카드와 드라이버), 비교한 두 커밋, 모델 파일, 실행한 명령과 그 출력만 같이 적어
  주세요. 차이가 작으면 양쪽을 번갈아 두 번씩 돌려 보면 도움이 됩니다. 이 숫자는 그 카드에서 PR을 판정하고,
  그 카드의 별도 표로 공개할 수 있습니다(README, 문서, 릴리스 노트). 표 하나에 여러 카드를 섞지만 않으면 됩니다. 두 카드가
  같은 변경을 다르게 재면 둘 다 유효하고, 둘을 가르는 카드 성질로 변경을 나눠 켭니다.
- 규칙 전체는 [`docs/rust-quality.md`](docs/rust-quality.md)(R1–R29)이고, 리뷰는 규칙 번호로 말합니다.
  [`AGENTS.md`](AGENTS.md)는 사람과 에이전트 모두를 위한 작업 계약입니다. 첫 변경 전에 "Never" 목록은 꼭 읽어 주세요.

## AI 활용

코드도, 저희와 나누는 대화도 AI를 마음껏 써 주세요. bloomery의 상당 부분이 AI 도움으로 만들어졌고, 메인테이너도 답변과
리뷰와 이슈를 모두 AI와 함께 씁니다. 여기서 오가는 글은 양쪽 모두 AI를 쓴다고 가정하고 진행하니, AI를 썼는지 따로 밝히지
않으셔도 됩니다. 이슈, 댓글, PR을 AI로 편하게 쓰시고, 언어도 편한 것으로 쓰시면 됩니다. 에이전트에게는 `AGENTS.md`를 먼저
읽히세요. 저희가 보는 것은 결과물입니다. 단순한 설계, 코드 품질, 그리고 실제로 작동하는지입니다.

## 문의

질문이나 아이디어는 [Discussions](https://github.com/midagedev/bloomery/discussions)에, 버그는 이슈 템플릿으로
남겨 주세요. 한국어로 쓰셔도 됩니다.
