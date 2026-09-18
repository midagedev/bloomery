# mulle — agent context

한 워크스테이션을 위한 LLM 추론 엔진. Rust 호스트, CUDA Rust 커널(cuda-oxide, 뒤에 cutile-rs). 목표는 DeepSeek-V4.1-Flash, 디딤돌은 DeepSeek-V2-Lite-Chat Q3_K_M. 단계와 게이트는 [`docs/plan.md`](docs/plan.md).

## 추적

이슈는 셀프호스트 트래커 **MUL** 프로젝트(`gadak --workspace gdk`, Mac에서는 `GADAK_HOME=$HOME/.gadak`, 빌드는 `/opt/homebrew/bin/gadak` — PATH의 dev 빌드는 미러 스키마를 못 읽는다). 단계마다 이슈 하나(MUL-1 ~ MUL-5), 기계 변경은 별도(MUL-6). 측정 수치는 [rig-log](https://github.com/midagedev/rig-log)의 `log/`에 먼저 기록하고 이슈에서 링크한다. `TODO.md`를 열지 않는다.

## 빌드·실행

개발은 박스의 RTX 3090에서. Mac에서 편집하고 `tools/box.sh <명령>`으로 트리를 rsync한 뒤 박스에서 돈다(환경은 박스의 `~/mulle-env.sh`: nightly-2026-08-28, LLVM 21.1.8 타르볼, CUDA 13.0, 3090 핀). 박스에서 직접 편집하지 않는다 — rsync가 `--delete`다. A6000은 서빙·야간 학습이 쥐고 있으니 건드리지 않는다.

## 관례

- 숫자는 측정한 것만, 도출이면 도출이라 밝힌다. 틀리면 선을 그어 정정한다(rig-log와 같다).
- 커널은 Rust만. 참조 하네스(`tools/ref/`)는 ggml에 링크하는 C++이고 그 목적은 진실값과 기준 속도다.
- 각 crate의 `rust-toolchain.toml`이 nightly를 고정한다. 바꾸는 것은 cuda-oxide 핀이 움직일 때만.
- 이 리포 안 산문은 한국어, 코드 주석과 업스트림으로 나가는 글은 영어.
