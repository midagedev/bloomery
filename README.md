# mulle

한 워크스테이션을 위한 LLM 추론 엔진. Rust 호스트, CUDA Rust 커널(cuda-oxide, 뒤에 cutile-rs), 목표 모델은 DeepSeek-V4.1-Flash, 디딤돌은 DeepSeek-V2-Lite-Chat Q3_K_M. 모든 주장은 측정한 것이고, 측정은 [rig-log](https://github.com/midagedev/rig-log)에 기록한다.

단계와 게이트는 `docs/plan.md`.
