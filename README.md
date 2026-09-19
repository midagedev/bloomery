# bloomery

한 워크스테이션을 위한 LLM 추론 엔진. Rust 호스트, CUDA Rust 커널(cuda-oxide, 뒤에 cutile-rs), 목표 모델은 DeepSeek-V4.1-Flash, 디딤돌은 DeepSeek-V2-Lite-Chat Q3_K_M. 모든 주장은 측정한 것이고, 측정은 [rig-log](https://github.com/midagedev/rig-log)에 기록한다.

단계와 게이트는 `docs/plan.md`.

## 참조

커널은 ggml의 k-quant 경로를 읽고 구현했다. Q3_K 블록 기하, 활성값 Q8 양자화, AVX2 정수
내적 사슬의 알고리즘이 거기서 왔고, 정확도는 지금도 ggml 출력과 대조해 정의한다. 알고리즘을
가져온 자리는 소스 주석이 원본 파일과 줄 번호로 가리킨다 — Rust 포팅에서 흔한 방식이고,
[candle](https://github.com/huggingface/candle)의 `k_quants.rs`가 같은 형태다.
기준 속도는 [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)로 같은 기계에서 잰다.
둘 다 MIT다.

라이선스는 MIT, `LICENSE`.

