# Stage 0 결과 — Q3_K gemv (Rust/cuda-oxide) vs ggml mmvq

같은 텐서(`blk.1.ffn_gate_exps.weight`, DeepSeek-V2-Lite-Chat Q3_K_M),
같은 카드(RTX 3090), 같은 분(2026-09-18T21:33Z)에 `tools/ref/measure.sh`로
연속 측정. 프로토콜: shape당 20 warm-up + 200 timed. CPU 참조값은
`ggml_internal_get_type_traits(GGML_TYPE_Q3_K)->to_float` 역양자화 + f32 내적.

## 본표 (µs/launch, GB/s = weight 바이트/시간, max rel err)

| engine | shape | M | µs | GB/s | max rel err | rust/ggml (GB/s) |
|---|---|---|---|---|---|---|
| ggml mmvq | expert0 | 1 | 10.48 | 118.26 | 4.361e-03 | — |
| mulle rust | expert0 | 1 | 16.96 | 73.07 | 1.879e-07 | 0.62 |
| ggml mmvq | expert0 | 8 | 16.13 | 76.83 | 4.429e-03 | — |
| mulle rust | expert0 | 8 | 71.27 | 17.39 | 1.786e-07 | 0.23 |
| ggml mmvq | stack | 1 | 238.18 | 332.93 | 4.119e-03 | — |
| mulle rust | stack | 1 | 565.74 | 140.17 | 1.315e-07 | 0.42 |
| ggml mmvq | stack | 8 | 473.05 | 167.63 | 3.928e-03 | — |
| mulle rust | stack | 8 | 2426.80 | 32.68 | 1.297e-07 | 0.19 |

- 정확도 게이트(≤ 1e-4) 통과: rust 전 shape ~1e-7. ggml 오차 ~4e-3은
  q8_1 activation 양자화 탓으로 예상 범위이며 실패가 아니다.
- 성능 목표(stack에서 ggml의 0.9배 이상) 미달: stack M=1 0.42배,
  M=8 0.19배. 아래 최적화 1pass 전후 기록 후 중단한다.

## 최적화 1pass 전후 (stack GB/s, 같은 박스·같은 카드에서 실측)

| 버전 | stack M=1 | stack M=8 | 비고 |
|---|---|---|---|
| v0: warp/row, smem 없음, 바이트 로드 | 60.49 | 8.49 | 첫 동작 버전 (rel err ~1e-7 확인) |
| v1: x를 블록당 smem에 1회 스테이징, warp 락스텝 | 123.92 | 16.32 | x 재읽기 제거가 주효 |
| v2: 32행/블록(1024스레드), qs/hmask u32 로드 | 126.37 | 29.85 | M=8 스테이징 트래픽 1/4 |
| v3: 스칼라 누산기(a0–a7), aux if-chain | 140.17 | 32.68 | local-memory 배열 제거 (본표 수치) |

`#[unroll]`은 시도했으나 cuda-oxide 장치 코드젠 ICE
(`APInt::shl: bitwidth mismatch (64 vs 32)`)로 컴파일이 안 돼 제외 —
툴체인 버그로 upstream 보고 대상. expert0은 L2 상주로 latency-bound라
튜닝 대상에서 제외(스펙대로 보고만).

## 증인 블록 (실측 로그 `measure-final.log`에서 발췌)

```
--- witness pre-ref 2026-09-18T21:33:19Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 30.73 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 297.75 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.50 1.72 1.88 2/1388 2431066
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1041247660
--- witness post-ref 2026-09-18T21:33:21Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 0 %, 30.47 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 295.34 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.50 1.72 1.88 2/1391 2431217
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1041251579
--- witness pre-rust 2026-09-18T21:33:21Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 6 %, 76.53 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 295.34 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.50 1.72 1.88 2/1391 2431233
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1041251579
--- witness post-rust 2026-09-18T21:33:23Z ---
index, name, memory.used [MiB], utilization.gpu [%], power.draw [W]
0, NVIDIA GeForce RTX 3090, 1 MiB, 100 %, 237.30 W
1, NVIDIA RTX A6000, 38742 MiB, 100 %, 296.48 W
compute-apps-3090:
pid, used_gpu_memory [MiB]
loadavg: 1.54 1.73 1.88 2/1393 2431400
pressure-io avg10: some avg10=0.00 avg60=0.00 avg300=0.00 total=1041252135
```

- 3090 compute-apps 질의는 timed section 직전 2회 모두 빈 목록 → 대기 없이 진행.
- A6000은 100 %로 바쁜 타 작업이 상주 (예상 범위, 기록만).
- 빌드: `cargo oxide run`에 `--release` 플래그 없음(help로 확인).
  기본 빌드가 이미 `release` profile [optimized]이며 `--arch sm_86` 지정.
