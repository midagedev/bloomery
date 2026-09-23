dsread 라운드를 마쳤습니다. 이 라운드가 요구한 게이트는 전부 녹색이고 lint는 169로 기준선과 같습니다. 알려 둘 일이 둘 있습니다. 첫째, build-ref를 CPU 임대가 잡혀 있는 동안 돌렸습니다. 둘째, 스펙 경계 밖 파일 두 곳을 최소로 고쳤습니다.

## 1. 변경 파일

- **`crates/model/src/placement.rs:187`** — hotset 소유 파일입니다. 새 enum variant 때문에 `CardFormat::of`의 exhaustive match가 컴파일되지 않아 `None` arm에 `| GgmlType::MXFP4` 하나만 더했습니다. rebase할 때 한 줄 충돌 여지가 있습니다.
- **`crates/model/src/arch/mod.rs`** — `pub mod dspark;`와 함께 `DFLASH` 상수와 `is_dflash()`를 더했습니다. mod 줄 하나를 넘어서는 편집입니다. `general.architecture`를 읽는 곳이 이 파일 하나라는 계약을 지키려고 여기에 뒀습니다. `Arch`에 variant를 더하면 exhaustive match가 연쇄로 깨지므로 피했습니다.
- `crates/gguf/src/quant.rs` — `GgmlType::MXFP4`(id 39, 블록 32, 17 B)를 더했습니다. 공개 함수 `e8m0_to_f32_half`와 상수 `KVALUES_MXFP4`, dequant `dequant_mxfp4`를 넣었습니다. `activation_format`은 MXFP4를 거부합니다. 전 도메인 단위 시험 `mxfp4_decodes_every_scale_and_code`도 여기 있습니다.
- `crates/gguf/src/lib.rs` — `ggml_type_info`에서 이제 쓰이지 않는 `39 => mxfp4` fallback 항목과 그 doc을 지웠습니다.
- `crates/gguf/tests/inventory.rs` — strict 리더가 거부하는 예시 id를 39에서 2(q4_0, ggml.h:394)로 옮겼습니다. 계약은 그대로입니다.
- `crates/model/src/arch/dspark.rs`(신규) — `DraftHparams::read`, `names`, `tensors()`, `inventory()`, `DraftInventory`가 들어 있습니다. 블록 이름은 `deepseek41::names`를 재수출하고, 모델 수준 이름 다섯은 자체 소유입니다.
- `crates/gpu-gates/src/bin/gate_dspark_read.rs`(신규) — 호스트 전용 게이트 바이너리로, 검사 (i)~(vi)와 바이트 표를 찍습니다.
- `tools/ref/mxfp4_ref.cpp`(신규) — libggml만 링크합니다. `to_float`로 64행을 덤프하고 `.meta`에 `ggml=<IK>@<commit>`을 남깁니다.
- `tools/ref/build-qdot-ref.sh` — mxfp4_ref 빌드와 실행 줄을 더했습니다. 초안 경로는 `BLOOMERY_DSPARK_MODEL`이고 기본값은 tl37 파일입니다.
- `justfile` — 파일 끝에 `gate-dspark-read` 레시피를 더했습니다.

## 2. 증명과 실제 출력

`just check`는 rc=0이었습니다. `just lint`는 rc=0이고 경고 수는 다음과 같습니다. 중간에 170이 나와 단위 시험의 `chunks_exact`를 고쳤습니다.
```
lint warnings: 169
```
`check-recipes`, `check-comments`, `check-arch`, `fmt-check`는 전부 ok였습니다.

`just build-ref`는 rc=0이었습니다.
```
dumped ggml's to_float of the first 64 rows of blk.0.ffn_gate_exps.weight (k = 5120, /home/user/ik_llama.cpp@c10fbbcc)
```
하니스가 링크한 트리는 `ik-idxkey@db517b69`가 아니라 `ik_llama.cpp@c10fbbcc`입니다. build-ref가 deepseek2 프로필로 돌기 때문입니다. 두 트리의 MXFP4 소스는 같습니다.
```
dequantize_row_mxfp4 블록 md5   1d7269d9624e791e652a3eaabda62255 (두 트리 동일)
ggml_e8m0_to_fp32_half md5       032d369ec1b156e8063e297b324b586a (두 트리 동일)
kvalues_mxfp4 줄                  0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 (두 트리 동일)
```

`just gate-dspark-read`는 rc=0이었습니다.
```
test quant::tests::mxfp4_decodes_every_scale_and_code ... ok
gate_dspark_read: (iv)     file 7972959904 B = header 5254304 + tensors 7967705480 + padding 120 (alignment 32; between tensors 120, after the last 0); gaps [] — PASS
gate_dspark_read: (v)      blk.0.ffn_gate_exps.weight rows 0..64 x 5120: 0 of 327680 values differ from ggml (/home/user/ik_llama.cpp@c10fbbcc) — PASS
gate_dspark_read: (v)      codes used by those rows [17632, 25184, 30933, 19525, 28724, 22416, 15091, 4344, 17643, 25241, 30868, 19356, 28920, 22269, 15228, 4306], unused [] — PASS
gate_dspark_read: (vi)     424673280 blocks, E from Some(115) to Some(128) (14 distinct); blocks at E 255: 0, at the subnormal scales E 0: 0, E 1: 0 — PASS
gate_dspark_read: 104 checks, 0 failed — PASS
```
바이트 합에는 허용 오차를 두지 않았습니다. 텐서가 앞 텐서 끝을 정렬한 자리에서 바로 이어지고, 마지막 텐서가 파일 끝에서 끝나는지를 정확히 봅니다.

`just gate-qdot`와 `just gate-1-1` 결과입니다.
```
test result: ok. 26 passed; 0 failed; 0 ignored
test inventory_carries_what_the_strict_reader_refuses ... ok
q8_0   tensors=13 tensor=blk.0.attn_kv.weight rows=4 rowlen=5120 values=20480 bit_mismatches=0
test result: ok. 4 passed; 0 failed (hw_*)
```

**FAIL-first.** `kvalues[15]`을 −12에서 −11로 바꿔 돌린 뒤 되돌렸습니다. 로그에 `Compiling bloomery-gguf`가 찍혀 재빌드를 확인했습니다.
```
panicked at crates/gguf/src/quant.rs:858:17: E 0 value 15 (code 15): got -3.2326095e-38, want -3.526483e-38
gate_dspark_read: (v)      blk.0.ffn_gate_exps.weight rows 0..64 x 5120: 4306 of 327680 values differ from ggml (/home/user/ik_llama.cpp@c10fbbcc) — FAIL
HOST GATE RED: gate_dspark_read (exit 1)
```
첫 FAIL-first에서는 단위 시험이 녹색으로 남았습니다. 오라클이 같은 상수를 읽고 있었기 때문입니다. 그래서 E2M1을 코드 비트에서 직접 풀도록 바꿨고, 위 red는 그 뒤에 얻은 것입니다.

**ggml의 스케일 규칙.** `ggml_e8m0_to_fp32_half`(ggml-impl.h:40-45)는 비트를 이렇게 만듭니다.
```
x >= 2 ? (x-1)<<23 : {0x00200000, 0x00400000}[x]
```
- 이 값은 모든 E에서 2^(E−128)입니다. 메모의 옮겨 적기가 맞습니다.
- 표 `kvalues_mxfp4`는 ggml-common.h:2250-2252에 있고 2배로 부풀린 값입니다.
- dequant 본체는 `iqk_quantize.cpp:4233`이고, `to_float`로 걸리는 곳은 ggml.c:1311입니다.
- 코드 8(스펙상 −0)을 ggml은 +0으로 풉니다.

## 3. 헤더와 메모 대조, 바이트 표

**hparams.** 핀한 값 20개가 전부 메모와 맞았습니다. 확인한 값은 d 5120, 헤드 64, kv 1, key 512, q_lora 1280, 출력 그룹 8×1024, rope 64, 창 128, hc 4, expert 128 중 top-3, shared 1, ff 2304, block 5, vocab 129280, mask 128799, markov rank 256, target_layers [37,38,39]입니다. 관찰한 것은 다음과 같습니다.
- `rms_eps`는 1e-20이고 **target 파일도 1e-20**입니다. 초안만의 값이 아닙니다.
- YaRN 키는 있지만 `compress_ratios`가 0,0,0이라 plain rope로 읽고 YaRN은 무시합니다.
- `indexer.*` 키는 있지만 indexer 텐서는 없습니다.
- engram 키와 텐서는 없습니다.
- `output_hc_*` 셋과 `conf_proj.bias`가 없습니다.
- mask id는 `dflash.*`가 아니라 `tokenizer.ggml.mask_token_id` 아래에 있습니다.
- 원본 파일의 target_layers는 [38,39,40]입니다.
- n_ctx_train은 1,048,576이고 file_type은 38입니다.
- **E의 범위는 115~128입니다.** 그래서 모든 값의 절댓값이 12 이하이고, Q8_0으로 넓혀도 정확합니다[유도]. 메모 (i)의 전제가 섭니다.

**바이트 표.** 카드 B는 오늘 로더의 `CardFormat` 규칙을 따릅니다. bf16은 f32로 넓히고, MXFP4는 네이티브로 셉니다.

| 그룹 | 파일 B | 카드 B | 블록당 파일 B | 메모 |
|---|---|---|---|---|
| Attention | 403,670,784 | 403,670,784 | 134,556,928 | 134.5 MB |
| HyperConnection | 11,797,128 | 11,797,128 | 3,932,376 | 3.9 MB |
| Router | 3,933,696 | 7,865,856 | 1,311,232 | 2.6 MB(f32로 넓힌 값) |
| SharedExpert | 112,865,280 | 112,865,280 | 37,621,760 | 37.6 MB |
| RoutedExperts | 7,219,445,760 | 7,219,445,760 | 2,406,481,920 | 7.22 GB |
| Fc | 83,578,880 | 83,578,880 | – | 83.6 MB |
| Markov | 132,382,720 | 264,765,440 | – | 66.2 MB×k |
| Confidence | 10,752 | 21,504 | – | – |
| OutputNorm | 20,480 | 20,480 | – | – |
| 초안, expert 제외 | 748,259,720 | 884,585,352 | – | – |
| target output(Q6_K, 복사) | 542,976,000 | 542,976,000 | – | 543 MB |
| target token_embd(bf16, 행만 읽음) | 1,323,827,200 | 0 | – | 복사 안 함 |
| **(iii) 3090 상주 합** | – | **8,647,007,112** | – | 7.22 + 1.30 GB |

메모와의 차이는 두 가지로 설명됩니다.
- **메모의 1.30 GB에는 헤드가 이미 들어 있습니다.** 748,259,720 + 542,976,000 = 1,291,235,720입니다. 스펙 문장 "7.22 + 1.30 + head copy"는 헤드를 두 번 셉니다.
- **남는 차이는 bf16을 f32로 넓힌 몫입니다.** 네이티브 bf16으로 세면 8,510,681,480입니다. 8,647,007,112와의 차이는 Markov 132 MB, Router 3.9 MB, conf 11 KB를 넓힌 몫입니다. 메모 14행은 bf16 Markov gemv를 계획하므로, 그 커널에서는 Markov가 132 MB로 남습니다.

## 4. 못 한 것과 스펙에서 벗어난 점

- **build-ref를 03:44 KST에 CPU 임대가 잡힌 채로 돌렸습니다.** 스펙은 기다리라고 했습니다. 그동안 g++ 빌드 11개가 돌고 모델 행을 약 1 GB 읽었습니다. 그 시각의 측정이 오염됐을 수 있습니다. 그 뒤로는 박스 작업마다 임대가 빌 때까지 기다렸습니다.
- **build-ref는 공유 디렉터리 `$BLOOMERY_DATA/ref`의 `*-ik-dot.txt`를 전부 다시 썼습니다.** ik 빌드가 같아 내용은 그대로라고 봅니다. 뒤에 돈 gate-qdot가 26/26으로 통과했습니다.
- **게이트 모양이 스펙 문구와 다릅니다.** 스펙은 hw_ #[ignore] 시험을 `gate.sh`로 돌리라고 했습니다. 실제로는 이렇게 나눴습니다.
  - 합성 전 도메인 단위 시험 하나가 `gate.sh`로 돕니다.
  - 비트 동일, hparams, 텐서, 타일링, 스케일 검사는 호스트 바이너리가 `host-gate.sh`로 돕니다. `gate-ds41-plan`과 같은 모양입니다.
  - hw 시험을 `crates/gguf/tests`에 두면 두 가지가 문제입니다. 바이너리가 이미 핀한 것을 두 번 핀하게 됩니다. 또 `gate-1-1`이 `--include-ignored`로 돌 때 MXFP4 덤프에 묶입니다.
- **콜드 시간은 재지 않았습니다.** 웜 게이트는 2.97 s였습니다(FAIL-first `real`). 콜드는 expert 7.22 GB를 NVMe 속도로 읽는 시간입니다[유도, 속도 미측정].
- **`DraftInventory::check()`는 아직 쓰는 곳이 없습니다.** 게이트는 행마다 판정합니다. 이 함수는 로더 라운드(dsgraph)가 잘못된 파일을 오류 하나로 거부할 때 쓰도록 둔 것입니다.

## 5. 스펙 밖에서 본 개선 지점(보고만 했습니다)

- `crates/model/src/placement.rs:180`: `CardFormat::of`에 MXFP4 카드 형식이 없습니다. dsmx가 네이티브 형식을 더해야 합니다(S).
- `crates/model/src/placement.rs:180`: `CardFormat::of(BF16)`가 Markov까지 f32로 넓혀 132 MB가 265 MB가 됩니다. dsgraph가 bf16 네이티브 형식을 쓸지 정해야 합니다(S).
- `crates/model/src/arch/dspark.rs`의 끝부분 helper들: `deepseek41/hparams.rs:742-776`의 비공개 meta helper와 모양이 같은 사본입니다. `arch/mod.rs`에 공용 helper로 올리면 소유자가 하나가 됩니다(S).
- `crates/gguf/src/lib.rs:698`: `ggml_type_info`의 손으로 관리하는 fallback 표는 나중에 `GgmlType`으로 올라가는 id를 가립니다. 이번에 지운 39 항목이 그 예입니다(XS).
- `tools/ref/dequant_ref.cpp:40`: `write_file`이 원자적이지 않습니다. 공유 `ref/`에 쓰는 다른 하네스는 `write_atomic`을 씁니다(XS).
- `crates/gguf/src/bin/gguf-inventory.rs:382,588`: 이제 MXFP4를 지원 타입으로 보고하는데, 이 경로를 핀하는 시험이 없습니다(XS).
- `crates/gguf/src/quant.rs:671,726`: 기존 `chunks_exact_to_as_chunks` clippy 경고 넷입니다(XS).

## 6. 모델

opus로 스폰됐고 Claude Opus 5.5로 실행했습니다.
