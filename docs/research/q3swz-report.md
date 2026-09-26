# q3swz 보고 (GQA 프리필 flash의 스위즐 K/V 타일과 프롤로그 다이어트, 2026-09-26)

> 리드 메모(2026-09-26). 아래는 구현 라운드 `q3swz`(opus)의 보고 원문이다. 설계는 `q3fstep-design-report.md`의 S와 레버 3이다.
> K/V 타일을 256 B 행으로 두고 16 B 청크를 `row & 7`로 XOR 스위즐했다. 스테이징 저장, cp.async 목적지, ldmatrix 주소가 모두
> `staged_word` 한 함수를 거친다. 쿼리 프롤로그는 `LDG.64` 32개를 모두 띄운 뒤 `cvt.rn.f16x2.f32`와 NaN 선택으로 변환한다. 그 변환은
> `gate_p4`의 `half_encode` 절이 f32 2³²개 전부로 확인한다(FAIL-first: NaN 선택을 빼면 `nan_mismatches=33554428`, 부호를 빼면
> `16777214`). 비트는 main과 같다(ubatch `--dump` md5가 기준 빌드와 같다). 시팅 18에서 층 24 한 런치가 q3fa의 2,765,883에서
> 2,319,186 사이클이 됐다. L2→SM 섹터는 바이트 하한보다 236개 많았고, 텐서 파이프는 59 %에서 70.4 %로 올랐다(rig-log
> 09-26#q3swz-ncu). 레지스터 196, 블록당 smem 32,768 B, spill 0이다. 5절 D의 출력은 가중치와 활성값의 이름이 서로 바뀌어 있다. 리드
> 스펙의 오류를 그대로 이어받은 것이다. 1.500배 행이 활성값 q6 복사이고, `[R104+…]`의 1.75·1.625·1.607배 행이 가중치 qs다
> (`q3gemmsrc-report.md` 2b절).

라운드는 끝났습니다. `gqa_prefill_flash`의 K/V 타일을 256 B 행에 `chunk ^ (row & 7)` XOR 스위즐로 바꿨고, 프롤로그 다이어트와 전수 게이트, 소스 페이지 계측기를 넣었습니다.

- **비트:** 베이스 `a2a4ed5`와 비트 동일합니다. `--dump` md5 여섯 개가 모두 같습니다.
- **게이트:** 지정된 게이트가 전부 녹색이고 lint는 144입니다.
- **예측을 빗나간 두 항:**
  - 레지스터: 예측 200–210에 대해 첫 빌드가 242였고, lane 기저를 명시해 196으로 고쳤습니다.
  - 프롤로그: 예측 450–650에 대해 실측 999입니다. 쿼리 스테이징 구간만 보면 526으로 밴드 안이고, 밴드가 넣지 않은 나머지 구간 약 470이 초과분입니다.
- **시간 측정:** 하지 않았습니다. 섹터와 사이클 판정은 리드의 카드 몫입니다.

## 1. 바뀐 파일 (워크트리 `/Users/hckim/repo/bloomery-q3swz`, 기준 `a2a4ed5`, git 상태 변경 없음)

- `crates/gpu/src/flash_gqa_prefill.rs`
  - `ROW_W`(272 B 패딩)를 없애고, 256 B 무패딩 행과 단일 소유자 `staged_word(row, w)`를 두었습니다.
  - 레이아웃 사실을 컴파일 시점 단언으로 박았습니다: `staged_pairs_hold`, `staged_offsets_hold`, 128 B 줄, `ROW_CHUNKS % 8`, `KEY_TILE % 8`, `PASS_KEYS % 8`.
  - KT/VT는 `SharedArray<u32, TILE_WORDS, 128>`로 128 B 정렬했습니다.
  - ldmatrix lane 기저를 K와 V에 4개씩(`HALF_PAIRS`) 루프 밖에서 만들고, 나머지 주소는 즉치로 붙입니다.
  - 프롤로그는 LDG.64 32개를 먼저 띄우고, 위치당 기저 하나와 즉치로 주소를 만든 뒤 `f32x2_to_f16x2_bits`로 변환합니다.
  - 모듈 문서의 스테이징 문장을 지금 레이아웃에 맞게 고쳤습니다.
- `crates/gpu/src/flash.rs`: `dev_exp` 옆에 `f32x2_to_f16x2_bits`를 새로 두었습니다(`cvt_f16x2_f32` + 레인별 NaN이면 `sign|0x7e00` 선택). `mma_row_words`는 그대로입니다.
- `crates/gpu/src/elem.rs`: `half_encode_check` 커널(2³² 패턴, atomics, 2²⁰ 참조 표본)과 `enqueue_half_encode_check`, 상수 `HALF_ENCODE_{SAMPLES,MUL,REPORT}`를 넣었습니다.
- `crates/gpu-gates/src/bin/gate_p4.rs`: `half_decode` 옆에 `shape op=half_encode …` 절을 추가했습니다.
- `tools/ref/ptx-shapes.tsv`: `half_encode_check 0 0`을 generate_ds41과 gate_e2e 두 곳에 핀했습니다.
- `tools/ref/q3pp.py`: `source` 하위 명령을 추가했습니다. 명령별 L2 Theoretical/Ideal을 런치 페이지 합으로 내고, 1.05를 넘으면 FLAG를 답니다.
- `tools/ref/ncu-gpu.sh`: `source_sectors`를 gemm과 q3pp 두 폼의 요약 뒤에서 호출하고, dry 출력과 머리말 설명도 넣었습니다.

## 2. 종이 분해 (코드 전에 작성, 원본 `…/scratchpad/q3swz/paper.md`)

설계와 다른 것은 레지스터와 프롤로그 두 항뿐입니다(아래 표시).

- **스위즐 식:** `row·64 + (((w>>2) ^ (row&7))<<2) + (w&3)`
  - XOR은 청크 비트 0–2만 바꾸므로 128 B 반행 안에서만 움직입니다. 행마다 전단사입니다.
- **쌍 규칙:**
  - 워프 wid, lane l의 키는 `2wid + l/16 + 8i`입니다.
  - 소스 lane (2j, 2j+1)은 정렬된 32 B 섹터 하나입니다.
  - 목적지 `(2j)^r`, `(2j+1)^r`는 `{2m, 2m+1}`이라 정렬된 smem 섹터 하나에 들어갑니다(타일 베이스 32 B 정렬 필요, 그래서 ALIGN 128).
  - 명령당 쌍은 16개입니다(이전 24, 1.5×). 128 B 줄 규칙으로 봐도 1.0×입니다.
- **뱅크:** ldmatrix 한 위상은 8행 × 같은 논리 청크입니다. K plain은 `row&7 = lane&7`이고 V `.trans`와 qa도 같은 꼴입니다. 물리 청크 `c^r`이 8개 모두 달라 32뱅크를 한 번씩 씁니다.
- **Q STS:** 워프 32 lane이 한 반행의 논리 청크 8개를 쓰므로 스위즐 뒤에도 32뱅크를 한 번씩 씁니다.
- **목적지 상수:** `(tid/16 + 8i) & 7 = tid/16`이므로 목적지는 `staged(tid/16, 4·(tid%16)) + 512i` 워드입니다. 복사당 추가 명령은 0입니다.
- **레지스터:** 예측 194 → 200–210.
  - **빗나감:** 첫 빌드는 242였습니다. 주소를 `staged_word(key, 8kk+…)`로 그대로 쓰면 ptxas가 ldmatrix 주소 64개에 레지스터를 하나씩 줬습니다. XOR된 주소를 "기저 + 즉치"로 접지 못한 것입니다.
  - lane 기저 4+4개를 owner fn으로 한 번 만들고 `staged_offsets_hold`로 오프셋을 증명하는 형태로 바꾸자 **196**이 되었습니다.
  - 이 레이아웃은 명시적 기저가 있어야 레지스터 수가 유지됩니다.
- **smem:** 블록당 32,768 B입니다(ptxas 실측 `32768 bytes smem`). 2블록/SM이 유지됩니다.
- **프롤로그:** 예측 3,877 → 450–650. 실측은 루프 머리까지 정적 **999**, 경로 921(q3fa 2,647)입니다.
  - **빗나감:** 쿼리 스테이징 구간(첫 쿼리 LDG부터 첫 BAR까지)은 3,360 → **526**으로 밴드 안입니다.
  - 초과분은 설계 밴드에 없던 나머지 구간입니다(카운트·fault 216, BAR 뒤 qa·K(0)·lane 기저 257). 이 구간은 원래 약 470이었고 거의 그대로입니다.
  - 헬퍼는 쌍당 약 11 명령으로, 설계 8–9보다 많습니다.
- **첫 대기 전 쿼리 로드:** q3fa 8 → **26**/32입니다(예측 32).
  - 설계의 "5"는 표의 `LDG<wait`(카운트 로드 포함)이고, 8은 같은 q3fa 리스팅을 쿼리 로드만으로 읽은 값입니다.
  - ptxas가 앞 값의 FSETP.NAN을 마지막 6개 로드보다 먼저 배치했습니다.
- **C의 FAIL-first 예측:** 셀렉트를 빼면 33,554,428 / `0x7f800001`, 부호를 빼면 16,777,214 / `0xff800001`. 둘 다 적중했습니다(5절).
- **자원 시간선(층 24, P=4096, 한 런치):**
  - 텐서 1.62 M(59 %), L2→SM 수요 2.16 M(최상위 유닛)이고, S 뒤에는 ×69.34/102.96 ≈ 1.45 M로 봅니다[유도].
  - DRAM 약 0.45 M, 호스트·PCIe·NVMe는 0입니다.
  - 벽시계는 K/V 버스트의 cp.async 도착 지연입니다(SM당 48 KB → 32 KB).
- **설계 참조:**
  - FA2의 `Swizzle<3,3,3>`(`/home/user/mistral.rs/mistralrs-flash-attn/kernels/kernel_traits.h:70-83`)와 반행 안의 순열이 같습니다.
  - 다른 점: FA2는 d=128 행을 8×64 아톰 두 개로 떨어뜨려 둡니다(`kBlockKSmem=64`; `:81` 주석 "kHeadDim gives wrong results for d=128"). 우리는 256 B 행을 연속으로 두고 XOR을 반행 안에 가둡니다. 뱅크와 섹터 동작은 같고 행 포인터는 하나입니다.
  - exllamav3의 prefill attention은 Triton(`triton_paged.py`)이라 비교할 smem 레이아웃이 없습니다.
  - cuda-oxide에 `cuda_device::swizzle`(`Swizzle<B,M,S>`)이 있지만, 스펙이 모듈 전용 단일 소유자를 요구해서 쓰지 않았습니다. 선택지로만 적어 둡니다.

## 3. 증거 (SASS, ptxas)

**루프 스텝** (`sass_inflight.py --step 0x3e70 --decisions 0x50a0=t,0x5510=t`, q3fa는 `--step 0xf250 --decisions 0x10480=t,0x108f0=t`로 같은 경로):
```
q3fa: instrs=624 … LDGSTS=16 … IMMA=128 … FP32=194 BRA/WARPSYNC=33   other LDSM:64
swz : instrs=621 … LDGSTS=16 … IMMA=128 … FP32=194 BRA/WARPSYNC=33   other LDSM:64
```
- 비-HMMA는 496 → 493입니다.
- ldmatrix 주소는 K가 `[R172|R14|R13|R12 + imm]`, V가 `[R15|R10|R9|R8 + imm]`로 기저 4+4개입니다.

**LDGSTS 목적지:** `[R11+0x800·i]`(V), `[R11+0x4000+0x800·i]`(K)로 레지스터 하나입니다. R11 슬라이스:
```
S2R R10, SR_TID.X ; IMAD.SHL R13,R10,0x4 ; SHF.R.U64 R174,R10,0x4 ; LOP3 R69,R13,0x3c ;
IMAD.SHL R68,R174,0x100 ; SHF.R.U64 R69,R69,0x2 ; LOP3.LUT R11,R69,R174,0x3c(XOR) ; IMAD.U32 R11,R11,0x10,R68
```
- 즉 목적지 = `256·(tid>>4) + 16·((tid&15) ^ (tid>>4))`입니다. 스위즐 식 그대로입니다.

**워프 0, 패스 0의 32 lane 바이트 오프셋:** 32 B 섹터 16개, 쌍 규칙 성립.
```
0x000 0x010 0x020 0x030 0x040 0x050 0x060 0x070 0x080 0x090 0x0a0 0x0b0 0x0c0 0x0d0 0x0e0 0x0f0
0x110 0x100 0x130 0x120 0x150 0x140 0x170 0x160 0x190 0x180 0x1b0 0x1a0 0x1d0 0x1c0 0x1f0 0x1e0
```
- 워프 1도 섹터 16개이고 쌍 규칙이 성립합니다.

**ptxas -v:** `Used 196 registers, used 1 barriers, 32768 bytes smem`, `0 bytes spill stores`. 첫 빌드는 `Used 242 registers`였습니다.

**프롤로그:** 루프 머리까지 정적 3,877 → 999, 경로 2,647 → 921입니다. 쿼리 구간은 3,360 → 526이고, 첫 사용 전 쿼리 로드는 8 → 26입니다.

**마지막 두 편집(lint 수정, `HALF_PAIRS` 이름 붙이기) 뒤:** 명령 텍스트 diff가 `SASS-IDENTICAL`입니다. 코드 생성이 바뀌지 않았으므로 아래 게이트 결과가 유지됩니다.

**q3pp 유도:** `q3pp.flash_consts(<tree>)`가 거부 없이 돌았습니다.
```
{'key_tile': 64, 'positions': 8, …, 'warps': 4, 'threads': 128, 'hmma_per_tile': 128}
```
`mma_m16n8k16` 호출처 4개, `blocks_for`, `live`, `warp_hi`, `qt` 줄은 그대로입니다. 카드 조건의 그리드와 HMMA 수에는 영향이 없습니다.

## 4. 게이트 판정 줄 (실제 출력)

- **`just gate-gpu-p4`** (복원 뒤 재빌드, `Compiling bloomery-gpu`):
  - `shape op=half_encode patterns=4294967296 non_nan_mismatches=0 nan_mismatches=0 first_diff=none ref_sample=1048576 ref_mismatches=0 ref_first_diff=none PASS`
  - `shape op=half_decode … PASS`, `PASSED: elem kernels …`
- **`just gate-gpu-qwen3moe-flash`:** `prefill flash seeded, fault, graph, refusal PASS`, `gate_qwen3moe_flash: PASS`
- **`just gate-gpu-qwen3moe-e2e`:** `gate_qwen3moe_e2e: PASS`가 2회입니다(`flash_mma=true`와 `BLOOMERY_GQA_MMA=0` 두 팔). ubatch 줄은 `…bit for bit PASS`입니다.
- **`just gate-ptx-spill`:** `ptx-spill bin=generate_ds41 entries=158 pinned=158 … violations=0 PASS`, `ptx-spill bin=gate_e2e entries=101 pinned=101 … violations=0 PASS`
- **`just lint`:** `grep -c '^warning:'` = **144**, error 0입니다. 중간에 151이 나왔고, `manual_is_multiple_of` 1개와 매크로 안 `undocumented_unsafe_blocks` 6개를 고쳤습니다.
- **정적 점검:** `just check`와 `just fmt-check`는 rc 0, `check-recipes: ok`, `check-comments: ok`, `check-arch: ok`입니다.
- **`--dump` md5** (`gate_qwen3moe_e2e --gemm-only --dump`): 우리 트리와 베이스(`git archive a2a4ed5`, 원격 `~/repo/bloomery-q3swz-base`)가 **여섯 개 모두 동일**합니다.
```
da5e44eff434b556ed62bfda61c7f05c  u1025-gemm.kv
71a1d02eaac2225f7f1c02d9df0f7cc9  u1025-gemm.logits
013080fc78b3cd41368b940220c94d4e  u1025-gemm.tokens
250cc75a665bbdcb28c36e4885883be9  u1300-gemm.kv
f36cc0052cbf0521f5626cca3081f8da  u1300-gemm.logits
cc6de13d8357fd2ca9994c07dcfcccf8  u1300-gemm.tokens
```
- **`just affected a2a4ed5`:** `crates/gpu`가 바뀌어 GPU 게이트 전부를 고릅니다.
  - 제가 돈 것: gate-gpu-p4, qwen3moe-flash, qwen3moe-e2e, gate-ptx-spill, 그리고 always 정적 7개.
  - **돌지 않은 것:** gate-gpu-p0 p1 p2 p3 p5 p6 p9 q4k-sel iq ds41-hc e2e hybrid p0b p8 moe ds41-moe ds41-chain-ffn p10 load-v41 load-v41-lock qwen3moe-qknorm qwen3moe-rope qwen3moe-router qwen3moe-down gemm qwen3moe-experts qwen3moe-kernels head mcol lib ds41-rope ds41-comp ds41-index ds41-lib ds41-engram ds41-chain-glue ds41-woa ds41-attn ds41-chain-attn ds41-step ds41-skew ds41-long ds41-faults ds41-draft ds41-dspark-loop ds41-prefill ds41-prefill-q3ksplit ds41-chat ds41-serve, gate-ds41-load, gate-gpu-p8b, dspark-hc dspark-experts dspark-kv dspark-graph, gate-gpu-vision.
  - `unmapped`로 나온 파일(어느 게이트도 읽지 않음): `ncu-gpu.sh`, `q3pp.py`, 카드. D의 증거는 아래 오프라인 실행뿐입니다.

## 5. FAIL-first (C)와 D의 오프라인 출력

| 변이 | md5 | 결과 |
|---|---|---|
| 정상본 | `5c52435756a0cfc435244fef3d7e4279` | |
| ① NaN 셀렉트 제거(헬퍼 = `cvt_f16x2_f32(lo, hi)`만) | `50488d8188c0d8c3d5e740e20ebf8c51` | 빨강 |
| ② 부호 제거(`0x7e00`만) | `8bad57ccbf7498791bcaacf98ab70f88` | 빨강 |

- 두 변이 모두 `Compiling bloomery-gpu v0.1.0 (…/crates/gpu)`가 찍혔고, rc 1입니다.
- ① 빨간 줄: `shape op=half_encode patterns=4294967296 non_nan_mismatches=0 nan_mismatches=33554428 first_diff=0x7f800001:host=0x7e00 … FAIL`
  - 예측과 정확히 같습니다.
  - `non_nan_mismatches=0`이라는 점이 `cvt.rn.f16x2.f32`가 NaN이 아닌 모든 f32에서 소프트웨어 반올림과 같다는 증거입니다.
- ② 빨간 줄: `… nan_mismatches=16777214 first_diff=0xff800001:host=0xfe00 … FAIL` (예측 적중)
- 복원 뒤 md5가 `5c52…`로 돌아왔고, 재빌드 로그에 `Compiling`이 다시 찍혔으며 p4는 녹색입니다.

**D** (`python3 tools/ref/q3pp.py source` on `/root/bloomery-data/ncu/ncu-gemm-gemm_q4k_moe_t4096-082817.source.csv`, 원본은 읽기만 하고 원격 `target/q3swz/`에 복사해 실행):
- 런치 페이지 4개를 합산했습니다. 명령 70개 중 34개가 1.05를 넘었고, 섹터는 221,484,072 대 이상치 135,938,944(1.629×)입니다. rc 0.
- 가중치 타일 LDGSTS 1.500 (`+0xca0 @P3 LDGSTS.E.BYPASS.128 [R11][R4.64]`, `+0x7350`, `+0x9b00` …)
- 활성 복사 1.750 / 1.625 / 1.607 (`+0x1040 [R104+0x10]`, `+0x1a70 [R104+0x2810]`, `+0x43d0 [R104+0x10]` …)
- 그 밖: 2.000 ×6(`[R115]`, `[R17+0x5400]` …), 4 B `LDGSTS.E` 7.1–7.98(접근 크기 효과), `LDG.E +0x710` 1.193, STG 전부 1.000.

## 6. 하지 못한 것

- 시간·ncu 측정은 금지라 하지 않았습니다. 섹터 69.34 M과 사이클 2.14–2.50 M 판정은 카드 `docs/cards/q3swz-ncu.card`(읽기만, 편집 안 함)의 몫입니다.
- 프롤로그 450–650과 레지스터 200–210은 위처럼 빗나갔습니다. 레지스터는 고쳤고, 프롤로그 쿼리 구간은 밴드 안입니다.
- 첫 대기 전 로드 32는 26으로 끝났습니다. ptxas의 스케줄이라 소스에서 더 밀지 않았습니다.
- **리드가 정리할 것:** 박스에 원격 보조 디렉터리 `~/repo/bloomery-q3swz-base`(전체 `target/` 포함)가 남아 있습니다. 워크트리를 지운 뒤 `just box-tracks --remove`가 aux로 지웁니다. `box-gc`는 처음과 끝에 돌렸고, 결과는 `found 0 process(es)`와 `gc-done`입니다.

## 7. 스펙 밖 개선 지점 (보고만, 손대지 않음)

- `crates/gpu/src/flash.rs:345-357` `f32x2_to_f16x2_bits`가 쌍당 약 11 SASS입니다. PRMT로 부호를 모으고 LOP3 하나로 NaN 패턴을 만들면 쌍당 −3, 프롤로그 −96쯤입니다. XS.
- `crates/gpu/src/gemm.rs:115` `WT_ROW_W = 20`과 `:106` `B_COL_W = 36`는 같은 부류입니다. D의 출력으로 확인됐습니다(가중치 타일 1.50, 활성 1.607–1.75, 4 B 헤더 7.1–8.0×). S–M.
- 같은 gemm 페이지의 `[R115]`, `[R17+0x5400]`, `[R122+0x5800]` 명령 6개가 정확히 2.000입니다. 목적지와 소스 정렬이 둘 다 어긋난 경우로 보이며, 따로 읽을 가치가 있습니다. XS(판독).
- 스크래치 `seg.py`, `path.py`, `prowalk.py`, `qpro.py`를 `tools/sass_inflight.py --segments`나 `--prologue`로 올릴 것. 이번 프롤로그 수치를 전부 손 스크립트로 냈습니다. S.
- `tools/ref/q3pp.py` summary에 기대 섹터(Σhi·512 + Q + n_keys)를 한 줄로 찍으면 카드 조건 69.34 M ±0.1 %를 도구가 판정합니다. XS.
- `bloomery-gpu` 빌드마다 나오는 `DynamicSharedArray … shared_mem_bytes` 경고 1개는 제 변경과 무관한 기존 것입니다. 출처를 한 번 읽을 가치가 있습니다. XS.
- 기기 쪽 소프트웨어 `f32_to_f16_bits` 사용처는 이제 전수 게이트가 있는 헬퍼로 비트 동일하게 바꿀 수 있습니다: `flash_gqa.rs:532-533`, `flash.rs:462-463` 부근, `rope_neox.rs:215-219`, `fused.rs:369, :395-396`. 각 S.

## 8. 모델

opus(Opus 5.5)로 스폰되어 그대로 돌았습니다.
