# q3gemmc 보고 (gemm_q4k 스텝을 배리어 하나로, 착륙하지 않음, 2026-09-26)

> 리드 메모(2026-09-26). 아래는 구현 라운드 `q3gemmc`(opus)의 보고 원문이다. 설계는 `q3gemmsrc-report.md`의 F1과 F2다. F1은 스텝당
> 배리어를 하나로 줄이고 다음 스텝의 복사를 이번 스텝의 계산 안에서 낸다. 풀타일은 n-타일 1 뒤에, 부분 타일은 배리어 바로 뒤에
> 낸다. F2는 스테이징의 열 주소를 타일당 한 번 u32 바이트 오프셋으로 계산한다. 비트는 main과 같았다(`gate_gemm` `y_fnv` 328/328,
> ubatch `--dump` md5 여섯 개). 시팅 19에서 T = 4096 한 런치가 1.746 M → 1.715 M 사이클(−1.8 %)로 카드 밴드(1.41–1.53 M) 위였다
> (rig-log 09-26#q3gemmc-ncu). 배리어는 예측대로 줄었지만 그 몫이 계산 안의 대기로 옮겨 갔다. 첫째, 계산 안의 복사 묶음은 주소를
> 레지스터 한 쌍(R52:R53)으로 돌려 쓴다. 그래서 R52를 덮어쓰는 명령마다 앞 LDGSTS가 MIO 큐에서 그 레지스터를 읽어 갈 때까지
> 기다린다(워프-스텝당 약 266 워프-사이클, long_scoreboard). 둘째, 부분 타일 루프는 배리어 바로 뒤의 복사 묶음이 첫 n-타일의 LDS를
> 막아 3,243 → 3,533으로 느려졌다(short_scoreboard 364 → 857). 보고의 유도대로 P = 512의 타일이 모두 부분 타일이라면 pp512가
> 느려질 공산이 커서, 이 트리는 착륙하지 않았다. 다음 라운드는 이 보고와 그 절에서 시작한다. 복사 주소가 비행 중인 LDGSTS의
> 레지스터를 덮어쓰지 않음을 SASS로 먼저 증명하고, 부분 타일 배치(묶음 복사, 옛 두 배리어, n-타일 1 뒤)를 종이 위에서 가른다.
> F2만 있는 판도 재어 바닥으로 삼는다. 보고 2절의 「부분 타일은 배리어 직후 스테이징이 compute 전체로 덮는다」는 복사 도착만
> 따진 것이고, 계산의 LDS가 복사 뒤에 줄 서는 비용은 세지 않았다.

F2와 F1 둘 다 넣었고 D(`srcpage.py`)와 카드도 만들었습니다. 네 GEMM 엔트리 모두 레지스터 128 이하, spill 0입니다. `gate_gemm`의 `y_fnv` 328개가 base와 전부 같고, ubatch `--dump` md5 6개도 base와 같습니다. 타이밍과 프로파일은 돌리지 않았습니다(스펙대로 리드 몫).

도중에 하나를 고쳤습니다. 처음 판은 풀타일과 부분타일 모두 n-타일 1 뒤에 복사를 냈습니다. 그러면 부분타일에서 복사 지연을 덮을 계산이 모자랍니다. 그래서 부분타일은 배리어 직후에 복사를 내도록 바꾸고 전 게이트를 다시 돌렸습니다. 처음 판도 게이트를 모두 통과했으니(e2e 두 패스, dump 6/6) 되돌릴 때 폴백으로 쓸 수 있습니다.

### 1. 바뀐 파일
- `/Users/hckim/repo/bloomery-q3gemmc/crates/gpu/src/gemm.rs`
  - F2: `gemm_stage_cols!`가 스테이징 열 오프셋을 타일당 한 번, u32 바이트 오프셋으로 계산합니다(청크 2개와 d8 블록 1개). 스텝은 여기에 자기 오프셋만 더합니다.
  - `STAGE_CHUNKS`, `STAGE_OFF_MAX`와 그 경계 증명 `const assert`를 추가했습니다.
  - F1: `gemm_step`이 `wait_group(0)` → 배리어 → compute 순서가 됐고 스텝 끝에서 한 번 commit합니다. 다음 스텝 복사는 풀타일이면 n-타일 `STAGE_NT`(=1) 뒤에 한 묶음으로, 부분타일이면 배리어 직후에 냅니다. 행이 없는 워프는 계산 없이 복사만 합니다.
  - `GEMM_THREADS` 주석(옛 77-83)과 `gemm_step`, `STAGE_NT` 주석을 지금 참인 내용으로 다시 썼습니다.
- `/Users/hckim/repo/bloomery-q3gemmc/tools/ref/srcpage.py` (새 파일): 소스 페이지 판독기.
- `/Users/hckim/repo/bloomery-q3gemmc/docs/cards/q3gemmc-ncu.card` (새 파일): profile 카드.

### 2. 종이 위 분해

**자원 시간선.** 스텝을 채우는 유닛은 없습니다(발행 50.6 %, IMMA 35.4 %, L1TEX 52.6 %, L2 66.5 %). 그래서 워프-스텝은 직렬 위상의 합입니다: staging 765 + bar1 357 + compute 3,473 + bar2 304 = 4,899 워프-사이클. F1은 비동기 레버로, staging을 직렬 경로에서 빼서 compute 아래에 겹칩니다. F2는 그 staging의 명령 수를 줄입니다.

**F2**
- 루프 불변이었던 것:
  - acol LDS 3개(0x71d0, 0x7360, 0x74a0).
  - q6 열 주소를 64비트로 계산하던 사슬: IMAD, IMAD.WIDE, LEA/LEA.HI.X, IADD3 쌍 둘. 청크 둘 각각.
  - s8/d8 사슬: `MOV R4,UR21`/`MOV R5,UR22`, IMAD.WIDE 2개, IMAD 2개, IADD3 2개.
- 예측은 워프-스텝당 −20…−32였습니다. 실측은 hh0 630 → **599**(−31), hh1 682 → **650**(−32)입니다. 헤더 복사가 있는 경로는 685 → 653입니다.
  - 0x74b0과 0x9c60의 `MOV R4,UR21` WAR는 사라졌습니다.
  - 루프 한 번당 LDS는 122 → 104, IMAD는 528 → 512입니다.
- **보고서와 다른 점**
  - 0x9d80 IADD3는 루프 불변이 아닙니다. 스텝마다 바뀌는 uniform 베이스에 이미 hoist된 `ws.2`를 더하는 명령입니다. 그 lsb 스톨은 앞 LDGSTS가 아직 읽는 R4에 대한 WAR입니다. 어느 레지스터를 쓰느냐는 ptxas가 정하므로 hoist로는 없앨 수 없습니다.
  - `@!PT LDS` 더미는 스텝당 9 → 3입니다(0이 아님). ptxas는 LDS 뒤 첫 LDGSTS 앞에 더미 3개를 넣고, 이 LDS는 직전 compute의 것입니다.
  - 처음에는 워드 오프셋에 `.add()`를 썼습니다. 이러면 64비트 zero-extend를 ptxas가 보지 못해 615/659가 나왔고 `@!UPT UIADD3` 패딩이 루프 한 번당 26개 늘었습니다. u32 바이트 오프셋과 `byte_add`로 바꾼 뒤 −31이 나왔습니다.
  - 레지스터(보고서 예측 +4): q4k 128, q5k 128, q6k 124, q3k 123.

**F1 레이스 표.** 버퍼는 b[h&1] 두 개입니다.

| # | 쓰기 | 읽기 | 버퍼 | 순서를 주는 것 |
|---|---|---|---|---|
| 1 | stage(h+1) cp.async (스텝 h의 bar 뒤) | compute(h−1) LDS | b[(h+1)&1]=b[(h−1)&1] | bar(h): 모든 워프가 compute(h−1)을 마치고 도착 |
| 2 | stage(h) (스텝 h−1) | compute(h) | b[h&1] | 자기 스레드는 `wait_group(0)`(대기 중인 그룹은 h의 것 하나), 다른 스레드는 bar(h) |
| 3 | stage(h+1) | compute(h) | 서로 다른 버퍼 | 필요 없음 |
| 4 | stage(h+2) (스텝 h+1) | compute(h) | b[h&1] | bar(h+1) |

- 배리어는 `active`(워프 균일)와 `next`(블록 균일) 분기보다 앞에 있습니다. 그래서 스텝당 배리어는 정확히 하나입니다.
- commit은 스텝마다 정확히 한 그룹(비어 있을 수 있음)입니다. 귀납적으로 매 wait에서 대기 그룹은 하나이고, 마지막 스텝 뒤에는 남은 복사가 없습니다.

**복사가 `wait(h+1)` 전에 착지하는 이유.** n-타일 하나는 약 434 워프-사이클입니다(페이지의 compute 3,473 ÷ 8, 유도).
- 풀타일: n-타일 1 뒤에 내므로 6 × 434 ≈ **2,600**이 남아 덮습니다.
- 부분타일: 첫 n-타일 앞에 내므로 compute(h) 전체가 덮습니다. main의 compute 커버와 같습니다.
- 처음 판(모든 타일이 n-타일 1 뒤)에서는 부분타일의 커버가 (nt_live−2) × 434였고 nt_live ≤ 2이면 0이었습니다. P=512에서는 모든 타일이 부분(nt_live 3–5)이라 고쳤습니다.
- q5k와 q3k는 `plain` 걸음이라 늘 부분 경로, 즉 배리어 직후 스테이징입니다.

**비트 논증.** 산술은 하나도 옮기지 않았습니다.
- 복사 주소는 식으로 이전과 같습니다: q6 `(c·col_words + 32i + 4q) + off`, s8 `4·(c·2·n_sb + h)`, d8 `c·2·n_sb + h`.
- compute(h)는 같은 stage를 읽고, IMMA 입력과 isum·에필로그 순서도 그대로입니다.

**보고서가 틀린 항(스펙도 예상하지 않음): F1은 명령 수를 늘립니다.** F2 대비 풀타일 스텝당 +9/+20입니다. 원인은 ptxas 패딩입니다. IMMA의 목적지가 B 조각 소스와 겹칠 때 `@!UPT UIADD3 URZ`가 붙고(루프 본문에 정적으로 0 → 23), hh1의 `next` 분기에 BSSY/BSYNC/BRA가 붙습니다. 시간 이득은 명령 수가 아니라 배리어와 MIO 흐름에서 와야 합니다.

**배치 실측** (풀타일 루프 한 번 / 스텝 평균):

| 배치 | 루프 한 번 | 스텝 평균 |
|---|---:|---:|
| 조각을 n-타일 0–3에 흩음 | 1,311 | 655.5 |
| n-타일 0 뒤 한 묶음 | 1,285 | 642.5 |
| **n-타일 1 뒤 한 묶음 (채택)** | 1,278 | 639 |
| 배리어 직후 | 1,265 | 632.5 |

풀타일에 n-타일 1을 고른 이유는 보고서의 MIO 산개 논리입니다. 배리어 직후보다 스텝당 6.5명령 비쌉니다.

**예측 재유도.**
- staging 명령 약 31개가 compute 안으로 들어가 +60…+190.
- 배리어 하나 300…450 (bar2 304에 bar1이 싣던 도착 차이 최대 +150).
- 패딩 +30…+90.
- 합: 4,899 → 3,863…4,203 (−14.2…−21.1 %).
- 두 루프가 샘플의 89.9 %이므로 런치는 1.746 M × (0.810…0.872) = **1.41–1.53 M** 사이클입니다. 보고서 밴드는 1.43–1.59였습니다.
- P=4096 창: −26…−39 ms (유도).

### 3. 컴파일 증명 (`sass_inflight.py --step`, gate_gemm의 ptxas SASS)

| 판 | hh0 | hh1 (헤더 복사 / 안 함) | BAR/스텝 | DEPBAR | 경로의 `@!PT LDS`/루프 한 번 |
|---|---:|---:|---:|---|---:|
| main 56c1b47 | 630 | 685 / 682 | 2 | SB0,0x1 | 18 |
| F2 | 599 | 653 / 650 | 2 | 0x1 | 6 |
| 최종 (F1+F2) | 608 | 670 / 668 | **1** | 0x0 | 6 |

- 풀타일 루프는 0x6c50–0xc060, BAR는 0x6c80과 0x9440입니다.
- 부분타일 루프는 0x1510–0x6770, BAR는 0x1530과 0x3d10입니다. 스테이징 LDGSTS 묶음은 BAR 직후 0x15a0–0x1710에 있습니다.
- 부분타일 루프 한 번당 명령 수 (nt_live 1…7):
  - main: 485 / 611 / 737 / 863 / 989 / 1115 / 1241
  - 최종: 423 / 551 / 679 / 807 / 935 / 1065 / 1195
- 레지스터 (ptxas와 jit 같음, spill 0, jit_local 0, depot 없음):

| 엔트리 | main | F2 | 최종 |
|---|---:|---:|---:|
| gemm_q4k | 128 | 128 | 128 |
| gemm_q5k | 126 | 128 | 128 |
| gemm_q6k | 126 | 124 | 127 |
| gemm_q3k | 118 | 123 | 125 |

- ptx-scan에서 바뀐 엔트리는 `gate_gemm`과 `generate_qwen3moe` 둘 다 gemm_q3k/q4k/q5k/q6k 넷뿐이고(공유 매크로 때문), 나머지 96개는 md5가 같습니다.
  - q4k `0fe9fbec…` → `45ff4ca2…`
  - q3k `901049e2…` → `901b2190…`
  - q5k `bf622ad5…` → `cbec16e2…`
  - q6k `425c3521…` → `dbbecb65…`

### 4. 게이트 (최종 소스로 실행)
- `just gate-gpu-gemm`: `PASSED: gate_gemm — every run inside its derived band … bit for bit the contract's transcription …`
  - rc=0, PASS 줄 428개, FAIL 0개.
  - `y_fnv` 328개를 순서대로 base 런과 diff한 결과 `y_fnv 328/328 identical to base`.
  - `q4k_stack_misaligned=refused`, ragged 33줄 전부 PASS, `contract_bits_differ` 0이 아닌 줄 0개.
- `just gate-gpu-qwen3moe-kernels`: qknorm, rope, router, down, flash, experts 모두 `…: PASS`.
- `just gate-gpu-qwen3moe-e2e`: `gate_qwen3moe_e2e: PASS` 두 번(flash_mma=true / false).
- ubatch `--dump` md5는 base와 최종이 6개 모두 같습니다: `da5e44ef…` `71a1d02e…` `013080fc…` `250cc75a…` `f36cc005…` `cc6de13d…` (u1025 / u1300의 kv, logits, tokens).
- `just gate-ptx-spill`: `ptx-spill bin=generate_ds41 … violations=0 PASS`, `ptx-spill bin=gate_e2e … violations=0 PASS`.
- `just check` rc=0, `just lint` 경고 **144**(gemm.rs 0개), `just fmt-check` rc=0.
- `check-recipes: ok`, `check-comments: ok`, `check-arch: ok`, `check-rustflags: ok`. Cargo.lock md5는 HEAD와 같습니다.
- 카드: `card: ok kind=profile (no ruler: not an ab card)`.
- 게이트 뒤에 주석만 두 줄 고쳤습니다. 주석을 뺀 코드는 게이트 소스와 같고, 첫 번째 주석 수정 뒤 ptx-scan md5가 그대로임을 확인했습니다. 두 번째 수정은 `///` 주석 한 줄뿐이라 ptx-scan은 다시 돌리지 않았습니다.

### 5. D: 기존 페이지(…-082817)에 돌린 결과
- 중복 블록: `launches: 2 distinct, 2 repeated block(s) dropped`, CPI 7.82 × 발행 명령 291,069,258 → 샘플 하나당 32,377.1 워프-사이클.
- 위상표(워프-스텝당, 보고서와 같음):

| 위상 | 명령 수 | 워프-사이클 |
|---|---:|---:|
| staging | 62.7 | **765** |
| bar1 | 2.0 | **357** |
| compute | 578.5 | **3,473** |
| bar2 | 9.4 | **304** |

- 섹터:
  - 가중치 23,122,944 / 14,229,504 **×1.625**
  - 활성값 19,091,580 / 12,727,720 **×1.500** (루프 안만: 16.42 M / 10.94 M, ×1.500)
  - 헤더 ×2.0, s8 ×2.0, d8 ×7.91
  - 합 55,371,018 / 33,984,736 **×1.629**
  - 보고서의 활성값 19.32 M과 d8 2.93 M은 그룹 경계가 다른 탓입니다. 둘의 합은 같습니다(22.24 M).
- `--listing`을 `sass_inflight --step 0x71d0`에 넣으면 630/685가 나옵니다. JIT 페이지 주소와 ptxas 주소가 같습니다.

### 6. 못 한 것, 달라진 것
- **main에는 ubatch `--dump`가 없습니다.** q3fa의 미커밋 `gate_qwen3moe_e2e.rs` hunk를 base 아카이브 트리와 내 트리의 사본에 똑같이 적용해 md5를 냈습니다. `diff -r` 결과 두 사본이 원본과 다른 곳은 그 파일뿐입니다. 워크트리는 건드리지 않았습니다. 그 hunk는 리드가 따로 착륙시켜야 합니다.
- 임대 확인을 처음에는 `flock -n`으로 했고, common.md가 바뀐 18:4x 이후 `flock -s -n`으로 바꿨습니다.
- `just affected 56c1b47`은 게이트 61개와 상시 검사 7개를 고릅니다. 이 라운드에서는 gemm, qwen3moe-kernels, qwen3moe-e2e, ptx-spill 게이트 넷과 상시 검사 7개만 돌렸습니다. V4.1과 dspark 게이트 등 나머지는 리드의 착륙 배치 몫입니다.
- 박스에 남은 것(정리 대상): aux 디렉터리 `bloomery-q3gemmc-base`, `bloomery-q3gemmc-dump`, 그리고 `~/q3gemmc-dump/{base,mine,split}`. 워크트리와 원격 디렉터리 제거는 리드가 합니다. box-gc는 처음과 끝에 돌렸고 둘 다 0개였습니다.
- 스크래치에 둔 증거(세션이 다시 시작되면 지워짐): `/private/tmp/claude-501/-Users-hckim-repo-bloomery/b4e4a3ef-e524-4e47-91af-3afadc891c1e/scratchpad/q3gemmc/`
  - `*-q4k.raw`: 각 판의 SASS
  - `s-*.txt`: 게이트 로그
  - `nt1-fmt.diff`: 폴백 판

### 7. 스펙 밖 개선 지점 (보고만, 손대지 않음)
- `tools/sass_inflight.py:353`: segs를 BAR 두 개마다 자르기 때문에, 배리어가 하나인 스텝에서는 hh0과 hh1을 가르지 못합니다. `--bars-per-step` 옵션을 추가하면 됩니다. XS.
- ptxas 패딩 두 종류는 F3/F4의 설계 상수입니다. (a) LDS 뒤 첫 LDGSTS 앞에 `@!PT LDS` 3개가 붙으니 복사는 한 묶음으로 둬야 합니다. (b) 레지스터 128에서 IMMA의 목적지가 B 소스와 겹치면 `@!UPT UIADD3`가 붙습니다(F1의 비용). s8/d8의 64비트 IV 레지스터 4개를 푸는 쪽이 레버입니다. S.
- `crates/gpu-gates/src/bin/gate_qwen3moe_e2e.rs:86-89`: ubatch dump가 없습니다. q3fa hunk(36줄)를 따로 착륙시키면 GEMM 라운드마다 교차 빌드 비트 핀을 쓸 수 있습니다. S.
- `tools/ref/ncu-gpu.sh`: gemm 폼이 source page 뒤에 `srcpage.py` 위상표까지 찍으면 한 명령으로 끝납니다. 이번 웨이브에서는 q3swz 소유 파일입니다. XS.
- 아래 셋은 F3 몫입니다.
  - `gemm.rs:656`: 부분 확장에서 s8/d8 복사가 predicate가 아니라 BRA/BSSY로 컴파일됩니다. XS.
  - `gemm.rs:790`: 헤더 복사가 워프 0–3에만 있고 s8/d8은 워프 0–1에만 있어 도착 차이가 생깁니다.
  - d8 4바이트 복사가 섹터 ×7.98입니다.

### 8. 모델
Opus 5.5(`claude-opus-5-5`)로 스폰되어 돌았고, 사용자 직접 입력은 없었습니다.
