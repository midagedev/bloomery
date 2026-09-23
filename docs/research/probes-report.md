`verdict-diff.py:46`은 줄 접두어(`^stat (step|summary) `)로 가리고, `time-gate.sh`는 환경을 지우지 않습니다. 보고를 씁니다.

## 라운드 `probes` 보고

두 probe 모두 박스에서 끝까지 동작함을 확인했습니다. E4에는 bench에 읽기 전용 팔 둘(`read-mmap`, `read-thp`)을 넣었습니다. E8에는 `stat` 줄에 `vram_free`를 넣었습니다. 게이트는 전부 녹색이고 lint는 169로 기준선과 같습니다. 배관 실행에서 나온 수치는 측정이 아닙니다.

### 1. 변경 파일

- `/Users/hckim/repo/bloomery-probes/crates/model/src/bin/bench_v41_host.rs`: `Arm`의 bool 둘을 `Shape` enum으로 바꿨고 `read-mmap`·`read-thp` 모양을 넣었습니다. 읽기 디스패치(`read_group`, `lane_bounds`, `fold_bytes`)와 `ThpCopy`(익명 mmap, `MADV_HUGEPAGE` 뒤 `MADV_COLLAPSE`, 검증, Drop 시 munmap)를 추가했습니다. 모듈 doc과 USAGE도 고쳤습니다.
- `/Users/hckim/repo/bloomery-probes/crates/gpu-gates/src/bin/generate_ds41.rs`: `Probe`에 `vram_free`를 넣었습니다. 값은 `m.stages()[0].gpu().mem_info()`로 읽으므로 `crates/gpu`는 고치지 않았습니다. `stat step`에 `vram_free=`를, summary에 `vram_free_load=`와 `vram_free_min=`을 붙였습니다.
- `/Users/hckim/repo/bloomery-probes/AGENTS.md`: 기존 `BLOOMERY_STEP_STATS` 항목 하나만 고쳤고 새 줄은 추가하지 않았습니다. 새 env 레버가 없어서입니다. 병렬 라운드와 겹치면 손으로 다시 적용할 hunk는 아래와 같습니다.
```
-step, host-tier `HybridStats` deltas and `getrusage` page faults, and a `stat
-summary`; unset, nothing is read).
+step, host-tier `HybridStats` deltas, `getrusage` page faults and the card's
+`cuMemGetInfo` free bytes (`vram_free`), and a `stat summary` with
+`vram_free_load` and `vram_free_min`; unset, nothing is read).
```

바뀐 팔 문법은 다음과 같습니다.
```
an arm is <engine|engine-sep|per-matrix|read-mmap|read-thp>:<n_host>[x<rows>]
```
읽기 모양은 한 행만 받습니다. `x<rows>`를 주면 거부합니다.

### 2. 게이트와 증거 (실제 출력)

```
just check        → rc=0, "Checking bloomery-model", "Checking bloomery-gpu-gates", Finished
just lint         → rc=0; grep -c '^warning:' = 169 (both runs); 0 hits in bench_v41_host / generate_ds41
just check-recipes → check-recipes: ok
tools/check-comments.sh → check-comments: ok
tools/check-arch.sh     → check-arch: ok
just fmt-check    → rc=0
```

`--check`는 `just bench-cpu-v41-host-check`로 돌렸고 rc는 0입니다. 아래 check 줄은 `docs/research/ktok-report.md:19`의 줄과 바이트 단위로 같습니다.
```
check layers=[0, 1, 2, 7, 14, 27, 34, 39] experts_per_layer=6 sites=200 failed=0 worst_rel_err=8.689e-7 band=1e-5
PASSED: bench_v41_host check — every site within 1e-5 of its f64 reference
```

`fold_bytes`의 코드 생성도 확인했습니다. 박스 objdump에서 32 B ymm 메모리 피연산자 add가 64 B 블록마다 두 번 나옵니다. 넓은 로드라는 뜻입니다.
```
357d0:	vpaddq 0x20(%rax),%ymm1,%ymm1
357d5:	vpaddq (%rax),%ymm0,%ymm0
```

**E4 배관 점검입니다. 측정이 아닙니다.** 러너는 `just time-cpu-v41-host --threads 32 --rounds 1 --seconds 5 --warmup 1 --arms read-mmap:5,read-thp:5,engine:5`였고 `BLOOMERY_HOST_BOUND=300`을 줬습니다. 돌리기 전에 `flock -n`으로 임대가 비었는지 확인했습니다(rc=0). 러너가 임대를 잡았습니다.

1회차는 collapse 코드가 들어가기 전입니다.
```
v41host thp_copy bytes=16172974080 mem_available=263647285248 thp_enabled=madvise copy_s=9.44 anon_huge_kb=10584064 verified=[4096 B at both ends of every matrix, every byte of layers [0, 20, 39]]
summary threads=32 arm=read-mmap:5 ... gbps_mean=140.28 gbps_best=141.51 admissible=yes
summary threads=32 arm=read-thp:5 ... gbps_mean=141.21 gbps_best=142.24 admissible=yes thp_enabled=madvise anon_huge_kb=10584064
summary threads=32 arm=engine:5 ... gbps_mean=130.44 gbps_best=131.19 admissible=yes
v41host read_sink=0xd20abc9278a332ae
dispatch ... arm=read-mmap:5 kind=gate+up bytes=50688000 ... | dispatches_per_token=80.00 expected=80
```

2회차는 collapse 코드가 들어간 뒤이고, 다른 트랙이 빌드하던 창이었습니다.
```
v41host thp_copy bytes=16172974080 mem_available=263129448448 thp_enabled=madvise copy_s=4.51 anon_huge_kb_copied=15200256 collapse=refused(Cannot allocate memory (os error 12)) collapse_s=0.05 anon_huge_kb=15222784 verified=[...]
summary ... read-mmap:5 gbps_mean=131.27 | read-thp:5 gbps_mean=138.94 (tokens=67) | engine:5 gbps_mean=124.95
```

**E8은 3090의 plan_gate에서 기능 실행만 했습니다.** 명령은 `BLOOMERY_BOX_ENV='BLOOMERY_STEP_STATS=1' just gen-ds41 --depth 6 -n 6`입니다.
```
stat step 1 served=40 ... majflt=932 minflt=28962 vram_free=1404895232
stat step 2 ... vram_free=1404895232
stat step 3 ... vram_free=1404895232
stat step 4 ... vram_free=1404895232
stat step 5 ... vram_free=1404895232
stat summary steps=5 ... majflt=3977 minflt=132971 vram_free_load=1404895232 vram_free_min=1404895232
```

### 3. 예측과 결과

- **E8 예측(유도):** 재생은 할당을 하지 않으므로 `vram_free`는 적재 직후 값에서 변하지 않는다고 봤습니다. **결과:** 다섯 스텝 모두 같은 값이고 `vram_free_min`과 `vram_free_load`가 같습니다. 예측대로입니다.
  - 이 값은 3090의 plan_gate입니다. 스펙에 적힌 1,510,998,016 B는 A6000의 plan_a 값이라 둘을 한 표에 넣을 수 없습니다. E8이 필요로 하는 수는 아래 명령처럼 A6000에서 1,000스텝을 돌려야 나옵니다.
  - `cuMemGetInfo`는 스텝과 스텝 사이, 시간 측정 구간 밖에서 한 번씩 불립니다. `--time`과 같이 쓸 때는 STEP_STATS를 끈 A/A 팔로 p50이 움직이는지 확인하면 됩니다.
- **E4:** 판정용 예측은 두지 않았습니다. 리드가 쓸 가설만 1회차 방향에서 유도했습니다(유도, 측정 아님).
  - `read-mmap`과 `read-thp`가 거의 같아서 매핑은 틈의 원인이 아닐 가능성이 큽니다.
  - 읽기 팔이 약 141인데 STREAM은 147.7입니다. 이 차이는 정적 분할 풀의 자체 상한으로 보입니다.
  - `engine`(130)과 읽기(141)의 차이는 커널·claim·양자화 몫입니다.
  - 읽기 팔은 엔진의 비용 공식으로 레인을 자르고, 참가자마다 자기 레인만 통째로 읽습니다. steal도 활성화 단계도 없습니다. 그래서 읽기 팔은 "커널 없는 엔진 모양"의 상한 역할을 합니다.

### 4. 하지 못한 것, 주의점

- **THP 비율이 실행마다 달랐습니다.** 1회차는 약 65 %, 2회차는 약 94 %였습니다(유도). defrag가 `madvise`라 fault가 즉석에서 compaction을 시도하는데, 실패하면 4 KiB 페이지로 떨어집니다(`thp_fault_fallback 2588`). `MADV_COLLAPSE`는 ENOMEM으로 거부됐고, 거부는 치명적이지 않게 출력만 합니다.
  - 리드는 summary 줄의 `anon_huge_kb`를 먼저 읽고 그 팔을 믿을지 정해야 합니다.
  - 2회차에서 `--warmup 1`의 첫 read-thp 토큰이 느려 토큰 수가 67로 줄었습니다. 실제 측정에서는 기본값 `--warmup 3`을 쓰면 됩니다.
  - collapse 호출을 유지할지는 리드가 정할 일입니다. 시간 측정 밖의 준비 단계이고, 어느 쪽이든 결과가 출력됩니다.
- **스펙의 "더 작은 n_host로 줄이기"는 복사 크기를 줄이지 못합니다.** 복사본은 n_host와 무관하게 작업 집합 전체입니다(24 × 40층, 16,172,974,080 B). 대신 `MemAvailable`이 두 배(복사본과 작업 집합의 페이지 캐시)를 담지 못하면 거부하게 했습니다. 실측 `MemAvailable`이 263 GB라 폴백은 필요 없었습니다.
- **복사는 populate보다 먼저 합니다.** 복사가 밀어낸 페이지를 populate가 다시 올립니다. 2회차의 `resident_at_open=3951341/3951384`가 그 흔적입니다.
- **스펙의 "mlock'd mmap"은 bench에 없습니다.** bench는 populate만 합니다. mlock은 페이지 크기를 바꾸지 않으므로 TLB 질문에는 같은 조건입니다.
- **`BLOOMERY_STEAL_BLOCKS`는 `engine`, `engine-sep`, `engine:NxR` 팔에만 닿습니다.** 경로는 `group_core → run_group → run_row_pool`(ops.rs:1902)입니다. `per-matrix`는 `matmul_q`의 열 디스패치(ops.rs:1814)로 가고, 읽기 팔은 steal을 하지 않습니다. 값은 프로세스마다 한 번 읽으므로 E3의 각 값은 따로 `just` 줄이 됩니다.
- **E3 행의 `BLOOMERY_HYBRID_OVERLAP=0`은 bench가 읽지 않습니다.** GPU 하이브리드 레버라서 E3에서 이 부분은 `generate_ds41`/`depth-ds41` 쪽 팔입니다. plan.md는 고치지 않았습니다.

**측정 자리에서 리드가 돌릴 명령입니다.** 벽시계 시간은 실측을 바탕으로 한 유도입니다. 프로세스당 준비는 복사 4.5–9.4 s, populate 0.1 s, check 수 초입니다. 1회차는 3팔 × 5 s × 1스레드 수에 25 s가 걸렸습니다. 모두 CPU 임대이고, 합계는 약 10분으로 30분 안입니다.
```
# E4 + E2 together (~4 min: 6 arms × 2.5 s × 2 thread counts + ~20 s setup each)
just time-cpu-v41-host --threads 16,32 --rounds 6 --seconds 6 --arms read-mmap:5,read-thp:5,engine:5,read-mmap:6,read-thp:6,engine:6
# E2 alone if wanted separately (~1 min)
just time-cpu-v41-host --threads 32 --rounds 6 --seconds 6 --arms engine:5,engine:6
# E3 — one process per value, the last line is the cross-process A/A (~1 min each, ~4 min)
BLOOMERY_BOX_ENV='BLOOMERY_STEAL_BLOCKS=4'  just time-cpu-v41-host --threads 32 --rounds 6 --seconds 6 --arms engine:5,engine:6
BLOOMERY_BOX_ENV='BLOOMERY_STEAL_BLOCKS=16' just time-cpu-v41-host --threads 32 --rounds 6 --seconds 6 --arms engine:5,engine:6
BLOOMERY_BOX_ENV='BLOOMERY_STEAL_BLOCKS=64' just time-cpu-v41-host --threads 32 --rounds 6 --seconds 6 --arms engine:5,engine:6
BLOOMERY_BOX_ENV='BLOOMERY_STEAL_BLOCKS=4'  just time-cpu-v41-host --threads 32 --rounds 6 --seconds 6 --arms engine:5,engine:6
# E8 — functional, A6000 placement (a), no lease (~1–2 min at ~40–60 ms/step, derived)
BLOOMERY_CARD=a6000 BLOOMERY_BOX_ENV='BLOOMERY_STEP_STATS=1' just gen-ds41 --place a --depth 6 -n 1001
```
E3는 프로세스마다 다른 창에서 돕니다. 창 사이 흔들림이 약 5 %라서, 1 % 미만의 효과는 이 방식으로 판정할 수 없습니다. 아래 improvement spot의 러너 개선이 그 해법입니다.

### 5. 스펙 밖의 개선 지점 (보고만 했고 손대지 않았습니다)

- `tools/ref/host-rate.sh:47-52`: env 레버를 한 임대 안에서 번갈아 돌리는 방법이 없습니다. ab-decode의 `BLOOMERY_AB_ENVS` 같은 방식이 없어서 E3의 STEAL_BLOCKS 팔은 창이 다른 별도 프로세스로 돕니다. 러너에 env 팔 루프를 넣는 S 크기 작업입니다.
- `tools/ref/host-rate.sh:22-26`: `--threads`는 첫 인자일 때만 인식됩니다. 한 줄짜리 문서 또는 파싱 수정입니다(XS).
- `wave-m4/common.md`와 `spec-probes.md`의 임대 확인 지시: `cat /root/bloomery-cpu.lock`은 파일이 원래 비어 있어서 아무것도 알려주지 않습니다. `flock -n /root/bloomery-cpu.lock true; echo $?`로 바꾸면 됩니다(XS).
- `docs/plan.md:620`: E3 행이 `BLOOMERY_HYBRID_OVERLAP=0`을 bench_v41_host 팔로 적었지만 bench는 그 변수를 읽지 않습니다(XS, 리드 문서).
- `bench_v41_host.rs`의 `Bench::round`: `--warmup 1`이면 느린 토큰 하나가 토큰 수를 정합니다. 설계대로의 동작이라 사용 주의일 뿐 버그는 아닙니다.
- 3090 plan_gate의 E8 기능 실행에서 스텝마다 majflt가 500–1100이었고 호스트 leg는 51–112 ms였습니다. telem이 이미 적은 페이징 압박과 같은 현상이고, 적재 라운드가 조사할 대상입니다.

### 6. 모델

opus로 스폰됐고 Opus 5.5로 실행했습니다.

박스 정리는 처음과 끝 모두 `box-gc` 0건이었습니다. git 상태는 바꾸지 않았습니다. 로그는 `/private/tmp/claude-501/-Users-hckim-repo-bloomery/e5714b41-3850-42be-a722-79e1f225c3b5/scratchpad/probes/`에 있습니다.
