# V4.1 게이트 fixture 설계 — 원문

2026-09-26. opus 서브에이전트 설계 라운드 `fixture`의 보고 원문이다(스펙 `specs/wave-m6/spec-fixture.md`, 박스 없음). [유도] 수치는 라운드의 추론이다. 리드가 원본에서 확인한 것과 처분은 `docs/plan-triage.md`의 fixture 항목에 적는다.

## 보고 (`fixture`, 워크트리 `/Users/hckim/repo/bloomery-fixture`, base `d2e25a5`)

박스는 쓰지 않았고 파일도 쓰지 않았습니다. git 상태는 바꾸지 않았고, 사용자 직접 입력은 없었습니다.

**결론.** 이 V4.1 fixture는 층당 실제 모양을 그대로 두고 **9층(층 종류마다 하나), ≈61 GB**입니다[유도]. 크기가 줄어서 이득이 나는 게 아닙니다. 264 GB RAM 안에 통째로 들어가서 매 적재가 warm이 된다는 것이 이득입니다. 지금 V4.1 게이트는 적재할 때마다 카드 23 GB를 NVMe에서 콜드로 읽습니다(`CARD_DONTNEED` 기본값). 스텝 계산도 40층 → 9층이라 약 4배 줄어듭니다. 무거운 엔진 게이트 9개의 항목 시간 합은 1,797 s → 319–456 s로 줄고[유도], 착륙 묶음 벽시계는 ≈42분 → ≈11–13분으로 줄 것으로 봅니다[유도].

## 0. 스펙 전제 하나를 고칩니다: `gate-ptx-spill` 887 s는 콜드 빌드가 아닙니다

- **빌드는 합쳐 75 s입니다.** `…/scratchpad/fixup6/gb1/g-gate-ptx-spill.log`의 빌드 셋이 30.12 + 41.13 + 3.79 s입니다. 같은 로그 9·26행에 `Blocking waiting for file lock on build directory`가 두 번 있습니다.
- **나머지는 3090 게이트 락 대기입니다.**
  - 이 항목은 B 차선에서 05:17:21에 시작해 05:32:08에 끝났습니다.
  - 그 사이 A 차선은 gpu-lib(05:17:36–05:18:33), prefill(05:18:33–05:24:58), prefill-2(–05:31:24), step(–05:32:05)을 돌렸습니다.
  - `tools/ptx-scan.sh:44,105`의 JIT(`oxart_jit`)는 `tools/gpu-gate.sh`로 돌고, 로그에도 `jit-card=NVIDIA_GeForce_RTX_3090`이 찍혀 있습니다.
  - 러너는 이 레시피를 "디바이스 코드 없음"으로 분류합니다(`tools/gate-batch.sh:26-27`).
  - 조용한 gb3 묶음에서는 59 s였습니다.
- **7번 예측에 들어가는 입력입니다.** 러너 분류를 고치지 않으면 fixture 티어 뒤에도 ptx-spill은 A 차선 뒤에 줄을 섭니다.

## 1. 게이트 표

범례: (a) 자기 일관성, (b) 오라클, (c) 실파일. "헤더" 표시는 모듈 `//!` 머리말과 검사 상수·코드 줄만 읽었다는 뜻입니다. 헬퍼 다섯이 429로 죽어서 제가 직접 읽은 그룹이 여기에 해당합니다(10번).

### 1-A 엔진 전체를 여는 게이트 (검사별)

| 레시피·바이너리 | 검사 (`path:line`) | 등급 | 실파일에 묶는 끈 |
|---|---|---|---|
| **prefill** `gate_deepseek41_prefill` (justfile:882) | attention 개수 초과 → fault `:20-23` · 라우터 decode = batch, −inf/NaN raise `:24-27` · raise 사이트 `:28-35` · fault → reset → clean `:36-43` | a | 없음. 합성 입력, 카드 expert 필요 |
| | 오라클: 4,097 디코드 스텝 `:44-51,:146` · 케이스 12개가 스텝과 비트 동일(ring·state·rows·keys·logits·shadow·taps·cuts) `:52-58,:151` · 넓은 탭, rollback, take back `:63-71` | a | 코퍼스 `corpus-prose.ids` `:205-207`(실어휘 id). **탭 층 = 드래프트 헤더의 target_layers** `:261-263`([37,38,39]) |
| | 분할 (700+400), (1800+1000), (300+2700) `:59-62,:153` | a | **끈**: 1800·2700은 B = 8 + 128·19 = 2440(디코더 19층) 양쪽에 걸리게 잡은 값입니다 |
| | 보고 줄 "layer 20 block cut" `:1352` | 출력만 | **끈**: 리터럴 20 |
| 시간 | 빌드 4.9 + 적재 **35.1** + 오라클 **156.7** + 케이스 ≈188 s (로그의 `loaded in`·`oracle … in` 줄). slot 팔은 0.1 + 30.2 + 165.6 + ≈189 | | |
| **step** `gate_deepseek41_step` (:812) | `--structure`: 종류별 런치 = 조각 보고 `:8-15`(`predict kind=` 8줄에서 계산, 실파일 1,169 노드) · 주입 상태 재생 = eager `:15-19` · shadow 표 `:19-23` · 층 0 창 행 NaN → fault → poisoned → reset `:24-29` | a (재생 입력은 세트) | step4·d1·d2 세트 |
| | `--sets`(step4, d1n): engram id·라우터 id(근접 동률 면제)·스트림 envelope·argmax `:30-40`. 동률 밴드는 `:2176-2192` | b | 세트 |
| | `--select`(d1, d2): 정확 top-k(a) + envelope·ik 목록 동률 밴드(b) `:41-60` | a+b | 세트 |
| | `--greedy` `:61-67` / `--ppl` `:68-72`(게이트 아님: run-ds41-*) | b/c | prompts.tsv, ik greedy, ikppl |
| 시간 | 41 = 빌드 2.6 + 적재 **30.2** (`:315` 줄) + 8.2 | | |
| **skew** (:819) | `--sets`: pair = 두 스텝 비트 동일, rollback + t2, cut 허가·거부 `:7-21` · `--structure`: 2배 캡처·순서·shadow `:22-32` · `--api`: step_pair = 스텝, 깊은 cut `:33-46` | a (입력은 step4·d1 세트) | `SETS` `:96`, `ORACLE_BUILD="db517b69"` `:98`, d1의 id(헤더) |
| 시간 | 116 = 빌드 1.5 + 적재 1회 ≈30[유도] + ≈84 | | |
| **long** (:827) | 모든 이음매 유한성 `:3-7,:36-38` · eager argmax = 엔진 `:40-41` | a | 없음 |
| | 주기 1·2로 8토큰 붕괴 `:38-40,:152-159` | c (텍스트 성질) | 무작위 가중치에서 거짓 빨강 위험은 낮다고 봅니다[유도]. 첫 fixture 실행이 판정합니다 |
| | `--free` 대 ik greedy(`greedy-ik-cpu-64-p7.tsv`)·EOS `:41-47,:520-571,:243` | b | 행 7 id, 파일의 EOS |
| | `--trigger`: 311 id → 층 34 점수 0 `:15-22,:173` | c | TRIGGER 리터럴 |
| **faults** (:837, solo) | 스텝 2부터 majflt 0, minflt ≤ 2 `:24-34,:161-168` | c | `PIN(2026-09-25)` 실파일 실측 |
| 시간 | long 106 = 적재 ≈30 + 위치 673개 × 탐침[유도] · faults 51 | | |
| **ds41-load** (:990) | (i) 세그먼트 = 계획·슬롯 맵 `:9-17` · (ii) KvLayout `:18-21` · (iv) 캡처, seed 거부 `:30-31` · (v) 재적재 반환 `:32-36` · (vi) 슬롯 0·n/2·n−1의 `_sel` = 평범 gemv `:37-41` | a | 계획은 파일에서 계산. n_l ≥ 3 필요 |
| | (iii) 위치 4·301·1025의 스텝 이미지 대 세트 입력 `:22-29,:88` | b | `STEP_SETS` |
| 시간 | 74 = 빌드 1.3 + 적재 **30.1** + **31.4** (`load 1/2` 줄) + 11 | | |
| **load-v41 / -lock** (:287, :293, solo) | ① 상주 = 설계 §5 계획 `gate_load_v41.rs:12-14` · ② 메모리 잔차 0(맞춘 반올림 규칙) `:15-25` · ④ 샤드 방출 뒤 mincore 전부 상주 `:35-43` · `--lock` VmLck 정확 `:45-50` | c | `CUT = 20` (`workstation.rs:68`), 실제 호스트 집합 190.6 GB |
| | ③ 독립 패킹 대비 되읽기 비트 동일 `:26-29` | a | 없음 |
| 시간 | 163 = A6000 **61.5** + 3090 **28.3** + populate **50.4** + ≈22. lock 165 (+ lock 4.4 s) | | |
| **draft** (:866) | plain = lookup 토큰(lcg, 코드 코퍼스 128 id)(파이프라인 `cmp`) · 코드 프롬프트 제안 ≥ 1 | a | 코퍼스 id(실어휘). 첫 제안은 프롬프트 자체의 n-gram에서 나옵니다(`draft.rs:4`) → fixture에서도 섭니다[유도] |
| 시간 | 170 = 적재 4회 + 64스텝 × 4 | | |
| **dspark-loop** (:875) | `--structure`: 탭 = 평범 + 탭 층·행당 커널 1개 `dsloop.rs:6-11` · `--tap`: 호스트 평균 = eager·graph·pair `:12-20` · plain = dspark 토큰 | a | 탭 = 드래프트 헤더, 실제 드래프트 8.5 GB |
| | 수락 ∉ {0, 전부} (justfile:875) | c (실텍스트의 드래프트 품질) | 무작위 fixture 드래프트는 수락 ≈ 0입니다 |
| 시간 | 168 = 적재 3회 + 드래프트 **13.2** (`load draft=… in 13.2 s`) + ≈65 | | |
| **chat** (:891, `bloomery_chat.rs:8-27`) | 프롬프트 id = 행 id | c (어휘) | 어휘를 그대로 복사하면 섭니다 |
| | greedy id가 generate의 접두 · text_consistent · p7 stop=length · seed 7 두 번 동일 | a | EOS가 16토큰 안에 나올 확률 ≈ 16/129,280[유도] |
| 시간 | 266 = 적재 6회 + 프로세스당 ≈12 s[유도] | | |
| **serve** (:901, `gate_ds41_serve.rs:9-38`) | /props = 계획 · /completion = generate · 스트림 = 비스트림 · reset · keep_point · 접두 재사용 · 두 턴 | a | 없음 |
| | /tokenize = 행 id `:23` | c (어휘) | 어휘 복사로 섭니다 |
| 시간 | 86 = 적재 2회 + ≈22 | | |

### 1-B op·조각 게이트 (엔진을 열지 않음 — 텐서 몇 개 + 덤프)

| 레시피 | 등급 | 끈 | 시간 |
|---|---|---|---|
| hc, rope, comp, index, engram(헬퍼 1, 줄 번호 있음) | b 중심, (a)는 커널 = 호스트 규칙·재생 = eager | 층은 파일·덤프에서 읽습니다. `comp.rs:557` STREAMS=["csa","hca"] 리터럴(확인함) → 두 스트림을 유지해야 합니다 | 5·3·3·3·6 s |
| ds41-lib | a (합성) | 파일 없음. ced 테스트는 40층 입력 `ced.rs:294` | 57 s (빌드) |
| woa `:13-18`, attn `:1-18`, moe `:6-18` (헤더) | b (+ a는 우리 규칙) | moe `:12` Q5_K down 층은 호스트 | 32·16·63 s |
| chain-glue `:7-40`, chain-attn `:15-35`, chain-ffn `:14-45` (헤더) | 조각 = op 경로(a) + 덤프 대조(b) | 노드 수는 종류에서 예측(glue `:38-40`) | 5·12·13 s |
| mcol `:1-18` (헤더) | a (m열 = 단일 열, 파일 형식으로 고름) | 없음 | 배치에 없음 |
| gemm V4.1 몫 `gate_gemm.rs:1940-1952` | a (f64 대조) | 첫 Q3_K `ffn_gate_exps` 층 | 182 s(Qwen3 포함) |

### 1-C DSpark 게이트 (헤더)

| 레시피 | 등급 | 비고 |
|---|---|---|
| dspark-read `gate_dspark_read.rs:4-16` | c (실제 드래프트의 타일링·E=255 스캔) + b (mxfp4_ref) | 실파일에 남깁니다 |
| dspark-hc, -experts (justfile:1051,1056) | a (호스트 규칙), ik는 진단 | 2·5 s |
| dspark-kv `:4-16`, -graph `:1-16` | b (dsref 세트) | graph는 fixup6에서 **빨강**이었습니다(로그 `:40` `ik-rule b0 dsv4_dflash_hc_init … false`). 범위 밖입니다 |

### 1-D 헤더·호스트·데이터 게이트 (전부 실파일에 남습니다. 싸고 (b)·(c)가 본체입니다)

| 레시피 | 등급 |
|---|---|
| ds41-meta (`ds41_meta.rs:6-38`: ik 적재 로그 `after-idxkey.log`·그래프 매니페스트 대조) | c+b |
| placement (justfile:525) | c |
| ds41-plan (`gate_deepseek41_plan.rs:1-14`) | b |
| ds41-oracle (:738) | b (기반) |
| ds41-host (`ds41_host.rs:1-14`: 샤드 걸침 층 7·14·27·34) | b + a (clamp) |
| union (`union.rs:1-14`: 라우팅 목록 입력) | a |
| qdot q5_K 몫, 1-1 V4.1 몫 | b |
| engram (NVMe 테이블) | c |
| tokenizer | c |
| ds41-kld | c |
| deepseek4-meta | V4-Flash 파일, 해당 없음 |
| ptx-spill · serve(mock) · vision | 모델 없음 / mmproj |

## 2. 컴파일 시점 상수 표

| 상수 | `path:line` | 강제 여부 (거부 지점) |
|---|---|---|
| `N_EXPERT`=384, `N_USED`=6 | `router.rs:60,63` | **강제**: `chain/ffn.rs:841`, `router.rs:717,780` |
| `HC_STREAMS`=4, `HC_MIX`=24 | `hc.rs:64,66` | **강제**: `body.rs:2339`, `chain/attn.rs:819`, `ffn.rs:849`, `glue.rs:611` |
| `ROW`=5120 (n_embd) | `engram_gate.rs:57` | **강제**: `glue.rs:611` |
| `LATENT`=512, compress `WIDTH`=512 | `attn.rs:70`, `compress.rs:62` | **강제**: `chain/attn.rs:816-817` |
| index key `WIDTH`=128 | `index_key.rs:40` | **강제**: `chain/attn.rs:818` |
| 인덱서 `HEADS`=32, `HEAD_DIM`=128 | `indexer.rs:65,67` | **강제**: `indexer.rs:1006` |
| `HC_PIECE`=512, `HC_MAX_TOKENS`=8 (= prefill `CHUNK`) | `hc.rs:69,81`, `body/prefill.rs:72` | 커널 기하. 파일 값 아님 |
| `HC_FIN_BATCH`=40 | `hc.rs:93` | 강제 아님: 5120 × 4 / 512의 조각 수에 맞춘 적재 묶음이고, `column_sum`은 임의 n을 받습니다(`hc.rs:370-397`). **층 수가 아닙니다** |
| ff (expert 폭) | `ffn.rs:849` | 256의 배수만 요구 → 더 작게 **허용** |
| 드래프트 `N_EXPERT`=128, `N_USED`=3, `MARKOV_RANK`=256 | `experts_mxfp4.rs:63,66`, `markov.rs:41` | **강제**: `draft/load.rs:276,285` |
| 런타임(Hparams): n_layer, n_head 64, q_lora 1280, o_groups 8(`chain/attn.rs:820-821` 나눗셈만), window 128, rope 64(`:809`), n_vocab, top_k, 비율, engram 치수 | — | 데이터 |

**더 작게 해도 되는 곳과 잃는 것.**
- **층 수** (허용): 1,169 노드, 19층 디코더 삼각형, 런치 큐 포화(Q ≈1,100), 계획 (b)의 20층 절단, 40층에 걸친 envelope 성장을 잃습니다.
- **ff** (허용): 실제 down K = 2304 walk(9 SB), gate·up의 N = 2304 격자, 호스트 union의 W·a·c, 16,773,120 B/expert 산수를 잃습니다.
- **어휘** (런타임): 헤드 129,280행 기하와 토크나이저·chat·serve·코퍼스를 잃습니다.
- **engram 행** (허용): NVMe 규모의 콜드 행을 잃습니다.

이 이유로 기본안은 실제 모양 그대로입니다.

## 3. fixture 설계

**층 사상 (최소 9층).** step 게이트가 로그에서 세는 종류 8개(`predict kind=` 8줄)에 층 하나씩 대응시키고, 여기에 f8을 더합니다.

| f | 원본 | 종류 (`predict kind`) | down | 카드 대상 |
|---|---|---|---|---|
| f0 | L0 | window+host-only | Q5_K | ✗ (`placement.rs:366-367` Q5_K 제외) |
| f1 | L1 | window+engram+host-only | Q5_K | ✗ |
| f2 | L2 | source-r2+indexer+card | Q4_K | ✓ |
| f3 | L3 | reader-r2+card | Q4_K | ✓ |
| f4 | L14 | source-r2+indexer+engram+card | Q4_K | ✓ |
| f5 | L20 | source-r1+indexer+card (마지막 압축기 = CED 경계) | Q4_K | ✓ |
| f6 | L21 | reader-r1+card | Q4_K | ✓ |
| f7 | L24 | reader-r1+indexer+card | Q4_K | ✓ |
| f8 | L25 | reader-r1+card, **topk_source(f7) ≠ kv_source(f5)** | Q4_K | ✓ |

- **빼도 되는 층은 없습니다.** f8은 노드 수로는 f6과 같지만, 두 source가 갈리는 유일한 자리라 남깁니다.
- **Q5_K 층이 둘인 이유:** 호스트 전용 종류 둘(window, window+engram)이 갈리고, 호스트 tier의 세 경로(`load host_tier q3_K k=5120 / q5_K k=2304 / q4_K k=2304`)가 모두 뜹니다.
- **CED:** 디코더는 d = 3(f6–f8)이고 B = 8 + 128·3 = **392** [유도]입니다. 392 mod 128 = 8이고 512와도 다릅니다. 그래서 ubatch 이음매·창 경계와 겹쳐 서로 가리는 일이 없습니다. 케이스 127–129는 B 아래, 511–4096은 B 위라 두 체제를 모두 덮습니다.
- **샤드 셋:** 한 층을 샤드 경계에 걸쳐 둡니다(실파일의 걸침 층 7·14·27·34와 같은 경로).

**텐서와 형식.** 원본 층의 텐서를 이름(blk.L → blk.f)·dims·타입 그대로 실헤더에서 복사합니다. 층당 주요 텐서와 바이트 [유도]:
- attention q_a·q_b·kv·wo_a·wo_b, shexp gate/up, hc_*_fn: Q3_K
- down·shexp_down: Q4_K (f0·f1은 Q5_K)
- ffn_gate_inp: BF16
- F32 norm류
- 소유층에는 compressor·indexer(+4.6 MB)
- routed 스택: Q3_K 1,946,419,200 × 2 + down 2,548,039,680 (Q5_K 3,114,270,720)
- 층 합: 6.52 GB, f0·f1은 7.08 GB
- 전역: token_embd Q3_K 284,416,000, output Q6_K 542,976,000

**바꿔 쓰는 메타데이터.**
- `block_count`=9
- `compress_ratios`=[0,0,2,2,2,1,1,1,1]+꼬리 [0,0,0]: 실배열이 43개이고 ik는 앞 `block_count`개만 읽습니다(`hparams.rs:1211-1233`)
- swiglu 층별 f32는 정확히 9개(`:1236-1246`)
- `engram.layer_ids`=[1,4]
- tokenizer.*는 원본 그대로
- `bloomery.fixture.*`(버전, 시드, 원본 층, 실헤더 sha, 카드 예산)

**어휘:** 실어휘 그대로입니다(827 MB, 1.4 %). chat·serve·코퍼스 게이트가 그대로 섭니다.

**engram:** 해시 상수가 파일 데이터(`hash.rs:1-40`)라서, 사이트당 2^14 근처 소수 24개와 offsets = 접두합으로 사이트당 ≈43 MB [유도]를 만듭니다. multipliers·token_map·pad는 복사합니다. NVMe 동작은 `gate-engram`(실파일)에 남깁니다.

**드래프트:** 실제 드래프트 모양(≈8.5 GB, justfile:874)에 target_layers=[6,7,8], 무작위 가중치로 만듭니다. 헤더만 읽는 곳(prefill·dsloop)도 `Split::open` 엄격 리더를 거치므로(`ds41_dspark.rs:64-69`) 온전한 파일이어야 합니다.

**배치:** plan_gate에 `BLOOMERY_CARD_BUDGET`≈6.9 GB(6,580M)를 줍니다 [유도].
- 바닥 = dense ≈1.40 + 반올림 ≈0.10 + KV 0.085 + ctx 0.537 + scratch 0.067 + margin 1.074 GB
- 여기에 217 × 16,773,120 B를 더하면 n_l ≈ **31** (실제 gate 배치의 30–31과 같음)
- 카드 ≈5.1 GB, 호스트 ≈55.5 GB
- plan 줄의 n_l이 검산입니다

**바이트:** 층 59.93 GB + 전역 0.83 + 테이블 ≈0.09 → **≈60.9 GB**, 드래프트를 더하면 ≈69.4 GB [유도]. `/models/<fixture 디렉터리>`에 둡니다. 원장이 stat 키로 잡습니다.

**적재 시간** (fixture 티어는 `BLOOMERY_CARD_DONTNEED=0`):
- **콜드 1회:** 카드 5.1 GB ÷ 0.80–0.83 GB/s + populate 55.5 GB ÷ ≈3.8–4 GB/s → ≈21 s [유도]
  - 근거: `gb2/g-gate-gpu-load-v41.log`의 `card …: loaded … in 61.5/28.3 s`, `host_populate=190634930176 in 50.4 s`
- **warm (게이트마다):** 2–8 s [유도]
  - 카드 업로드 0.35–6.4 s — **미교정 항**. 첫 실행의 `gate-ds41-load` `load 1/2: … in X s`가 잽니다.
  - populate ≤0.5 s: step의 30.2 s − 같은 카드의 콜드 28.3 s ≈ 2 s / 238 GB에서
- **오늘 적재가 ≈30 s인 이유:** `weights.rs:181-195`와 `hybrid.rs:184-198`이 업로드한 카드 페이지를 버립니다. 그래서 매 게이트가 23 GB를 NVMe에서 콜드로 읽습니다. `ds41-load`의 두 적재가 30.1·31.4 s로 같은 것이 그 증거입니다.

## 4. 가중치 스케일 유도

- **스케일 규칙:** 서브층 입력은 전부 RMS norm이라 스케일 불변입니다. 행렬 원소를 표준편차 1/√K로 두면 출력 rms ≈ 1입니다.
  - Q3_K: E[w²] = d²·341·5.5, 즉 d = 1/√(1876K): K = 5120 → 3.2e-4, 20480 → 1.6e-4.
  - Q4_K: m = sc, dmin = 7.5d로 두면 d ≈ 1.24e-4 (K = 2304).
  - Q5_K: sc ≤ 31로 제한하면 1.26e-4.
  - Q6_K: |sc| ≤ 8로 두면 1.5e-4.
  - 규칙: 블록 d는 모두 **정규 f16** [2^-13, 2^-10] 안에 두고, 스케일 필드 범위를 그에 맞춥니다.
  - norm gain = 1, sink = 0, HC base 0·scale 1, `exp_probs_b` = 0, 라우터 BF16 σ = 1/√5120 → 로짓 N(0,1) [유도].
- **유한성:** 잠재·압축 행·인덱스 키(f16)는 rms ≈1이라 65,504에서 한참 멉니다. 스트림은 √18 ≈ 4.2배로 자랍니다. 로짓 O(1)이고 softplus·Sinkhorn도 안전합니다. 그래서 fault 워드는 나지 않습니다[유도]. 첫 실행에서 `long --free`의 유한성 탐침이 판정합니다.
- **라우팅:** 균등해서 expert당 6/384입니다. 4,097토큰이면 expert마다 ≈64회 뽑힙니다.
  - P(층·토큰의 6개가 모두 호스트) = (353/384)^6 = 0.60 → 카드·호스트가 매 스텝 둘 다 돕니다.
  - union m_e ~ Binomial(512, 1/64): 평균 8, 최대 ≈17 [유도].
  - 선택: 층 2개에 bias ≈0.67로 hot expert를 심으면 실제 쏠림의 긴 열 런을 재현할 수 있습니다[유도].
- **동률 (결론 먼저):** iid 가우시안 라우터의 동률 확률은 **σ_r과 무관**합니다. 두 엔진의 입력 거리 r만 따릅니다.
  - 순서 동률 ≈ **111·r**, 집합 동률 ≈ **30·r** / 층·토큰 [유도]. 상위 6개 간격이 σ/(n·φ(z_k))이고 Σ1/간격 = 55.6입니다.
  - 실파일 step4 세트는 층 0–8에서 순서 동률 4개, 층 9에서 집합 동률이었습니다(step 로그의 `near tie: waived` 줄). r ≈ 4e-3에 해당합니다.
  - 따라서 스케일은 유한성에만, 동률은 분포 모양에만 걸립니다. (b) 게이트를 fixture로 옮겨도 실파일과 같은 깊이에서 멈춥니다. (a) 게이트에는 동률이 무관합니다.
- **파일을 쓰는 것:** 새 도구가 r8conv의 `gguf::write`를 씁니다. main에는 테스트 writer만 있습니다(`lib.rs:1027-1070`: U16/I32/Str, F32 4값). 도구가 writer에 요구하는 것:
  - 문자열 배열(토큰 129,280개, merges)과 u32/u64/f32 배열
  - 선언 순서 스트리밍과 알려진 형식의 nbytes 검사
  - MXFP4
  - 분할은 도구가 파일마다 writer로 처리합니다(분할 키, 첫 샤드에만 메타데이터, `lib.rs:420-432`)
  - 추가로 청크 단위 텐서 쓰기가 있으면 좋습니다(없어도 됨)
  - 무작위 블록은 양자화기 없이 코드·스케일을 직접 생성합니다

## 5. (b) 오라클 계획

- **ik가 적재하는가 — 우리 기록 기준, 조건부입니다.**
  - 우리 포트가 옮긴 ik 검사에는 층 수·층 번호 리터럴이 없습니다: engram 키(`llama-hparams.cpp:2075-2096`), 비율 앞 n개(`:2115-2121`), 걷기 검사 넷(`:2147-2176`), 세그먼트(`:2181-2188`), dsv4 플래그(`:2191-2192`), o_groups(`:2201-2205`), gating(`:2214,2219-2221`), 층별 배열(`llama-model-loader.cpp:845-882`), 분할(`:361-431`).
  - 박스 없이는 확정할 수 없습니다. 읽어야 할 것:
    - `llama-hparams.cpp` deepseek41 분기(≈2060–2230)
    - `llama-load-tensors.cpp` deepseek41 분기(`TENSOR_NOT_REQUIRED`, `:3326,3468`)
    - `build_deepseek4.cpp:1543-1545`
    - engram 행 수 대 primes 합의 단언
    - `/home/user/ik-dspark-draft`의 target_layers 처리
- **비용** [유도]: rig-log 09-23의 V4.1 CPU PPL(65,536토큰 × 40층, 19–20분) → 토큰·층당 0.43–0.46 ms → fixture 토큰당 ≈4 ms.
  - 세트 여섯의 프리필 ≈2,968토큰 ≈12 s (every-node 스케줄은 ×2–5)
  - greedy 두 행, KLD 기준 둘 ≈66 s
  - 합쳐 **5–10분**. 30분 승인선 아래입니다.
- **불가능하면** (b)는 실파일에 남고 fixture는 (a)만 싣습니다. dspark-*와 dsref는 ik 드래프트 트리 결과에 달렸습니다.

## 6. 티어 표와 예측 묶음 시간

| 레시피 | 착륙 | 파동 경계 | fixture에서 미루는 검사 | 실측 s → fixture s [유도], warm 업로드 저·고 |
|---|---|---|---|---|
| prefill ×2 | fixture | 실파일 | 없음 | 770 → 176 / 188 |
| step | fixture | 실파일 | --sets·--select (fixture 세트 전까지) | 41 → 7 / 13 |
| skew | fixture | 실파일 | --sets | 116 → 25 / 31 |
| long | fixture(--free) | 실파일 | --trigger(c), vs_ik(b) | 106 → 12 / 18 |
| ds41-load | fixture | 실파일 | (iii) | 74 → 8 / 20 |
| draft | fixture | 실파일 | 없음 | 170 → 22 / 46 |
| dspark-loop | fixture | 실파일 | 수락 ∉ {0, 전부} | 168 → 17 / 40 |
| chat | fixture | 실파일 | 없음 | 266 → 36 / 72 |
| serve | fixture | 실파일 | 없음 | 86 → 16 / 28 |
| faults, load-v41, -lock | 없음 | 실파일 (solo) | — | 379 → 0 |
| op·DSpark·헤더·호스트 게이트 | 실파일 | 실파일 | — | 그대로 |

- **fixture 모델:** 스텝 38.2 → **9.5 ms** [유도]. 실측 4,097스텝 156.7 s를 호스트 0.687 ms/층 + 카드 0.245 ms/층 + 헤드 0.9 ms로 나눈 값입니다. 층 비례 계산은 ×0.225입니다.
- **묶음 (fixup6의 44항목을 한 묶음으로)** [유도]:
  - A 고정 = 327 + 319–456 = 646–783 s
  - 균형 702 s (ptx-spill 분류 수정 포함)
  - X = 0
  - **벽시계 ≈ 674–783 s (11–13분)**. 오늘 같은 목록은 A 2,124 + X 379 ≈ **2,503 s (42분)**이고, 실측은 세 묶음 합 2,578 s였습니다.
  - 파동 경계는 오늘과 같은 ≈42분입니다.
- **러너 제안** (만들지 않았습니다):
  - `GROUPS`(`gate-batch.sh:331`)에 `fixture`와 `real`(티어 등급, 차선과 직교)을 더합니다.
  - `--tier fixture|real` 플래그를 두고, 기본값은 `real`이라 조용히 바뀌지 않게 합니다.
  - `fixture` 항목은 **맥 쪽** `BLOOMERY_V41_MODEL`로 돌립니다. `box.sh:63-66`이 프로필보다 먼저 싣기 때문입니다. `BOX_ENV`는 프로필 뒤(`:78-89,:129`)라 `BLOOMERY_REF_MODEL`이 따라오지 않습니다.
  - 드래프트 경로, `CARD_DONTNEED=0`, 파일 헤더의 예산을 함께 싣습니다.
  - `real` 항목은 `rc=defer tier=real`로 이름을 찍고, 시간 키에 `#fixture`를 붙입니다.
  - 원장은 바꿀 것이 없습니다: 키가 맥 쪽 변수·BOX_ENV·모델 디렉터리 stat을 품으므로 두 티어의 녹색은 자동으로 다른 키입니다.
  - 미룬 검사는 바이너리가 `bloomery.fixture.version` 헤더로 알아채고 `deferred(real): <검사>`로 찍습니다.

## 7. fixture로 교정할 수 있는 V4.1 프리필 항

- **교정 가능** (실제 모양의 단일 커널·호출, 층 종류별):
  - m = 8 프로젝션 넷 (`cardroute-design-report.md:83-86,99`)
  - 청크별 attention seg·merge·commit·append, 종류·깊이별 (`:87-88,100`)
  - 작은 커널 (`:89-91,101`)
  - source·indexer 추가분 (`:102`. top-k 히스토그램은 데이터 의존이라 부분적)
  - route norm_quant·꼬리·engram 호출 (`:103-105`)
  - 카드 shadow 그룹 GEMM의 expert당 비용 (`:149`)
  - PCIe D2H/H2D의 layer-batch당 비용 (`:150`)
  - 호스트 union t(m)의 W·a·c (`plan.md:73`)
  - 호스트 자체 발행의 런치당 비용 (`:144`)
  - 브리지 36.0 + 119.5·k_host (`plan.md:74`)
  - **route 잔여 10.9 ms의 종류별 분포** (`:108`): `BLOOMERY_STEP_STATS=1`의 층별 `card_out`에서 읽습니다. fixture는 층 하나가 종류 하나입니다.
- **교정 불가:**
  - 큐 막힘 ≈12 ms: fixture의 route 항목은 ≈1,633 × 9/40 ≈ 367로 Q ≈1,100 아래라 막히지 않습니다 (`:30-31,144`)
  - 프롬프트 전체 벽시계
  - 실제 호스트 집합 규모의 DRAM 경합, populate, NVMe
  - 실제 라우팅 쏠림 아래의 층별 union 합 (균등 m_e)
  - engram 콜드 행

## 8. 이 설계가 부르는 구현 라운드

| 순서 | 라운드 | 파일 | 크기 | 충돌 |
|---|---|---|---|---|
| 0 | (선행) r8conv 착륙 | `crates/gguf` write·open_private | — | bloomery-ee |
| 1 | `fxgen` | 새 `crates/model/src/fixture.rs`, 새 `bin/v41fixture.rs`, `lib.rs`의 `pub mod` 한 줄(마지막에 편집), 새 테스트 | M | r8conv(lib.rs 한 줄), uniondispatch는 무관 |
| 2 ‖ | `fxseam` | `gguf/src/v41.rs`(FIXTURE, `_fx`), `models/deepseek41.sh`, `box.sh`(DSPARK 맥 쪽 운반, REF/V41 분리 거부), `gpu-gates/src/lib.rs`(REF == V41 단언, fixture 판별) | S | 없음 |
| 2 ‖ | `fxgates` | prefill `:153`·`:1352`, long·step·skew·load의 미룸 줄, justfile:875 수락 줄, `ced.rs` 9층 테스트 | S | ds41proj(prefill.rs·attn.rs·hc.rs·dense.rs·gpu lib.rs)와 겹치지 않음 |
| 3 | `fxtier` | `gate-batch.sh`(티어 + ptx-spill 분류), justfile 속성 12개(마지막에 편집), check-recipes | S | justfile 공유 |
| 4 | (리드·박스) 생성 ≈70 GB, 1분 미만 [유도] + 첫 `--tier fixture` 묶음 | — | — | **`/models` 디스크는 사용자 승인** (r8 sidecar와 같은 절차) |
| 5 | `fxoracle` | ik 적재기 독해 → fixture 세트·greedy·KLD → (b) 미룸 해제 | M, 박스 5–10분 | CPU 임대 |

## 9. 범위 밖 개선 후보 (보고만, 손대지 않음)

1. `tools/gate-batch.sh:26-27` + `tools/ptx-scan.sh:44`: ptx-spill의 JIT가 3090 락을 잡습니다. 스크립트 안 호출도 분류하거나 `any`로 돌리게 하세요 (S).
2. `crates/gguf/src/v41.rs:58-64`: 공개 파일이 아닌 모든 파일이 `""`를 받아 셋째 파일이 혼합 파일 세트명과 충돌합니다 (XS, 확인함).
3. `tools/box.sh:78-89,129`: BOX_ENV의 `BLOOMERY_V41_MODEL`만 바뀌고 `REF_MODEL`은 그대로라 두 소유자가 갈립니다 (XS, 확인함).
4. `crates/gpu-gates/src/lib.rs:118`, `oracle/mod.rs:96`: V4.1 게이트에서 `ref_model_path == v41::model`을 단언하세요 (XS, 헬퍼).
5. `crates/model/src/placement/workstation.rs:68`: `CUT = 20` 리터럴 (S).
6. `gate_deepseek41_prefill.rs:153` SPLITS, `:1352` 층 20 리터럴 (XS).
7. `crates/gpu/src/weights.rs:181-195`: 실파일 게이트는 적재마다 카드를 콜드로 읽습니다(묶음당 ≈25 × 23 GB [유도]). 문서에 적어 두세요 (XS).
8. `workstation.rs:131`: plan_gate가 3090을 이름으로 지정해 V4.1 게이트가 A 차선에 고정됩니다. fixture 티어를 양 카드로 돌리면 벽시계가 더 줄어듭니다 (S–M).
9. `gate-gpu-dspark-graph` 빨강(fixup6, 로그 `:40`) — 리드가 인지하고 있는 사안.
10. `justfile:762`, `gate_deepseek41_rope.rs:29-31`: rope 표 테스트가 `crates/gpu/src/rope_table.rs:282`로 옮겨졌다는 헬퍼 판독 (XS).
11. `gate_deepseek41_index.rs:822-835`: 합성 (a) 검사가 세트 다섯이 다 열린 뒤에야 돕니다 (S, 헬퍼).
12. `gate_deepseek41_engram.rs:449`: 캐시된 테이블에서는 콜드 경로가 돌지 않습니다 (S, 헬퍼).
13. `gate_deepseek41_comp.rs:562-570`: source 층을 덤프에서만 읽습니다. 파일과 대조하세요 (XS, 헬퍼).
14. `justfile:102-103`, `:778`: 주석의 "1·14" 층 리터럴 (XS, 헬퍼).

## 10. 모델

- opus로 돌았습니다.
- 게이트 독해 팬아웃(비전 아님, 3만 5천 줄)을 위해 opus 헬퍼 6개를 띄웠습니다. ops1(hc·rope·comp·index·engram·lib) 하나만 완주했습니다.
- 나머지 다섯은 07:1x KST에 세션 한도(429, 10:20 리셋)로 초반에 죽었습니다. 그 몫은 제가 머리말·상수·로그로 직접 읽었습니다(1번 표의 "헤더" 표시).
- 헬퍼 1의 결론 중 F1(`v41.rs:58-64`), F3(`box.sh:47-51`), comp STREAMS 리터럴은 원본에서 확인했고, 나머지 줄 번호는 헬퍼 보고 그대로입니다.
