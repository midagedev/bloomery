# Round markov 보고: DSpark Markov 헤드 단독 수락률 (E6)

**결론:** Markov 헤드 단독의 acc/positions는 다섯 코퍼스 모두 0.059–0.077입니다. 손익분기 0.34와 0.36을 모두 크게 밑돌아, 헤드 단독 초안 후보는 닫힙니다. 모든 표의 목표는 코퍼스 텍스트 자체라서, 엔진 수락률이 아니라 대리 척도입니다.

## 1. 변경 파일

- `/Users/hckim/repo/bloomery-markov/crates/model/src/bin/markov-accept.rs`: 새 bin입니다. bf16을 f32로 한 번 넓혀 헤드를 로드하고, AVX2 FMA 커널로 위치별 경로와 이전 토큰별 argmax 표를 계산합니다. E5 markov1 재계산, `--tokens`·`--table-out`·`--self-test`가 들어 있습니다. 이 crate는 bin 목록을 두지 않고 자동으로 찾으므로 `Cargo.toml`은 고치지 않았습니다.
- `/Users/hckim/repo/bloomery-markov/justfile`: 맨 끝에 `markov-accept` 레시피를 추가했습니다. self-test를 돌린 뒤 다섯 코퍼스를 실행하고, 둘 다 `tools/host-gate.sh`를 거칩니다.

## 2. 증명

아래는 실제 출력 줄입니다.

```
lease=0                       (첫 실행 전, check 전, 레시피 실행 전 세 번 모두)
check rc=0
lint: grep -c '^warning:' = 169   (첫 lint 175 = 기준 169 + 내 bin 6줄 → as_chunks로 고친 뒤 169, 내 파일 0건)
check-recipes: ok / check-comments: ok / check-arch: ok / fmt-check rc=0
```

**self-test**는 `BLOOMERY_THREADS=1`과 32스레드 둘 다 통과했습니다.

```
markov-accept: self-test kernel = scalar sum order, bit for bit (stride 16, 13 rows) — ok
markov-accept: self-test kernel = scalar sum order, bit for bit (stride 40, 5 rows) — ok
markov-accept: self-test kernel = scalar sum order, bit for bit (stride 256, 3 rows) — ok
markov-accept: self-test synthetic table = known answer 9 f(p mod 16), 144 entries, ties to the lowest — ok
markov-accept: self-test fnv1a64 of "" and "abcd" — ok
markov-accept: self-test per-position argmax = table, rank = brute-force count, 299 positions — ok
markov-accept: self-test markov1 = hand-counted proposals — ok
markov-accept: self-test ok (pool 32 threads)
```

**FAIL-first:** 처음 만든 합성 헤드는 어휘가 17이었습니다. 이 크기에서는 범위 안 동률 규칙을 `>=`로 바꾼 변이가 32스레드에서 살아남았습니다. 청크당 행이 하나 이하라 청크 안 동률이 생기지 않았기 때문입니다. 그래서 합성 헤드를 어휘 144로 바꾸고, 연속된 9행이 같은 토큰을 나타내게 했습니다. 이제 스레드 수 36 이하에서는 청크 안 동률과 청크 사이 동률이 모두 생깁니다.

```
mutation A (in-range x >= l.best):  self-test failed: synthetic table = known answer ... → t32 mutated rc=1
mutation B (merge later.best >= self.best): self-test failed: synthetic table = known answer ... → t32 mutated rc=1
restored   (cmp로 원본과 같음을 확인)
```

**실제 실행**은 `just markov-accept --table-out /tmp/markov-argmax.u32`이고, CPU만 쓰며 시작할 때 lease가 비어 있었습니다.

```
markov_w1/w2 rank 256 x vocab 129280, widened to f32 in 0.10 s; pool 32 threads
argmax table: 129280 entries, 517120 bytes as u32, fnv1a64 8ceb99396ccddf97, 14117 distinct drafts, 6.67 s
markov-accept: table = per-position argmax at every position — PASS
md5 967d4b43de1dcad22f7308fa63dbb645  /tmp/markov-argmax.u32
total 23.682 s   (레시피 전체)
```

해시는 lint 수정 전후 두 빌드에서 같았습니다.

**표 E6.** 각 스트림의 앞 50,000토큰이고, 목표는 코퍼스 텍스트 자체입니다. 헤드는 항상 제안하므로 proposed와 positions가 같습니다.

| corpus | positions | acc/positions | top-2 | top-4 | markov1 (E5 보고서에서 복사) | wall s |
|---|---:|---:|---:|---:|---:|---:|
| code | 49999 | 0.077 | 0.109 | 0.144 | 0.405 | 2.95 |
| prose | 49999 | 0.063 | 0.100 | 0.135 | 0.270 | 3.11 |
| prose-all | 49999 | 0.063 | 0.099 | 0.135 | 0.271 | 2.68 |
| korean | 49999 | 0.059 | 0.086 | 0.122 | 0.238 | 3.67 |
| threads | 49999 | 0.076 | 0.109 | 0.141 | 0.326 | 3.19 |

bin이 E5 규칙으로 markov1을 다시 계산했습니다. 다섯 코퍼스 모두 소수 셋째 자리까지 E5와 같고, proposed 값 47842, 44141, 44144, 46854, 45432도 같습니다. 위치별 경로와 표의 불일치는 다섯 코퍼스 모두 0입니다.

**10,000위치 구간별 acc/positions** (헤드 / markov1)

| corpus | seg 0 | seg 1 | seg 2 | seg 3 | seg 4 |
|---|---:|---:|---:|---:|---:|
| code | 0.107 / 0.438 | 0.075 / 0.408 | 0.076 / 0.376 | 0.061 / 0.382 | 0.068 / 0.419 |
| prose | 0.052 / 0.237 | 0.059 / 0.271 | 0.076 / 0.225 | 0.060 / 0.234 | 0.067 / 0.380 |
| korean | 0.061 / 0.232 | 0.064 / 0.228 | 0.065 / 0.243 | 0.057 / 0.237 | 0.047 / 0.250 |
| threads | 0.084 / 0.261 | 0.085 / 0.299 | 0.051 / 0.374 | 0.077 / 0.395 | 0.083 / 0.300 |

## 3. 예측과 결과

실제 실행 전에 두 가지를 적어 두었습니다. 첫째, 헤드가 앞쪽 구간에서는 markov1을 이기고 반복이 많은 code의 뒤쪽에서는 지며, 전체는 0.2–0.4라는 예측입니다. 둘째, 벽시계 추정 15–30 s입니다.

- **수치는 틀렸습니다.** 실제는 0.059–0.077이고, seg 0을 포함한 모든 구간에서 markov1에 4–6배 뒤집니다.
- **틀린 항은 "헤드는 체크포인트의 bigram"이라는 전제입니다.** 헤드는 draft 본체의 logits에 더해지는 rank-256 보정항이고, 본체와 함께 학습됐습니다. 단독 argmax는 bigram 최대우도 예측이 아닙니다.
- **버그가 아니라는 근거는 스크래치 프로브입니다.** 박스의 `/tmp/markov-probe.py`로 토큰 문자열을 확인했습니다.
  - 결정적인 이어쓰기는 맞힙니다. `Ġgg→ml`은 727/727, `ĠLL→M`은 541/552이고, `ĠUnited→ĠStates`와 `Ġof→Ġthe`도 맞습니다.
  - 스트림 대부분을 차지하는 엔트로피 높은 앞 토큰에서는 엉뚱한 토큰을 냅니다. `,→Ġso`는 1/2028, `.→</think>`는 0/754입니다.
  - 표 전체에서 서로 다른 초안은 14117개로 어휘의 11 %뿐입니다.
- **벽시계 추정은 맞았습니다.** 코퍼스당 2.7–3.7 s, 표 6.7 s, 전체 23.7 s였습니다.

## 판정

| corpus | 헤드 단독 acc/positions | 0.34 | 0.36 |
|---|---:|---|---|
| code | 0.077 | 불통과 | 불통과 |
| prose | 0.063 | 불통과 | 불통과 |
| prose-all | 0.063 | 불통과 | 불통과 |
| korean | 0.059 | 불통과 | 불통과 |
| threads | 0.076 | 불통과 | 불통과 |

- **top-4도 0.12–0.14입니다.** k-way 트리 초안으로도 건지지 못합니다.
- **표 조회 대 엔진 내장:** 앞선 토큰만으로 정해지므로 형태로는 517 KB u32 조회표(D≈0)가 0.15 ms 커널보다 낫습니다. 하지만 이 수락률에서는 두 형태 모두 이득이 없어서, 헤드 단독 후보는 닫힙니다.
- **이 수치는 전체 draft 안에서 헤드가 기여하는 몫의 하한이 아닙니다.** 그 몫은 본체 logits와 함께 재야 합니다. 이 결과는 "헤드만 떼어 쓰는 것"만 기각합니다.

## 4. 하지 못한 것

- 없습니다.
- ik의 헤드 경로에서 BF16 mul_mat은 활성값을 bf16으로 반올림합니다. 이 bin은 f32로 계산하므로 근접 동률에서 argmax가 다를 수 있지만, 그 차이는 재지 않았습니다. 0.34와의 차이에 비하면 판정을 바꿀 크기는 아닐 것으로 봅니다.
- 박스 `/tmp`의 `markov-argmax.u32`와 `markov-probe.py`는 남겨 두었습니다.

## 5. 스펙 밖 개선 지점 (보고만)

- `crates/gpu-gates/src/bin/gate_dspark_read.rs:35`: draft 절대경로 리터럴이 새 `markov-accept` 레시피에 한 번 더 들어갔습니다. `BLOOMERY_DSPARK_MODEL`을 `box.sh` 프로필에서 export하면 경로의 소유자가 하나가 됩니다. 한 줄 크기입니다.
- `docs/research/spec-decoding-report.md:57`: "Markov 헤드 단독"의 수용률이 아직 "모른다"로 적혀 있습니다. 이번 결과로 갱신할 자리입니다. 한 줄 크기입니다.
- **손익분기 값이 문서마다 다릅니다.** 0.34와 0.36이 섞여 있다는 것은 E5가 이미 보고했습니다. 이번에도 판정을 두 값으로 써야 했습니다.

## 6. 모델

opus(Opus 5.5)로 실행했습니다.
