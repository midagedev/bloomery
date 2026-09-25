# h3tile-b-design 보고 — r8 바이트의 자리와 그 바이트를 읽는 곳 전부 (설계, 2026-09-26 새벽)

> 리드 메모(2026-09-26). 아래는 `h3tile-b-design` 라운드(opus, 파일 변경 없음, 박스는 평문 읽기만)의 보고 원문이다.
> 라운드 도중 리드가 한 번 멈추고 조건 둘(카드 쪽 독자 셋, 형식 태그)을 더해 재개했다.
> - 권고는 **② r8 사이드카**다. 원본 공개 파일은 그대로 두고, 라우팅 gate/up만 비공개 타입 id로 담은 GGUF를 하나 더
>   둔다(+155.7 GB, /models 여유 1.1 T). 카드 업로드와 빌림 재업로드는 원본을 그대로 읽어 코드가 0줄이고, r8을 읽는
>   곳은 호스트 계층과 hoststream 링 채움(장치 un-permute)뿐이다. 형식 태그는 `strict_tensors`가 모르는 타입 id를
>   이름 붙여 거부하는 기존 코드에서 나온다.
> - 리드 처분은 셋이다.
>   1. **R1 uniondispatch**(층·배치당 디스패치 161 → 5, 비트 동일, +3…+8 %[유도])는 r8과 무관하게 연다. hostserve
>      착륙 뒤다(같은 `moe.rs`·`ops.rs`).
>   2. **R2–R4(사이드카 프로그램)**는 사용자 결정으로 올린다. 디스크 +155.7 GB가 들고, V4.1 교차 엔진 시팅에서 참조
>      엔진과 우리가 페이지를 못 나눠 엔진을 바꿀 때마다 양쪽이 차가워진다(§10.4). hoststream 뒤 값은 pp512 +11 %,
>      pp4096 +3.6 %[유도]다.
>   3. §10.1(빌림 재업로드 원천이 DONTNEED 뒤라 NVMe ≈ 1.8 s, recal의 90 ms는 pinned 전제)은 r8과 무관한
>      hoststream 결함이라 69에게 바로 보냈다. §5.3의 꼬리 호출 스크래치(`UnionScratch::new(embd, ff, 4096)` ≈
>      2.25 GB[유도])는 hostserve 리뷰에서 본다.

---

**h3tile-b-design 보고 (bloomery-ee 앞)**

설계 라운드라서 어떤 파일도 고치지 않았습니다. 빌드·게이트·타이밍도 돌리지 않았고, 박스는 `ssh ws` 평문 읽기만 했습니다. 이 라운드가 도는 사이에 main이 `b822f1a`(q3fix2 착륙)로 움직였기 때문에, 아래 `path:line`은 전부 이 트리 기준입니다.

## 0. 결론

- **② r8 사이드카를 권합니다.** 원본 공개 파일은 그대로 두고, 라우팅 층 40개 × 전문가 384개의 gate/up만 r8로 담은 GGUF를 하나 더 둡니다. 디스크는 +155.7 GB이고 /models 여유 1.1 T 안에 들어갑니다.
- **카드 쪽 읽는 곳 셋 중 둘은 코드를 한 줄도 안 바꿉니다.** 적재 업로드와 대여 슬롯 재업로드는 원본 Q3_K를 그대로 읽습니다. r8을 읽는 것은 호스트 계층과 hoststream 링 채움뿐입니다.
  - 링에는 장치 un-permute를 둡니다. 17.6 KB 그룹마다 제자리에서 풀고, 전문가당 31 µs입니다[유도]. 흐름의 그늘 안에 듭니다.
- **형식 태그는 이미 있는 코드에서 거부가 나옵니다.** 디스크의 타입 id를 비공개 값(1011 제안)으로 두면 `strict_tensors`가 코드 변경 없이 이름 붙여 거부합니다(`crates/gguf/src/lib.rs:788-792`). 사이드카를 여는 길은 호스트 계층의 새 입구 하나뿐입니다.
- **RAM과 따뜻한 재시작은 오늘과 같습니다.** 214.0 GB, populate ≈ 1.65 s입니다.
  - ③(적재 때 리팩)은 시작마다 ≥ 95 s를 더 냅니다[유도].
  - ①(파생 파일 전체)은 +347.3 GB가 들고, 카드 로더 둘에 un-permute를 넣어야 합니다.
- **값은 hoststream이 착륙하면 크게 줄어듭니다.**

  | 조건 | pp512 | pp4096 |
  |---|---|---|
  | 오늘 흐름 | +23 % | +22 % |
  | hoststream G8 R128 | +11 % | +3.6 % |
  | 거기에 카드 route 절반(파트 C) | +14 % | +6.7 % |

  모두 [유도], 중앙값입니다. 이 값을 사이드카 155.7 GB와 라운드 넷에 견줘 저울질하는 것은 리드 몫입니다.
- **첫 라운드는 커널과 무관한 디스패치 재구성을 권합니다.** 층·배치당 디스패치를 161 → 5로 줄입니다. 비트는 같고, 그것만으로 +3…+8 %[유도]입니다. per-call/per-expert 질문도 이 라운드가 닫습니다.
- **사이드카가 들어오면 교차 엔진 시팅 프로토콜이 깨집니다(§10).** 리드가 알아야 합니다.

## 1. 바꾼 파일

저장소 파일은 없습니다. 계산 스크립트는 스크래치에만 있습니다.
- 스크립트: `/private/tmp/claude-501/-Users-hckim-repo-bloomery/b4e4a3ef-e524-4e47-91af-3afadc891c1e/scratchpad/h3b/{pred.py,pred_add.py,timeline.py,dma.py,partc.py}`
- 출력: 같은 디렉터리의 `*.out`
- hoststream-recal 모형(`…/02d044bc…/scratchpad/hs/recal-model.py` 등)은 읽기 전용으로 import했습니다(`dont_write_bytecode`).

## 2. 종이 위 분해 — 항과 이미 잰 값

| 항 | 값 | 출처 |
|---|---|---|
| Q3_K 행 | 20 sb × 110 B = 2,200 B. gate 또는 up 한 전문가 = 2,304 × 2,200 = 5,068,800 B. 8행 그룹 = 17,600 B = 20 × 880 B, **두 레이아웃에서 같은 구간** | [유도] |
| 라우팅 gate/up 전체 | 40 × 384 × 2 × 5,068,800 = 155.7 GB | [유도] |
| 호스트 집합 (a) | 214.0 GB, 12,692 전문가(AGENTS.md). gate/up 12,692 × 10,137,600 = 128.7 GB, down 85.3 GB | [유도] |
| 카드 전문가 | 2,668개. gate/up 27.0 GB, 전체 ≈ 44.75 GB. 밀집 카드 바이트 7.66 GB(`docs/research/v41-audit-load-report.md:18`) | [유도] |
| 공개 파일 | 9 샤드 합 347,271,472,288 B. /models 여유 1.1 T(`df -h`) | 박스 읽기 |
| 호스트 union (S14, P512 lcg) | 72.3 ms/층-배치. 비-union 32.3, 카드 route 30.4 | rig-log 09-25#v41-prefill-s14 |
| h3tile-a 적합 | union a 30.2 · c 22.20, unionq 51.6 · 23.78, unionr8 59.4 · 14.63 µs | rig-log 09-25#h3tile-ab |
| NVMe populate | 1.33 GB/s. 쓰기 5.74–5.85 GB/s | rig-log 09-23 |
| 적재 | 따뜻할 때 92 s, 일부 밀려났을 때 132 s. 호스트 사본 ≈ 10 s[유도] | rig-log 09-25:123, `v41-audit-load-report.md:47` |

박스 실행이 필요한 항은 하나입니다. 큐 고정비가 호출당인지, 디스패치당인지, 전문가당인지입니다. 기대값과 함께 §5.2에 적었습니다. 나머지는 모두 산수로 닫힙니다.

## 3. Part A — 선택지 표

| 항목 | ① r8 전용 파생 파일 | **② r8 사이드카** | ③ 적재 때 익명 메모리로 리팩 |
|---|---|---|---|
| 디스크 | +347.3 GB. 원본은 게이트·오라클·참조 엔진 때문에 남습니다. 여유 → ≈ 0.75 T | +155.7 GB → ≈ 0.94 T | 0 |
| 정상 상태 RAM | 페이지 캐시 214.0 GB | 페이지 캐시 214.0 GB(사이드카 128.7 + 원본 down 85.3) | 익명 128.7 + 캐시 85.3 GB |
| 적재 정점 | 오늘과 같습니다 | 오늘과 같습니다 | + 스택 하나 원본 ≤ 1.95 GB(384 × 5,068,800)[유도] |
| 따뜻한 적재 | populate ≈ 1.65 s | ≈ 1.65 s + 신원 검사 < 0.1 s | 따뜻한 적재가 없습니다. 원본 gate/up 페이지를 버렸으므로 ≥ 96.7 s(128.7 ÷ 1.33). 벤치 셋업의 끝-끝 속도 0.375 GB/s면 343 s[유도] |
| 차가운 적재 | 266.4 GB ÷ 1.33 = 200 s | 200 s(같은 바이트, 파일 둘) | 200 s + 패커 |
| 카드 업로드 | r8을 읽습니다 → un-permute 27.0 GB | **원본 Q3_K, 코드 그대로** | 원본 Q3_K |
| 재업로드 128 슬롯 | r8 → 1.30 GB un-permute | 원본 Q3_K(un-permute 0) | 원본 Q3_K |
| 링 채움 | r8 → 장치 un-permute | 사이드카 r8 → 장치 un-permute | 익명 r8 → 장치 un-permute, 또는 등록 메모리에서 직접 DMA |
| 디코드 | r8, C = 1 | 사이드카 r8, C = 1(필수, §5.1) | 익명 r8 |
| 참조 엔진과 공유하는 페이지 | 0 | down 몫만 | down 몫만 |
| 낡거나 어긋난 파일 | 자기완결입니다. 게이트·참조가 열면 타입 id에서 이름 붙은 거부 | 적재 때 원본 신원을 대조해 이름 붙은 거부 | 없습니다 |
| 바꿔야 하는 읽는 곳 | 카드 셋 + 호스트 | 호스트 + 링 | 호스트 + 링 + 적재 리팩 |

**un-permute 가격.** 호스트 AVX2는 880 B당 약 1,100–1,250 연산입니다[유도, 손 계수]. 연산 수는 이렇게 나옵니다.
- 코드 ≈ 1,016: 쌍 8개 × (로드 24 + 필드 추출 ~20 + 8×8 전치 24), (행, 쌍) 64개 × 7, 저장 24.
- 스케일 40–80, d 10.

IPC 2.52(rig-log 09-25#h1fold-ipc) × 3.96 GHz를 넣으면 코어당 7–9 GB/s입니다. 여러 코어를 쓰면 읽기 + 쓰기 DRAM에 묶입니다. 장치에서는 CTA 하나가 17,600 B 그룹 하나를 제자리에서 풉니다. 전문가 gate/up 하나가 20.3 MB 왕복 ÷ ≈ 650 GB/s = **31 µs**입니다[유도].

| 읽는 곳 | 바이트 | 호스트 | 장치 | 그 읽는 곳의 창 | 판정 |
|---|---|---|---|---|---|
| 적재 업로드(①만) | 27.0 GB | 계산 0.11 s, DRAM 바운드 0.39 s | 83 ms(2,668 × 31 µs) | 적재 ≈ 92 s | ≤ 0.4 %. ①은 이 비용이 아니라 읽는 곳 수와 공유에서 집니다 |
| 재업로드(①만) | 1.30 GB | 18.5 ms | 4.0 ms | H2D 89 ms(2.34 GB ÷ 26.3, pinned 가정) | 그늘 안 |
| 링 채움(셋 다) | 슬롯당 10.1 MB | 채움 스레드가 ≈ 2코어를 씀 = union −6 %(15.8 GB/s ÷ ~8 GB/s) | 31 µs = 슬롯 H2D 641 µs의 4.9 %, 다른 엔진이라 겹침. P4096 G8에서 층당 4.2–4.9 ms | union이 묶는 자원입니다 | **장치** |

- **장치 쪽은 벌크로 발사합니다.** 착륙한 슬롯 묶음(링 한 바퀴, ≤ 128 슬롯)마다 한 번 발사합니다. 슬롯마다 발사하면 134 × 40 = 5,360회이고, c_node 3–5 µs로 16–27 ms입니다[유도].
- **r8을 직접 읽는 카드 GEMM**은 `gemm_block!` Q3_K `@load`에 비트 패킹 해독을 새로 넣어야 합니다. 커널 계열 하나, ptx 핀, 비트 게이트 하나가 새로 듭니다. 아끼는 것은 층당 ≤ 4.9 ms뿐이고, 이 시간은 P512에서 흐름의 여유 안에 있습니다(host_done 74.8 대 stream_done 50.3 ms, `timeline.out`). 그래서 더 싸지 않습니다.
- **재업로드 원천은 세 선택지 모두의 문제입니다.** `BLOOMERY_CARD_DONTNEED`가 적재 때 카드 세그먼트 페이지를 버리므로, 재업로드는 NVMe에서 2.34 ÷ 1.33 = 1.8 s가 걸립니다[유도]. 89 ms 예산은 pinned 원천을 전제합니다(§10-1).
- **③의 유일한 장점**은 등록한 익명 메모리에서 직접 DMA하는 것입니다. gate/up이 DRAM을 1회만 건너고, fill_steal이 0.04 → 0.016으로 줄어 hoststream pp512가 177.2 → 185.6(+4.7 %), pp4096이 468.9 → 472.0(+0.7 %)입니다[유도, `dma.out`]. 시팅마다 시작당 ≈ 95 s를 낼 값은 아닙니다. 서빙 모드에서 다시 볼 만합니다.
- exllamav3식 memfd·tmpfs 변형은 ②와 같은 128.7 GB를 쓰는데 회수가 안 되고, 붙잡는 프로세스가 필요하며, 박스의 shmem THP가 never입니다. 열세입니다.

**적재 자원 시간선(②).** 사이드카는 바이트가 같으므로 어느 항도 바꾸지 않습니다.

| 단계 | 호스트 CPU | PCIe | NVMe(차가울 때) | 벽시계 따뜻 / 차가움 |
|---|---|---|---|---|
| 파일 10개 열기 + 신원 검사(헤더 digest, 표본 640 KiB) | < 0.1 s | 0 | 160 × 4 KiB × 130 µs = 21 ms | < 0.1 s / < 0.1 s |
| 카드 업로드 52.4 GB(원본) | 사본 ≈ 10 s | 2.0 s pinned / 2.5 s pageable | 39 s | ≈ 10 s / ≈ 39 s |
| populate 214.0 GB(사이드카 128.7 + 원본 85.3) | 페이지 테이블 | 0 | 161 s | 1.65 s(실측) / 161 s |

단계는 차례로 돌아 합이 벽시계입니다. populate를 업로드 사본 밑에 겹치면 따뜻할 때 ≤ 1.65 s, 차가울 때 0을 법니다. 차가울 때는 어느 쪽이든 NVMe 266 GB = 200 s에 묶입니다[유도]. 92 s의 나머지(CUDA 초기화, 그래프 등)는 이번 독해로 나누지 못했습니다.

**프롬프트 층 자원 시간선(② + r8, 오늘 흐름 P512 lcg, 층-배치당).**

| 자원 | 오늘(S14) | r8 뒤 |
|---|---|---|
| 카드 SM(route) | 30.4 ms | 30.4 |
| 호스트 CPU(union) | 72.3 ms | ≈ 52.5(밴드 50.9–59.6)[유도: 72.3 × 모형비] |
| 호스트 DRAM | 5.35 GB(317.3 × 16.86 MB) → 74 GB/s | 102 GB/s. 바닥 40.1 ms |
| PCIe | ≈ 21 MB → ≈ 1 ms | 같음 |
| NVMe | 0 | 0 |
| 벽시계 | 합 72.3 + 32.3 = 104.6 → pp512 122(실측 121.4) | 52.5 + 32.3 = 84.8 → ≈ 151[유도] |

- route를 기다려 union이 돌고, union을 기다려 다음 층이 돕니다. 그래서 **합**입니다.
- r8 다음 레버는 비동기입니다. route(b+1)를 union(b) 밑에 겹치면(hoststream G, 또는 P512의 부분 배치 스큐) max(30.4, 52.5)가 됩니다.
- r8 뒤에는 호스트 전문가의 36–51 %가 W 바닥에 앉습니다(Bin(512, 7.61/512), m* = 6.8–7.4)[유도]. 그다음 호스트 레버는 명령이 아니라 **바이트**입니다. 하나는 k이고, 다른 하나는 down 활성값을 q8_K로 바꾸는 것입니다(트리아지 109행, 사용자 결정).

## 4. 형식 태그 — 설계와 routed gate/up을 읽는 곳 전수

**설계**
- **디스크**
  - 타입 id **1011**(규칙: bloomery 비공개 = 1000 + 원 ggml id). 메인라인 0..42(`ggml.h:433` `GGML_TYPE_COUNT   = 43,`)와 ik 0..399(`ggml.h:491-492` `GGML_TYPE_Q8_K_R8   = 399,` 다음 COUNT) 밖이고, ik의 다음 id와도 멉니다.
  - 두 번째 방벽으로 `general.architecture = "bloomery-r8"`을 둡니다. **211은 절대 쓰면 안 됩니다.** ik가 자기 `Q3_K_R4`(`ggml.h:465`)로 조용히 읽습니다.
  - 텐서 이름·dims·nbytes는 원본과 같습니다.
  - KV에는 `bloomery.r8.layout`(qdot 레이아웃 버전)과 원본 신원을 둡니다. 신원은 샤드별 이름·크기·헤더 바이트 sha256, 텐서별 앞·뒤 4 KiB sha256, 텐서별 전체 sha256(오프라인 verify용)입니다.
- **메모리**
  - `GgmlType`에 변형을 **추가하지 않습니다**. `Gguf::open_private(path, &[(1011, 256, 110)])`만 사이드카를 열고, 텐서는 `ty = GgmlType::Unknown(1011)`(`quant.rs:91`)로 남습니다.
  - 호스트 계층은 id·dims·nbytes를 원본 Q3_K 스택과 대조한 뒤 자기 타입 `R8Stack`으로 감쌉니다. 커널 선택은 레버가 아니라 **바이트가 정합니다**. R8Stack이면 row-lane 타일, 파일 Q3_K면 `dot_row_cols`입니다.
  - V4.1에서 사이드카가 없으면 이름 붙은 거부를 권합니다(§10-7).
- 이름 붙은 변형을 넣으면 `strict_tensors`가 사이드카를 받아들이게 되어 여는 순간의 거부가 약해지고, 모든 exhaustive match에 새 arm이 생깁니다.

**읽는 곳 전수**

| 읽는 곳 | 오늘 1011을 만나면 | 뒤 |
|---|---|---|
| `crates/gguf/src/lib.rs:788-792` `strict_tensors` — `let blck = ty.blck_size().ok_or_else(\|\| LoadError::UnsupportedType {` | 여는 순간 거부: `tensor …: ggml type … has no size in this build's table`(:113). `Split::open`·`Gguf::open*`을 쓰는 아래 전부가 여기서 멈춥니다 | 그대로. `open_private`만 예외 |
| `crates/gpu-deepseek41/src/chain/ffn.rs:514-515` `CardStacks::of` — `Some(DevWeight::KQuant { ty, w: stack, .. }) if *ty == want => Ok(stack),` | Q3_K가 아니면 `GpuError::Tensor` | 그대로(원본을 읽음) |
| `crates/gpu/src/weights.rs:494` `upload_segment` — `if s != t.shard \|\| info.ty != t.ty \|\| …` | 이름 붙은 `placed_refusal` | 그대로. 재업로드도 같은 함수 |
| `crates/gpu/src/gemm.rs:137` `GemmWeight::from_ggml` — `other => Err(GpuError::shape(` | 이름 붙은 오류 | 링 슬롯이 타입을 들고 다니므로, R4 전에 R8 슬롯을 GEMM에 넣으면 여기서 거부 |
| `crates/qdot/src/lib.rs:681-682` `check_row` — `if !has_kernel(w) {` | `UnsupportedType`(`dot_row`·`dot_row_cols` 전부) | r8 바이트는 `dot_q3k_r8_cols`(:597)로만. 그 입구의 `check_r8`(:636)은 길이만 보므로 `R8Stack` 타입이 방벽입니다 |
| `crates/gguf/src/quant.rs:377-378` `dequant_row` — `if !ty.has_dequant() {` | `QuantError::Unsupported`(exact_ref 경로) | 그대로 |
| `crates/model/src/placement.rs:299` `CardFormat::of` — `\| GgmlType::Unknown(_) => None,` | 거부가 아니라 호스트에 남깁니다(약점) | 배치는 원본만 보므로 사이드카를 만날 일이 없습니다. §13-3 |
| `crates/model/src/arch/deepseek41/host.rs:27` — `names::ffn_gate_exps(layer),` → `HostLayer::build`(`moe.rs:1489`) | 원본을 읽음 | gate/up은 사이드카 `R8Stack`, down은 원본 |
| `crates/gpu-deepseek41/src/chain/ffn.rs:1440` `Ds41Host::build` — `.map(\|l\| host::layer(&file, hp, l))`, `body.rs:2161` | 원본 | 사이드카 split을 넘기는 서명 변경(카드 쪽 파일) |
| `crates/gpu/src/hybrid.rs:232` `HostResidency::at_load` — `let set = HostSet::of(split, plan, keep)…` | 원본 호스트 집합 | gate/up 구간은 사이드카 split에서 populate·lock |
| 게이트·빈: `gpu-gates/src/bin/exact_ref.rs:432`, `gate_gemm.rs:1852`(Q3_K를 명시), `rawx_floor.rs:101`, `gate_deepseek41_chain_ffn.rs:360,793`, `gate_deepseek41_moe.rs:1469,1571`, `gate_mcol.rs:709`, `bench_v41.rs:387,395` | 원본 → 사이드카를 주면 여는 순간 거부 | 그대로 |
| 호스트 계층을 세우는 게이트: `gate_deepseek41_chain_ffn.rs:1642`, `model/tests/union.rs:283`, `gate_deepseek41_step.rs:2293`, `bench_v41_host.rs`(unionr8은 셋업에서 리팩) | 원본 | 사이드카가 필요합니다(박스에 변환기 선행, `build-ref`처럼) |
| DSpark 초안 파일: `draft/load.rs:341,449`, `gate_dspark_experts.rs:1085` | 자기 파일 | 영향 없음 |
| hoststream 채움(미착륙) | — | 스팬 서술자가 `ty`를 들고, 슬롯이 `ty`를 기록하며, GEMM이 Q3_K를 요구합니다. R4의 un-permute가 슬롯을 Q3_K로 옮깁니다 |

## 5. Part B

**5.1 디코드(m = 1..2)**
- **오늘 경로:** `hybrid.rs:1806` `self.host.experts_into` → `moe.rs:763` `experts_into` → `serve`(:906) → `matmul_q_group_into` → `run_row_pool`(`ops.rs:2466`) → `compute_rows` → `qdot::dot_row`(`lib.rs:313`) → `dot_q3k_q8k_avx2`. 행 super-block당 124 명령입니다(h3tile-a 역어셈).
- **r8 경로:** `dot_q3k_r8_cols`를 C = 1로 부르고, 65.4 명령입니다.
- **명령 항(토큰·층당, 호스트 전문가 4.713개).**

  | 경로 | 명령 | 계산 시간 |
  |---|---|---|
  | 오늘 | 53.9 M | 168 µs |
  | r8 | 28.4 M | 88 µs |

  32스레드 × 10 G명령/s[유도]로 잡은 값입니다.
- **DRAM 항:** 4.713 × 10.14 MB ÷ 133 GB/s = 359 µs입니다.
- **판정:** 둘 다 DRAM에 묶이므로 **벽시계 Δ = 0**입니다[유도]. m = 2에서도 오늘 216 µs, r8 131 µs로 둘 다 < 359 µs입니다.
- **비트:** 정수 경로 재배열이므로 같습니다. 근거는 `lib.rs:584-585` "`out[c][r]` is `dot_row(Q3_K, row r, acols[c], k)` bit for bit"이고, gate-qdot이 C = 1..8을 덮습니다.
- ②에서는 원본의 호스트 gate/up 페이지가 populate되지 않으므로 **디코드 r8은 필수**입니다. 디스패치 경로를 건드리므로 같은 임대 디코드 A/B를 증명 칸에 넣습니다.

**5.2 union 큐 — unionq가 12–16 % 느린 이유**

벤치 팔 `4x8u0.125`는 호출 하나에 **전문가 4개**입니다(round(0.125 × 4 × 8), `bench_v41_host.rs:80-83`). 그래서 "호출당 expert 하나"라는 표현은 부정확합니다. 두 팔의 호출 구조는 gate/up 디스패치 **안쪽**만 다릅니다.

| 항 | union(`moe.rs:1185` `serve_union`) | unionq(`bench_v41_host.rs:958` `union_r8_call`) | 단위 |
|---|---|---|---|
| plan | `plan.build` | 같음 | 호출당 |
| x 양자화, m = 8(좁은 호출) | 한 참가자가 슬롯의 열을 전부 양자화하고 31명이 `wait_ready`에서 스핀(`ops.rs:2124-2128`, `:2138-2142`) | `QuantClaims` 열 단위 claim(:1156-1160) | 호출당. **union만 더 냅니다** |
| x 양자화, m = 16 | `xq.fill` 1(`moe.rs:1210`) | `for_each_chunk` 1 | 호출당, 같음 |
| gate/up | `run_row_pool` 레인: 연속 행, 레인별 카운터 `repr(align(64))`(`ops.rs:2373`) | 1,152 단위를 `next.fetch_add` 하나로(:1164) | 디스패치 1회는 같고, **단위 비용이 전문가당**(전문가당 288) |
| down, `sum_col` | `SwigluClamp` | 같음 | 같음 |

**Δ 분해.** 전문가당 Δ는 m = 8에서 (38.70 − 33.26)/160 = 34.0 µs, m = 16에서 46.6 µs입니다. 따라서 Δa 21.4, Δc 1.58입니다.

- **열당 몫:** Δc를 단위로 옮기면 1.58 × 32/288 = 0.18 µs(스레드 시간, 열·단위당)입니다. 출력 쓰기 `(t0 + t) * ff + R8 * g`(:1194)는 8행 = 32 B, 곧 반 라인이라 이웃 단위와 64 B 라인을 나눕니다. gate와 up 두 라인이 ≈ 90 ns씩 오가면 이 값과 맞습니다[유도, 거짓 공유].
- **고정 몫:** 21.4 × 32/288 = 2.38 µs/단위입니다. 정적 계수로는 0.45–1.1 µs만 설명됩니다.
  - claim RFO + 같은 라인의 `stop`(:1132-1133, :1163) ≈ 0.2 µs
  - 단위마다 17.6 KB 스트림 두 개를 새로 시작 0.2–0.8 µs
  - 단위 셋업 0.05–0.1 µs

  나머지는 종이로 닫지 못합니다. 또 좁은 경로의 직렬 양자화는 union만 내므로, 실제 전문가당 벌점은 21.4 이상입니다.

**박스 실행 하나(R1 라운드용, 기대값 선기록).** `just time-cpu-v41-host`를 `union|unionq:4x64u0.125`와 `4x128u0.0625`로 돌립니다. 호출당 전문가 32개, 청크 4개이고, 벤치가 행 64/128을 받는다는 전제입니다. 여기에 unionq의 claim 블록 B = 18 팔을 더합니다(R1이 레버를 넣음).

| 해석 | 기대 Δa(µs/전문가) |
|---|---|
| 전문가당(내 예측) | +21 ± 3 |
| 디스패치당 | ≈ 10.7 |
| 호출당 | ≈ 2.7 |
| B = 18 | Δa 0–3, Δc ≈ 0(144행 = 576 B = 열마다 온전한 9라인) |

4x8 스윕이 팔 6개에 177 s였으므로 박스 시간은 ≈ 3–5분입니다.

**엔진 설계**
- **디스패치:** `run_row_pool`의 레인(홈 레인을 먼저, 그다음 훔치기, 레인별 카운터)을 층의 모든 (전문가, 8행 그룹) 단위에 전문가 순으로 깝니다.
- **훔치기 블록:** 최대 18그룹(144행)으로 캡을 씌웁니다. 오늘의 lane/4(`ops.rs:2546-2547`)는 층 전체 디스패치에서 ≈ 11 ms 조각이 됩니다. 디코드 폭(lane/4 ≈ 170행)에서는 캡이 사실상 바꾸는 것이 없습니다.
- **열 런:** ≤ 8열씩, ⌈m/8⌉등분으로 고르게 자릅니다. 비트는 분할과 무관합니다.
- **비용표:** (그룹 sb, 런)당 270 + 253·C 명령입니다[유도: C = 1에서 65.4 × 8, C = 8에서 35.8 × 8/열]. 이것이 r8의 `tile_units`(`ops.rs:2348-2355`)를 대신합니다.
- **x 양자화:** 호출당 한 번(`xq.fill`), 8열 이하는 claim입니다.
- **디스패치 수:** 층-배치당 161 → 5입니다. 오늘은 fill 1 + 40 × (gate/up, down 전처리, down) + stash 39 + sum 1입니다.
- **스크래치:** 슬롯 예산 3,072(= 512 × 6)으로 잡습니다. gate/up 56.6 + 결합 28.3 + down 저장 62.9 = 148 MB이고, 오늘은 281 MB입니다(`moe.rs:1096-1099`)[유도].

**5.3 hoststream-host 이음새와 착륙 순서**
- **XS:** `hybrid.rs:1555` HOST 판정에서 목록이 줄어들 뿐입니다. 큐는 plan의 단위만 보므로 안쪽 변경이 없고, 한 번만 건드립니다.
- **꼬리(G × 512 ≤ 4096열):** 타입이 있는 용량으로 받습니다. 슬롯 예산 덕에 폭과 무관하고, x 양자화는 24.2 MB, down 저장은 호출의 슬롯 수로 잡습니다.
  - 현실값 174–229 MB, 최악 503 MB입니다. 현재 `UnionScratch::new(embd, ff, 4096)`는 **2.25 GB**입니다[유도].
  - 이 크기를 hostserve 리뷰에 가져가십시오.
  - `tensor.rs:201`은 카드 활성값 용량이고 꼬리는 호스트 전용이므로 변경이 없습니다.
- **순서:** hostserve(xview+XS+꼬리, 비행 중) → R1 → R3 → R4이고, R2는 R1과 병렬입니다.
  - 두 번 건드리는 줄은 꼬리 용량과 xview의 x 읽기, ≈ 30–60줄입니다[유도].
  - 반대 순서도 줄 수는 비슷하지만, 비행 중인 라운드를 다시 하게 됩니다.

## 6. Part C — 라운드

| 라운드 | 파일·함수 | 소유 | 변경 계층·증명 | 게이트·FAIL-first | 크기 |
|---|---|---|---|---|---|
| **R1 uniondispatch** | `moe.rs` `serve_union`(:1185), `UnionScratch`(:1059), `UnionPlan`(:972), `check_union_call`(:1114). `ops.rs` `run_row_pool` 블록 캡(:2546) | ee | 구조 보존 + 디스패치 수 변경. Δt = −156 × 20–50 µs = −3.1…−7.8 ms/층[유도] | gate-union·gate-gpu-hybrid·gate-gpu-ds41-prefill 비트 동일, 디스패치 수 통계(161 → 5), gate-alloc. **같은 임대 A/B**(pp512 124.9–131.3 예측) + §5.2 벤치 실행 | M |
| **R2 r8conv** | 새 빈(변환기), `gguf` `open_private` + 쓰기기(테스트 전용 `lib.rs:1027`을 승격), `qdot` `unpack_q3k_r8`(`r8_code`/`r8_scale` :1481-1505 활용) | ee(gguf 변경은 ee 쪽, 69에 통지) | 새 도구 | 왕복 게이트 unpack(repack(x)) = x(FAIL-first: W2/W5 상위 비트 교환 → 빨강). strict open이 사이드카를 거부(FAIL-first: id 11로 쓴 사이드카 → 통과해서 빨강). 변환기 verify는 모든 텐서가 원본으로 풀리는지. 박스 ≈ 5분, 최악 15분[유도: 읽기 117 s + 쓰기 27 s + verify] — 30분 미만 | M |
| **R3 r8host** | `HostLayer`(:1477)·`R8Stack`, `serve`/`serve_union`의 r8 레인, `hybrid.rs:211-232` 이중 HostSet, `arch/deepseek41/host.rs`, 신원 대조. **카드 쪽 서명**: `chain/ffn.rs:1432`, `body.rs:2161` | ee(+69 서명) | 정수 경로 재배열 + 로더 | gate-union(원본 `dot_row` 오라클과 비트 동일), gate-gpu-hybrid, ds41 step/prefill 비트, 신원 거부 게이트(FAIL-first: 다른 헤더 digest), gate-alloc. **같은 임대 A/B** 프리필 + 디코드(Δ 0 예측) | M–L |
| **R4 r8ring** | 장치 `q3k_r8_unpermute`(그룹당 CTA 하나, 제자리, 링 한 바퀴당 발사 하나), 슬롯 상태 R8 → Q3_K, 재업로드 원천 상주 | 69 | 새 커널 + ptx 핀 | 장치 un-permute = 원본 Q3_K md5(FAIL-first: 인덱스 맵 변조). R8 슬롯을 직접 enqueue하면 이름 붙은 거부. gate-ptx-spill 행. DEMOTE 게이트 그대로 | M |

**R3과 R4의 착륙.** hoststream ②가 R3보다 먼저 착륙하면, R3 뒤의 채움은 populate되지 않은 원본 gate/up을 NVMe로 읽게 됩니다. 조용한 성능 절벽입니다. 그래서 R3이 스팬에 `ty`를 싣고, R4 전까지 사이드카 + hoststream 조합은 이름 붙여 거부해야 합니다.

## 7. 예측 (lcg, A6000, 같은 임대 앵커 union 30.2/22.20 = 76.1 ms/층)

"오늘 흐름" 행은 S14 실측/모형 비로 앵커했습니다. 밴드는 prop·add 잔차 × 커널 밴드 × 디스패치 레버입니다.

| 흐름 | pp512 | pp4096 |
|---|---:|---:|
| 오늘(S14 실측) | 121.4 | 169.4 |
| R1만(레버 −3…−8) | 124.9–131.3 | 174.1–182.6 |
| + r8 중앙(38.0/13.05, 레버 −3) | 149–155 | 206–214 |
| + r8 전체 밴드 | **138–168** | **192–231** |
| hoststream G1 R128 / G8 R128, 오늘 호스트 | 159.6 | 452.7 |
| hoststream + r8 밴드(중앙) | 170–190(174–181) | 465–476(467–471) |
| hoststream + 카드 route 절반, 오늘 → r8 | 182.6 → 200–210 | 614.5 → 645–660 |

- r8은 P4096 G8 R128에서 최적 k를 층당 159 → 134로 낮춥니다. 스트리밍하는 바이트가 −16 %입니다[유도].
- 리드의 ≈ 65 / ≈ 57을 확인했습니다. 전문가당 65.2, 호출당 57.0 ms/층(prop)입니다.
- 디스패치당 해석이면 오늘의 40청크에서 60.7이고, R1 뒤에는 57.0입니다. **R1을 먼저 착륙시켜야 하는 근거**입니다.
- 모든 값은 [유도]입니다. 커널 쪽 c와 a는 unionr8이 익명 메모리를 읽으며 잰 값이고(rig-log :538 단서), 엔진은 파일 페이지를 읽습니다. 둘 다 4 KiB 페이지라 차이는 예상하지 않지만 재지는 않았습니다.

## 8. 엔진 비교

- **ik:** `-rtr`가 mmap을 끕니다(`common.cpp:2220-2222` `params.use_mmap = false;`, `llama-model-loader.cpp:589-591`).
  - 적재 뒤 호스트 버퍼에서 제자리로 리팩합니다(`llama.cpp:5203-5214`, `iqk_quantize.cpp:8731-8736` 스레드 뒤 `tensor->type = r.new_type;`). `repack_q3_k`는 4행 단위입니다(:6583).
  - **오프라인 형식도 있습니다**(`quantize.cpp:59` `"Q3_K_R4" … "Q3_K_S repacked"`).
  - 모르는 타입은 `GGML_ASSERT(0 <= info->type && info->type < GGML_TYPE_COUNT);`(`ggml.c:31417`)로 abort합니다.
  - 우리 방식에 가장 가깝지만, 전체 익명 메모리에 매 시작 재독입니다.
- **llama.cpp:** CPU_REPACK은 익명 CPU 버퍼입니다(`repack.cpp:5168-5180`).
  - `set_tensor`가 파일 바이트를 리팩하고(:5150-5160), 타입은 Q4_K로 두고 `extra`로 표시합니다(:5143-5144). AVX2 Q4_K는 8x8이고(:5006-5009), Q3_K 리팩은 없습니다.
  - **디스크 형식은 제거했습니다**(`ggml.h:421` `// GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files`). 로드에서 범위를 검사합니다(`gguf.cpp:722-724`).
- **exllamav3:** 프로세스별 익명 arena로 적재 때 복사합니다(`moe_cpu_host.py:247-255` `rehome`).
  - 스위즐은 AVX-512 티어에서만 하고, AVX2는 원래 레이아웃입니다(:90-94, :321-326).
  - opt-in으로 memfd arena를 CUDA에 등록해 직접 DMA합니다(:104-116, "~0.2 s per GiB at load"). ③의 장점과 같은 설계입니다.
- **mistral.rs:** GPU Marlin 리팩, "a second full set of weights"입니다(`packed_affine.rs:282-297`). CPU 디스크 레이아웃은 없습니다.
- **정리:** 넷 모두 적재 때 자기 버퍼로 리팩하고, 그 비용(RAM 두 벌 또는 매 시작 재독)을 치릅니다. 우리는 214/264 GB와 잦은 재시작 때문에 그것을 치를 수 없어서, ik식 비공개 타입 + 오프라인 파일을 **원본 옆 사이드카**로만 둡니다.

## 9. 추천과 근거

②를 권합니다. 근거는 다섯입니다.
1. 카드 업로드와 재업로드가 바뀌지 않은 바이트를 바뀌지 않은 코드로 읽습니다. un-permute도, 카드 쪽 형식 코드도 0입니다.
2. RAM과 따뜻한 재시작이 오늘과 같습니다.
3. 디스크 155.7 GB가 들어갑니다.
4. 태그가 여는 순간 이미 있는 코드에서 거부됩니다.
5. ③은 시작당 ≥ 95 s, ①은 +347.3 GB에 로더 둘 + 공유 0입니다.

우선순위는 **R1 먼저**(비트 동일, +3…+8 %, 큐 모호성 해소), 그다음 R2 ∥ R1, 이어서 R3 + R4입니다. hoststream 뒤 r8의 값은 P512에서 +11 %, P4096에서 +3.6 %입니다. 파트 C로 route가 반이 되면 +14 % / +6.7 %로 커지는 레버라는 점을 저울에 올리십시오.

## 10. 리드가 물어야 할 것

1. **(69) 재업로드 원천.** DONTNEED 뒤라 NVMe에서 1.8 s가 걸립니다. 89 ms 예산은 pinned를 전제합니다. 적재 때 pinned 사본 2.34 GB(권장)와, DONTNEED 제외 + mlock 중 무엇으로 할지 정해야 합니다.
2. **(69)** 채움이 스팬을 한 함수로 읽는지(그 함수가 `ty`를 받도록), GEMM enqueue가 슬롯 `ty`를 확인할지, un-permute 발사 위치(복사 스트림, 링 한 바퀴당 하나)를 물어야 합니다.
3. **(사용자) 디스크.** /models +155.7 GB와 변환기 1회 ≈ 5–15분입니다.
4. **(사용자·리드) 시팅 프로토콜.** 오늘은 참조 엔진과 우리가 같은 페이지를 나눠 씁니다. 우리 팔 뒤의 ik r2 주 폴트가 423회였습니다(rig-log 09-25).
   - ②에서는 참조의 gate/up(최대 128.7 GB)을 캐시가 함께 담지 못합니다. 엔진을 바꿀 때마다 양쪽이 차가워집니다: 참조 ≤ 97 s, 우리 재-populate ≤ 97 s[유도].
   - 회전 순서의 `depth-ds41.sh`에서는 매 바퀴 참조 행이 무효가 됩니다. 엔진별로 몰아 돌리고, 전환마다 버림 실행을 하나씩 두어야 합니다. ①은 214 GB 전부라 더 나쁩니다.
5. **(사용자)** 비공개 id 1011과 아키텍처 이름 `bloomery-r8`을 정해 주십시오.
6. **(리드)** 이 프로그램이 hoststream 뒤 수치에 견줄 가치가 있는지입니다(§9).
7. **(리드)** V4.1에 사이드카가 없을 때 이름 붙은 거부(권장)로 할지, 오늘 경로를 돌리고 load 줄에 표기할지 정해야 합니다.

## 11. 실행한 계산 (게이트 절 대신; Mac Python만, 박스 실행 없음)

    PYTHONDONTWRITEBYTECODE=1 python3 pred.py
    RAW512 union same-lease 63.95 ms, R' = 1.1900; report anchor RAW512(28.1,22.98) = 65.14, R_PROP = 1.1682
      unionr8 as-is, per expert                65.2 | ...
      unionr8, per call (+29.2us x4 / 317)     57.0 | ...
      lanes+r8 central                         55.3 |  153.7  210.9 ( 149.4  206.5) |  163.8  177.2 |  405.6  468.9 (k 133.6)
    a 38.0 c 13.05: m* = 6.77, share of host experts at the W floor = 0.361, P(m=0) = 0.00047
    PYTHONDONTWRITEBYTECODE=1 python3 pred_add.py
    RAW512 union 63.95 ms, residual 12.15 ms/layer; R_ADD = 5.035 us per host slot (SLOT_TOK 4.713)
      lanes+r8 central   (add, lever 0)        U512   58.6 | today pp512  144.1 pp4096  199.6 ... | hoststream P512 G1 R128  173.8  P4096 G8 R128  467.1
      unionr8, per dispatch x40 chunks (prop)  U512   60.7
    python3 timeline.py
    lanes+r8 central  P512 G1 layer  0 T=  512: no-stream wall   86.9 (host   86.6 attn   30.4) | stream k= 78: wall   75.1 host_done   74.8 attn   30.4 stream_done   50.3 pcie   50.0
    union             P4096 G8 layer  0 T= 4096: ... stream k=178: wall  325.2 host_done  322.8 attn  282.1 stream_done  319.8 pcie  114.2
    python3 dma.py
    lanes+r8 central  | staging: P512 G1 R128  177.2  P4096 G8 R128  468.9 | direct: P512 G1 R128  185.6  P4096 G8 R128  472.0
    python3 partc.py
    lanes+r8 central  Part C (route 15.2 ms/512): P512 nostream  185.7 | P4096 nostream  252.1 | P512 G1 R128  207.7 | P4096 G8 R128  655.9

## 12. 못 한 것

- 적재 92 s 중 CUDA 초기화·그래프 몫은 이번 독해로 나누지 못했습니다.
- ik `--n-cpu-moe 33`이 어느 층을 CPU에 두는지 확인하지 못해, 공유 손실은 상한(128.7 GB)으로만 적었습니다.
- prose 프롬프트 팔은 스펙대로 계산하지 않았습니다.

## 13. 스펙 밖 개선 지점 (보고만, 손대지 않음)

1. `crates/model/src/bin/bench_v41_host.rs:1132-1133` — `next`·`stop`이 인접한 스택 원자값이라, claim 때마다 모든 참가자가 폴링하는 라인(:1163)을 무효화합니다. XS.
2. `bench_v41_host.rs:1194` — 32 B 출력 쓰기가 이웃 단위와 라인을 나눕니다. unionq의 Δc 1.58과 크기가 맞는 거짓 공유입니다. S(R1의 144행 블록으로 닫힘).
3. `crates/model/src/placement.rs:265-267,299` — 모르는 타입의 라우팅 스택은 거부가 아니라 호스트 배치입니다(조용한 재계획). 이름 붙은 거부 후보입니다. S.
4. `crates/model/src/ops.rs:2546-2547` — 훔치기 블록 lane/4가 층 전체 디스패치에서 ≈ 11 ms입니다. 캡이 필요합니다. S(R1).
5. `crates/model/src/moe.rs:1096-1099` — `UnionScratch`가 전문가별 최악으로 잡혀 512열 281 MB, 4096열 2.25 GB입니다. 슬롯 기준으로 바꿔야 합니다. M(R1/hostserve).
6. `bench_v41_host.rs:655-659` — `vec![0u8; …]`로 9.73 GB를 영으로 채운 뒤 덮습니다. 25.96 s는 패커 속도가 아니라 폴트·NVMe 속도입니다(패커는 연산 수로 스레드당 ~0.4 GB/s[유도]). XS.
7. 스펙의 "single thread" — `bench_v41_host.rs:670-687`은 풀 참가자마다 스레드 하나(32개)입니다. XS 문서.
8. rig-log `log/2026-09-25.md:540` "호출 하나에 expert 하나" — 실제는 4개입니다. 선을 그어 정정할지는 리드 몫입니다. XS.
9. ik `ggml/src/ggml.c:31417` — 모르는 타입에서 이름 붙은 오류가 아니라 abort입니다(메인라인 `gguf.cpp:722-724`는 오류를 냅니다). 업스트림 후보이고 우선순위는 낮습니다. XS.

## 14. 모델

이 라운드는 opus(Opus 5.5)로 스폰되어 돌았습니다.
