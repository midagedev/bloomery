# 감사자 ③ 보고: 적재, 배치, 하이브리드 경계, engram

`2f0182f`을 읽기 전용으로 감사했습니다. 박스 명령은 하나도 돌리지 않았습니다.

가장 큰 발견은 세 가지입니다.
- **engram 비용 어긋남**: 계획 문서가 적은 engram 조회 "0.31 ms [측정]"은 행마다 `WILLNEED`를 거는 측정 팔의 값입니다. 엔진 코드는 그 팔이 아니라 QD1 demand-fault 팔을 돕니다.
- **호스트 expert 잠금 누락**: 호스트 expert를 잠그는 `HostLock`은 설계에 있고 코드도 있습니다. 그런데 엔진 적재 경로는 그것을 한 번도 부르지 않습니다.
- **B12 대상 좁아짐**: V4.1 쪽 B12는 커널을 바꾸지 않아도 되고, 적재기와 배치만 고치면 됩니다.

## 1. 발견 표

### 성능 (Δ 큰 순)

| path:line | 시선 | 종류 | 기제 | Δ [유도 — 항, 조건] | 변경 클래스 + 증명 | 크기 | 비행 중 간섭 |
|---|---|---|---|---|---|---|---|
| `crates/gpu-deepseek41/src/chain/glue.rs:680-760` (`StepRows::fill`), 호출 `body.rs:693` | perf | async | engram 48행을 스텝 스레드가 런치 전에 `Site::copy_rows`로 mmap에서 곧바로 읽습니다. `WILLNEED`도 헬퍼도 없습니다(`Prefetcher`·`RowCache`를 쓰는 곳은 엔진에 0개입니다). 이것은 rig-log 09-22-p의 "콜드, 행 하나씩 demand fault(QD1)" 팔입니다. 계획 문서의 0.31 ms(`v41-placement.md:236`, `plan.md` B5 결정)는 같은 기록의 선행 읽기 팔(310–335 µs)이라 엔진 코드와 맞지 않습니다. 콜드 페이지가 하나 나오면 GPU와 풀이 모두 쉬는 동안 1:1로 스텝에 더해집니다 | **콜드 행**: 코드 스트림 미스 8.38행 × 1.0625페이지 × 75.4–92 µs = **+0.67–0.82 ms/스텝**. 산문 25.72행이면 **+2.06–2.51 ms**, 균등 id 최악은 **+3.84–4.69 ms**[측정 팔]. **페이지 캐시에 있으나 매핑되지 않은 행**: b5step이 잰 minor fault 120–250/스텝 × 2.5 µs = 0.30–0.63 ms. 조건 @ n=1, 스텝마다, A6000 + 호스트 | 행을 스텝 이미지에서 빼서 매핑 버퍼로 옮깁니다. argmax 직후 헬퍼에 제출하고, `engram_kv` 앞에 wait 노드를 1–2개 둡니다(아래 넷째 행과 함께). 헬퍼는 측정된 헬퍼 팔의 모양이고, 스텝 몫은 gap 500에서 1.0–1.1 µs[측정 09-23]입니다. 클래스는 런치 수만 바뀌는 것(Δ노드 +1–2 × `c_node`)이고, 증명은 G1 `engram_out`와 호스트 행 id = int 행의 비트 동일입니다 | M | 없음. `ktok`는 k토큰이면 행 수가 k배라는 점을 알아야 합니다 |
| `crates/model/src/placement/host_lock.rs:59` (`HostLock::lock`) 대 `crates/gpu-deepseek41/src/body.rs:76-99`·`:870-873` | perf | inefficiency | 설계는 "호스트 expert를 샤드 mmap에서 잠가서 쓴다"(`v41-placement.md:167`, 축출 방지)입니다. 그런데 `HostLock`을 부르는 곳은 `gate_load_v41.rs:645` 하나뿐이고, `body::open`·`generate_ds41`은 부르지 않습니다. `Split::open`은 lazy(`open_shard`, populate 없음)입니다. 그래서 새 프로세스는 호스트 expert 페이지를 **처음 만질 때마다** fault를 냅니다. b2h의 128 GB/s 벤치는 `pages.populate()`(`bench_v41_host.rs:1368`)를 한 뒤에 쟀으므로 이 항을 모릅니다. 구조적 원인도 있습니다. `HostLock<'a>`가 `&'a Split`을 빌리므로 Split을 소유한 `Body`가 그 잠금을 가질 수 없습니다(자기 참조가 됩니다). `Ds41Host`는 Split을 따로 하나 더 엽니다(`body.rs:873`) | 정상 상태 값은 원인 축소 C1이 먼저입니다. 상한 신호는 깊이 표의 역전입니다. 깊이 6이 40.31 ms, 깊이 400이 39.31–39.49 ms, 4096이 39.4 ms로, **얕을수록 0.8–1.0 ms 느립니다**. 비측정 프롬프트가 짧을수록 비측정 구간에서 한 번도 안 만진 expert가 많습니다[유도, 균등 라우팅: 15스텝 뒤 호스트 expert 12,946개 중 ~25 %만 만짐]. 잠금이 있으면 적재가 218 GB ÷ 1.33–7 GB/s = **+31–164 s** 늘어납니다[유도] | 적재에서만 바뀝니다. 스텝 비트 동일, 노드 수 동일입니다. 증명은 스텝별 fault 계수 = 0입니다(트리아지 ⑪의 텔레메트리) | S–M (`Arc<Split>` 소유 또는 span 소유형 잠금) | 없음 |
| `crates/gpu/src/weights.rs:299-340` (`upload_segment` → `g.data(info)`) | perf | inefficiency | 카드에 올리는 파일 바이트를 mmap으로 읽으므로 전부 페이지 캐시에 남고, 다시는 읽히지 않습니다. 남는 자리 43.78 GB는 호스트 가용 270,071,001,088에서 expert 218,277,273,600, `token_embd` 1,323,827,200, OS 6,694,629,376을 뺀 값입니다. 카드 파일 바이트는 약 48.15 GB(밀집 7,657,593,280 + expert 2,414 × 16,773,120)입니다. 그래서 적재 한 번이 **≥ 4.37 GB**를 밀어내며, 따뜻하던 호스트 expert나 engram 행이 대상입니다[유도] | 정상 상태는 0입니다. 적재 뒤 첫 토큰들에 4.37 GB ÷ 4 KiB × 75–92 µs ÷ 16 스레드 = **~5–6 s**의 재폴트가 퍼집니다[유도]. b5time의 무효 행 r1("재부팅 뒤 첫 적재, 페이지인")과 같은 모양입니다 | 업로드한 span마다 `posix_fadvise(DONTNEED)`를 한 줄 걸거나 `O_DIRECT`로 읽습니다. 적재에서만 바뀌고 비트 동일입니다 | XS–S | 없음 |
| `crates/gpu-deepseek41/src/body.rs:480` (`glue.enqueue_engram_kv`가 0층 attention **앞**) | perf | async | 카드는 "`engram_wkv` 0.51 ms는 0층 그늘로 간다"라고 적습니다(`plan.md` B5 결정 셋째, `v41-placement.md:277`). `glue.rs:11-15` 문서도 "the step puts them in layer 0's host-leg shadow"라고 적습니다. 코드는 첫 go 앞, 임계 경로 위에 둡니다. 트리아지 b5step "hook 없음"과 같은 자리이고, 새로운 것은 **R1 예측이 이 항을 그늘로 셌다**는 점입니다. 측정 39.4 ms와 예측 39.7 ms가 맞은 것은 이 0.48 ms를 품은 우연입니다. 둘째 사이트(14층)의 행은 13층 동안 필요 없습니다 | **−0.48 ms/토큰**[측정 런치당 237.99 µs × 2사이트, rig-log 09-23 B11c 표] @ n=1, 모든 깊이, A6000 | 이동 클래스입니다. 노드 수 동일(1,110), eager = replay, e2e 집합 동일이 증명입니다. 첫째 행과 합치면 행 대기도 그 그늘로 갑니다 | S | b5prof의 nsys 귀속 표가 이 0.48을 임계 경로에서 보게 됩니다(참고) |
| `crates/model/src/placement/workstation.rs` `MARGIN`(1 GiB) + 미할당분 | perf | algorithm | 적재 뒤 A6000에 1,510,998,016 B가 비어 있습니다[측정 b11c]. 이는 expert 90개어치입니다(÷ 16,773,120). 이 중 margin 1 GiB(= expert 64.0개)는 [가정] 값이고, context 512 MiB와 scratch 64 MiB도 [가정]입니다(트리아지 ⑯: 조각 scratch 추정 67 MB 대 실측 2.7 MB) | 접두 규칙에서 카드 expert 하나 = 16,773,120 × 6/384 B ÷ 128.8 GB/s = 2.03 µs/토큰이므로 **최대 −0.18 ms**(90개)[유도] @ n=1, A6000 + 호스트. B12의 뜨거운 목록에서는 한계 슬롯 값이 달라집니다 | 원인 축소 C4가 먼저입니다. 그 뒤 n_l 재핀(적재 게이트 ②, 배치 핀)을 합니다 | S | 없음 |
| `crates/gpu/src/hybrid.rs:1059` (`payload_f32_into(X_W, …, &mut self.x.data)`) | perf | inefficiency | 층마다 20,480 B 활성값을 매핑 페이지에서 `Tensor2`로 복사한 뒤 호스트 expert에 넘깁니다. 이 페이지는 캐시 가능한 핀드 메모리(`DEVICEMAP`만, WC 아님)라 제자리에서 읽어도 됩니다. 호스트 다리는 임계 경로입니다 | 40층 × ~1–2 µs = **−0.04–0.08 ms/토큰**[유도, 단일 스레드 memcpy ~10–20 GB/s] | 정수 경로 재배열(비트 동일)입니다. `gate-gpu-hybrid`와 ds41 step이 증명입니다 | XS–S | ② CPU 감사자의 `experts_into` 입력 계약과 맞물립니다 |
| `glue.rs:716-736` (`embd` 행) + `token_embd` 미잠금 | perf | inefficiency | 처음 보는 토큰의 bf16 행 10,240 B가 3–4페이지에 걸칩니다. 콜드면 스텝 스레드가 QD1로 읽습니다. 1.32 GB 표는 `HostLock` 예산에 이미 있지만(`v41-placement.md:163`) 잠그지 않습니다 | 콜드 토큰 하나에 3 × 75–92 µs = **0.23–0.28 ms**[유도]. 따뜻하면 0 | 둘째 행의 잠금 `keep`에 `token_embd`를 넣습니다. 적재에서만 바뀝니다 | XS | 없음 |

### 품질

| path:line | 시선 | 종류 | 기제 | Δ | 변경 클래스 + 증명 | 크기 | 간섭 |
|---|---|---|---|---|---|---|---|
| `crates/gpu/src/arch/deepseek2/mod.rs:566-612` (`hybrid_row`)와 `:393-434` | quality | duplication | `placement::place_routed`·`card_segment`의 접두 분할을 손으로 다시 적은 둘째 주인입니다(R14). 거기에 **총계가 0인 가짜 `Plan`**(usable 0)을 만들어 `Weights::load_placed`에 넘깁니다. 이 Plan에 `violations()`를 부르면 틀린 답이 나옵니다. B12는 이 파일과 `placement.rs` 둘 다 고쳐야 합니다 | — | 이동 클래스입니다. V2-Lite 하이브리드 게이트와 e2e 비트 동일이 증명입니다. `load_placed`가 `Plan` 대신 `&[Row]`를 받게 하면 끝납니다 | S | B12 |
| B12 접두 가정 목록(질문 3) | quality | gate | V4.1 커널 경로는 **이미 목록을 받을 준비가 돼 있습니다**. 핸드오프가 `sel[s] = map[row_off + id]`로 쓰고(`chain/ffn.rs:106-107`), `ds41` gate·up(`experts.rs:172`)·`q4k_sel`·`ds41_ffn_post`는 슬롯과 `n_card`를 읽습니다. `SlotMap::from_rows`와 `plan_slots`(`body.rs:1098`)도 준비돼 있습니다. **접두를 가정하는 곳**은 다음과 같습니다: `placement.rs:780-808`(`place_routed`), `:339-362`(`Segment::span`이 연속 범위 하나), `weights.rs:196-201`(카드 위 세그먼트 둘 거부), `:320-324`("rows do not lead" 거부), `:457-459`(`bytes.get(..n)`), `gate_deepseek41_load.rs:315-316`(기대 맵을 `plan.n_l` 접두로 만듦), `gate_load_v41.rs:27,403`(첫 접두 세그먼트를 읽어 비교), `host_lock.rs`의 host span(목록이면 여러 범위가 됨), 위의 `hybrid_row`. `footprint`·`Held::Experts`는 개수만 보므로 id와 무관합니다. **주의**: expert마다 세그먼트를 따로 할당하면 2 MiB 반올림이 expert당 1,222,656 × 2 + 1,753,088 = 4,198,400 B(25 %)를 버립니다. 2,414개면 10.1 GB, 즉 expert 604개어치입니다[유도]. 그래서 스택마다 버퍼 하나에 모아(gather) 올려야 하고, `Segment.experts`는 `Range`가 아니라 목록 타입이 되어야 합니다 | — | 카드 명세입니다: 항등 목록에서 게이트 전부 비트 동일 | M (B12 몫) | B12 |
| 트리아지 ⑥ B12 행 | quality | gate | 적힌 네 판독자는 확인했습니다. 셋은 줄이 맞고(`q4k_sel.rs:89`, `q5.rs:736`, `moe_fused.rs:99`), `lib.rs:1323`은 지금 **`lib.rs:1343`**입니다. 넷 다 V2-Lite 전용입니다(id를 슬롯으로 씀). 더할 카드 쪽 판독자는 없습니다. `experts.rs:172`는 이름이 `id`일 뿐 슬롯을 읽습니다 | — | — | XS | — |
| `crates/gpu-deepseek41/src/chain/glue.rs:11-15` | quality | error | 문서가 코드와 다릅니다. 넷째 성능 행의 "그늘"은 거짓입니다(AGENTS 주석 규칙) | — | 문서만 바뀝니다 | XS | — |
| `crates/model/src/placement/host_lock.rs:47-66` | quality | error | `HostLock<'a>`가 빌림이라 엔진 소유자 구조에 들어갈 수 없습니다. 게이트 전용 도구로 굳었고, 설계가 전제한 서빙 조건이 코드에 없습니다(둘째 성능 행의 구조 원인) | — | 소유형(`Arc<Split>` 또는 span 소유)으로 바꿉니다 | S | — |
| `generate_ds41.rs:295-302`, `body.rs:870-873`, `glue.rs:625` | quality | duplication | 적재 한 번에 같은 파일을 셋이 따로 엽니다. 본체 Split, `Ds41Host`의 둘째 `Split::open`(mmap 한 벌 더), `Engram::open`(9샤드 헤더를 `inventory_of`로 다시 파싱)입니다. 여기에 `generate_ds41`이 `PlanInputs::read`와 `plan`을 두 번 합니다(출력용과 `body::open`) | 적재 시 ms | 이동 클래스 | S | — |
| `crates/engram/src/lib.rs:77` (`Io(#[from] io::Error)`) | quality | error | `Site::open`의 `File::open(path)?`·`advise(...)?`와 `prefetch`의 `advise_range(...)?`가 경로와 사이트 이름을 잃습니다(R11). `Prefetch`·`Cache`·`Reuse(&'static str)`도 문맥이 없습니다 | — | 오류 변형만 바뀝니다 | XS | B3c 몫과 겹칩니다 |
| `crates/gguf/src/quant.rs:448`·`:490` | quality | error | 라이브러리의 `blk[4..16].try_into().unwrap()`입니다(R10). `first_chunk`/`as_chunks`로 바꿉니다 | — | 비트 동일 | XS | — |
| `crates/gpu/src/lib.rs:1378` | quality | error | `Gpu` 문서가 "context on device 0"이라고 적지만 지금은 `for_card`로 아무 카드에나 만듭니다 | — | 문서 | XS | — |
| 트리아지 ① 경로 | quality | duplication | `one_tensor_file`은 `model/tests/ops.rs`가 아니라 `crates/model/src/ops.rs:2441`(`#[cfg(test)]`)에 있습니다. 새 GGUF 작성기는 없습니다(`b"GGUF"`를 쓰는 곳은 셋 그대로) | — | — | XS | — |

## 2. 모델이 기각한 것 (산수)

- **B3c(`io_uring`)의 스텝 값 ≈ 0**: 첫째 성능 행이 들어오면 스텝 스레드 몫은 헬퍼 대기 1.0–1.1 µs[측정 gap 500]입니다. 콜드 48행 선행 읽기 0.31–0.34 ms는 0층 그늘 ~0.98 ms[유도 12.95/40 + 26.54/40] 안에 숨고, 14층 사이트에는 13층분의 여유가 있습니다. B3c에 남는 값은 헬퍼 CPU와 페이지 캐시 반환뿐입니다.
- **rope 표**: 토큰당 128–224 `sin_cos` × 20–40 ns = 3–9 µs입니다. 역방향 두 표가 정방향과 같은 `sin_cos`를 다시 계산하는 몫(`params.rs:672-693`)은 64회, 1.3–2.6 µs입니다. 둘 다 스텝의 < 0.03 %입니다. 트리아지의 "rope 표를 적재 때"도 같은 크기입니다.
- **이미지 H2D 24.3 KB(pageable)**: 전송 ~2 µs(~12 GB/s)에 API 고정비 ~5 µs입니다. 매핑으로 바꿔도 ≤ −7 µs입니다.
- **이미지 조립**: `words.fill(0)`, engram 행 이중 복사(`copy_rows` 뒤 바이트를 워드로 다시 포장, `params.rs:707-714`), embd u16을 u32로 다시 포장. 합쳐 23 KB로 1–3 µs입니다.
- **해시**: `rows_into`는 u64 `%` 48회 ≈ 1,700 cycle ≈ 0.5 µs이고, `plan_into`는 push ~30회로 < 1 µs입니다.
- **`token_embd`를 카드에**: 1.32 GB VRAM = expert 79개 × 2.03 µs = +0.16 ms/토큰을 내고, 얻는 것은 10 KB H2D와 호스트 복사 ~2 µs입니다. 손해입니다.
- **적재 시간 쪽**: `spread`는 O(E×U), 2,414 × ~1,300 할당으로 0.1–0.3 s입니다. `Q8_0Planes`의 `Vec<Q8Block>`, 두 평면, `words_of`까지 호스트 사본이 세 벌이라 48 GB × 2 ÷ ~10 GB/s ≈ 10 s입니다. 둘 다 적재에서만 들고 스텝 값은 0입니다.

## 3. 원인 축소 항목

- **C1 깊이 역전(6 > 400·4096에서 +0.8–1.0 ms)**: `generate_ds41`에 스텝별 `RUSAGE_SELF` major·minor 차를 찍습니다(트리아지 ⑪). 깊이 6을 `--warm 400`으로 돌리는 팔과 `HostLock` 또는 populate 팔을 같은 임대에 둡니다. first-touch fault와 다른 항이 이것으로 갈립니다.
- **C2 스텝 스레드 minor fault 120–250/스텝(b5step)**: engram 51페이지와 embd 3–4페이지를 넘는 몫이 어느 매핑에서 오는지는 코드로 설명되지 않습니다. 스텝 스레드에 `perf record -e page-faults -d`를 걸고 주소를 engram, `token_embd`, expert VMA에 대응시킵니다.
- **C3 engram 콜드 비용이 표에 없음**: `depth-ds41.sh`의 팔은 매번 같은 `lcg_prompt`와 greedy 수열을 돕니다. 그래서 r2와 r3에서는 engram 행이 페이지 캐시에 이미 있습니다. 측정 전에 스트림의 행을 `Site::evict_rows`로 비우는 콜드 팔, 또는 라운드마다 다른 프롬프트를 두고 `read_bytes`를 증인에 넣습니다.
- **C4 VRAM margin**: 캡처 뒤 재생 1,000회 동안 `cuMemGetInfo` free의 최솟값을 봅니다. margin과 context 가운데 실제로 쓰이는 몫이 이것으로 나옵니다.
- **C5 적재 2–3분의 구성**: plan, 카드 읽기·포장·H2D, 호스트 티어, 캡처를 단계별로 잽니다. 잠금의 +31–164 s를 서빙에서 감수할지는 이 값이 정합니다.
- **C6 뜨거운 engram 행을 고정하는 값**: `engram-reuse`로 한 코퍼스에서 배운 행 집합이 다른 코퍼스의 스트림 내 강제 미스를 얼마나 덮는지 잽니다. 네 코퍼스가 이미 있고, B12 라우터 교차 검증과 같은 모양입니다.

## 4. 리드 질문에 대한 답

- **Q1 런치 전 스텝 스레드 일**: 순서는 `check_pos`, `plan_into`(< 1 µs, **위치만**), `fill`의 embd(**토큰만**, 변환 1–3 µs + fault 0–0.28 ms), `fill`의 engram(**토큰과 앞 3토큰**, 해시 0.5 µs + 복사 1 µs + fault 0.13 ms에서 4.69 ms), `image.build`(rope 3–9 µs는 **위치만**, 나머지 1–3 µs), H2D ~7 µs입니다. 토큰에 달린 일은 앞 스텝의 argmax가 나와야 하므로 한 스텝 앞당길 수 없습니다. 위치에만 달린 일(~10 µs)은 앞 재생의 꼬리, 곧 헤드 ~0.8 ms로 옮길 수 있지만 값이 거의 0입니다. 스텝에 남은 적재 시점 일은 없습니다. 이름 조회는 없고, 매 스텝 도는 것은 `data(info)`의 경계 검사 하나뿐입니다.
- **Q2 engram**: 사이트마다 24행입니다(3차수 × 8헤드, `hash.rs`). 두 사이트면 48행 × 272 B = 13,056 B이고 51.0페이지입니다. 미스는 스텝을 1:1로 세우며, 비용은 첫째 성능 행에 적었습니다. 고정 예산은 43,775,270,912 B[유도: `v41-placement.md` b11c 여유 + 행 캐시 예약] ÷ 272 = 1.61억 행으로, 전체 7.68억 행의 21 %입니다. 측정된 스트림의 서로 다른 행은 0.49–1.65 GiB라 예산은 넉넉합니다. 값이 남는 곳은 스트림을 건너는 강제 미스뿐입니다(C6).
- **Q3 배치**: `n_l`은 개수로서 예산 변수로 남기면 됩니다. B12에 필요한 것은 층마다 id 순서(항등 순서 = 오늘)와 목록형 세그먼트입니다. 가정 목록은 품질 표 둘째 행에 있습니다.
- **Q4 3090에 둘째 모델**: 보고만 합니다. 걸리는 곳은 다음과 같습니다.
  - `body::open`이 카드 둘 이상을 거부합니다(`body.rs:84-92`).
  - `GpuModel::step`은 스테이지 하나를 요구합니다(`model.rs:684`).
  - `MappedHost`는 `DEVICEMAP`만 쓰고 `PORTABLE`을 안 써서(`hybrid.rs:333`) 3090 컨텍스트가 그 페이지를 매핑하지 못합니다.
  - 코드에 peer 경로가 없고, P2P는 측정상 없습니다(`v41-placement.md:186`, GNS).
  - 호스트 서비스는 프로세스 전역 `threads::pool()`을 씁니다.
  - q8_0 `_sel` 커널이 없습니다(`_sel`은 q3_K, q4_K, q5_0뿐).

  가장 싼 길은 `PORTABLE|DEVICEMAP` 페이지를 거치는 호스트 경유입니다. 3 × 5120 f32 = 61,440 B를 A6000이 쓰고(~2.5 µs @ ~25 GB/s[유도]) 스트림 memop 플래그를 올리면, 3090이 `cuStreamWaitValue32`로 기다렸다가 매핑으로 읽습니다. 3090에는 약 9–10 GB가 남습니다[유도: 24,176 MiB − 512 MiB − 14 GB]. 다만 3090은 게이트 카드입니다. V4.1 GPU 게이트(`plan_gate`)가 카드 전체를 예산으로 잡으므로 상주 드래프트와 정면으로 충돌합니다.
- **Q5 품질**: 품질 표에 있습니다. 적재 경로에서 텐서 바이트를 두 번 읽는 곳은 없습니다. 페이지 캐시에 남는 카드 바이트(셋째 성능 행)가 그와 가장 가까운 문제입니다.

## 5. 읽지 못한 것

- `crates/gguf/src/lib.rs`는 파서 본문(~600–1000줄)을 훑기만 했습니다.
- `crates/gguf/tests/oracle.rs`와 `crates/model/tests/placement.rs`는 윤곽과 핀만 봤습니다.
- `hybrid.rs`는 620–940줄(go/wait 노드 빌더)을 건너뛰었습니다.
- `tools/ref/lease.sh`는 열지 않았습니다.
- `models/deepseek41.sh`는 트리아지 몫이라 따로 보지 않았습니다.
- 비행 중인 `cpu5`의 `attn.rs`와 `b5prof`의 `nsys-ds41.sh`에는 손대지 않았습니다.
- advisor 호출은 속도 제한으로 실패해 검토 없이 냈습니다.

## 6. 모델

opus로 스폰됐고, Opus 5.5로 돌았습니다.
