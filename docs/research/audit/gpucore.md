# audit-gpucore 보고 (GC)

범위는 `crates/gpu/src`의 GPU 코어다. 결론부터: `GpuModel`은 사실상 이미 "카드 하나 + 호스트 티어"인데 타입은 여러 스테이지를 전제한다(GC1). 판정이 끝난 계기 팔이 코드와 게이트 시간을 계속 쓴다(3장). 순서는 삭제 → move 등급 구조 변경 → 사용자 결정이 필요한 GC8이다.

라운드 정의가 요구한 두 항목:
- **종이 분해**: 읽기 전용 감사라 박스 실행과 예측 카드가 없다. 숫자는 전부 이미 있는 실측이다(rig-log, `tools/flow/constants.tsv`, `~/.cache/bloomery/gate-times.tsv`). 여기에 이번에 돌린 grep과 `wc -l`을 더했고, 계산한 값에는 [derived]를 붙였다.
- **자원 시간선**: V4.1 프롬프트 층-배치 하나의 시간선(호출 스레드, 호스트 작업자, 카드)을 GC3에 그렸다. GC8의 값도 그 시간선으로 매겼다. 나머지는 호스트 쪽 구조 변경이라 커널이 움직이지 않는다. 증명은 `just ptx-scan` 동일이다.

## 0. Scope read

**커밋**
- 4786c1e에서 읽기 시작했고, 읽는 동안 main이 fa61362로 움직였다.
- `git diff --stat 4786c1e HEAD`로 보면 범위 안 변경은 031b842 하나다.
  - elem.rs +107: `half_encode_check` 커널과 `enqueue_half_encode_check`
  - flash.rs +24: `f32x2_to_f16x2_bits`
- elem.rs와 flash.rs의 줄 번호는 fa61362에서 다시 grep했다. 나머지 인용 파일은 두 커밋에서 같다.
- plan-triage.md는 254행 뒤만 바뀌어 인용 줄은 그대로다.

**크기** [measured, wc -l]
- crates/gpu/src는 51파일 40,640줄이다(4786c1e).
- 범위 밖 파일을 빼면 범위는 32,226줄이다 [derived]. 뺀 것은 gemm.rs 2,415, flash_gqa_prefill.rs 623, arch/qwen3moe 5,376이다.
- fa61362에서 elem.rs는 1,517줄, flash.rs는 3,674줄이다.

**읽은 정도**

| 정도 | 파일 [measured 줄 수] |
|---|---|
| 전부 | model.rs 1,368, model/launcher.rs 615, model/kernels.rs 737, model/probe.rs 414, model/lookup.rs 56, graph.rs 600, hybrid.rs 2,414 [03], fault.rs 522 [03], head.rs 332, tensor.rs 316, weights.rs 745, arch/mod.rs 15, arch/deepseek2/mod.rs 676, arch/deepseek2/taps.rs 530 |
| 부분 | lib.rs 3,234(1–1120과 2140–3234는 본문, 가운데 커널은 fn 목록), deepseek2/dispatch.rs 1,544(1–560과 1240–1420은 본문, 560–1240은 fn·`probe_cfg`·`tick` 목록), flash.rs(1–120 본문, 커널·enqueue·`TWICE` grep) |
| 머리와 fn 목록만 | q8f32 1,835, q5 1,426, elem 1,517(argmax 부분만 본문), cores 1,370, q4k_sel 1,285, iq 1,259, fused 911 [03], flash_gqa 882, deepseek2/scratch 752(머리, probe 필드, `Drop`), router 528, join_probe 483, moe_fused 394, q6k_sel 333, rope_neox 332, rope_table 329, mxfp4 283, probe.rs 233, deepseek2/pins 185, route_core 98, deepseek2/seed 55 |

**안 본 것**
- deepseek2/names.rs(73) 본문.
- "머리만" 파일들의 디바이스 커널 본문. 커널의 수치 정확성은 이 감사가 판단하지 않았다.

**quant.rs**: 스펙은 [03] 파일로 적었지만 crates/gpu/src에 없다(`ls`로 확인).

**근거용으로 읽은 범위 밖 자료**
- gpu-deepseek41: body.rs, body/prefill.rs, chain/ffn/batch.rs(교환부 ~1300–1415), draft/mod.rs 머리
- gpu-gates: engine.rs, bind.rs 일부, 게이트 호출부
- docs: arch-split.md, gpu-design.md, v41-placement.md, plan-triage.md
- rig-log: 09-22-b, 09-22-n, 09-25(launch-thread, load3), 09-26(cardtile, prefillgroup)
- 참조 엔진: ik `ggml-cuda.cu`·`ggml-backend.cpp`, mistral.rs `cuda_graph.rs`·`device_map`·`loaders`, 박스의 exllamav3 `exl3_kernel_map.cu`(읽기 전용 ssh)

## 1. What we know now

1. **여러 스테이지는 코드상 한 번도 두 카드가 아니었다.**
   - `load_staged(cuts)`를 부르는 곳은 `load`(model.rs:381, `&[]`) 하나다.
   - `load`의 호출자는 없다: `grep -rnE '(GpuModel(::<[^>]*>)?|Deepseek2Model|Deepseek41Model|Qwen3moeModel)::load\(' --include='*.rs' crates` → 빈 결과(exit 1).
   - `load_staged`가 만드는 스테이지는 전부 `Gpu::new()`이고, 이것은 `with_device(0)`이다(lib.rs:2246-2248).
2. **배치 계획은 카드마다 층 범위와 헤드 여부를 주지만, 스텝 경로는 0층 시작만 받는다.**
   - 계획: `placement.rs:163-185` `Card { layers, head }`.
   - `load_placed`는 둘을 받는다(model.rs:585, 592-596).
   - 그러나 `step`, `step_pair`, `run_rows`는 `layers.start != 0`을 거부한다(model.rs:833, 926, 1294). 0층에서 시작하지 않는 카드는 적재는 되고 스텝은 못 한다.
3. **3090을 쓰는 데 스테이지 절단은 필수가 아니다.**
   - (b′), 즉 3090을 expert 전용 층으로 쓰는 안은 (b)보다 R1에서 3 % 낮다.
   - 지금 배치 코드는 "카드의 `layers`가 곧 스테이지"라 (b′)를 표현하지 못한다(docs/v41-placement.md:19).
   - 카드 사이 P2P는 없다(GNS, docs/gpu-design.md:37).
4. **DSpark 드래프트는 이미 `GpuModel` 밖에 있다.**
   - `DraftBody`가 자기 그래프를 가진다(gpu-deepseek41/src/draft/mod.rs:111-125).
   - `Gpu::for_card`로 열린다(gpu-gates/src/bin/shared/ds41_dspark.rs:110).
   - 검증 패스는 두 행 패스 위에 있다(`MAX_ROWS=2`, draft/mod.rs:44-50).
5. **`ChainBody`는 11메서드에서 20메서드로 컸다.**
   - 설계 스케치는 11메서드였다(docs/arch-split.md:89-143, 별도로 `Engine` 6).
   - 지금은 model.rs:55-246에 20메서드다(grep).
   - 기본값이 거부인 7개는 각각 구현자가 하나뿐이다(grep: deepseek2/mod.rs:394,414; gpu-deepseek41/src/body.rs:2048-2081).
   - 같은 문서가 `Arch` 트레이트를 거절한 이유는 "아키텍처 둘로는 무엇이 정말 공통인지 모르고, 지어낸 추상은 셋째 모델이 올 때 틀린 자리에 있다"였다(arch-split.md:144-145). 셋째 모델은 이미 왔다.
6. **그래프 소유자가 다섯이다.**
   - `GpuModel`의 `step_graph`·`pair_graph`(model.rs:335-375)
   - `Stage::graph`(model.rs:273-286, deepseek2/taps.rs만 씀)
   - `Head::graph`(head.rs:78, 게이트만 씀)
   - Qwen3 `prefill.graphs`(qwen3moe/prefill.rs:596-599)
   - `DraftBody`의 그래프
7. **카드와 호스트 사이 교환 프로토콜이 둘이다.**
   - 스텝용: 매핑된 호스트 워드와 스트림 memop(hybrid.rs:704-1104, 1970-2218).
   - 프롬프트 배치용: page-locked 세트, 이벤트, 상태 기계. 이것은 gpu-deepseek41 chain/ffn/batch.rs(~1300-1415)에 있다.
   - 둘 다 `Hybrid<H>`에 매달린다(`serve_batch`, hybrid.rs:1796-1943).
8. **V4.1 프롬프트 층-배치는 호스트에 묶여 있다.**
   - 조건: A6000, lcg 프롬프트, P 4096, G 2.
   - 층-배치당 union 68.6–69.3 ms, enqueue 4.3–4.4 ms, card_out 21.9 ms, card_in 11.7 ms, wait 0.83 ms다.
   - 벽 약 74 ms 가운데 union이 69 ms다(rig-log 2026-09-26#prefillgroup-ab, log/2026-09-26.md:330,332).
9. **`run_rows`를 쓰는 두 본체가 fault word를 다른 자리에서 읽는다.**
   - V4.1은 그룹 끝마다 읽는다(gpu-deepseek41 body/prefill.rs:1205).
   - Qwen3는 마지막 우배치의 헤드 판독에 맡긴다(qwen3moe/prefill.rs:614-625, model.rs:1309 `head.token`).
   - 그래서 Qwen3에서 N개 중 첫 우배치의 폴트는 N번째 우배치 뒤에 보고된다 [derived, 코드 독해].
10. **`GpuError::Shape`가 만능 변형이 됐다.**
    - 생성 587곳이다: `shape(` 361, `Shape {` 226(gpu, gpu-deepseek41, gpu-gates의 src를 grep).
    - 호출자가 변형으로 가르는 것은 `Fault`와 `Poisoned`뿐이다(grep: prefill.rs:313,1800,1803; model.rs:897; gate_deepseek41_prefill.rs:1363,1392,1503; gate_deepseek41_step.rs:453).
    - 게이트 7곳이 오류 문구를 `to_string().contains`로 대조한다.
11. **Q3_K split-K는 설계상 판정할 수 없는데 게이트 시간은 매 묶음 쓴다.**
    - 기본값이 꺼져 있다(`Q3K_SPLIT_WIDTH = 1`, lib.rs:207-217).
    - 주석은 예측 이득이 "자의 해상도 아래"라고 적는다(lib.rs:212-215).
    - 임대는 자 안의 A/B를 거부한다(AGENTS "Derive first"). 그래서 판정할 길이 설계상 막혀 있다.
    - 그런데도 이 커널을 도는 레시피 `gate-gpu-ds41-prefill-q3ksplit`(justfile:930-936)이 3090 레인에서 매 묶음 337–357 s를 쓴다 [measured, gate-times.tsv 마지막 네 행].
12. **`BLOOMERY_LAUNCH_THREAD`의 판정은 보류 상태다.**
    - 레버 on은 +0.96 % ± 2.1 %, on에 PIN_MAIN=0을 더하면 −1.2 % ± 3.4 %다(n = 3, rig-log 2026-09-25#launch-thread-lever-ab).
    - 스텝 사이 유휴의 큰 몫은 1,169노드 `cuGraphLaunch` 자체(0.32–0.43 ms)다(#load3-engram-after-launch-ab, log/2026-09-25.md:82).
    - hoststream 카드가 이 레버를 전제로 적는다(plan-triage.md:43).
13. **V4.1 게이트 픽스처는 작은 모델이 아니다.**
    - 0b95904의 픽스처는 실모양 9층이고, 대상이 60.9 GB다 [derived, 커밋 메시지].
    - 전부 카드에 올릴 수 있는 것은 606 MB짜리 부분 집합이다(같은 커밋).
14. **참조 엔진의 해당 설계**
    - mistral.rs는 그래프를 모양 키로 캐시한다(`pipeline/cuda_graph.rs:922-932` `CudaDecodeGraphKey`, 버킷 `:394-460`).
    - mistral.rs에서는 모델이 자기 층 바이트와 활성 크기를 매퍼에 신고한다(`pipeline/loaders/mod.rs:792` `DeviceMappedModelLoader`).
    - exllamav3는 모양 선택기 하나로 커널을 고른다(`exllamav3_ext/quant/exl3_kernel_map.cu:23` `select_gemm_shape`, 표는 `:132`).
    - ik는 8열에서 mmvq와 mmq를 가른다(`ggml/src/ggml-cuda.cu:2698-2721`, `mmvq.cuh:10`).
    - ik는 분할 입력을 배치 크기와 무관한 이벤트 한 벌로 넘긴다(`ggml/src/ggml-backend.cpp:2020-2043`).

## 2. Structural findings

순위는 증명 등급 순이다. move 등급이 먼저 오고, 사용자 결정이 필요한 것(GC7 3단계, GC8)은 마지막이다.

### GC1 — `GpuModel`은 이미 "카드 하나 + 호스트 티어"다. `Vec<Stage>`와 `Option<Residency>`를 걷고 그래프 소유를 하나로

**위치**
- model.rs:273-286(`Stage`), 335-375(필드), 378-432(`load`, `load_staged`), 1164-1265(`block0_parts`, `layer_slot`, `capture_stage`, `launch_graph`, `stage_stream`)
- 거부 줄 26개: `grep -nE 'no stage|carries no residency|stages\.len\(\) != 1|single whole-model stage' crates/gpu/src/model.rs` → 767…1308, `-c`로 26

**지금 모양**
- 스테이지마다 자기 `Gpu`, 층 범위, 그래프, `Option<Residency>`를 가진다.
- 스텝 경로마다 "스테이지 하나, 0층부터, 상주함"을 런타임에 다시 확인한다.
- 층 캡처(`capture_stage`, `launch_graph`)는 deepseek2/taps.rs(303, 315, 343, 351, 360)만 부른다.

**쌓인 경위**
- 60d8c9f(09-21): 결정 7 "`GpuModel` = `Vec<Stage>`, `load_staged(cuts)`". 약속한 "2스테이지 = 1스테이지 비트 동일" 게이트는 지어지지 않았다(gpu-design.md:37, :95).
- 715118b(09-23, V4.1 적재 이음): 카드 하나를 원소 하나짜리 벡터로 감쌌다.
- 3464bdf(09-24): `step_graph` 옆에 `pair_graph`를 붙였다.

**오늘 짓는다면**
- 타입: `GpuModel<B> { gpu, weights, body: B, head: Head, host: Option<HostTier>, graphs: Graphs<Chain>, mode, pos, poisoned }`.
- 생성자는 늘 상주한 모델을 돌려준다. 그러면 `Option<Residency>`와 "no stage"는 표현할 수 없게 된다.
- 둘째 카드는 모델 타입 밖에 둔다.
  - 드래프트: 둘째 `GpuModel`(지금도 그렇다).
  - (b′)의 3090: 호스트 티어 뒤의 expert 서버(GC3의 포트 하나).
- 그래프는 체인 모양을 키로 한 캐시 하나가 갖는다. mistral.rs `CudaDecodeGraphKey`(cuda_graph.rs:922-932)가 같은 형태다.
- 층 캡처는 deepseek2 계기로 옮기거나 GC7 1단계에서 지운다.

**이득**
- 없어지는 것: 거부 줄 26 [measured], `load`와 `load_staged` 55줄 [derived, 줄 번호], taps만 부르는 배관 102줄 [derived, 1164-1265].
- "상주 안 한 모델"과 "0층이 아닌 스테이지"라는 거부 부류가 타입으로 닫힌다.
- A2-2/N4(plan-triage.md:29, :107)의 설계가 맞는 모양에서 출발한다. 3090 DSpark 걸림돌 중 "카드 하나"(plan-triage.md:127, `body.rs:84-92`)도 이 모양을 전제로 한다.

**비용과 증명**
- move 등급이다. `just ptx-scan` 동일에 구조 줄 셋(그래프 노드 수 핀, eager = replay, e2e 집합 동일)이 더해진다.
- 게이트: gate-gpu-e2e, -ds41-step, -ds41-prefill, -qwen3moe-e2e, -p8, -p8b.
- 시간 A/B는 없다.

**위험·소유자·순서**
- 소유자는 aa다.
- faultstep 착륙(같은 파일의 `note_fault`, model.rs:896) 뒤, A2-2/N4 설계 라운드 앞에 둔다.
- GC2와 같은 파일이라 한 라운드로 묶거나 GC1을 먼저 한다.

### GC2 — `ChainBody` 20메서드 중 7개는 "한 본체만 구현, 나머지 거부"다. 핵심 트레이트를 작게 줄이고 능력별 트레이트와 모델별 생성자로 나누며, 엔진 표면을 하나로

**위치**
- 거부 기본값 일곱: model.rs:133 `hybrid_weights`, 149 `load_hybrid`, 169 `load_placed`, 194 `decode_pair`, 210 `enqueue_pair`, 231 `serve_replay_pair`, 240 `rollback`.
- 거꾸로 V4.1은 핵심 메서드 셋을 거부한다: gpu-deepseek41/src/body.rs:1893 `load`, 1999 `seed_depth`, 2007 `set_probe`. Qwen3는 기본값이 아닌 `set_probe`를 거부한다(qwen3moe/body.rs:430-438).
- 엔진 표면이 셋이다: `gpu::Engine`(model.rs:252-259), `serve::Engine`(crates/serve/src/engine.rs:97), `AnyEngine`(gpu-gates/src/engine.rs, 113줄).
- 제품 코드가 게이트 크레이트에 있다: `Ds41Engine`(gpu-gates/src/bind.rs, 599줄), DSpark 드라이버(bin/shared/ds41_dspark.rs, 296줄), 서버 바이너리 `bloomery_serve_ds41` [measured wc -l].

**쌓인 경위**
- `git log -L '/^pub trait ChainBody/,+200:crates/gpu/src/model.rs'`로 보면 커밋 일곱이다: 0e12f66 → c179548 → 4e4e643(hybrid) → 715118b(placed) → 3464bdf(pair) → 8c07db4 → 823aac8(`take_host_refusal`).
- 라운드마다 메서드 하나를 거부 기본값과 함께 얹었다.

**오늘 짓는다면**
- 핵심 트레이트: `arch`, `decode_input`, `refresh`, `enqueue_chain`, `reset`, `head_eps`, `resident_bytes`.
- 생성자: 모델별 고유 생성자(V2-Lite `open_hybrid`, V4.1 `open_placed`, Qwen3 `open`)가 `GpuModel<Self>`를 돌려준다.
- 선택 능력은 작은 트레이트로 나눈다: `HostServed`, `Rows`(pair), `Rollback`, `Instrumented`(`seed_depth`, `set_probe`; V2-Lite 전용).
- `step_pair`는 `where B: Rows`일 때만 존재한다. 런타임 거부가 컴파일 오류로 바뀐다.
- fault word의 판독 지점은 `run_rows`의 계약 하나로 둔다(1장 9번).
- thread-local `REPLAY`(hybrid.rs:1373-1403, 075ceef)는 인자로 넘긴다([03]).
- 엔진 표면은 엔진 크레이트의 `serve::Engine` 하나로 모은다. 서빙 바인딩과 DSpark 드라이버도 그 크레이트로 옮긴다. mistral.rs는 투기 디코딩을 엔진 모듈로 둔다(`mistralrs-core/src/speculative/`).

**이득**
- 거부 기본값 7과 거부 재정의 3이 없어진다.
- 제품 코드 895줄 [measured 합]이 게이트 크레이트 밖으로 나간다.
- 프롬프트 폴트 판독 계약이 하나가 된다.
- DSpark w+1 검증을 위해 pair를 `Rows(n)`로 넓히는 일은 커널 작업이라 이 GC에 넣지 않는다. 다만 그 라운드가 붙을 자리가 여기서 생긴다.

**비용과 증명**
- 분할과 이동은 move 등급이다: `ptx-scan` 동일 + 구조 줄. 게이트는 gate-ds41-serve, -ds41-draft, -dsloop, -e2e다.
- 폴트 판독 통일은 결정 하나가 필요하다.
  - Qwen3에 우배치마다 판독을 넣으면 우배치마다 호스트와 카드가 합류하게 된다. 흐름 변경이라 A/B가 든다.
  - V4.1을 호출 끝 판독으로 옮기면 gfix(80e9c9c)가 세운 "폴트는 실패한 호출을 넘지 않는다"를 다시 따져야 한다.

**위험·소유자·순서**
- 소유자는 aa다. `REPLAY`와 qwen3moe prefill은 [03]이다.
- GC1 뒤에 둔다.

### GC3 [03]+aa — 교환 프로토콜 두 벌이 두 소유자에게 나뉘어 있다. `HostTier` 하나 아래 `StepPort`와 `BatchPort`

**위치**
- [03] hybrid.rs(2,414 [measured]):
  - 플래그 프로토콜: `Boundary`/`RowPage`/`go_batch`/`release`(704-1104), `serve`/`serve_one`(1970-2154), `wait_go`(2187-2218)
  - 프로토콜 밖 입구: `serve_batch`(1796-1943)
  - batch 통계가 섞인 `HybridStats`(1148-1210)
- aa: 배치 상태 기계는 gpu-deepseek41 chain/ffn/batch.rs(~1300-1415)에 있다. 흐름은 D2H → `set.routed.record` → `Stage::{Free,Routed,Served}` → 이중 버퍼 → `synchronize` → `hybrid.serve_batch(…, &[], sum)`(1372) → H2D다.
- 한 사실에 소유자가 둘인 곳이 셋이다.

  | 사실 | 소유자 1 | 소유자 2 |
  |---|---|---|
  | 폴트 판독 | `Gpu::fault`(lib.rs:2317-2355) | `read_fault`(hybrid.rs:2329-2358, 동기 `cuMemcpyDtoH`) |
  | 매핑 메모리 | `MappedHost`(hybrid.rs:464-594) | `HostFlags`(graph.rs:328-464) |
  | 캡처 중 여부 | `capturing`(hybrid.rs:678-685) | `capturing`(graph.rs:301) |

**쌓인 경위**
- 4e4e643(09-23): 스텝 프로토콜.
- 43cd107(09-25): `serve_batch`.
- c58cb37: 제외 집합.
- a9fa916: `Hybrid::reset` 의미 변경.
- 배치는 플래그 프로토콜을 늘인 것이 아니라 옆에 둘째 프로토콜을 세웠다. 그런데 그 상태 기계는 V4.1 크레이트에 있고, 입구·통계·poison은 `Hybrid<H>`에 있다.

**스코프 질문("스텝용 프로토콜을 배치로 늘인 것을 오늘도 그렇게 짓겠나")의 답**
- 메커니즘 둘은 오늘도 둔다. 캡처된 그래프 안 호스트 서비스에는 memop이 필요하고, eager 배치에는 이벤트가 맞다.
- ik에는 그래프 안 호스트 서비스가 없다. 스케줄러가 이벤트 한 벌로 처리한다(ggml-backend.cpp:2020-2043). 그래서 `StepPort`는 ik 대응물이 없는 우리 고유의 것이다.
- 바꿀 것은 소유다.
  - `HostTier` 하나(슬롯 맵, `HostExperts`, poison과 거부 명명, 통계) 아래 `StepPort`와 `BatchPort`를 둔다.
  - 폴트 판독, 매핑 할당기, `capturing`은 하나씩만 둔다.
  - (b′)의 3090 expert 서버는 셋째 포트다.

**자원 시간선**(lcg, P 4096, G 2, 층-배치당, rig-log 09-26#prefillgroup-ab)

| 자원 | 하는 일 | ms |
|---|---|---:|
| 호출 스레드 | enqueue, 그다음 union 서비스 | 4.3–4.4 |
| 호스트 작업자 | union | 68.6–69.3 |
| 카드 | card_out + card_in, union 아래에서 | 21.9 + 11.7 = 33.6 [derived] |
| 호스트 대기 | wait | 0.83 |

- 벽은 약 74 ms ≈ 69 + 4.4 + 0.8 [derived]로, 호스트 합이다. 카드는 그늘 아래라 lcg에서 값이 0이다.
- union 밖의 호스트 항은 enqueue 하나다. 발행 비용 t_issue는 2.42 µs/항목이다 [derived, constants.tsv:60]. 카드 쪽 틈 gap_act 0.898 µs [measured, constants.tsv:106]보다 크다. 즉 발행은 호출 스레드의 비용이다.

**이 GC가 여는 흐름 레버**(가설이며 이 GC의 본체가 아니다)
- 벌크: 층-배치 route를 그래프로 잡는다. 1,169노드 런치가 0.32–0.43 ms라는 실측(log/2026-09-25.md:82)에 비추면 최대 −4.3 ms/층-배치, 벽의 약 5.8 %다 [derived]. 걸림돌은 CED 폭이 층마다 달라 그래프 키가 많아진다는 점이다.
- 비동기: 발행을 SMT 형제 스레드로 옮겨 union 아래로 넣는다. 배치 코드는 launcher.rs:255-310에 이미 있다. 걸림돌은 union 작업자와의 코어 간섭이다. launch-thread A/B가 3090에서 같은 가설을 적었다.
- 두 레버 모두 hoststream과 h3tile-b가 union을 바꾼 뒤에 값을 다시 매긴다.

**이득**
- 중복 셋이 각각 하나로 준다.
- 배치 상태 기계가 프로토콜 소유자 곁으로 온다.
- (b′)와 DSpark(`MappedHost`에 `PORTABLE` 없음, plan-triage.md:127)가 붙을 자리가 생긴다.

**비용과 증명**
- 소유 이동은 move 등급이다: `ptx-scan` 동일 + 구조 줄. gate-gpu-hybrid, -ds41-prefill(`CARD_EXPERTS` 세 팔), -ds41-step, -faults가 비트 그대로여야 한다.
- 흐름 레버는 launch-count 등급이라 카드와 A/B가 따로 든다.

**위험·소유자·순서**
- 소유자: [03](hybrid.rs), aa(ffn/batch.rs, body/prefill.rs).
- r8host(`HostSet`, 비행 중) 뒤에 둔다.

### GC4 — `GpuError`가 호출자가 가를 정보를 버린다. `Shape`가 만능이고 복합 실패는 문자열이 된다

**위치**
- lib.rs:80-142(변형 14개). 도우미(144-196)는 `pub(crate)`라서 다른 크레이트는 리터럴로 만든다.
- 범위 안의 부정직한 자리:
  - launcher.rs:263-306: OS의 친화성 거부를 `Shape`로 낸다.
  - launcher.rs:151, 154, 224: 스레드 실패를 `State`로 낸다.
- 스트림 대기는 범위 안에서 정직하다. `synchronize()?`는 `From`으로 `Driver`가 된다(lib.rs:488-492, 2353, 3168; hybrid.rs:1645).
- 스코프가 물은 "실패한 스트림 대기에 Shape"는 범위 밖에 있다.
  - gpu-deepseek41 body/prefill.rs:1804-1811 `fault_or`와 :324-338 `take_back`이 원래 `GpuError`를 `{e}`로 문자열화해 `Shape`에 담는다.
  - 둘 다 `Fault`는 형태를 지킨다(:313, :1800-1803). 그러나 드라이버 실패는 입력 거부와 같은 변형으로 떨어진다.

**쌓인 경위**
- R9와 R11(docs/rust-quality.md:36, 38): "변형은 호출자가 다르게 처리할 만큼만".
- 실제로 가르는 것은 `Fault`와 `Poisoned`뿐이었고, 그 사이 `Shape`가 "나머지 전부"가 됐다.

**오늘 짓는다면**
- R11대로 가르되, 가르는 축을 셋으로 둔다.
  - 이 호출이 거부됐다.
  - 모델이 오염됐다.
  - 카드나 컨텍스트가 사라졌다.
- 셋째는 이 저장소에서 실제로 일어났다(Xid 79 두 번, AGENTS "Never"). 서버는 셋째 경우에만 재시작해야 한다.
- 변형: `Refused`, `Driver`, `Os { source: io::Error }`, `Fault`, `Poisoned`, `Recovery { first: Box<GpuError>, then: Box<GpuError> }`. 도우미는 `pub`로 연다.
- 참조 엔진에 본보기는 없다. mistral.rs도 `Error::msg`·`Msg` 류가 1,213곳, `.context(`가 286곳이다 [measured grep, mistralrs-core/src].

**이득**
- "카드 소실"이 "입력 거부"로 가려지는 부류가 닫힌다.
- `{e}`를 박은 `Shape` 10곳의 원인이 타입으로 남는다.

**비용과 증명**
- 호스트 전용이다(`ptx-scan` 동일).
- 문구를 대조하는 게이트 7곳은 문구를 지키거나 변형 대조로 바꾼다: gpu-gates ptx.rs, gate_gemm.rs, gate_hybrid.rs, gpu-deepseek41 params.rs, gpu fault.rs, deepseek2/pins.rs. 찾은 grep은 `grep -rnE 'to_string\(\)\.contains\('`다.
- 시간 A/B는 없다.

**위험·소유자·순서**
- 소유자는 aa, fault.rs 부분은 [03]이다.
- 독립적이다. 범위 밖 `fault_or`·`take_back`과 한 라운드로 묶는다.

### GC5 — 카드 바이트의 소유자가 둘이다. 계획 산술과 실제 할당이 적재 게이트의 잔차로만 맞춰진다

**위치**
- 계획 쪽: `crates/model/src/placement/workstation.rs`(CONTEXT 512 MiB, `SCRATCH 64 MiB [assumed]`, MARGIN 1 GiB, GRANULE 2 MiB).
- 적재 입구가 넷이다: weights.rs:151 `load`(층 범위), :188 `load_placed`(계획), :216 `load_rows`(손으로 만든 행), :231 `load_where`(술어).
- 적재 레버 `HostLevers`(`BLOOMERY_HOST_POPULATE`, `_LOCK`, `BLOOMERY_CARD_DONTNEED`)와 `HostResidency`가 호스트 티어 모듈에 있다(hybrid.rs:174-273).
- V2-Lite의 행 분류는 GPU 크레이트 안의 `role_of`다(deepseek2/mod.rs:570-588).
  - catch-all이 둘이다: `blk.`이 아니면 `Head`, 나머지는 전부 `Attention`.
  - 0층 dense FFN을 `Role::SharedExpert`로 적는다. `Role::DenseFfn`(placement.rs:49)이 있는데도 그렇다 [code reading].
  - model 크레이트의 분류기에는 catch-all이 없다(qwen3moe/roles.rs:3, deepseek41/roles.rs:6).
- 계획 밖의 카드 할당:
  - V4.1 배치 버퍼: 첫 프롬프트 때 잡는다(gpu-deepseek41 body/prefill.rs:985-988). G 2에서 238 MB를 쓴다(`vram_free_load` 1.020 → 0.783 GB, rig-log 09-26#prefillgroup-ab).
  - 폴트 워드 2 MiB(lib.rs:2226-2234).
  - pinned 매핑 2 MiB와 `join_rows`의 순간 ~20 MB(plan-triage.md:127).
- 계획과 실제는 적재 게이트의 잔차로만 대조된다(gate_deepseek41_load.rs:127).

**쌓인 경위**
- `load_rows`와 `role_of`는 4e4e643(V2-Lite 하이브리드)에서 생겼다.
- `load_placed`는 715118b에서 생겼다.
- 배치 버퍼는 43cd107과 eb3e08a(G)에서 커졌다.

**오늘 짓는다면**
- 모든 적재가 한 길을 지난다: classify(model 크레이트, catch-all 없음) → `Plan` → `load_placed`.
- 본체가 자기 카드 항을 계획에 신고한다: 가중치, KV, 스크래치, G별 배치 버퍼, 폴트 워드.
- mistral.rs `DeviceMappedModelLoader`(pipeline/loaders/mod.rs:792)가 이 모양이다(`layer_sizes_in_bytes`, `non_mapped_size_in_bytes`, `mapped_max_act_size_elems`).
- 적재 레버는 `Residency` 모듈로 옮긴다.

**이득**
- 적재 입구 넷이 하나로 준다.
- `[assumed]` 값이 신고값이 된다.
- "적재는 됐는데 첫 프롬프트에서 카드가 모자람" 부류가 적재 때의 이름 붙은 오류로 닫힌다. G 상한은 8인데, G 8이 들어가는지는 재지 않았다(가설).

**비용과 증명**
- 호스트 전용이고 적재 때만 움직인다(`ptx-scan` 동일).
- 게이트: gate-gpu-load-v41, -load-v41-lock, placement 게이트, gate-gpu-hybrid. 같은 계획이면 같은 슬롯이 나와야 한다.
- 시간 A/B는 없다.

**위험·소유자·순서**
- 소유자는 aa, `HostLevers`는 [03]이다.
- r8host(`Ds41Host::build`) 뒤, GC3와 같은 파동에 둔다.

### GC6 — 가중치 형식을 적재 때 타입으로 굳히지 않아서 enqueue마다 다시 검사한다. 모양 선택도 호출자에 흩어져 있다

**위치**
- `DevWeight::KQuant { ty: GgmlType, w: DeviceTensor<u32>, k }`(weights.rs:32-44) 하나가 Q3_K, Q4_K, Q6_K를 다 담는다.
- `kq_weight`(model/lookup.rs:30)는 `&DeviceTensor<u32>`만 돌려주며 형식을 지운다.
- 받는 쪽마다 행 바이트를 다시 계산한다 [measured, `grep -rnE '110 ?\* ?n_sb|n_sb ?\* ?110'` 등].

  | 형식 | 범위 안 | gpu + gpu-deepseek41 |
  |---|---:|---:|
  | Q3_K `110*n_sb` | 34 | 57 |
  | Q4_K `36*n_sb` | 21 | 30 |
  | Q6_K `210*n_sb` | 9 | 13 |

- `launch_u32(`가 589곳이다.
- 범위 안 모듈에 `enqueue_*`가 71개다 [derived 합, 파일별 grep]. 그중 gemv·quantize 변형이 29개다(`_mcol`, `_groups`, `_split`, `_sel`, `_grouped`, `_tiles`, `_heads`, `_heads_pair`, 양자화 입구 여섯은 lib.rs:2519-2717).
- 모양 선택이 흩어져 있다: `BLOOMERY_CARD_EXPERTS` 분기, `Q3K_SPLIT` 규칙(lib.rs:207-263), m ≤ 8 검사.
- 같은 이름의 `RouterKernels`가 셋이다: crates/gpu/src/router.rs, gpu-deepseek41/src/router.rs:614, qwen3moe/router.rs.

**쌓인 경위**
- 커널 코어는 m ≤ 8열에 `[f32; 8]` 누산기, 즉 디코드 모양이다. P0–P10, a5gemm, ds41bulk, B1, T가 그 위에 변형과 입구를 하나씩 얹었다.
- `Q8Act`는 소비자가 한 순열만 읽어도 순열 셋과 s8, d8을 다 쓴다(tensor.rs:218-226, lib.rs:518-691). `GemmAct`에도 같은 항목이 트리아지에 있다(plan-triage.md:221).

**오늘 짓는다면**
- 적재 때 `KqRows<F: KFormat>` 핸들로 한 번 해석한다. `F::BLOCK_BYTES`가 행 바이트를 아는 유일한 자리가 된다.
- enqueue는 `&KqRows<Q3k>`처럼 형식이 박힌 타입을 받는다.
- 가족마다 모양 선택기 하나를 둔다: `Shape { rows, cols: One | Few(m≤8) | Groups | Tiles }` → 커널과 격자.
- exllamav3 `select_gemm_shape`와 `[K][cb][shape_idx]` 표(exl3_kernel_map.cu:23, 132)가 이 형태다.
- 변형 커널은 남는다. 고르는 자리만 한 곳으로 모인다.

**이득**
- 행 바이트 식 64곳(범위 안)이 한 곳으로 모인다.
- 형식이 틀린 가중치를 넘기는 실수가 컴파일 오류가 된다.
- GC8의 GEMM 열을 넣을 자리가 생긴다.

**비용과 증명**
- 호스트 전용이다. `ptx-scan` 동일 + 노드 수, eager = replay, e2e 집합.
- 호출 지점이 수백 곳이라 두 라운드로 나눈다: 형식 핸들 → 선택기.
- 시간 A/B는 없다.

**위험·소유자·순서**
- 소유자는 aa, fused.rs의 입구는 [03]이다.
- GC7 1단계 뒤(지울 변형을 옮기지 않도록), GC8 결정 앞에 둔다.

### GC7 — V2-Lite: 코어는 빠른 게이트 모델로 남기고 붙은 계기를 걷는다. 은퇴는 대체 게이트가 생긴 뒤 사용자 결정

**판결: 코어는 남긴다.** V2-Lite만 주는 커버리지가 둘이다.
- 호스트 티어가 전부 카드 실행과 비트 동일함: gate-gpu-hybrid, 55–108 s [measured gate-times.tsv]. V4.1은 전부 카드에 올릴 수 없고, Qwen3에는 호스트 티어가 없다.
- ik CUDA greedy와의 토큰 대조(33프롬프트 × 32스텝): gate-gpu-e2e, 141–194 s(A6000 레인) [measured].

**비용은 대부분 계기에서 나온다**
- `StepProbe`(model/probe.rs, 414줄)
- taps.rs(530줄)
- dispatch.rs의 `probe_cfg` 분기 8곳(177, 484, 525, 586, 587, 660, 737, 770)과 `tick(` 43곳 [measured grep]
- 스칼라 flash 패스와 탐침 커널(flash.rs:1103-1760)
- V2-Lite 전용 줄: arch/deepseek2가 3,815 [derived 합]이다. V2-Lite만 부르는 모듈을 더하면 6,577 [derived]이다(router.rs 528, q5.rs 1,426, moe_fused.rs 394, model/probe.rs 414). flash.rs 커널 대부분도 여기에 속한다.
  - 근거: `.flash()`는 dispatch.rs에서 8번, gate_p0b에서 1번 불린다. `.router()`, `.q5()`, `.moe_fused()`도 deepseek2와 게이트뿐이다(접근자 grep).
- V2-Lite 게이트 바이너리 여덟이 합 7,231줄이다 [measured]: gate_hybrid 1,688, gate_p10 1,220, gate_p8 1,187, gate_e2e 1,028, gate_moe_fused 725, generate 566, gate_p8b 483, forced_probe 334.
- 공유 코드에 제약을 건다. `mma_segment_walk!`(flash.rs:434-785)을 V4.1 쪽에서 고치면 V2-Lite PTX 바이트 동일을 따로 확인해야 한다(plan-triage.md:220).

**쌓인 경위**: P0–P10의 디딤돌이었고, 단계마다 탐침이 엔진 안에 남았다. 머리 주석의 "P5", "P6", "(P8 prep)", "(package P0b …)"가 그 흔적이다.

**오늘 짓는다면 세 단계로 간다**
1. 판정 난 팔을 지운다(3장).
2. V2-Lite 전용 커널과 계기를 `arch/deepseek2` 아래로 모은다. flash.rs에서는 공유 도우미(`f32_to_f16_bits`, `f32x2_to_f16x2_bits`, `dev_exp`)와 `mma_segment_walk!`만 공용 모듈에 남긴다. 그러면 V2-Lite를 지우는 일이 디렉터리 하나의 결정이 된다.
3. 은퇴는 사용자 결정이다. 조건이 둘이다.
   - gate_hybrid의 계약을 V4.1 픽스처 부분 집합(606 MB, 전부 카드 가능) 위에서 돌릴 수 있어야 한다.
   - 작은 실모델로 ik와 토큰을 대조하는 게이트를 계속 둘지 정해야 한다.
   - 오늘은 둘 다 대응물이 없다.

**이득**
- 1단계: 3장의 줄과 게이트 시간이 빠진다. q3ksplit 레시피가 3090 레인에서 337–357 s, e2e의 둘째 프로세스가 A6000 레인에서 빠진다.
- 2단계: 공용 모듈이 V2-Lite에 묶이지 않는다.

**비용과 증명**
- 1단계는 삭제다. 커버리지 변경마다 날짜 사유를 붙인다.
- 2단계는 move 등급이다.
- 3단계는 커버리지 변경이다.

**소유자와 순서**: 소유자는 aa다. 1단계를 가장 먼저 하고, 2단계는 GC6 앞에 둔다.

### GC8 — 모델마다 프리필 계약이 다르다. 이 항목은 사용자의 프리필 계약 결정이 있어야 움직이고, 이 보고서의 다른 항목은 그 결정에 기대지 않는다

**위치와 모양**
- V4.1 배치 경로는 "프리필 = 스텝 비트" 계약 때문에 m ≤ 8 디코드 코어 위에 섰다. 그 결과가 `ColGroups`(lib.rs:265-366), `_groups`/`_split_groups`(1901, 1976), `_tiles`/`_grouped`(q4k_sel.rs:325, 549)다.
- Qwen3는 IMMA GEMM 우배치를 쓴다. 스텝과는 밴드로 판정하고, 비트는 우배치 크기와 무관하다(Qwen3 e2e 게이트 머리의 (p)(q)(u)(w)).
- V4.1의 hoststream on 팔은 이미 밴드 판정이다(사용자 09-25 밤, plan-triage.md:43).
- V4.1 프롬프트는 gemm.rs를 쓰지 않는다: `grep -rnE 'gemm::|GemmKernels|enqueue_gemm|GemmAct' crates/gpu-deepseek41` → 빈 결과.

**쌓인 경위**: 43cd107의 "비트 그대로" 계약 안에서 B1, T, G가 m ≤ 8 코어를 넓혔다. a5gemm(7fd6f2a)은 "Qwen3 먼저; ds41 프리필 게이트는 이 뒤 밴드로"라고 적었다(plan-triage.md:37).

**오늘 짓는다면**
- 모델 공통 계약 하나: "프롬프트는 스텝과 밴드, 비트는 배치·그룹·우배치 크기와 무관".
- 그 위에서 열 수로 gemv 코어와 GEMM을 가른다. ik가 이렇게 한다(`src1->ne[1] ≤ 8` → mmvq, 아니면 mmq; ggml-cuda.cu:2698-2721).
- GC6의 선택기가 그 분기의 자리다.

**이득(지금은 작다, 시간선 기준)**
- lcg에서는 카드 33.6 ms가 union 69 ms 그늘 아래라 0이다.
- 산문 P 4096(T 뒤, G 전, log/2026-09-26.md:291)에서는 카드 44.2 ms [derived: card_out 22.5 + card_in 21.7]가 union 39.1 ms보다 조금 길 뿐이다.
- IMMA 예측(prose ~3 ms 대 T 14.6 [11.5–17.8] ms/층-배치, 둘 다 [derived], cardnext-design-report.md:161)이 맞아도, 벽은 호스트 항까지만 준다. 산문 G 행은 아직 없다.
- 값은 hoststream과 h3tile-b가 호스트 항을 줄인 뒤에 생긴다.

**비용과 증명**
- float 합산 순서 등급이다: σ, gate-gpu-e2e 개수 핀, 확인 A/B 1회.
- V4.1 프리필 게이트를 비트에서 밴드로 다시 쓴다. FAIL-first와 날짜 사유가 필요하다.

**위험·소유자·순서**: 사용자 결정 뒤 aa와 [03](gemm.rs, q3gemmd 뒤)이 맡는다. GC6 선택기와 hoststream 뒤에 둔다.

## 3. Deletions

항목마다 근거와, 지웠을 때 달라지는 게이트 커버리지를 적는다.

1. **`GpuModel::load`·`load_staged`**(model.rs:378-432)
   - 근거: 1장 1번 grep이 빈 결과다.
   - 커버리지: 변화 없음.
2. **`argmax_rows` 커널**(elem.rs:787)과 **`enqueue_argmax_rows`**(1429)
   - 근거: `grep -rnE '\b(enqueue_)?argmax_rows\b' --include='*.rs' crates | grep -v '^crates/gpu/src/elem.rs'` → head.rs:169의 문서 주석 하나뿐이다. 헤드가 쓰는 것은 `_fault` 판이라 그 주석도 틀렸다.
   - 커버리지: 게이트 없음.
3. **비폴트 `argmax`·`enqueue_argmax`**(elem.rs:753, 1403)
   - 근거: 호출자가 gate_p4.rs(520, 523, 615, 644, 681)뿐이다.
   - 커버리지: gate_p4의 argmax 절을 `_fault` 판으로 옮긴다.
4. **`Head::graph`·`capture`·`launch`**(head.rs:78, 254, 263)
   - 근거: 호출자가 gate_head_gpu.rs:202-224와 gate_mcol.rs:764-766뿐이다.
   - 커버리지: 헤드 단독의 "eager = replay"와 "같은 그래프에 둘째 입력"(낡은 입력 FAIL-first) 핀이 없어진다. 같은 성질은 스텝 그래프의 eager = replay와 연속 토큰 재생(gate-gpu-e2e, -ds41-step)이 헤드까지 포함해 지킨다. 잃는 것은 결함 위치를 헤드로 좁히는 해상도다.
5. **스칼라 flash 세그먼트 패스와 탐침**
   - 대상: `flash_latent_seg`(1103), `seg_pass`(1166), `enqueue_flash_latent_seg`(2933), 탐침 커널 다섯(1423-1760), `flash_merge2_q8`(1947), `TWICE_*`(290-302)와 `latent_range`·`merge_row`의 `TWICE` 분기, `enqueue_flash_latent_seg_twice`(3105), `enqueue_flash_merge2_q8`(3318), `StepProbe`의 flash 팔과 `keyaxis_arms`(model/probe.rs:245-287), `BLOOMERY_FLASH_MMA=0`.
   - 지우지 않는 것: `flash_latent`·`flash_latent_q8`은 짧은 캐시의 기본 경로라 남는다(dispatch.rs:440, 451).
   - 근거: `log/2026-09-22-n-the-mma-pass-became-the-default.md:37` "스칼라 세그먼트 패스와 그 계기 8개는 이제 두 번째 경로다. 지울지는 다음 결정이다." 그리고 AGENTS "Performance first".
   - 커버리지: gate-gpu-e2e의 둘째 프로세스와 keyaxis 팔이 없어진다(justfile:112-116). MMA의 ik 대조는 남는다.
6. **`StepProbe`의 split 팔 넷**: `split_heads`, `split_kqvc`, `split_flash_quant`, `split_moe_quant`(dispatch.rs:177, 586-587, 660, 1298-1355)
   - 근거: `log/2026-09-22-b-the-barrier-got-a-price.md:32-40`. 접기가 −31.9, −35.9, 둘 다 하면 −69.3 µs/스텝으로 판정됐다(8바퀴, 노드 701 → 648). 이 팔들은 "value-neutral rollback lever"로 남았다(gate_e2e.rs:209-215: 648/702/729/755).
   - 커버리지: gate_e2e의 중첩 팔 검사, gate_p8.rs:342, generate.rs의 레버 37곳이 바뀐다.
7. **Q3_K split-K 가족**: `q3k_gemv_split`, `_split_mcol`, `_split_groups`(lib.rs:1698, 1770, 1976), `Q3K_SPLIT_*`(207-263), 레시피 `gate-gpu-ds41-prefill-q3ksplit`(justfile:930-936)
   - 근거: 설계상 판정할 수 없다(1장 11번).
   - 커버리지: 이 커널을 도는 유일한 게이트가 없어진다. 3090 레인에서 337–357 s [measured]가 빠진다.
8. **`q4k_gemv_grouped`**(q4k_sel.rs:325)와 **`enqueue_gemv_q4k_grouped`**(948), **`BLOOMERY_CARD_EXPERTS=expert|slot`**(gpu-deepseek41과 걸침)
   - 근거: rig-log 2026-09-26#cardtile-ab(tile/expert 1.245 ± 0.060과 1.167 ± 0.011, decide-in). 엔진 호출자는 expert 팔(ffn/batch.rs:2181)과 gate_q4k_sel뿐이다.
   - 커버리지: gate-gpu-ds41-prefill의 두 팔 실행이 없어진다.
9. **`BLOOMERY_HYBRID_OVERLAP=0`**
   - 근거: 이 팔을 도는 레시피와 게이트가 0이다(plan-triage.md:126 Q3).
10. **stage-0 도우미**: `gemv_q4k`, `probe_q4k_launch_us`, `check_q4k_geometry`(lib.rs:3153-3230), `Q8Act::new`(k=2048, tensor.rs:231)
    - 근거: gpu-spike와 gate-gpu-p0(justfile:70-71)만 쓴다.
    - 커버리지: gate-gpu-p0가 없어진다. 사용자 판단이 필요하다.
11. **`join_probe.rs`(483)와 probe.rs의 `gap_*`·`two_phase`**
    - 근거: bench_join, gate_p8 1곳, gpu-spike 4곳만 쓴다. 결과는 rig-log에 이미 있다.
12. **`BLOOMERY_LAUNCH_THREAD`: 판정하든지 지우든지 둘 중 하나다**
    - 판정: 예측 상한 1.1–1.7 %는 6바퀴(±0.8 %)면 자 밖이다. 13바퀴(±0.5 %)면 점추정 +0.96 %까지 가른다 [derived, AGENTS 자, 09-26 정정 값].
    - 유지 비용: launcher.rs 615줄 [measured], `REPLAY`/`ReplayWatch`(hybrid.rs:1373-1403), 친화성 셋째 사본(plan-triage.md:129).
    - 지우려면 hoststream 카드의 전제(plan-triage.md:43)부터 고쳐야 한다.
13. **plan-triage.md:127의 `Boundary`·`param_view` 누수 항목**
    - 근거: bfac033(09-25)이 이미 닫았다. hybrid.rs:2370 `hw_boundary_gives_back_its_context_handles`와 scratch.rs:242-255 `Drop`이 그 증거다.

## 4. Cross-cutting patterns

- **출구 없는 레버**: 판정 뒤에도 "롤백"으로 남는다(split 팔, 스칼라 flash, expert/slot, split-K). 레버에 "판정 → 삭제 날짜" 칸이 없다.
- **한 사실에 소유자 둘 이상**:
  - `capturing`, 매핑 메모리, 폴트 판독
  - per-head gemv(model/kernels.rs와 q8f32.rs:1470)
  - 친화성 세 벌, 그래프 소유자 다섯
  - 모델별 프리필 계약
  - 역할 분류(`role_of`와 `classify`)
  - 같은 이름의 `RouterKernels` 셋
  - 카드 바이트(계획과 할당)
- **엔진 타입 안의 계기**: `Head::graph`, `Stage::graph`, `StepProbe`, `tick` 바이트 회계, `step_layer_hybrid`·`hybrid_stats`·`hybrid_words`(deepseek2/mod.rs:611-676, gate_hybrid만 씀).
- **타입 대신 런타임 거부**: "no stage" 26줄, 거부 기본값 7개.
- **머리 주석에 남은 라운드 이름**: lib.rs:1 "stage-0", 2627-2629 "P8's business", flash.rs:1 "P5", router.rs "P6", moe_fused.rs "(P8 prep)", fused.rs "(package P0b …)" [03], weights.rs "(package P10)", model/kernels.rs "P1–P7", head.rs:73-77의 사건 서술. `tools/check-comments.sh`는 이슈 번호와 날짜만 잡고 라운드 이름은 못 잡는다.
- **게이트 크레이트 안의 제품 코드**: 서빙 바인딩, DSpark 드라이버, 서버 바이너리, `generate_ds41`.
- **여러 일을 하는 모듈**
  - lib.rs: 컨텍스트, 오류, 발사 도우미, split 규칙, 열 그룹, q8_1 디바이스 코드, K-quant 커널, enqueue, 폴트, stage-0.
  - flash.rs: V2-Lite 커널, 공유 변환, 공유 매크로.

## 5. Looks historical but stays

- **iq.rs**: 계획된 IQ 모델용이다(c55cf13, plan-triage.md:104).
- **graph.rs**: 캡처 catch_unwind, `cuGraphUpload`, Branch/Forked, HostFlags. `capturing`의 소유자는 이쪽으로 모은다.
- **fault.rs [03]**: AGENTS 계약 그대로다. 판독 중복만 GC3에서 다룬다.
- **`mma_segment_walk!`**: V4.1 attn.rs:258과 공유한다. 위치만 옮긴다.
- **`serve_batch`의 `exclude`**: c58cb37로 착륙한 hoststream API다(plan-triage.md:36).
- **`enqueue_gemv_q3k_sel`**: fused MoE 게이트의 참조 쌍둥이다.
- **단형 제네릭 `GpuModel<B>`**: 토큰 경로에 `dyn`이 없다.
- **`step(&[u32])`**: 배치 프리필의 비트 기준이다.
- **`Gpu::for_card`의 이름 선택**: 바꿀 것은 `Gpu::new`의 장치 0 쪽이다.
- **load3의 순서**: rig-log 09-25#load3-engram-after-launch-ab가 "구조 정정으로 코드로는 남긴다"고 적었다.

## 6. If we rebuilt this scope today

**모듈**

| 모듈 | 내용 |
|---|---|
| `gpu::ctx` | `Gpu` 하나 = 카드 하나. 스트림, 모듈 적재. 장치는 `for_card`로만 고른다 |
| `gpu::error` | 거부 / 오염 / 드라이버 / OS / 복구 실패. 도우미는 `pub` |
| `gpu::fault` | 폴트 판독의 유일한 소유자 |
| `gpu::graph` | `Graph`, `HostFlags`(매핑 메모리의 유일한 할당기), `capturing`, 모양 키 캐시 `Graphs<K>` |
| `gpu::weights` | `KqRows<F>`. 입구는 `Plan`으로 적재하는 `load_placed` 하나 |
| `gpu::launch` | 가족별 모양 선택기, `launch_u32` |
| `gpu::kernels::{kquant, act, elem, attn_mma, convert}` | 모델과 무관한 변형 커널 |
| `gpu::host` | `HostTier` + `StepPort`(플래그·memop) + `BatchPort`(세트·이벤트·상태 기계; 지금 ffn/batch.rs에 있는 것) |
| `gpu::model` | `GpuModel<B>`(카드 하나, 상주, `Option<HostTier>`, `Graphs<Chain>`), 작은 `ChainBody`, `HostServed`/`Rows`/`Rollback`/`Instrumented` |
| `gpu::arch::deepseek2` | V2-Lite 전용 커널과 계기 전부 |

**모듈 밖**
- 배치 산술은 model 크레이트가 소유하고, 본체가 자기 카드 항을 신고한다.
- 서빙 바인딩과 DSpark 드라이버는 엔진 크레이트에 둔다.

**흐름**
- 디코드는 지금과 같다: 캡처된 스텝 하나 + `StepPort`.
- 프롬프트: `run_rows`가 폴트 판독 지점을 소유하고, 층-배치 교환은 `BatchPort`가 맡는다. 발행을 벌크로 할지 비동기로 할지는 그 포트 안의 레버다.

**이행 경로**(단계마다 게이트 녹색)

| 단계 | 내용 | 증명 등급 |
|---|---|---|
| 1 | GC7 1단계 삭제(3장 1–11) | 삭제. 남는 `ptx-scan` 표는 지운 항목만 빠짐, 커버리지 변경마다 날짜 사유 |
| 2 | GC1 | move: `ptx-scan` 동일 + 노드 수, eager = replay, e2e 집합 |
| 3 | GC2 | move, gate-ds41-serve 포함 |
| 4 | GC4 | `ptx-scan` 동일 + 문구 대조 게이트 7곳 |
| 5 | GC3, r8host 뒤 [03] 공동 | move + hybrid, ds41 prefill/step/faults 비트 그대로 |
| 6 | GC5 | 적재·placement 게이트, 같은 계획이면 같은 슬롯 |
| 7 | GC7 2단계 | move |
| 8 | GC6, 형식 핸들 → 선택기 두 라운드 | move, 호스트 전용 |
| 9 | GC8(사용자 결정 뒤), GC3의 흐름 레버 | GC8은 float 순서 등급(σ, 개수 핀, 확인 A/B 1회, V4.1 프리필 게이트 재저작). 흐름 레버는 launch-count 등급이라 카드와 A/B |

## 7. 범위 밖 개선 지점

| 위치 | 내용 | 크기 |
|---|---|---|
| gpu-deepseek41 body/prefill.rs:1804-1811 `fault_or`, :324-338 `take_back` | 드라이버 실패를 `Shape` 문자열로 삼킨다(GC4와 함께) | S |
| gpu-deepseek41 chain/ffn/batch.rs ~1300-1415 | 배치 교환 상태 기계의 소유(GC3) | M |
| gpu-gates/src/bind.rs, bin/shared/ds41_dspark.rs, bin/bloomery_serve_ds41.rs | 엔진 크레이트로 옮긴다(GC2) | M |
| gpu-deepseek41 body/prefill.rs:985-988 | 배치 버퍼를 계획 밖에서 첫 프롬프트 때 잡는다(GC5) | S–M |
| model/src/placement/workstation.rs `SCRATCH 64 MiB [assumed]` | 신고값으로 바꾼다 | S |
| gpu-deepseek41 chain/attn.rs:1691 등 12곳 | `Dense::of`를 enqueue마다 부른다 → 적재 때 해석 | S |
| gpu-deepseek41 chain/attn/batch.rs | `HC_MAX_TOKENS=8` 청크 루프(GC8 뒤) | L |
| gpu-deepseek41의 `GpuError::Shape {` 리터럴 | 도우미로 바꾼다. 기계적 | S–M |
| head.rs:169 | 죽은 `argmax_rows`를 가리키는 주석 | XS |
| plan-triage.md:127 | bfac033이 닫은 누수 항목 | XS |

이미 트리아지에 있는 항목(dispatch.rs:1316, 1333 `h_exp`, launcher.rs:210 대기 기한, gpu-deepseek41 attn.rs:188 조용한 자르기)은 되풀이하지 않는다.
