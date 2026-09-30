# 층 프로그램(DS6) 설계 — 설계 라운드 `layerprogdesign`의 메모

**상태.** 읽기 전용 설계 라운드 `layerprogdesign`(2026-09-27, main `e039209`)의 보고를 그대로 옮겼다. 4파동 `layerprog`의 설계이고, 03의 Qwen3.6 본체가 지금 맞출 인터페이스(§1b)를 먼저 답한다.

**리드의 처분 (2026-09-27, aa).**
- **L1 크레이트 배치**: 권고대로 셋으로 나눈다. 순수 계획(`order()`, CED 걷기, `StoreRule`)은 `crates/runtime`(호스트 전용), enqueue 드라이버는 `GpuModel` 옆(`crates/gpu`), 프로그램은 새 `crates/programs`. `rebuild.md` §2의 `models`가 "층 프로그램"을 갖는다는 줄은 이 결정으로 고친다 — 프로그램은 ops를 필요로 하니 `models`(gguf만 의존)에 둘 수 없다(modelspec L1).
- **L2**: Qwen의 `CedState::Off("no window-only layer")`는 전제조건으로 받는다(리더 거부가 아니다; modelspec §4 "Fact 2"와 같은 갈래).
- **L3**: V4.1의 T-b("decode m = 1과 prompt m = 1이 같은 커널 집합")는 오늘 빨강이므로 `PIN(2026-09-27)`을 붙여 빨강으로 착륙시키고 `batchwide` 뒤에 초록으로 만든다. 게이트 완화가 아니라 알려진 빨강의 핀이다.
- **L4**: R0 `launchlog`는 gpumodel 착륙 직후 3파동에 넣는다(S, 박스 약 14분[유도]). `graph.rs`는 hostone A와 겹치므로 03에게 순서를 맞춘다: A가 먼저면 R0는 그 위에서.
- **L5**: 생기지 않는다. 고른 Qwen3.6 파일은 lmstudio Q4_K_M(Q4_K/Q6_K/F32뿐)이라 dense 프롬프트는 `gemm_q4k`로 간다. 보고 §0-8·§1b 마지막 행의 "Qwen3.6 dense는 전부 Q8_0"은 UD-Q4_K_XL 기준이다.
- **03에게 넘긴 것**: §1b의 hunk 표 전체와 §0 1–8(정정 포함)을 09:2x에 전달했고, 03이 q35body·q36prog 스펙의 구속 조항으로 적었다. K1–K6은 03이 답한다.
- **라운드 순서(§6 표)**: R0 launchlog → R1 progshape(session·hostone B 뒤) → R2 progprompt(hostone C–E·act-planes B·modelspec 뒤) → R3 rowsm(Q3-3) ∥ R4 candwire(opslib R1 뒤) → R5 q36prog(S) → R6 lanes(seqstate 뒤). `batchwide`는 R2 다음의 별도 라운드다.

---

# `layerprogdesign` 보고 — 층 프로그램(DS6) 설계와 03의 Qwen3.6 본체가 지금 맞출 인터페이스

읽기 전용 라운드로 끝났습니다. 트리는 `~/repo/bloomery-lpd`(`e039209`)이고, 편집·git·빌드·게이트·임대는 없었습니다. 박스에서는 참조 트리 읽기(`cat`, `sed -n`, `grep`, `ls`, `find`)와 `/usr/local/cuda/include/cuda.h` grep 한 번만 했습니다. 참조 트리의 커밋은 이렇습니다: ik `db517b69`(`ik-idxkey`, 브랜치 `v41/idxkey-fix`), llama.cpp mainline `53ed051ce`, mistral.rs `d5ae0f18f`, exllamav3 `0740edc`. 사용자가 이 세션에 직접 입력한 것은 없었습니다.

**종이 위 분해.** 이 라운드의 항은 모두 코드에서 센 값이거나 이미 기록된 값입니다.
- 층당 런치 수: `dispatch.rs:199-205`
- 노드 핀: Qwen3 604/601/673, V4.1 1169/1172/2344
- 상태 바이트: 헤더 차원에서 계산
- 줄 수: `wc`

박스 실행이 필요한 항은 없었고, 그래서 카드도 쓰지 않았습니다.

**자원 시간선.**
- V4.1 디코드의 벽은 호스트 다리가 정합니다(토큰당 22.3 ms[유도], `rebuild.md` §1).
- Qwen3의 벽은 카드가 정합니다.
- 층 프로그램은 m = 1의 노드 목록을 바꾸지 않으므로, 어느 자원의 바쁜 시간도 움직이지 않습니다(Δ = 0, 구성상).
- 설계가 새로 비용을 더하는 항은 둘뿐입니다. 하나는 Q4의 레인 쓰기((m − 1)·S / verify)이고, 다른 하나는 끄면 0인 launch 기록기입니다.
- 흐름 레버 가운데 비동기(pair, G)는 이미 쓰고 있습니다. SIMT와 벌크는 `batchwide`의 몫이고 layerprog의 몫이 아닙니다.

---

## §0 권고 (한 줄씩, 03의 hunk 목록이 먼저)

**03에게: Qwen3.6 본체**
1. **새 본체는 `arch/qwen3moe/` 안에 두세요.** 형제 디렉터리 `arch/qwen35moe/`를 만들면 `dispatch.rs`의 `attention`/`ffn`을 import할 수 없어 사본이 생깁니다(check-arch ①, `tools/check-arch.sh:52-66`).
2. **`dispatch.rs:230-249`의 `layer(c, kv, s, io, m, embed, out)` 모양을 그대로 쓰세요.** m행 블록 위의 층 프로그램이 이미 이 모양입니다. 믹서만 `match plan.mixer { Gqa, Delta }`로 가르면 됩니다.
3. **층 종류는 `Hparams`의 층별 벡터(텐서 존재로 정함)에서 읽으세요.** `(l+1) % 4`에서 읽으면 안 됩니다. `LayerPlan::resolve(w, &hp, l)`(`body.rs:207-251` 패턴) 한 곳이 나중에 `&spec.layers[l]`로 바뀌면 스왑이 끝납니다.
4. **GDN 상태는 `RecStore { state: [lanes][32][128][128] f32, conv: 위치 색인 링 }`로 두세요.** `lanes`는 적재 매개변수(오늘 1)입니다. 레인 인덱스와 위치는 파라미터 이미지의 디바이스 워드로 읽고, 커널 인자로 넘기지 않습니다.
5. **GDN 커널은 토큰 j의 상태를 lane `(c + j) mod lanes`에 쓰세요.** lanes = 1이면 제자리 쓰기입니다. 그러면 나중에 레인을 늘리는 것은 `StoreRule` 변경이고 본체·커널 변경이 아닙니다.
6. **`reset`은 GDN 상태와 conv 링을 0으로 지워야 합니다.** Qwen3의 "지울 것 없음"(`body.rs:398-402`)을 복사하면 이전 시퀀스의 상태가 조용히 남습니다.
7. **짓지 말 것:**
   - `ubatch.rs` 같은 두 번째 층 몸체
   - 본체 수준 `history`나 위치 사본
   - 본체 소유 그래프 캐시를 새로 늘리는 것
   - 모델 전용 겹침 코드
   - 층 번호 규칙
8. **Qwen3.6의 dense 투영은 전부 Q8_0이라 GEMM이 없습니다**(act-planes §2.3). m ≤ 8 Q8_0 gemv(`q8f32.rs`, 열마다 m = 1 순서)로 패스만 가능합니다. 8을 넘는 프롬프트 경로의 소유자는 §9에서 묻습니다.

**리드에게: layerprog 설계**

9. 층 프로그램은 `front`/`shadow`/`back` 세 단계를 가진 트레이트로 둡니다. 호스트 다리가 없는 층은 `front`만 씁니다.
10. 겹침 부품은 `Overlap { units, cols, port }` 하나로 둡니다.
    - 네 매개변수 점: 디코드 (1, 1, Step), pair (2, 1, Step), G (g, T, Batch), 앞으로의 `Rows(m)` (1, m, Step)
    - uniongroup의 `(lanes, cols)`가 곧 `(units, cols)`입니다.
    - 항목 순서는 호스트 전용 순수 함수로 두고, 단위 시험으로 오늘의 네 순서를 재현합니다.
11. `Ced::new(&[LayerSpec])`의 `every[l]`은 그 층 저장소의 `StoreRule`에서 유도합니다. 오늘의 `compressor || index_keys`(`ced.rs:229`)를 GQA·GDN 층에 쓰면 K/V와 상태 쓰기를 조용히 건너뜁니다.
12. 첫 라운드는 `launchlog`로 둡니다. launch 기록기와 캡처 노드의 커널 이름(`cuFuncGetName`, 박스 `cuda.h:17350`, CUDA 13.0)이 Q5의 자와 Q7의 탐침을 겸합니다. 오늘 "launch 목록 동일"은 개수로만 근사됩니다.
13. "decode m = 1과 prompt m = 1이 같은 커널 집합" 시험은 Qwen3에서는 오늘 초록이고 V4.1에서는 오늘 빨강입니다. PIN을 붙여 착륙시키고, `batchwide` 뒤에 초록으로 만듭니다.
14. 03의 Qwen3.6을 층 프로그램으로 만드는 라운드는 `q36prog`입니다. 권고 1–7을 따르면 S 크기이고, 03 파일의 hunk 약 150줄 이하입니다[추정].

---

## §1 Q1 — 03의 Qwen3.6 본체가 드러낼 인터페이스

### 1a. 층 프로그램의 모양 (layerprog가 만들 것)

```rust
// wave 4: crates/programs (§9 L1) — until then crates/gpu/src/program/
/// An m-token block as a program sees it. Everything that changes per call is on the device.
pub struct Block<'a> {
    pub m: usize,      // columns: 1 (decode), ≤ 8 (verify), ≤ T/U (prompt); ≤ the arena's cap
    pub io: Io<'a>,    // device windows: tokens, pos, n_keys, rope rows, lane word — never launch args
}
pub struct Cx<'a> { pub gpu: &'a Gpu, pub w: &'a Weights, pub layer: u32, pub sink: FaultSink, pub eps: f32 }

/// One layer kind's program: no allocation, no sync, no host round trip in any method,
/// so the same body runs eager and under capture (qwen3moe dispatch.rs:1-3).
pub trait LayerProgram {
    type Plan;    // resolved once at load from LayerSpec: names, types, instance keys
    type Stores;  // this layer's state view (KV planes / GDN lanes / latent rows), borrowed
    type Arena;   // block scratch at the schedule's cap
    /// Up to the host handoff; the whole layer when the layer has no host leg.
    fn front(&mut self, cx: &Cx, p: &Self::Plan, b: &Block, st: &mut Self::Stores, a: &mut Self::Arena) -> Result<(), GpuError>;
    /// Card work that runs under the host leg (default: none).
    fn shadow(&mut self, cx: &Cx, p: &Self::Plan, b: &Block, st: &mut Self::Stores, a: &mut Self::Arena) -> Result<(), GpuError> { Ok(()) }
    /// From the join on (default: none).
    fn back(&mut self, cx: &Cx, p: &Self::Plan, b: &Block, st: &mut Self::Stores, a: &mut Self::Arena) -> Result<(), GpuError> { Ok(()) }
}
```

- **"층 종류"는 `LayerSpec`의 (mixer, ffn, residual) 셋입니다.** 프로그램은 서브층 프로그램들의 합성입니다. 믹서 종류(Gqa·Latent·DeltaRule)마다 하나, ffn 종류(카드 Moe·호스트 서빙 Moe·Dense)마다 하나가 있고, residual 종류가 입출력 배선(Plain: 마지막 런치의 잔차 합; Hc: HC_PRE/POST와 fold)을 정합니다. 층마다 enum `match` 한 번으로 고르고, 이 비용은 enqueue 때 호스트 ns이며 재생 때는 0입니다.
- **V4.1은 오늘 이미 세 단계로 나뉘어 있습니다.**
  - `front` = `Parts::engram` + `attn` + `FfnPiece::enqueue_handoff`(`body.rs:1493`, `:1529`; `chain/ffn.rs:1074`)
  - `shadow` = `enqueue_shadow` + engram `extra` 작업(`chain/ffn.rs:1075-1078`)
  - `back` = `enqueue_join_half`(대기, join, 탭; `chain/ffn.rs:1085-1100`, `body.rs:1632-1669`)
- **Qwen3와 Qwen3.6은 호스트 다리가 없어서 `front` = 층 전체입니다.**

### 1b. 03의 파일별 hunk 목록

| 파일 | 지킬 모양 | 피할 모양 | 이유 한 줄 |
|---|---|---|---|
| (디렉터리) | 새 본체를 `crates/gpu/src/arch/qwen3moe/` 안에 둔다(예: `body35.rs` 또는 본체 enum). 이름 변경(`qwen`)은 layerprog의 이동 몫이다 | 형제 `arch/qwen35moe/` | check-arch ①이 형제 디렉터리끼리의 `use`를 막는다(`tools/check-arch.sh:52-66`). 막히면 `attention`/`ffn` 사본이 생기고, 이는 원칙 3("층 종류마다 하나", `rebuild.md` §2-1)을 어긴다. 호스트 쪽 리더도 계열 하나(`qwen`)다(modelspec §6 step 2) |
| `crates/model/src/arch/…/hparams` (Qwen3.6 리더) | `layers: Vec<LayerKind>`를 텐서 존재로 정한다(`ssm_a` ⇒ Delta, `attn_q` 행 = 2·heads·head_dim ⇒ Gqa{out_gate}). `full_attention_interval`·`recurrent_layers`와 어긋나면 이름 붙은 거부 | `is_recurrent(l) = (l+1) % 4 != 0` | `docs/arch-split.md:38`. modelspec §2b의 새 거부. 이 벡터 한 칸이 `spec.layers[l]`로 바뀌면 스왑이 끝난다 |
| `body.rs:42-59, :207-251` (`LayerNames`, `resolve`) | `LayerPlan { mixer: MixerPlan::{Gqa(GqaPlan{names, out_gate, v_ty}), Delta(DeltaPlan{names})}, ffn: MoePlan{names, down_ty, shared: Option<SharedPlan{sigmoid_gate}>} }`를 `load`에서 층마다 한 번 푼다(`kq_site`/`f32_site`와 같은 Q8_0 검사를 더함) | 커널 호출 자리에서 이름을 조립하거나 종류를 판정 | 적재 때 한 번 푸는 것이 AGENTS "The step does no load-time work" 규칙이다. `resolve(w, &hp, l)` → `resolve(w, &spec.layers[l], &spec)`가 스왑 호출 하나다(`body.rs:330-333`) |
| `body.rs:120-156` (`pins`) | 종류별 핀: GQA flash HEAD 256·GROUP 8(16/2), rope dims 64, 라우터 N_EXPERT 256, top-k는 런타임 인자(K1) | 모델 전역 상수 비교 한 줄 | 03의 K1(런타임 인자와 컴파일 키의 경계)과 같은 판단이다. 층 종류가 둘이면 핀도 종류별이다 |
| `dispatch.rs:230-249` (`layer`) | 지금 모양 그대로, `attention(...)` 자리를 `match p.mixer { Gqa(g) => attention(c, g, …), Delta(d) => delta(c, d, …) }`로 바꾼다. m은 인자이고 커널 선택은 op 안에서 한다(`:369-398`처럼, V4.1 `router.rs:5-9`처럼) | `layer_decode`와 `layer_pass` 한 쌍, 스케줄을 인자로 받는 분기 | 이 함수가 이미 "m토큰 블록 위의 층 하나"다(모듈 문서 `:1-10`: "One layer body serves both passes"). 이 자리에 이미 모양이 있으니 새로 지을 일이 없다 |
| `dispatch.rs:254-347` (`attention`) | 같은 함수에 `g.out_gate`(σ(gate)·attn, q 투영 행 2배)와 IMROPE(텍스트는 64/256 부분 NeoX, K3 [유도])를 계획 필드로 더한다 | `attention_gated` 사본 | 게이트 GQA와 Qwen3 GQA는 같은 층 종류의 매개변수다 |
| 새 `delta.rs`(같은 디렉터리) | `delta(c, d, st: &mut RecStore, s, io, m)`: conv(링은 `pos mod R`) → L2 → β·decay → 재귀(토큰 루프 안에서 레지스터에 상태를 둠) → gated RMSNorm → out. m ≤ 8의 열 j는 m = 1과 같은 연산 순서를 쓴다 | 프롬프트용 사본, 위치를 호스트 카운터로 넘기는 것 | verify(m ≤ 8)는 pair와 같은 "비트로 스텝 m번" 계약이다(`model.rs:910-916`). 무손실 드래프트 핀(plain = draft 토큰)이 여기에 기댄다. 상태는 f32 저장·재적재와 같은 순서면 비트가 같다 |
| `dispatch.rs:359-449` (`ffn`) | `p.shared: Option<SharedPlan>`로 공유 expert를 더한다(σ(x·w_gate_shexp) 스케일은 combine에서). 라우터 인스턴스는 E = 256 | 공유 expert 전용 두 번째 `ffn` | Moe 한 종류의 매개변수다(modelspec §1 `Shared.sigmoid_gate`) |
| `scratch.rs:14-19, :117-122` (`SP_*`, `Io`) | `Io`에 `lane: &DeviceBuffer<u32>`를, `StepParams`에 `SP_LANE`을 더한다. prefill 슬롯 블록(`prefill.rs:54-63`)에도 같은 워드를 둔다 | 레인이나 링 헤드를 커널 인자(값)로 넘기는 것 | `compress.rs:10-13`: "never from launch arguments, so one captured graph serves every position". 캡처는 인자를 굽는다 |
| `scratch.rs:31-49` 옆 (`KvPlanes`) | `enum LayerStore { Kv(KvPlanes), Rec(RecStore) }`, `RecStore { state, conv, lanes }`. 각 저장소가 자기 `StoreRule`을 알린다(Kv: `Positional`; conv: `Window{slots: (w−1)+MAX_ROWS}`; state: `Recurrent{lanes, committed}`) | 본체 필드 곳곳에 흩어진 GDN 버퍼 | wave 4의 `runtime::state`와 `SeqState`가 규칙을 읽는다(§4). mistral.rs의 층별 `HybridLayerCache::{Attention, Recurrent}`(`qwen3_5_moe/text.rs:826-870`)와 같은 모양이다 |
| `body.rs:398-402` (`reset`) | Rec 저장소의 모든 레인과 conv 링을 0으로 지우고, 레인 워드를 0으로 둔다 | "Nothing to clear" 복사 | 재귀 상태는 위치 색인이 아니라 제자리 덮어쓰기다(`linear-attn.md` §0 항목 4). 지우지 않으면 다음 시퀀스가 앞 상태를 읽는다(조용한 오답) |
| `body.rs:374-392` (`decode_input`/`refresh`) | 토큰·위치·키 수·rope 행에 레인 워드를 더해 H2D 한 번 | 캡처 본체 안의 `if pos == 0 { zero }` | 호스트 분기는 캡처에 남지 않는다. 시작 상태는 `reset`이나 프롬프트 호출 시작이 맡는다 |
| `ChainBody::rollback` / `keep` | 기본 거부 그대로(`model.rs:240-245`); 유지 가능한 위치는 `pos`만 | 호스트 스냅숏 되감기 | session §3d: "never host snapshots on the token path". 레인은 `seqstate`가 연다 |
| `prefill.rs:72, :289-328` (본체 소유 m별 패스 그래프) | Qwen3.6 패스가 필요하면 이 구조체를 확장한다(같은 `enqueue_pass` 걷기) | 두 번째 m별 그래프 캐시 | Q3-3의 "m행 패스 주인 둘"(`rebuild.md` §3 B)을 셋으로 늘리지 않는다. 주인은 `GpuModel`의 `Rows(m)` 키다 |
| `ubatch.rs:471-748` | Qwen3.6에는 손대지 않는다. 8을 넘는 프롬프트가 필요하면 GEMM 분기를 `attention`/`delta`/`ffn` 안에서 op별로(`if m > 8 && gemm_of(ty).is_some()`) 고른다 | `qwen35 ubatch` 사본, 즉 층 루프를 가진 둘째 몸체 | DS6의 축소판이다. act-planes §2.3: Qwen3.6 dense는 Q8_0이라 GEMM이 거부된다 |
| 폴트 | 모든 새 커널이 `c.sink`를 받는다(`dispatch.rs:60` `layer_sink`). 디바이스 워드로 인덱싱하는 곳은 레인 하나이므로 새 사이트는 `DeltaLane` 하나다(c ≥ lanes면 올리고, 접근은 `mod lanes`로 정의된 자리) | 범위 밖 레인을 clamp해 조용히 쓰기 | AGENTS의 fault word 규칙. 값의 NaN은 같은 층 expert 양자화기의 QuantColumn이 잡는다. `fault.rs`의 사이트 순서 표가 아키텍처별이라(`:207-229`) 층 종류가 둘인 모델에는 층 종류별 순서가 필요하다(§7) |

### 1c. 캡처 경계와 fault word가 요구하는 것 (Qwen3.6에 그대로 적용)

- **enqueue 함수 안에 할당·동기화·호스트 왕복이 없어야 합니다**(`dispatch.rs:1-3`). 호출마다 바뀌는 값(토큰, 위치, 키 수, rope 행, 레인)은 전부 파라미터 이미지에 있어야 합니다.
- **형상 상수는 런치 인자가 될 수 있습니다.** heads, d, conv 폭, k-head 매핑 `Tiled`가 여기에 해당하는데, 적재 뒤 바뀌지 않기 때문입니다(K1).
- **fault word는 (layer, site)만 싣고 m이나 스케줄은 싣지 않습니다.** 어느 스케줄에서 올라왔는지는 호스트 오류가 호출의 위치 범위와 함께 이름 붙입니다(§7).

---

## §2 Q2 — 프로그램과 세 스케줄

### 2a. 스케줄이 앉는 자리

| 스케줄 | 호출 | 그래프 | 블록 | 겹침 |
|---|---|---|---|---|
| decode | `Session::step` → `GpuModel::step` | 키 `Rows(1)`, 캡처 | m = 1, `StepParams` 이미지 | `Overlap{1, 1, Step}`(V4.1), 없음(Qwen) |
| verify | `Session::verify` → `GpuModel::step_rows(m)` | 키 `Rows(m)`, m마다 캡처 하나, m ≤ `MAX_ROWS` ≤ 8 | m, 블록 이미지를 슬롯에 H2D 한 번(Qwen3 `prefill.rs:240-251` 패턴) | `B::layout(m)` |
| prompt | `Prompt::prompt` | eager | 층-배치 T ≤ 512(V4.1) 또는 U(Qwen) | `Overlap{g, T, Batch}` |

**작은 모양만 캡처하는 근거.** exllamav3 `modules/dsv4.py:1307`은 `if dsv4_batch_graph and B <= 8 and S <= 16 and R <= 32:`일 때만 배치 단계 그래프를 씁니다. GDN도 `(bsz, seqlen)` ≤ (8, 16)에서만 층 전체를 내부 그래프로 재생합니다(`gated_delta_net.py:1019-1036`, `bc_attn.py:52-53`).

**스케줄이 쥐는 것.**
- 그래프 캐시 키
- 블록 이미지 배치
- 호출 계획: 배치, 그룹, CED need, Qwen의 `PrefillPlan`(`prefill.rs:396-425`)
- 겹침 매개변수
- 캡(cap)별 아레나

**프로그램에 넘기는 것.** `Block{m, io}`, 층 번호와 fault sink(`Cx`), 층의 저장소 뷰, 아레나 뷰(ActPlanes 뷰, `cols = m`)입니다. 프롬프트에서는 층의 need `(part, full)`도 넘깁니다. 프로그램은 `part`부터 잠재 부분(K/V·상태·잠재 행 쓰기)을, `full`부터 블록 전체를 돕니다. decode와 verify에서는 둘 다 블록 시작입니다.

**프로그램에 남는 것.**
- op별 커널 선택. 입력은 (m, 평면 집합, 게이트 레버 `PrefillPath`)이고 스케줄은 입력에 없습니다. 오늘 `dispatch.rs:369-398`(m = 1 라우터 접기)과 `router.rs:5-9`(V4.1 두 모양)가 이 모양입니다.
- 위치를 넘어 상태를 나르는 연산(링 append·attend·commit, 압축기, 인덱서, 재귀)의 청크 루프.
- 층별 계획.

**참조 넷이 갈리는 곳과 따른 설계.**
- **mainline**: ubatch마다 빌더 하나를 짓고 `if (hparams.is_recr(il))`로 믹서를 고릅니다(`src/models/qwen35moe.cpp:178-195`). 캡처는 백엔드가 그래프 모양마다 합니다.
- **mistral.rs**: `DecoderLayer { layer_impl: LayerImpl::{FullAttention, LinearAttention} }`(`qwen3_5_moe/text.rs:415-425`)이고, 층 저장소를 forward마다 체크아웃·커밋합니다(`:826-870`).
- **exllamav3**: `TransformerBlock(attn = GatedDeltaNet | Attention)`(`architecture/qwen3_5.py:316-420`)이고, 모듈이 모양에 따라 경로를 고릅니다.
- **ik**: 수치 기준이고 설계는 mainline과 같은 빌더 모양입니다.
- **따른 것**: mistral.rs의 모양(한 층 타입 안의 믹서 enum, 런타임이 가진 층별 저장소)과 exllamav3의 규칙(op가 모양으로 커널을 고름, 작은 모양만 캡처)입니다.
- **셋 모두와 다른 점**: 캡처 그래프 안에 호스트 티어가 있다는 것입니다(`StepPort`). 그래서 겹침 부품은 우리만의 부품입니다.

### 2b. 겹침 부품: 한 타입, 두 매개변수 집합

```rust
// crates/runtime (host-only): the order is a pure function, tested without a card
pub enum PortKind { Step /* mapped memop in the graph, async host */, Batch /* events + host sync */ }
pub struct Overlap { pub units: usize, pub cols: usize, pub port: PortKind }
pub enum Item { Front(usize, usize), Shadow(usize, usize), Serve(usize, usize), Back(usize, usize) } // (unit, layer)
pub fn order(o: &Overlap, layers: usize) -> Vec<Item>;
```

**순서 규칙.** 오늘 코드에서 전사했습니다[유도: `body.rs:1105-1147`, `:1175-1200`; `prefill.rs:1488-1511`].
- **Step**: (l, u)를 층 우선으로 돌며 `Back(u, l−1), Front(u, l), Shadow(u, l)`를 놓고, 끝에 `Back(u, L−1)`와 head를 둡니다. Serve 항목은 없습니다(풀이 go memop에 반응합니다).
  - units = 1이면 디코드의 front → shadow → back입니다.
  - units = 2이면 pair입니다.
- **Batch**: 항목 x = (l, u)를 층 우선으로 돕니다. 처음에 `Front(0)`를 놓고, x마다 `Shadow(x)`를 놓습니다. units ≥ 2이면 `Front(x+1)`, `Serve(x)`, `Back(x)` 순이고, units = 1이면 `Serve(x)`, `Back(x)`, `Front(x+1)` 순입니다. 이것이 `ahead = g >= 2`(`prefill.rs:1492`)입니다.
- **두 포트가 공유하는 불변식**: 호스트가 항목 x를 서빙하는 동안 카드에는 `Shadow(x)`와 `Front(x+1)`이 줄 서 있고, `Back(x)`가 그 뒤에 옵니다.
- **포트가 달라서 생기는 차이는 `Back`의 자리 하나입니다.** Step은 back이 스트림 대기라서 늦게, 의존하는 front 직전에 둡니다. Batch는 serve가 호출 스레드를 막으므로 반환 직후에 둡니다.

**매개변수 점.**

| 이름 | units | cols | port |
|---|---|---|---|
| 디코드 | 1 | 1 | Step |
| pair(오늘 `Rows(2)`) | 2 | 1 | Step |
| uniongroup | lanes | cols | Step |
| `Rows(m)`(V4.1 나중) | 1 | m | Step |
| G | g ∈ 1..=8(`GROUP_MAX`, `prefill.rs:485`) | T | Batch |

uniongroup의 `(lanes, cols)`(트리아지 `:21`, "(2, 1) = 오늘 Pair")가 곧 `(units, cols)`입니다.

### 2c. `step_rows(m)`: 03이 서명할 모양

1. **`trait Rows: ChainBody { const MAX_ROWS: usize; fn layout(m: usize) -> Overlap; }`**, 조건은 units·cols = m.
   - V4.1 오늘: `MAX_ROWS = 2`(`PAIR_ROWS`, `body.rs:113`), `layout(2) = {2, 1, Step}`.
   - Qwen3와 Qwen3.6: `MAX_ROWS = 8`, `layout(m) = {1, m, _}`, 호스트 다리 없음.
2. **`GpuModel::step_rows(ids: &[u32]) -> Result<RowsOut, GpuError>`**, 1 ≤ m ≤ `MAX_ROWS`.
   - 그래프 키는 `Rows(m)` 하나이고, 캡처는 명시적 `capture(Rows(m))`입니다.
   - 호출마다 블록 이미지를 H2D 한 번 합니다.
   - m행 head가 행마다 argmax와 폴트 사본을 씁니다.
   - Qwen3의 `Prefill.graphs`(`prefill.rs:72`)가 `Rows(m)` 캐시로 옮겨 갑니다.
3. **`HandoffLayout { units, cols, hidden, n_used }`**의 주인은 `host/step.rs` 하나입니다(host-tier §7 item 9와 aa 조건).
   - 단위마다 `Cnt`, `Lyr`, 이미지(cols·n_used id·가중치, cols·hidden x), hsum(cols·hidden)을 둡니다.
   - aa의 handoff 커널 상수는 여기에 컴파일 시간 단언 또는 단위 시험으로 묶입니다.
   - units·cols ≤ 8이고, union 인라인 열 수(`UNION_INLINE_COLS`, host-tier §2.3)가 그 근거입니다.

---

## §3 Q3 — `LayerSpec`이 스케줄에 유도해 주는 것 (적재 때 한 번)

| 필드 | decode / verify | prompt | V4.1 | Qwen3 | Qwen3.6 |
|---|---|---|---|---|---|
| `Mixer::Gqa{heads, kv_heads, head_dim}` | KV 평면 [kv][ctx][hd] f16 ×2, `Positional`; flash 인스턴스(HEAD, PACK) | 같은 평면; CED `every = true`(창 없음) | — | 48층, 32/4 × 128 | 10층(3, 7, …, 39), 16/2 × 256; 위치당 KV 20,480 B[유도] |
| `Gqa.out_gate` | q 투영 행 2배, σ(gate)·attn | 같음 | — | false | true |
| `Rope` | 위치별 rope 행을 이미지에(`refresh`) | 표를 적재 때 전부(`Ubatch` rope 표) | 창 층 1e4 평문, 스트림 층 YaRN(표 둘) | Neox 128 | Imrope [11,11,10,0] = 텍스트에서 64/256 부분 NeoX(K3 [유도]), 행당 32쌍 |
| `Mixer::DeltaRule{Gdn{Tiled}, 16, 32, 128, 4}` | 상태 [lanes][32][128][128] f32 = 층·레인당 2,097,152 B, `Recurrent{lanes, committed}`; conv 링 [(4−1)+MAX_ROWS][8192] f32, `Window` | 층당 한 런치(토큰 루프 안에서 상태를 레지스터에); CED `every = true` | — | — | 30층; `khead_map`은 런치 인자(v-head j → k-head j mod 16) |
| `Mixer::Latent{window}` | 링 `Window{slots 128, shadow}` | CED `slots` | 40층 | — | — |
| `Compress{ratio, rows: Own/From}` | 행은 소유자에만, ⌈ctx/ratio⌉행, `Ratio{ratio}` 상태 링(Holds) | CED 소스; 소유자는 `every` | 2·8·14·20 | — | — |
| `StreamTopK{keys, list, candidates}` | 목록은 소유자가 같은 위치에서 계산; 후보 마스크는 pair 행마다 비트마스크(candmask R2) | 청크·세트마다 비트마스크; CED 소스 넷째 칸 | keys 2·8·14·20, list 2, 8, …, 36, 후보 20 → 24·28·32·36 | — | — |
| `Ffn::Moe{E, top_k, …}` | 라우터 인스턴스 E, top-k는 런타임 인자; m = 1은 norm+router 한 런치(Qwen3), m > 1은 둘 | GEMM 라우트 표(Qwen3), T행 라우터(V4.1) | 384/6, 호스트 티어 | 128/8 | 256/8, `Shared{sigmoid_gate}` |
| `Residual::{Plain, Hc}` | Plain: 마지막 런치에서 잔차 합; Hc: 행마다 hc·fold 핑퐁 | 같음, 배치 세트마다 | Hc | Plain | Plain |
| `Extra::Engram` | 앞 층 FFN 그늘의 토큰 전용 작업 | 청크마다(DS2 대상) | 1, 14 | — | — |
| `BlockDraft.target_layers` | 탭(층 입력 평균) | CED가 탭 층을 마지막 `window` 위치로 넓힘 | 37–39 | — | — |

**CED: `Ced::new(&[LayerSpec])`가 layerprog에서 더하는 것.** modelspec step 4는 인자 타입만 바꿉니다. layerprog는 셋을 더합니다.

1. **`every[l]`을 `StoreRule`에서 유도합니다.**
   - 규칙: 층 l의 저장소 가운데 창 밖 위치에서 읽히는 것(`Positional` 또는 `Recurrent`)이 하나라도 있으면 `true`, 저장소가 `Window`뿐이면 삼각형 대상입니다.
   - V4.1에서는 오늘의 `compressor || index_keys`(`ced.rs:229`)와 층마다 같습니다[유도: 소유자만 `Positional` 행·키를 가짐]. 그래서 V4.1에는 이동입니다.
   - 오늘 규칙을 Qwen 명세에 쓰면 GQA와 GDN 층이 `every = false`가 됩니다. 그러면 삼각형이 마지막 `slots` 위치 밖의 K/V·상태 쓰기를 건너뜁니다. 게이트가 비트 계약이 아닌 곳(밴드)에서는 조용한 오답입니다.
2. **층별 `slots`를 `window`에서 읽습니다.** 오늘은 링 행 수 하나입니다(`ced.rs:220`).
3. **`sources()`에 후보 소스를 넷째 칸으로 더합니다**(candmask R2). 20층이 이미 `every`라서 V4.1의 need는 바뀌지 않고, `exact()`가 사실 1을 검사할 뿐입니다.

**Qwen에서 CED가 주는 것[유도].** 모든 층이 `every`이므로 걷기는 맨 위층의 "full"만 `end − 1`이 든 청크로 줄입니다. 그 아래 모든 층은 모든 위치를 돕니다. 절약은 맨 위 한 층의 q·flash·o·FFN(P − 청크) 위치분으로, Qwen3.6은 대략 1/40이고 Qwen3는 1/48입니다. 그런데 Qwen3의 q·k·v는 한 런치(`dispatch.rs:276-287`)라 잘라 쓰려면 런치를 나눠야 합니다. 권고는 이렇습니다: 창만 가진 층이 없으면 `CedState::Off("no window-only layer")`를 스케줄 전제조건으로 둡니다. 그러면 Qwen의 launch 목록이 그대로입니다.

**스케줄 전제조건으로 남는 사실.** 사실 2(인덱스 키는 압축기 층에만)는 리더의 거부가 아니라 `CedState::Off(why)`로 남습니다(`ced.rs:179-191`, `:220-236`; modelspec §4 "Fact 2"). 위의 "창만 가진 층 없음"도 같은 부류입니다.

---

## §4 Q4 — 재귀 상태와 롤백

**헤더에서 다시 계산한 값[유도].**
- S = 32 v-heads × 128 × 128 × 4 B × 30층 = **62,914,560 B**
- conv C = (4 − 1) × 8,192 × 4 B × 30 = **2,949,120 B**(8,192 = q 2,048 + k 2,048 + v 4,096)

"k = 4에서 252 MB"(`rebuild.md` §2-2 항목 1)는 m = 5 verify의 추가 레인 쓰기 4·S = 251,658,240 B입니다. conv는 들어 있지 않습니다. session-design §3b의 255.6 MB는 층당 conv 한 행을 추가로 셌습니다. 아래 설계에서는 conv가 위치 링이라 그 행은 스텝이 어차피 쓰는 바이트이므로 추가가 아닙니다. 시간은 0.373 ms입니다(674 GB/s로 나눔[유도]).

**GDN 프로그램이 저장소에서 필요한 것.**
- 층마다 `state[lanes]`
- 디바이스 스칼라 `committed` c. 모든 GDN 층이 공유하고 파라미터 이미지에 둡니다.
- conv 링 R = (w − 1) + MAX_ROWS행, 슬롯은 `pos mod R`
- 커널 규칙: 레인 c를 읽고 토큰 j의 상태를 레인 (c + j) mod lanes에 씁니다. decode는 m = 1이라 레인 c를 제자리에서 씁니다. commit은 c := (c + a − 1) mod lanes이고, 다음 `refresh`의 H2D 한 번에 실리므로 추가 비용이 0입니다.
- **conv에는 레인이 필요 없습니다[유도].** 수락 a ≥ 1이므로 되감을 위치 n ≥ p + 1입니다. n에 필요한 입력은 n − 3 .. n − 1이고, 링이 p + m − R 이상을 쥐므로 R ≥ m + 2면 충분합니다. R = 3 + MAX_ROWS ≥ m + 3이므로 성립합니다.
- **상주 바이트[유도].** MAX_ROWS = 8에서 레인 8·S = 503.3 MB, conv 링 (3 + 8)·8,192·4·30 = 10,813,440 B입니다(오늘 2.9 MB).

**`SeqState`가 필요한 것.** 저장소마다 규칙 하나(`Positional`, `Window{slots}`, `Ratio`, `Recurrent{lanes, committed}`)와 committed 워드의 자리입니다. 스냅숏 바이트는 0입니다. 레인 선택은 인덱스 하나입니다.

**한 `Rollback`에서 두 규칙이 만나는 자리.** 트레이트 하나에 저장소 규칙 여럿을 둡니다. `keepable(n)`은 저장소별 규칙의 최소입니다.
- `Positional`: n
- `Window`·`Ratio`: 오늘의 `keep_point`(`body.rs:1256-1283`)
- `Recurrent`: {pos} ∪ 직전 verify의 레인이 덮는 위치 ∪ 호스트 체크포인트

V4.1에는 `Recurrent` 저장소가 없어서 결과가 오늘과 같습니다. 그래서 거부로 미룰 필요가 없고, `seqstate` 라운드가 호스트 시험으로 닫습니다.

**03이 오늘 할 것.** lanes = 1로 짓되 레인 워드·레인 보폭·`(c + j) mod lanes` 쓰기를 처음부터 넣고, conv는 위치 링으로 짓습니다. 그러면 나중에 verify를 여는 일은 적재 매개변수 `lanes = MAX_ROWS`와 `SeqState`의 commit 두 가지이고, 커널과 본체는 그대로입니다.

---

## §5 Q5 — 증명

**decode (move).**
- `just ptx-scan`이 `generate_ds41`, `gate_e2e`, Qwen3 bin(`generate_qwen3moe --features gpu`, `gate_qwen3moe_e2e`)에서 base와 같아야 합니다.
- V4.1 스텝 구조 줄이 같아야 합니다: 커널 = 예측, memop 2·40 + `ARRIVALS`, memcpy = `ARRIVALS`, 위상(`gate_deepseek41_step.rs:1009-1023`, `:1064-1071`), eager = replay(`:1034-1049`).
- dsloop 핀 1172/2344(`gate_deepseek41_dsloop.rs:68-71`), e2e 세트.
- Qwen3 노드 604 / 601 / 673[유도]:
  - m = 1 패스는 1 + 48·12 + 24 = 601입니다(Q6_K v 24층, `pass_launches`, `dispatch.rs:199-205`).
  - m ≥ 2 패스는 1 + 48·13 + 2·24 = 673입니다.
  - 디코드 604 = 601 + head 3이고, 3은 차감한 값입니다.
- 타이밍 A/B는 없습니다(`rebuild.md` §6-0 ②).

**prompt (첫 단계).**
- 10개 plan 파일의 `--plan`이 바이트 단위로 같아야 합니다(`records-refresh` 뒤 `git diff`가 비어야 함).
- `BLOOMERY_STEP_STATS=1`의 `stat prefill front/lb` 개수가 같아야 하고, `gate-gpu-ds41-flowcounts`가 초록이어야 합니다.
- prefill 게이트는 비트 동일이어야 합니다.
- **정직한 한계.** `tools/flow/plans/*.rec`는 plan, lb, need 기록(파일당 93줄)이지 launch 목록이 아닙니다. `Entries`/`Tally`는 자리에서 손으로 올리는 **개수**입니다(`attn/batch.rs:182-196`). `NodeInfo`는 kind만 가집니다(`graph.rs:468-474`). 그래서 오늘은 순서가 증명되지 않고, `launchlog`(§6 R0)가 들어와야 "launch 목록 동일"이 문자 그대로 성립합니다.

**결정 5가 두 스케줄 프로그램에 뜻하는 것.**
- opslib가 뒤집기 전까지 V4.1의 모든 op에서 T > 1 커널은 열마다 1열 커널과 비트가 같아야 합니다. 오늘 배치 커널이 이미 그렇습니다(`router.rs:5-9` "each the step's number").
- 뒤집은 뒤 GEMM 분기는 8을 넘는 곳에서만 쓰고, 날짜 붙은 밴드 재핀을 합니다.
- verify(m ≤ 8)는 뒤집기와 무관하게 영원히 비트 = 디코드입니다. 무손실 드래프트 핀이 그 근거입니다.
- 그래서 op의 선택 규칙은 두 개입니다: "≤ 8: 열마다 m = 1 순서"와 "> 8: GEMM, 밴드". ik의 절단점(`mmvq.cuh:10`)과 같습니다.

**남는 것과 합치는 것.**
- **남는 것**: step, prefill, long, skew, faults, dspark-loop, draft, e2e, flowcounts, qwen3moe-e2e(여섯 ubatch `--dump` md5 포함), gemm. 전부 그대로입니다.
- **observer**: `Seam`(`body.rs:540-569`)과 `BatchSeam`(`prefill.rs:302-331`)을 `Seam { layer, sub, cols: Range, streams, fold, taps }` 하나로 합칩니다. 디코드는 `cols = 0..1`입니다. 타입 이동이고 게이트의 검사 내용은 그대로입니다.
- **교환**: `Boundary`와 `exchange`는 합치지 않습니다. 전송이 다른 두 포트(hostone B의 `StepPort`/`BatchPort`)로 남고, 프로그램은 `Overlap.port`로 부릅니다.

**새 구조 시험 넷.**
- **T-a 순서** (호스트, `order()`): 오늘의 네 걷기(decode, pair, G = 1, G ≥ 2)를 전사한 목록과 같아야 합니다. FAIL-first: Step 규칙에서 Shadow와 Back을 바꾸면 pair가 빨강입니다.
- **T-b 스케줄 무관 걷기**: 층 종류마다 블록 폭 m ∈ {1, 2, 8}에서 launch 이름 목록이 캡처 스케줄(decode·verify)과 eager 스케줄(prompt)에서 같아야 합니다.
  - Qwen3는 오늘 초록입니다. 패스 경로가 m = 1에서 디코드의 `layer`와 같습니다(601 = 604 − 3). FAIL-first는 변이로 합니다: m = 1 라우터 접기를 "디코드인가" 플래그에 묶으면 빨강입니다.
  - V4.1은 오늘 빨강이고 이것이 곧 FAIL-first입니다. 1토큰 배치도 `enqueue_batch_layer`와 라우터 두 런치를 씁니다. `KNOWN_DIVERGENCE` + `PIN(날짜)`로 착륙시키고 batchwide 뒤에 핀을 뺍니다.
  - Qwen3.6은 첫 커밋부터 초록이어야 합니다.
- **T-c 기록기 = 그래프**: 캡처한 걷기의 `LaunchLog` 이름 목록이 그래프 노드의 커널 이름 목록(`cuGraphKernelNodeGetParams` → `cuFuncGetName`, `cuda.h:18773`, `:17350`)과 같아야 합니다. FAIL-first: 기록기를 거치지 않는 런치 하나로 빨강입니다.
- **T-d CED 유도** (호스트): V4.1 명세에서는 오늘의 40층 `every` 벡터와 같고, Qwen3.6에서는 전부 `true`입니다. FAIL-first: 오늘 규칙(`compressor || index_keys`)을 Qwen3.6 명세에 쓰면 빨강입니다.
- **추가 권고**: V4.1 `capture_step`에도 Qwen3 `capture_pass`의 런타임 단언(노드 수 = 예측, `prefill.rs:319-325`)을 둡니다.

---

## §6 Q6 — 라운드, 순서, 크기

박스 분은 게이트 런타임에서 [유도]했습니다.
- V4.1 집계는 `gates-ds41.md` §0′입니다: 본체 게이트 6개 1,037 s, faults 61 s, 제품 615 s, op 157 s.
- 나머지는 `~/.cache/bloomery/gate-times.tsv`의 마지막 5행 중앙값(기록된 런타임이지 이 라운드의 측정이 아님)입니다: qwen3moe-e2e 187, qwen3moe-kernels 92, gemm 169, ds41-prefill 364, flowcounts 126(2행 평균), dspark-loop 107, draft 165, chat 228, ptx-spill 35 s.
- 값은 직렬 합입니다. 두 차선 배치에서는 max(A, B) + X입니다.

| # | 라운드 | 변경 클래스 | 파일 | 증명 | 박스 분 [유도] | 크기 | 선행 |
|---|---|---|---|---|---|---|---|
| R0 | `launchlog` | 호스트 계측 + 기록 kind 추가 | `graph.rs`(`NodeInfo.kernel`), op 런치 진입점 하나, `record.rs`, 레버 행 `BLOOMERY_LAUNCH_LOG` | ptx-scan 동일, 노드 수 동일, 스키마 diff, T-c | step 46 + prefill 364 + qwen3moe-e2e 187 + e2e 49 + gemm 169 + spill 35 = 850 s ≈ 14 | S | gpumodel(`graph.rs`를 hostone A와 순서 맞춤) |
| R1 | `progshape` | move | 새 `program/`(트레이트, `Block`, `Overlap`, `order()`, T-a), V4.1 `Parts` → V4.1 프로그램, decode·pair 걷기 → `Overlap` | ptx-scan 동일, 스텝 구조, eager = replay, 1172/2344, e2e 동일, decode·pair LaunchLog = base | 본체 1,037 + faults 61 + dspark-loop 107 + e2e 49 = 1,254 s ≈ 21 | M | session(호출 모양), hostone B(`StepPort`) |
| R2 | `progprompt` | move | V4.1 프롬프트 G 걷기 → `Overlap(g, T, Batch)`, `Seam` 하나, `Ced::new(&[LayerSpec])` + T-d | `--plan` 10파일 동일, stat 개수, flowcounts, prefill 비트, 프롬프트 LaunchLog = base | 본체 1,037 + faults 61 + flowcounts 126 + chat 228 = 1,452 s ≈ 24 | M | R1, hostone C–E(`BatchPort`), act-planes B, modelspec |
| R3 | `rowsm` (Q3-3) | V4.1은 move, Qwen3는 03 hunk | `Rows`, `step_rows(m)`, `Rows(m)` 키, `HandoffLayout(S, m)`; Qwen3의 `Prefill.graphs` → `Rows(m)`, m행 head | V4.1 `Rows(2)` 노드 = pair; Qwen3 601/673, e2e | 본체 1,037 + dspark-loop 107 + draft 165 + qwen3moe-e2e 187 + kernels 92 = 1,588 s ≈ 26 | M | R1, session |
| R4 | `candwire` (candmask R2) | launch 수만(+2 노드/스텝, 512 배치당 +128 런치), 거절은 삭제 클래스 | V4.1 프로그램의 `enqueue_select`, CED 넷째 소스 | 소유 게이트 비트, 노드 핀 재핀(PIN), candmask G2·G2b | 본체 1,037 + faults 61 + op 157 = 1,255 s ≈ 21 | M | R2, opslib R1 |
| R5 | `q36prog` | move(03 파일, hunk) | Qwen3.6의 `layer` → `front`, `LayerStore`가 `StoreRule`을 알림 | T-b 초록, Qwen3.6 e2e(03의 게이트) 불변 | qwen3moe-e2e 187 + kernels 92 + Qwen3.6 게이트(미정) ≈ 5 + α | **S**(권고 1–7을 따랐으면; 03 파일 약 150줄 이하 [추정]) | R1, 03의 본체 |
| R6 | `lanes` | 새 동작 | `lanes = MAX_ROWS`, `SeqState` commit, Qwen3.6 `step_rows` | verify(m) 토큰 = m 스텝(비트), seqstate 호스트 시험(commit 변이 FAIL-first) | Qwen3.6 verify 게이트(새) | M | R3, R5, seqstate |

- **03의 Qwen3.6을 프로그램으로 만드는 것은 R5입니다.** §1b를 따랐다면 `layer`는 이미 `front`이고 저장소도 이미 규칙을 가지므로, R5는 포장과 등록만 하는 S 라운드입니다. 따르지 않았다면(형제 디렉터리, ubatch 사본, 캐시 추가) R5는 03 파일의 재작성이 됩니다.
- **`batchwide`(DS2)는 layerprog 라운드가 아니라 R2 다음입니다.** 필요한 것은 둘입니다. 프로그램의 prompt 블록이 API에서 T 폭이어야 하고(`Block{m: T}`, 8은 커널 안 타일), op가 `cols = T` ActPlanes 뷰를 받아야 합니다(act-planes D). 클래스가 "launch 수 + 기하 + 산문 A/B 1회"라 move와 섞으면 증명이 무너집니다. R2가 끝난 뒤 청크 루프를 링, 압축기, 인덱서 op 안으로만 줄이는 라운드로 둡니다.

---

## §7 Q7 — 세 층

1. **구조적 모양.**
   - "층이 하는 일"의 주인은 서브층 프로그램입니다. 믹서 종류마다, ffn 종류마다 하나이고, residual 종류가 배선을 맡습니다.
   - m에 따른 커널 선택의 주인은 op(`crates/ops` enqueue 함수)이고, 입력은 (m, 평면, 게이트 레버)입니다.
   - 스케줄은 "어느 (unit, layer, block)을 어떤 순서로 어느 포트로"에서 끝나고, 프로그램은 "이 블록이 주어지면 이 층을 enqueue한다"에서 시작합니다.
2. **깎이지 않게 하는 것.**
   - T-a부터 T-d까지
   - `--plan` 덤프
   - `ptx-shapes` 래칫
   - `check-arch`에 규칙 ⑤ 추가 제안: `program/`와 ops 파일이 스케줄 이름(`Sched`, `Chain::`, `PortKind`)을 쓰지 않는다(grep, Mac).
3. **디버깅 용이성.**
   - observer는 `Seam` 하나입니다.
   - 탐침은 `BLOOMERY_LAUNCH_LOG=1`입니다. 층마다, 스케줄마다 `launch layer=l sched=decode|verify|prompt m=… unit=… kernel=<entry> planes=<set>` 기록 줄을 찍습니다. `record.rs`의 kind이고, `records.py`로 읽으며, R0가 이것입니다.
   - 폴트: `(layer << 8) | site`는 세 m에서 같은 값을 냅니다. 스케줄과 위치는 호스트가 붙입니다(`GpuError::Fault`에 `sched`, 위치 범위, m; 프롬프트는 그룹 끝에서 읽음, `prefill.rs:1305-1311`).
   - 사이트 순서 표는 아키텍처별(`fault.rs:207-229`)에서 층 종류별로 바꿉니다. Qwen3.6의 GDN 층과 GQA 층은 순서가 다릅니다. 오늘 `QWEN3MOE` 표에도 "ubatch 경로 순서가 다르다"는 주석이 이미 새어 있습니다(`:203-205`).

---

## §8 위험: 디코드가 이동이 아니게 되는 경우와 먼저 볼 것

1. **아레나 병합.** decode `s`(행 1)와 패스 `a`(행 8)를 한 캡 8 아레나로 합치면, 오늘 런처가 열 수를 `act.m()`에서 읽기 때문에(act-planes §3) m = 1 디코드가 8열을 계산할 수 있습니다. ActPlanes C(열 수 = 런치 인자) 전에는 합치지 않습니다. 그 뒤에도 스케줄 계열(캡처 ≤ 8, eager U)마다 아레나 하나이고, 평면별 캡은 필요 없습니다.
2. **m = 1에서 커널이 다르게 골라지는 경우.** Qwen3의 m = 1 라우터 접기(`dispatch.rs:369-379`)나 V4.1 디코드 한 런치 라우터가 "블록이 프롬프트인가"에 묶이면 노드 수는 같아도 커널이 바뀝니다. T-b와 ptx-scan이 아니라 LaunchLog가 잡습니다.
3. **겹침 순서.**
   - Step 포트의 Shadow/Back 자리가 바뀌면 노드 수는 같고 간선이 달라집니다. 스텝 게이트의 위상 검사(`ok_topo`)와 T-a가 잡습니다.
   - pair 이후 `hybrid.row_enqueued` 호출 순서(`chain/ffn.rs:1099`)가 바뀌면 `StepPort`의 순번 검사가 거부합니다(host-tier §1.3 step 4).
4. **그늘 안의 `extra` engram 작업과 `FeatureTap`의 자리.** `extra`(`body.rs:1603-1620`)와 `FeatureTap`(`:1661-1663`)이 `shadow`와 `back`으로 정확히 옮겨가지 않으면 dsloop 1172/2344가 움직입니다.
5. **먼저 볼 것**: `just ptx-scan` → 스텝 구조 게이트의 kind·위상 → decode와 pair의 LaunchLog diff 순서입니다.

---

## §9 열린 질문

**리드에게**
- **L1 크레이트 배치.** session-design은 `crates/runtime`를 호스트 전용으로 확정했습니다. `rebuild.md` §2는 `sched/`(캡처)와 `state/`(디바이스 저장소)를 runtime에 둡니다. modelspec L1은 프로그램이 ops를 필요로 하니 models에 둘 수 없다고 합니다. 권고는 셋으로 나누는 것입니다: 순수 계획(`order()`, CED 걷기, `StoreRule`)은 runtime, enqueue 드라이버는 `GpuModel` 옆, 프로그램은 `crates/programs`.
- **L2.** Qwen의 `CedState::Off("no window-only layer")`를 전제조건으로 받을지.
- **L3.** V4.1 T-b를 빨강·PIN으로 착륙시키고 batchwide 뒤에 초록으로 만드는 순서를 받을지.
- **L4.** R0를 gpumodel 직후 wave 3에 넣을지. `graph.rs`는 hostone A와도 겹칩니다.
- **L5.** Qwen3.6의 8을 넘는 프롬프트 경로(Q8_0 GEMM)의 소유자는 누구인지. opslib의 Q5_K 전문가 gemv와 같은 파동인지.

**03에게**
- **K1.** 같은 디렉터리(권고)인지, 형제 디렉터리(check-arch ①로 사본)인지.
- **K2.** GDN 커널의 레인 워드, 레인 보폭, `(c + j) mod lanes` 인터페이스를 지금 넣을지.
- **K3.** Qwen3.6 `reset`에서 상태와 conv를 0으로 지울지.
- **K4.** `Rows { MAX_ROWS, layout }`, `Rows(m)` 키, `HandoffLayout(units, cols, hidden, n_used)` 세 줄에 서명할지.
- **K5.** 층 종류별 폴트 사이트 순서.
- **K6.** Qwen3의 `dispatch::layer`와 `ubatch` 병합을 Q3-3에서 할지, R3의 03 hunk로 할지.

---

## §10 범위 밖 개선 지점 (보고만)

| `path:line` | 한 줄 | 크기 |
|---|---|---|
| `crates/gpu/src/graph.rs:468-474` | `NodeInfo`가 kind만 가진다. 커널 이름(`cuFuncGetName`, 박스 `cuda.h:17350`)을 붙이면 구조 게이트가 이름을 읽을 수 있다 | S |
| `crates/gpu/src/model.rs:690-715` | V4.1 `capture_step`에는 Qwen3 `capture_pass`(`prefill.rs:319-325`) 같은 런타임 노드 수 단언이 없다(게이트에만 있음) | S |
| `crates/gpu-deepseek41/src/body/ced.rs:229` | `every = compressor \|\| index_keys`가 V4.1에만 맞다. 재사용하면 조용한 오답이다. 인자를 `StoreRule`로 받게 | S |
| `crates/gpu/src/fault.rs:203-229` | 사이트 순서가 아키텍처별이다. ubatch 순서의 차이가 주석에만 있다. 층 종류별이 되어야 한다 | S |
| `crates/gpu/src/model.rs:68` | `fn arch() -> Arch`가 정적이라 한 Body가 두 GGUF arch를 섬기지 못한다. 인스턴스 메서드로 | XS–S |
| `crates/gpu-deepseek41/src/chain/attn/batch.rs:182-196`, `body/prefill.rs:804-875` | `Entries`/`Tally`는 자리에서 손으로 세는 개수라 순서를 못 본다. LaunchLog가 대신하면 DS5가 닫힌다 | S–M |
| `crates/gpu/src/arch/qwen3moe/body.rs:398-402` | "Nothing to clear"는 KV에만 참이다. 재귀 상태가 들어오는 자리의 함정이라 한 줄 주석이 필요하다 | XS |
| `docs/research/audit/ds41.md:48-49` | 줄 수 8,274/6,738은 `4786c1e` 값이다. `e039209`에서는 `cat body.rs chain/{attn,ffn,glue}.rs \| wc -l` = 8,384, `cat chain/*/batch.rs body/{prefill,ced}.rs \| wc -l` = 6,405 | XS |
| `crates/gpu/src/arch/qwen3moe/dispatch.rs:264, :368` | `let i = m - 1`로 m별 `Q8Act` 벡터를 색인한다. ActPlanes B에서 사라진다(확인만) | XS |

## §11 모델

opus(Opus 5.5)로 돌았습니다.
