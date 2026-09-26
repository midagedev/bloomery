# 감사 라운드 `audit-gates-core`(GG) 보고

**요약:** P0b–P10 사다리는 V2-Lite GPU를 띄울 때 쓴 발판입니다. 지금 이 사다리만 핀하는 커널은 V2-Lite 전용이거나 엔진이 부르지 않는 참조용뿐입니다. 공유 계약 다섯 갈래(아래 GG1)만 먼저 옮기면 사다리 전체를 걷어낼 수 있습니다. 배치 벽시계는 거의 줄지 않습니다. 줄어드는 것은 계약이 바뀔 때마다 이 사다리에 붙는 수정입니다.

## 0. 읽은 범위

- **읽은 커밋:** 시작할 때 `4786c1e`였고, 읽는 도중 main이 `fa61362`로 5커밋 전진했습니다. 범위 안 파일 중 바뀐 것은 `gate_p4.rs`(+56줄, `031b842`가 f16 인코드 전수 검사를 추가)와 `gate_qwen3moe_e2e.rs`(범위 밖)뿐입니다. gate_p4 변경은 아래에 반영했습니다.
- **전부 읽음:** 스펙 두 개, `justfile` 1,162줄 전체.
- **부분적으로 읽음:**
  - `crates/gpu-gates/src/lib.rs`: 1–680, 1100–1360, 1570–1760, 개요.
  - 라이브러리: `engine.rs` 18–113, `generate.rs` 20–170.
  - 바이너리: `gate_p6.rs` 715–830, `gate_p8.rs`(개요, 640–760, 940–975), `gate_load_v41.rs`(머리글, 87–190), `gate_hybrid.rs`(머리글, 함수 개요), `bin/generate.rs` 285–305, `gate_e2e.rs` 740–760.
  - `crates/model/tests`: `common/{manifest,oracle}.rs` 전부, `common/v41set.rs` 1–60, `ds41_host.rs` 100–224, `ds41_meta.rs` 575–724, `qwen3moe_meta.rs` 50–110, `alloc.rs` 20–133, `union.rs` 1008–1034.
- **머리글만 읽음:**
  - 범위 bin: p0b, p1–p6, p8, p8b, p9, p10, e2e, hybrid, head_gpu, iq, q4k_sel, mcol, moe_fused, block, generate, load_v41.
  - 라이브러리 모듈 전부: act_rule, kld, ptx, prompts, bind, block, draft, nodes, qwen3moe, hc_host, ds41_meta, ik_norm, ik_q8_2, rounding, oracle/*.
  - `crates/model/tests` 22개 파일 전부.
- **본문을 읽지 않음:** `gate_vision_encoder.rs`, `bench_join.rs`, `h2d_probe.rs`, `probe_host_register.rs`, `rawx_floor.rs`, `real_x.rs`, `oxart_{jit,ptx}.rs`. 이들은 justfile, tools, rig-log, 커밋 메시지로만 판단했습니다. `act_rule/kld/ptx/prompts/bind/block.rs`와 P 게이트 대부분도 본문은 읽지 않았습니다.
- `gate_p7.rs`는 한 번도 존재한 적이 없습니다(`git log --all -- …/gate_p7.rs` 결과 없음). P7은 블록 하니스이고, `gate_block.rs:1`이 "package P7b"라고 적고 있습니다.
- **잰 줄 수(`wc -l`):**
  - gpu-gates 비-bin 모듈 9,625줄.
  - 범위 bin 21,208줄.
  - `crates/model/tests` 11,744줄(common 포함).
  - 레시피 195개 중 `gate-*` 95개(`just --summary`).
- **게이트 비용:** `~/.cache/bloomery/gate-times.tsv`(677행, 첫 행이 09-26 06:10)의 최근 5행 중앙값입니다(측정값).

## 1. 지금 아는 것 (seed 목록 밖, 결론을 좌우하는 것)

1. **대상 프로필** (`docs/plan.md:141-143`)
   - T1 = 3090 한 장 + `--place gate`
   - T2 = 3090 두 장(두 카드 합류 경로 필요, 걸림돌 여섯)
   - 측정 기준 = A6000 plan (a)
   - design §5의 plan (b)(A6000 0–19층 + 3090 20–39층)는 이 셋 어디에도 없습니다. `plan_b`를 쓰는 곳은 `gate_load_v41.rs:180`, `tests/placement.rs:659`, `tests/deepseek4_meta.rs:472` 셋뿐이고, 엔진 경로는 없습니다.
2. **V2-Lite 커널을 부르는 곳은 V2-Lite 엔진뿐입니다.** 아래 커널을 부르는 곳은 `crates/gpu/src/arch/deepseek2/dispatch.rs`뿐입니다(:117, :259, :286, :440–548, :829, :860, :994, :1160, :1178, :1431, :1502).
   - embed_rows, rope, flash_latent 계열, router_topk, q5_0_gemv_sel, moe_fused, down_add_q5_1, gate_up_swiglu, V2-Lite의 fused.rs kv_norm_rope_append, card_sel
   - V4.1은 같은 이름의 자기 함수를 따로 씁니다(`gpu-deepseek41/src/rope.rs:374`).
   - V2-Lite를 도는 바이너리는 `gate_e2e`, `generate`, P 게이트뿐입니다.
3. **P 사다리는 배치 벽시계에서는 싸고, 계약 비용에서는 비쌉니다.**
   - 게이트 하나가 대부분 약 2 s입니다(§2 표).
   - `6123133`(q3fix3, 09-26)의 stat: 공유 q8_1 거부 계약을 넣으면서 gate_p0b 89줄, gate_p5 183줄, gate_p6 210줄, gate_p1 31줄이 바뀌었고, router.rs +69줄로 V2-Lite 라우터 경로를 고쳤습니다.
   - `7928a0f`(09-25): "V2-Lite's hybrid … a new `card_sel` kernel". V2-Lite 전용 디바이스 코드가 새로 생겼습니다.
   - `031b842`(오늘): Qwen3 `gqa_prefill_flash`가 쓰는 `flash::f32x2_to_f16x2_bits`의 전수 검사가 `gate_p4.rs`(V2-Lite elem 게이트)에 들어갔습니다(`gate_p4.rs:756-807`).
4. **무엇이 게이트인지는 레시피 이름 접두사가 정합니다.**
   - `tools/recipes.py:1087`은 `gate-`로 시작하는 이름만 고르고, `tools/gate-batch.sh:211`은 `"  gate-"`로 시작하는 줄만 읽습니다.
   - 그래서 f64 밴드 검사인 `bench-gpu-v41-check`(`justfile:251-252`, V4.1 gemv 21개 사이트)와 `bench-cpu-v41-host-check`(:355-356)는 착륙 묶음에 들어가지 않습니다.
5. **f64 심판 `exact_ref`와 교사강제 참값 파일은 V2-Lite 전용입니다.**
   - `exact_ref.rs:1-18`: 상태가 "`gate_e2e`'s forced arm"입니다.
   - AGENTS.md의 "Performance first, accuracy opt-in" 절이 심판으로 부르는 도구가 이것입니다.
   - V4.1과 Qwen3의 정확도 자는 ik 덤프 밴드와 KLD입니다.
6. **V4.1 스텝 게이트는 V2-Lite hybrid의 엔진 절을 이미 V4.1 위에서 하고 있습니다.** `--structure`가 fault → poison → reset → 호스트 release 워드 → 뒤 8스텝 비트 동일을 핀합니다(`gate_deepseek41_step.rs:24-39`).
   - 겹침 레버 off를 도는 레시피나 게이트는 0건입니다(`docs/plan-triage.md:126`).
   - rig-log에 겹침 A/B 판정은 없습니다(`grep 'HYBRID_OVERLAP|overlap 레버' log/*.md` 결과 없음).
7. **fixup6가 P 사다리를 손으로 뺐지만, gatesel 보고가 "증명되지 않았다"고 되돌렸습니다**(`docs/plan-triage.md:233`). 착륙 묶음은 이제 `just affected` 목록 전부입니다.
8. **V4.1 fixture는 "작은 모델"이 아닙니다.** 9층에 실제 모양이고 약 60.9 GB이며, 이득은 RAM에 상주해 매 적재가 warm이 되는 것입니다(`docs/plan-triage.md:238`, [유도]). 작은 모델 역할을 맡을 후보는 Qwen3(전 카드, e2e 172 s)입니다.
9. **두 융합 라운드의 시간 판정은 이미 기록돼 있습니다.**
   - P0b: 104.2 → 84.5 µs(`docs/gpu-design.md:79`)
   - MoE 융합: 63.1 → 57.6 µs(`:94`)
   - 둘 다 09-21입니다.

## 2. 구조적 발견 (payoff/cost 순)

### GG1. V2-Lite P 사다리는 띄우기용 발판이다. 공유 계약 다섯을 옮긴 뒤 한 번에 은퇴시킨다

- **현재 모양:** 커널 게이트가 V2-Lite를 띄울 때의 꾸러미 번호(P0–P10) 단위로 짜여 있습니다. 각 게이트는 V2-Lite 파일과 `ref_cuda_v2` 세트를 엽니다. 공유 커널의 핀도, V2-Lite 전용 커널의 핀도, 참조용 op 경로도 같은 바이너리에 섞여 있습니다.
- **P 사다리 표:** 비용은 최근 5행 중앙값(측정)입니다.

| 게이트 | 오늘 핀하는 것 | 엔진 사용처 | 다른 커버리지 | 비용 | 빠지면 유일 커버리지를 잃는 것 |
|---|---|---|---|---|---|
| p0 (`gpu-spike`, 345줄) | eager = replay 바이트, 0단계 `attn.q4k`의 q4k gemv ≤ 1e-2 | 공유 q4k | p1(1e-5), 모든 graph = eager 팔 | 64 s | 없음 |
| p0b | FFN 융합 4 = op 8 비트, ds2 `fused.rs:635` kv_norm_rope_append, CachePos | ds2만 | e2e(토큰) | 2 s | ds2 융합 비트와 CachePos(V2-Lite 전용). op 경로의 `gemv_q5_1`·`kv_append_pos_buf`는 엔진 호출 0 |
| p1 | q3k/q4k/q6k gemv의 KERNEL_BAND(V2-Lite 행), K = 2048 FNV 핀(0단계 포트), q8_1 NaN 거부 | 공유 | `bench_v41 --check`(게이트 아님), gate_mcol(비트), gate_gemm·q4k_sel(QuantColumn), qwen3moe_router(NormQuant) | 2 s | `FaultSite::Q5Quant`(fault.rs:71, V2-Lite 전용) 핀. 공유 gemv의 f64 밴드는 착륙 묶음 안에서 **p1이 유일**(bench-v41-check가 게이트가 될 때까지) |
| p2 | Q5_0/Q5_1 plain gemv | **호출 0** | — | 2 s | 없음 |
| p3 | F32 라우터 gemv, derived Q8_0 1e-5 | f32: ds2 dispatch.rs:1160, q8_0: V4.1 dense.rs:833 | ds41 woa/index/engram/moe, qwen3moe_router | 2 s | 없음 |
| p4 | elem 밴드(embed, rms_norm, rope, swiglu, add, weighted_sum, argmax, half_decode), TokenId, **컴파일 모양**(norm·argmax 폭), **f16 인코드 전수**(031b842) | rms_norm·add 공유, embed_rows·rope는 ds2, 나머지는 호출 0 | V4.1 사슬(맥락 안) | 2 s | **공유 elem 격리 밴드, 모양, half_encode 전수 → 옮겨야 함** |
| p5 | kv_append, flash_latent 계열, 곁 양자화 거부, 로컬 디포 없음 | ds2만 | e2e | 2 s | ds2 전용(디포 단언은 GG2로) |
| p6 | router_topk 64/6, expert_table, Router 거부, **q8_0_gemv·q8_0_gemv_heads FMA 바닥 + Q3_K cvt/clz**(:715-830) | router: ds2, expert_table: 0 | — | 62 s(행 3, 98, 117, 62, 61) | **공유 커널의 컴파일 모양 → 옮겨야 함** |
| p8 | 블록 0 스텝 대 오라클 밴드, 노드 핀 18, 재시드 + `--time/--profile/--bench-kernels` | ds2 | e2e | 3 s | ds2 전용. 단 `--bench-kernels`는 살아 있는 Qwen3 GEMM 벤치(GG3) |
| p8b | 층 1 MoE 스텝, 라우팅 프로브, 노드 핀 22 | ds2 | e2e | 2 s | ds2 전용 |
| p9 | `_sel` = plain 비트, OOR/ExpertId | q5_0_sel: ds2, q3k_sel: 0 | gate_q4k_sel(같은 `_sel` 계약) | 2 s | ds2 전용 |
| p10 | V2-Lite 상주 가중치: census, 독립 패커 대비 되읽기, staging, derived | weights.rs 공유 | gate_load_v41 3항(카드 형식마다 첫 텐서 되읽기) | 44 s | Q5_0/Q5_1·Q8_0Derived 패킹(ds2 전용) |
| head_gpu | head vs ref_cuda_v2 | Head 공유 | ds41_chain_glue(head 끝) | 2 s | 없음 |
| moe (moe_fused) | MoE 융합 4 = op 8 + `--time` | ds2 | e2e | 2 s | ds2 전용 |
| block | 하니스 자기 검증(V2-Lite 두 세트) | — | — | 5.5 s | V2-Lite 세트 전용 |

- **어떻게 쌓였나:**
  - 09-21 하루에 P1–P10이 순서대로 착륙했습니다(`3cd704e`, `18e7b92`, `ceaf559`, `8e5d7ba` …).
  - 그 뒤로 계약이 공유 커널에 새로 붙을 때마다 이 꾸러미 게이트에 절을 얹었습니다(fault 사이트마스크 `7928a0f`, q8_1 거부 `6123133`, f16 전수 `031b842`).
- **오늘 짓는다면:** 꾸러미가 아니라 **커널 계열 게이트**로 짭니다. 모델 파일 없이, 서빙 모델의 모양으로 만든 합성 행에 대해 f64 참조와 digest 줄을 냅니다.
  - `gate-gpu-kquant`: q3k/q4k/q5k/q6k/q8_0/f32 gemv, `_sel`, m열, tiles. 오늘의 `bench_v41 --check`, `gate_mcol`, `gate_q4k_sel`, p1·p3·p9의 공유분을 합칩니다.
  - `gate-gpu-elem`: norm, add, argmax_fault, half_encode, embed_rows_q4k.
  - q8_1 거부는 gemm·q4k_sel에 이미 있으니 q3k_quantize_q8_1과 norm_quant의 NaN lane만 보탭니다.
  - 기존 선례로 `gate_iq`가 이미 이 모양입니다(ref-synth 합성 행, 모델 없음).
  - mistral.rs도 같습니다. `mistralrs-quant/tests/grouped_mmq_packed_cuda_tests.rs:20-59`가 합성 입력을 CPU 참조와 TOLERANCE로 비교하고, 모델 파일은 쓰지 않습니다.
- **payoff:**
  - 사다리 bin 8,979줄(p0b, p2, p4, p5, p6, p8, p8b, p9, p10, head_gpu, moe_fused, block; `wc`)
  - p1·p3 714줄, lib의 V2-Lite 전용 927줄(`block.rs`, `engine.rs`, `oracle/deepseek2.rs`), gpu-spike 345줄 [합은 유도]
  - 줄 수보다 큰 것은 공유 계약이 바뀔 때 V2-Lite 게이트와 V2-Lite 커널에 두 번 적어야 하는 비용이 사라진다는 점입니다(`6123133`, `7928a0f`, `031b842`).
  - 배치 시간은 사다리 중앙값 합이 약 200 item-초[유도]이고 두 레인이 병렬이라, 벽시계 이득은 작습니다. 과장하지 않습니다.
- **비용과 증명 부류:**
  - 절 옮기기는 이동 부류입니다. 옮긴 절의 출력 줄이 같다는 것을 `just verdict-diff`로 보이고, 절마다 FAIL-first를 합니다.
  - 게이트 제거는 커버리지 변경이므로 게이트마다 날짜 붙은 사유를 적습니다(이 표가 그 기록).
  - 시간 A/B는 필요 없습니다.
- **위험·소유·순서:**
  - aa 소유입니다. 커널 삭제는 gpucore, `fused.rs`와 `fault.rs`는 `[03]`입니다.
  - V2-Lite **엔진**을 남긴 채 사다리만 걷으면, 그 엔진의 조용한 실패 금지 핀(CachePos ds2, Router ds2, Q5Quant, QuantColumn 곁, TokenId embed_rows, ExpertId q5_0_sel)이 사라집니다. 엔진 결정(GG1 다음 단계, §6의 9)과 함께 가야 합니다.
  - GG2와 GG3이 먼저입니다.

### GG2. 커널의 컴파일 모양이라는 사실이 네 곳에 있다. 래칫 표 하나로 모은다

- **현재 모양:**
  - `gate_p4.rs:780-845`(rms_norm·norm_quant·argmax의 `.reqntid` = 호스트 폭)
  - `gate_p5.rs:154`(`no_local_depot`)
  - `gate_p6.rs:715-790`(f32_gemv·q8_0_gemv·q8_0_gemv_heads FMA 바닥, router 폭)과 `:795-830`(Q3_K에 `clz` 없음, `cvt` 있음)
  - 이 셋은 모두 `lib.rs:1578-1608`의 `ptx_shapes`를 거칩니다. 즉 V2-Lite 모델을 올리고 카드를 잡는 런타임 게이트 안에서 컴파일 사실을 단언합니다.
  - 따로 `tools/ref/ptx-shapes.tsv`(274줄, generate_ds41 158행, gate_e2e 101행)와 `gate-ptx-spill`(`justfile:1089-1090`)이 spill과 jit_local만 잽니다.
- **쌓인 경위:** A6(`677481f`, 09-21)이 PTX에서 flash_latent 스필을 찾아낸 뒤 게이트마다 단언을 달았습니다. 그 뒤 스필 래칫이 생겼지만 게이트별 단언은 그대로 남았습니다.
- **오늘 짓는다면:** 엔트리마다 한 행을 둡니다(spill, jit_local, reqntid, depot/ld/st, fma_floor, 필수·금지 op). `tools/ptx-scan.sh`는 이미 블록 폭, 디포, 로컬 왕복, 명령 digest를 찍고 있으니(`justfile:1076-1080`, `121fb43`), 호스트 패스 하나가 판정하면 됩니다.
- **payoff:** 공유 커널(q8_0_gemv_heads, q3k gemv, norm_quant)의 모양 핀이 V2-Lite 파일, 카드, P 게이트에 기대지 않게 됩니다. GG1에서 P4, P5, P6을 걷을 수 있게 됩니다.
- **비용과 증명:** S. 컴파일 시점 래칫이라 비트와 무관하고, 핀 하나를 고쳐 FAIL-first를 봅니다.
- **소유와 순서:** aa. boxlease와 무관합니다(`ptx-spill-check.sh`는 boxlease 목록 밖).

### GG3. gate_p8이 살아 있는 Qwen3 GEMM 벤치를 V2-Lite 블록 0 게이트 뒤에 품고 있다

- **현재 모양:**
  - `gate_p8.rs:678-681`이 `--bench-kernels`를 받고, `:724-1089`의 `bench_kernels(&Deepseek2Model)`와 `:957`의 `gemm_arms`(Qwen3 모양 128 × 768 × 2048 Q4_K top-8)를 부릅니다.
  - 이 벤치를 부르는 레시피는 `justfile:211-212`(bench-gpu-kernels)와 `:220-221`(ncu-gpu-gemm)입니다. `tools/ref/ncu-gpu.sh:230`의 `GEMM_BIN` 기본값도 `target/release/gate_p8`입니다.
  - 사용 기록은 rig-log 09-25#a5gemm-a6000과 09-26 ncu 시팅입니다.
  - Qwen3 커널 하나를 재려면 V2-Lite 8 GB를 올리고, 임대 창 안에서 블록 0 정확성 팔을 먼저 돌려야 합니다.
- **쌓인 경위:** R-1 노드 가격 벤치(09-21)에 a5gemm의 GEMM 팔이 얹혔습니다.
- **오늘 짓는다면:** 모델 없는 `bench_kernels` bin입니다. 커널 계열별 팔을 두고, bench_v41의 `SITES` 표 같은 모양으로 짭니다. p8의 `--time`/`--profile`은 V2-Lite와 함께 갑니다.
- **payoff:** 03의 q3gemm 라운드가 V2-Lite에서 풀립니다. 임대 창에서 쓸데없는 적재와 정확성 팔이 빠집니다.
- **비용과 증명:** S. 이동 부류이고 팔의 줄과 digest가 같으면 됩니다.
- **소유와 순서:** 분리는 aa, GEMM 팔 내용은 `[03]`입니다. q3gemmd 뒤, ncu-gpu.sh 경로는 boxlease 뒤입니다.

### GG4. 앱 계층이 게이트 크레이트에 산다. V4.1 디코드 루프가 둘이고, chat은 프롬프트를 스텝으로 먹인다

- **현재 모양:**
  - `generate.rs`의 `Generator`(:27-61 `Place`, :138-144 `prefill` → `self.model.step(ids)`, 즉 id마다 본체 하나)를 chat과 serve가 씁니다.
  - `bloomery_chat.rs:292`는 `g.prefill(prompt)`를 부르므로 스텝 먹이기입니다.
  - 서버는 `body::prefill`을 클로저로 끼워 넣습니다(`bloomery_serve_ds41.rs:133, 167-175`).
  - `generate_ds41`은 자기 루프를 갖고 `Place`와 그 파싱을 복사해 둡니다(`generate_ds41.rs:205-226, 284-288`). 배치 경로는 `:869`에서 탑니다.
  - `engine.rs:25-56`의 `AnyEngine`을 부르는 곳은 둘뿐인데(`gate_e2e.rs:744-753`, `bin/generate.rs:287-303`), 둘 다 **모델을 다 올린 뒤** `Deepseek2` 팔만 받고 나머지는 오류로 끝냅니다.
  - bits는 같습니다. `gate-gpu-ds41-chat`이 chat id가 generate_ds41의 접두사임을 이미 확인합니다. 결함이 아니라 속도와 이중 소유의 문제입니다.
- **쌓인 경위:**
  - `0399efc`(09-24): Generator와 chat
  - `43cd107`(09-25): 배치 프리필이 generate_ds41과 body에 착륙
  - 서버는 클로저로 따라갔고, chat은 따라가지 못했습니다.
  - `8098a40`(09-23, M3): "open engines by detected architecture"
- **오늘 짓는다면:**
  - 디바이스 크레이트 위에 앱 크레이트를 두고 `Session` 하나를 둡니다.
  - `Session`은 `expect_arch`를 적재 전에 확인하고, 배치 규칙으로 프롬프트를 먹이고, step, 드래프트 정책, placement 이름 파싱을 소유합니다.
  - generate_ds41, chat, serve가 모두 이것을 씁니다. gpu-gates는 게이트만 남습니다.
  - "디바이스 크레이트 이름을 부르지 않는다"는 우회(`generate.rs:8-14`, `bind.rs:10-11`)도 필요 없어집니다.
  - 참고로 mistral.rs는 CLI와 서버가 엔진 크레이트 하나를 공유합니다(`mistralrs-cli/Cargo.toml:23-24`).
- **payoff:**
  - 프롬프트 먹이기의 소유자가 하나가 되고, chat도 배치 경로를 탑니다. AGENTS.md 프리필 문단에 따르면 착륙 시점 pp512 91.2가 스텝 먹이기의 2.97배였고, 그 뒤 빨라진 것은 배치 경로뿐입니다.
  - `AnyEngine` 113줄과 `Place` 사본이 사라집니다.
- **비용과 증명:** M. 이동 부류이고 chat/serve 게이트의 id 비교가 증명합니다.
- **소유와 순서:** aa. faultstep(model.rs의 스텝 경로) 착륙 뒤입니다.

### GG5. MANIFEST.tsv를 여섯 군데서 따로 읽는다. 호스트 전용 oracle 크레이트 하나로 모은다

- **현재 모양:**
  - `lib.rs:330-1470`의 `RefManifest`와 판독기.
    - pre-v2 호환: `:437-440`, `:1283-1291`, `:1348-1357`
    - 전역 암묵 디렉터리 API: `ref_dir()`의 기본값이 V2-Lite CUDA 세트이고 env 둘을 봅니다(`:1104-1119`). 이 API를 부르는 곳은 V2-Lite bin뿐입니다(앞의 grep).
  - 모델 테스트의 독자 파서:
    - `common/manifest.rs:11-53`과 `common/oracle.rs:26-119`(파일명 규칙을 따로 가짐, `:98-107`)
    - `common/v41set.rs:20-60`
    - `ds41_host.rs:106-224`: v41set의 거의 그대로 된 사본입니다. diff해 보면 이름과 상수, 단언 하나만 다릅니다.
    - `ds41_meta.rs:575-724`, `qwen3moe_meta.rs:59-95`
  - 이들이 열 위치를 손으로 박아 둡니다(`f[11]`, `f[13]`, `f[14]`, `len == 15`).
  - Python에도 넷이 더 있습니다(`tools/ref/{check-int-twins,router-hotlist,window-union,router-coverage}.py`).
- **쌓인 경위:** `52837f6`(09-23)이 "one owner of the dump file-name rule"을 선언했지만, 모델 테스트는 gpu-gates에 의존할 수 없습니다(gpu-gates가 model에 의존하므로 순환). 그래서 파일마다 따로 자랐습니다.
- **오늘 짓는다면:** `crates/oracle`(gguf에만 의존)에 v2 매니페스트, 세트 이름, 헤더 검사(build, arch, complete, model_file), 트윈을 둡니다. gpu-gates와 model tests가 둘 다 의존합니다.
- **payoff:** 모델 테스트 쪽 중복 파서 약 600줄[유도]이 사라지고, 열 위치의 소유자가 하나가 됩니다. pre-v2 호환은 v1 세트(V2-Lite CPU `ref`)가 은퇴할 때 함께 삭제합니다.
- **비용과 증명:** M. 이동 부류(출력 줄이 같음)이고 디바이스 코드와 무관합니다.
- **소유:** aa. `ds41_host.rs`와 `union.rs`는 `[03]` 인접입니다.

### GG6. 제품 계약(할당 래칫, 스레드 불변, 프로파일러)이 어떤 제품도 돌지 않는 V2-Lite CPU forward에만 핀돼 있다

- **현재 모양:**
  - `tests/alloc.rs:75-111`은 `arch::deepseek2::forward::step`에 `LIMIT = 660` 래칫을 걸고, PIN이 다섯 개 쌓여 있습니다.
  - `mt.rs`와 `profile.rs`도 같은 V2-Lite forward를 봅니다.
  - `model::{ffn,head,kv,profile}`와 `deepseek2::forward`를 쓰는 곳은 자기 테스트뿐입니다(앞의 grep, bloomery-decode 제외).
  - V4.1 쪽은 할당을 **찍기만** 합니다. `gate_deepseek41_step.rs:76-77`은 `--greedy`에서만 찍는데, 이는 `run-ds41-greedy`로 `gate-*`가 아닙니다. `gate_hybrid`도 판정하지 않고 찍기만 합니다.
  - 살아 있는 게이트가 V2-Lite 파일에 기댑니다.
    - `ops.rs:28-37`(gate-ops)
    - `union.rs:1008-1034`(자기 하드코딩 경로 `/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf`)
    - `gate_q4k_sel`(V2-Lite Q4_K 두 텐서)
- **오늘 짓는다면:**
  - 할당 래칫을 V4.1 호스트 서비스와 스텝에 게이트로 겁니다.
  - union 디스패치에 `BLOOMERY_THREADS` 1/3/32 자식 비교를 둡니다(mt.rs 패턴).
  - V2-Lite CPU forward 게이트 11개(3,716줄, `wc`)는 CPU 엔진과 함께 은퇴합니다. 이것은 리드와 사용자가 결정할 일입니다. bloomery-decode와 exact_ref가 여기에 걸려 있습니다.
- **payoff:** 제품 경로가 지금 없는 래칫을 얻고, 3,716줄이 빠집니다.
- **비용과 증명:** M. 새 게이트는 FAIL-first, 은퇴는 커버리지 변경입니다.
- **소유:** aa. moe.rs와 ops.rs 테스트 내용은 `[03]`입니다.

### GG7. 게이트 등록부가 이름 규칙과 셸 텍스트이고, 약 3.6k줄 도구가 그것을 추론한다. 선언적 표로 바꾼다

- **현재 모양:**
  - 레시피 66개가 `cargo oxide build … && bash tools/gpu-gate.sh` 모양을 공유하고, 그중 56개는 bin 하나에 호출 하나입니다.
  - 달라지는 축:
    - 프로필: 기본 deepseek2 25, deepseek41 30, qwen3moe 11
    - features: gpu 34, deepseek41 31, vision 1
    - `BLOOMERY_GATE_CARD:-any` 14개, DSpark export 주문 7개, solo 속성, ARGS와 env 팔
    - 모두 `just --dump` JSON에서 센 값입니다.
  - 셸로 짠 게이트: chat 3,498자, draft 1,224자, dspark-loop 1,248자(`justfile:912-946`)
  - 이것을 추론하는 도구:
    - `recipes.py` 2,470줄: `:148-410` 셸 토크나이저와 cargo 파서, `:79`의 `DEFAULT_PROFILE = "deepseek2"`(기본 프로필 박스 레시피 109개에 V4.1 레시피도 포함)
    - `gate-batch.sh` 1,060줄: `:262-330` 정규식과 스크립트 전이 걷기로 레인을 정함
    - `check-recipes.sh` 126줄: 손으로 짠 셸에서 생기는 실패 부류 다섯
  - 오분류 사례: ptx-spill의 887 s는 JIT가 3090 락을 기다린 시간이었는데, 러너는 이 레시피를 "디바이스 코드 없음"으로 분류했습니다(`plan-triage.md:238`, `gate-batch.sh:26-27`).
- **오늘 짓는다면:** `gates.toml`(또는 TSV)에 한 행이 게이트 하나입니다. 열은 kind(gate/bench-check/probe), runner, package/bin/test, features, 필수 profile, card(3090/any/both), group, args, env 팔 목록, data(DSPARK), tier(fxtier가 계획한 fixture/real 열)입니다. justfile에는 범용 `gate NAME`만 남기고, recipes.py와 gate-batch는 표를 읽습니다.
  - 참고로 mistral.rs의 등록부는 cargo 테스트 발견과 2행 feature 매트릭스입니다(`.github/workflows/ci_cuda.yaml:24-36, 76-79`, `#![cfg(feature = "cuda")]`).
  - 우리 `.config/nextest.toml`에도 이미 `hw` 프로필이 있는데 게이트 bin들이 이를 우회하고 있습니다.
- **payoff:** 셸 추론에서 오는 부류(레인 오분류, 걷기 누락, `||` 삼킴)가 구조적으로 닫힙니다. bench-check가 묶음에 들어오고, 기본 프로필 V2-Lite가 사라집니다.
- **비용과 증명:** L. `gate-batch --dry-run --classes` 출력이 앞뒤로 같으면 됩니다(구조적 증명).
- **소유와 순서:** aa. boxlease와 fxtier 뒤입니다.

### GG8. gate_load_v41의 기본값과 머리글이 09-24 이전 믿음이다. 두 solo 적재가 목표 아닌 배치를 매 착륙마다 돈다

- **현재 모양:**
  - `gate_load_v41.rs:52`에 "b, the serving target, by default"라고 적혀 있고, 기본값은 `:129`의 `PlanId::B`입니다.
  - 레시피 둘(`justfile:312-320`)은 모두 `BLOOMERY_CARD=both`로 solo 레인에서 돕니다. 중앙값은 163 s와 166 s이고, X 레인은 두 레인이 끝난 뒤 직렬로 돌기 때문에 착륙마다 약 329 s가 직렬로 붙습니다[유도].
  - 두 실행은 중복이 **아닙니다.** mlock이 populate 회귀를 가릴 수 있기 때문입니다(머리글 4항: "Red with `BLOOMERY_HOST_POPULATE=0`").
  - plan (b)는 트리에서 **유일한 두 카드 staging의 런타임 연습**입니다(`workstation.rs:112-122`).
  - T1의 호스트 상주는 `gate-gpu-ds41-faults`(solo, `justfile:876-884`)가 동작으로 핀합니다.
- **오늘 짓는다면:** 머리글과 기본값을 "다중 카드 staging 연습(T2의 전 단계)"으로 고칩니다. 두 레시피는 T2 작업이 시작될 때까지 다중 카드 tier로 옮기고, 산술은 헤더 전용 `placement.rs:659`가 계속 핀합니다.
- **payoff:** crates/gpu나 model이 바뀌는 착륙마다 약 329 s의 직렬 시간[유도]이 빠지고, 낡은 믿음이 사라집니다.
- **비용:** S. 커버리지 변경(날짜 사유)이고 aa 소유입니다.

## 3. 삭제

"아무도 안 부른다"는 판단에 쓴 grep:
- `grep -rn --include='*.rs' -E "\.$k\(|::$k\(" crates --exclude-dir=gpu-gates | grep -v 'pub fn'`
- 아래 커널들에서 결과 0줄

- **탐침 bin 판정표**

| bin | 줄 | 호출자 | 판정·마지막 사용 | 처분 |
|---|---|---|---|---|
| bench_join | 1,835 | `justfile:264-277`, `tools/ref/cstate-ab.sh` | rig-log 09-23 "캡처된 그래프가 호스트를 기다리는 왕복은 22–65 µs다" 절에서 memop 카운터로 결론. hybrid.rs 빌더와 약 150줄 중복(`plan-triage.md:187`) | 삭제(레시피 둘과 cstate-ab의 빌드 포함) |
| h2d_probe | 253 | `justfile:967-968` | 09-25#m2-h2d-a6000: READ_ONLY 등록 거부(801), pinned 26.28 / pageable 21.16 GB/s | 삭제 |
| probe_host_register | 357 | `:979-980` | rig-log, docs, research에 실행 기록 없음(grep 결과 없음) | **가설: 미사용.** M2b가 닫혔는지 리드가 확인 후 삭제 |
| real_x, rawx_floor | 727, 415 | `:328-329`, `:324-325` | rig-log 기록 없음. 질문(raw-x 바닥)이 KERNEL_BAND PIN(`lib.rs:43-48`)으로 닫힘 | GG1과 함께 삭제 |
| generate (V2-Lite CLI) | 566 | `:182-188`, `:276-277` | rig-log 마지막 09-23 | V2-Lite 엔진과 함께 |
| oxart_jit, oxart_ptx | 80, 79 | ptx-scan, sass-scan, ptx-spill | 살아 있음 | **유지** |

- **곧바로 지울 수 있는 것:**
  - gate_p0b와 gate_moe_fused의 `--time` 팔, `time-gpu-p0b`(`:195-196`), `time-gpu-moe`(`:280-281`). 판정이 `docs/gpu-design.md:79, :94`에 있습니다.
  - `lib.rs`의 죽은 pub 함수: `ref_tensor`(:1248), `ref_tensor_logical`(:1292), `topk_ids_logical`(:1955). 외부 호출 0입니다.
  - `ds41_host.rs:106-224`의 `Set` 사본 → `common/v41set.rs`로 교체합니다.
- **V2-Lite 게이트만 쓰는 커널 엔트리(그룹 A):** GG1 이후 gpucore로 넘깁니다.
  - 목록: `q5_0_gemv`, `q5_1_gemv`, `kv_append`(과 `kv_append_pos_buf`), `expert_table`, `weighted_sum`, plain `argmax`, `half_decode`, `flash_latent_split`(호스트 래퍼), `card_sel`(ds2 엔진 한 곳)
  - 이들이 `ptx-shapes.tsv`의 generate_ds41 행에도 들어 있다는 것은, V4.1 제품 CLI의 번들에 실려 있다는 뜻입니다.
  - 증명: 남는 엔트리의 `just ptx-scan`이 같아야 하고, `ptx-shapes.tsv` 행 삭제는 날짜 사유를 붙인 커버리지 편집입니다(표 머리글: 핀 행이 사라지면 빨강).
- **V4.1·Qwen3 게이트가 참조로 쓰는 엔트리(그룹 B, 삭제 대상 아님):** `q3k_gemv_sel`(bench_v41, ds41_load/moe, gate_gemm), `swiglu`(gate_gemm, qwen3moe_experts). §4-3에서 다룹니다.
- **justfile의 0단계:** 리드에게 넘깁니다.
  - `just gate`의 `build-gpu`와 `build-cpu`(`:61-62`, `:331-332`, `:744`), `measure-gpu`와 `measure-cpu`(`:341-345`, rig-log 사용 0), `gate-gpu-p0`(gpu-spike, 0단계 데이터 사용, 64 s)
  - AGENTS.md Known state(09-22)에 따르면 undocumented_unsafe 경고 58개가 전부 0단계 크레이트에 있습니다.

## 4. 가로지르는 패턴

1. **디딤돌 모델이 여전히 "기본값" 자리에 있다.** `ref_dir`의 기본값(`lib.rs:1104-1119`), `V2_LITE_ROUTER`(`:1695-1705`), `DEFAULT_PROFILE`, ptx-spill과 ptx-scan 증명 규약의 `gate_e2e`, V2-Lite 전용 심판, V2-Lite 텐서에 기대는 공유 커널 게이트(p1, p3, q4k_sel, ops, union)가 모두 같은 모양입니다.
2. **띄울 때 짠 꾸러미 조직이 새 계약을 계속 끌어당긴다.** `031b842`, `6123133`이 예입니다.
3. **디바이스 참조 커널이 호스트 참조를 대신한다.** 이런 커널은 모든 제품 번들에 엔트리를 싣고, 가로지르는 계약(fault)을 받을 때마다 같이 고쳐야 합니다.
4. **정확성 bin 안에 시간 모드가 있다.** p0b, moe_fused, p8, generate `--ab`가 그렇습니다. 시간 측정이 모델 적재와 정확성 팔을 임대 창 안에서 부담합니다.
5. **이름 접두사가 게이트 여부를 정한다.** bench-check가 그 때문에 묶음 밖에 있습니다.
6. **PIN 주석이 변경 이력이 됐다.** `gate_p8b.rs`의 `NODES_LAYER1`에 PIN 넷(26 → 25 → 24 → 22), `alloc.rs:75-111`에 다섯이 쌓여 있습니다. AGENTS.md가 허락하는 것은 날짜 붙은 한 줄입니다.
7. **머리글에 이력이 남아 있다.**
   - `lib.rs:8-13`은 "≤ 1e-2"라고 하지만 실제 값은 `:48`의 1e-5입니다.
   - `justfile:74`의 "KERNEL_BAND = 1e-2"
   - `gate_p3.rs` 머리글의 "far tighter than KERNEL_BAND"는 지금은 두 값이 같습니다.
   - `gate_load_v41.rs:52`, `generate.rs:1-4`
8. **공유 파일 형식의 열 번호를 여러 판독기가 손으로 박아 둔다**(GG5).

## 5. 역사처럼 보이지만 남는 것

- **oxart_jit/oxart_ptx와 `ptx.rs`:** 착륙 묶음마다 도는 ptx-spill의 계기입니다.
- **gate_iq와 `iq.rs`:** 엔진 호출은 아직 0입니다(`grep 'iq::|IqKernels|\.iq()'` 결과 없음). 그러나 M4 계획(`plan.md:26`)과 Qwen3 IQ4_XS(`plan-triage.md:104`)를 위해 미리 지은 것입니다.
- **gate_mcol, gate_q4k_sel, gate_vision_encoder:** 살아 있는 커널입니다. 단 q4k_sel의 V2-Lite 텐서는 합성 행으로 바꿔야 합니다.
- **`prompts.rs`, `kld.rs`, `act_rule`, `ik_*`, `rounding`, `hc_host`, `nodes`:** V4.1·Qwen3 게이트가 씁니다.
- **gate_load_v41:** 유일한 다중 카드 런타임 연습이라 opt-in으로 남깁니다.
- **`BLOOMERY_DRAFT=lookup`, `draft.rs`, `gate-gpu-ds41-draft`:** 판정이 아직 열려 있습니다(E5b/E19, `plan-triage.md:146`).
- **gate_e2e, gate_block, V2-Lite GPU 엔진:** 심판 결정(§6의 9) 전까지 남깁니다.

## 6. 오늘 다시 짓는다면

- **구성:**
  - `crates/oracle`: 호스트 전용 세트 판독기
  - `crates/app`: `Session`과 제품 bin 넷
  - `crates/gpu-gates`: 게이트만. 구성은 다음과 같습니다.
    - 호스트 규칙 모듈
    - 계열 게이트(kquant, elem, gemm, iq, flash_gqa): 모델 파일 없이 서빙 모델 모양의 합성 행
    - 사슬 게이트(V4.1은 fixture/real, Qwen3 e2e)
    - 호스트 티어 프로토콜 게이트(stub 티어)
    - 모델 없는 `bench_kernels`
  - `gates.toml`이 등록부이고, `ptx-shapes` 표가 컴파일 모양의 유일한 소유자입니다.
  - 착륙 묶음 = `affected`(크레이트 그래프) × 표 행이고, 레인, 카드, solo, tier는 모두 열에서 읽습니다.
- **이행 경로:** 각 단계는 게이트가 녹색인 채로 끝납니다.
  1. `bench-gpu-v41-check`와 `bench-cpu-v41-host-check`를 `gate-` 이름으로 바꿉니다. 레시피 이름만 바뀌고 출력은 같습니다.
  2. GG2: P4, P5, P6의 모양 단언을 래칫 열로 옮기고 핀 하나로 FAIL-first를 봅니다.
  3. GG3: `bench_kernels`를 분리합니다. q3gemmd `[03]` 뒤입니다.
  4. `gate-gpu-elem`: p4의 공유분(half_encode 포함)을 모델 없이 옮깁니다. `verdict-diff`와 절마다 FAIL-first로 확인합니다.
  5. `gate-gpu-kquant`: p1, p3, p9의 공유분, q4k_sel의 합성화, 1단계 결과를 합칩니다.
  6. gate_hybrid의 stub 팔(`:811-1500`)을 모델 없는 게이트나 hybrid.rs 테스트 `[03]`로 옮깁니다. hostreset과 faultstep 뒤입니다.
     - 겹침 레버의 on/off 핀은 V4.1 레시피로 옮기거나, 판정이 난 뒤 레버를 지웁니다(현재 기록 없음).
  7. P 사다리, gpu-spike, real_x, rawx_floor를 은퇴시킵니다. 게이트마다 날짜 사유를 붙입니다.
  8. 그룹 A 엔트리를 삭제합니다(gpucore, `[03]`). 남은 엔트리의 ptx-scan이 같아야 하고, ptx-shapes 행 삭제에 사유를 붙입니다.
  9. **사용자 결정:** 서빙 모델용 `exact_ref`를 먼저 만들지, ik를 정확도 자로 받아들일지 정합니다. 이 결정이 나야 gate_e2e, gate_block, generate, exact_ref, forced_probe와 V2-Lite GPU 엔진을 은퇴시킬 수 있습니다.
     - 이때 ptx-spill이 스캔하는 바이너리는 gate_e2e 대신 generate_qwen3moe가 되고, AGENTS.md의 ptx-scan 증명 규약도 고쳐야 합니다.
  10. GG6: 제품 경로에 래칫을 걸고, 그 뒤 V2-Lite CPU 게이트를 은퇴시킵니다(사용자 결정).
  11. GG5 oracle 크레이트. v1 세트가 은퇴하면 pre-v2 호환을 지웁니다.
  12. GG4 앱 크레이트.
  13. GG7 표. boxlease와 fxtier 뒤입니다.

## 7. 범위 밖에서 본 개선 지점

- `tools/ref/depth-gpu.sh`(201줄)와 `depth-decode.sh`(71줄)에는 레시피가 없습니다(`justfile`을 grep해도 결과 없음). 그런데 AGENTS.md의 깊이 문단은 depth-decode.sh를 인용합니다. boxlease 소관이고 크기는 S입니다.
- AGENTS.md의 디스패치 A/B 계약은 `just ab-decode`(V2-Lite CPU 디코드)를 가리킵니다. 제품 호스트 티어는 `time-cpu-v41-host`입니다. XS.
- AGENTS.md "Never"의 첫 규칙이 0단계 `q3k-gemv`를 기준으로 쓰여 있습니다. XS.
- `crates/gpu/src/model/probe.rs:50-110` `StepProbe`의 V2-Lite 계측 팔(pad, skip_quant, split_*, keyaxis `76c8673`)은 이미 닫힌 라운드의 계측기입니다. gate_e2e의 4번 팔이 이를 핀합니다. gpucore, M.
- `crates/gpu/src/flash.rs`는 V2-Lite latent flash이면서 V4.1과 Qwen3가 쓰는 공유 f16 도우미(`f32x2_to_f16x2_bits`)의 집이기도 합니다. 도우미를 분리해야 합니다. gpucore, S.
- `crates/gpu/src/fault.rs:71`의 `Q5Quant`는 V2-Lite 전용입니다. `[03]`, XS.
- `tools/ref/ncu-gpu.sh:230`의 `GEMM_BIN` 기본값은 GG3을 따라가야 합니다. XS.
- `crates/model/tests/union.rs:1008-1012`에 V2-Lite 경로가 하드코딩된 폴백이 있어 프로필 말고 소유자가 하나 더 생깁니다. `[03]`, XS.
