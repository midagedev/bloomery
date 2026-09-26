# audit-qwen3 보고 (범위 `qwen3`, 발견 접두 Q3, 전부 [03])

## 0. 읽은 범위

- **읽은 커밋**: `4786c1e`. 파일 편집, 박스 명령, git 상태 변경은 하지 않았습니다. 박스는 exllamav3 참조 트리를 `ssh ws 'grep …'`으로 읽기만 했습니다.
- **전부 읽음**(줄 수는 `wc -l` 실측)
  - `arch/qwen3moe/`: `body.rs` 454, `prefill.rs` 688, `dispatch.rs` 449, `ubatch.rs` 612, `scratch.rs` 290, `taps.rs` 151, `router.rs` 1,411, `experts.rs` 481, `proj.rs` 408, `mod.rs` 24
  - `generate_qwen3moe.rs` 245
  - `model/arch/qwen3moe/{mod 12, host 49}`
  - `docs/arch-split.md` 246
- **머리 문서와 윤곽만 읽음**
  - 커널 쪽: `head_argmax.rs` 408, `flash_gqa.rs` 882, `flash_gqa_prefill.rs` 623, `gemm.rs` 2,415(머리 1–70, 매크로·엔트리 목록, `q4k_dec` 190–260)
  - 모델 쪽: `hparams.rs` 443(1–60), `place.rs` 135
  - 게이트 머리 문서: `gate_qwen3moe_{e2e 1,464, router 1,115(740–870도), down 551, experts 595, flash 1,096, qknorm 162, rope 337}`, `gate_gemm` 2,144, `tests/qwen3moe_placement.rs` 157(1–40)
  - 도구: `depth-qwen3moe.sh` 650(머리 1–120, 함수 목록, `depth-ds41.sh` 651과 diff), `q3pp.py` 556(1–80), `ncu-gpu.sh` 60–110, justfile의 Qwen3 레시피
  - 공유 골격(접점만): `model.rs` 60–260·470–560·822–860·905–1000·1287–1320, `tensor.rs` 150–320, `gpu-gates/src/generate.rs` 60–175
- **읽지 않음**: `gpu-gates/src/qwen3moe.rs` 381, `oracle/qwen3moe.rs` 28, `tests/qwen3moe_meta.rs` 397, `names/roles/kv.rs` 본문, `gemm.rs` 커널 본문 대부분, 프리필 flash·head argmax 커널 본문, 게이트 본문(윤곽 밖), `tools/ref/models/qwen3moe.sh`
- **그 밖에 읽은 것**
  - rig-log: 09-24 E28, 09-25 Qwen3 기준선·q3graph, 09-26 목차
  - `docs/research/q3next-design-report.md`(머리, §3 표, §4)
  - plan·triage의 grep 줄, `~/.cache/bloomery/gate-times.tsv`
- **비행 중 라운드와 만나는 곳**
  - q3swz(`flash_gqa_prefill.rs`): 구조 발견이 없어 겹치지 않습니다.
  - q3gemmd(`gemm.rs`): Q3-5와 `gemm_q5k` 삭제는 그 뒤로 둡니다.
  - boxlease(`depth-*.sh`): Q3-6은 그 뒤입니다.
  - faultstep(`model.rs`): Q3-3·Q3-7은 그 뒤입니다.
  - r8host: 겹치지 않습니다.

## 1. 지금 아는 것 (시드 밖)

1. Qwen3 GPU 경로는 `1fb022a`(09-24 11:06)에서 시작했습니다. 범위 파일을 만진 커밋은 29개입니다.
2. 구동자는 `generate_qwen3moe`와 게이트뿐입니다. 서버는 없습니다(plan-triage.md:46 q3serve 보류). `bloomery-decode.rs:108-110`, `generate.rs:289-292`, `gate_e2e.rs:746-747`은 qwen3moe를 거부합니다.
3. 적재는 `load_full → load_blocks`이고 배치 계획을 읽지 않습니다(`model.rs:497-510`).
4. `Auto` 계획에서 패스는 많아야 한 번입니다(`prefill.rs:401-425`: P ≤ 8이면 패스 하나, 아니면 ubatch들 뒤 꼬리 ≤ 8을 패스 하나).
5. 실측 수치
   - 캡처된 8토큰 패스 20.75 ms(`6283084`, rig-log 09-25 q3graph 절)
   - 디코드 깊이 6: 205.97 tok/s(09-24 E28), q3unfold 뒤 211.5
   - pp512 7,706, pp4096 8,818(`ccdd3dc`)
6. 8 컷의 근거는 `dfa8b5d` 메시지의 유도 "equal at T = 8"이었습니다. 그 뒤 GEMM 경로만 네 라운드(q3pflash, q3ubatch, q3router, q3gemmb) 빨라졌고, T ≤ 8의 GEMM 비용은 잰 적이 없습니다.
7. 참조 엔진 셋 모두 8에서 **연산 하나 안에서** 가릅니다.
   - exllamav3: `modules/mlp.py:740`, `block_sparse_mlp.py:942`, `exllamav3_ext/libtorch/mlp.h:42`(`Graph graph_bszN[MAX_BSZN]`, 모듈 단위)
   - mistral.rs: `gguf/fast_mmvq.rs:53`, `mistralrs-quant/src/lib.rs:2236,2321,2439`
   - ik: `ggml-cuda/mmvq.cuh:10`, `ggml-cuda.cu:2701,2881`
   - 층 몸체를 둘 두는 곳은 없습니다.
8. 추측 디코딩
   - `qwen3spec`(m행 검증 그래프 + EAGLE-3)가 계획에 있습니다(plan-triage.md:213).
   - 골격에는 m = 2인 `step_pair`(`model.rs:917`)가 이미 있습니다.
   - `Head::with_m`(`head.rs:102`)은 `gate_mcol.rs:735,750`만 씁니다.
9. 프롬프트 흐름(q3next 보고 §4, 유도): 벽시계 = 호스트 직렬 앞단 4.5–12 ms(P 4096) + 카드 약 530 ms(합)입니다. 이 표의 합은 실측 546.9 ms와 +2.6 ms 차로 맞았습니다.
10. 착륙 묶음은 `just affected` 전부입니다(plan-triage.md:233). `crates/gpu`를 바꾸면 Qwen3 GPU 게이트가 전부 돕니다.
11. 다음 Qwen은 `qwen35moe`(Qwen3.6-35B-A3B)입니다: GQA 16/2 × 256, GDN 30층(plan.md:137, `linear-attn.md:121,129`). 지금 커널에 박힌 상수 HEAD 128, GROUP 8, N_EXPERT 128/N_USED 8, NORM_K 2048(`body.rs:120-131`, `router.rs:72-79,135`)과 맞지 않습니다.

## 2. 구조 발견 (이득/비용 순)

### Q3-1. 게이트가 두 번 돈다: 우산 레시피, 스칼라 팔 전체 재실행, 엉뚱한 바이너리에 든 계약 [03]

**어디**
- `justfile:606-607` `gate-gpu-qwen3moe-kernels`가 개별 여섯(`:580-604`)을 다시 짓고 차례로 돕니다.
- `justfile:612`가 e2e를 MMA로 한 번, `BLOOMERY_GQA_MMA=0`으로 한 번 더 돕니다.
- ubatch 라우터 계약이 둘로 갈려 있습니다: 비트는 `gate_gemm`의 `router` 케이스(머리 문서 62–72), 폴트는 `gate_qwen3moe_router.rs:797-826`.
- `qwen3moe_head_q6k_argmax`는 `gate_qwen3moe_down`의 4번 항에 들어 있습니다.
- 커널 하나(`head_norm_neox_append`)를 qknorm·rope 두 바이너리가 핀합니다.

**실측**: `gate-times.tsv`의 세 묶음(09-26 15:13–15:20, 15:59–16:05, 19:43–19:51, 모두 A6000)

| 묶음 | 개별 여섯의 합 | 우산 | e2e(두 실행 합) |
|---|---:|---:|---:|
| 15:13–15:20 | 93 s | 115 s | 194 s |
| 15:59–16:05 | 92 s | 88 s | 201 s |
| 19:43–19:51 | 155 s | 131 s | 161 s |

- 같은 묶음 안에 개별 여섯과 우산이 둘 다 있습니다.
- `tools/recipes.py`에 우산을 거르는 규칙이 없습니다(`grep -n -i 'umbrella\|subset\|dedup\|duplicate\|same bin\|superset\|covers'` → 무관한 1300행 하나).
- 이런 우산은 저장소에 이것 하나입니다.

**어떻게 쌓였나**
- 우산은 phase 2(`43944e4`)의 편의였습니다.
- 스칼라 재실행은 phase 2의 두 팔 규칙이었습니다. `dfa8b5d`부터 스칼라 팔은 (u) 절의 자로 쓰입니다(`gate_qwen3moe_e2e.rs:1146-1150` `set_flash_mma`).

**오늘 짓는다면**
- 계약을 소유한 모듈마다 게이트 바이너리 하나를 두고 우산은 두지 않습니다.
- 스칼라 GQA 커널은 운영 경로가 아니라 자입니다(AGENTS 「Performance first」). 선택자 둘(환경 레버 `flash_gqa.rs:106-114`·`body.rs:369`와 API `taps.rs:96`) 가운데 API 하나만 남깁니다.
- 레버의 판정 기록은 rig-log에 없습니다(`GQA_MMA|gqa_flash_seg_mma|gqa_mma` grep 빈 결과).
- 라우터 엔트리는 전부 router 게이트, head argmax는 자기 절, qknorm·rope는 한 바이너리의 두 절로 둡니다.

**스칼라 재실행이 핀하는 것**
- 스칼라 flash로 도는 (f)(c)(g)(r)(p)(q) 절입니다.
- 커널 자체는 `gate_qwen3moe_flash`가 ik `fa-L`에 대고 "in both passes"로 핀합니다. 스칼라 커널은 (u) 절 안에서도 eager로 돕니다.
- 그래서 재실행을 빼는 것은 사슬 수준의 커버리지 변경이고, 날짜 붙은 사유가 필요합니다.

**이득**
- 우산을 지우면 묶음당 88–131 s [실측]를 덜고 커버리지 변화는 없습니다.
- 재실행을 빼면 e2e 161–201 s 가운데 한 실행 몫을 덥니다. 한 실행 몫은 따로 잰 적이 없습니다.

**비용과 증명**
- 레시피 삭제: `just check-recipes`, `just affected` 목록이 한 줄 줄어듭니다.
- 레버 제거: 이동 클래스입니다. `ptx-scan` 동일, MMA 실행의 덤프 md5 동일. timed A/B는 필요 없습니다.

**소유와 순서**: [03], justfile만 만지므로 순서 제약이 없습니다.

### Q3-2. 엔진이 쓰지 않는 배치 계획과 호스트 층 명세, 그리고 이미 어긋난 계획 산술 [03]

**어디**
- `model/arch/qwen3moe/host.rs`(49줄)는 호출자가 없습니다.
  - `grep -rn 'qwen3moe::host\|qwen3moe::{[^}]*host' crates --include='*.rs'` → 빈 결과, rc 1.
  - `host::layer` 호출자 넷(`gpu-deepseek41/src/chain/ffn.rs:1530`, `gate_deepseek41_chain_ffn.rs:1642`, `tests/ds41_host.rs:481`, `tests/union.rs:384,639,875`)은 모두 `arch::deepseek41::host`를 들입니다.
- `place.rs`(135)와 `kv.rs`(63)의 호출자는 `tests/qwen3moe_placement.rs`(157) 하나입니다.
- 엔진은 `load_full → load_blocks`로 적재합니다(`model.rs:497-510`). 층 일부 적재를 거부하고(`body.rs:318-326`) 호스트 티어가 없습니다(`experts.rs:19-23`).

**어긋남**: 트리아지 원문(plan-triage.md:244) "`SCRATCH = 64 MiB`에 GEMM 프리필 아레나(U 4096이면 944 MB)가 빠져 Qwen3 whole-card 계획의 마진이 조용히 흡수". `workstation.rs:49`가 그 줄입니다. 배치 게이트가 초록인 것은 엔진이 잡는 바이트에 대한 진술이 아닙니다.

**어떻게 쌓였나**: phase 1(`1fb022a`)이 공유 배치 서비스가 둘째 아키텍처를 받는지 보이려고 지었습니다. 뒤 라운드는 `load_full`로 적재했고, 아레나(`6283084`, `dfa8b5d`, `ded6c51`)를 계획에 넣지 않았습니다.

**오늘 짓는다면**: 한 카드에 통째로 드는 모델에는 배치 코드를 두지 않습니다. 둔다면 계획이 몸체의 바이트 공식 `resident_bytes()`(`body.rs:445-453`, 아레나와 이미지를 이미 셉니다)를 소비해야 소유자가 하나가 됩니다. 이 대응을 가진 참조는 확인하지 않았습니다.

**이득**
- 404줄 [wc -l] 합, 레시피 하나, 틀린 초록 하나가 사라집니다. 게이트 시간은 1 s라 시간 이득은 거의 없습니다.
- `roles.rs`는 메타 게이트가 쓰므로 남습니다(`tests/qwen3moe_meta.rs:29`).

**비용과 증명**: 크기 S. `just check`, `gate-qwen3moe-meta`, `check-arch`. 커버리지 변화는 배치 게이트 제거이고, 사유는 "엔진이 계획을 소비하지 않고, 계획이 944 MB를 빠뜨렸다"입니다.

**순서**: 독립입니다.

### Q3-3. m행 디코드 패스의 소유자가 둘이고, Qwen3 쪽은 프롬프트 모양이다 [03, 골격은 aa와 함께]

**지금 모양**
- **Qwen3**: `Prefill`(`prefill.rs:54-367`)
  - m = 1..8마다 캡처 그래프 하나(`:72`)
  - ctx/8 블록짜리 프롬프트 이미지(`:138,151`)와 슬롯 스테이징(`:240-269`)
  - 사슬은 `dispatch::layer`의 m행 인스턴스(`dispatch.rs:152-164`)
  - 머리는 마지막 행을 복사한 1행 머리(`:168-188`)
- **골격**: `step_pair`(`model.rs:917-957`)
  - m = 2 고정, `pair_graph`/`pair_head`(`:347,354`), `Chain::Pair`(`hybrid.rs:1350`)
  - 트레이트의 pair 메서드 셋은 기본 구현이 거부입니다.
  - V4.1 드래프트가 이 길로 검증합니다.
- **셋째 조각**: `Head::with_m`, 엔진은 쓰지 않습니다.
- **다중 패스는 강제 팔 전용**: 여러 패스를 도는 것은 강제 `PrefillPath::Pass`뿐입니다. 호출자는 `gate_qwen3moe_e2e.rs:901,906,942`와 `generate_qwen3moe.rs:105`입니다.
- **계획과 모양이 어긋남**: 계획된 `qwen3spec`는 행마다 argmax가 필요한데, 지금 패스의 머리는 마지막 행 하나입니다.

**어떻게 쌓였나**
- 패스는 먼저 프롬프트용으로 지어졌습니다(`3c00e39` eager, `6283084` 캡처).
- 2.5시간 뒤 `dfa8b5d`가 P ≥ 9를 GEMM으로 가져갔습니다.
- pair는 V4.1 쪽에서 따로 골격에 들어갔습니다.

**오늘 짓는다면**
- 골격이 `step_rows(ids) -> [u32; m]`(1 ≤ m ≤ 8)을 소유합니다.
  - m마다 캡처 그래프 하나(`Chain::Rows(m)`)
  - 재생 입력은 고정 슬롯 하나(`ids`, `pos0` — Q3-4)
  - m행 머리는 행마다 1열 본체로 계산해 1행과 비트가 같게 합니다.
- 몸체는 `enqueue_rows(m)`와 `refresh_rows`만 줍니다.
- 인스턴스: V4.1 pair = m 2, Qwen3 짧은 프롬프트와 꼬리 = m ≤ 8, `qwen3spec` = 1 + k.
- 8 컷은 참조 엔진 셋과 같습니다(§1-7).

**이득**
- `prefill.rs:54-367`(314줄)의 이미지·슬롯·청크 기계와 골격의 pair 전용 필드가 한 경로로 합쳐집니다.
- 다중 패스 팔과 그 게이트 절(e2e (p)의 연결 프롬프트, `:821-933`)이 사라집니다.
- `qwen3spec`를 세 번째 구현 없이 지을 수 있습니다.
- k = 3 드래프트(plan-triage.md:235)의 검증 폭이 pair 고정 폭 밖이라는 문제의 문도 열립니다. 지금 루프가 그 k를 어떻게 검증하는지는 읽지 않았습니다.

**비용과 증명**: 크기 L, 세 단계입니다.
1. **Qwen3 안([03])**: 다중 패스 팔을 지우고 이미지를 슬롯 하나로 줄입니다. 이동 클래스입니다: `ptx-scan` 동일, `NODES_PASS_1` 601·`NODES_PASS_M` 673 불변(`gate_qwen3moe_e2e.rs:232,245`), eager = replay, 덤프 md5 동일. 연결 프롬프트 절 제거는 커버리지 변경이고 사유는 "`Auto`가 그 모양을 만들지 않는다"입니다.
2. **골격(aa)**: `Chain::Rows(m)`을 두고 pair를 m 2로 옮깁니다. pair 노드 핀, `gate-gpu-ds41-draft`·`-dspark-loop` 토큰 동일, 드래프트 팔 같은 임대 A/B 1회가 필요합니다.
3. **Qwen3를 행 경로로** 옮깁니다.

**선택 측정(선행 조건 아님)**: 같은 바이너리로 `--prefill pass|gemm`을 P = 1..8에서 잽니다. GEMM이 자 안이면 짧은 프롬프트도 GEMM으로 보내고, 행 경로는 검증 전용이 됩니다.

**순서**: faultstep 뒤, `model.rs`를 만지는 라운드는 한 시점에 하나입니다.

### Q3-4. 입력 이미지 셋과 토큰마다의 호스트 rope: 흐름의 직렬 앞단 [03]

**지금 모양**
- 같은 넷(토큰, 위치, 키 수, rope 행 128 f32)을 세 배치로 싣습니다.
  - `StepParams`(`scratch.rs:14-19,101-162`)
  - 패스 블록(`prefill.rs:54-63,164-211`)
  - `UbImage`(`ubatch.rs:197-301`)
- 셋 다 위치마다 호스트에서 `rope.push`를 부릅니다(`body.rs:388`, `prefill.rs:195`, `ubatch.rs:261`).
- `UbImage::write`는 프롬프트마다 ctx 크기 전체를 채워 통째로 복사합니다(`:255,269`). `n_keys`는 늘 `pos + 1`입니다(`:259`).

**자원 시간선**(q3next 보고 §4, 유도)

| 자원 | P 4096 | P 512 |
|---|---:|---:|
| 호스트 직렬 앞단 | 4.5–12 ms | 0.6–1.6 ms |
| 카드 | ≈530–536 ms | ≈78 ms |
| PCIe | 0.1–0.2 ms(앞단 안) | ≈0.05 ms |

벽시계는 앞단과 카드의 합입니다. 앞단은 첫 런치 전이라 그늘이 없습니다.

**어떻게 쌓였나**: 디코드의 `StepParams`(`a2ec74b`)를 패스(`6283084`)와 ubatch(`dfa8b5d`)가 각자 베꼈습니다. 트리아지는 레버로만 적었습니다(:245 "E 카드 rope 표(S)", :213 K5, :244).

**오늘 짓는다면**
- rope 표를 적재 때 카드에 둡니다. 같은 호스트 함수로 계산하므로 비트가 같습니다. 크기는 ctx 4096에 2 MiB, 32768에 16 MiB [유도]입니다.
- 커널이 pos로 표를 읽고, `n_keys`는 카드에서 냅니다.
- 입력은 세 경로 모두 `ids + pos0` 한 모양이 됩니다. Q3-3의 슬롯도 이것입니다.

**이득**
- 앞단 제거로 P 4096에서 +0.8…+2.2 % [유도]
- 이미지 배치 셋과 rope 루프 셋이 하나로
- ctx 32k에서 프롬프트 길이와 무관하게 드는 약 17 MB memset·복사(설계 보고)가 사라집니다.

**비용과 증명**
- 크기 M. `head_norm_neox_append` PTX가 움직입니다(호출자는 `dispatch.rs:298`, `ubatch.rs:491`). ptx-spill 재핀이 필요합니다.
- 비트: rope 게이트의 전사 비트 절, e2e 덤프 md5.
- timed: 효과가 네 바퀴 자(±1.0 %)에 걸치므로 여섯 바퀴 카드를 쓰거나, 속도 주장 없이 구조 이득으로 착륙합니다.

**순서**: 독립입니다. Q3-3 앞에 두면 슬롯이 먼저 단순해집니다.

### Q3-5. 활성값 타입이 옛 믿음을 담고 있다: m마다 한 벌, 세 순열 전부 기록 [03, 공유 타입은 aa와 함께]

**지금 모양**
- `Q8Act`(`tensor.rs:218-226`)는 m ≤ 8을 할당에 굳히고(`:238`) q3·q4·q6 순열을 다 가집니다.
- 패스 아레나는 활성 자리 넷마다 `Vec<Q8Act>`를 둡니다(`scratch.rs:65-96,204-208,245-247`). Q8Act 32개 × 버퍼 다섯 [코드에서 셈]입니다.
- 커널은 `act.m()`으로 열 수를 읽습니다(`proj.rs:305`, `experts.rs:290`).
- GEMM은 쌍둥이 `GemmAct`(`gemm.rs:1807`)를 쓰고 q6 순열만 읽습니다(`gemm.rs:55-58`). Qwen3 디코드는 q4·q6만 읽습니다.

**비용**(q3next 표, P 4096)
- 층당 쓰기 가운데 읽는 사람이 없는 것: `swiglu_quant` "q3·q4 면 50 MB", quantize ×3 "67 MB". 트리아지 :245는 112 MB로 적었습니다.
- 48층이면 프롬프트당 약 5.6 GB 쓰기, 8–9 ms ≈ 1.5–1.7 % [유도]입니다.

**어떻게 쌓였나**: stage 0의 활성 타입을 phase 1–3이 그대로 썼습니다. 패스(`3c00e39`)가 m마다 한 벌을 늘렸고, `7fd6f2a`가 4096열 쌍둥이를 만들었습니다.

**오늘 짓는다면**
- 활성값 하나를 (면 집합, 용량 열, k)로 둡니다.
- 면 집합은 로드 때 소비자 가중치 타입(`body.rs:57-58` `v_ty/down_ty`)에서 정합니다. GEMM 입력은 q6, V4.1 Q3_K 자리는 q3입니다.
- 열 수는 런치 인자로 넘깁니다.
- 같은 부류가 V4.1에도 있습니다(plan-triage.md:235 "Q8Act 24개 → 둘").

**이득**
- Vec 넷이 활성 넷으로, 쌍둥이가 하나로(트리아지 :221)
- P 4096 쓰기 약 5.6 GB 감소 [유도]
- m ≤ 8 타입 상한이 풀립니다(:190, :281).

**비용과 증명**
- 크기 M–L. 양자화기(`lib.rs:2580` 부근, `fused.rs`의 `norm_quant`는 [03] 소유)는 커널 아홉이 공유합니다(`6123133`). 그래서 affected가 GPU 게이트 전부입니다.
- 읽히는 면은 비트 동일이어야 합니다: `gate_gemm` `y_fnv` 328/328, e2e 덤프 md5, V2-Lite·V4.1 비트 게이트, ptx-spill 재핀.
- A/B는 네 바퀴 한 번입니다.

**순서**: q3gemmd 뒤입니다.

### Q3-6. depth 러너 둘이 같은 뼈대를 따로 가진다 [03, ds41 러너는 aa]

**지금 모양**
- `depth-qwen3moe.sh` 650줄과 `depth-ds41.sh` 651줄이 같은 이름의 함수 열둘(`tree_line`, `ref_cmd`, `ref_arm`, `ratio_table` 등)을 각자 정의합니다.
- 비주석 코드 397·438줄 가운데 172개 고유 줄이 글자 그대로 같습니다(`comm -12`, 앞 공백 제거).
- 앞 파일은 스스로 "depth-ds41.sh's shape"라고 적습니다(`:15`).

**어떻게 쌓였나**: `1785648`이 본떴고, `9048d04`, `5cdb7a5`, `ed3d6f3`, `66c121a`가 두 파일에 따로 얹었습니다.

**오늘 짓는다면**: 러너 핵심 하나(팔 해석, 회전, 참조 팔, pp 팔, 증인, 비율 표)를 두고, 이미 있는 `tools/ref/models/*.sh`가 "우리 팔"의 명령줄과 파서를 줍니다.

**이득**: 약 170줄과 두 번 적은 규약이 한 곳으로 모입니다.

**비용과 증명**: 크기 M.
- `BLOOMERY_BOX_ENV=BLOOMERY_DRY=1`로 고정 팔 목록의 명령줄이 바이트 동일한지 봅니다.
- 저장된 시팅 로그를 새 파서로 다시 읽어 비율 표가 같은지 봅니다.
- shellcheck 수가 오르지 않아야 합니다. timed 측정은 필요 없습니다.

**순서**: boxlease 뒤입니다.

### Q3-7. 프롬프트 표면이 아키텍처마다 다르고, Qwen3 CLI는 공유 생성기를 쓰지 않는다 [03, `Engine`·`generate.rs`는 aa]

**지금 모양**
- `Engine`에 프롬프트 메서드가 없습니다.
- 모양이 둘입니다: Qwen3 `prefill_with(tokens, PrefillPath)`(`prefill.rs:554`) 대 V4.1 `body::prefill_with(m, ids, Some(rows))`(`generate_ds41.rs:1148`).
- `Generator::prefill`은 id마다 스텝입니다(`generate.rs:141-143`).
- `generate_qwen3moe`는 자기 파서(`:78-91`)와 행 형식(`:203-241`)을 가집니다.

**오늘 짓는다면**
- 공유 표면에 `prompt(ids)` 하나를 두고, 몸체가 ubatch와 행 경로 가운데 고릅니다.
- 두 generate 바이너리는 같은 생성기의 인스턴스가 됩니다.
- 강제 경로(`Pass`/`Gemm`)는 게이트 API로만 남깁니다.

**이득**: Qwen3 서버가 생길 때 다시 짓지 않아도 되고, 러너가 읽는 행 형식의 소유자가 하나가 됩니다.

**비용과 증명**: S–M. 출력 줄(`load`, `step 0`, `time prompt`, `SMOKE`)이 같아야 합니다.

**순서**: Q3-3과 faultstep 뒤입니다.

### Q3-8. ncu 도구가 Rust 소스를 읽어 런치 순서를 다시 유도한다 [03, `ds41pp.py`는 aa]

**지금 모양**
- `q3pp.py plan`은 `body.rs`·`prefill.rs`·`ubatch.rs`의 줄을 읽어 launch skip을 유도하고, 줄이 없으면 이름 붙은 거부를 냅니다(`q3pp.py:17-35`, `ncu-gpu.sh:65-83`, `0c0603d`).
- 런치 순서라는 사실의 소유자가 Rust와 Python 둘입니다.
- NVTX는 엔진·도구·ik·mistral.rs 어디에도 없습니다(네 곳 grep 모두 빈 결과).

**오늘 짓는다면**: 엔진이 자기 순서를 냅니다. 프롬프트 경로의 (ubatch, 층) NVTX 범위와 ncu `--nvtx-include` 조합, 또는 바이너리가 찍는 `--launch-plan` 가운데 하나입니다. 참조 셋은 이런 도구가 없으니 우리 쪽 설계입니다.

**이득**: `plan` 유도 부분이 사라지고, Rust 줄 이동이 도구를 깨뜨리는 결합이 풀립니다. HMMA 개수 증명(`ncu-gpu.sh` 약 86행, 34,078,720 @ P 4096 [유도])은 남깁니다.

**비용과 증명**: S–M. 전후 같은 런치를 고르는지(HMMA 수 동일)로 봅니다. NVTX를 붙이는 비용은 재지 않았습니다. 가설로, 디코드 스텝에는 넣지 않습니다.

## 3. 삭제

1. **`qwen3moe_router` 엔트리 + `RouterKernels::enqueue` + `ROUTER_THREADS`**(`router.rs:486-533,1170-1200,82`)
   - 엔진 호출 0입니다. `router\.enqueue(` grep 결과가 없고, 엔진은 `dispatch.rs:370,390`, `ubatch.rs:566`만 부릅니다. 호출은 `gate_qwen3moe_router.rs:111,1021`뿐입니다.
   - `qwen3moe_router_route`(n = 1, `:1024-1070`)가 같은 `route_warp`를 돕니다.
   - `ptx-shapes.tsv`에서 한 행이 빠지고, 나머지 행이 같으면 증명입니다.
2. **`gemm_q5k`**(`gemm.rs:1508-1578`)
   - 엔진 호출 0(`ubatch.rs:97-98,530,579`는 Q4K·Q6K만)이고, plan·triage에 이름이 없습니다.
   - 공유 매크로 변경이 이 엔트리의 PTX까지 움직입니다(`ccdd3dc` "moved four entries, not two").
   - 조건: V4.1 q5_K 2텐서(arch-split.md:46)가 IMMA shadow(사용자 결정 대기)를 타면 필요해집니다. 그 결정 뒤에 지웁니다.
   - 커버리지: `gate_gemm` Q5_K 합성 스택 32회 [유도: 머리 문서의 T 8종 × 라우팅 4종]가 빠집니다.
3. **`enqueue_combine` 1토큰 래퍼**(`experts.rs:409-430`): 호출자는 `gate_qwen3moe_experts.rs:551,570`뿐입니다.
4. **`RouterOut::new`**(`router.rs:1091-1093`): qwen3moe 호출자는 게이트뿐(`gate_qwen3moe_router.rs:485,486,710,891`)입니다.
5. **`qwen3moe/host.rs`**: Q3-2 참조.
6. **우산 레시피 `justfile:606-607`**: Q3-1 참조.
7. **`BLOOMERY_GQA_MMA` 환경 레버**: Q3-1 참조. 판정 기록이 없습니다.
8. **`probs` 쓰기**(선택, 낮은 우선)
   - 엔진은 읽지 않고, 게이트가 소프트맥스를 핀하는 데 씁니다.
   - 비용은 P 4096에 96 MiB, 약 0.15 ms [유도]로 거의 0입니다.
   - 지우면 소프트맥스 핀이 선택된 8개의 weights로 좁아집니다(커버리지 변경).
9. **`gate_qwen3moe_experts.rs`의 `PIN(2026-09-25): removed — …` 줄**: PIN 예외는 재핀 상수의 귀속용인데, 이 줄은 지운 절의 이력을 싣습니다. 커밋 메시지로 옮깁니다.
10. **Q3-3 1단계**: `PrefillPath::Pass`의 다중 패스 팔과 e2e (p)의 연결 프롬프트 절.

## 4. 다른 곳에서도 되풀이될 모양

1. 우산 레시피, 그리고 비기본 팔로 게이트 전체를 다시 돌리기. V2-Lite `gate-gpu-e2e`도 두 flash 팔을 돕니다(AGENTS 레버 목록).
2. 한 사실에 선택자 둘(환경 레버 + setter): `GQA_MMA`/`set_flash_mma`, `QWEN3_UBATCH`/`set_ubatch`. 뒤의 것은 §5에 따라 남깁니다.
3. 할당에 굳힌 열 수 때문에 m마다 버퍼를 한 벌씩 둡니다: Qwen3 32개, V4.1 24개(:235), `RouterOut::{with_tokens,for_ubatch}`.
4. 도구가 엔진 사실을 소스에서 다시 유도합니다: `q3pp.py`, `ds41pp.py`.
5. 라운드가 만진 게이트에 새 계약이 끼어듭니다: ubatch 라우터 → `gate_gemm`, head argmax → `gate_down`.
6. 엔진이 안 쓰는 계획, 그리고 엔진 할당에서 어긋난 계획 산술이 있습니다: SCRATCH 대 944 MB. V4.1 바이트 공식이 여러 벌인 것(:235)도 같은 부류입니다.
7. 형상 커널이 아키텍처 이름을 달고 `arch/` 아래 있습니다(`qwen3moe_head_q6k_argmax`, `_token_major`, `_combine`, `_gate_up_swiglu_q4k` — arch-split.md:19-21 원칙 2와 어긋남). 반대로 공유 커널이 다른 아키텍처의 규약을 들고 있습니다(`q6k_sel.rs:82` `HOST`). `qwen35moe`가 오면 재사용 단위는 형상 커널인데, 지금은 상수가 박혀 있습니다.
8. PIN 예외로 이력을 싣습니다.

## 5. 역사처럼 보이지만 남는 것

- **디코드 flash(split-K)와 프리필 flash(FA2 모양) 둘**: T 1에서 split-K가 없으면 깊이에서 kv 헤드 4블록뿐입니다. llama.cpp의 vec/mma 분업과 같은 이유입니다.
- **8이라는 컷 자체**: 참조 셋 모두 8입니다.
- **`gemm_q3k`**: B4 결정(plan-triage.md:227)과 prefill-lit2가 이름으로 부릅니다.
- **스칼라 GQA 커널**: (u) 절의 자입니다.
- **`PrefillPlan`**(ubatch 계획)과 **`run_rows`**(`model.rs:1287-1316`, V4.1과 공유하는 작은 골격).
- **`BLOOMERY_QWEN3_UBATCH`**: A/B 팔이 아니라 메모리 손잡이입니다(U 4096 아레나 943.6 MB). 토큰 비트는 U와 무관합니다(e2e (w)).
- **GEMM 가족의 매크로 뼈대**(`gemm.rs:205-905`): llama.cpp `mmq.cuh`의 형식별 템플릿과 같은 모양입니다. Q4_K만 가중치를 스테이징하는 것은 형식별 최적화이지 두 번째 설계가 아닙니다.
- **`Kq`와 층별 `v_ty`/`down_ty`**(`body.rs:33-59`): 파일의 사실(24층씩 Q4_K/Q6_K)에 맞는 타입입니다.
- **`dispatch::layer` 하나가 m 1과 m ≤ 8을 같이 도는 것**: 좋은 모양입니다.
- **m 1에서 남긴 접기 둘**(라우터 안 norm, head argmax): `bf458db` 실측 판정입니다.

## 6. 오늘 이 범위를 다시 짓는다면

**배치**
- **골격(aa)**
  - `step`(m 1, 그래프)
  - `step_rows`(1 ≤ m ≤ 8, m별 그래프, m행 머리 — V4.1 pair는 m 2)
  - `prompt`(ubatch들, eager)
  - pair, host, placed는 능력 트레이트로 둡니다.
- **`arch/qwen3moe/`**
  - `plan.rs`: 핀, `LayerNames`, `Kq`, 자리별 활성 면 집합
  - `arena.rs`: max(8, U)행 아레나 하나, `Act{planes, cap, k}`, `KvPlanes`, 카드 rope 표
  - `layer.rs`: 층 함수 하나를 두고 연산마다 행 수로 고릅니다. 1행은 디코드 접기, ≤ 8행은 열마다 1열 본체로 도는 gemv 가족, 그보다 많으면 GEMM과 프리필 flash입니다. exllamav3, mistral.rs, ik와 같은 자리에서 가릅니다.
  - `router.rs`: 엔트리 셋(norm-fused, fused, logits+route)
- **형상 커널**은 크레이트 루트로 옮기고 상수를 인자로 받게 합니다.
- **게이트**는 모듈 계약마다 하나, e2e는 한 번만 돕니다.
- **도구**: depth 러너 핵심 하나 + 모델 프로필, ncu는 엔진이 내는 순서를 씁니다.

**흐름**: 프롬프트는 [ids H2D, P 4096에 16 KB] → 카드 사슬 약 530 ms → D2H 한 번이고, 호스트 앞단은 거의 0입니다.

**옮기는 순서**(각 단계 초록)
1. Q3-1 게이트 정리: 레시피·절 이동, 레버 제거(이동 클래스). 스칼라 재실행 제거는 날짜 붙은 커버리지 변경입니다.
2. §3의 삭제 1·3·4·5: ptx-scan은 한 행 적고 나머지는 동일, `check`, 메타 게이트.
3. Q3-4 카드 rope 표와 ids 입력: rope PTX 이동, 비트 동일, 재핀.
4. Q3-5 자리별 활성 면(q3gemmd 뒤): 읽는 면 비트 동일, 공유 게이트 전부.
5. Q3-3: Qwen3 슬롯 하나로(이동) → 골격 `step_rows`와 pair = m 2(faultstep 뒤, 드래프트 A/B 1회) → Qwen3 행 경로.
6. (선택) 층 함수 하나로: 각 단계의 커널열이 불변이면 이동 클래스입니다(ptx-scan, 노드 수, 덤프 md5).
7. 도구: Q3-6(boxlease 뒤), Q3-8, Q3-7.

## 7. 범위 밖에서 본 것

- **`gpu-gates/src/generate.rs:141-143` `Generator::prefill`이 id마다 스텝입니다.** `bloomery_chat.rs:292`가 V4.1 프롬프트를 이 길로 먹입니다. 배치 피드로 바꾸면 되고, 서버 쪽 피드는 따로인지 확인이 필요합니다(aa, S).
- **`model.rs` `ChainBody`**: 60–260에서 센 메서드 20개 가운데 기본 구현이 거부인 것이 7개입니다(hybrid 둘, placed, pair 셋, rollback). 능력 트레이트로 바꾸면 빠진 능력이 컴파일 오류가 됩니다(aa, M).
- **`hybrid.rs:1350` `enum Chain`**: 골격의 재생 선택자(`model.rs:972-1000`)인데 호스트 티어 모듈에 있습니다(XS).
- **`q6k_sel.rs:82`**: 호스트 티어 없는 사슬에서도 `HOST`를 예외로 둡니다. 트리아지에 있고 [03] 몫입니다.
- **`ds41pp.py`**: Q3-8과 같은 부류(aa).
- **`depth-ds41.sh`**: Q3-6의 짝(aa).
- **`tensor.rs:231` `Q8Act::new`**: "k = 2048, stage-0" 생성자입니다. 호출자는 확인하지 않았습니다(XS).
