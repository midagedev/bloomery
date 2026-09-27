# Qwen3.8-Flash-Next(`qwen4exp`) — Qwen 리더 팔의 변형으로 붙이기

라운드 `qwen4arch`(재구성 파동 3, 읽기 전용 설계, 2026-09-27)의 메모를 보고한 그대로 두고, 위에 리드의 처분을 적는다.
메모의 수치는 출처를 이름으로 대는 것 말고는 전부 [유도]다(코드를 돌리지 않았다). 인용한 URL은 라운드가 curl로 받아
스크래치에 둔 파일에서 grep한 줄이다(WebFetch 요약이 아니다).

## 리드의 처분 (2026-09-27)

- **Q1 파일: 있는 것을 쓴다, 검증은 리드가.** unsloth UD-Q4_K_XL 네 샤드(111,334,654,784 B)가 `/models/Qwen3.8-Flash-Next/`에
  HF lfs 크기와 바이트 단위로 같게 있다(09-16 받음). 새 다운로드는 없다. 다만 받은 스크립트가 86행 문법 오류로 검증 단계를
  건너뛰고 `FETCH_DONE`을 찍었다 — 조용한 실패다. 리드가 §1의 sha256 넷과 대조한다(읽기 111 GB, CPU sha가 한계라 수 분[유도];
  30분 안이지만 페이지 캐시를 V4.1 파일과 다투므로 03의 묶음이 도는 동안은 하지 않고 박스가 빈 틈에 한다). 박스 쪽
  `fetch-gguf.sh:86`은 레포 밖 파일이라 rig-log `configs/` 쪽 정리 항목으로 남긴다.
- **Q2 첫 프로그램은 위치 0–2,050까지 dense, 그 너머는 이름 붙은 거부.** 받는다. 완성 블록이 512개 이하(p + 1 ≤ 2,051)이면
  QSA가 가시 토큰을 전부 고르므로 dense GQA와 비트 단위로 같다(§3 C, transformers `:757`·메인라인 `:682`). 선택기(K7 `q38qsa`)는
  e2e가 도는 뒤의 라운드다. 원시 인덱서 키 캐시 쓰기는 첫 프로그램부터 넣는다(선택기가 붙을 때 캐시 모양이 바뀌지 않게).
  선례는 V4.1의 16,384 위치 거부(`ef80083`).
- **Q3 리더는 `qwen35moe` 팔의 변형(R-a `q38reader`, aa, S–M).** 새 크레이트도 새 팔도 아니다(rebuild §2-1 원칙 2). 변형 표
  §2가 그대로 스펙이다. `ModelSpec` 쪽 다섯 가지(`Arch::Qwen4Exp`, `Gqa.select`, `HcSpec { streams, kind: Mhc | Gated{rank} }`,
  `Extra::Ple`(EngramSpec의 일반화), `DeltaKind::Gdn.gate: Silu|Sigmoid`)는 `crates/models`를 만지므로 **glm5next 착륙 뒤**에
  직렬로 간다 — glm5next가 `Selector::TokenPool`과 mHC `HcSpec`의 모양을 먼저 정하고, q38reader는 그 위에 얹는다. §7의 5·6은
  지금 정한다: `expert_feed_forward_length`가 층별 배열인 파일은 이름으로 거부(이 파일은 스칼라), 이미지 토큰 입력은 리더의
  텍스트 전용 거부. `ple.eos_token_id`(248,044)는 반드시 그 키로 읽는다 — tokenizer eos(248,046)로 대신하면 조용한 실패다.
- **Q4 새 연산 넷은 사촌이다 — 셋은 aa, 하나는 03.** gated-residual HC(K5 `q38hc`, aa, M: op 라이브러리의 새 모듈, `gpu-deepseek41`
  이 아니다; ex `hc_mix.cu`의 `gr_mix` 쌍이 설계 참조, ik 탭 `hc_mixed`·`hc_combine` 밴드), PLE(K6 `q38ple`, aa, M: `engram::Hash`의
  접두어·token_map·EOS 리셋 일반화, IQ4_NL 행 역양자화, `engram_gate` ROW 제네릭, 팽창 conv 커널 — V4.1 engram 엔트리 md5 불변이
  증명, 움직이면 q35attn 선례대로 thin 엔트리), QSA 선택기(K7 `q38qsa`, aa, L: GLM k-pool과 같은 `TokenPool` 계열 — ik·ex처럼
  **풀링된 블록 키를 캐시**한다, 메인라인의 매 스텝 재풀링이 아니다; 768 n B 대 3,072 n B), 카드 routed **Q5_1** down `_sel`
  (K4 `q38exp`, 03, M: k 640 = 20블록; routed Q8_0 down 다섯 층은 카드 `_sel` 또는 qdot Q8_0 fused 중 하나). q5kexp(Q5_K)와
  형식이 다르니 합치지 않는다.
- **Q5 03의 인스턴스 셋(K1–K3)은 Qwen3.6 사슬 뒤.** GDN sigmoid 게이트 엔트리(`norm_gate.rs`, `act::<ACT>` 본체는 있음), 라우터
  `gated` 두 번째 인스턴스(512/10; norm 융합 불필요 — 입력이 HC mix), GQA flash (256, PACK 4). §8의 두 지적(`gated` 모듈의
  `N_EXPERT`·`N_USED` 상수, `flash_gqa.rs:78` `GROUP = 8`)은 modelvocab 처분과 같은 것이라 03에게 03 DONE 릴레이 때 한 줄로
  전한다 — 03의 `kernelshape`가 GROUP을 const generic으로 열면 K3는 인스턴스 하나다.
- **Q6 시간선: 긴 막대는 카드 dense와 호스트 expert, 레버는 흐름 먼저.** 받는다. 토큰당 카드 dense 4.83 GB(700 GB/s에서 6.9 ms)
  + 호스트 expert 4.9–10.9 ms(대역 가정 둘 사이의 밴드) → 약 12–18 ms, 55–80 tok/s @ depth ≤ 2k, A6000, 균등 라우팅[유도].
  카드 expert 1.1–1.2 ms는 호스트의 그늘 아래라 그 커널을 줄이는 라운드는 열지 않는다. 순서: ① 두 카드(3090에 expert 16–20 GB
  → 호스트 몫 20–25 %, 호스트 항이 카드 front 아래로) ② hot list ③ MTP verify k+1 배치(카드 dense를 k+1 토큰이 나눠 읽음 — 이
  모델의 가장 큰 레버; 초안은 unsloth 공유 MTP 2.79 GB, 오라클은 ik). output 675 MB·F32 라우터 252 MB를 줄이는 것은 정확도가
  바뀌는 선택이라 트리아지에 「사용자 결정」으로 둔다.
- **Q7 스펙 입력의 오류 셋을 고친다.** 풀 어텐션 헤드 차원은 **256**(스펙의 128은 틀림; config `head_dim`, 카드 L57, GGUF
  `attention.key_length`), MoE는 **48층 전부**(스펙의 36은 GDN 층 수다), `m-latest 53ed051ce`는 박스에 없어 `3d82ef6`을 읽었다.
  rebuild §2-3의 Qwen3.8 문장을 같은 커밋에서 고친다. 스펙 사본(`specs/wave-r2/qwen4arch.md`)은 세션 스크래치라 그대로 둔다.
- **Q8 §7의 나머지 열린 질문은 그 라운드에서.** MTP 초안 경로(`DraftSpec::Mtp{file, borrows}` 대 `Block{file}`)는 q38mtp에서,
  top-k 동점 순서와 풀링 캐시 dtype(ex는 f32 평균 뒤 원래 dtype, 메인라인은 f32 합)은 K7에서 오라클과 함께, 도구 파서·채팅
  템플릿의 Qwen3.6 대비는 q38reader에서 한다. 사슬은 트리아지 「q38 사슬」 행이다: R-a → {K1 ‖ K2 ‖ K3 ‖ K4 ‖ K5 ‖ K6} → P1
  `q38prog`(layerprog R1·R2·R5, session, hostone B–E 뒤) ‖ E1 `q38oracle`(ik 덤프 1회, 30분을 넘을 수 있어 사용자 승인) → K7.
- **§8 범위 밖 지점의 처분.** `engram::Hash`의 `KEY_PREFIX` 하드코딩·`engram_gate.rs` `ROW = 5120`·`dequant_row` IQ4_NL
  `Unsupported`는 K6에 들어간다. `qdot::has_features`의 Q8_0 arm 부재(routed Q8_0이 f32 역양자화 대체 경로로 감)는 K4다.
  `Gqa.select`·`HcSpec` mHC 전용 필드는 R-a다. 나머지 둘은 Q5(03)와 Q1(박스 스크립트)이다.

## 메모 원문 (라운드 `qwen4arch`, 2026-09-27)

# `qwen4arch` 설계 메모: Qwen3.8-Flash-Next(`qwen4exp`)를 Qwen 리더의 변형으로 붙이기

리드 bloomery-aa에게 보냅니다. 읽기 전용 라운드였고, 저장소 파일과 git 상태는 바꾸지 않았습니다. 박스에서는 `ssh ws`로 읽는 명령만 썼고, 빌드·바이너리·임대는 쓰지 않았습니다.

인용 약어는 다음과 같습니다.
- **lc**: 메인라인 llama.cpp `3d82ef62d` (2026-09-20, `/root/worker-scratch/llama.cpp-3d82ef62d47fd74e18f36c5eccbdcf965b617b17`). 박스에서 가장 새 트리입니다. `930e2fa59`(`/home/user/llama.cpp`)와는 `qwen4exp.cpp`의 hc_mix 융합 부분만 다릅니다.
- **ik**: `/home/user/ik_llama.cpp` @`c10fbbcc`.
- **ex**: exllamav3 `/home/user/exllamav3-src` @`0740edc`.
- **tf**: transformers 5.17.0 `modeling_qwen4_exp.py` (md5 `edbc627d…`, 박스의 세 사본이 같음).
- **ms**: 리더 팔 `/Users/hckim/repo/bloomery-modelspec` @`8641b17`.

메모리 규칙상 스크래치에 둔 사본은 `/private/tmp/claude-501/-Users-hckim-repo-bloomery/02d044bc-8d2a-4055-91fa-4ae541a4916f/scratchpad/qwen4arch/{lc,ex,tf,hf}/`입니다.

## §0 권고 한 줄씩

1. **이 기계에서 돌릴 수 있습니다.** 양자화는 unsloth **UD-Q4_K_XL**(111,334,654,784 B)이고, A6000 한 장과 호스트 티어로 돕니다. 파일 네 샤드는 이미 `/models/Qwen3.8-Flash-Next/`에 HF lfs 크기와 바이트 단위로 같게 있습니다(09-16). **새 다운로드는 필요 없습니다.**
   - 다만 fetch 스크립트가 86행 문법 오류로 끝나 sha256 검증 흔적이 없습니다. 리드가 §1의 oid 넷과 대조해야 합니다.
2. **첫 프로그램은 가시 위치 2,051개(위치 0–2,050)까지만 돌리고, 그 너머는 이름 붙은 거부로 끝냅니다.** 이 범위에서는 QSA가 dense GQA와 비트 단위로 같으므로 선택기가 없어도 됩니다(§3 C).
   - 필요한 것은 dense GQA와 원시 인덱서 키 캐시 쓰기뿐입니다. V4.1의 16,384 위치 거부(`ef80083`)가 같은 모양의 선례입니다.
3. **리더는 새 크레이트가 아니라 `qwen35moe` 리더 팔에 `qwen4exp` 행을 더하는 변형입니다.** 약 +500줄이고 파일 목록은 §5 R-a에 있습니다.
   - `crates/models/src/**`는 진행 중인 `glm5next`의 경계입니다. 그래서 glm5next 착륙 뒤에 직렬로 돕니다.
4. **새 연산은 넷입니다.** 모두 기존 연산의 사촌입니다.
   - gated-residual HC: V4.1 mHC의 저랭크·원소별 사촌입니다.
   - PLE: V4.1 engram 해시와 게이트를 일반화하고, 팽창 conv를 더합니다.
   - QSA 선택기: GLM k-pool과 같은 `TokenPool` 계열입니다.
   - 카드의 routed Q5_1 down `_sel`입니다.
5. **인스턴스는 셋이 더 필요합니다.** GDN sigmoid 게이트 엔트리 하나, GQA flash (HEAD 256, PACK 4), 라우터 512/10입니다.
6. **디코드 시간선의 긴 막대는 두 개입니다.** 하나는 카드 dense 4.83 GB/토큰(700 GB/s에서 6.9 ms)입니다. 다른 하나는 호스트 expert로, 균등 라우팅에서 4.9–10.9 ms입니다.
   - 카드 expert는 약 1.1–1.2 ms로 호스트의 그늘 아래 있으니, 카드 expert 커널을 줄이는 라운드는 가치가 0입니다.
   - 레버 순서는 흐름 레버가 먼저입니다: 두 카드 → hot list → MTP verify 배치.
7. **스펙 입력의 오류 셋을 바로잡습니다.**
   - 풀 어텐션 헤드 차원은 **256**입니다(스펙의 128은 틀림). 근거는 config `"head_dim": 256`, 카드 L57 `- Head Dimension: 256`, GGUF `attention.key_length = 256`입니다.
   - MoE는 **48층 전부**입니다(스펙의 36층은 틀림).
   - `m-latest 53ed051ce`는 박스의 어떤 llama.cpp 트리에도 없습니다(`git cat-file` 실패). 그래서 `3d82ef6`을 읽었습니다.

## §1 공개 현황

**체크포인트.** `Qwen/Qwen3.8-Flash-Next`, 라이선스 `other`(`qwen-community-1.0`), BF16 179,999,981,459 파라미터(HF API `"total":179999981459`), 생성일 `"createdAt":"2026-08-24T08:24:59.000Z"`입니다.
- 카드 L26: "This experimental preview of the architecture that will underpin Qwen4 …"
- 카드 L46: "Number of Parameters: 125B with 6B activated, plus 51B n-gram embedding and 4B MTP"
- 같은 조직에 `Qwen/Qwen3.8-Flash-Next-FP8`도 있습니다(검색 목록).

**GGUF (`unsloth/Qwen3.8-Flash-Next-GGUF`, tree API에서 바이트를 합산).**

| 양자화 | 샤드 | 바이트 |
|---|---|---|
| UD-IQ1_S / IQ1_M | 3 | 72,546,461,344 / 74,538,755,776 |
| UD-Q2_K_XL | 3 | 78,869,128,864 |
| UD-IQ3_XXS / Q3_K_XL | 3 | 81,961,823,936 / 89,986,353,824 |
| UD-IQ4_XS | 3 | 93,682,584,224 |
| **UD-Q4_K_XL** | **4** | **111,334,654,784** |
| UD-Q5_K_XL / Q6_K_XL | 6 | 158,286,406,650 / 169,165,382,688 |
| Q8_0 | 6 | 188,225,033,248 |
| BF16 | 8 | 354,029,930,496 |
| MTP/(초안 머리 6개) | 6 | 24,616,077,888 |

UD-Q4_K_XL 샤드와 lfs sha256입니다. 박스 파일의 크기는 넷 모두 일치합니다(`find -printf %s`).
- `-00001` 10,946,624 `4448186216b3…8082`
- `-00002` 49,859,583,136 `3f342f1c1580…a6c9`
- `-00003` 49,376,141,504 `56758f40269c…cbd3`
- `-00004` 12,087,983,520 `753bda48b98b…510a`

박스에는 MTP 초안 둘도 있습니다. `mtp-…-Q8_0.gguf` 4,137,429,120 B와 `mtp-…-shared-Q8_0.gguf` 2,786,568,256 B입니다. `/models` 여유는 743,959,007,232 B입니다(`df -B1`).

**메인라인.** #27742 "model: add Qwen3.8-Flash-Next (qwen4exp)"는 머지됐습니다. API JSON에서 `"merged": true`, `"merged_at": "2026-08-27T19:32:31Z"`, `"merge_commit_sha": "6c84c7d5d8833c6e0df69628f75a0f599797934e"`입니다. 박스 `/home/user/llama.cpp`의 `git log --diff-filter=A`가 `6c84c7d5d Thu Aug 27 … (#27742)`로 같은 결과를 보였습니다.
- 뒤이은 수정은 다음과 같습니다: #27880(그래프 split), #28023(인덱서 헤드 합), #28123(순환 상태 롤백), #27941(seq_cp, 블록 위치), #28068(GDN norm), #28896(rms_norm+mul 융합).
- **MTP**는 #28243 "models: Qwen3.8-Flash-Next MTP"가 `"state": "open"`, `"merged": false`입니다. unsloth MTP README L26: "**A stock `ggml-org/llama.cpp` build cannot use these.** Mainline has no MTP graph for the `qwen4exp` architecture…"
- **ik**는 `src/graphs/build_qwen4exp.cpp`(728줄)를 가지고 있습니다. 빌드된 `build/src/libllama.so`는 09-15 22:15에 만들어졌고(커밋 22:00 뒤) 문자열 `qwen4exp`가 31번 나옵니다. `llama-spec-features.cpp:242`의 `llama_model_share_qwen4exp_mtp_tensors`가 공유 MTP 초안을 받습니다. 그래서 **ik가 수치 오라클과 MTP 오라클 둘 다 됩니다.**
- mistral.rs에는 qwen4exp가 없습니다(`models/`에 `qwen3_next.rs`까지).

**URL은 이번 라운드에 curl로 받았습니다(WebFetch가 아님).** 인용 줄은 받은 파일에서 grep한 것입니다(`…/scratchpad/qwen4arch/hf/`).
- `https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next`
- `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json`
- `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/README.md`
- `https://huggingface.co/api/models/unsloth/Qwen3.8-Flash-Next-GGUF/tree/main?recursive=true`
- `https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/raw/main/MTP/README.md`
- `https://api.github.com/repos/ggml-org/llama.cpp/pulls/27742`, `…/pulls/28243`
- 샤드 헤더 네 개: `…/resolve/main/UD-Q4_K_XL/…-0000{1..4}-of-00004.gguf`, 범위 0–25 MB

## §2 변형 표: 리더의 `Hparams` 필드, `qwen35moe` × `qwen4exp`

출처는 두 곳입니다. qwen4exp 열은 UD-Q4_K_XL 헤더에서 읽은 값이고, 로더는 `lc:qwen4exp.cpp:26-147`입니다. qwen35moe 열은 `ms:crates/model/src/arch/qwen35moe/hparams.rs`와 modelspec 보고 §1입니다. 두 arch 모두 키 접두어는 `split.arch_key`가 `general.architecture`로 붙입니다.

| 필드 (키) | qwen35moe | qwen4exp | 판정 |
|---|---|---|---|
| `n_layer` (`block_count`) | 40 | 48 | 같음 |
| `n_embd` (`embedding_length`) | 2048 | 2560 | 같음 |
| `n_head` / `n_head_kv` | 16 / 2 | 24 / 2 (GROUP 12) | 같음 |
| `head_dim` (`attention.key_length`, `value_length`) | 256 | 256 | 같음 |
| `rope_dims` / `rope_sections` / `rope_base` | 64 / [11,11,10,0] / 1e7 | 같은 값 | 같음 |
| `rms_eps`, `n_vocab`, `n_ctx_train` | 1e-6, 248,320, 262,144 | 같은 값 | 같음 |
| `n_expert` / `n_used` | 256 / 8 | 512 / 10 | 같음 |
| `expert_ff` (`expert_feed_forward_length`) | 512 | 640. 로더가 층별 배열도 받습니다(`get_key_or_arr`, `:27`) | 같음. 리더는 배열을 받거나 이름으로 거부합니다 |
| `shared_ff` | 512 | 640 | 같음 |
| `conv`, `state`, `k_heads`, `v_heads`, `ssm.inner_size` | 4, 128, 16, 32, 4096 | 4, 128, 16, **48**, 6144 | 같음. `inner = v·state` 검사도 같습니다 |
| `interval` / `attention.recurrent_layers` | 4 / 선택 | 4 / 선택(`:123-131`) | 같음 |
| `nextn_predict_layers` | >0이면 거부 | 없음. 변환기가 MTP를 버립니다(`conv_qwen4exp.py:29-30` `no_mtp = True`) | 같음 |
| `expert_gating_func` / `_weights_norm` / `_weights_scale` | 없으면 기본값 | 없음. 그래프가 SOFTMAX·norm·scale 0(곱 없음)을 넘깁니다(`qwen4exp.cpp:1001`) | 같음 |
| `hc_streams` (`hyper_connection.count`) | 없음 | 4. 1 이하는 거부(`:48-51`) | **새것.** V4.1 키 이름을 빌렸습니다 |
| `hc_rank` (`hyper_connection.low_rank`) | 없음 | 320 | **새것**(qwen4exp 전용) |
| `hyper_connection.sinkhorn_iterations` / `epsilon` | 없음 | 없음 | **없음이 판정입니다.** low_rank가 있고 sinkhorn이 없으면 Gated, 둘 다 있으면 이름으로 거부합니다 |
| `idx_heads` / `idx_dim` (`attention.indexer.head_count` / `key_length`) | 없음 | 4 / 128 | **새것.** V4.1·GLM과 이름이 같습니다 |
| `idx_budget` (`attention.indexer.top_k`) | 없음 | 2048 | **새것, 이름은 같고 뜻이 다름.** V4.1에서는 압축 행 512개이고, 여기(와 GLM)에서는 토큰 수입니다 |
| `compress` (`attention.compress_ratios`) | 없음 | 48칸 `[0,0,0,4,…]` | **새것, 이름은 같고 뜻이 다름.** V4에서는 압축 스트림 비율이고, 여기서는 QSA 블록 크기입니다. GDN 층에서는 0, 어텐션 층에서는 0보다 커야 합니다. `kinds`와 교차 검사합니다 |
| `ple_layers` (`ple.layers`) | 없음 | `[1]`(0-based; HF config는 1-based로 `[2]`) | **새것.** 정확히 1개여야 하고, GDN 층이어야 합니다(`:73-78`, `:133-138`) |
| `ple_ngram` / `ple_heads_per_ngram` / `ple_conv` | 없음 | 3 / 8 / 4 → 헤드 (3−1)·8 = 16, 팽창 = ngram 3 | **새것.** 2 ≤ n ≤ 8, 헤드 ≤ 64(`:97-102`) |
| `ple_mult` / `ple_offsets` / `ple_vocab` | 없음 | u64 3개 / 16개 / 16개(소수 크기 20,000,003…20,000,171) | **새것.** 길이 검사, i32 안에 들어감, 표 행 수 ≥ max(off+size)(`:103-121`, `:171-185`) |
| `ple_eos` (`ple.eos_token_id`) | 없음 | **248,044**. tokenizer의 eos는 248,046입니다 | **새것.** tokenizer eos로 대신하면 조용한 실패입니다. 반드시 이 키를 읽습니다 |
| `ple_image` (`ple.image_token_id`) | 없음 | 248,056(선택) | 새것. 텍스트만 돌린다면 이미지 입력을 이름으로 거부합니다 |
| `ple_row` (`embedding_length_per_layer_input`) | 없음 | 160 | **새것, 이름은 gemma3n에서 빌림** |
| GDN 출력 게이트 활성함수 | SiLU | **sigmoid**(`qwen4exp.cpp:476-486` "the one numerical difference from Qwen3.5's GDN") | **키 없음, arch가 정함.** 리더가 arch 이름으로 명시해서 set합니다 |
| 층 norm | `attn_norm`, `post_attention_norm`, `output_norm` | 없음. HC 모듈의 grouped RMSNorm과 머리 믹서가 대신합니다(`:158-159`) | **arch가 정함.** roles에서 텐서가 있고 없는 것으로 교차 검사합니다 |
| pre-tokenizer | `qwen35` | `qwen35` | 같음 |

**`ModelSpec` 쪽 변경.** 모두 `crates/models/src/lib.rs`입니다.
- `Arch::Qwen4Exp`를 더합니다.
- `Gqa`에 `select: Option<Selector>`를 더합니다. 오늘은 `Latent`만 `select`를 가집니다.
- `Selector::TokenPool`에 풀링 규칙(Mean / GLM의 것), 키 norm(RMS / Layer), 키 rope(블록 시작 위치), 헤드 가중(균등 / 학습)을 더합니다. glm5next가 정한 모양 위에 얹습니다.
- `HcSpec`을 `{streams, kind: Mhc{sinkhorn, eps, mix, collapse} | Gated{rank}}`로 바꿉니다. rebuild §2-2 초안과 같은 모양입니다.
- `Extra::Ple`는 `EngramSpec`을 n-gram 사이트 하나로 일반화해서 둡니다. 필드는 conv(kernel, dilation), pad 규칙(EOS 리셋 / pad id), token_map 유무, 행 폭 160입니다.
- `DeltaKind::Gdn`에 `gate: Silu|Sigmoid`를 더합니다.

**Role 매핑.** 모두 기존 `Role`에 들어가고 새 역할은 없습니다.
- `hc_*` → HyperConnection
- `output_hc_*` → Head
- `indexer.*` → Attention
- `ple_key`/`ple_value` → EngramDense
- `ple_norm_*`, `ple_conv1d` → EngramGain
- `per_layer_token_embd` → EngramTable

## §3 층 종류와 연산 순서

**A. 트렁크(층마다).** 출처는 `lc:qwen4exp.cpp:353-458`(graph 생성자), `ex:qwen4_exp.py:236-302`입니다.

`res_hc = [emb]×4`(`:388-392`, f32 스트림 4개 × 2560). 층 `l`마다:
1. PLE 층이면 `res_hc = build_ple(…)`(`:397-399`)
2. `cur, inject = hc_mix(res_hc, hc_attn_*)`(`:402-407`)
3. GDN(`:412`) 또는 QSA-GQA(`:414`)
4. `res_hc = hc_combine(res_hc, cur, inject)`(`:427`)
5. `cur, inject = hc_mix(res_hc, hc_ffn_*)`(`:429-434`)
6. `ffn`(`:436`)
7. `hc_combine`(`:439`)

마지막은 `hc_mix(res_hc, output_hc_*)`를 inject 없이 부르고(`:446-450`, 이것이 출력 norm) → `output`입니다.

**HC mix** (`:267-321`; tf `:1003-1040`; ex `hyperconnections.py:217-239`):
- `xn = rms_norm_per_stream(x)·γ[2560,4]`, 변환기가 1+w로 접었습니다(`:279-281`)
- `lo = silu(down[10240→320]·xn / 4)`(`:285-286`)
- `g = up[320→10240]·lo`
- `mixed = mean_c(xn_c ⊙ σ(g_c))`(`:297-309`)
- `inject = W_inj[10240→4]·xn`

**HC combine** (`:323-351`): `res_c += 2σ(inject_c/4)·y`(`:332-333`). 스트림 사이 섞임(comb 행렬)도, Sinkhorn도 없습니다. mainline은 `ggml_dsv4_hc_post(…, comb = nullptr)`로 V4 커널을 identity comb로 재사용합니다(`:338`).
- **V4.1·GLM mHC와의 차이.** V4.1은 24값 머리 gemv(pre 4, post 4, comb 16)에 Sinkhorn을 씁니다(`gpu-deepseek41/src/hc.rs` 머리 주석, `ds41_hc_post`). qwen4exp는 원소별 저랭크 게이트와 평균을 쓰고, 스칼라 주입 4개만 있습니다.

**GDN 층** (`:861-986`, `build_qkvz` `:460-474`, tf `:465-600`):
- `qkv = attn_qkv[2560→10240]·x`, `z = attn_gate[2560→6144]·x`
- `β = σ(ssm_beta·x)` [48], `g = softplus(ssm_alpha·x + dt_bias)·ssm_a` [48]
- conv4 + SiLU over 10,240 채널(conv 상태 3열)
- q, k를 L2 norm하고 k-head는 tiled로 반복합니다(`:955-960` `ggml_repeat_4d`). 변환기가 qwen35moe와 같은 `_LinearAttentionVReorderBase`(`conv_qwen4exp.py:18`)를 씁니다.
- delta 재귀(state [48][128][128] f32)
- `RMSNorm(o)·w ⊙ σ(z)`(`:476-486`, **sigmoid**) → `ssm_out[6144→2560]`

**QSA 층** (`:775-859`; 인덱서 `:542-691`; 희소 어텐션 `:695-773`):
- `q|gate = attn_q[2560→12288]`(헤드마다 인터리브), `k, v = attn_k/v[2560→512]`
- q/k RMSNorm(256), IMROPE 64/256
- 인덱서는 원시 `k = indexer.k_proj[2560→128]`을 **정규화도 회전도 하지 않은 채** 캐시합니다(`:597-603`).
- 완성된 4토큰 블록을 평균냅니다(`:612-621`) → RMSNorm(`index_k_norm`) → **블록 시작 위치**로 IMROPE를 겁니다(`:630`, `blk_pos`).
- `q_idx = indexer.q_proj[2560→512]` → 4×128, RMSNorm, 쿼리 위치로 rope를 겁니다.
- 점수 = `Σ_h relu(q_h·k_blk)`(`:645-659`). tf는 `/√128`을 붙입니다(`:755`). 양수 상수라 top-k 순서는 바뀌지 않습니다.
- 블록 점수를 토큰으로 펼치고 마스크를 더한 뒤 `top_k(width = min(n_kv, 2048 + 4 − 1))`(`:682-684`)
- GQA 마스크를 선택된 칸만 열도록 바꿉니다(`:717-744`) → `attn ⊙ σ(gate)`(`:849-852`) → `attn_output`

**MoE** (`:988-1036`): softmax 512 → top-10 → renorm(`:1001`). 공유 expert(640)는 `σ(ffn_gate_inp_shexp·x)`로 게이트합니다(`:1020-1026`). 라우터 입력은 HC mix 출력이고, **앞에 RMSNorm이 없습니다.**

**B. PLE 임베딩** (`:1037-1127` 해시, `:1185-1204` gather, `:1206-1294` 층; tf `:1041-1257`; ex `ple.py:15-28`, `ngram_embedding.py:12-24`):
- **해시(호스트).** `ctx[0] = t_p`, `ctx[s] = t_{p−s}`입니다. 창 안에 EOS(248,044)가 있거나 이전 토큰이 없으면 그 뒤는 모두 EOS로 채웁니다(`:1101-1110`).
- n = 2, 3에 대해 `mixed = ⊕_j ctx[j]·m[j]`(u64)을 만들고, 헤드 8개마다 `row = mixed % vocab[h] + off[h]`(`:1112-1123`)입니다. 토큰당 16행입니다.
- **gather.** `per_layer_token_embd` IQ4_NL [160, 320,001,536]에서 16행 × 160 = 2560값을 모읍니다.
- **층.** `key = ple_key[2560→10240]·e`, `value = ple_value[2560→2560]·e`
- 스트림 c마다 `s_c = (RMS(key_c)·γk)·(RMS(h_c)·γq)/√2560`, `gate_c = σ(sgn(s)·√max(|s|,1e-6))`(`:1223-1231`)
- `gv_c = value·gate_c`
- `conv = SiLU(Σ_k w_k ⊙ RMS(gv)·γc [t − (3−k)·3])`: kernel 4, dilation 3, 이력 9위치 × 10,240채널(`:1245-1291`)
- `h ← h + gv + conv`(`:1293`)
- **V4.1 engram과의 차이.** 게이트 식(signed sqrt, sigmoid, 공유 value, 스트림별 key)은 `ds41_engram_gate`와 **같습니다**(`engram_gate.rs:1-40`). PLE에는 여기에 둘이 더 있습니다: ① `gv`의 grouped norm, 팽창 conv, SiLU, 가산 ② 해시에 token_map이 없고 EOS로 리셋합니다.

**C. QSA와 dense가 같아지는 경계 [유도].**
- tf: 쿼리 위치 p의 가시 토큰은 p+1개이고, 완성 블록은 ⌊(p+1)/4⌋개입니다. `min(512, 완성 블록)`개와 꼬리를 고릅니다(`:757`, `:762`).
- 완성 블록이 512개 이하이면, 즉 p+1 ≤ 2,051이면 모든 가시 토큰이 선택됩니다.
- 경계에서: p = 2,049와 2,050은 dense입니다. p = 2,051은 완성 블록 513개라 블록 하나가 빠집니다.
- mainline도 같습니다. `width = min(n_kv, 2051)`이고, 가시 칸 ≤ width이면 top-k가 유한값을 모두 담고 가려진 칸은 `kq_mask`로 −∞로 남습니다.
- 그러므로 **위치 0–2,050은 선택기 없이 비트 동일**입니다. 원시 인덱서 키는 그래도 써야 합니다.

**D. 선택기 비교.**

| | V4.1 | GLM-5.3-Flash | qwen4exp QSA |
|---|---|---|---|
| 키 | 압축 스트림(학습 compressor) 행 | 4토큰 pool | 4토큰 **평균** |
| 인덱서 | 32헤드 × 128 | 32헤드 × 128 | 4헤드 × 128, MQA 키 1 |
| 선택 | top 512행 + 2단 후보 마스크(2048블록 × 8) | top 2048토큰, LayerNorm 키, `Selector::TokenPool` | top 512블록 + 꼬리 |
| 키 전처리 | — | LayerNorm | RMSNorm, 블록 시작 rope |
| 헤드 합 | — | — | 균등 relu 합 |

- 캐시 설계는 ik와 ex가 **풀링된 블록 키를 캐시**합니다. ik는 `build_qwen4exp.cpp:317-319` "only the blocks this ubatch wrote need pooling again"이고, ex는 `qsa_indexer.py:17-19`와 `:428`의 pooled plane입니다.
- mainline은 **매 스텝 원시 키 전체를 다시 풀링합니다**(`:606-621`).
- 우리는 ik·ex 설계를 따릅니다. 바이트가 1/4이고 풀링·norm·rope가 블록당 한 번입니다(§4).

## §4 연산별 재사용과 빠진 것

| 연산 | 트리 상태 | 필요한 것 | 크기 | 비용 항 [유도] |
|---|---|---|---|---|
| GDN | 03 `crates/gpu/src/linear/`(@`61979df`): `HEAD 128`, `CONV_TAPS 4`는 고정이고 `n_k`/`n_v`는 런치 인자입니다(`mod.rs:47,50,108-112`). `KHeadMap::Tiled` 있음 | k16/v48이 그대로 맞습니다. **`GATE_SIGMOID` 엔트리 하나**가 없습니다(`mod.rs:98-103` "No entry is built for it yet"). `norm_gate.rs`의 `act::<ACT>` 본체는 이미 있습니다 | S | 03 기준 4.69 MB/층·6.7 µs(Qwen3.6, v32). v48로 비례하면 약 6.9 MB·약 9.9 µs/층, 36층에 약 0.36 ms/토큰 |
| GQA flash 256 | q35attn(@`7ef4e38`) `flash_gqa.rs`: **`GROUP = 8`이 모듈 상수**(`:78`)이고 HEAD만 const generic입니다. `assert!(GROUP == 8 && HEAD_256 == 256)`(`:121`), `MMA_ROWS == 2·GROUP` | GROUP 12 → PACK 4를 const generic으로 만들고, decode·prefill (256, 4) 인스턴스를 둡니다. 기존 엔트리는 md5가 움직이므로 q35attn 선례대로 새 thin 엔트리로 더합니다 | M | dense: 24,576 n B/토큰(12층). 2,051에서 50.4 MB |
| QSA 희소 GQA | 없음. flash는 연속 구간만 받습니다 | 블록 목록(4토큰 연속)을 받는 flash 변형. KEY_TILE 32 = 블록 8개 | M | n > 2,051에서도 50.4 MB/토큰으로 고정 |
| QSA 인덱서 | 없음. V4.1 히스토그램 정확 top-k와 candmask는 행 단위입니다 | 원시 키 쓰기, 풀링 캐시(블록 완성 때 평균·RMS·rope), 4헤드 점수, top 512 블록 + 꼬리. GLM k-pool과 한 op | M–L | 풀링 캐시 768 n B/토큰(f16), mainline 방식은 3,072 n B. 262,144에서 201 MB 대 805 MB |
| 라우터 512/10 | q35moe(작업 중) `router.rs` `pub mod gated`: `N_EXPERT 256`·`N_USED 8`이 **모듈 상수**입니다. `NORM_K 2048` 한계는 norm 융합 경로에만 걸립니다(`router_width(…,128,NORM_K)`, `:2759`). logits·fused 경로는 `usize::MAX`입니다(`:2595`, `:2648`) | `gated` 모듈의 두 번째 인스턴스(PER_LANE 16, USED 10, `UBATCH_TOKENS` 32768/11 = 2,978). norm 융합은 필요 없습니다(입력이 HC mix) | S | F32 가중치 5.24 MB/층, 48층에 252 MB/토큰 |
| 공유 expert + sigmoid 게이트 | Qwen3.6 경로(q35moe) | 같은 규칙, ff 640, Q8_0 | — | 5.22 MB/층 |
| 카드 routed gate/up Q4_K | `qwen3moe_gate_up_swiglu_q4k`(k 런타임) | 그대로 씁니다 | — | expert당 921,600 B × 2 |
| 카드 routed down **Q5_1**(43층) | 카드에는 V2-Lite `enqueue_down_add_q5_1`(m=1, `fused.rs:823-845`)만 있고, `_sel`은 Q4_K/Q6_K뿐입니다 | **Q5_1 `_sel`**(k 640 = 20블록). 프리필 grouped GEMM Q5_1은 뒤로 | M(+L) | expert당 1,228,800 B |
| routed down **Q8_0**(층 2·4·30·46·47) | 카드 `_sel` 없음. 호스트 `qdot::has_features`에도 Q8_0 arm이 없어(`qdot/src/lib.rs:165-187`) dequant 대체 경로로 떨어집니다 | 카드 Q8_0 `_sel` 또는 호스트 Q8_0 fused 중 하나 | S–M | expert당 1,740,800 B |
| routed gate/up **Q5_K**(층 2) | `CardFormat::of_routed`가 카드에서 거부합니다(`placement.rs:367-368`) → 호스트. qdot는 Q5_K를 지원합니다 | 없음(자동으로 호스트) | — | — |
| 호스트 티어 | V4.1 hybrid(Q3_K r8 전용 경로 포함). qdot는 Q4_K, Q5_1 × q8_2_x4를 지원합니다 | 첫 Qwen 호스트 티어입니다. hostone 포트를 모델 무관하게 씁니다 | M(prog 안) | §5 |
| gated-residual HC | 없음. V4.1 `ds41_hc_pre`는 Q3_K gemv와 Sinkhorn 융합이고, `ds41_hc_post`는 comb 행렬입니다 | mix(grouped RMS, down/up Q8_0 저랭크, σ, 평균) + combine(2σ, identity) + 머리 믹서. ex `hc_mix.cu`의 `gr_mix` 쌍이 설계 참조입니다 | M | down+up+inject 약 14.3 MB/층 × 48 ≈ 0.69 GB/토큰(dense 안) |
| PLE 해시 | `crates/engram::Hash`: 같은 식입니다(`hash.rs:8-19`). `KEY_PREFIX = "deepseek41.engram."` 하드코딩, token_map 필수, pad id | 접두어 인자, token_map 선택, EOS 리셋·EOS pad, 이미지 토큰 | S | 호스트 µs |
| PLE 표 행 | IQ4_NL: `dequant_row`가 `Unsupported`를 냅니다. 표 `KVALUES_IQ4NL`은 `iq_tables.rs:252`에 있습니다 | IQ4_NL 행 역양자화(호스트) | XS | 16 × 90 = 1,440 B/토큰 gather |
| PLE 게이트 | `engram_gate.rs` `ROW = 5120`(`:57`), `PER_THREAD == 5·GROUP` 단언(`:60`) | ROW를 const generic으로 하는 2560 인스턴스, key/value 두 gemv(Q8_0) | S | 27.85 + 6.96 MB/토큰(층 1만) |
| PLE conv | 없음. GDN `conv.rs`는 conv+SiLU+L2+β 융합이라 재사용 불가 | grouped norm + 팽창 conv(k4, d3, 10,240채널, 이력 9) + SiLU + 가산 | S | 이력 368,640 B/시퀀스 |
| 순환 상태 | session(3파동)이 만듭니다 | GDN state 113,246,208 B + conv 36 × 3 × 10,240 × 4 = 4.4 MB + PLE conv 이력 0.37 MB + PLE 이전 토큰 2개. 롤백 스냅숏 k=4에 약 453 MB(modelvocab) | (session) | — |
| pre-tokenizer | `qwen35`(Qwen3.6 항목과 같음) | — | — | — |

**예상 coverage 목록** [유도; 코드를 돌리지 않았습니다].
- GDN(sigmoid) 36층
- router softmax 512/10 48층
- shared ff 640 48층
- GQA flash head 256 pack 4 12층
- QK norm + IMROPE 64/256 12층
- 출력 게이트 12층
- TokenPool(QSA) 12층
- HC Gated 4×320 48층 + 머리
- PLE 사이트 층 1
- 순환 상태 슬롯
- 혼합 트렁크 프로그램
- 형식: routed Q5_1 down(카드) 43층, routed Q8_0 down 5층, IQ4_NL 표 행, q8_0 attention·head·token embedding(Qwen3.6 UD와 같은 항목), BF16 인덱서 투영(`Bf16AsF32` 있음)
- 도구 파서

## §5 종이 위 비용과 시간선 [모두 유도]

**바이트.** 헤더의 텐서 정보에서 셌습니다. 합계 111,323,630,080 B + 메타데이터가 파일 크기와 맞습니다.

| 항 | 값 |
|---|---|
| routed expert 전체 | 77,017,907,200 B. expert당 3,072,000 B/층(Q4_K 921,600 × 2 + Q5_1 1,228,800) |
| PLE 표 | 28,800,138,240 B (IQ4_NL [160, 320,001,536]) |
| 나머지 | 5,505,584,640 B |
| **카드 dense 바이트/토큰**(나머지에서 token_embd 675,430,400 뺌) | **4,830,154,240 B.** output 675 MB와 F32 라우터 252 MB 포함 |
| routed 바이트/토큰(균등; expert 크기가 층 안에서 같아 정확) | 1,504,256,000 B |
| KV/위치(f16) | GQA 24,576 B + 인덱서 원시 키 3,072 B = 27,648 B. 32k에 0.91 GB, 262k에 7.25 GB |
| PLE/토큰 | 1,440 B gather(호스트) |

**배치** (A6000은 `nvidia-smi` 49,140 MiB). 카드에 둘 것:
- 비-expert 약 4.83 GB(+token_embd 0.68 GB, 카드 gather를 고르면)
- KV(32k에 0.9 GB), 상태 0.12 GB와 스냅숏, 작업공간
- expert 약 38–43 GB(바이트의 49–56 %)

호스트에 둘 것: 나머지 expert 34–39 GB, PLE 표 28.8 GB. 박스 RAM은 `free -b` 기준 270,071,025,664 B입니다. 표는 ex처럼 NVMe에서 행만 읽어도 됩니다(`ngram_embedding.py:16-18`). 속도 항은 아닙니다.

3090 한 장(24,576 MiB)이면 expert 약 16 GB(21 %)만 카드에 올라가고, 호스트 몫은 79 %가 됩니다.

**디코드 시간선(토큰당).** 구조는 V4.1의 front / shadow / back입니다(`docs/research/layerprog-design.md` §1a). 층마다 `front(카드) + max(shadow 카드 expert, 호스트 expert + PCIe 왕복) + back`이고, 층 사이는 직렬입니다(다음 층 입력이 결합을 기다림).

| 자원 | 항 | ms/토큰 |
|---|---|---|
| 카드 SM(front: dense) | 4.83 GB ÷ 700 GB/s | **6.90**, 상태 0.36과 어텐션(≤ 50 MB, 0.07) 별도 |
| 카드 SM(shadow: 카드 expert) | 1.504 GB × 0.49–0.56 ÷ 700 | 1.06–1.20 |
| 호스트 DRAM(호스트 expert) | 1.504 GB × 0.44–0.51, 층당 13.8–15.9 MB | 136 GB/s에서 4.9–5.6. facts.md:108의 디스패치당 바이트 법칙(16.8 MB에 75 GB/s)으로는 9.5–10.9 |
| PCIe | 층당 D2H 10 KB + H2D 10 KB. 스텝당 48 × 2회 동기 | `c_sync` × 96. 값은 측정 안 됨 |
| NVMe | 0(PLE 표를 RAM에 둘 때) | — |

- **벽시계**는 6.9 + 0.4 + max(1.1, 4.9…10.9 + PCIe)이므로 약 12–18 ms, 약 55–80 tok/s @ depth ≤ 2k, A6000, 균등 라우팅입니다. 이 범위는 호스트 대역 가정 둘 사이의 밴드이고, 단일 숫자가 아닙니다.
- **카드 expert 항은 호스트의 그늘 아래 있습니다.** 카드 expert 커널을 줄이는 것은 가치 0입니다.

**흐름 레버, 순서대로.**
1. **두 카드.** 3090에 expert 약 16–20 GB를 더 두면 호스트 몫이 약 20–25 %로 내려갑니다. 그러면 호스트 항이 약 2–3 ms가 되어 카드 front 아래로 들어갑니다. 오늘 엔진은 두 카드 배치를 거부하고(`body.rs:127-135`, modelvocab 인용), 두 카드 적재의 선례는 V4.1 plan (b)(`stage-gpu-load-v41`)입니다.
2. **hot list.** 토큰 몫을 카드로 옮깁니다.
3. **MTP verify k+1 배치.** 카드 dense 4.83 GB를 k+1 토큰이 나눠 읽습니다. 카드 front가 긴 막대인 이 모델에서는 가장 큰 레버입니다. 초안은 unsloth 공유 MTP(2.79 GB)이고 오라클은 ik입니다.

그 다음이 항 단축입니다. output 675 MB를 줄이는 것(예: 어휘 부분 head)과 F32 라우터 252 MB를 줄이는 것(적재 때 변환)은 **정확도가 바뀌는 선택**이라 AGENTS 「Performance first」 판단이 필요합니다.

**프롬프트.**
- 토큰 사이의 재귀는 **GDN delta 하나뿐**입니다(03의 프리필 delta는 토큰 루프를 커널 안에 둡니다).
- 나머지는 T 폭 GEMM이나 병렬로 돕니다: 모든 투영, HC down/up/inject, PLE key/value, 라우터·공유·routed grouped GEMM, PLE 팽창 conv(이력 9 + ubatch 안 병렬), QSA 선택(쿼리마다 독립), 프리필 flash(P ≤ 2,051에서는 dense).
- 밴드 계약을 따릅니다(rebuild §7-5).
- 호스트 union은 V4.1 실측 31.5 µs/슬롯(AGENTS)을 expert MAC 비 0.139로 줄이면 약 4.4 µs/슬롯입니다. P = 512에서 층당 슬롯 2,250–2,600개라 약 10–11 ms/층, 512토큰당 0.48–0.55 s이고, 호스트 union 한계는 약 0.9–1.1 k tok/s입니다. 가정이 강한 [유도]이고 밴드로만 씁니다.

## §6 라운드

리더 변경은 변형 표이고, 새 크레이트도 새 리더도 아닙니다(원칙 2). 순서는 원칙 2를 따라 dense-우선 단계입니다.

| 라운드 | 소유 | 경계 | 선행 | 증명 클래스 | 크기 | 박스 [유도] |
|---|---|---|---|---|---|---|
| **R-a `q38reader`** | aa | `crates/models/src/lib.rs`(+~60: Arch, `Gqa.select`, `HcSpec` kind, n-gram spec, Gdn gate), `need.rs`(+~60), `crates/model/src/arch/qwen35moe/{hparams.rs(+~150), roles.rs(+~35), spec.rs(+~90), mod.rs(+3)}`, `arch/mod.rs`(+~4 디스패치 `"qwen35moe" \| "qwen4exp"`), `coverage.rs`(+~15), 새 `tests/qwen4exp_meta.rs`(+~150), `justfile` 레시피 1개. 합계 약 +550줄 | modelspec 착륙, **glm5next 착륙**(models 크레이트) | move + 이름 붙은 거부. qwen4exp는 UnknownArchitecture에서 coverage 목록으로 바뀝니다(FAIL-first). Qwen3.6 목록은 Gdn gate가 SiLU로 찍혀 문자열이 불변이어야 합니다. 바뀌면 날짜를 붙여 재핀합니다 | S–M | 메타 게이트 4–5개, 호스트 cargo만, 약 10–15분 |
| K1 `q38gdn` | 03 | `crates/gpu/src/linear/norm_gate.rs`(엔트리 1), `gate_linear.rs` 절(n_v 48, sigmoid) | q35gdn 착륙 | 더하기(새 행만) | S | ptx-scan 3개 + gate_linear, 약 15분 |
| K2 `q38router` | 03 | `arch/qwen3moe/router.rs` `gated` 두 번째 인스턴스 | q35moe 착륙 | 더하기 | S | 약 15분 |
| K3 `q38gqa` | 03 | `flash_gqa.rs`, `flash_gqa_prefill.rs` (256, 4) | q35attn 착륙, 03 `kernelshape` | 더하기(기존 md5 불변) | M | 약 20분 |
| K4 `q38exp` | 03 | 카드 Q5_1 `_sel`(새 파일), Q8_0 routed(카드 `_sel` 또는 qdot) | K2, hostone | 더하기 | M | 약 20분 |
| K5 `q38hc` | aa | op 라이브러리 쪽 새 모듈(gated mix, combine, 머리). `gpu-deepseek41`이 아닙니다 | opslib R1 | 더하기. ik 탭 `hc_mixed`, `hc_combine`과 밴드 | M | 약 20분 |
| K6 `q38ple` | aa | `crates/engram/src/hash.rs`(일반화), `crates/gguf/src/quant.rs`(IQ4_NL), `engram_gate.rs` ROW 제네릭, 새 PLE conv 커널 | — | V4.1 engram 엔트리 md5 불변을 증명합니다. 움직이면 q35attn 선례대로 thin 엔트리. 해시는 ik `set_input`과 행 id 대조 | M | 약 25분 |
| P1 `q38prog` | aa + 03 hunk | Qwen3.6 프로그램 + `Residual::Hc(Gated)` 배선 + PLE extra(층 1, HC mix 앞) + 호스트 티어 MoE + 2,051 위치 이름 거부 | layerprog R1 `progshape`, R2 `progprompt`, R5 `q36prog`, `session`, hostone B–E, K1–K6 | 새 모델 e2e | M–L | 111 GB 첫 적재(NVMe 속도는 이번에 안 잼) + e2e, 약 20–30분 |
| E1 `q38oracle` | aa | refset 계열 행 `qwen4exp`(ik 노드 덤프), 메인라인 토큰 대조 | P1과 병행 | 참조 세트 | S–M | ik 덤프 1회. 30분을 넘을 수 있어 리드 승인 대상입니다 |
| K7 `q38qsa` | aa | 인덱서 풀링 캐시 + 선택 + 희소 flash(K3 위) | P1, GLM k-pool op(glmops 쪽) | 새 연산. 2,051 거부를 해제합니다 | L | 약 30분 |

**선행 작업(리드).**
- 샤드 넷의 sha256을 §1 oid와 대조합니다. 읽기 111 GB이고 CPU sha가 한계라 수 분입니다[유도].
- **페이지 캐시 경합**: 박스 RAM 270 GB에 V4.1 파일과 이 111 GB가 동시에 들어가지 않습니다. 적재가 겹치는 배치는 느려집니다(`box-saturation` 사건).

## §7 열린 질문

1. **MTP 초안 경로.** unsloth는 별도 파일로 두고, `shared-`는 target의 embedding과 head를 빌립니다. `DraftSpec`에 `Mtp{file, borrows}`를 둘지, `Block{file}`로 볼지 정해야 합니다. 메인라인 #28243은 열려 있고, ik는 지원합니다.
2. **top-k 동점 순서.** ggml `top_k`와 우리 히스토그램 top-k가 2,052 위치부터 갈릴 수 있습니다. 밴드 계약에서 허용하는지 판단이 필요합니다.
3. **도구 파서와 채팅 템플릿.** Qwen3.6과 같은지 비교하지 않았습니다.
4. **풀링 정밀도.** ex는 f32로 평균낸 뒤 원래 dtype으로 되돌립니다(`qsa_indexer.py:116`). mainline은 f32 합입니다. 풀링 캐시의 저장 dtype(f16/f32)을 정해야 합니다.
5. **`expert_feed_forward_length`가 층별 배열인 파일.** 받을지 거부할지 정해야 합니다. 이 파일은 스칼라입니다.
6. **이미지 입력.** PLE가 이미지 위치를 `image_token_id`로 해시합니다. 텍스트 전용 거부를 어디에 둘지 정해야 합니다.
7. **GLM k-pool의 정확한 필드.** glm5next의 `TokenPool` 확정을 보고 QSA 필드를 얹어야 합니다. 이번 라운드에서 GLM 풀링 규칙 원문은 읽지 않았습니다.

**하지 못한 것.**
- sha256 대조는 읽기 전용 명령 목록 밖이라 하지 않았습니다.
- coverage 목록은 코드를 돌리지 않은 예측입니다.
- NVMe 순차 읽기 속도와 `c_sync`는 측정값이 없습니다.

## §8 범위 밖 개선 지점 (보고만)

- `crates/engram/src/hash.rs`의 `const KEY_PREFIX = "deepseek41.engram."`: 모델 이름이 공유 해시에 박혀 있어 PLE가 재사용할 수 없습니다. S.
- `crates/gpu-deepseek41/src/engram_gate.rs:57,60`의 `ROW = 5120`과 `PER_THREAD == 5·GROUP` 단언: modelvocab 보고 뒤에도 그대로입니다. S.
- `crates/gguf/src/quant.rs`의 `fn dequant_row` IQ4_NL arm이 `Unsupported`입니다. 표는 `iq_tables.rs:252`에 있습니다. XS.
- `crates/qdot/src/lib.rs`의 `fn has_features`에 Q8_0 arm이 없어 routed Q8_0 행이 f32 역양자화 대체 경로로 갑니다. S–M.
- 박스 `/home/user/fetch-gguf.sh:86`: 문법 오류로 검증 단계가 돌지 않았는데 `FETCH_DONE`을 찍습니다. 조용한 실패입니다(레포 밖). XS.
- q35moe(작업 중) `crates/gpu/src/arch/qwen3moe/router.rs`의 `pub mod gated`: `N_EXPERT`·`N_USED`가 모듈 상수입니다. 두 번째 모델이 모듈 사본을 부릅니다(`dflash_router` 선례). const generic 모듈이 맞습니다. S–M.
- q35attn `crates/gpu/src/flash_gqa.rs:78,121`: `GROUP = 8` 상수와 단언이 GROUP 2·4·5·6·12·16 변형을 막습니다(modelvocab 지적 그대로). M.
- ms `crates/models/src/lib.rs`의 `struct Gqa`에 `select`가 없고, `struct HcSpec`은 mHC 전용 필드뿐입니다. qwen4exp와 MiniMax-M3 블록 희소가 막힙니다. S.
- 스펙 입력(`specs/wave-r2/qwen4arch.md` 26–29행): "GQA 24/2 × 128"과 "36 MoE layers"가 틀렸습니다. rebuild §2-2·§2-3의 Qwen3.8 행에는 헤드 차원이 없습니다. 리드가 문서에 반영할 때 256과 48층으로 고칩니다. XS(문서).

## §9 모델

Opus 5.5(`claude-opus-5-5`)로 실행했습니다.
