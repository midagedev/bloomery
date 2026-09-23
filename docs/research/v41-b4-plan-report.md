# V4.1 B4 op 계획 — b4plan 원문 보고

읽기 전용 계획 라운드 b4plan(opus, 2026-09-23)의 보고를 그대로 옮겼다. 결정과 라운드 순서는 [`plan.md`](../plan.md)의 B4 카드가 정본이고, op 라운드의 스펙은 여기 §1의 블록을 복사해 간다. 보고 안의 스크래치 경로는 세션 스크래치라 남아 있지 않을 수 있다.

`b4plan` 보고. 읽기 전용 라운드라 저장소와 박스 데이터는 바꾸지 않았다. 표시: **[측정]** 이번 라운드에 덤프 파일을 맥으로 복사해 numpy로 재계산한 값 또는 rig-log의 측정값, **[읽음]** GGUF 헤더·매니페스트·문서에서 읽은 값, **[유도]** 계산한 값, **[가정]** 가정.

**먼저 알아야 할 것 넷.**
1. **ik `c10fbbcc`(우리 PR #2455)의 인덱스 키가 rope가 걸린 잠재에서 투영된다.** 참조 구현은 rope 전 잠재를 쓰고, ik 주석(`build_deepseek4.cpp:719`)도 그렇게 적혀 있다. [측정] 근거는 블록 E에 있다. 이 배선은 포트 첫 커밋 `0c6e934a`부터 있었다. 그러니 B5의 PPL 2.2355(c2048, top-k가 켜지는 길이)도 이 결함이 있는 상태에서 잰 값이다. 수정은 XS이고 결정은 리드 몫이다(블록 E).
2. **5토큰 세트 안의 op는 거의 전부 ik 규칙을 호스트로 옮기면 덤프와 비트 동일하게 재현된다** [측정]. HC_PRE, HC_POST, 접기, DS4_COMP, RMS, SUM_ROWS, Hadamard, 라우터 사슬이 그렇다. 조건은 두 가지다: FMA를 거는 자리가 ik와 정확히 같아야 하고, 합 두 종류(RMS·SUM_ROWS)는 f64로 누산해야 한다. 예외는 FA다. ik의 generic FA는 V를 **f16으로 누산**한다. 이 규칙을 시뮬레이션하면 덤프를 1.6e-7까지 재현하고, f32 누산 규칙과는 6–8e-4만큼 차이 난다.
3. **게이트 하니스가 V4.1 세트의 이름 없는 뷰를 못 읽는다.** 덤퍼는 파일 이름의 공백을 `_`로 바꾸는데(`dump_ref.cpp:89-96`) `RefRow::file_name`(`gpu-gates/src/lib.rs:277-280`)은 바꾸지 않는다. 박스에 있는 파일은 `_(view).13.f32`이고 ` (view).13.f32`는 없다. HC 게이트는 전부 이 뷰를 읽는다. 그래서 선행 라운드 하나가 필요하다(§2 `b4-infra`).
4. 인덱서 분기, 고른 행, 디코드 한 스텝, 깊이에 따라 달라지는 것은 5토큰 세트에 없다. 이를 닫으려면 덤퍼에 "프리필은 조용히, 디코드 한 스텝만 덤프" 기능이 필요하다. 벽시간은 30분 한참 아래로 본다 [유도, §3].

---

## 1. op별 블록

게이트는 공통으로 `gate_p4` 형식을 따른다(`gate_p4.rs:300-322`). ① 커널을 **우리 규칙의 호스트 전사**와 대조해 비트 동일 또는 `KERNEL_BAND` 1e-5(`gpu-gates/src/lib.rs:30`)를 본다. ② **ik 규칙의 호스트 시뮬**을 덤프와 대조해 의미론을 증명한다. ③ `ik_rel`(커널 대 덤프)을 찍고, 밴드는 규칙 차이에서 유도한다. 파일 이름은 `<공백→_ 이름>.<occ>.f32`이고, 정수는 매니페스트 `int` 행의 `file` 열을 따른다.

### A. 하이퍼커넥션 — 앞 norm·gemv(바뀜) + HC_PRE·HC_POST(새 것) + 접기(바뀜)
1. **노드**. 층 2의 attention 서브층을 예로 든다.
   - `hc_pre-2`/0 RMS_NORM [20480,5] ← `l_out-1 (reshaped)`
   - `hc_pre_mixes-2`/0 MUL_MAT [24,5] ← `blk.2.hc_attn_fn.weight`(q3_K [20480→24] [읽음]) × `hc_pre-2`
   - `node_247` HC_PRE [120] ← `hc_pre_mixes-2`, `blk.2.hc_attn_scale`[3]. src2인 `hc_attn_base`[24]는 매니페스트 열에 안 나온다.
   - 출력 배치: pre `[S·t]`, post `[S(T+t)]`, comb `[2ST+S²t]`(`ggml.c:24430-24433`)
   - 뷰: `_(view).15` = pre(오프셋 0, 합 7.6985, **다음 서브층이 씀**), `.13` = post(20, 1.2400), `.14` = comb(40, 19.99998)
   - `hc_attn_post-2` HC_POST [5120,4,5] ← x `attn_out-2`, post, res `l_out-1`, comb
   - `hc_ffn_pre-2` MUL_MULTI_ADD ← `hc_attn_post-2` × pre(.15)
   - FFN 서브층: `hc_pre-2`/1 → `node_277` → `l_out-2`
   - 초기값: `hc_init`은 REPEAT로 4벌 복제하고, `hc_pre_init`은 one-hot [1,0,0,0]이다(`build_deepseek4.cpp:1531-1539`).
   - 출력 접기: `node_3197`(마지막 토큰) × `node_3200` → `hc_out` [5120,1]
   - 지연: 서브층 s의 입력은 s−1의 HC_PRE가 낸 pre로 접힌다(`llama-hparams.cpp:2191`, `build_deepseek4.cpp:1550-1566`). 그래서 HC_PRE는 서브층 본체의 임계 경로 밖에 있다. 그 출력을 쓰는 것은 자기 HC_POST와 다음 서브층뿐이다.
2. **ik 산술**.
   - RMS(`ggml.c:17463-17466`): f32 제곱을 f64로 합한다. `mean=(float)`, `1/sqrtf(mean+eps)`. eps는 파일 값 9.999999682655225e-21 [읽음].
   - gemv: q3_K × q8_K 활성값.
   - HC_PRE(`24396-24487`), 토큰마다 S=4:
     - `pre=σ(x·s0+b)+eps`, `post=2σ(x·s1+b)`
     - comb 로짓 `x·s2+b`에 행 softmax(max 차감, expf, 직렬 합, `/sum` 뒤 `+eps`)
     - 열 정규화: 합은 eps에서 시작한다.
     - 이어서 {행, 열} 정규화를 19회 반복한다(각 합은 eps에서 시작).
     - `iters`=20, `eps`=9.999999974752427e-07. **두 키 모두 파일에 있다** [읽음]. 그러니 조용한 기본값은 없다.
   - HC_POST(`24494-24607`, T==1 경로 `24552-24577`): `out[i][d]=x[d]·post[i]+Σ_j comb[j·S+i]·res[j][d]`
   - MUL_MULTI_ADD(`iqk_cpu_ops.cpp:430-501`): `y=x0·w0`, `y+=xj·wj`
   - [측정] 각 규칙의 재현 결과:
     - HC_PRE: 아핀을 FMA로 두면 node_247과 120/120 비트 동일. FMA가 없으면 72/120, 최대 16 ulp.
     - HC_POST: `fma(x,post_i, round(comb_0i·r_0))`를 먼저 하고 `fma(comb_ji,r_j,·)`(j=1..3)을 잇는 형태가 `hc_attn_post-2`·`l_out-2`·`hc_attn_post-0`에서 각 102,400/102,400. 평범한 FMA 사슬은 90,921.
     - 접기: `x0·w0` 뒤 FMA 사슬이 25,600/25,600이고 T=1(`hc_out`)에서도 5,120/5,120. FMA 없이는 21,000과 3,701.
     - RMS 20480: f64로 102,400/102,400.
3. **재사용**.
   - `rms_norm`(`elem.rs:264-319`, gain과 f32 트리) → 무게 없는 변형으로 쓴다.
   - q3k gemv(`lib.rs:1137`)는 **`Q8Act`가 k ≤ 10944라 K=20480을 거부한다**(`tensor.rs:106`, B4 카드가 이미 적어 둠). 게다가 24행이면 격자가 3블록이다.
   - `weighted_sum`(`elem.rs:409-443`)은 FMA가 없다. 그래서 `mul_add` 순서를 박은 새 엔트리가 필요하다.
   - HC_PRE와 HC_POST는 새로 만든다. 설계안은 두 커널이다 [유도]:
     - [HC_POST + 다음 입력 접기]를 한 커널로. 접기 가중치는 HC_POST보다 먼저 준비돼 있다.
     - [RMS + split-K gemv + HC_PRE]를 한 커널로.
4. **게이트**. 5토큰 세트가 80 서브층 전부를 덮는다(T=5 경로).
   - op 게이트는 덤프한 mixes를 입력으로 받는다. HC_POST와 접기는 `mul_add` 순서를 박으면 비트 동일이다. HC_PRE는 exp를 f64로 계산해 반올림하면(토큰당 40값) 비트 동일이고, CUDA `expf`로 하면 ≤4 ulp다 [가정: CUDA expf ≤2 ulp].
   - 사슬 게이트(norm+gemv+HC_PRE) 밴드 [유도]: gemv 밴드 ε(기존 q3_K 게이트의 활성값 양자화 밴드)를 입력으로 둔다. comb 로짓 오차는 s2·ε·max|mix| = 0.1557·ε·69 ≈ 11ε다([읽음] 층 2의 scale [0.0977, 0.0224, 0.1557], [측정] |mix| 최대 69·412).
   - 구멍: T==1 경로.
5. **비용**.
   - 가중치 16,904,640 B/토큰 [읽음 §3] → 29 µs. HC_POST와 접기가 서브층마다 ~205 KB로 토큰당 16.4 MB → 28.5 µs [유도].
   - **진짜 비용은 3블록 gemv의 바닥이다.** q3_K의 바닥은 K=512에서 1.45 µs, K=5120에서 5.5–5.8 µs다 [측정 B9/B11]. 두 점을 외삽하면 K=20480에서 ≈12 µs이고 80회면 ≈1.0 ms/토큰이다 [유도]. split-K(K 조각 512)로 바꾸면 ≈0.2 ms다 [유도].
   - 노드: 융합하지 않으면 400 × 0.852 µs = 0.34 ms, 2노드로 융합하면 0.14 ms [유도].

### B. rope(바뀜) + ROPE_BACK(새 것)
1. **노드**.
   - `kv_rope-N` ← `kv_norm-N (reshaped)`, `inp_pos`
   - `q_rope-N` ← `q_b-N (reshaped)(reshaped)` [512,64,5]
   - `csa_state_compress-2`/2 ← /1, `dsv4_csa_write_pos` [0,2]
   - 인덱스 키 rope `_(reshaped)_(view)`/0
   - `attn-N` ROPE_BACK ← `attn_raw-N`(0·1층) 또는 `attn_shared-N`
   - 디코드에서는 `indexer_q`가 더해진다.
   - 게이트 파일 예: `kv_norm-0`/`kv_rope-0`(10k), `kv_norm-2`/`kv_rope-2`(160k YaRN), `attn_shared-2`/`attn-2`
2. **ik 산술**(`ggml.c:21070-21273`).
   - `theta_scale=powf(base,-2/n_dims)`(`:21121`), 캐시 θ는 f32로 거듭 곱한다(`20808-20822`). `rope_yarn`(`20790-20806`), ramp(`20784-20787`), corr_dims floor/ceil(`20768-20782`).
   - NORM은 인접 쌍이고 `x0·c−x1·s`, `x0·s+x1·c`(`21214-21227`)이다. **`op_params[15]=1` → offset = ne0−n_dims**(`21154-21155`). 512 헤드에서는 448부터 **꼬리 64**, 인덱스 128 헤드에서는 64부터 꼬리다. ROPE_BACK은 sin 부호만 −1이다.
   - 층별 인자(`build_deepseek4.cpp:1092-1100`):
     - 0·1층: base 10000, fs 1, ext 0
     - 나머지와 압축 행·인덱스 q/k: base 160000, fs 1/16, ext 1, β 32/1, orig 65536 [읽음]
     - attn_factor = 1/(1+0.1·ln16)(`:23-29`). mscale을 곱하면 f32에서 정확히 1.0이다 [측정].
     - corr_dims는 [15, 25] [유도].
   - 압축 행과 인덱스 키의 위치는 그룹 시작 `pos+1−ratio`다(`llama-dsv4.cpp:767-770`).
   - [측정] 수축 없는 평범한 연산 + 정확 반올림 cos/sin으로 `kv_rope-0` 2559/2560, `kv_rope-2` 2559/2560, `attn-2` 163,787/163,840이 비트 동일하다. 나머지는 1 ulp~~(삼각함수·powf 구현 차이)~~다. FMA를 걸면 2520–2526/2560으로 더 나빠진다. **정정(09-23, b4rope가 원인을 확인)**: 1 ulp의 원인은 삼각함수·powf가 아니다. ik의 CPU 빌드에서 GCC가 `rope_yarn`의 θ 혼합 `theta_interp·(1 − ramp_mix) + theta_extrap·ramp_mix`와 `1 + 0.1·ln(1/freq_scale)`을 FMA로 수축한 것이 원인이다. 표에서 그 두 FMA를 재현하고 회전은 수축 없이 두면 rope 사이트 794개가 전부 비트 동일하다[측정, b4rope 게이트]. 위의 "FMA를 걸면 더 나빠진다"는 회전에 건 FMA 이야기다. `kv_rope-0`의 0–447차원은 입력과 같다.
3. **재사용**.
   - `elem.rs:325-364` `rope`: 호스트 cs 캐시, 인접 쌍, `rope_pair_core`(`:164-166`)가 ik 순서다.
   - **작은 변형으로 된다**: 행 폭 512/128 + 오프셋 448/64 + in-place.
   - ROPE_BACK은 (cos, −sin) 표를 넘긴 같은 커널이다.
   - 호스트가 표 둘을 ggml 레시피로, 박스 libm으로 짓는다.
4. **게이트**. 5토큰으로 모든 사이트를 덮는다. 호스트 대비 비트 동일(PLAIN_BAND 1e-6)을 보고, ik_rel ≤ 1 ulp다. 구멍은 큰 위치다(세트는 0..4만). D2(pos 1025)와 호스트만의 큰 p(~1M) 표 점검으로 닫는다.
5. **비용**. 바이트는 ~9 µs로 무시할 만하다. 따로 띄우면 120노드 ≈ 0.1 ms이므로 q_b 에필로그, append, merge에 융합한다 [유도].

### C. 잠재 K/V 경로(바뀜: K/V 투영, 윈도우 KV 추가)
1. **노드**.
   - `kv_b-N` MUL_MAT [512,5](q8_0 5120→512) → `kv_norm-N` FUSED_RMS_NORM(`attn_kv_a_norm`) → `kv_rope-N`
   - `dsv4_raw_k_write-N` SET_ROWS f16 → `cache_k_lN` [512, n_ctx=512], `dsv4_raw_k_write_idxs`.i32 [0..4]
   - 읽기: `raw_k-N` VIEW f16 [512,1,256]
2. **ik**.
   - FUSED_RMS_NORM(`ggml.c:17503-17557`)의 출력은 (scale·g)·x다. f16 쓰기는 RNE다.
   - **raw 캐시는 128행 링이 아니라 KV 셀 그대로다.** 쓰기 슬롯은 `kv.head+i`(`llama-dsv4.cpp:376-389`), 읽기는 시퀀스의 전 셀을 256 패딩한 것이다(`393-477`). 링(압축)은 `--swa-compress`일 때만이고(`llama.cpp:1207-1233`) 덤프에서는 꺼져 있다.
   - 창은 SWA 마스크로 정해진다: `pos−cell.pos ≥ 128`이면 가린다(`llama.cpp:6010`).
   - 디코드에서는 nton=512 뷰로 자른다(`build_deepseek4.cpp:1272-1285`).
   - [측정] f16 쓰기는 RNE로 1024/1024 비트 동일이다(같은 op인 `csa_k_write-2`에서).
3. **재사용**.
   - `fused.rs:281` `kv_norm_rope_append`는 MLA 576 전용이다 → 512 norm + 꼬리 rope 변형을 만든다.
   - `flash.rs:343/376` `kv_append(_pos_buf)`는 `f32_to_f16_bits`로 f16을 만든다 → 폭만 인자로 받으면 된다.
   - 링을 쓸지는 우리 선택이다. 게이트는 위치로 대조한다.
4. **게이트**. 5토큰 40층에서 캐시 행 비트 동일, `kv_norm`은 f64면 비트 동일이다. `kv_b`는 기존 q8_0 밴드를 쓴다.
5. **비용**. `attn_kv` 11.98 µs × 40 = 0.48 ms [측정 B11], append는 무시할 만하다.

### D. 압축기(새 것: 풀링) — 상태 gemv·persist·DS4_COMP·norm·rope·f16 캐시
1. **노드**. 층 2(csa, 비율 2):
   - `csa_state_kv-2`/`csa_state_score-2` MUL_MAT(q3_K [5120→512])
   - `csa_source_kv`/`_score` CONCAT(상태 `leaf_62`/`leaf_68`, 현재) [512,7], `dsv4_csa_state_read`.i32 [2,3,4,5]
   - `csa_state_compress-2`/0 DS4_COMP [512,2] → /1 norm → /2 rope → `csa_k_write-2` SET_ROWS f16 → `leaf_74` [512,256] @ `dsv4_csa_state_write`.i64 [0,1]
   - persist: `csa_persist_kv/score-2` GET_ROWS(src ~~[3,4]~~ [4,3] — 정정 09-23 b4plan2: dst 순서로 적으면 이렇고, plan 게이트가 비트 동일로 확인했다) → `csa_k_state_persist-2` SET_ROWS(dst [0,1])
   - 소스는 2·8·14(csa)와 20(hca, 게이트 없음, 비율 1)이다.
   - 별칭 [읽음]: 3–7은 `leaf_74`, 9–13은 `leaf_236`, 15–19는 `leaf_402`, 21–39는 `leaf_563`.
2. **ik**.
   - type1(`ggml.c:24764-24830`)은 16레인 청크마다 비율 행들에서 max를 잡고, `w=expf(s−max)`, `sum+=w`, `res+=w·kv`(행 순서), `y=res/sum`이다. 비율 1이면 정확한 항등이다.
   - 계획(`llama-dsv4.cpp:664-832`): `n_visible=(pos+1)/ratio`(`:738`). 그룹이 끝나면 행 `pos/ratio`에 위치 `pos+1−ratio`로 쓴다(`763-770`). persist는 슬롯마다 마지막 토큰이다(`751-761`). n_kv는 256으로 패딩한다(`816`).
   - [측정] FMA 규칙이 1024/1024(평범하면 920). ik의 상태는 0으로 시작한다. 참조는 −inf다(`model.py:455`). 다만 실제 토큰이 persist되기 전에는 읽히지 않는다 [유도].
3. **재사용**. 상태 gemv는 q3k gemv를 쓴다. 풀링+norm+rope+쓰기는 작은 새 커널 하나로 묶는다.
4. **게이트**. 5토큰 층 2/8/14/20, **첫 스텝뿐**이다(state_read가 전부 현재 배치를 가리킨다). op 게이트는 비트 동일(f64 exp)이거나 ≤2 ulp다. 구멍은 persist된 상태 읽기와 디코드 중 그룹 완성이다 → D1/D2.
5. **비용**. 작은 q3_K 사이트는 1.45–5.8 µs이고, 압축기·인덱서를 합쳐 토큰당 19런치 90 µs다 [측정 B9]. 풀링 자체는 무시할 만하다.

### E. 인덱스 키(새 것) — mm 512→128·k_norm·rope·Hadamard·f16 쓰기
1. **노드**.
   - 층 2: `node_201` MUL_MAT [128,2] ← `blk.2.indexer.attn_k`(q3_K [512→128] [읽음]) × `csa_state_compress-2`(매니페스트는 occurrence를 말하지 않는다) → `node_202` norm → `_(reshaped)_(view)`/0 rope → `lid_k_new-2` HADAMARD → `lid_k_write-2` → `leaf_79` [128,256]
   - 층 20: `node_1672` → `lid_k_new-20` → `leaf_568`
2. **ik**. `build_deepseek4.cpp:1153-1170`. `fast_ht`(`iqk_cpu_ops.cpp:503-520`)는 h=1..64 버터플라이를 돌고 단계마다 f32 `scale*=0.707106781f`, 마지막에 곱한다.
   - **발견 [측정]**: `*pre_rope = comp`(`:721`) 뒤에 같은 텐서의 reshape에 `ggml_rope_ext_inplace`가 걸린다(`:724-725`). 매니페스트 순서를 보면 이 rope가 node_201보다 먼저 돈다.
   - q8_K 활성값을 시뮬레이션해 대조했다. **rope 뒤 입력**은 node_201과 2.5e-7(층 2), ≤3.3e-7(층 20 행 0–4)로 맞는다. **rope 전 입력**은 pos>0 행에서 21 %(층 2), 11–101 %(층 20) 어긋난다. 행 0은 둘 다 같다(pos 0).
   - 참조는 rope 전 잠재다(`model.py:434`, `:528`, `:544`, `:749`). phylliida는 `ggml_dup`을 해 두었다(`phylliida/src/graphs/build_deepseek41.cpp:599-600`). skelectric은 같은 패턴이다(`skelectric/.../build_deepseek4.cpp:898-906`, 재지 않음).
   - [측정] `node_202`는 f64 규칙으로 256/256, `lid_k_new-2`는 `fast_ht`로 256/256 비트 동일이다.
3. **재사용**. q3k gemv(K=512 바닥 1.45 µs), rms 변형, B의 rope(오프셋 64), `kv_append`(폭 128). Hadamard는 새로 만들되 작고 융합한다.
4. **게이트/결정**. 5토큰 층 2/8/14/20에서 행 0은 깨끗하다. pos>0 행은 ik의 결함을 따른다. 리드 결정:
   - **(a)** #2455를 고친다(XS) → 다시 뜬다(~6분) → PPL을 다시 잰다.
   - **(b)** 우리는 참조를 따르고, 재덤프 전까지 행 0과 호스트 시뮬로 게이트한다.
   - 밴드: norm·rope·Hadamard는 비트 동일이다. mm은 활성값 양자화 밴드를 따른다. K=512에서 f32 활성값과 q8_K의 차가 1.1e-2 max/max다 [측정 node_201]. 그러니 우리 양자화의 호스트 전사로 대조한다. **덧붙임(2026-09-23, b4comp)**: 우리 q3_K gemv의 활성값은 f32가 아니라 q8_1(128값 블록, half-away, ±127)이다 — 게이트의 전사도 그 규칙이다.
5. **비용**. 1.45 µs × 4층, 그룹 완성 때만(csa는 2토큰마다) [측정 바닥].

### F. 인덱서 쿼리·점수·top-512(새 것)
1. **노드**. 세트에 없다(`:1344-1348`, 512 < 256이 거짓). 빌드될 때(`809-910`) 생기는 노드:
   - `lid_q`(q8_0 [1280→4096]) → `indexer_q` rope → `lid_q_hadamard`
   - `lid_weights`(q3_K [5120→32]) × 1/√4096
   - 기본은 융합 `lid_top_k` INDEXER_TOPK(i32 [512,n])이다. 비융합이면 `lid_kq`, `lid_score` SUM_ROWS, `lid_score_masked`, TOP_K가 생긴다.
   - 발행층은 2·8·14와 20·24·28·32·36이다(`llama-hparams.cpp:2183-2188`). 나머지 층은 스트림의 id를 다시 쓴다.
2. **ik**. 융합 경로(`iqk_mul_mat.cpp:1993-2295`):
   - `score = mask + Σ_{h=0..31} w_h·relu(q_h·k)`이고 f32로 h 순서대로 FMA(`2141-2163`).
   - 디코드는 `iqk_bucket_topk`(`1800-1914`, 버킷 64개, 마지막 버킷만 정렬 — 동률 순서 미정)를 쓰고, 프리필은 `partial_sort`(안정 정렬 아님)를 쓴다.
   - 비융합 경로는 헤드 합이 f64(`ggml.c:15010`→`4169-4179`)이고, ARGSORT 동률은 큰 색인이 먼저다(`iqk_cpu_ops.cpp:251`).
   - 참조는 q/k를 fp4로 시뮬레이션하고 Hadamard는 없다(`model.py:546,552,555`). ik는 fp4를 뺐다.
3. **재사용**. `q8_0_gemv`(`q8f32.rs:477`; 사이트 12.5 µs [측정 B9]), q3k gemv, rope, Hadamard. **점수 커널과 커지는 집합에서 top-512를 고르는 커널은 새로 만든다.**
4. **게이트**. D1/D2 덤프(§3)가 필요하고, 비융합으로 떠서 점수를 노출해야 한다.
   - 점수 밴드: 호스트 전사 대비 1e-5. 덤프 대비는 128길이 f32 내적의 한계 128·2⁻²⁴·Σ|q·k| ≈ 7.6e-6 상대 [유도].
   - id: 512번째 점수와의 거리가 밴드 안인 것은 빼고 집합으로 일치를 본다. 순서는 상관없다.
5. **비용**. q gemv 0.10 ms + 가중치 gemv ~0.044 ms [측정 바닥]. 키 훑기는 1,664·D B/토큰 [유도, placement §4와 같음]:

   | D | 훑는 바이트 | 575 GB/s에서 |
   |---:|---:|---:|
   | 4,096 | 6.8 MB | 11.9 µs |
   | 32,768 | 54.5 MB | 95 µs |
   | 1,048,576 | 1.74 GB | 3.0 ms |

   FLOP은 53,248·D다. 4096에서 218 MFLOP이면 f32 코어로 충분하다 [유도, 가정 A6000 ~38 TFLOP/s]. **깊이에 비례하는 항은 이것 하나다.**

### G. 고른 행 모으기 + mask_to_index(새 것)
1. **노드**. 세트에 없다.
   - n_tokens==1이면(`:1350-1356`) `comp_kv_getrows`(행 gather, f16 그대로)와 마스크 열 gather(`op_params[0]=1`)가 돈다.
   - raw 512 ⧺ 고른 512, n_eff 640 → PAD 768 < 1024이므로 `mask_to_idx`(`ggml.c:24633-24677`)가 돌고, 그 결과가 `fattn` src[5]가 되어 iqk 경로로 간다.
2. **ik**. 바이트 복사다(`19823-19890`, 빌더 `9038-9073`). 디코드 id에는 −1이 없다 [유도: 보이는 행 > 512].
3. **재사용**. FA가 id로 제자리에서 읽으면 G는 사라진다(H2에 흡수).
4. **게이트**. D2에서 비트 동일.
5. **비용**. 복사하면 19.9 MB → 35 µs, 제자리에서 읽으면 0 [유도].

### H. 어텐션(바뀜) — K=V, 헤드별 sink, 창 ⧺ 압축/고른 행
1. **노드**.
   - 층 0: `fattn-0` ← q `q_rope-0 (view)(permuted)`, k=v `raw_k-0 (permuted)` f16 [512,256], mask `dsv4_raw_mask_padded-39`(입력 f16 [256,16]), sinks `blk.0.attn_sinks`(열에 없음), scale 1/√512(`:1131`) → `attn_raw-0`
   - 층 2: k=v `csa_k_all-2` [512,512](raw 256 ⧺ csa 256), mask `csa_kq_mask-2` [512,16] → `attn_shared-2`
   - 층 2에서 보이는 키 [측정]: {0}, {0,1,256}, {0–2,256}, {0–3,256,257}, {0–4,256,257}
2. **ik**.
   - 선택이 없으면 `IQK_DISABLED`(`:584-589`) → **generic CPU FA**(`ggml.c:22957-23243`).
   - **0·1층은 `n_compressed=−1`(`:1403`)이라 깊이와 상관없이 항상 generic이다.**
   - generic의 산술:
     - Q를 f16으로 바꾼다(`23144`).
     - `vec_dot_f16`(`3079-3119`, 4×8 FMA 누산기, 고정 트리)
     - **VKQ16을 f16으로 누산한다**(`vec_scale_f16` `3369-3398`, `vec_mad_f16` `3207-3237`).
     - `S=S·ms+vs`
     - sink(`23212-23226`)
     - `1/S`(`23229-23230`)
   - 선택이 있으면(디코드 2–39층, 마스크 폭 > PAD(n_eff)) iqk로 간다(`iqk_flash_attn.cpp:161-277`). x86에서는 f32 누산이다(`fa/iqk_fa_templates.h:975-981`). **`iqk_fa_512_512.cpp`는 읽지 못했다.**
   - [측정] ik의 f16 규칙을 시뮬레이션하면 ~~`fattn-0` 157,401/163,840, `fattn-2` 152,834/163,840이 비트 동일이고 최대 1.6e-7이다~~ **정정(2026-09-23, b4attn)**: 이 수는 `S=S·ms+vs`를 FMA로 수축한 시뮬의 것이다. 수축하지 않으면 5토큰 세트 40층 전부 163,840/163,840이 비트 동일이다(`gate-gpu-ds41-attn`의 `ik_sim_bits`). f32 누산(우리 규칙)은 6.1e-4(L0), 8.0e-4(L2) max/max, rms 2.6e-4/3.1e-4 차이 난다.
3. **재사용**.
   - `flash_latent`/`_seg`(`flash.rs:424/634`)는 `rope_dims`가 자유라 0으로 두면 512 폭 K=V가 된다(`2609-2637`).
   - `flash_latent_mma`(`818`)는 폭 576에 고정돼 있다(`:158`, 호스트 검사 `:2826`) → 512 인스턴스가 필요하다.
   - 64헤드 = MMA 그룹 4(`:238`).
   - merge(`1587/1659/1749`)에 sink를 더한다.
   - 키 소스 둘(창, 압축 prefix)을 세그먼트로 나눈다. 분할 설계가 이미 고정 순서로 병합한다.
   - 디코드 마스크는 개수 두 개면 된다.
4. **게이트**. 5토큰 40층:
   - (i) ik 규칙 시뮬 대 덤프 ≤2e-7(마스크·sink·scale·가시성 증명)
   - (ii) 커널 대 우리 f32 규칙 ≤1e-5
   - (iii) 커널 대 덤프 ≤2e-3 [유도: 측정 8.0e-4의 2.5배]
   - **f16 누산기는 복제하지 않는다**(더 부정확한 규칙이고 성능 우선이다).
   - 구멍: 고른 행, 창 자르기, 128 밖 가림, iqk 경로, T=1.
5. **비용**. ≤640행 × 1 KiB → 25.2 MB/토큰 → 44 µs, 3.2 GFLOP/토큰, 노드 80 → 68 µs [유도]. ~~작은 항이다.~~ **정정(09-23, b5plan)**: 3.2 GFLOP/토큰을 시간에 넣지 않았다. 키 하나가 131 kFLOP(128 FLOP/B)이라 이 커널은 바이트가 아니라 연산에 묶이고, 대역은 0.39–0.57 ms/토큰이다[유도].

### I. 출력 투영(바뀜) — 그룹 wo_a, wo_b
1. **노드**. `attn-2`(ROPE_BACK) → [4096,5,8] → `attn_wo_a-2` [1024,5,8] ← `attn_output_a (reshaped)`(q8_0 [4096→8192]를 [4096,1024,8]로; 덤프에서 skip) → cont [8192,5] → `attn_out-2`(q8_0 8192→5120). T=1은 reshape 분기다(`:1430-1434`).
2. **ik**. 그룹 g는 행 g·1024..와 헤드 8g..8g+7을 쓴다. q8_0 × q8 활성값.
3. **재사용**. **`q8_0_gemv_heads`(`crates/gpu/src/model/kernels.rs:94`)가 바로 이 모양이다.** n_rows 8192, k 4096, n_heads 8, rows_per_head 1024, x/y 보폭 4096/1024로 한 런치에 끝난다 [유도, 계약에서]. 그러니 plan B4 카드와 rig-log의 "그 커널은 아직 없다"는 틀렸다. 이 형상에서 재지 않았을 뿐이다.
4. **게이트**. `attn_wo_a-N`을 기존 q8_0 밴드로 본다.
5. **비용**. 따로 8런치면 11.92 µs × 320 = 3.81 ms/토큰이다. 같은 바이트를 한 런치로 두면 **−1.64 ms/토큰**(하한) [측정 B11 ABAB]. B4에서 가장 큰 단일 항이다.

### J. 라우터(바뀜)
1. **노드**. `ffn_moe_logits-2`(bf16 [5120→384], PREC_F32) → `_probs` SQRT_SOFTPLUS → `_probs_biased` ADD → `(sort)` ARGSORT → `ffn_moe_topk-2`(읽을 것: `.0.logical.i32`) → `_weights` GET_ROWS → `_sum` → `_norm` DIV → `_scaled` ×1.5
2. **ik**. `llama-build-context.cpp:1443-1702`:
   - `1502` √softplus = `x>20?x:logf(1+expf(x))` 뒤 sqrtf(`ggml.c:3410`)
   - `1511-1515` 편향은 선택에만 쓴다.
   - `1533` top_k(6), 동률은 큰 색인이 먼저(`iqk_cpu_ops.cpp:251`)
   - `1537-1538` 편향 없는 확률을 모은다.
   - `1551` 합은 f64
   - `1554-1560` DEEPSEEK41에는 1e-20이 없다.
   - `1567-1570` ×1.5 [읽음 1.5]
   - [측정]
     - 로짓 대 f64는 3.0e-7이다.
     - √softplus는 1891/1920이 비트 동일이고 나머지는 1 ulp다.
     - 편향 덧셈은 1920/1920이다.
     - **200행(40층×5토큰) 전부에서 순서까지 top-6이 일치한다.** 상위 7에 정확한 동률은 없고, 6·7위 최소 간격은 1.56e-4(층 21, 토큰 4)다.
     - 합은 f64로 5/5, f32 순차로는 1/5다. 나눗셈과 ×1.5는 30/30이다.
3. **재사용**. `router.rs`는 V4.1 산술을 하나도 덮지 않는다. `N_EXPERT=64` 상수(`:30`), softmax, 편향·재정규화 없음, 동률은 작은 id다. `expert_table`(`296-345`)의 형태만 쓴다. 로짓은 `f32_gemv` 또는 B8로 낸다.
4. **게이트**. 5토큰 40층에서 id 정수 일치, 가중치 비트 동일. 동률 규칙은 세트가 건드리지 않으므로 합성 단위 테스트로 본다.
5. **비용**. f32 로짓 13.49 µs × 40 = 0.54 ms [측정 B11]. 격자 48이라 바닥에 묶이고, 라우터 산술은 gemv 마지막 블록에 융합한다 [유도].

### K. 전문가(바뀜) — SwiGLU ±10, shared q8_0, 결합
1. **노드**. `ffn_moe_gate_par-N` MOE_FUSED_UP_GATE(q3_K + 클램프) → `ffn_moe_down-N` MUL_MAT_ID(q4_K, 0·1층 q5_K) → `ffn_moe_out-N` MUL_MULTI_ADD. `ffn_up_gate-N` FUSED_UP_GATE(q8_0 + 클램프) → `ffn_shexp-N` → `ffn_out-N` ADD.
2. **ik**(`iqk_mul_mat.cpp:136-170`). `min(silu(g),10)`(`155-157`) · `clamp(u,±10)`(`168-169`), 40층 전부 10.0이다 [읽음]. 참조는 `silu(min(g,10))`다(`model.py:844-846`). g>10일 때만 ≤4.5e-5 차이 난다 [유도]. [측정] 층 2 shared에서 max g 1.94, max|u| 4.30이라 **클램프가 안 걸린다.**
3. **재사용**.
   - `moe_fused.rs:76`에 클램프를 넣은 ds41 엔트리를 만든다(공유 core를 호출). V2 커널은 두고, 다운 경로는 `q4k_sel.rs:66`을 쓴다.
   - `fused.rs:415`는 q3_K 전용이라 **shared q8_0용 fused gate/up + 클램프는 새로 만든다.**
   - 결합은 FMA 순서 변형이다.
4. **게이트**. 5토큰, 기존 gemv 밴드. 클램프는 10을 가로지르는 합성 입력으로 본다.
5. **비용**. shared 2.98 ms [측정 B11]. 클램프는 공짜다.

### L. engram 게이트(새 것; 조회는 B3)
1. **노드**. 층 1:
   - `engram_rows-1`: **.i32로만 정확하다**, 최대 383,377,333 [읽음]
   - `node_76` GET_ROWS(q8_0) → `engram_kv-1`(q8_0 6144→25600)
   - 값 뷰 /0(오프셋 20480) → `node_82` ×4
   - 키 뷰 /1 → `node_86` RMS(5120행마다) → `node_90` × `engram_k` 이득(bf16 [5120,4])
   - 질의: `l_out-0 (cont)` → `node_94` RMS → `node_98` × `engram_q`
   - `node_100` → `node_101` SUM_ROWS [1,4,5] → SCALE → SGN/ABS/CLAMP/SQRT/MUL → `engram_gate-1` → `engram_out-1`
   - **스트림 넷마다 게이트가 따로 있고, attention 앞에서 돈다**(`:1543-1547`).
2. **ik**(`985-1047`). 이득은 get_rows로 f32(`1016-1023`), norm 뒤 따로 MUL(`1025-1031`), `σ(sgn·√clamp(|s|,1e-6))`(`1035-1039`). [측정] SUM_ROWS 5120은 f64로 20/20(f32 쌍합은 12/20), 나머지 단계는 20/20씩이다.
3. **재사용**. rms 변형, `q8_0_gemv`. 나머지는 작은 새 커널 하나다.
4. **게이트**. 층 1·14를 덤프 입력으로 비트 동일.
5. **비용**. `engram_wkv` 0.52 ms [측정 B11]. 게이트는 무시할 만하다.

### M. 임베딩·복제·출력 접기·헤드
1. **노드**. `inp_embd` GET_ROWS(bf16) → `hc_init`. 출력: `node_3197`·`node_3200` → `hc_out` → `result_norm` → `result_output`(q6_K [5120→129,280]). argmax는 그래프 밖이다.
2. **ik**. bf16은 정확하다. 첫 접기는 one-hot이라 항등이다. [측정] `hc_out` FMA 사슬이 5120/5120이다.
3. **재사용**. `embed_rows`는 Q3_K 2048 전용이지만(`elem.rs:232-256`) 배치 문서가 호스트에서 모은다(§3). `head.rs`는 텐서 크기로 정해지니 그대로 쓴다. argmax는 동률이면 작은 색인이다.
4. **게이트**. 기존 헤드 게이트 형식, 마지막 토큰.
5. **비용**. 헤드 775.6 µs [측정 B9].

### N. 호스트 계획 입력(정수)
1. **노드**. `dsv4_raw_k_write_idxs`, `dsv4_raw_mask_padded`, `dsv4_{csa,hca}_{kq_mask,state_read,state_write(i64),write_pos,persist_src,persist_dst}`, `inp_pos`, `inp_out_ids`, engram id
2. **ik**. `llama-dsv4.cpp:329-480`, `664-832`, `842-882`(보이면 0, 아니면 −inf f16), `llama.cpp:6010`
3. **재사용**. `crates/model/src/arch/deepseek41/kv.rs`. `plan.rs`는 arch-split이 B4에 배정했다(`arch-split.md:58`).
4. **게이트**. CPU 전용, 정수 사본과 비트 동일. 5토큰은 첫 스텝만 덮는다.
5. **비용**. 호스트 몫이고 스텝 파라미터 버퍼로 올린다.

**토큰당 합** [유도]: 새 비-gemv op는 융합하지 않으면 ~0.5–1 ms다. 기준 토큰은 40.85 ms(배치 (a) 직렬 D=4096, 24.48 tok/s [유도]). 큰 항은 셋이다: 그룹 wo_a(−1.64 ms), hc gemv 바닥(split-K 없이 ~1 ms), 노드 수(융합).

---

## 2. 라운드 표

스펙은 파일을 `crates/gpu/src/<op>.rs`에 둔다고 했다. 하지만 **`arch-split.md:62,222-224`와 닫힌 S0는 V4.1 디바이스 코드를 새 크레이트 `crates/gpu-deepseek41`(엔트리 접두 `ds41_`)에 둔다.** 아래 표는 arch-split을 따랐다. 어느 쪽인지는 리드가 정한다.

| id | 내용 | 파일(새 / 접촉) | 게이트 | 앞 | 크기 | GPU 게이트 부담 |
|---|---|---|---|---|---|---|
| `b4-infra` | 크레이트 뼈대(선택적 의존 + `gpu` 피처, `ds41_`, **계획한 op `mod` 줄 전부 미리**), `RefRow::file_name`이 safe_name을 따르게, int 사본 리더, `oracle/deepseek41.rs` | 새 크레이트, 루트 `Cargo.toml`, `gpu-gates/src/lib.rs`, `oracle/*` | `_(view).13.f32`·`engram_rows-1.0.input.i32` 읽기(FAIL-first: 지금은 실패) | — | S | 없음 |
| `b4-dump` | §3의 덤퍼 기능 | `tools/ref/dump_ref.cpp`, `dump.sh`, `models/deepseek41.sh` | 기본 모드로 5토큰 재덤프 → 매니페스트 합 동일 | — | S | 없음(리드가 CPU 임대 아래 실행) |
| `b4-hc` | A | `hc.rs`, `gate_deepseek41_hc.rs`, **`crates/gpu/src/tensor.rs`**(Q8Act — 이 라운드만) | 80 서브층 | infra | M | 가벼움 |
| `b4-rope` | B(+C의 append 변형) | `rope41.rs` + 호스트 표, 게이트 | 사이트별 비트 동일 | infra | S | 가벼움 |
| `b4-attn` | H1(창 ⧺ 압축 prefix, sink, 폭 512 MMA) | `attn41.rs`, 게이트(`flash.rs`는 읽기만, cores 호출) | 40층 (i)(ii)(iii) | infra | L | **무거움** |
| `b4-moe` | J + K | `router41.rs`, `experts41.rs`, 게이트 | id 정수 일치·가중치·합성 클램프 | infra | M | 중간(선택된 expert만 올린다) |
| `b4-engram` | L | `engram_gate.rs`, 게이트 | 층 1·14 | infra | S | 가벼움 |
| `b4-woa` | I: `q8_0_gemv_heads` 배선 + `bench_v41` 사이트 | 게이트, `bench_v41.rs` | 밴드 + 리드 A6000 시간(예측: −1.64 ms 근처) | infra | S | 가벼움 |
| `b4-plan` | N | `crates/model/src/arch/deepseek41/plan.rs`, 테스트 | 정수 비트 동일(CPU) | infra | S | 없음 |
| `b4-comp` | D + E | `compress.rs`, `index_key.rs`, 게이트 | 층 2/8/14/20, E의 결정 대기 | infra, rope, **#2455 결정** | M | 가벼움 |
| `b4-index` | F | `indexer.rs`, 게이트 | D1/D2 점수·id | dump + 리드 덤프, rope, comp | M | 중간 |
| `b4-attn-sel` | G + H2(간접 행 id, T=1) | `attn41.rs`(attn이 소유) | D2 `fattn-N` src[5] | attn, index, D2 | M | 무거움 |

- **파동**: {infra ‖ dump} → {hc ‖ rope ‖ attn(무거운 것 하나) ‖ moe, 빈 자리에 engram·woa·plan} → {comp ‖ 리드 D1/D2 덤프} → index → attn-sel.
- **충돌 지점**:
  - `justfile`: 리드가 머지 때 한 줄씩 붙인다.
  - `gpu-deepseek41/src/lib.rs`: infra가 미리 적어 둔다.
  - `tensor.rs`: hc만 만진다.
  - `gpu-gates/src/lib.rs`: infra만 만진다. op 라운드는 도움 함수를 자기 bin에 둔다.
- 트랙마다 워크트리와 `BLOOMERY_REMOTE`를 따로 준다.

## 3. 5토큰 세트의 구멍과 닫는 덤프

| 구멍 | 왜 없나 | 닫는 덤프 |
|---|---|---|
| 인덱서 분기 전체(q·가중치·점수·top-k) | 512 < 마스크 폭 256이 거짓(`:1344-1348`) | D1 또는 D2 |
| 고른 행 모으기·mask_to_index·iqk FA | T=1 + 선택 | D1/D2 디코드 스텝 |
| T=1 분기: wo_a reshape(`:1430-1434`), HC_POST T=1(`24552-24577`, 수축 패턴 미검증) | 프리필만 | D1/D2 |
| 창 128 밖 가림, 창 자르기(PAD>512) | 위치 0..4 | 가림은 D1, 자르기는 D2 |
| persist된 상태 읽기, 디코드 중 그룹 완성 | 첫 스텝 | 홀수 pos에서 디코드 |
| 큰 위치의 rope·YaRN | p ≤ 4 | D2 + 호스트 표 점검 |
| 점수 값 | 융합 op가 숨긴다 | `fused_idx_topk=false`로 한 번 더 |
| SwiGLU 클램프, 라우터 동률, top-k 동률 | 값이 안 닿는다 [측정] | 합성 단위 테스트 |

- **D1(후보, 싸다)**: `--override-kv deepseek41.attention.indexer.top_k=int:64`, 프리필 301토큰 + 디코드 pos 301(csa 그룹 완성), `-c 512`, 융합 켬·끔 두 번.
  - 그래프는 `hparams.indexer_top_k`를 직접 읽고(`:879,1345`) 키가 `get_key`를 지난다(`llama-hparams.cpp:2026`).
  - **미검증이다.** 매니페스트에 `lid_top_k`가 생기는지로 확인한다.
- **D2(실제 설정)**: 프리필 1,025토큰 + 디코드 pos 1025(csa 보이는 행 513, hca 1026), `-c 2048`, 융합 끔 + 기본.
- **덤퍼 기능(`b4-dump`)**:
  - 조용한 프리필 뒤 한 스텝 덤프. `dump_ref.cpp:452`는 지금 한 번에 모든 토큰을 돈다.
  - `--tokens-file`(router_trace의 `--ids`처럼)
  - `--no-fused-idx-topk`: CLI가 없다. `common.cpp:1939`는 켜기만 한다.
  - `dump.sh:98`에 `--defer-experts`: rtrace 트리아지에서 1,200/1,200을 확인했다.
  - REF_CTX와 세트 이름 재정의(5토큰 세트를 대체하지 않는다)
  - 선택: src 열에 occurrence와 src2–5를 추가
  - 조용한 프리필은 콜백 일정 때문에 융합 산술로 돈다(rig-log 09-23). 게이트는 입력을 덤프에서 읽으므로 상관없다. 매니페스트 머리에 적어 둔다.
- **벽시간 [유도]**: router_trace가 51,200토큰(2,048 × 25, 같은 CPU 설정)을 662.9 s에 돌았다 → D2는 적재 + ~13 s + 쓰기(~2 GB [유도])다. 한 번에 몇 분이고 **30분 한참 아래라 승인이 필요 없다.** 네 번을 한 자리에서 몰아 돌린다.

## 4. 네 질문

1. **f64나 특별한 합 순서가 필요한가.**
   - f64가 필요한 것은 두 합뿐이다: RMS 계열(`17463-17466`)과 SUM_ROWS(`4169-4179`) [측정: 라우터 5/5 대 1/5, engram 20/20 대 12/20, hc 20480 비트 동일].
   - 순서가 중요한 것은 FMA 자리다: HC_PRE 아핀, HC_POST의 첫 fma, 접기, DS4_COMP. rope는 수축이 **없어야** 한다.
   - FA는 반대로 참조가 덜 정확하다(f16 누산).
   - f64 합은 크기가 작아 비용 ~0이다. 그러니 비트 동일 게이트를 위한 편의로 쓰는 것이지 엔진 요구가 아니다.
2. **"이름만 같은" 것**(10개 대조):
   - attn·ffn·최종 norm은 같은 FUSED_RMS_NORM(eps 1e-20)이다. 다만 입력이 hc 접기이고, **최종 norm 앞에는 마지막 토큰의 `hc_out` 접기가 새로 온다.**
   - q_a·그 norm(1280), q_b는 같다(q8_0). 다만 q 헤드 norm이 없고(`llama-hparams.cpp:2192`) rope가 헤드 꼬리에 걸린다.
   - 잠재 norm은 같다. 출력은 곧장 K=V f16 캐시로 간다.
   - shared add는 같지만 shared expert 자체는 q8_0 + 클램프로 바뀌었다.
   - 헤드 q6_K는 같다.
   - argmax는 그래프 밖이고 우리 규칙은 작은 색인이다.
3. **Hadamard**: `c10fbbcc` 그래프에 있다. `lid_k_new-{2,8,14,20}`이 있고, 디코드에서는 q 쪽도 8층에 있다(`:849`). 참조 `model.py`에는 없다. 정규직교라 q·k를 보존하는 포트 유물이다. 비용은 융합하면 ≈0이다 [유도]. [측정] `fast_ht`가 256/256 비트 동일이다. 키 캐시를 비트로 게이트하려면 남기고(비용 ≈0), 빼면 점수로 게이트한다.
4. **라우터**: 사슬 전부(√softplus·편향·top-6/384·모으기·f64 합·나눗셈·×1.5)가 우리 설계에서 GPU 그래프 안에서 돈다. id가 `_sel`을 몬다. 로짓은 B8이 맡는다. **`router.rs`는 어느 것도 덮지 않는다**(64 고정, softmax, 편향 없음, 반대 동률). 형태만 재사용한다.

## 5. 문서와 그래프의 모순
- `docs/research/v41-op-map.md:37`, `v41-op-map-report.md:54,130`: "앞 64" → **꼬리 64**(`ggml.c:21154-21155`) [측정].
- `v41-op-map.md:43`과 ik 주석 `build_deepseek4.cpp:719`: 인덱스 키는 rope 전 잠재 → 그래프는 **rope 뒤 잠재**를 쓴다 [측정].
- `v41-op-map.md:37`, `v41-ops.md:43`, `v41-placement.md` §4: 층마다 "128행 링" → ik raw는 n_ctx행 셀이고 창은 SWA 마스크다(`llama-dsv4.cpp:376-417`, `llama.cpp:6010`).
- `v41-op-map.md:40`: `weighted_sum` → `GGML_OP_MUL_MULTI_ADD`의 FMA 사슬.
- `v41-op-map.md:42-43`: top-k → 기본은 융합 INDEXER_TOPK(`llama.cpp:8469`)이고 산술과 동률이 다르다.
- `v41-op-map.md:72`, `v41-ops.md:77`: 조용한 기본값 → 파일에 20과 1e-6이 있다 [읽음].
- `v41-op-map.md:95`: n_swa 미확인 → 파일에 128 [읽음].
- `v41-op-map.md:94`: 본체 다섯을 안 읽었다 → 이번에 읽었다(위 인용).
- `v41-ops.md:36`: 라우터·전문가 융합 "그대로" → 라우터는 새로 만들고, 클램프와 q8_0 shared가 빠져 있다.
- `v41-ops-report.md:210`: `silu(clamp_max(g))` → ik는 `min(silu(g),10)`(`iqk_mul_mat.cpp:155-157`)이고 이 이탈은 어디에도 기록돼 있지 않다.
- `docs/oracle.md:62`: "top-k 노드" → 인덱서 호출 전체, get_rows_ext, mask_to_idx까지 없다.
- `docs/v41-placement.md:117`: 1,048,576은 헤더 값이 아니다 → `deepseek41.context_length`=1048576 [읽음].
- plan.md B4 카드, rig-log 09-23: "묶음 wo_a 커널은 없다" → `model/kernels.rs:94`가 그 모양이다.
- FA의 f16 누산(`ggml.c:23177-23206`)은 어느 문서에도 없다.

## 6. 범위 밖 개선 기회
- `ik build_deepseek4.cpp:719-727`: rope 전에 `ggml_dup`(phylliida 선례). **업스트림 PR 후보다**(우리 #2455). PPL c2048 전후 비교와 함께. XS
- `crates/gpu-gates/src/lib.rs:277-280`: `file_name`이 `dump_ref.cpp:89-96` 규칙을 따르게. XS
- `gpu-gates/src/lib.rs:736`: `.i32/.i64` 사본 리더(`int` 행의 `file` 열). S
- `tools/ref/dump_ref.cpp:272`: src 열에 occurrence와 src2–5. S
- `tools/ref/dump.sh:98`: `--defer-experts`와 REF_CTX·세트 이름 재정의. XS
- 제자리 연산 별칭 검사기: 제자리 op가 덮은 버퍼를 나중에 읽는 노드를 찾는다(이번 결함 클래스를 구조적으로 막는다. 위 src 열 항목이 선행). `tools/`, S
- `crates/gpu/src/tensor.rs:106`: Q8Act 상한 20480(b4-hc). XS
- `bench_v41.rs`: `q8_0_gemv_heads` 사이트 추가(b4-woa). XS–S
- `docs/oracle.md`: 파일 이름 규칙과 "세트의 FA는 f16 누산 generic" 한 문장씩(한국어라 리드). XS
- `ik llama-cparams.h:47`, `common/common.h:433`: "off by default" 주석이 실제 값 true와 모순. 업스트림 XS
- `ik llama-build-context.cpp:1554-1562`: DEEPSEEK41에 1e-20(이미 `v41-ports.md:85`에 있음). XS
- skelectric `build_deepseek4.cpp:898-906`: 같은 패턴 가능성(재지 않음). 알림만. XS

## 7. 실행 모델
- Opus 5.5(`claude-opus-5-5[1m]`)로 돌았다. 사람 사용자는 이 세션에 아무것도 입력하지 않았다.
- 박스에 쓴 것은 `/root/bloomery-scratch/b4plan/`뿐이다. GGUF에서 추출한 작은 텐서 바이트와 스크립트 둘, 28 MB이고 지워도 된다.
- 검증 스크립트는 세션 스크래치 `/private/tmp/claude-501/-Users-hckim-repo-bloomery/e5714b41-3850-42be-a722-79e1f225c3b5/scratchpad/b4plan/`에 있다: `chk1.py`–`chk10.py`, `chk8b.py`, `ggufti.py`, `sweep.py`, `d/`는 복사한 덤프 파일.
- 리드에게는 인덱스 키 발견을 한 줄로 먼저 보냈다.
