# q3tail 보고 (Qwen3 프리필 창에서 grouped GEMM 밖의 시간, 설계, 2026-09-26 아침)

> 리드 메모(2026-09-26). 아래는 설계 라운드 `q3tail`(opus)의 보고 원문이다. 박스 실행은 없다 — 박스 PTX를 ptxas·cuobjdump로
> 조립해 SASS를 읽고, q3router 시팅의 nsys sqlite를 다시 읽었다(A6000, main `ef00f78`, P = 4096 한 ubatch, 창 511.1 ms). 결론:
> 꼬리는 평평하다 — 레버 하나로 창의 2 %를 넘는 것은 flash 명령 다이어트(FA, −9…−17 ms, 비트 동일)뿐이고, 사다리 전체가
> 창 −55…−72 ms(라운드 여섯일곱). 상위 둘(H1: head_norm 합을 f32로 — 비트 이동, 디코드 포함; Q: `GemmAct`가 아무도 안 읽는
> q3·q4 면을 층당 117.4 MB 쓰는 것을 끊기 — 비트 동일, q3fix3 뒤)로 pp4096 8,024 → 8,280–8,330, pp512 6,815 → 7,000–7,030[유도].
> q3gemmlat의 GEMM 레버와 시간이 더해져 상위 둘을 얹으면 ≈ 9,420[유도]. flash는 텐서 수요 56–60 %·발행 36–38 %로 어느 유닛도
> 스텝을 채우지 않고, 합 모형(텐서 + 비-HMMA 발행)이 실측과 −2…−8 %로 맞아 max 모형을 기각했다 — 위상 레버(FS/FB)를 열지는
> ncu 한 번(스톨 구성)이 가른다. 스펙의 전제 하나를 정정했다: ubatch 경로는 그래프가 아니다(창의 965 커널 전부 일반 launch; 호스트가
> 5.4 ms에 다 넣고 카드보다 최대 505 ms 앞서 결론은 같다). `f32::max`의 네 명령 내림은 리드가 PTX로 확인해
> `docs/upstream/nvlabs-ledger.md` #27에 올렸다.

리드 세션 `bloomery-ee`에게. 조건은 모두 A6000 300 W, main `ef00f78`의 nsys 한 프롬프트(P = 4096, ubatch 1개, 창 511.1 ms)입니다. 커널 수치는 박스 PTX(04:10)를 ptxas·cuobjdump로 조립한 SASS에서 읽었습니다.

## 요약

- **꼬리는 평평합니다. 1순위 레버 H1조차 창의 1.4–2.0 %로 2 %에 못 미치고, 이것 자체가 이번 라운드의 발견입니다.**
  - 한 번에 2 %를 넘을 수 있는 레버는 FA(flash 명령 수 줄이기, 1.8–3.3 %)와 H2의 위쪽 대역뿐입니다.
  - 사다리를 끝까지 다 해도 창 −55…−72 ms(11–14 %)입니다. 한 라운드로 끝나는 일이 아니고 S·M 라운드 여섯일곱 개가 필요합니다.
- **상위 둘**은 H1(head_norm의 제곱합을 f64 대신 f32로, S, 비트 이동)과 Q(GemmAct에 q6·s8·d8만 쓰기, S, 비트 동일, q3fix3 뒤)입니다. 둘을 합치면 창 −15.6…−18.5 ms이고, pp4096 8,024 → **8,280–8,330**, pp512 6,815 → **7,000–7,030**을 예측합니다[유도].
- **꼬리 커널의 부류**
  - 바이트 바운드 여섯: combine 694 GB/s, swiglu_quant 689, add 682, quantize 634/647, rms_norm 563(82 %).
  - FP64 바운드 하나: head_norm. FP64 수요가 커널 사이클의 85–90 %입니다.
  - 지연 바운드 둘: router_route, 그리고 한 블록으로 도는 gemm_route.
- **아무도 읽지 않는 쓰기가 층당 117.4 MB 있습니다.** GemmAct의 q3·q4 면을 읽는 곳이 없습니다. 읽는 곳은 `gemm.rs:2043-2045` 하나이고, 거기서도 q6·s8·d8만 읽습니다.
- **flash는 어느 유닛도 스텝을 채우지 않습니다.**
  - 텐서 수요 56–60 %, 발행 36–38 %입니다.
  - 합 모형(텐서 + 비-HMMA 발행) 6,684 사이클/짝이 실측 6,795–7,247과 −2…−8 %로 맞습니다(±15 % 안). max 모형은 34–41 % 모자라서 기각했습니다.
  - 따라서 한 스케줄러의 두 워프는 서로의 HMMA 구간과 ALU 구간을 겹치지 못합니다. 남은 레버는 명령 수와 위상 둘입니다.
- **스펙 전제 하나를 정정합니다. ubatch 경로는 그래프가 아닙니다.** 창의 965 커널이 모두 일반 launch입니다. 다만 호스트가 5.4 ms 안에 전부 넣고 카드보다 최대 505 ms 앞서 있으므로 결론은 같습니다. 카드가 유일한 바쁜 자원입니다.

## 1. 변경 파일

없습니다. 레포와 박스 어디에도 파일을 고치지 않았고, 빌드·게이트·타이밍·프로파일 실행과 git 상태 변경도 없습니다. 박스에는 `/tmp/q3tail-sass`만 썼고 마지막에 지웠습니다.

## 2. 실행한 명령과 판정 줄

**임대·홀드 확인.** 박스 CPU 도구를 쓸 때마다 `flock -n /root/bloomery-cpu.lock true`와 `test ! -e /root/bloomery-ee-hold`를 먼저 통과시켰습니다.
```
try 3 21:09:11Z lease/hold rc: 1 0
try 4 21:10:11Z lease/hold rc: 0 0
FREE
try 1 21:11:32Z rc=3
try 2 21:12:33Z rc=0
```

**md5 비교 (Mac main 대 박스 `/root/repo/bloomery`).** 스펙이 지정한 일곱 파일에 flash.rs, fused.rs, router.rs를 더해 봤습니다.
```
ba6d1e003bb56173eada1fe05016b40d  flash_gqa_prefill.rs   (같음)
246bcc031b0a9a619461007db25c9ffe  rope_neox.rs           (같음)
7e4a5a2f0224393576d503cf9f9028b7  elem.rs                (같음)
3f65dec2c03c33e813069afb35aff489  lib.rs                 (같음)
459427cbbf1d8cfd0b69dda5fa1b36a2  gemm.rs                (같음)
89d1ec21372f2b98ac060f9c14239f17  arch/qwen3moe/experts.rs (같음)
85c0a5a6… (Mac) / 69f00251… (box)  arch/qwen3moe/ubatch.rs  ← 다름
500bd28e… (Mac) / e175b96e… (box)  arch/qwen3moe/router.rs  ← 다름
```
커널 SASS는 박스 PTX에서 가져왔습니다. ubatch.rs와 router.rs는 Mac 트리만 읽었고, router 커널의 SASS는 쓰지 않았습니다.

**ptxas·cuobjdump (박스 PTX → `/tmp/q3tail-sass`)**
```
ptxas_rc=0
Function properties for head_norm_neox_append
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Function properties for gqa_prefill_flash
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
cuobjdump_rc=0 lines=28608
=== gqa_prefill_flash: 6168 instructions
   loop 0xfbc0-0x14c70 (BRA): 1292 instr; HMMA=128 LDSM=64 LDGSTS=16 LDGDEPBAR=3 DEPBAR=2 MUFU=34 BAR=2 SHFL=4 FMUL=200 FADD=66 FMNMX=2 F2FP=16 HADD2=32 ISETP=166 FSETP=108 FSEL=68 LDS=48 STS=16 IMAD=95 IADD3=64 LEA=33 LOP3=16 BRA=9 CS2R=15 BSYNC=3 BSSY=3 WARPSYNC=4
=== head_norm_neox_append: 472 instructions
   f64-pipe ops: {'F2F.F64.F32': 2, 'DADD': 7, 'DMUL': 1, 'F2F.F32.F64': 1}
=== qwen3moe_combine: 352 instructions
   by class: MUFU=2 FADD=1 FFMA=17 ...   loop 0x630-0xad0 (@!P1BRA): 75 instr; FFMA=8 ...
```

**nsys sqlite (`nsys-qwen3moe-pp4096-n4-graph-201044.sqlite`, CUPTI 열)**
```
window kernels: 965 first-to-last ms 511.127 sum ms 510.064
kernel                           grid     n    tot_ms   mean_us    min_us    max_us   blk  regs  ssmem  dsmem
gqa_prefill_flash                2048    48    88.983   1853.81   1676.98   1888.18   128   255  34816      0
head_norm_neox_append          147456    48    21.861    455.44    408.49    462.53    64    30     16      0
qwen3moe_router_logits            512    48    13.494    281.12    251.72    284.87   512   127  49152      0
q3k_quantize_q8_1               65536    96     9.097     94.76 …
q3k_quantize_q8_1              131072    48     8.901    185.45 …
q3k_quantize_q8_1                  16     1     0.002      2.34 …
rms_norm                         4096    96    11.437    119.14 …
rms_norm                            1     1     0.005      5.47 …
gemm_route                          1    49     2.709     55.29     10.50     70.21  1024    32  32896      0
window graphNodeId non-null: 0 of 965 launchType set: Counter({1: 965})
kernels total / with graphNodeId: (2777, 1812)
launch API us: median 2.27 mean 2.39 max 14.03 sum 2.306 ms
kernel start - API end (ms): min -0.001 median 252.436 max 505.364
host enqueue span 5.404 ms vs window 511.127 ms
memcpy near window: [(-0.151, 2280448, 1, 128.2), (510.735, 8192, 8, 2.0), (511.133, 12, 2, 1.9), (511.156, 524, 1, 0.6)]
```
같은 파일의 enum 표로 읽으면 launchType 1은 `CUDA_KERNEL_LAUNCH_TYPE_REGULAR`이고, memcpy kind 1·2·8은 각각 HtoD·DtoH·DtoD입니다. 층별 줄에서 flash는 층 0이 1,701–1,707 µs, 뒤쪽 층이 1,869–1,874 µs이고, head_norm은 408 대 458 µs입니다.

**로컬 산술 (`calc.py`, Mac에서 코드 모양만으로 계산)**
```
T=4096 blocks=2048 block-tiles=66560 avg/slot=396.2 LPT=397.0 (+0.20%) ascending=427.0 (+7.8%) longest=64
T=512 blocks=256 block-tiles=1152 avg/slot=6.9 LPT=8.0 (+16.67%) ascending=11.0 (+60.4%) longest=8
T=4096 warp-tiles=266240 issued=139.586 GFLOP/layer (x48 6.700 TFLOP) exact=137.473 GFLOP/layer (x48 6.599 TFLOP) pad=1.54%
T=512 warp-tiles=4608 issued=2.416 GFLOP/layer (x48 0.116 TFLOP) exact=2.152 GFLOP/layer (x48 0.103 TFLOP) pad=12.28%
```

**정리.** `rm -rf /tmp/q3tail-sass; ls -d /tmp/q3tail-*`의 출력은 비어 있었습니다(rc 2는 남은 것이 없다는 뜻입니다).

## 3. 예측과 결과

### 3.1 꼬리를 커널별로 (T = 4096)

| 커널 (호출 자리) | launch 수 × grid × block | 바이트 R + W (MB) | 평균 µs | GB/s | 690 GB/s 바닥 µs | 부류: 유닛 수요 대 커널 사이클 |
|---|---|---|---|---|---|---|
| `qwen3moe_combine` (`ubatch.rs:600-611`) | 48 × 32,768 × 256 | 302.1 + 33.6 = 335.7 | 483.37 | 694 | 486.5 | 바이트: DRAM ≈ 100 % |
| `head_norm_neox_append` (`:491`) | 48 × 147,456 × 64 | 86.0 + 83.9 = 169.9 | 455.44 | 373 | 246.2 | **FP64**: 294,912 워프 × f64 파이프 명령 11 × 16 사이클 ÷ 84 SM = 617.9 K SM-사이클. 커널 사이클은 683–729 K(1.5–1.6 GHz)라 85–90 %. DRAM은 54 % |
| `gemm_swiglu_quant` (`:589`) | 48 × 196,608 × 32 | 201.3 + 79.4 = 280.8 | 407.35 | 689 | 406.9 | 바이트 |
| `q3k_quantize_q8_1` attn-norm `:474`, ffn-norm `:565` | 96 × 65,536 × 32 | 33.6 + 26.5 = 60.0 | 94.76 | 634 | 87.0 | 바이트 (92 %) |
| `q3k_quantize_q8_1` attn-out `:527` | 48 × 131,072 × 32 | 67.1 + 53.0 = 120.1 | 185.45 | 647 | 174.0 | 바이트 (94 %) |
| `q3k_quantize_q8_1` head (한 토큰) | 1 × 16 × 32 | — | 2.34 | | | 무시 |
| `rms_norm` attn `:465-473`, ffn `:556-564` | 96 × 4,096 × 256 | 33.6 + 33.6 = 67.1 | 119.14 | 563 | 97.3 | 바이트 82 %: 토큰당 한 블록이 8 KB를 두 번 훑고, 그 사이에 블록 전체 리덕션이 있어 블록마다 지연이 남음 |
| `rms_norm` head | 1 × 1 × 256 | | 5.47 | | | 지연 |
| `add` (`:538-539`) | 48 × 32,768 × 256 | 67.1 + 33.6 = 100.7 | 147.70 | 682 | 145.9 | 바이트 |
| `gemm_route` MoE (`:575`) | 48 × 1 × 1,024 | 작음 | ≈ 56.2 | | | 지연: 32,768 슬롯의 count·prefix·타일 목록을 블록 하나가 돌고 나머지 83 SM은 놂 |
| `gemm_route` dense (`:403` `enqueue_route_dense`) | 1 × 1 × 1,024 | | 10.50 | | | ubatch당 한 번 |
| `qwen3moe_router_route` (`:574`) | 48 × 512 × 256 | 2.1 + 2.4 = 4.5 | 23.97 | 186 | 6.5 | 지연: 토큰당 워프 하나가 128 logits 위에서 top-8 여덟 라운드를 직렬로 돎 |

- **launch 수의 내역**
  - quantize 145 = 96 × grid 65,536 + 48 × grid 131,072 + 1 × grid 16입니다. 블록 하나가 128값 q8_1 블록 하나(32 레인 × 4값)입니다.
  - rms_norm 97 = 96 + head 1입니다.
  - gemm_route 49 = MoE 48 + dense 1입니다.
- **꼬리 합계.** router_logits를 빼면 105.0 ms로, 커널 합의 20.6 %입니다.
- **죽은 쓰기**
  - GemmAct는 q3·q4 면을 값당 1 B씩 씁니다(`gemm.rs:1528-1529`).
  - 비중은 quantize 쓰기 26.5 MB 중 16.8, attn-out 53.0 중 33.6, swiglu 79.4 중 50.3입니다. 층당 117.4 MB입니다.
  - 읽는 곳은 `gemm.rs:2043-2045`(`&act.q6,` `&act.s8,` `&act.d8,`)뿐입니다.
  - GemmAct 인스턴스는 `ubatch.rs:115-164`의 셋(`act_hid`, `act_attn`, `act_h`)이 전부이고, head는 `Q8Act`를 씁니다(`head_argmax.rs:351`).

### 3.2 `gqa_prefill_flash`

**일의 양**

- **정확한 인과 FLOPs**(QKᵀ + PV, 쿼리·키 쌍 하나에 헤드당 4·128)는 P = 4096에서 층당 137.47 GFLOP(48층 6.599 TFLOP), P = 512에서 층당 2.152 GFLOP(0.103 TFLOP)입니다.
- **실제로 발행된 FLOPs**는 64키 타일을 통째로 돌기 때문에 139.59 / 2.416 GFLOP이고, 패딩은 1.54 % / 12.3 %입니다.
- **달성 처리율**
  - 139.59 G ÷ 1.8538 ms = 75.3 TFLOPS(발행 기준)입니다.
  - A6000 dense FP16·FP32 누적 피크 154.8 TFLOPS(1.80 GHz 부스트, GA102 whitepaper Table 3, `docs/research/q3gemm-lit-report.md:56`에 인용)의 48.6 %입니다.
  - 전력 캡 아래 클록 1.5–1.6 GHz의 피크(129–138 TFLOPS) 기준으로는 55–58 %입니다.
- **바이트**
  - Q는 f32로 67.1 MB 읽고, Y는 f32로 67.1 MB 씁니다. K/V는 f16이고 고유 바이트는 8.4 MB입니다.
  - L2→SM 트래픽은 블록-타일 66,560 × 32 KB = launch당 2.18 GB(1.18 TB/s)입니다.
  - DRAM은 launch당 ≈ 150–165 MB로, 약 12 %입니다[유도].

**블록 하나의 스텝**

- **타일 모양.** q 타일은 `POSITIONS` 8위치 × GQA 8헤드 = 64행이고, 4워프가 16행씩 맡습니다. 키 타일 `KEY_TILE`은 64이고 절대 위치에 정렬됩니다.
- **워프-타일당 mma**는 QK 8×8 = 64에 PV 16×4 = 64를 더해 `HMMA.16816.F32` 128개입니다.
- **루프 한 바퀴의 나머지 1,164 명령**
  - 소프트맥스 부동소수 ≈ 530: FMUL 200, FSETP 108, FSEL 68, FADD 66, MUFU 34, HADD2 32, F2FP 16.
  - 정수·주소 374: ISETP 166, IMAD 95, IADD3 64, LEA 33, LOP3 16.
  - 적재 149: LDSM 64, LDS 48, LDGSTS 16, STS 16, DEPBAR 5.
  - 제어 21, 그 외 ≈ 90.
- **배리어 사이의 순서.** 다음 K를 cp.async로 스테이징 → V 대기 → QK HMMA 64 → 마스크, 러닝 max, exp, f16 팩, 행 합 → O 재스케일(FMUL 64) → PV HMMA 64 → 배리어. V는 QK 동안, 다음 K는 PV 동안 옵니다(2단, 배리어 2개).
- **자원.** 레지스터 255(CUPTI), smem 34,816 B 정적, spill 0입니다.
- **점유율**
  - 레지스터 할당은 256 × 128 = 블록당 32,768이라 65,536 ÷ 32,768 = 2블록입니다.
  - smem은 (34,816 + 1,024) × 2 = 71,680 ≤ 102,400이라 2블록이 들어갑니다. 3블록이면 107,520이라 안 됩니다.
  - 결과적으로 SM당 8워프(48 중 16.7 %)이고, 스케줄러당 2워프(블록마다 하나)입니다.

**그리드와 LPT**

- 그리드는 2,048(q 타일 512 × KV 헤드 4)이고 슬롯 168개에 나뉘어 슬롯당 12.2블록입니다.
- 블록 비용은 키 타일 수 floor(qt/8)+1에 비례하고 1에서 64까지이므로, 첫 q 타일과 마지막 q 타일의 비용 비는 1:64입니다.
- deepest-first 순서(`flash_gqa_prefill.rs:164`, `qt = n_tiles - 1 - b / nkv`)는 LPT 스케줄입니다.
  - 메이크스팬은 슬롯당 397.0 타일로 완전 균형보다 +0.20 %입니다. 오름차순이었다면 427.0(+7.8 %)입니다.
  - 그래서 P = 4096에서는 launch의 7.0 %(창 약 6.2 ms)를, P = 512에서는 27 %를 벌고 있습니다.

**모형과 적합**

- **한 짝의 수요(스케줄러 하나에서 두 워프가 타일을 하나씩)**
  - 텐서: 2 × 128 × 16 = 4,096 사이클. 파티션당 128 FMA/clk인 전속으로 읽은 값이고, 84 × 4 × 128 × 2 × 1.8 GHz = 154.8 T와 맞습니다. 반속으로 읽으면 이 커널이 피크의 97 %여야 하므로 배제했습니다.
  - 비-HMMA 발행: 2 × 1,164.
  - 배리어와 파이프 채움·비움: ≈ 260.
- **두 모형**
  - 합 모형: 6,684 사이클.
  - max 모형: ≈ 4,300–4,500 사이클.
- **실측.** 1.8538 ms × f ÷ (396.2 × 1.033) = 6,795(1.5 GHz)…7,247(1.6 GHz)입니다. 1.033은 LPT +0.2 %와, 블록당 프롤로그·에필로그가 타일 하나쯤 드는 몫을 합친 값입니다.
- **예측.** 2.736 M 사이클 → 1.71–1.82 ms이고, 1.854 대비 −2…−8 %로 ±15 % 안입니다.
  - 거꾸로 읽으면 합 모형은 뒤쪽 층에서 1.46 GHz, 층 0에서 1.61 GHz를 뜻합니다.
  - 이것은 다른 증거와 같은 대역입니다. ncu 증인은 1,515 MHz @ 299.3 W였고, head_norm의 FP64 수요는 층 0에서 1.51 GHz 이상을 요구합니다.
  - 남은 차이는 클록 불확실성과 구별되지 않습니다.
- **max 모형 기각.** max 모형이면 1.10–1.23 ms(−34…−41 %)가 나와야 하므로 기각합니다.
- **가설 두 개.** 어느 쪽인지는 §4의 ncu 한 번이 가릅니다.
  - ① 락스텝: LPT가 길이가 같은 블록을 32개씩(qt 8개 × KV 헤드 4) 내보냅니다. 그래서 한 SM의 두 슬롯이 같이 시작하고, 두 블록의 HMMA 구간이 겹칩니다.
  - ② 사슬: q3gemmlat이 쓴 「명령 수 × CPI」 형입니다. 워프마다 자기 의존 사슬이 스텝을 정하고, CPI는 5.3–5.6입니다.
  - 두 가설 모두 비-HMMA 명령 수를 레버로 만듭니다. 위상까지 레버로 만드는 것은 ①뿐입니다.
- **유닛 수요 대 스텝**
  - 텐서 56–60 %, 발행 36–38 %, MUFU 8 %, smem ≈ 36–45 %, L2→SM ≈ 55–60 %, DRAM ≈ 12 %.
  - 가장 큰 수요가 스텝의 60 %이므로 AGENTS 09-26 규칙에 따라 한 유닛의 항만 줄이는 레버는 0으로 예측됩니다.
- **천장.** 텐서 바운드(짝당 4,096 사이클)면 launch당 1.05–1.12 ms이고, 창에서 −35…−38 ms(7 %)가 flash에서 딸 수 있는 전부입니다.

**레버 (창 ms, P = 4096)**

| 레버 | Δ창 | 레지스터 | smem | 점유율 | 비트 |
|---|---|---|---|---|---|
| **FA** 명령 수 줄이기 (내용은 표 아래) | 비-HMMA 1,164 → ≈ 510–700, 짝당 −900…−1,300 사이클 → **−9…−17 ms** | ≤ 255 | 같음 | 2블록/SM | **유지** (조건은 표 아래) |
| **FS** 스태거: SM별 도착 카운터(`%smid` + atomic)로 두 번째 블록만 첫 웨이브에서 반 타일 한 번 대기 | ①이면 −22…−31 ms, ②면 0 | +2 | 0 | 같음 | 유지 |
| **FB** 핑퐁: 8워프 블록을 64행 그룹 둘로 나누고, named barrier로 반대 위상에 두고, K/V 링 3–4단을 공유 | FA 전 −22…−33, FA 뒤에는 천장 근처 | 255 × 256 → 1블록/SM (SM당 8워프로 같음) | 52–70 KB | 1블록 | 유지 |
| q 타일 16위치 (128행, 8워프) | ≈ 0 (±3 %): L2 트래픽이 반으로 줄지만 L2가 바운드가 아님. 같은 스케줄러의 두 워프가 한 블록이라 배리어로 위상이 묶임 | 1블록 | 34,816 | 8워프 | 유지 |
| q 타일 4위치 | 나빠짐: smem 때문에 2블록 → SM당 4워프, L2 트래픽 2배 | | | | 유지 |
| 키 타일 128 | 나빠짐: S/P 레지스터 +32로 255 초과(spill), smem 69.6 KB로 스케줄러당 1워프 | > 255 | 69,632 | 4워프 | 이동 |
| 키 타일 32 | 나빠짐: HMMA당 배리어·소프트맥스 고정비 2배. 레지스터 때문에 2블록에 묶임 | | 17,408 | | 이동 |
| cp.async 3단 | 0: 적재는 이미 가려져 있음. 게다가 2 × 53,248 = 106,496 > 102,400이라 들어가지 않음 | | +17,408 | | 유지 |
| 긴 타일 split-KV | P = 4096에서 되찾을 몫이 0.2 %뿐. P = 512에서도 ≈ 0.25 ms에 합치기 패스가 추가됨 | | | | 이동 (대역 + PIN) |
| K/V 레이아웃 | 0: 헤드 우선 행(`rope_neox.rs:8-10`)이라 타일당 16 KB가 연속이고, `ROW_W` 68 패딩으로 뱅크 충돌 없음 | | | | 유지 |
| 3블록/SM (레지스터 ≤ 168, 스위즐 33.8 KB) | −15…−25 %. 레지스터 다이어트 위험(주소를 빼고도 Q 32 + O 64 + S 32) | ≤ 168 | ≤ 33,792 | 12워프 | 유지 |
| f16 O 누적 (llama.cpp의 선택) | ≈ 0 (A6000에서는 같은 속도) | −32 | | | 이동, 정밀도 하락 ✗ |

- **FA에 들어가는 변경**
  - 내부 타일은 원소별 마스크(`:377`, `:423`, `:428`)를 거치지 않는 빠른 경로로 돕니다.
  - 키와 개수를 `usize`에서 `u32`로 바꿉니다.
  - `f32::max`를 `max.f32` 한 명령으로 바꿉니다. 지금은 cuda-oxide가 4명령으로 내립니다.
  - f16 가중치의 exp에만 `ex2.approx.ftz`를 씁니다.
  - cp.async 주소를 32비트 증분으로 계산하고, 내부 타일에서는 `ka < hi` 검사를 뺍니다.
  - 워프 vote로 max가 바뀐 행이 없으면 재스케일을 건너뜁니다.
- **FA가 비트를 유지하는 조건**
  - `ex2.approx(0) = 1`이 정확해야 합니다.
  - ftz는 f16 반올림이 어차피 0으로 만드는 값(2⁻²⁵ 미만)에만 적용하고, 재스케일 인자에는 쓰지 않습니다.
  - `max.f32`의 NaN 의미는 `f32::max`와 같고, 0의 부호는 exp(s − m)의 비트에 닿지 않습니다.
- **오늘 flash를 거는 게이트 절**
  - `gate_qwen3moe_flash.rs:69-81`: 한 행 launch가 정확값·ik `fa-L` 대역 안에 있을 것, 재실행과 NaN 패딩에서 비트 동일, 행이 단독 launch와 비트 동일(시드 입력이 타일 경계 63/64/65, 127/128/129를 지남), `key_count` 결함, 캡처 재생 비트 동일.
  - e2e 게이트: (u)(`:52-65`, 대역 `:160`·`:177`의 `PIN(2026-09-25)`)와 (w)(`:66-70`), 그리고 `BLOOMERY_GQA_MMA=0` 쌍둥이 비교.
  - **이 절들은 전부 관계형이거나 대역입니다. prefill의 절대 비트를 못박는 절은 없습니다.** `--dump`(`:86-89`)도 pass 경로만 덤프하므로(`:863-870`), 비트 동일을 주장하는 ubatch 라운드에는 교차 빌드 비교 수단이 없습니다(§5-1).

### 3.3 fold 사다리

c_node는 창의 간격 1.06 ms ÷ 964 = 1.1 µs로 측정했습니다.

| fold | 없어지는 커널 µs/층 | 호스트 그리드가 새로 떠안는 일 | Δ µs/층 → 창 | 비트 | 호스트 레지스터·발행 | 판정 |
|---|---|---|---|---|---|---|
| **F1** swiglu_quant → gate/up GEMM 에필로그 | 407.35 | 128행 타일이 q8_1 블록 하나라 모양은 맞음. 그러나 같은 (슬롯, ff)의 gate와 up을 한 블록이 다 가져야 함 → 누산기 32 → 64로 128 레지스터 캡(`launch_bounds(256,2)`) 초과(아니면 smem +21.5 KB) → 1블록/SM. 지연 바운드 GEMM(q3gemmlat: CPI 7.26)이 워프를 절반 잃어 층당 +0.54…+1.36 ms | 순손실 +0.2…+1.0 ms/층 | 유지 가능 | 레지스터 다이어트와 정면 충돌 | ✗ |
| **F2** rms_norm + quantize를 한 패스로 (두 자리, q6만) | 두 자리 각 213.9 → attn 65–79, ffn 108–134 (router가 `a.normed`를 읽으므로 ffn은 normed도 써야 함) | 토큰당 한 블록(256 × 8값). q8_1 블록이 토큰 안에 있으므로 반복·대기 없음 | Q 뒤 −161…−201 → **−7.7…−9.6 ms** | 유지: `rms_partial_sq` → `rms_warp_tree` → `rms_scale` 순서와 같은 기하를 재현하고 `q8_1_quant_vals`를 그대로 씀 | 새 커널 | 염 (M) |
| **F3** attn-out quantize → flash 에필로그 | 185.45 (Q 뒤 133.7) | launch당 +≈17 µs: 비는 KT/VT smem(32 KB)으로 재배치한 뒤 행마다 한 번 양자화(헤드 하나 = 128값 블록). 블록마다 다시 양자화하던 qwen3fuse와 달리 반복이 없음. Y f32 쓰기는 사라짐 | Q 뒤 −118 → **−5.7 ms** | 유지. codes = q8_1(y) 절 신설 필요 | 루프 뒤라 255 그대로 | 염 (FA 라운드 안이면 S) |
| **F4** add → O GEMM 에필로그 | 147.70 | O GEMM(grid 1,040, 886.9 µs, 지연 바운드)에서 블록당 한 번 스레드당 LDG 32 + FADD 32 + 주소 ≈ 16 (+0.3 %). resid 33.6 MB 읽기는 가려짐 → +3…+12 µs | −137…−146 → **−6.6…−7.0 ms** | 유지 (덧셈은 교환법칙이 성립, 피연산자 순서만 맞춤) | 에필로그 전용이라 k 루프 할당 불변. 별도 엔트리 | 염 (S–M) |
| 대안: add → FFN rms_norm | | x 재읽기 33.5 MB | −49 → −2.4 ms | 유지 | | F4보다 약함 |
| **F5′** combine → 다음 층 norm+quant (F2 뒤) | x 재읽기 ≈ 49 | combine을 토큰당 블록으로 다시 짬 | −1.2…−2.4 ms | 유지 (FFMA 축약은 SASS에서 확인, §5-2) | | 나중 |
| combine → down GEMM | 483.4 | 한 토큰의 8슬롯이 서로 다른 블록 8개에 있어 블록 간 합이 필요. 결정적인 형태는 오늘의 고정 순서 두 번째 패스뿐이고 atomics는 순서가 비결정적 ✗. 줄일 수 있는 것은 16비트 슬롯 행(−9.4 ms [유도]) | | 이동 (f16은 넘침 위험, bf16은 가수 손실) | | fold ✗, 16비트 슬롯은 순위 낮음 |
| **F6** head_norm → QKV 에필로그 | 455.44 | 128행 타일 = 헤드 하나라 헤드는 통째로 들어옴. 그러나 8워프에 걸친 smem 리덕션, (i, i+64) rope 쌍 교환, f64 트리가 128 레지스터 GEMM 안에 들어가야 함 | | | | H1 + H2가 대체 |

**ubatch 경로가 오늘 두 launch로 도는 이유**

- `enqueue_norm_quant`(`fused.rs:595-655`)는 `&mut Q8Act`를 받습니다. `Q8Act`는 decode 형식이고 열 수가 `Q8ACT_MAX_SLOTS` 3,072 이하입니다(`tensor.rs:194-200`). ubatch가 쓰는 `GemmAct`는 열이 32,768까지이고 면 배치도 다릅니다.
- `norm_quant`의 기하는 decode용(토큰당 1,024 스레드)입니다. T = 4096에 그대로 쓰면 지연 바운드(≈ 150–200 µs [유도])라 지금의 두 launch보다 낫지 않습니다.
- 거부 의미도 서로 달랐습니다.

### 3.4 자원 시간선 (층당, P = 4096)

| 자원 | 바쁜 시간 | 근거 |
|---|---|---|
| 카드 SM | 10.63 ms | 커널 20개가 한 스트림에서 직렬로 돌고, 각각 앞 커널의 출력을 읽음 |
| 호스트 CPU | 0.11 ms | 프롬프트 전체 launch 965개를 5.404 ms에 넣음(API 평균 2.39 µs). 카드보다 최대 505 ms 앞서므로 카드의 그늘 아래 |
| PCIe | ≈ 0 | 첫 커널 0.151 ms 전 HtoD 2,280,448 B(128 µs), 끝에 12 B + 524 B |
| 호스트 DRAM, NVMe | 0 | 가중치가 카드에 상주하고, Qwen3 경로에는 호스트 티어가 없음 |
| 커널 사이 간격 | 0.022 ms | 창 전체 1.06 ms (0.2 %) |

- **벽시계는 카드 커널의 합에 간격을 더한 것**이고, 흐름을 바꿀 레버는 카드 안에만 있습니다.
- **그래프가 아닙니다.** 창의 커널은 전부 일반 launch이고, 파일 전체의 그래프 노드 1,812개는 decode 스텝입니다. 로그의 `capture prefill_graphs=8`은 m ≤ 8인 pass 그래프를 가리킵니다. ubatch를 그래프로 캡처해도 얻는 것은 간격 몫(0.2 %)이 상한입니다.
- **두 번째 스트림**(quantize와 router logits를 나란히, `ubatch.rs:564-573`)
  - router logits는 블록당 65,024 레지스터를 써서 SM당 1블록이고, 남는 레지스터가 512개라 같은 SM에 quantize가 들어갈 수 없습니다.
  - 겹칠 수 있는 곳은 router의 마지막 웨이브뿐입니다. 512 ÷ 84 = 6.1이라 마지막에 8블록이 약 46 µs를 돕니다. 그 동안 나머지 SM에서 quantize를 돌리면 층당 ≤ 42 µs, 창 ≤ 2.0 ms를 법니다.
  - 둘 다 바이트 바운드인 짝은 ≈ 0입니다. F2가 들어오면 quantize가 norm 패스 안으로 사라지므로 이 레버도 사라집니다.

### 3.5 순위와 예측

| # | 레버 | Δ창 (P = 4096) | 각 끝을 정하는 항 | 크기 | 소유 게이트 | 비트 | 선행 조건 |
|---|---|---|---|---|---|---|---|
| 1 | **H1** head_norm 합을 f32로 (`elem::rms_norm`, llama.cpp `norm.cu:77`과 같은 부류) | −7.4…−10.0 (1.4–2.0 %) | 아래 끝: 발행 상한 414 K 사이클(정적 472 명령 기준, ≤ 276 µs). 위 끝: DRAM 바닥 246 µs | S | qwen3moe-kernels(qknorm), qwen3moe-e2e | **이동**. decode도 같이 움직임(`dispatch.rs:298`). `rope_neox.rs:12-30`을 다시 쓰고, `NORM_BAND` 10u와 호스트 전사를 다시 유도하고, e2e 대역을 재확인 | 없음 (파일이 겹치는 라운드 없음) |
| 2 | **Q** GemmAct에 q6·s8·d8만 씀 (GemmAct 전용 quantize 엔트리 신설, decode PTX는 불변) | −8.2…−8.5 (1.6–1.7 %) | 커널마다 자기 실측 GB/s로 계산 | S | gemm(`gate_gemm.rs:1553-1555`), qwen3moe-kernels, e2e, ptx-scan | 유지 | q3fix3 |
| 3 | **FA** flash 명령 수 줄이기 | −9…−17 (1.8–3.3 %) | 아래 끝: 제거한 명령 일부가 원래 HMMA 그늘에 있던 경우 | M | flash prefill 절, e2e, 덤프 추가(§5-1) | 유지. SASS 루프 1,292 → ≤ 750을 타이밍 전에 컴파일로 확인 | 없음 |
| 4 | **F4** add → O GEMM 에필로그 | −6.6…−7.0 | 에필로그 비용 +3…+12 µs | S–M | gemm(resid 절), e2e | 유지 | q3fix3, q3gemmb |
| 5 | **H2** q-노름 + rope → flash 프롤로그 (flash가 재현할 수 있는 트리를 H1에서 고름: 행이 쿼드 4레인 × 32값, rope 쌍은 같은 레인의 청크 c와 c+4) | −9…−12 | head_norm이 k·v만 처리(39–50 µs), 프롤로그 +4…+10 µs | M | qknorm, flash, e2e | H1과 비트 동일 | H1, FA |
| 6 | **F2** norm + quant 한 패스 | −7.7…−9.6 | ffn 자리의 normed 쓰기 | M | qwen3moe-kernels, gemm, e2e | 유지 | q3fix3, Q |
| 7 | **F3** attn-out quantize → flash 에필로그 | −5.7 | 에필로그 +17 µs | M (FA 안이면 S) | flash (절 신설), e2e | 유지 | q3fix3, FA |
| 8 | **FS / FB** 위상 | FS: ①이면 −22…−31, ②면 0 | ncu 스톨 구성 | S / L | flash, e2e | 유지 | ncu 판정 |
| 9 | F5′ | −1.2…−2.4 | | M | experts, e2e | 유지 | F2 |
| — | 기타 | QKV 한 launch ≈ −3.8(GEMM 쪽 일), gemm_route 다중 블록 ≤ 2, 두 번째 스트림 ≤ 2.0(F2와 충돌), 16비트 슬롯 −9.4(비트 이동) | | | | | |

**예측**[유도]

- **상위 둘(H1 + Q):** 창 −15.6…−18.5 ms → pp4096 **8,280–8,330**.
  - P = 512에서도 두 레버의 그리드가 8,192블록 이상이라 T에 선형으로 줄어듭니다. −1.95…−2.31 ms → pp512 **7,000–7,030**.
- **상위 다섯:** pp4096 8,710–8,980, pp512 7,210–7,300.
- **1–7번과 9번 전부:** pp4096 8,990–9,350, pp512 7,400–7,540.
- **q3gemmlat 예측과의 합산:** 시간은 더해집니다. q3gemmlat의 GEMM 레버만으로 −58.9 ms이고, 상위 둘을 얹으면 ≈ 9,420, 사다리 전체를 얹으면 ≈ 10,500입니다.

**P = 512를 ÷8로 줄이는 계산이 틀리는 곳**

- **flash:** LPT 꼬리가 +16.7 %, 패딩이 12.3 %라 flash는 ≈ 1.8 ms/창(2.4 %)입니다. flash 레버는 P = 512에서 거의 0입니다.
- **웨이브가 두 개 미만인 커널:** gemm_route, router 둘, rms_norm(512블록)은 ÷8보다 덜 줄어듭니다. 그래서 F2의 P = 512 수치는 하한으로 읽어야 합니다.
- **GEMM:** T에 따라 5.2배만 줄고 8배가 되지 않습니다.
- **간격:** T와 무관하게 1.06 ms라 P = 512 창의 1.4 %입니다.
- 이 네 항 때문에 실측 P = 512 창 75.1 ms가 선형 축소값(≈ 54.4 ms)보다 약 20.7 ms 깁니다.

### 3.6 엔진 비교

**llama.cpp** (`/home/user/llama.cpp-mainline`. `git log`이 "detected dubious ownership"으로 막혀 커밋을 읽지 못했습니다)

- **flash 타일은 우리와 같습니다.**
  - Ampere 설정은 `fattn-mma-f16.cuh:59-62` `GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 2, 128,  64,  64,  64, 2, true);`이고, GQA 비가 4보다 크면(`gqa_ratio > 4`) ncols2 = 8이 됩니다.
  - 결과는 64열, 4워프, 2블록/SM, 64키, 2단, Q를 레지스터에 두는 구성입니다.
- **다른 점**
  - VKQ를 half2로 누적합니다(`:1090` `using T_C_VKQ = tile<16,  8, half2>;`). 우리 f32 O 계약과 다릅니다.
  - exp는 `expf`이고(`:803`), 재스케일에 FTZ 문턱이 있습니다(`:913`).
  - 가려진 KV 타일은 `flash_attn_mask_to_KV_max`(`fattn-common.cuh:666`)로 건너뜁니다.
  - stream-k는 Ada 이전 NVIDIA에서 `tiles_efficiency_percent < 75`일 때만 켜지는데, 이 모양은 93.8 %라 A6000에서는 꺼져 있습니다.
- **융합**
  - `rms_norm_f32<block_size, do_multiply, do_add>`(`norm.cu:77`)는 f32로 합칩니다.
  - norm+mul+rope(`ggml-cuda.cu:2704`)와 rope+set_rows(`:2670`)가 있고, 둘 다 f32입니다. H1과 같은 방향입니다.
  - GLU를 matmul에 접는 것은 벡터 경로(`:3763-3773`)뿐입니다. prefill에서는 `ggml_cuda_op_swiglu`(`:2203`)를 돌고 나서 `quantize_mmq_q8_1_cuda`(`mmq.cu:156`, `:236`)를 따로 돕니다. 우리가 한 패스로 하는 일이 거기서는 두 패스입니다.
  - 전문가 합은 `moe_weighted_reduction_f32`로, 열마다 순서대로 더하는 우리와 같은 부류입니다.
- 처리량은 mmq GEMM에서 나옵니다.

**mistral.rs**

- **flash:** FA2 포트이고, sm8x causal hdim128에서 `Flash_fwd_kernel_traits<Headdim, 64, 64, 4, …>`를 씁니다(`flash_fwd_launch_template.h:274`, 주석 `:268` "64 x 64 is the fastest for causal").
  - CTA 하나가 쿼리 헤드 하나를 맡으므로 K/V L2 트래픽이 우리의 8배입니다.
  - 공개된 pp 행은 flash-attn 없이 잰 값입니다(`mrs-pp4096-report.md:18`).
- ~~**MoE:** `let down_in = (up * gate.apply(&config.act)?)?;`(`moe/experts/backends.rs:1181`)로 활성을 따로 돕니다. combine은 `to_dtype(F32).broadcast_mul(..).sum(D::Minus2)?.to_dtype(..)`(`:1214-1217`)로 네 패스이고 bf16이라 우리 계약과 다릅니다.~~ 정정(09-26, mrsq3): 위는 gather 경로다. 32토큰 이상 프롬프트는 grouped 경로(`backends.rs:1121-1133`)로 가서, gate·up 뒤 `quantize_mmq_q8_1_glu`가 SiLU·곱·양자화를 한 번에 하고, 가중합은 `moe_weighted_reduce_flat_bf16` 한 패스다(`moe_grouped.cu:679-703`).

**exllamav3**

- **attention:** Triton이고 설정은 `cfg = (128, 32, 8, 2) if blackwell else (128, 64, 8, 2)`(`triton_paged.py`)입니다. 128행이라 우리 레버 표의 「q 타일 16위치」에 해당합니다.
- **MoE:** 융합 커널 `ext.exl3_moe` 안에서 활성을 돕니다(`exl3_moe_kernel.cuh:127` `const bool gated = act_function != MOE_ACT_RELU2_NOGATE;`). 에필로그 안인지는 읽지 않았습니다.
  - combine `exl3_moe_gather_kernel`(`exl3_moe.cu:387`, "every column thread streams its column of the listed slots in k order")은 결정적입니다.
  - 비융합 경로의 `index_add_`는 atomic이라 순서가 비결정적입니다 ✗.

**정리**

- 셋 중 어느 엔진도 prefill에서 norm+quant를 접지 않고, combine을 down GEMM에 접지도 않습니다. 셋 다 토큰당 고정 순서로 모으는 결정적 형태입니다.
- 우리 비트 계약과 속도를 동시에 이기는 선택은 셋에 없습니다.
- H1은 우리 q-노름을 llama.cpp CUDA 노름과 같은 f32 부류로 옮기는 것입니다.

### 3.7 비행 중 라운드와 만나는 곳

- **q3fix3**
  - Q·F2·F3는 모두 `q8_1_quant_vals`(`lib.rs:420`)를 지나는데, q3fix3가 이 함수를 거부의 단일 소유자로 만드는 중입니다. 그래서 세 레버는 q3fix3 뒤에 옵니다.
  - q6 전용 변형은 소유자 위에 면 마스크를 얹는 식으로 만들고, 검사를 복제하지 않습니다.
  - Q의 swiglu 부분은 q3fix3가 지금 고치는 `gemm_swiglu_quant`(`gemm.rs:1432-1492`)의 쓰기 경로 그 자체입니다.
  - F3는 flash가 이미 가진 싱크(`flash_gqa_prefill.rs:151`, `:213`)를 씁니다.
- **q3gemmlat / q3gemmb**
  - F4는 `gemm.rs:784`의 에필로그에 명령 ≈ +80, 에필로그 전용 레지스터를 더합니다.
  - q3gemmb가 SASS 스텝 수를 래칫으로 쓰니, 다른 GEMM 엔트리의 SASS가 움직이지 않도록 별도 엔트리로 둡니다.
  - F1은 누산기를 두 배로 만들어 레지스터 다이어트와 정면으로 부딪칩니다.
  - FA의 방법(64비트 `usize` 제거, 분기 없는 풀 타일, 컴파일 시점 명령 수 래칫)은 q3gemmb와 같습니다.

## 4. 못 한 것

- **ncu를 돌리지 않았습니다.** 그래서 클록(1.46–1.61 GHz는 유도값)과 flash의 두 가설이 아직 확정되지 않았습니다. 유도로 닫히지 않는 항은 이것 하나이고, 닫는 실행도 하나입니다.
  - 실행: ncu로 `gqa_prefill_flash` 한 launch(층 24, P = 4096)를 봅니다.
  - 기대값
    - `sm__cycles_elapsed.avg.per_second` 1.46–1.55 GHz
    - 텐서 파이프 활성 56–60 %
    - 스케줄러당 발행 0.36–0.40
  - 스톨 구성으로 판정
    - `math_pipe_throttle`이 25 % 이상이면 ①(락스텝) → FS를 시도합니다.
    - `short_scoreboard` + `wait`가 지배하면 ②(사슬) → FA만 합니다.
    - `no_instruction`이 10 %를 넘으면 i-cache 문제입니다(루프 본문 20.7 KB).
    - 텐서가 90 % 이상이면 전속 읽기가 틀린 것이고 FA ≈ 0입니다.
  - 임대가 필요하므로 리드의 시팅에서 돌려야 합니다.
- **FA가 제거할 명령 수(≈ 450–650)는 추정입니다.** 루프 SASS의 부류별 개수에서 셌고, 정확한 값은 바꾼 코드를 컴파일해야 나옵니다.
- **head_norm의 동적 명령 수를 모릅니다.** 정적 472는 상한이고, H1 대역의 위 끝이 이 상한에서 나왔습니다.
- **gemm_route와 router_route는 "지연"까지만 분해했습니다.** 둘을 합쳐 창의 0.8 %입니다.
- **flash 루프의 LDS 48개가 어느 소스 적재인지 추적하지 못했습니다.** FA 라운드에서 셀 항목입니다.
- **advisor 검토가 없습니다.** 앞에서는 도구가 없었고, 마지막 호출은 시간 초과로 끝났습니다.
- **박스의 ubatch.rs와 router.rs가 main과 달랐습니다.** 그래서 두 파일은 Mac 트리만 읽었습니다.

## 5. 스펙 밖 개선 지점 (보고만 하고 손대지 않았습니다)

1. `crates/gpu-gates/src/bin/gate_qwen3moe_e2e.rs:86-89`: `--dump`에 GEMM ubatch 경로가 없습니다. (u)의 logits와 K/V를 덤프에 넣으면 ubatch 라운드의 교차 빌드 md5 비교가 됩니다 (S).
2. `crates/gpu/src/elem.rs:273-276`: 주석은 "one plain multiply then add per term (no fused multiply-add)"라고 하지만, `acc += wv * dv;`(`:302`)는 SASS에서 FFMA로 컴파일됩니다(combine의 FFMA 17개). 주석이 틀렸거나, 원래 의도대로라면 `mul_rn_f32`가 필요합니다. 후자는 비트를 옮깁니다 (XS/S).
3. cuda-oxide가 `f32::max`를 4명령으로 내립니다. flash에서 워프-타일당 32번 나옵니다. `docs/upstream/nvlabs-ledger.md`에 한 줄 올릴 감입니다 (S).
4. `crates/gpu/src/flash_gqa_prefill.rs`: 루프 안에서 `usize`로 비교하고 주소를 계산합니다. FA에 포함됩니다 (S).
5. `crates/gpu/src/gemm.rs:1528-1529`: 아무도 읽지 않는 q3·q4 면에 117.4 MB를 할당하고, 층마다 117.4 MB를 씁니다. Q에 포함됩니다 (S).
6. `crates/gpu-gates/src/bin/gate_gemm.rs:1553-1555`: 읽는 곳이 없는 면을 못박고 있습니다 (XS).
7. `crates/gpu/src/gemm.rs:1480`: refusal을 올리고도 양자화해서 씁니다. 이미 q3fix3 1번이 소유한 항목입니다.
8. `crates/gpu/src/elem.rs:427-475`: rms_norm이 82 %에 머뭅니다. F2가 대체합니다 (S).
9. `crates/gpu/src/gemm.rs:927-1124`: gemm_route가 블록 하나로 돕니다. 층당 56 µs이고, 다중 블록으로 바꾸면 창 ≤ 2 ms를 법니다 (S–M).
10. head의 rms_norm(grid 1)과 quantize(grid 16)는 한 토큰짜리라 `norm_quant` 한 launch로 충분합니다 (XS).

## 6. 모델

opus(Opus 5.5)로 스폰되어 그대로 돌았습니다.
