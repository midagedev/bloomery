# 동류·업스트림 조사 원본 (2026-09-21)

agy(gemini-3.8-flash-high) 조사 라운드 여섯의 보고서 원문이다. 영어 그대로 둔다 — 업스트림과 남의 코드를 가리키는 글이다. **읽는 법**: 코드 인용(`path:line`)과 이슈 번호는 조사자가 읽은 것이고, 속도·우열 판정은 전부 조사자의 추론이다(어느 쪽도 돌려 보지 않았다). 리드가 원본에서 다시 확인한 것만 아래에 적는다. 나머지는 쓰기 전에 확인한다.

| 파일 | 무엇 | 리드가 원본에서 확인한 것 |
|---|---|---|
| `census-cuda-oxide.md` | cuda-oxide 공개 사용처 전수(24개), LLM 추론 셋, 남들이 밟은 결함 | FeLLM의 커널 파일과 함수들(5,332줄), localMoE의 `#[kernel]` 52개, 두 저장소의 존재·날짜 |
| `census-cutile.md` | cutile-rs 사용처, mistral.rs·candle에서 cutile이 들어오는 자리 | candle #3934·mistral.rs #2413의 존재와 머지 날짜, mistral.rs cutile 디렉터리에 K-quant 파일이 없다는 것 |
| `fellm-vs-bloomery.md` | FeLLM과 우리 커널을 나란히 | FeLLM의 Q6_K가 `q6k_gemv_row`로, 배치 1 MoE가 행당 스레드 하나의 스칼라 `dot_q4k`로 간다는 것 |
| `localmoe-v4.md` | DeepSeek V4를 cuda-oxide로 돌린 코드에서 V4.1 단계가 물려받을 것 | 없음(전부 미확인) — V4 수식은 공식 참조 구현과 대조한 뒤에 쓴다 |
| `upstream-cuda-oxide.md` | 장부 #1·#4의 제출 전 조사 | `const_fold.rs`의 시프트 폴딩과 `convert_shift`, 그리고 그 위에서 만든 패치의 끝단 A/B |
| `upstream-cutile-graph.md` | 장부 #3의 제출 전 조사 | cuda-oxide #107의 메인테이너 답, #585 닫힘, cutile-rs #270 열림 |

가져올 후보(처분은 HANDOFF): 잔차를 gemv 저장에 접기(P0b에 들어 있음), 활성 블록 128 → 32(FeLLM도 32 — 블록 게이트 뒤에 판단), m=1에서 `rms_norm`이 워프 8개를 띄우고 7개가 바로 돌아가는 기하, V4의 FP4 → E4M3 레지스터 LUT·Sinkhorn 20회·√softplus 라우터·어텐션 싱크·라우터 구동 전문가 프리페치. 베끼지 않을 것: sm_89 전용 MMA 인트린식(3090·A6000에 없다), 전문가 호출마다의 할당, `read_volatile` 우회.
