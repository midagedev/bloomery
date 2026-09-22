# bloomery — agent context

@AGENTS.md

위 파일이 이 레포의 작업 계약이고 정본이다. 여기에는 Claude 세션에만 해당하는 것만 적는다.
같은 사실을 두 곳에 적지 않는다.

## 추적

이슈는 셀프호스트 트래커 **MUL** 프로젝트. `GADAK_HOME=$HOME/.gadak gadak --workspace gdk`이고
Mac에서는 `/opt/homebrew/bin/gadak`을 쓴다(PATH의 dev 빌드는 미러 스키마를 못 읽는다).
댓글은 `gadak --workspace gdk comment <KEY> "<본문>"`이다 — `issue comment`가 아니다.
단계마다 이슈 하나(MUL-1 ~ MUL-5), 기계 변경은 MUL-6, 업스트림은 MUL-7.
측정 수치는 [rig-log](https://github.com/midagedev/rig-log)의 `log/`에 먼저 쓰고 이슈에서 링크한다.
`TODO.md`를 열지 않는다.

## 위임

~~구현 라운드는 `outsource` 스킬로 내보낸다.~~ 2026-09-22부터 위임은 opus 서브에이전트(Agent 도구, `model:"opus"` 명시)다.
라운드 운영(열린 카드·파동·트리아지)은 `docs/plan.md`의 「라운드 운영」, 예측·증명 규칙은 `AGENTS.md`의
"Derive first, measure the gap"이 정본이다. 워크트리를 트랙마다 따로 주고, 스펙에는
`AGENTS.md`의 해당 조항을 복사해 넣는다 — 위임받는 모델이 그 파일을 읽는다고 가정하지 않는다.
게이트는 위임 결과를 받은 뒤 리드가 자기 소유로 다시 돌린다. 커밋·푸시는 리드 전용이다.

## 이 레포의 한국어

산문은 한국어로 쓴다. `AGENTS.md`와 코드 주석, 업스트림으로 나가는 문서는 영어다.
한국어 산문은 위임하지 않고 리드가 직접 쓴다.

## rig-log에 쓸 때

측정 기록은 `~/repo/rig-log`의 `log/`에 쓴다. 이 세션에서는 그 레포의 규칙이 자동으로 로드되지 않으므로,
기록을 쓰기 전에 `~/repo/rig-log/CLAUDE.md`를 먼저 읽는다. 놓치기 쉬운 것 셋:

- **공개 레포다.** 커밋 전에 `192.168`, `100.` 대역 주소, `.ts.net`, `admin`을 grep한다. BMC 주소·자격증명,
  박스 `/root`의 nvidia-bug-report 아카이브(호스트명 포함)는 넣지 않는다.
- 측정한 것만 쓰고, 틀리면 지우지 않고 선을 그어 정정한다.
- 산문은 한국어, 업스트림으로 나가는 본문은 영어(AI 도움은 산문으로 밝히고 Claude 배지는 넣지 않는다).
