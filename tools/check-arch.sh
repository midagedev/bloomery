#!/usr/bin/env bash
# 아키텍처 축 점검 — 맥에서 돈다(grep뿐, 빌드 없음). docs/arch-split.md 「검사」가 정본이다.
#
# 세 가지를 본다:
#   ① arch/deepseek41/ 아래가 deepseek2 를 use 하지 않고, 그 반대도 없다.
#   ② blk.N.<name> 문자열 리터럴과 "deepseek2. / "deepseek41. 키 접두는 crates/*/src/arch/ 와
#      tools/ref/models/ 밖에 없다 — 커널 파일은 모델 이름을 모른다(결정 6).
#   ③ general.architecture 를 읽는 자리는 crates/model/src/arch/mod.rs 하나다.
#      crates/gguf 의 접근자는 저장소 쪽이라 허용한다.
#
# 계약: --allow-pending(또는 CHECK_ARCH_PENDING=1)은 ②·③을 경고로 낮춘다. M1·M2·M3 이관이
# 끝나기 전에는 둘이 실제로 빨강이라 레시피를 지금 붙이려면 이 문이 필요하고, M1 이 머지되면
# 리드가 레시피에서 이 플래그를 뺀다. ①은 어느 모드에서도 엄격하다 — 지금은 두 디렉터리가
# 아직 없어서 통과하고, 생기는 순간부터 값을 한다.
set -euo pipefail
cd "$(dirname "$0")/.."

PENDING=${CHECK_ARCH_PENDING:-0}
for a in "$@"; do
  case $a in
    --allow-pending) PENDING=1 ;;
    *) echo "check-arch: unknown argument '$a' (only --allow-pending)" >&2; exit 2 ;;
  esac
done

fail=0
report() { # report <번호> <설명> <위반 줄들>
  local n=$1 what=$2 hits=$3
  [ -n "$hits" ] || return 0
  if [ "$n" != 1 ] && [ "$PENDING" = 1 ]; then
    echo "warning: check-arch $n ($what) — $(printf '%s\n' "$hits" | wc -l | tr -d ' ') hit(s), pending the arch move:"
    printf '%s\n' "$hits" | sed 's/^/    /'
  else
    echo "check-arch: $n ($what):" >&2
    printf '%s\n' "$hits" | sed 's/^/    /' >&2
    fail=1
  fi
}

# ① 두 아키텍처 디렉터리가 서로를 use 하지 않는다. 디렉터리가 아직 없으면 grep 대상이 없어
#    통과한다 — arch/deepseek41/ 은 B4 의 첫 op 가 만든다.
cross=
cross_one() { # cross_one <이 아키텍처 디렉터리> <여기서 use 하면 안 되는 이름>
  local dirs hits
  dirs=$(find crates -type d -path "*/src/arch/$1" 2>/dev/null || true)
  [ -n "$dirs" ] || return 0
  # shellcheck disable=SC2086
  hits=$(grep -rnE "^[[:space:]]*(pub )?use .*\b$2\b" $dirs --include='*.rs' || true)
  [ -z "$hits" ] || cross="${cross:+$cross$'\n'}$hits"
}
cross_one deepseek41 deepseek2
cross_one deepseek2 deepseek41
report 1 "arch dirs use each other" "$cross"

# ② 모델을 아는 문자열은 arch/ 와 도구 프로필 안에만.
# 훑는 범위는 crates 의 러스트와 tools/ref 의 셸이다. 빼는 것: 시험(오라클 탭 이름을 고정한다),
# gpu-gates 의 bin(오라클 탭 — M3 가 표로 옮긴다), engram(SITE_NAMES 는 추적 중인 잔여), gguf 의
# bin(인벤토리 도구는 이름을 찍는 저장소 쪽이다 — ③이 gguf 접근자를 허용하는 것과 같은 논거),
# 그리고 주석 줄(문서의 예시 이름은 코드가 아니다). 이 파일 자신은 tools/ 바로 아래라 안 걸린다 —
# 점검기의 설명문이 자기 점검에 걸리면 어떤 트리에서도 빨강이다.
lits=$(grep -rnE '"blk\.|blk\.\{|"deepseek2\.|"deepseek41\.' crates tools/ref --include='*.rs' --include='*.sh' 2>/dev/null \
  | grep -vE '^crates/[^/]+/src/arch/' \
  | grep -vE '^tools/ref/models/' \
  | grep -vE '^crates/[^/]+/tests/' \
  | grep -vE '^crates/gpu-gates/src/bin/' \
  | grep -vE '^crates/engram/' \
  | grep -vE '^crates/gguf/src/bin/' \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true)
report 2 "model-aware string literals outside arch/" "$lits"

# ③ general.architecture 는 한 곳에서만 읽는다.
archread=$(grep -rn 'general\.architecture' crates --include='*.rs' 2>/dev/null \
  | grep -vE '^crates/gguf/' \
  | grep -vE '^crates/model/src/arch/mod\.rs:' \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true)
report 3 "general.architecture read outside crates/model/src/arch/mod.rs" "$archread"

if [ "$fail" = 1 ]; then
  echo "check-arch: failed" >&2
  exit 1
fi
if [ "$PENDING" = 1 ]; then
  echo "check-arch: ok (pending checks downgraded to warnings)"
else
  echo "check-arch: ok"
fi
