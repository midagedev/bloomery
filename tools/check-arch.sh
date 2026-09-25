#!/usr/bin/env bash
# 아키텍처 축 점검 — 맥에서 돈다(grep뿐, 빌드 없음). docs/arch-split.md 「검사」가 정본이다.
#
# 네 가지를 본다:
#   ① 아키텍처 디렉터리끼리 서로를 use 하지 않는다.
#   ② blk.N.<name> 문자열 리터럴과 "<아키텍처>. 키 접두는 crates/*/src/arch/ 와
#      tools/ref/models/ 밖에 없다 — 커널 파일은 모델 이름을 모른다(결정 6).
#   ③ general.architecture 를 읽는 자리는 crates/model/src/arch/mod.rs 하나다.
#      crates/gguf 의 접근자는 저장소 쪽이라 허용한다.
#   ④ 공유 파일(crates/*/src 가운데 arch/ 밖)은 아키텍처 모듈 경로(deepseek2:: · qwen3moe:: …)를 쓰지 않는다.
#      Arch 를 구체 모델로 잇는 디스패치 지점만 파일과 구문(함수·타입 별칭·use 선언) 단위로 허용한다.
#   아키텍처 자신의 크레이트(crates/gpu-<arch>/src/)는 ①②④에서 그 아키텍처의 arch/ 로 센다.
#   아키텍처 목록은 손으로 적지 않는다: crates/*/src/arch/ 아래 디렉터리 이름이 목록이다. 새 아키텍처는
#   디렉터리가 생기는 순간 넷 모두의 대상이 된다.
#
# 계약: `just check-arch`는 넷 다 엄격하게 돈다. --allow-pending(또는 CHECK_ARCH_PENDING=1)은 ②·③을
# 경고로 낮추는 문이다 — 이관 라운드가 한동안 둘을 빨강으로 두어야 할 때 그 라운드 안에서만 쓴다.
# ①·④는 어느 모드에서도 엄격하다.
set -euo pipefail
cd "$(dirname "$0")/.."

PENDING=${CHECK_ARCH_PENDING:-0}
for a in "$@"; do
  case $a in
    --allow-pending) PENDING=1 ;;
    *) echo "check-arch: unknown argument '$a' (only --allow-pending)" >&2; exit 2 ;;
  esac
done

# 아키텍처 이름들: crates/*/src/arch/<name>/ 디렉터리.
ARCHES=$(find crates -mindepth 4 -maxdepth 4 -type d -path 'crates/*/src/arch/*' 2>/dev/null \
  | sed 's|.*/||' | sort -u | tr '\n' ' ')
[ -n "$ARCHES" ] || { echo "check-arch: no crates/*/src/arch/<name>/ directory found" >&2; exit 2; }
ALT=$(printf '%s' "$ARCHES" | sed 's/ *$//; s/ /|/g')         # deepseek2|deepseek41|…
# 모듈 없이 다른 모듈이 읽는 아키텍처 문자열(원칙 3의 예외, docs/arch-split.md)은 규칙 ②의 키 접두에 들어간다.
KEYS=$(printf '"%s\\.|' $ARCHES deepseek4 | sed 's/|$//')     # "deepseek2\.|"deepseek41\.|"deepseek4\.
OWN="^crates/([^/]+/src/arch/|gpu-($ALT)/src/)"

fail=0
report() { # report <번호> <설명> <위반 줄들>
  local n=$1 what=$2 hits=$3
  [ -n "$hits" ] || return 0
  if [ "$n" != 1 ] && [ "$n" != 4 ] && [ "$PENDING" = 1 ]; then
    echo "warning: check-arch $n ($what) — $(printf '%s\n' "$hits" | wc -l | tr -d ' ') hit(s), pending the arch move:"
    printf '%s\n' "$hits" | sed 's/^/    /'
  else
    echo "check-arch: $n ($what):" >&2
    printf '%s\n' "$hits" | sed 's/^/    /' >&2
    fail=1
  fi
}

# ① 두 아키텍처 디렉터리가 서로를 use 하지 않는다. 한쪽 디렉터리가 없으면 그쪽은 grep 대상이 없어
#    통과하고, 생기는 순간부터 값을 한다.
cross=
cross_one() { # cross_one <이 아키텍처 디렉터리> <여기서 use 하면 안 되는 이름>
  local dirs hits
  dirs=$(find crates -type d \( -path "*/src/arch/$1" -o -path "crates/gpu-$1/src" \) 2>/dev/null || true)
  [ -n "$dirs" ] || return 0
  # shellcheck disable=SC2086
  hits=$(grep -rnE "^[[:space:]]*(pub )?use .*\b$2\b" $dirs --include='*.rs' || true)
  [ -z "$hits" ] || cross="${cross:+$cross$'\n'}$hits"
}
for a in $ARCHES; do
  for b in $ARCHES; do
    [ "$a" = "$b" ] || cross_one "$a" "$b"
  done
done
report 1 "arch dirs use each other" "$cross"

# ② 모델을 아는 문자열은 arch/ 와 도구 프로필 안에만.
# 훑는 범위는 crates 의 러스트와 tools/ref 의 셸이다. 빼는 것: 시험(오라클 탭 이름을 고정한다),
# gpu-gates 의 bin(오라클 탭 — M3 가 표로 옮긴다), engram(SITE_NAMES 는 추적 중인 잔여), gguf 의
# bin(인벤토리 도구는 이름을 찍는 저장소 쪽이다 — ③이 gguf 접근자를 허용하는 것과 같은 논거),
# 그리고 주석 줄(문서의 예시 이름은 코드가 아니다). 이 파일 자신은 tools/ 바로 아래라 안 걸린다 —
# 점검기의 설명문이 자기 점검에 걸리면 어떤 트리에서도 빨강이다.
# 맨 접두 리터럴 "blk." 하나는 이름이 아니다(2026-09-23): 모든 아키텍처가 텐서를 `blk.<층>.` 아래에
# 두는 GGUF 규약이고, 공유 로더가 층 번호를 파싱하는 자리(gpu/weights.rs 의 block_index)가 그것을 쓴다.
# 규칙이 막는 것은 `blk.N.<name>`이므로, 줄에서 그 리터럴을 지운 뒤에도 패턴이 남는 줄만 잡는다 —
# `"blk.{l}.ffn_up"`·`"blk.0.attn_q"`와, 맨 접두와 이름이 한 줄에 같이 있는 줄은 그대로 걸린다.
# `blk.{` 앞에 `.`나 소문자가 붙은 이름(비전 탑의 `v.blk.{n}.…`)은 다른 탑의 이름공간이라 이 규칙의 대상이
# 아니다 — 그 표는 어차피 그 크레이트의 arch/ 아래에 있다.
lits=$(grep -rnE "\"blk\\.|(^|[^.a-z])blk\\.\\{|$KEYS" crates tools/ref --include='*.rs' --include='*.sh' 2>/dev/null \
  | grep -vE "$OWN" \
  | grep -vE '^tools/ref/models/' \
  | grep -vE '^crates/[^/]+/tests/' \
  | grep -vE '^crates/gpu-gates/src/bin/' \
  | grep -vE '^crates/engram/' \
  | grep -vE '^crates/gguf/src/bin/' \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' \
  | awk -v keys="$KEYS" '{ body = $0; sub(/^[^:]*:[0-9]+:/, "", body); gsub(/"blk\."/, "", body)
           if (body ~ /"blk\.|(^|[^.a-z])blk\.[{]/ || body ~ keys) print }' || true)
report 2 "model-aware string literals outside arch/" "$lits"

# ③ general.architecture 는 한 곳에서만 읽는다.
archread=$(grep -rn 'general\.architecture' crates --include='*.rs' 2>/dev/null \
  | grep -vE '^crates/gguf/' \
  | grep -vE '^crates/model/src/arch/mod\.rs:' \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true)
report 3 "general.architecture read outside crates/model/src/arch/mod.rs" "$archread"

# ④ 공유 파일은 아키텍처 모듈 경로를 쓰지 않는다. 옛 크레이트 루트 경로(model::attn 등)는 지워져
#    컴파일이 막으므로, 여기서는 컴파일러가 못 잡는 것 — 공유 파일이 `…::deepseek2::…`를 직접 적는 줄 — 만 본다.
# 빼는 것은 ②와 같은 모양이다: arch/ 아래, gpu-gates 의 bin(deepseek2 게이트다), 주석 줄. 시험(crates/*/tests/)은
# 훑는 범위(crates/*/src) 밖이다. 디스패치 지점은 파일과 구문 단위로 하나씩, 이유 한 줄과 함께 허용한다.
#
# An entry names a file and one construct in it, never a line's text: `fn <name>` (a top-level
# function, from its signature to the `}` that closes it in column 0), `type <name>` (a type alias,
# from its first line to its `;`) or `use` (each `use` declaration of the file, to its `;`). Every
# hit inside the construct passes, whatever it spells — a field whose type changes, an argument
# added — and a path anywhere else in the same file still fails. A construct that is renamed or
# moved allows nothing, so its hits fail until the entry follows it.
dispatch=(
  # GpuModel<B>에 deepseek2 몸체를 꽂는 별칭 — AnyEngine(crates/gpu-gates/src/engine.rs)은 이 별칭으로만 deepseek2 를 부른다.
  'crates/gpu/src/lib.rs type Deepseek2Model'
  # 같은 모양의 qwen3moe 별칭 — AnyEngine 은 이 별칭으로만 qwen3moe 를 부른다.
  'crates/gpu/src/lib.rs type Qwen3moeModel'
  # gpu-vision 은 deepseek41v 탑 자신의 크레이트다(이름이 gpu-<arch> 가 아닐 뿐) — 인코더가 그 탑의 이름표를 읽는다.
  'crates/gpu-vision/src/encoder.rs use'
  # CPU 디코드 바이너리: Arch::detect 로 다른 아키텍처를 거절한 뒤 deepseek2 순전파를 돈다.
  'crates/model/src/bin/bloomery-decode.rs use'
  # r8 사이드카 변환기: Hparams::read 가 deepseek41 이 아닌 파일을 이름 붙여 거절한 뒤 V4.1 의 routed 술어를 쓴다.
  'crates/model/src/bin/r8conv.rs use'
  # 오라클 표의 디스패치: for_arch 가 Arch 를 그 아키텍처의 표로 잇는다.
  'crates/gpu-gates/src/oracle/mod.rs fn for_arch'
  # 디스패치가 아닌 유일한 항목: 하네스의 기본 참조 세트. ref_dir 가 Arch 를 받기 전까지 남는다.
  'crates/gpu-gates/src/lib.rs fn ref_dir'
)
# construct_lines <file> <fn|type|use> [name]: "first last" for each line span of that construct.
construct_lines() {
  awk -v kind="$2" -v name="${3:-}" '
    !first {
      if (kind == "fn" && $0 ~ ("^(pub(\\([a-z]+\\))? )?fn " name "[(<]")) first = NR
      else if (kind == "type" && $0 ~ ("^(pub(\\([a-z]+\\))? )?type " name "[ <=]")) first = NR
      else if (kind == "use" && $0 ~ /^[ \t]*(pub(\([a-z]+\))?[ \t]+)?use[ \t]/) first = NR
    }
    first && ((kind == "fn" && /^}/) || (kind != "fn" && /;[ \t]*$/)) {
      print first, NR
      first = 0
      if (kind != "use") exit
    }
  ' "$1"
}
allowed=$(for d in "${dispatch[@]}"; do
  read -r file kind name <<< "$d"
  [ -f "$file" ] || continue
  construct_lines "$file" "$kind" "$name" | sed "s|^|$file |"
done)
# 세 모양을 본다: 경로 안의 `deepseek2::…`, 모듈을 통째로 들여오거나 재수출하는 use 줄
# (`use …::arch::deepseek2 as x;`, `pub use …::deepseek2;` — 뒤에 `::`가 없어 첫 모양에 안 걸린다),
# 그리고 rustfmt가 여러 줄로 나눈 `use …::{`의 한 줄에 모듈 이름만 남은 것(`    deepseek2,`).
archpath=$({ grep -rnE "\\b($ALT)::" crates/*/src --include='*.rs' 2>/dev/null
             grep -rnE "^[[:space:]]*(pub(\\([a-z]+\\))?[[:space:]]+)?use[[:space:]][^;]*\\b($ALT)\\b" \
               crates/*/src --include='*.rs' 2>/dev/null
             grep -rnE "^[[:space:]]*($ALT)([[:space:]]+as[[:space:]]+[A-Za-z_][A-Za-z0-9_]*)?,?[[:space:]]*\$" \
               crates/*/src --include='*.rs' 2>/dev/null; } | sort -u \
  | grep -vE "$OWN" \
  | grep -vE '^crates/gpu-gates/src/bin/' \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' || true)
outside=
while IFS= read -r hit; do
  [ -n "$hit" ] || continue
  file=${hit%%:*} rest=${hit#*:}
  line=${rest%%:*} inside=0
  while read -r f lo hi; do
    if [ "$f" = "$file" ] && [ "$line" -ge "$lo" ] && [ "$line" -le "$hi" ]; then inside=1; break; fi
  done <<< "$allowed"
  [ "$inside" = 1 ] || outside="${outside:+$outside$'\n'}$hit"
done <<< "$archpath"
report 4 "architecture module path in a shared file, outside arch/ and the dispatch points" "$outside"

if [ "$fail" = 1 ]; then
  echo "check-arch: failed" >&2
  exit 1
fi
if [ "$PENDING" = 1 ]; then
  echo "check-arch: ok (pending checks downgraded to warnings)"
else
  echo "check-arch: ok"
fi
