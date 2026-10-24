#!/usr/bin/env bash
# 아키텍처 축 점검 — 맥에서 돈다(grep뿐, 빌드 없음). docs/arch-split.md 「검사」가 정본이다.
#
# 다섯 가지를 본다:
#   ① 아키텍처 디렉터리끼리 서로를 use 하지 않는다.
#   ② blk.N.<name> 문자열 리터럴과 "<아키텍처>. 키 접두는 crates/*/src/arch/ 와
#      tools/ref/models/ 밖에 없다 — 커널 파일은 모델 이름을 모른다(결정 6).
#   ③ general.architecture 를 읽는 자리는 crates/model/src/arch/mod.rs 하나다.
#      crates/gguf 의 접근자는 저장소 쪽이라 허용한다.
#   ④ 공유 파일(crates/*/src 가운데 arch/ 밖)은 아키텍처 모듈 경로(deepseek2:: · qwen3moe:: …)를 쓰지 않는다.
#      Arch 를 구체 모델로 잇는 디스패치 지점만 파일과 구문(함수·타입 별칭·use 선언) 단위로 허용한다.
#   ⑤ crates/*/src 의 스레드 spawn 은 threads::helper::spawn_helper 를 거친다 — 고정된 스텝 스레드가
#      띄운 스레드는 그 한 CPU 마스크를 물려받는다. 예외는 tools/spawn-allow.txt 의 파일별 개수와 이유다.
#   아키텍처 자신의 크레이트(crates/gpu-<arch>/src/)는 ①②④에서 그 아키텍처의 arch/ 로 센다.
#   크레이트 이름은 디렉터리 이름의 밑줄을 하이픈으로 쓴다(arch/gemma_embedding2/ 의 크레이트는 crates/gpu-gemma-embedding2/).
#   아키텍처 목록은 손으로 적지 않는다: crates/*/src/arch/ 아래 디렉터리 이름이 목록이다. 새 아키텍처는
#   디렉터리가 생기는 순간 넷 모두의 대상이 된다.
#
# 계약: `just check-arch`는 넷 다 엄격하게 돈다. --allow-pending(또는 CHECK_ARCH_PENDING=1)은 ②·③을
# 경고로 낮추는 문이다 — 이관 라운드가 한동안 둘을 빨강으로 두어야 할 때 그 라운드 안에서만 쓴다.
# ①·④는 어느 모드에서도 엄격하다.
# Exit: 0 clean, 1 a violation, 2 no verdict: an unknown argument, no architecture directory, or a listing or scan whose
# grep or find fails (a file it cannot read never passes as a file with no hit). `--self-test` runs the architecture
# listing and each scan on a missing path, rule ②'s scan on fixed lines, and exits.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=tools/grep-answer.sh
source tools/grep-answer.sh

PENDING=${CHECK_ARCH_PENDING:-0}
SELFTEST=0
for a in "$@"; do
  case $a in
    --allow-pending) PENDING=1 ;;
    --self-test) SELFTEST=1 ;;
    *) echo "check-arch: unknown argument '$a' (--allow-pending, --self-test)" >&2; exit 2 ;;
  esac
done

# 아키텍처 이름들: crates/*/src/arch/<name>/ 디렉터리.
# arches <crates dir>: the names, space-separated, into ARCHES. A find that fails ends the check with no verdict.
arches() {
  local dirs
  dirs=$(find "$1" -mindepth 4 -maxdepth 4 -type d -path "$1/*/src/arch/*") || {
    echo "check-arch: cannot list the architecture directories under $1 (find exit $?, above), no verdict" >&2
    exit 2
  }
  ARCHES=$(printf '%s' "$dirs" | sed 's|.*/||' | sort -u | tr '\n' ' ')
}
arches crates
[ -n "$ARCHES" ] || { echo "check-arch: no crates/*/src/arch/<name>/ directory found" >&2; exit 2; }
ALT=$(printf '%s' "$ARCHES" | sed 's/ *$//; s/ /|/g')         # deepseek2|deepseek41|…
# 모듈 없이 다른 모듈이 읽는 아키텍처 문자열(원칙 3의 예외, docs/arch-split.md)은 규칙 ②의 키 접두에 들어간다.
KEYS=$(printf '"%s\\.|' $ARCHES deepseek4 | sed 's/|$//')     # "deepseek2\.|"deepseek41\.|"deepseek4\.
OWN="^crates/([^/]+/src/arch/|gpu-(${ALT//_/-})/src/)"

fail=0
report() { # report <번호> <설명> <위반 줄들>
  local n=$1 what=$2 hits=$3
  [ -n "$hits" ] || return 0
  if [ "$n" != 1 ] && [ "$n" != 4 ] && [ "$n" != 5 ] && [ "$PENDING" = 1 ]; then
    echo "warning: check-arch $n ($what) — $(printf '%s\n' "$hits" | wc -l | tr -d ' ') hit(s), pending the arch move:"
    printf '%s\n' "$hits" | sed 's/^/    /'
  else
    echo "check-arch: $n ($what):" >&2
    printf '%s\n' "$hits" | sed 's/^/    /' >&2
    fail=1
  fi
}

# The scans of rules ① to ⑤ below: each reads only the paths it is given, and a grep or find that fails ends the
# check through nov.
nov() { echo "check-arch: rule $*, no verdict" >&2; exit 2; }
cross_one() { # cross_one <crates dir> <this architecture> <the names it must not use…>: rule ①'s hits into cross
  local dirs root=$1 this=$2 alt
  shift 2
  dirs=$(find "$root" -type d \( -path "*/src/arch/$this" -o -path "$root/gpu-${this//_/-}/src" \)) ||
    nov "1 cannot list the directories of $this under $root (find exit $?, above)"
  [ -n "$dirs" ] || return 0
  alt=$(printf '%s|' "$@")
  # shellcheck disable=SC2086
  cross_uses "(${alt%|})" $dirs
}
cross_uses() { # cross_uses <name> <dir…>
  local name=$1 hits
  shift
  hits=$(grep_answer check-arch -rnE "^[[:space:]]*(pub )?use .*\b$name\b" "$@" --include='*.rs') ||
    nov "1 cannot read $* (exit $?, above)"
  [ -z "$hits" ] || cross="${cross:+$cross$'\n'}$hits"
}
scan_lits() { # scan_lits <path…>: rule ②'s hits into lits
  # PIN(2026-10-22): the metal bins are exempt beside the gpu-gates bins: they are gates and probes, and naming the
  # fixture's tensors is their job.
  lits=$(grep_answer check-arch -rnE "\"blk\\.|(^|[^.a-z])blk\\.\\{|$KEYS" "$@" --include='*.rs' --include='*.sh' \
    | grep_answer check-arch -vE "$OWN" \
    | grep_answer check-arch -vE '^tools/ref/models/' \
    | grep_answer check-arch -vE '^crates/[^/]+/tests/' \
    | grep_answer check-arch -vE '^crates/gpu-gates/src/bin/' \
    | grep_answer check-arch -vE '^crates/metal/src/bin/' \
    | grep_answer check-arch -vE '^crates/engram/' \
    | grep_answer check-arch -vE '^crates/gguf/src/bin/' \
    | grep_answer check-arch -vE '^[^:]+:[0-9]+:[[:space:]]*//' \
    | KEYS=$KEYS awk '{ body = $0; sub(/^[^:]*:[0-9]+:/, "", body); gsub(/"blk\."/, "", body)
             if (body ~ /"blk\.|(^|[^.a-z])blk\.[{]/ || body ~ ENVIRON["KEYS"]) print }') ||
    nov "2 cannot read $* (exit $?, above)"
}
scan_archread() { # scan_archread <path…>: rule ③'s hits into archread
  archread=$(grep_answer check-arch -rn 'general\.architecture' "$@" --include='*.rs' \
    | grep_answer check-arch -vE '^crates/gguf/' \
    | grep_answer check-arch -vE '^crates/model/src/arch/mod\.rs:' \
    | grep_answer check-arch -vE '^[^:]+:[0-9]+:[[:space:]]*//') ||
    nov "3 cannot read $* (exit $?, above)"
}
# 세 모양을 본다: 경로 안의 `deepseek2::…`, 모듈을 통째로 들여오거나 재수출하는 use 줄
# (`use …::arch::deepseek2 as x;`, `pub use …::deepseek2;` — 뒤에 `::`가 없어 첫 모양에 안 걸린다),
# 그리고 rustfmt가 여러 줄로 나눈 `use …::{`의 한 줄에 모듈 이름만 남은 것(`    deepseek2,`).
scan_archpath() { # scan_archpath <path…>: rule ④'s hits into archpath
  archpath=$(grep_answer check-arch -rnE -e "\\b($ALT)::" \
               -e "^[[:space:]]*(pub(\\([a-z]+\\))?[[:space:]]+)?use[[:space:]][^;]*\\b($ALT)\\b" \
               -e "^[[:space:]]*($ALT)([[:space:]]+as[[:space:]]+[A-Za-z_][A-Za-z0-9_]*)?,?[[:space:]]*\$" \
               "$@" --include='*.rs' | sort -u \
    | grep_answer check-arch -vE "$OWN" \
    | grep_answer check-arch -vE '^crates/gpu-gates/src/bin/' \
    | grep_answer check-arch -vE '^crates/metal/src/bin/' \
    | grep_answer check-arch -vE '^[^:]+:[0-9]+:[[:space:]]*//') ||
    nov "4 cannot read $* (exit $?, above)"
}
scan_spawns() { # scan_spawns <path…>: rule ⑤'s spawn count per file into spawned
  spawned=$(grep_answer check-arch -rnE 'thread::spawn\(|\.spawn\((move\b|\|)' "$@" --include='*.rs' \
    | grep_answer check-arch -vE '^crates/oxide-ice-unroll/' \
    | grep_answer check-arch -vE '^[^:]+:[0-9]+:[[:space:]]*//' | cut -d: -f1 | sort | uniq -c) ||
    nov "5 cannot read $* (exit $?, above)"
}

# The listing and each scan on a missing path end with no verdict, and each scan on /dev/null (no hit) is clean.
# Rule ②'s awk reads a key's dot as a dot: the bare "blk." beside a name that only starts like a key is clean, and a
# key with its dot beside it is a hit.
self_test() {
  local fails=0 out rc n scan d
  expect() {
    [ "$2" = "$3" ] || { echo "check-arch: self-test $1: got [$2], want [$3]" >&2; fails=$((fails + 1)); }
  }
  rc=0 out=$( (arches /nonexistent) 2>&1) || rc=$?
  expect arches "$rc ${out##*$'\n'}" \
    "2 check-arch: cannot list the architecture directories under /nonexistent (find exit 1, above), no verdict"
  rc=0 out=$( (cross_one /nonexistent a b) 2>&1) || rc=$?
  expect cross-find "$rc ${out##*$'\n'}" \
    "2 check-arch: rule 1 cannot list the directories of a under /nonexistent (find exit 1, above), no verdict"
  rc=0 out=$( (cross_uses b /nonexistent/a.rs) 2>&1) || rc=$?
  expect cross-grep "$rc ${out##*$'\n'}" "2 check-arch: rule 1 cannot read /nonexistent/a.rs (exit 2, above), no verdict"
  rc=0 out=$( (cross_one /dev/null a b && cross_uses b /dev/null && printf '%s' "${cross-}") 2>&1) || rc=$?
  expect cross-clean "$rc $out" "0 "
  n=1
  for scan in lits archread archpath spawns; do
    n=$((n + 1))
    rc=0 out=$( ("scan_$scan" /nonexistent/a.rs) 2>&1) || rc=$?
    expect "$scan" "$rc ${out##*$'\n'}" "2 check-arch: rule $n cannot read /nonexistent/a.rs (exit 2, above), no verdict"
    rc=0 out=$( ("scan_$scan" /dev/null && printf '%s' "${lits-}${archread-}${archpath-}${spawned-}") 2>&1) || rc=$?
    expect "$scan-clean" "$rc $out" "0 "
  done
  d=$(mktemp -d)
  printf '%s\n' 'let k = ("blk.", "deepseek2x");' > "$d/a.rs"
  rc=0 out=$( (scan_lits "$d" && printf '%s' "$lits") 2>&1) || rc=$?
  expect lits-dot "$rc $out" "0 "
  printf '%s\n' 'let k = ("blk.", "deepseek2.expert_count");' > "$d/a.rs"
  rc=0 out=$( (scan_lits "$d" && printf '%s' "$lits") 2>&1) || rc=$?
  expect lits-key "$rc $out" "0 $d/a.rs:1:let k = (\"blk.\", \"deepseek2.expert_count\");"
  rm -rf "$d"
  [ "$fails" = 0 ] || { echo "check-arch: self-test: $fails failed" >&2; return 1; }
  echo "check-arch: self-test ok"
}
if [ "$SELFTEST" = 1 ]; then
  self_test
  exit $?
fi

# ① 두 아키텍처 디렉터리가 서로를 use 하지 않는다. 한쪽 디렉터리가 없으면 그쪽은 grep 대상이 없어
#    통과하고, 생기는 순간부터 값을 한다.
#    한 아키텍처마다 디렉터리를 한 번 찾고, 다른 이름 전부를 한 grep으로 본다.
cross=
for a in $ARCHES; do
  others=()
  for b in $ARCHES; do
    [ "$a" = "$b" ] || others+=("$b")
  done
  [ "${#others[@]}" = 0 ] || cross_one crates "$a" "${others[@]}"
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
scan_lits crates tools/ref
report 2 "model-aware string literals outside arch/" "$lits"

# ③ general.architecture 는 한 곳에서만 읽는다.
scan_archread crates
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
  # The same shape for nemotron_h — the nemotron seat and its gates name the body through this alias only.
  'crates/gpu/src/lib.rs type NemotronHModel'
  # The same shape for nemotron_h_moe — its gate names the body through this alias only.
  'crates/gpu/src/lib.rs type NemotronHMoeModel'
  # gpu-vision 은 deepseek41v 탑 자신의 크레이트다(이름이 gpu-<arch> 가 아닐 뿐) — 인코더가 그 탑의 이름표를 읽는다.
  'crates/gpu-vision/src/encoder.rs use'
  # The same crate's second tower (the Qwen3-VL merger of Clef, Qwen3.6 and Qwen3.8) reads its file's
  # names and hyperparameters, and `open` is the one place that maps a file's projector type to a tower.
  'crates/gpu-vision/src/qwen3vl.rs use'
  # The MiniCPM-V tower (MiniCPM-V 4.7's mmproj) reads its file's names, hyperparameters, card figure and image-side types the same way.
  'crates/gpu-vision/src/minicpmv.rs use'
  # The glm5v tower (GLM-5.3-Flash's mmproj) reads its file's names, hyperparameters, card figure and host rules the same way.
  'crates/gpu-vision/src/glm5v.rs use'
  # The RADIO tower (Nemotron 3.5 Bolt's mmproj) reads its file's names, hyperparameters and card figure the same way.
  'crates/gpu-vision/src/radio.rs use'
  'crates/gpu-vision/src/lib.rs use'
  'crates/gpu-vision/src/lib.rs fn open'
  # The V4.1 media span's one owner (ds41_media, the media gate and the serve seat share it):
  # Hparams::read refuses a non-deepseek41v file by name, and the delimiter names and the card
  # figure it reads are the tower's own.
  'crates/gpu-gates/src/ds41_media.rs use'
  # The Qwen3.6 fixture's shape (q36_fixture, the e2e gate's census) names its layers by the header's own kind
  # (qwen35moe::hparams::Kind), the one owner of that fact, not a copy.
  'crates/gpu-gates/src/q36_fixture.rs use'
  # The vision seat's one table that maps a projector type (`clip.projector_type`) to its host crate's module: the ids
  # its span carries, the text rope it needs, the row width and the image-token cap are that module's own facts.
  'crates/gpu-gates/src/vision_seat.rs use'
  'crates/gpu-gates/src/vision_seat.rs fn rules'
  # The shared prepared image holds one MiniCPM tile group (`EncoderInput::Tiles`). Prepare stays in the model's
  # own module; this file only names that group.
  'crates/vision/src/media.rs use'
  # CPU 디코드 바이너리: Arch::detect 로 다른 아키텍처를 거절한 뒤 deepseek2 순전파를 돈다.
  'crates/model/src/bin/bloomery-decode.rs use'
  # r8 사이드카 변환기: Hparams::read 가 deepseek41 이 아닌 파일을 이름 붙여 거절한 뒤 V4.1 의 routed 술어를 쓴다.
  'crates/model/src/bin/r8conv.rs use'
  # The ModelOpt converter: Checkpoint::parse refuses another architecture by name before mapping qwen4exp's tensors.
  'crates/model-hf/src/bin/prepare.rs use'
  # 호스트 union 벤치: 파일의 아키텍처가 qwen4exp 일 때만 qwen35moe 의 Hparams·텐서 이름·host::layers 로 routed 층을 연다.
  'crates/model/src/bin/bench_v41_host.rs use'
  # 오라클 표의 디스패치: for_arch 가 Arch 를 그 아키텍처의 표로 잇는다.
  'crates/gpu-gates/src/oracle/mod.rs fn for_arch'
  # AnyEngine::open 이 파일의 아키텍처를 그 몸체로 잇는다 — 이 파일의 use 묶음이 그 디스패치의
  # 수입면이다: qwen3moe 팔이 그 family 의 불러오기 선택(KvQ8 포함)을 root 별칭과 arch 경로로 받는다.
  'crates/gpu-gates/src/engine.rs use'
  # 디스패치가 아닌 유일한 항목: 하네스의 기본 참조 세트. ref_dir 가 Arch 를 받기 전까지 남는다.
  'crates/gpu-gates/src/lib.rs fn ref_dir'
  # The --hf resolve's draft name: it reads only the file name of qwen4exp's DRAFT constant.
  'crates/gpu-gates/src/model_file.rs fn draft_name'
  # The --hf resolve's draft predicate: the plan's own rule (qwen35moe's PlanInputs::mtp_borrows, which
  # the qwen4exp seat plans with), called read-only so the fetch and the plan keep one owner.
  'crates/gpu-gates/src/model_file.rs fn draft_usable'
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
scan_archpath crates/*/src
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

# ⑤ A thread inherits its spawner's affinity mask: one spawned beside a step thread pinned to one cpu
#    shares that cpu. Every spawn in crates/*/src (`thread::spawn(`, a Builder's or a scope's
#    `.spawn(move`/`.spawn(|`) outside a comment is counted per file against tools/spawn-allow.txt
#    (`<file> <count> <why>`); a file not listed, a file with more spawns than its entry, and an entry
#    above its file's count (the list follows the tree down) each fail.
ALLOW=tools/spawn-allow.txt
scan_spawns crates/*/src
spawn_bad=
while read -r n f; do
  [ -n "${f:-}" ] || continue
  want=$(awk -v f="$f" '$1 == f { print $2 }' "$ALLOW")
  if [ -z "$want" ]; then
    spawn_bad="${spawn_bad:+$spawn_bad$'\n'}$f: $n spawn(s), not in $ALLOW (use threads::helper::spawn_helper)"
  elif [ "$n" -gt "$want" ]; then
    spawn_bad="${spawn_bad:+$spawn_bad$'\n'}$f: $n spawn(s), $ALLOW allows $want"
  fi
done <<< "$spawned"
while read -r f want _; do
  case "$f" in ''|'#'*) continue ;; esac
  have=$(awk -v f="$f" '$2 == f { print $1 }' <<< "$spawned")
  if [ "${have:-0}" -lt "$want" ]; then
    spawn_bad="${spawn_bad:+$spawn_bad$'\n'}$ALLOW: $f allows $want spawn(s), the file holds ${have:-0} (lower the entry)"
  fi
done < "$ALLOW"
report 5 "thread spawn outside threads::helper::spawn_helper" "$spawn_bad"

if [ "$fail" = 1 ]; then
  echo "check-arch: failed" >&2
  exit 1
fi
if [ "$PENDING" = 1 ]; then
  echo "check-arch: ok (pending checks downgraded to warnings)"
else
  echo "check-arch: ok"
fi
