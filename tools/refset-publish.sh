#!/usr/bin/env bash
# The fixture tier's `fx_` reference sets as one Hugging Face dataset repo (midagedev/bloomery-refsets). Runs on the box
# through tools/box.sh (`just refset-publish-dry`); the set directories are read where they are and never written.
#   tools/refset-publish.sh [--dry-run] [--data DIR] [--out DIR] [--jobs N]
#       The default and the only mode that touches nothing outside --out. Lists, per family directory, every file: its repo
#       path, its source path, its bytes and its sha256, with each directory's and the grand total, and the sets left out
#       with their reasons. Writes SHA256SUMS, MANIFEST.tsv, README.md and LISTING.txt into --out (default
#       /tmp/refset-publish; never under /models or --data).
#   tools/refset-publish.sh --upload
#       Refused (64) unless REFSET_PUBLISH_GO=1. Stages the files as hardlinks (REFSET_PUBLISH_STAGE, default beside
#       --data, which must be the same filesystem) plus README.md, SHA256SUMS and LICENCES/, and runs `hf upload --repo-type dataset`
#       on it with HF_TOKEN_PATH pointing at the token file; this script never reads the token.
#   tools/refset-publish.sh --self-test    # the assignment, the named errors, the left-out block and the --upload refusal
#                                          # on a fake tree; check-recipes runs it
#
# The assignment. A set directory is a candidate when its name starts `fx_` under --data or --data/ref-mtp (refset's
# naming of a fixture set, crates/refset/src/arch/*/fixture.rs). Its family is read from the set's own MANIFEST.tsv:
# `# model` must be `<fixture root>/<dir>/<file>`, where the fixture root and the `(arch, dir)` rows are those of
# crates/refset/src/fixture.rs (DEFAULT_ROOT, DIRS), `# arch` must be the arch of that dir, and `# fixture` must be
# there; every set of a family must carry the same `# fixture` line. A name is never read for a family.
# A set that cannot be placed is a named error (65) and nothing is hashed. Left out, by name with the reason: a fixture
# variant (its `# model` is under fixture-variants/<tag>, a tag of tools/fixture-variants.tsv) and any set of the
# DSpark draft (`dspark` or a `/draft/` directory in `# model` or `# draft_model`).
# Exit codes: 64 usage or refusal, 65 a set the rule cannot place or data that cannot be published, 66 an input that is not
# there.
set -euo pipefail

TAB=$(printf '\t')
NL=$'\n'
ROOT=${REFSET_PUBLISH_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
REPO=midagedev/bloomery-refsets
BOX_USER_HOME=/home/user
HF=${REFSET_PUBLISH_HF:-$BOX_USER_HOME/ft/bin/hf}
HF_TOKEN_FILE=$BOX_USER_HOME/.cache/huggingface/token
DATA=${BLOOMERY_DATA:-/root/bloomery-data}
OUT=/tmp/refset-publish
JOBS=

die() {
  local rc=$1
  shift
  echo "refset-publish: $*" >&2
  exit "$rc"
}

# The sha256 command and the stat format of this host: the box has GNU, the Mac (the self-test) BSD.
if command -v sha256sum > /dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
if stat -c %s / > /dev/null 2>&1; then STATFMT=gnu; else STATFMT=bsd; fi

# value of `# KEY<TAB>value` in a set's MANIFEST.tsv, empty when the line is not there
mline() {
  awk -F'\t' -v k="# $2" '$1 == k { sub(/^[^\t]*\t/, ""); print; exit }' "$1/MANIFEST.tsv"
}

# The one place a set the rule cannot place is recorded; the run ends 65 after every set was read.
unplaced() { printf '%s\t%s\n' "$1" "$2" >> "$W/errors.tsv"; }
leftout() { printf '%s\t%s\n' "$1" "$2" >> "$W/leftout.tsv"; }

# The fixture root, the (arch, dir) rows and the variant tags, read from where the code owns them.
read_rules() {
  local rs=$ROOT/crates/refset/src/fixture.rs tags=$ROOT/tools/fixture-variants.tsv
  [ -f "$rs" ] || die 66 "no $rs: the fixture root and the arch/dir rows are read from it"
  [ -f "$tags" ] || die 66 "no $tags: the variant tags are read from it"
  FIXTURE_ROOT=$(sed -n 's/^pub const DEFAULT_ROOT: &str = "\(.*\)";$/\1/p' "$rs")
  [ -n "$FIXTURE_ROOT" ] || die 66 "$rs has no DEFAULT_ROOT line of the form it was read by"
  awk '/^const DIRS:/ { on = 1; next } on && /^\];/ { exit }
       on && match($0, /\("[a-z0-9]+", "[a-z0-9]+"\)/) {
         s = substr($0, RSTART, RLENGTH); gsub(/[()" ]/, "", s); split(s, p, ","); print p[1] "\t" p[2] }' "$rs" > "$W/dirs.tsv"
  [ -s "$W/dirs.tsv" ] || die 66 "$rs has no DIRS rows of the form they were read by"
  grep -v '^#' "$tags" | cut -f1 | sed '/^$/d' > "$W/tags.txt"
}

# Place one set (a path under $DATA): published with its family, left out with a reason, or unplaced.
place() {
  local set=$1 dir=$DATA/$1 model arch fixture draft low rest fdir file family tag
  if [ ! -f "$dir/MANIFEST.tsv" ]; then
    unplaced "$set" "no MANIFEST.tsv: nothing states what the set was dumped from"
    return 0
  fi
  model=$(mline "$dir" model)
  arch=$(mline "$dir" arch)
  fixture=$(mline "$dir" fixture)
  draft=$(mline "$dir" draft_model)
  if [ -z "$model" ]; then
    unplaced "$set" "no '# model' line"
    return 0
  fi
  low=$(printf '%s %s' "$model" "$draft" | tr 'A-Z' 'a-z')
  case $low in
    *dspark* | */draft/*)
      leftout "$set" "a set of the DSpark draft ('# model' $model, '# draft_model' ${draft:-none}): its source is not verified"
      return 0
      ;;
  esac
  case $model in
    */fixture-variants/*)
      tag=${model#*/fixture-variants/}
      tag=${tag%%/*}
      if ! grep -qx "$tag" "$W/tags.txt"; then
        unplaced "$set" "'# model' is under fixture-variants/$tag, which is no tag of tools/fixture-variants.tsv"
        return 0
      fi
      leftout "$set" "a set of the fixture variant '$tag': the variant file is requantized on the box and cannot be made again byte for byte off it"
      return 0
      ;;
  esac
  case $model in
    "$FIXTURE_ROOT"/*/*) ;;
    *)
      unplaced "$set" "'# model' $model is not $FIXTURE_ROOT/<dir>/<file>"
      return 0
      ;;
  esac
  rest=${model#"$FIXTURE_ROOT"/}
  fdir=${rest%%/*}
  file=${rest#*/}
  case $file in
    */*)
      unplaced "$set" "'# model' $model is not $FIXTURE_ROOT/<dir>/<file>"
      return 0
      ;;
  esac
  family=$(awk -F'\t' -v d="$fdir" '$2 == d { print $1; exit }' "$W/dirs.tsv")
  if [ -z "$family" ]; then
    unplaced "$set" "'# model' names the fixture dir '$fdir', which is no dir of crates/refset/src/fixture.rs DIRS"
    return 0
  fi
  if [ "$arch" != "$family" ]; then
    unplaced "$set" "'# arch' is '${arch:-none}' but the fixture dir '$fdir' of '# model' belongs to the family '$family'"
    return 0
  fi
  case $fixture in
    bloomery.fixture.*) ;;
    *)
      unplaced "$set" "no '# fixture' line: the set was not dumped from a fixture file"
      return 0
      ;;
  esac
  printf '%s\t%s\t%s\n' "$family" "$set" "$fixture" >> "$W/published.tsv"
}

# Every candidate under $DATA, placed; ends 65 on any unplaced set or a family whose sets disagree on `# fixture`.
assign() {
  local d set
  : > "$W/published.tsv"
  : > "$W/leftout.tsv"
  : > "$W/errors.tsv"
  [ -d "$DATA" ] || die 66 "no data directory $DATA"
  for d in "$DATA"/fx_* "$DATA"/ref-mtp/fx_*; do
    [ -d "$d" ] || continue
    set=${d#"$DATA"/}
    place "$set"
  done
  if [ -s "$W/errors.tsv" ]; then
    echo "refset-publish: $(wc -l < "$W/errors.tsv" | tr -d ' ') set(s) the assignment rule cannot place (nothing was hashed):" >&2
    awk -F'\t' '{ printf "  %s: %s\n", $1, $2 }' "$W/errors.tsv" >&2
    exit 65
  fi
  [ -s "$W/published.tsv" ] || [ -s "$W/leftout.tsv" ] || die 65 "no fx_ set under $DATA or $DATA/ref-mtp"
  local fam n
  for fam in $(cut -f1 "$W/published.tsv" | sort -u); do
    n=$(awk -F'\t' -v f="$fam" '$1 == f { print $3 }' "$W/published.tsv" | sort -u | wc -l | tr -d ' ')
    [ "$n" = 1 ] || die 65 "the family $fam: its sets carry $n different '# fixture' lines, so they were not all dumped from one fixture"
    grep -q "^$fam$TAB" "$ROOT/tools/refset-publish/sources.tsv" || die 65 "the family $fam has no row in tools/refset-publish/sources.tsv (model and licence)"
    licence_ok "$fam"
  done
}

# The family's licence copy exists and hashes to its sources.tsv row; anything else is a named error (65), since the
# published LICENCES/<fam>.txt is the notice the licence asks every copy to carry.
licence_ok() {
  local fam=$1 want got f=$ROOT/tools/refset-publish/licences/$1.txt
  want=$(awk -F'\t' -v f="$fam" '$1 == f { print $5; exit }' "$ROOT/tools/refset-publish/sources.tsv")
  [ -n "$want" ] || die 65 "the family $fam: its sources.tsv row has no sha256 of the licence text"
  [ -f "$f" ] || die 65 "the family $fam: no licence text at $f"
  got=$($SHA "$f" | cut -d" " -f1)
  [ "$got" = "$want" ] || die 65 "the family $fam: $f hashes to $got, not the $want its sources.tsv row records"
}

# size<TAB>path of every file, NUL-separated names in, from stdin
sizes_of() {
  if [ "$STATFMT" = gnu ]; then
    xargs -0 stat -c "%s${TAB}%n"
  else
    xargs -0 stat -f "%z${TAB}%N"
  fi
}

# Hash and size every file of the published sets: W/sums.tsv (sha<TAB>path) and W/sizes.tsv (size<TAB>path).
hash_files() {
  local set odd
  : > "$W/files.nul"
  for set in $(cut -f2 "$W/published.tsv" | sort); do
    odd=$(find "$DATA/$set" ! -type f ! -type d | head -3)
    [ -z "$odd" ] || die 65 "the set $set holds an entry that is neither a file nor a directory (a link or a device), which a dataset cannot carry as it is: $odd"
    odd=$(find "$DATA/$set" -type f \( -name '*\\*' -o -name "*$TAB*" -o -name "*$NL*" \) | head -3)
    [ -z "$odd" ] || die 65 "the set $set holds a file whose name has a backslash, a tab or a newline: $odd"
    find "$DATA/$set" -type f -print0 >> "$W/files.nul"
  done
  xargs -0 -n 64 -P "$JOBS" $SHA < "$W/files.nul" > "$W/sums.raw" || die 65 "hashing failed"
  awk '{ h = $1; sub(/^[0-9a-f]+  /, ""); print h "\t" $0 }' "$W/sums.raw" | LC_ALL=C sort -t "$TAB" -k2,2 > "$W/sums.tsv"
  sizes_of < "$W/files.nul" | LC_ALL=C sort -t "$TAB" -k2,2 > "$W/sizes.tsv"
  local a b
  a=$(wc -l < "$W/sums.tsv" | tr -d ' ')
  b=$(wc -l < "$W/sizes.tsv" | tr -d ' ')
  [ "$a" = "$b" ] && [ "$a" != 0 ] || die 65 "$a hashes but $b sizes: a file changed while it was read"
}

# W/manifest.tsv: family, repo path, source, bytes, sha256 — sorted by repo path. The repo path is the path under --data
# (`fx_<set>/<file>`, `ref-mtp/fx_<set>/<file>`), so one download into $BLOOMERY_DATA is the install; the family is a column.
build_manifest() {
  awk -F'\t' -v OFS='\t' -v data="$DATA/" '
    FILENAME == ARGV[1] { fam[$2] = $1; next }
    FILENAME == ARGV[2] { size[$2] = $1; next }
    {
      path = $2; rel = substr(path, length(data) + 1)
      n = split(rel, c, "/")
      set = (c[1] == "ref-mtp") ? c[1] "/" c[2] : c[1]
      if (!(set in fam)) { print "refset-publish: " path " belongs to no placed set" > "/dev/stderr"; bad = 1; next }
      if (!(path in size)) { print "refset-publish: no size for " path > "/dev/stderr"; bad = 1; next }
      print fam[set], rel, path, size[path], $1
    }
    END { exit bad }' "$W/published.tsv" "$W/sizes.tsv" "$W/sums.tsv" | LC_ALL=C sort -t "$TAB" -k2,2 > "$W/manifest.tsv" || die 65 "a file is in no placed set"
  mkdir -p "$OUT"
  { printf '# family\trepo_path\tsource\tbytes\tsha256\n'; cat "$W/manifest.tsv"; } > "$OUT/MANIFEST.tsv"
  awk -F'\t' '{ printf "%s  %s\n", $5, $2 }' "$W/manifest.tsv" > "$OUT/SHA256SUMS"
}

gb() { awk -v b="$1" 'BEGIN { printf "%.2f", b / 1e9 }'; }

# The summary (stdout) and LISTING.txt (the summary and every file).
report() {
  local fam set files bytes sets total_f total_b total_s
  {
    echo "refset-publish: dry run"
    echo "repo: $REPO (dataset)"
    echo "data: $DATA"
    echo
    total_f=0 total_b=0 total_s=0
    for fam in $(cut -f1 "$W/published.tsv" | sort -u); do
      sets=$(awk -F'\t' -v f="$fam" '$1 == f' "$W/published.tsv" | wc -l | tr -d ' ')
      files=$(awk -F'\t' -v f="$fam" '$1 == f' "$W/manifest.tsv" | wc -l | tr -d ' ')
      bytes=$(awk -F'\t' -v f="$fam" '$1 == f { s += $4 } END { printf "%.0f", s }' "$W/manifest.tsv")
      echo "$fam/  sets=$sets files=$files bytes=$bytes ($(gb "$bytes") GB)"
      for set in $(awk -F'\t' -v f="$fam" '$1 == f { print $2 }' "$W/published.tsv" | sort); do
        awk -F'\t' -v f="$fam" -v s="$set" -v data="$DATA/" '
          $1 == f && index($3, data s "/") == 1 { n++; b += $4 }
          END { printf "  %-48s files=%d bytes=%.0f\n", s, n, b }' "$W/manifest.tsv"
      done
      total_s=$((total_s + sets))
      total_f=$((total_f + files))
      total_b=$((total_b + bytes))
    done
    echo
    echo "total: sets=$total_s files=$total_f bytes=$total_b ($(gb "$total_b") GB)"
    echo
    echo "left out ($(wc -l < "$W/leftout.tsv" | tr -d ' ')):"
    awk -F'\t' '{ printf "  %s: %s\n", $1, $2 }' "$W/leftout.tsv"
  } > "$W/summary.txt"
  {
    cat "$W/summary.txt"
    echo
    echo "files (family, repo path, source, bytes, sha256):"
    cat "$W/manifest.tsv"
  } > "$OUT/LISTING.txt"
  cat "$W/summary.txt"
  echo
  echo "out: $OUT (SHA256SUMS MANIFEST.tsv README.md LISTING.txt)"
}

# README.md from tools/refset-publish/README.tmpl
render_readme() {
  local tmpl=$ROOT/tools/refset-publish/README.tmpl src=$ROOT/tools/refset-publish/sources.tsv
  [ -f "$tmpl" ] || die 66 "no $tmpl"
  local fam dir model lic link sets files bytes
  : > "$W/table.md"
  : > "$W/fetch.txt"
  : > "$W/licences.md"
  : > "$W/setlist.md"
  printf '| family | fixture dir | sets | files | size |\n|---|---|---|---|---|\n' >> "$W/table.md"
  for fam in $(cut -f1 "$W/published.tsv" | sort -u); do
    IFS=$TAB read -r _ model lic link _ < <(grep "^$fam$TAB" "$src")
    dir=$(awk -F'\t' -v f="$fam" '$1 == f { print $2; exit }' "$W/dirs.tsv")
    sets=$(awk -F'\t' -v f="$fam" '$1 == f' "$W/published.tsv" | wc -l | tr -d ' ')
    files=$(awk -F'\t' -v f="$fam" '$1 == f' "$W/manifest.tsv" | wc -l | tr -d ' ')
    bytes=$(awk -F'\t' -v f="$fam" '$1 == f { s += $4 } END { printf "%.0f", s }' "$W/manifest.tsv")
    printf '| `%s` | `/models/fixtures/%s` | %s | %s | %s GB |\n' "$fam" "$dir" "$sets" "$files" "$(gb "$bytes")" >> "$W/table.md"
    printf -- '- `%s`: %s\n' "$fam" "$(awk -F'\t' -v f="$fam" '$1 == f { print "`" $2 "`" }' "$W/published.tsv" | sort | paste -sd, - | sed 's/,/, /g')" >> "$W/setlist.md"
    printf 'fixture fetch %s /models/fixtures/%s\n' "$fam" "$dir" >> "$W/fetch.txt"
    printf -- '- %s (family `%s`): %s, [`LICENCES/%s.txt`](LICENCES/%s.txt), a copy of [the base model'"'"'s LICENSE](%s)\n' "$model" "$fam" "$lic" "$fam" "$fam" "$link" >> "$W/licences.md"
  done
  if [ -s "$W/leftout.tsv" ]; then
    awk -F'\t' '{ printf "- `%s`: %s\n", $1, $2 }' "$W/leftout.tsv" > "$W/leftout.md"
  else
    echo "Nothing." > "$W/leftout.md"
  fi
  local total_f total_b total_s line
  total_f=$(wc -l < "$W/manifest.tsv" | tr -d ' ')
  total_b=$(awk -F'\t' '{ s += $4 } END { printf "%.0f", s }' "$W/manifest.tsv")
  total_s=$(wc -l < "$W/published.tsv" | tr -d ' ')
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in
      @FAMILY_TABLE@) cat "$W/table.md" ;;
      @FETCH_COMMANDS@) cat "$W/fetch.txt" ;;
      @LICENCE_LIST@) cat "$W/licences.md" ;;
      @LEFT_OUT@) cat "$W/leftout.md" ;;
      @SET_LIST@) cat "$W/setlist.md" ;;
      *)
        line=${line//@REPO@/$REPO}
        line=${line//@SET_COUNT@/$total_s}
        line=${line//@FILE_COUNT@/$total_f}
        line=${line//@TOTAL_GB@/$(gb "$total_b")}
        printf '%s\n' "$line"
        ;;
    esac
  done < "$tmpl" > "$OUT/README.md"
  if grep -q '@[A-Z_]*@' "$OUT/README.md"; then
    die 65 "README.tmpl has a placeholder this script does not fill: $(grep -o '@[A-Z_]*@' "$OUT/README.md" | sort -u | tr '\n' ' ')"
  fi
}

# Hardlink every published file into STAGE at its repo path, and write README.md, SHA256SUMS and LICENCES/<family>.txt beside them. STAGE must be on the
# filesystem of --data: a link that fails is a named error (65) and nothing is copied.
stage_tree() {
  local stage=$1 rp src fam
  rm -rf "$stage"
  mkdir -p "$stage"
  while IFS=$TAB read -r _ rp src _; do
    mkdir -p "$stage/$(dirname "$rp")"
    ln "$src" "$stage/$rp" 2> /dev/null || die 65 "cannot hardlink $src to $stage/$rp: the stage must be on the filesystem of $DATA (set REFSET_PUBLISH_STAGE); nothing is copied"
  done < "$W/manifest.tsv"
  cp "$OUT/README.md" "$OUT/SHA256SUMS" "$stage/"
  mkdir -p "$stage/LICENCES"
  for fam in $(cut -f1 "$W/manifest.tsv" | sort -u); do
    cp "$ROOT/tools/refset-publish/licences/$fam.txt" "$stage/LICENCES/" || die 65 "cannot copy the licence text of $fam into $stage/LICENCES"
  done
}

upload() {
  [ "${REFSET_PUBLISH_GO:-}" = 1 ] || die 64 "--upload publishes $REPO to Hugging Face; it runs only with REFSET_PUBLISH_GO=1 (the lead's go, after the 0.2.10 tag)"
  [ -x "$HF" ] || die 66 "no hf at $HF"
  local stage=${REFSET_PUBLISH_STAGE:-$(dirname "$DATA")/refset-publish-stage}
  stage_tree "$stage"
  HF_TOKEN_PATH=$HF_TOKEN_FILE "$HF" upload --repo-type dataset "$REPO" "$stage"
}

# ---------------------------------------------------------------------------------------------------------------------
self_test() {
  local T fails=0 here
  here=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
  T=$(mktemp -d)
  trap "rm -rf '$T'" EXIT
  # set NAME MODEL ARCH [DRAFT]: a set of two files and a manifest under $T/data
  mkset() {
    local d=$T/data/$1
    mkdir -p "$d"
    printf 'aaaaaaaaaa' > "$d/a.f32"
    printf 'bbbbbbbbbbbbbbbbbbbb' > "$d/b(reshaped).0.f32"
    {
      printf '# dump_ref\n# model\t%s\n' "$2"
      printf '# fixture\tbloomery.fixture.seed=1 bloomery.fixture.source_header_sha256=ab\n'
      [ -z "${4:-}" ] || printf '# draft_model\t%s\n' "$4"
      printf '# arch\t%s\n' "$3"
    } > "$d/MANIFEST.tsv"
  }
  say() { echo "refset-publish self-test: $*" >&2; }
  want() { # NAME CONDITION-RESULT
    if [ "$2" != 0 ]; then say "FAIL $1"; fails=$((fails + 1)); fi
  }
  run() { REFSET_PUBLISH_ROOT=$ROOT BLOOMERY_DATA=$T/data bash "$here" "$@"; }

  mkset fx_ref_glm5next_a /models/fixtures/glm5next/g-00001-of-00002.gguf glm5next
  mkset ref-mtp/fx_prose64 /models/fixtures/glm5next/g-00001-of-00002.gguf glm5next
  mkset fx_ref_qwen4exp /models/fixtures/qwen38/q-00001-of-00001.gguf qwen4exp
  mkset fx_iq4xs_ref_glm5next /models/fixture-variants/iq4xs/glm5next/g-iq4xs-00001-of-00002.gguf glm5next
  mkset fx_ref_deepseek41_dspark /models/fixtures/v41/v-00001-of-00004.gguf deepseek41 /models/fixtures/v41/draft/dspark-00001.gguf
  mkdir -p "$T/data/glm5next" # a family's corpus dir is no candidate
  printf 'x' > "$T/data/glm5next/corpus-prose.ids"

  # 1. the assignment, the left-out block, the files
  local o1 rc
  o1=$(run --out "$T/out1" --jobs 2 2>&1) && rc=0 || rc=$?
  want "the fake tree is placed (rc $rc)" "$rc"
  case $o1 in *"glm5next/  sets=2 files=6 "*) rc=0 ;; *) rc=1 ;; esac
  want "glm5next holds its two sets and six files" "$rc"
  case $o1 in *"qwen4exp/  sets=1 files=3 "*) rc=0 ;; *) rc=1 ;; esac
  want "qwen4exp holds its set" "$rc"
  case $o1 in *"deepseek41/"*) rc=1 ;; *) rc=0 ;; esac
  want "the DSpark set is not published" "$rc"
  case $o1 in *"left out (2):"*"fx_iq4xs_ref_glm5next: a set of the fixture variant 'iq4xs'"*) rc=0 ;; *) rc=1 ;; esac
  want "the variant is left out by name and reason" "$rc"
  case $o1 in *"fx_ref_deepseek41_dspark: a set of the DSpark draft"*) rc=0 ;; *) rc=1 ;; esac
  want "the DSpark set is left out by name and reason" "$rc"
  case $(sed -n 2p "$T/out1/MANIFEST.tsv" | cut -f2) in fx_ref_glm5next_a/MANIFEST.tsv) rc=0 ;; *) rc=1 ;; esac
  want "a repo path is the path under the data dir" "$rc"
  grep -q "^glm5next$TAB""ref-mtp/fx_prose64/a.f32$TAB" "$T/out1/MANIFEST.tsv" && rc=0 || rc=1
  want "an MTP set keeps its ref-mtp directory and its family column" "$rc"
  grep -q "^[0-9a-f]\{64\}  ref-mtp/fx_prose64/a.f32$" "$T/out1/SHA256SUMS" && rc=0 || rc=1
  want "SHA256SUMS paths are relative to the data dir" "$rc"
  local want_bytes got_bytes
  want_bytes=$( (find "$T/data/fx_ref_glm5next_a" "$T/data/ref-mtp/fx_prose64" "$T/data/fx_ref_qwen4exp" -type f -print0 | xargs -0 cat | wc -c) | tr -d ' ')
  got_bytes=$(grep -v '^#' "$T/out1/MANIFEST.tsv" | awk -F'\t' '{ s += $4 } END { printf "%.0f", s }')
  [ "$want_bytes" = "$got_bytes" ] && rc=0 || rc=1
  want "the bytes total ($got_bytes, cat | wc -c says $want_bytes)" "$rc"
  local bad=0 fam rp src by sh
  while IFS=$TAB read -r fam rp src by sh; do
    [ "$(wc -c < "$src" | tr -d ' ')" = "$by" ] || bad=1
    [ "$($SHA < "$src" | cut -d' ' -f1)" = "$sh" ] || bad=1
    grep -qx "$sh  $rp" "$T/out1/SHA256SUMS" || bad=1
  done < <(grep -v '^#' "$T/out1/MANIFEST.tsv")
  want "every row's bytes, sha256 and SHA256SUMS line match a fresh read" "$bad"
  grep -q '^fixture fetch glm5next /models/fixtures/glm5next$' "$T/out1/README.md" && rc=0 || rc=1
  want "the README names the fetch command of a family" "$rc"

  # 1b. the staging: hardlinks at the repo paths with README.md and SHA256SUMS beside them, and a named error when a link fails
  mkdir -p "$T/w"
  grep -v '^#' "$T/out1/MANIFEST.tsv" > "$T/w/manifest.tsv"
  ( DATA=$T/data OUT=$T/out1 W=$T/w; stage_tree "$T/stage" ) > /dev/null 2>&1 && rc=0 || rc=1
  want "stage_tree stages the fake tree" "$rc"
  [ "$T/stage/ref-mtp/fx_prose64/a.f32" -ef "$T/data/ref-mtp/fx_prose64/a.f32" ] && [ -f "$T/stage/README.md" ] && [ -f "$T/stage/SHA256SUMS" ] && rc=0 || rc=1
  want "a staged file is a hardlink of its source (same inode), README.md and SHA256SUMS are beside them" "$rc"
  cmp -s "$T/stage/LICENCES/glm5next.txt" "$ROOT/tools/refset-publish/licences/glm5next.txt" && [ "$(ls "$T/stage/LICENCES" | paste -sd' ' -)" = "$(cut -f1 "$T/w/manifest.tsv" | sort -u | sed 's/$/.txt/' | paste -sd' ' -)" ] && rc=0 || rc=1
  want "the stage carries the licence text of each published family, and only those" "$rc"
  grep -q '(family `glm5next`): MIT, \[`LICENCES/glm5next.txt`\](LICENCES/glm5next.txt), a copy of \[the base model.s LICENSE\](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/[0-9a-f]\{40\}/LICENSE)$' "$T/out1/README.md" && rc=0 || rc=1
  want "the README links the published licence text and the base model's LICENSE at a revision" "$rc"
  mkdir -p "$T/lroot/tools/refset-publish/licences"
  cp "$ROOT/tools/refset-publish/sources.tsv" "$T/lroot/tools/refset-publish/"
  { cat "$ROOT/tools/refset-publish/licences/glm5next.txt"; echo; } > "$T/lroot/tools/refset-publish/licences/glm5next.txt"
  o2=$( ( ROOT=$T/lroot; licence_ok glm5next ) 2>&1 ) && rc=0 || rc=$?
  { [ "$rc" = 65 ] && case $o2 in *"the family glm5next: $T/lroot/tools/refset-publish/licences/glm5next.txt hashes to "*", not the "*) true ;; *) false ;; esac; } && rc=0 || rc=1
  want "a licence copy that differs from its recorded sha256 ends 65 by name" "$rc"
  rm "$T/lroot/tools/refset-publish/licences/glm5next.txt"
  o2=$( ( ROOT=$T/lroot; licence_ok glm5next ) 2>&1 ) && rc=0 || rc=$?
  { [ "$rc" = 65 ] && case $o2 in *"the family glm5next: no licence text at "*) true ;; *) false ;; esac; } && rc=0 || rc=1
  want "a missing licence copy ends 65 by name" "$rc"
  printf 'glm5next\tfx_gone/a.f32\t%s/data/fx_gone/a.f32\t1\tab\n' "$T" > "$T/w/manifest.tsv"
  o2=$( ( DATA=$T/data OUT=$T/out1 W=$T/w; stage_tree "$T/stage2" ) 2>&1 ) && rc=0 || rc=$?
  { [ "$rc" = 65 ] && case $o2 in *"cannot hardlink $T/data/fx_gone/a.f32 to $T/stage2/fx_gone/a.f32"*) true ;; *) false ;; esac; } && rc=0 || rc=1
  want "a link that fails ends 65 naming both paths" "$rc"

  # 2. a stray set is a named error and nothing is hashed
  mkset fx_ref_stray /models/fixtures/nosuchdir/x-00001-of-00001.gguf glm5next
  mkset fx_ref_archmix /models/fixtures/glm5next/g-00001-of-00002.gguf qwen4exp
  mkdir -p "$T/data/fx_ref_nomanifest"
  local o2
  o2=$(run --out "$T/out2" 2>&1) && rc=0 || rc=$?
  [ "$rc" = 65 ] && rc=0 || rc=1
  want "a set the rule cannot place ends 65" "$rc"
  for n in fx_ref_stray fx_ref_archmix fx_ref_nomanifest; do
    case $o2 in *"$n: "*) rc=0 ;; *) rc=1 ;; esac
    want "the error names $n" "$rc"
  done
  [ ! -e "$T/out2/MANIFEST.tsv" ] && rc=0 || rc=1
  want "nothing was written on a named error" "$rc"
  rm -rf "$T/data/fx_ref_stray" "$T/data/fx_ref_archmix" "$T/data/fx_ref_nomanifest"

  # 3. two fixtures in one family
  mkset fx_ref_glm5next_b /models/fixtures/glm5next/g-00001-of-00002.gguf glm5next
  sed -i.bak 's/source_header_sha256=ab/source_header_sha256=cd/' "$T/data/fx_ref_glm5next_b/MANIFEST.tsv" && rm -f "$T/data/fx_ref_glm5next_b/MANIFEST.tsv.bak"
  o2=$(run --out "$T/out3" 2>&1) && rc=0 || rc=$?
  { [ "$rc" = 65 ] && case $o2 in *"2 different '# fixture' lines"*) true ;; *) false ;; esac; } && rc=0 || rc=1
  want "a family whose sets name two fixtures ends 65" "$rc"
  rm -rf "$T/data/fx_ref_glm5next_b"

  # 4. the refusals: --upload without the go, an --out inside the data
  o2=$(run --upload --out "$T/out4" 2>&1) && rc=0 || rc=$?
  { [ "$rc" = 64 ] && case $o2 in *REFSET_PUBLISH_GO=1*) true ;; *) false ;; esac; } && rc=0 || rc=1
  want "--upload refuses without REFSET_PUBLISH_GO=1 (64)" "$rc"
  [ ! -e "$T/out4" ] && rc=0 || rc=1
  want "a refused --upload wrote nothing" "$rc"
  o2=$(REFSET_PUBLISH_GO=0 run --upload 2>&1) && rc=0 || rc=$?
  [ "$rc" = 64 ] && rc=0 || rc=1
  want "--upload with the go set to 0 refuses" "$rc"
  o2=$(run --out "$T/data/out" 2>&1) && rc=0 || rc=$?
  [ "$rc" = 64 ] && rc=0 || rc=1
  want "an --out under the data directory is refused (64)" "$rc"

  if [ "$fails" = 0 ]; then
    echo "refset-publish: self-test ok"
  else
    echo "refset-publish: self-test $fails failed" >&2
    return 1
  fi
}

# ---------------------------------------------------------------------------------------------------------------------
MODE=dry
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) MODE=dry ;;
    --upload) MODE=upload ;;
    --self-test) MODE=self ;;
    --data) DATA=${2:?--data DIR}; shift ;;
    --out) OUT=${2:?--out DIR}; shift ;;
    --jobs) JOBS=${2:?--jobs N}; shift ;;
    *) die 64 "unknown argument $1 (--dry-run, --upload, --self-test, --data DIR, --out DIR, --jobs N)" ;;
  esac
  shift
done
[ "$MODE" != self ] || { self_test; exit $?; }
# The upload's refusal comes before anything is read.
if [ "$MODE" = upload ] && [ "${REFSET_PUBLISH_GO:-}" != 1 ]; then
  die 64 "--upload publishes $REPO to Hugging Face; it runs only with REFSET_PUBLISH_GO=1 (the lead's go, after the 0.2.10 tag)"
fi
case $JOBS in
  '') JOBS=$(nproc 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || echo 4); [ "$JOBS" -le 16 ] || JOBS=16 ;;
  *[!0-9]* | 0) die 64 "--jobs is a whole number of at least 1, got '$JOBS'" ;;
esac
case $OUT in
  /models | /models/* | "$DATA" | "$DATA"/*) die 64 "--out $OUT is under /models or the data directory $DATA: this tool writes nowhere there" ;;
esac
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
read_rules
assign
hash_files
build_manifest
render_readme
report
if [ "$MODE" = upload ]; then upload; fi
