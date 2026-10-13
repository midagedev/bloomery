#!/usr/bin/env bash
# One headless Claude Code run of the review bot (.github/workflows/review.yml).
#
#   tools/review/run-claude.sh reviewer|validator BASE_TREE PR_TREE CTX OUT TRACE
#
# The prompt is .github/review/<role>.md from BASE_TREE plus the four absolute paths. The run's working directory is
# OUT, an empty directory of ours, so nothing in the PR tree is loaded as project configuration (a PR's
# .claude/settings.json hooks, .mcp.json or CLAUDE.md never apply); the PR tree, the base tree and CTX are added
# read-only. Tools: Read, Grep, Glob, and Write/Edit under OUT only. No Bash, no web, no subagents. Settings come from
# the user scope only (the runner has none). The full stream-json transcript goes to TRACE/<role>.ndjson and its token
# and tool accounting (tools/review/usage.py) to TRACE/<role>.usage.json; TRACE is outside OUT, so the run cannot
# touch it and nothing in it is posted. The exit code is claude's, and a result that reports an error is exit 1.
set -euo pipefail

role=${1:?role}; base=${2:?base tree}; pr=${3:?pr tree}; ctx=${4:?ctx}; out=${5:?out}; trace=${6:?trace}
case "$role" in
  reviewer) turns=60 ;;
  validator) turns=30 ;;
  *) echo "run-claude.sh: role is reviewer or validator, not '$role'" >&2; exit 64 ;;
esac
for d in "$base" "$pr" "$ctx" "$out" "$trace"; do
  case "$d" in /*) ;; *) echo "run-claude.sh: '$d' is not an absolute path" >&2; exit 64 ;; esac
  [ -d "$d" ] || { echo "run-claude.sh: no directory $d" >&2; exit 64; }
done
instructions="$base/.github/review/$role.md"
[ -f "$instructions" ] || { echo "run-claude.sh: no $instructions" >&2; exit 66; }

prompt=$(cat "$instructions"; printf '\n## Paths for this run\n\n- PR_TREE: %s\n- BASE_TREE: %s\n- CTX: %s\n- OUT: %s\n' \
  "$pr" "$base" "$ctx" "$out")

transcript="$trace/$role.ndjson"
rc=0
(
  cd "$out"
  claude -p "$prompt" < /dev/null \
    --model "${REVIEW_MODEL:-claude-sonnet-5-5}" \
    --effort high \
    --max-turns "$turns" \
    --output-format stream-json --verbose \
    --setting-sources user \
    --strict-mcp-config \
    --permission-mode dontAsk \
    --add-dir "$pr" "$base" "$ctx" \
    --allowedTools "Read(/$pr/**)" "Read(/$base/**)" "Read(/$ctx/**)" "Read(/$out/**)" "Edit(/$out/**)" "Write(/$out/**)" \
    --disallowedTools "Bash" "WebFetch" "WebSearch" "Task" "Agent" "NotebookEdit" "Read(//proc/**)" "Read(//sys/**)"
) > "$transcript" || rc=$?

prc=0
python3 "$base/tools/review/usage.py" "$role" "$transcript" "$trace/$role.usage.json" || prc=$?
[ "$rc" -ne 0 ] || rc=$prc
exit "$rc"
