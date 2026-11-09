#!/usr/bin/env bash
# eyeball-serve.sh — start a server on a model file, send a few chat prompts through /v1/chat/completions and print the
# prompts with the answers the model decoded, so a person can read whether the file answers sensibly. No timing: the
# numbers are the sitting runners' (tools/ref/depth-serve.sh); this prints text.
#
#   tools/ref/eyeball-serve.sh <seat> <model file> <prompts file> [--place P] [--ctx C] [--max-tokens N] [--port PORT]
#
# The prompts file holds one prompt a line (a blank line or a `#` line is skipped). Each prompt is one user turn with no
# system prompt, temperature 0, `max_tokens` N (default 400). The answer's `reasoning_content`, when the model gives one, is
# printed before its `content`. The server runs from the tree's target/release/bloomery-serve on the loopback port
# (default 18190) and is stopped at the end, or by a trap on exit. Exit 65 for a bad argument, 69 when the server does not
# answer /health within 20 min, else the number of prompts that got no answer.
set -u
usage() { echo "usage: $0 <seat> <model file> <prompts file> [--place P] [--ctx C] [--max-tokens N] [--port PORT]" >&2; exit 65; }
[ $# -ge 3 ] || usage
seat=$1 model=$2 prompts=$3; shift 3
place='' ctx='' max_tokens=400 port=18190
while [ $# -gt 0 ]; do
  case $1 in
    --place) place=$2; shift 2 ;;
    --ctx) ctx=$2; shift 2 ;;
    --max-tokens) max_tokens=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    *) usage ;;
  esac
done
[ -f "$model" ] || { echo "no model file $model" >&2; exit 65; }
[ -f "$prompts" ] || { echo "no prompts file $prompts" >&2; exit 65; }
bin=target/release/bloomery-serve
[ -x "$bin" ] || { echo "no $bin: build the tree first" >&2; exit 65; }
args=(--model "$seat" -m "$model" --host 127.0.0.1 --port "$port")
[ -z "$place" ] || args+=(--place "$place")
[ -z "$ctx" ] || args+=(--ctx-size "$ctx")
log=$(mktemp -t eyeball-serve.XXXXXX)
# BLOOMERY_REF_MODEL (tools/box.sh sets a default) would name a second model: the server takes -m alone.
env -u BLOOMERY_REF_MODEL "$bin" "${args[@]}" > "$log" 2>&1 &
spid=$!
trap 'kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null' EXIT
echo "[eyeball] server pid $spid: $bin ${args[*]} (log $log)"
t0=$SECONDS
until curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
  kill -0 "$spid" 2>/dev/null || { echo "[eyeball] the server exited before /health answered; its log:"; tail -n 20 "$log"; exit 69; }
  [ $((SECONDS - t0)) -lt 1200 ] || { echo "[eyeball] no /health in 1200 s; its log:"; tail -n 20 "$log"; exit 69; }
  sleep 2
done
echo "[eyeball] loaded in $((SECONDS - t0)) s"
grep -m1 'listening' "$log" || true
bad=0 n=0
while IFS= read -r line; do
  case $line in '' | '#'*) continue ;; esac
  n=$((n + 1))
  req=$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]}],"temperature":0,"max_tokens":int(sys.argv[2])}))' "$line" "$max_tokens")
  echo
  echo "=== prompt $n: $line"
  t1=$(date +%s.%N)
  resp=$(curl -s --max-time 1800 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' -d "$req")
  t2=$(date +%s.%N)
  python3 - "$resp" "$t1" "$t2" <<'PY' || bad=$((bad + 1))
import json, sys
r = json.loads(sys.argv[1])
wall = float(sys.argv[3]) - float(sys.argv[2])
ch = r["choices"][0]
m = ch["message"]
if m.get("reasoning_content"):
    print("--- reasoning:")
    print(m["reasoning_content"])
u = r.get("usage", {})
print("--- answer (finish=%s, tokens=%s):" % (ch.get("finish_reason"), u.get("completion_tokens")))
print(m.get("content"))
t = r.get("timings")
if t:
    # the server's own timings (llama-server's fields): no lease, no witness, so a reading, not a sitting number
    print("--- timings (the server's, not a sitting's): prompt %s tok in %.1f s = %.2f tok/s; decode %s tok in %.1f s = %.2f tok/s; wall %.1f s"
          % (t.get("prompt_n"), t.get("prompt_ms", 0) / 1e3, t.get("prompt_per_second", 0),
             t.get("predicted_n"), t.get("predicted_ms", 0) / 1e3, t.get("predicted_per_second", 0), wall))
else:
    print("--- timings: none in the reply; prompt %s tok, completion %s tok, wall %.1f s = %.2f tok/s over both"
          % (u.get("prompt_tokens"), u.get("completion_tokens"), wall,
             ((u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)) / wall if wall > 0 else 0))
PY
done < "$prompts"
echo
echo "[eyeball] $n prompts, $bad without an answer"
exit "$bad"
