#!/usr/bin/env bash
# One user-path probe of a GGUF (a sparse header stub or a real file): bloomery-serve with no flags loads it, then
# one completion whose prompt is longer than a gemv unit runs a prompt call and a few decode steps. Prints one line:
#   probe <label> load=ok|fail prompt=ok|fail|- secs=<load s> err="<first error line, or empty>"
# Run on the box from the remote dir, under the card lock (it is target/release/stub-probe there):
#   bash tools/gpu-gate.sh stub-probe <first shard> <label>
# The server is signalled only by the pid captured at its spawn. Bounds: STUB_LOAD_BOUND (default 900 s) for the
# load, 300 s for the completion.
set -u
model=$1 label=$2
bin=${STUB_SERVE_BIN:-target/release/bloomery-serve}
w=${STUB_OUT:-target/stub-probe}/$label
mkdir -p "$w"
# The user's path: no BLOOMERY_* setting reaches the server (the box's profile sets some, e.g. BLOOMERY_REF_MODEL).
# The card is the lock's, through CUDA_VISIBLE_DEVICES.
unset_args=()
for v in $(compgen -e | grep '^BLOOMERY_'); do unset_args+=(-u "$v"); done
env "${unset_args[@]}" "$bin" -m "$model" --port 0 > "$w/serve.log" 2>&1 &
pid=$!
echo "$pid" > "$w/serve.pid"
stop() {
  kill -TERM "$pid" 2> /dev/null
  for _ in $(seq 1 20); do kill -0 "$pid" 2> /dev/null || break; sleep 0.5; done
  kill -KILL "$pid" 2> /dev/null
  wait "$pid" 2> /dev/null
}
trap stop EXIT
t0=$(date +%s) port='' load=fail prompt=-
while kill -0 "$pid" 2> /dev/null; do
  port=$(sed -n 's/.*listening on [^ ]*:\([0-9][0-9]*\).*/\1/p' "$w/serve.log" | head -1)
  [ -n "$port" ] && break
  [ $(($(date +%s) - t0)) -le "${STUB_LOAD_BOUND:-900}" ] || break
  sleep 1
done
secs=$(($(date +%s) - t0))
rc=-
if [ -z "$port" ] && ! kill -0 "$pid" 2> /dev/null; then
  wait "$pid"
  rc=$?
fi
if [ -n "$port" ]; then
  load=ok
  text=$(printf 'The quick brown fox jumps over the lazy dog while the river runs past the old mill. %.0s' $(seq 1 24))
  body=$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "n_predict": 4, "temperature": 0}))' "$text")
  code=$(curl -s -o "$w/completion.json" -w '%{http_code}' -m 300 -H 'Content-Type: application/json' \
    -d "$body" "http://127.0.0.1:$port/completion")
  [ "$code" = 200 ] && prompt=ok || prompt="fail($code)"
fi
# The server's own fatal line is its last `bloomery-serve:` line; an earlier refusal can be a verdict it went past
# (the whole-fit line before a placed plan).
err=$(grep -E '^bloomery-serve[a-z0-9-]*: ' "$w/serve.log" | grep -v 'listening on' | tail -1 | cut -c1-300 | tr '"' "'")
[ "$prompt" = ok ] || [ -n "$err" ] || err=$(tail -1 "$w/serve.log" | cut -c1-300 | tr '"' "'")
echo "probe $label load=$load prompt=$prompt rc=$rc secs=$secs err=\"$err\""
