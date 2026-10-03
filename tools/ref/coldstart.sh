#!/usr/bin/env bash
# Start-to-/health wall of a serve binary under the driver's JIT cache states, and the count of device
# bundle loads one start makes. Functional, not timed: it takes no lease, so its seconds are not
# admissible as a benchmark; they say how much of a start is JIT and module loading.
#
#   tools/ref/coldstart.sh <binary> [the server's own flags…]
#     e.g. tools/ref/coldstart.sh target/release/bloomery-serve \
#            -m /models/clef-flash/Cloudflare_clef-flash-Q3_K_S.gguf \
#            --head /models/clef-flash/hf/joint_head.safetensors
#
# Arms, in this order (COLDSTART_ARMS overrides, space separated):
#   count    an empty cache, the start under `strace -f -e trace=openat`: every embedded-bundle load
#            (cuda_host::load_embedded_module_from_anchor) re-opens the binary it reads its bundle from
#            with O_NONBLOCK|O_NOFOLLOW, one open per `cuModuleLoadData` of a bundle, so the count of
#            those opens of <binary> is the count of bundle loads (`loads=`)
#   nocache  CUDA_CACHE_DISABLE=1: every load JITs its bundle's PTX
#   empty    an empty CUDA_CACHE_PATH: the first load of each bundle JITs, the rest hit
#   warm     the cache `empty` left; `warm2` the same again
# Each arm prints `coldstart <arm> ok=<0|1> secs=<s> cache_bytes=<n>` (`count` adds `loads=<n>`), ok=1
# when /health answered within COLDSTART_BOUND seconds (default 300). The server binds a free port
# (`--port 0`) and the port is read from its `listening on` line. Every server is signalled by the pid
# captured at its spawn and waited for.
set -u
B=${1:?usage: coldstart.sh <binary> [server flags…]}
shift
BIN=$(readlink -f "$B") || { echo "coldstart: no binary $B" >&2; exit 2; }
[ -x "$BIN" ] || { echo "coldstart: $BIN is not executable" >&2; exit 2; }
BOUND=${COLDSTART_BOUND:-300}
ARMS=${COLDSTART_ARMS:-count nocache empty warm warm2}
W=$(mktemp -d /tmp/coldstart.XXXXXX)
C=$W/cache
mkdir -p "$C"
PID=
cleanup() {
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then kill "$PID"; wait "$PID" 2>/dev/null; fi
  rm -rf "$W"
}
trap cleanup EXIT

# run <arm> <prefix…> -- <env…>: one start; the server's stderr and stdout go to $W/<arm>.log.
run() {
  local arm=$1 port='' ok=0 t0 t1
  shift
  local -a pre=() envs=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do pre+=("$1"); shift; done
  shift
  envs=("$@")
  t0=$(date +%s.%N)
  # The flags name the model once: the gates' BLOOMERY_REF_MODEL beside them would be a second name.
  env -u BLOOMERY_REF_MODEL "${envs[@]}" "${pre[@]}" "$BIN" "${SERVER_ARGS[@]}" --port 0 >"$W/$arm.log" 2>&1 &
  PID=$!
  local deadline=$(( $(date +%s) + BOUND ))
  while [ "$(date +%s)" -lt "$deadline" ] && kill -0 "$PID" 2>/dev/null; do
    if [ -z "$port" ]; then
      port=$(sed -n 's#.*listening on http://[^:]*:\([0-9]*\).*#\1#p' "$W/$arm.log" | head -1)
    fi
    if [ -n "$port" ] && curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 0.2
  done
  t1=$(date +%s.%N)
  # Under strace the spawned pid is strace's; the server is its child, signalled first by that pid.
  local child
  for child in $(cat /proc/"$PID"/task/*/children 2>/dev/null); do kill "$child" 2>/dev/null; done
  kill "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  PID=
  local extra=''
  if [ "$arm" = count ]; then
    local n
    n=$(grep -F "\"$BIN\"" "$W/strace.txt" | grep -c 'O_NONBLOCK|O_NOFOLLOW')
    extra=" loads=$n"
  fi
  echo "coldstart $arm ok=$ok secs=$(echo "$t1 - $t0" | bc) cache_bytes=$(du -sb "$C" | cut -f1)$extra"
  if [ "$ok" != 1 ]; then tail -5 "$W/$arm.log" | sed "s/^/coldstart $arm log: /"; fi
}

SERVER_ARGS=("$@")
for arm in $ARMS; do
  case $arm in
    count)
      command -v strace >/dev/null || { echo "coldstart: count needs strace" >&2; exit 2; }
      rm -rf "$C"; mkdir -p "$C"
      run count strace -f -qq -e trace=openat -o "$W/strace.txt" -- CUDA_CACHE_PATH="$C"
      rm -rf "$C"; mkdir -p "$C" ;;
    nocache) run nocache -- CUDA_CACHE_DISABLE=1 ;;
    empty) rm -rf "$C"; mkdir -p "$C"; run empty -- CUDA_CACHE_PATH="$C" ;;
    warm|warm2) run "$arm" -- CUDA_CACHE_PATH="$C" ;;
    *) echo "coldstart: unknown arm $arm" >&2; exit 64 ;;
  esac
done
