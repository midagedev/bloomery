# swappace: resv41 A′'s diagnostic arm — generate_ds41 --place bp, two prose:64 arms under BLOOMERY_RESIDENCY=mid-p40-s1, through
# gpu-gate.sh (both gate locks, its 600 s bound), with a watchdog: once the first `arm` record is out,
# 90 s with no new output dumps generate_ds41's threads (gdb, else /proc task stacks) and ends it. The
# watched pid is found only among the descendants of the pid this script spawned, by comm.
# A witness line before and after the run: loadavg, IO and CPU pressure, and every compute process on the cards.
set -u
OUT=target/swappace-diag
rm -rf "$OUT"; mkdir -p "$OUT"
export BLOOMERY_GATE_V41_LOAD=1 BLOOMERY_RESIDENCY=mid-p40-s1 BLOOMERY_STEP_STATS=1 BLOOMERY_GATE_BOUND=600
echo "build $(date +%T)"
cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features gpu,deepseek41 --release --bin generate_ds41 > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; exit 1; }
grep 'Compiling bloomery' "$OUT/build.log"
witness() {
  echo "witness $1 $(date +%T) loadavg $(cut -d' ' -f1-3 /proc/loadavg) io $(head -1 /proc/pressure/io) cpu $(head -1 /proc/pressure/cpu)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed "s/^/witness $1 app /"
}
witness pre
echo "run $(date +%T)"
setsid bash tools/gpu-gate.sh generate_ds41 --place bp --arm prose:64 --arm prose:64 -n 96 --time > "$OUT/run.log" 2>&1 < /dev/null &
top=$!
echo "$top" > "$OUT/top.pid"
echo "spawned $top comm $(cat /proc/$top/comm)"
find_gen() {
  local p=$1 c
  if [ "$(cat /proc/$p/comm 2>/dev/null)" = generate_ds41 ]; then echo "$p"; return; fi
  for c in $(cat /proc/$p/task/*/children 2>/dev/null); do find_gen "$c"; done
}
gen='' last=-1 still=0 t0=$SECONDS
while kill -0 "$top" 2>/dev/null; do
  sleep 5
  if [ -z "$gen" ]; then
    gen=$(find_gen "$top" | head -n 1)
    [ -n "$gen" ] && { echo "$gen" > "$OUT/gen.pid"; echo "watching $gen comm $(cat /proc/$gen/comm) at $((SECONDS - t0)) s"; }
  fi
  if [ -n "$gen" ] && [ -z "${masks:-}" ] && grep -q '^arm ' "$OUT/run.log"; then
    masks=1
    for t in /proc/"$gen"/task/*; do
      c=$(cat "$t/comm" 2>/dev/null)
      case $c in swap-staging|engram-prefetch|generate_ds41) echo "mask $c tid ${t##*/} $(grep Cpus_allowed_list "$t/status")";; esac
    done > "$OUT/masks.txt"
    echo "== masks at the first arm"; cat "$OUT/masks.txt"
  fi
  sz=$(stat -c %s "$OUT/run.log")
  if [ "$sz" = "$last" ]; then still=$((still + 5)); else still=0; last=$sz; fi
  if [ -n "$gen" ] && grep -q '^arm ' "$OUT/run.log" && [ "$still" -ge 90 ]; then
    echo "no output for $still s at $((SECONDS - t0)) s: dumping $gen"
    if command -v gdb > /dev/null; then
      timeout 120 gdb -p "$gen" -batch -ex 'thread apply all bt' > "$OUT/stacks.txt" 2>&1
    else
      for t in /proc/"$gen"/task/*; do echo "== $t $(cat "$t/comm") wchan=$(cat "$t/wchan")"; cat "$t/stack" 2>/dev/null; done > "$OUT/stacks.txt"
    fi
    kill -TERM "$gen"; sleep 10; kill -0 "$gen" 2>/dev/null && kill -KILL "$gen"
    break
  fi
  if [ $((SECONDS - t0)) -ge 660 ]; then echo "watch bound 660 s"; break; fi
done
wait "$top"; rc=$?; witness post; echo "gpu-gate rc=$rc $(date +%T)"
echo "== run.log"; cat "$OUT/run.log"
[ -f "$OUT/stacks.txt" ] && { echo "== stacks"; cat "$OUT/stacks.txt"; }
exit 0
