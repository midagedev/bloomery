#!/usr/bin/env bash
# The stub probe sitting, on the box in the remote dir: header stubs of the picked repos' sets (sparse, real size),
# then one user-path probe per pick (picks.txt: repo, set, tier, label), V4.1 first, each under the 3090's gate lock.
# Results: target/stub-probe/results.txt, one `probe` line per pick.
set -u
D=target/stub-probe STUBS=${STUBS:-/models/census-stubs} H=tools/ref/stubprobe
mkdir -p "$D/census"
[ "${STUB_SKIP_CENSUS:-0}" = 1 ] || python3 -I "$H/census.py" "$D/census" --stubs "$STUBS" $(awk '{print $1}' "$H/picks.txt" | sort -u) > "$D/census.log" 2>&1
echo "stubs: $(grep -c . "$D/census.log") sets read, $(grep -c FAILED "$D/census.log") failed"
cp "$H/stub-probe.sh" target/release/stub-probe
chmod +x target/release/stub-probe
: > "$D/results.txt"
while read -r repo set tier label; do
  first=$(find "$STUBS/$repo" \( -name "$set.gguf" -o -name "$set-00001-of-*.gguf" \) | head -1)
  if [ -z "$first" ]; then
    echo "probe $label load=nostub prompt=- secs=0 err=\"no stub for $repo $set\"" | tee -a "$D/results.txt"
    continue
  fi
  BLOOMERY_GATE_CARD=3090 BLOOMERY_GATE_BOUND=1500 bash tools/gpu-gate.sh stub-probe "$first" "$label" \
    < /dev/null 2>> "$D/gate.err" | grep '^probe ' | tee -a "$D/results.txt"
done < "$H/picks.txt"
