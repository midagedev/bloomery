#!/usr/bin/env bash
# The engine thread's waits on the card on the step and prompt path are bounded (Mac, no build).
# Each row below names a function. Its body must call host::await_done (bounded by host::ENGINE_BOUND,
# a named GpuError::Stalled past it) before any blocking copy, and must never call `.synchronize()`:
# a driver wait behind a stream that never drains spins forever and prints nothing, which is what a
# driver that runs every stream on one hardware queue turns a stuck stream memory wait into.
# The rule reads the function's own lines only, not its callees.
# Exit codes: 0 every row holds; 1 a row breaks the rule (each named); 2 a row's file or function is
# not found, or a function is found more than once.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# file|the function's signature, as a fixed prefix of its `fn` line after the indent
ROWS=(
  "crates/gpu/src/head.rs|pub fn tokens(&self, gpu: &Gpu)"
  "crates/gpu/src/host/batch.rs|pub fn routed_ids(&mut self, key: BatchKey)"
  "crates/gpu/src/host/batch.rs|fn serve_tiered("
  "crates/gpu/src/lib.rs|pub fn fault(&self) -> Result<Option<Fault>, GpuError>"
  "crates/gpu/src/arch/qwen3moe/body38.rs|fn reset(&mut self, gpu: &Gpu)"
)

bad=0
for row in "${ROWS[@]}"; do
  file="${row%%|*}"
  sig="${row#*|}"
  if [ ! -f "$file" ]; then
    echo "check-waits: $file: no such file" >&2
    exit 2
  fi
  # The body: from the signature's line to the line where the braces it opened close.
  verdict="$(awk -v sig="$sig" '
    function strip(s) { sub(/^[ \t]+/, "", s); return s }
    !inbody && index(strip($0), sig) == 1 { found++; inbody = 1; depth = 0; opened = 0; waited = 0 }
    inbody {
      line = $0
      sub(/\/\/.*/, "", line)
      if (line ~ /\.synchronize\(\)/) { print "sync " NR }
      if (line ~ /await_done\(/) { waited = 1 }
      if (!waited && line ~ /(to_host_vec|copy_to_host|copy_from_host)\(/) { print "copy " NR }
      n = gsub(/\{/, "{", line); m = gsub(/\}/, "}", line)
      depth += n - m
      if (n > 0) { opened = 1 }
      if (opened && depth == 0) { if (!waited) { print "nowait " NR }; inbody = 0 }
    }
    END { print "found " found + 0 }
  ' "$file")"
  found="$(printf '%s\n' "$verdict" | awk '/^found /{print $2}')"
  if [ "$found" != 1 ]; then
    echo "check-waits: $file: '$sig' found $found times (want 1)" >&2
    exit 2
  fi
  while read -r kind at; do
    case "$kind" in
      sync) echo "check-waits: $file:$at: '$sig' calls .synchronize(); wait with host::await_done" >&2; bad=1 ;;
      copy) echo "check-waits: $file:$at: '$sig' copies before its host::await_done" >&2; bad=1 ;;
      nowait) echo "check-waits: $file:$at: '$sig' ends with no host::await_done" >&2; bad=1 ;;
    esac
  done < <(printf '%s\n' "$verdict" | grep -v '^found ')
done
if [ "$bad" = 0 ]; then
  echo "check-waits: ${#ROWS[@]} engine waits bounded"
fi
exit "$bad"
