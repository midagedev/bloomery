#!/usr/bin/env bash
# The one-bundle-load ratchet (Mac, grep only, no build): a `#[cuda_module]` family's generated
# `load` reads its crate's whole embedded bundle, so calling it outside the shared slot is one more
# whole-bundle cuModuleLoadData per call — the class the shared_module! path closed (one load per
# (device, crate bundle) per process, counted by bloomery_gpu::bundle_loads). This file owns the
# detection rule:
#   families every `#[cuda_module]`-attributed `mod NAME` under crates/ except oxide-ice-unroll
#            (the reproducer outside the workspace)
#   uses     every code line under crates/<dir>/src that calls `<family>::load(` literally
#   allowed  the bundle-path closure idiom alone — `::load(anchor).map(|loaded|
#            loaded.as_cuda_module().clone())`, the body shared_module! and start_bundle_load's
#            closures spell — and pure `//` comment lines; the macro's own `$module::load` is a
#            metavariable and never matches a family name
# Every other family load goes through `crate::shared_module!` (inside bloomery-gpu) or
# `bloomery_gpu::shared_module!` (outside it), which binds to the slot's one module.
# Exit: 0 no raw load (a line naming the family count). 1 each raw load, its file:line and the
# family. 70 the family scan found nothing (the collector rotted; no silent pass).
# Runs under the Mac's bash 3.2: no associative arrays, no mapfile.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export LC_ALL=C

say() { printf 'check-loads: %s\n' "$*"; }

# The generated load's one legitimate caller shape: the closure that hands the loaded module to
# bundle_module/start_bundle_load. Matching it (not a file exemption) keeps lib.rs itself covered.
idiom='::load\(anchor\)\.map\(\|loaded\| loaded\.as_cuda_module\(\)\.clone\(\)\)'

families="$(
    find crates -type f -name '*.rs' -not -path 'crates/oxide-ice-unroll/*' | sort |
        xargs awk '
        /^[[:space:]]*#[[:space:]]*\[[[:space:]]*cuda_module\]/ { want = 1; next }
        want {
            if ($0 ~ /^[[:space:]]*(\/\/|#!?\[)/) next
            if ($0 ~ /^[[:space:]]*(pub([[:space:]]*\([a-z]+\))?[[:space:]]+)?mod[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
                name = $0
                sub(/.*mod[[:space:]]+/, "", name)
                sub(/[^A-Za-z0-9_].*/, "", name)
                print name
            }
            want = 0
        }
        ' | sort -u
)"
[ -n "$families" ] || { say "no #[cuda_module] family found under crates/ — the scan collector rotted" >&2; exit 70; }
n_families="$(printf '%s\n' "$families" | wc -l | tr -d ' ')"

pattern="$(printf '%s\n' "$families" | paste -sd'|' -)"
status=0
while IFS= read -r hit; do
    [ -n "$hit" ] || continue
    say "raw generated-module load: $hit"
    status=1
done < <(
    find crates -type f -name '*.rs' -path '*/src/*' -not -path 'crates/oxide-ice-unroll/*' | sort |
        xargs grep -nE "(^|[^A-Za-z0-9_])(${pattern})::load\(" |
        grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' |
        grep -vE "$idiom" || true
)

if [ "$status" -eq 0 ]; then
    say "$n_families #[cuda_module] families, every load through the shared slot"
fi
exit "$status"
