#!/usr/bin/env bash
# The ptx-shapes name check (Mac, text only, no build): every entry of tools/ref/ptx-shapes.tsv names a kernel its
# binary's device crates declare, and every kernel those crates declare has a row — the two halves of
# tools/ptx-spill-check.sh's verdict that need no build or scan. A half-applied kernel change (a kernel deleted
# or added without its row) failed only on the box, inside a landing batch (train0211i, 2026-10-11).
# The reader is tools/check-ptx-names.py (its header: how an entry is derived from source, the macro family,
# the refusals, what it cannot see, the exit codes).
set -euo pipefail
exec python3 "$(dirname "$0")/check-ptx-names.py" check "$@"
