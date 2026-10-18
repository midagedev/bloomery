#!/usr/bin/env bash
# The model-name ratchet (Mac, text only, no build): how many code lines in common code name a model, held two-sided
# per (file, family) by tools/model-tax.tsv, so that a model addition puts its name into registry files and its own
# paths, and a lift that removes names lowers the entry in the same change.
# The reader is tools/model_tax.py (its header: what is counted, the three tables, the failures, `rows` and `lower`);
# this wrapper only names it for `just check-model-tax`, tools/mac-static.sh and check-recipes.
# Exit: 0 clean, 1 a violation, 2 no verdict: an argument, a table or a file the reader cannot read.
# `--self-test` runs the reader's cases on fixture trees: each rule's FAIL and its pass.
set -euo pipefail
exec python3 "$(dirname "$0")/model_tax.py" "${@:-check}"
