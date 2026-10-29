#!/usr/bin/env bash
# The docs an agent or a user starts a model from, held to the code they repeat (Mac, text only, no build): the README's
# Use table and docs/models.md's Recommended file against the model catalog (tools/catalog.tsv), docs/agents.md's exit
# table, event names, stop causes and loading phases against crates/serve, and the pointers to docs/agents.md.
# The reader is tools/check-catalog-docs.py (its header: what each copy is held to and the exit codes), its self-test is
# listed in tools/check-recipes.sh with the other Python tools'.
set -euo pipefail
exec python3 "$(dirname "$0")/check-catalog-docs.py" check
