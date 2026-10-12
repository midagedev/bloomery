#!/usr/bin/env bash
# The record-kind registration check (Mac, text only, no build): a shared module that
# prints record kinds is a group of record.rs's `groups`, every seat list is composed of
# its own kinds and the groups it uses, and the module's entry requires its group at run
# time (`record::require`). Twice a seat reached a shared module's print without
# registering its kind (the GLM seat and `vision`; the GLM and V4.1 seats and the `hf`
# fetch lines) and both shipped, because the flag path that reaches the print was never
# started by a gate. This check holds each group equal to what its module constructs and
# every printer that two or more binaries reach a group.
# The reader is tools/check-records.py (its header: what a printing file, a root and a
# reach are, the blind spots, and the `schema` subcommand that predicts which checked-in
# schema files `just records-refresh` rewrites).
set -euo pipefail
exec python3 "$(dirname "$0")/check-records.py" check "$@"
