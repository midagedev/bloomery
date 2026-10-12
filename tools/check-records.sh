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
# reach are, the blind spots). Its `schema --strict` holds each checked-in schema file's
# kind names, in order, to the list its binary registers: a list that gains or loses a
# kind fails here until `just records-refresh` rewrites the file. A kind's fields are
# rendered by record.rs's `schema`, in a crate this Mac cannot build, so a field change
# stays the box test's (`checked_in_schemas_are_current`).
set -euo pipefail
python3 "$(dirname "$0")/check-records.py" check "$@"
exec python3 "$(dirname "$0")/check-records.py" schema --strict "$@"
