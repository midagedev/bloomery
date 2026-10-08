#!/usr/bin/env bash
# The defaults-twins check (Mac, text only, no build): a lever a gate pins off its
# default needs a defaults twin — a clause that runs the same model path at the
# default — or a named, dated reason why not, as a row of tools/defaults-twins.tsv.
# GitHub #3 shipped because every gate that ran the MTP draft pinned
# BLOOMERY_MTP_WIDTH=fixed, so no gate ran the default width chooser across a
# server's request sequence; this check holds every pin of that shape to a twin.
# The reader is tools/check-defaults.py (its header: what a pin is, what the scanner
# sees, its blind spots, the TSV's columns and the exit codes) — one reader for the
# census and the check, so the table cannot drift from what the tree pins.
set -euo pipefail
exec python3 "$(dirname "$0")/check-defaults.py" check
