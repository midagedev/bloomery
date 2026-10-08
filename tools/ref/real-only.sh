#!/usr/bin/env bash
# The fixture tier's real-only guard: bash tools/ref/real-only.sh <recipe>
#
# A recipe whose gate has no fixture-tier conversion opens its box command with this call. Under BLOOMERY_TIER=fixture
# (tools/box.sh exports the tier into every box command, from the environment or from a BLOOMERY_TIER entry of
# BLOOMERY_BOX_ENV) it stops the command by name, exit 66 as tools/ref/ref-paths.sh does for a family with no fixture, before
# a build, a lock or a file: a gate never runs on a fixture it was not written for. The real tier, and no tier, pass.
#
# tools/gate-batch.sh reads the same call from the recipe's text: in a `--tier fixture` batch the item is deferred to the
# real tier without being called, so the batch never meets this exit (a run by hand does). The call must be the first command
# of every box.sh command of the recipe and name the recipe itself; gate-batch refuses a recipe that does not.
#
# Exit codes: 66 the fixture tier, 64 a missing recipe name or a BLOOMERY_TIER that is neither tier.
set -euo pipefail
if [ $# != 1 ] || [ -z "$1" ]; then
  echo "real-only.sh: usage: bash tools/ref/real-only.sh <recipe>" >&2
  exit 64
fi
case "${BLOOMERY_TIER:-real}" in
  real) ;;
  fixture)
    echo "$1: real-only: it has no fixture-tier conversion, so BLOOMERY_TIER=fixture does not run it; run it in the real tier (exit 66)" >&2
    exit 66
    ;;
  *)
    echo "real-only.sh: BLOOMERY_TIER is real or fixture (unset: real), got '${BLOOMERY_TIER}'" >&2
    exit 64
    ;;
esac
