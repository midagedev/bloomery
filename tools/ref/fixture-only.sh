#!/usr/bin/env bash
# The real tier's fixture-only guard: bash tools/ref/fixture-only.sh <recipe>
#
# The mirror of tools/ref/real-only.sh. A recipe whose gate has sets on the fixture tier only (no real file or real-tier set
# exists for it yet) opens its box command with this call. Under BLOOMERY_TIER=real, or no tier (tools/box.sh exports the tier
# into every box command, from the environment or from a BLOOMERY_TIER entry of BLOOMERY_BOX_ENV), it stops the command by
# name, exit 66 as real-only.sh does, before a build, a lock or a file: a gate never runs on a tier it has no sets for. The
# fixture tier passes.
#
# tools/gate-batch.sh reads the same call from the recipe's text: in a `--tier real` batch the item is deferred to the fixture
# tier without being called, so the batch never meets this exit (a run by hand does). The call must be the first command of
# every box.sh command of the recipe and name the recipe itself; gate-batch refuses a recipe that does not. Drop the call when
# the recipe has a real-tier run.
#
# Exit codes: 66 the real tier, 64 a missing recipe name or a BLOOMERY_TIER that is neither tier.
set -euo pipefail
if [ $# != 1 ] || [ -z "$1" ]; then
  echo "fixture-only.sh: usage: bash tools/ref/fixture-only.sh <recipe>" >&2
  exit 64
fi
case "${BLOOMERY_TIER:-real}" in
  fixture) ;;
  real)
    echo "$1: fixture-only: it has sets on the fixture tier only, so the real tier does not run it; run it under BLOOMERY_TIER=fixture (exit 66)" >&2
    exit 66
    ;;
  *)
    echo "fixture-only.sh: BLOOMERY_TIER is real or fixture (unset: real), got '${BLOOMERY_TIER}'" >&2
    exit 64
    ;;
esac
