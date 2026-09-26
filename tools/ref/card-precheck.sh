#!/usr/bin/env bash
# The card check a timed recipe runs first in its box command, before it builds anything: a run
# whose card is missing or refused stops here, not after minutes of cargo. It is lease_card
# (tools/ref/lease.sh) itself — tools/ref/card.py's lease mode with the same --rounds rule
# (ROUNDS, else BLOOMERY_AB_ROUNDS) — so it refuses exactly what lease_take would, with card.py's
# exit code. A runner whose own ROUNDS default differs from the card's is not seen here (the runner
# sets ROUNDS after this runs); lease_take stays the enforcement point and checks it again.
#
#   bash tools/ref/card-precheck.sh [docs/cards/<slug>.card] && cargo … && bash tools/ref/<runner>.sh …
#
# With no argument the card is the environment's BLOOMERY_LEASE_CARD (tools/box.sh exports what
# BLOOMERY_BOX_ENV names); with one it is the card a recipe passes itself, which must then agree with
# BLOOMERY_LEASE_CARD when that is set. Runs on the box (and on the Mac in the stub tests); builds
# nothing, takes no lock.
set -uo pipefail
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
# A dry run (BLOOMERY_DRY, which the depth, nsys and ncu runners read) prints command lines and takes
# no lease, so it needs no card; this says so and passes. A runner that ignores BLOOMERY_DRY still meets
# lease_take, which checks the card as always.
if [ -n "${BLOOMERY_DRY:-}" ]; then
  echo "card-precheck: BLOOMERY_DRY is set — a dry run takes no lease, so no card is checked here"
  exit 0
fi
if [ $# -gt 1 ]; then
  echo "card-precheck: usage: card-precheck.sh [docs/cards/<slug>.card]" >&2
  exit 64
fi
if [ $# = 1 ]; then
  if [ -n "${BLOOMERY_LEASE_CARD:-}" ] && [ "$BLOOMERY_LEASE_CARD" != "$1" ]; then
    echo "card-precheck: the recipe's card $1 and BLOOMERY_LEASE_CARD=$BLOOMERY_LEASE_CARD name two cards" >&2
    exit 64
  fi
  BLOOMERY_LEASE_CARD=$1
fi
lease_card
