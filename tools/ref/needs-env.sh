#!/usr/bin/env bash
# The env guard: bash tools/ref/needs-env.sh <recipe> <VAR>
#
# A recipe whose gate needs an environment variable the batch cannot know (the address of a node outside the box, say) opens its
# box command with this call. When VAR is unset or empty in the box command's environment (tools/box.sh exports the entries of
# BLOOMERY_BOX_ENV into every box command), it stops the command by name, exit 67, before a build, a lock or a file: a gate
# never runs without the node it is written for, and the failure names the variable instead of a count of addresses.
#
# tools/gate-batch.sh reads the same call from the recipe's text: an item whose env (`NAME@K=V`) and the batch's
# BLOOMERY_BOX_ENV do not set VAR to a value is deferred, in either tier, without being called, so the batch never meets this
# exit (a run by hand does). The call must be the first command of every box.sh command of the recipe and name the recipe
# itself; gate-batch refuses a recipe that does not.
#
# Exit codes: 67 VAR is unset or empty, 64 a missing recipe name or variable, or a variable that is no name.
set -euo pipefail
if [ $# != 2 ] || [ -z "$1" ] || [ -z "$2" ]; then
  echo "needs-env.sh: usage: bash tools/ref/needs-env.sh <recipe> <VAR>" >&2
  exit 64
fi
case $2 in
  [A-Za-z_]*) ;;
  *)
    echo "needs-env.sh: '$2' is no variable name (usage: bash tools/ref/needs-env.sh <recipe> <VAR>)" >&2
    exit 64
    ;;
esac
case $2 in
  *[!A-Za-z0-9_]*)
    echo "needs-env.sh: '$2' is no variable name (usage: bash tools/ref/needs-env.sh <recipe> <VAR>)" >&2
    exit 64
    ;;
esac
if [ -z "${!2:-}" ]; then
  echo "$1: needs-env: it needs $2, which is unset or empty here, so it does not run; give it through BLOOMERY_BOX_ENV='$2=<value>' or the item's NAME@$2=<value> (exit 67)" >&2
  exit 67
fi
