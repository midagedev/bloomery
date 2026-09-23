#!/usr/bin/env bash
# Bring the box's Cargo.lock back to this tree after a box-side cargo run.
#
# box.sh syncs one way (Mac -> box). After a dependency edit, cargo rewrites the
# box's copy of Cargo.lock and nothing carries it back: the next sync restores the
# old lock, cargo rewrites it again, and every timing runner refuses its binary as
# older than its sources (the lock's box-side mtime keeps moving). `just check`
# calls this last, so the refreshed lock lands in the worktree and in the round's
# diff. Prints one line when the lock changed; silent when it did not.
set -euo pipefail
HOST=${BLOOMERY_BOX:-ws}
HERE=$(cd "$(dirname "$0")/.." && pwd)
REMOTE=${BLOOMERY_REMOTE:-"~/repo/$(basename "$HERE")"}
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
# $REMOTE expands here on purpose; its "~" stays literal and the box shell expands it.
# shellcheck disable=SC2029
ssh "$HOST" "cat $REMOTE/Cargo.lock" > "$tmp"
if [ ! -s "$tmp" ]; then
  echo "lock-back: the box's $REMOTE/Cargo.lock is empty or missing" >&2
  exit 1
fi
if ! cmp -s "$HERE/Cargo.lock" "$tmp"; then
  cp "$tmp" "$HERE/Cargo.lock"
  echo "lock-back: Cargo.lock refreshed from the box ($REMOTE) — commit it with the dependency edit"
fi
