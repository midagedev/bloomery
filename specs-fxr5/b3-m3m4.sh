#!/usr/bin/env bash
# fxr5's last FAIL-first runs, finished by the lead: m3 (GDN state not carried) through the fixture-tier e2e gate, m4 (eh
# halves swapped) through the fixture-tier mtp gate. Each: apply, run, revert, md5 back to the base. box-gc first and last.
set -u
cd /Users/hckim/repo/bloomery-fxr5 || exit 64
export BLOOMERY_REMOTE='~/repo/bloomery-fxr5'
O=specs-fxr5/out
base=$(git diff | md5)
echo "b3 base md5: $base" | tee "$O/b3-md5.log"
just box-gc > "$O/b3-gc-start.log" 2>&1; echo "box-gc start rc=$?"
run() { # $1 mutant, $2 recipe, $3 log stem
  git apply "specs-fxr5/mutants/$1.patch" || { echo "$1: apply failed"; return 90; }
  echo "$1 applied md5: $(git diff | md5)" | tee -a "$O/b3-md5.log"
  T0=$(date +%s)
  BLOOMERY_TIER=fixture just "$2" > "$O/$3.log" 2>&1
  rc=$?
  echo "local_rc=$rc wall=$(( $(date +%s) - T0 ))" >> "$O/$3.log"
  git apply -R "specs-fxr5/mutants/$1.patch" || echo "$1: REVERT FAILED"
  after=$(git diff | md5)
  echo "$1 reverted md5: $after $([ "$after" = "$base" ] && echo EQUAL || echo DIFFERS)" | tee -a "$O/b3-md5.log"
  echo "$1 rc=$rc"
}
run m3-gdn-carry gate-gpu-qwen4exp-e2e b3-m3-e2e
run m4-eh-halves gate-gpu-qwen4exp-mtp b3-m4-mtp
./tools/box.sh 'true' > "$O/b3-resync.log" 2>&1; echo "resync rc=$?"
just box-gc > "$O/b3-gc-end.log" 2>&1; echo "box-gc end rc=$?"
