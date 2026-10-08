#!/usr/bin/env bash
# The nightly's mail step. Runs as root, as bloomery-nightly.service's `ExecStopPost=+` (the `+` keeps root although the service
# runs as bnightly), after run.sh has ended or been killed: systemd hands it $SERVICE_RESULT (success, exit-code, timeout, signal,
# …), $EXIT_CODE, $EXIT_STATUS and $INVOCATION_ID. Installed by tools/nightly/install.sh to /usr/local/lib/bloomery-nightly/mail.sh,
# root-owned: root never runs a file bnightly can write.
#
#   RED    the service did not end in `success`, or the run's status is not GREEN, or this invocation left no run dir
#          (the lock was held, the state dir is missing): mail midagedev@gmail.com, subject
#          `bloomery nightly RED <short sha> <date>`, body = summary, units table, failing tests, the last 80 lines of
#          the first failure.
#   GREEN  no mail, except one on each Monday (Asia/Seoul) — a heartbeat, so silence is never ambiguous; one a Monday, a marker
#          keeps a second run that day quiet.
#
# The mail goes out as the `openclaw` user, the only one `gog` is authorised under: `runuser -l` (the login form) is what gives
# it the keyring password its profile sets — `runuser -u` has none and gog fails at the keyring. This script never reads or copies
# that credential. Everything read from the run dir is read as bnightly (`as_b`), so a symlink planted there cannot make root
# read or write another file. Three attempts, 20 s apart.
#
# For tests: NIGHTLY_STATE moves the state dir, NIGHTLY_TODAY (YYYY-MM-DD) the day the Monday rule reads, NIGHTLY_MAIL_DRY=1 prints
# the decision and the mail instead of sending it, NIGHTLY_GOG names another sender (a failing stub proves the failure path).
# The unit sets none of them.
#
# The outcome is recorded in <run>/summary (`mail: sent id=…` or `mail: FAILED …`) and <run>/mail.json holds gog's reply. A failed
# mail ends this script nonzero, so the unit is `failed` and `systemctl status` shows it: no silent failure.
set -uo pipefail

STATE=${NIGHTLY_STATE:-/var/lib/bloomery-nightly}
TO=midagedev@gmail.com
GOG=${NIGHTLY_GOG:-/home/linuxbrew/.linuxbrew/bin/gog}
RESULT=${SERVICE_RESULT:-unknown}
as_b() { runuser -u bnightly -- "$@"; }
say() { printf 'nightly-mail: %s\n' "$*" >&2; }

today=${NIGHTLY_TODAY:-$(TZ=Asia/Seoul date +%F)}
weekday=$(TZ=Asia/Seoul date -d "$today" +%u) # 1 = Monday

# the run this invocation made, if any
RUN=""
cur=$(as_b cat "$STATE/current" 2> /dev/null | head -c 64)
if [[ $cur =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}(-[0-9]+)?$ ]] && [ "$(as_b cat "$STATE/$cur/invocation" 2> /dev/null)" = "${INVOCATION_ID:-none}" ]; then
  RUN=$STATE/$cur
fi
status=NONE
sha=unknown
date_of=$today
if [ -n "$RUN" ]; then
  status=$(as_b cat "$RUN/status" 2> /dev/null | head -c 16)
  c=$(as_b cat "$RUN/commit" 2> /dev/null | head -c 32)
  [[ $c =~ ^[0-9a-f]{8}(\+local)?$ ]] && sha=$c
  date_of=${cur:0:10}
fi

verdict=RED
if [ "$RESULT" = success ] && [ "$status" = GREEN ]; then verdict=GREEN; fi

if [ "$verdict" = GREEN ]; then
  marker=$STATE/heartbeat-$today
  if [ "$weekday" != 1 ] || [ -e "$marker" ]; then
    say "green, no mail ($sha $date_of; weekday $weekday)"
    exit 0
  fi
  subject="bloomery nightly GREEN $sha $date_of (Monday heartbeat)"
else
  subject="bloomery nightly RED $sha $date_of"
fi

body() {
  echo "systemd: result=$RESULT exit_code=${EXIT_CODE:-?} exit_status=${EXIT_STATUS:-?}  run=${cur:-none} status=$status"
  if [ -n "$RUN" ]; then
    echo
    as_b head -c 4000 "$RUN/summary"
    echo
    echo "units (unit, kind, rc, passed, failed, ignored, seconds):"
    as_b head -c 6000 "$RUN/units.tsv" 2> /dev/null
    if as_b test -s "$RUN/failed"; then
      echo
      echo "failing:"
      as_b head -c 6000 "$RUN/failed"
    fi
    if as_b test -s "$RUN/first-failure.tail"; then
      echo
      echo "last 80 lines of the first failure:"
      as_b head -c 24000 "$RUN/first-failure.tail"
    fi
    echo
    echo "full log: $RUN/log on the VPS"
  else
    echo
    echo "this invocation left no run dir: run.sh did not start its units (lock held, state dir missing) or died before it wrote one."
    echo "journal: journalctl -u bloomery-nightly.service -n 80"
  fi
}

record() { # record <line>: append to the run's summary as bnightly
  if [ -n "$RUN" ]; then as_b bash -c 'printf "%s\n" "$1" >> "$2/summary"' _ "$1" "$RUN"; fi
}

if [ -n "${NIGHTLY_MAIL_DRY:-}" ]; then
  echo "dry run, would send: $subject"
  body
  exit 0
fi

send="$GOG gmail send -a $TO --to $TO --subject $(printf %q "$subject") --body-file - -j --no-input"
reply=""
rc=1
for attempt in 1 2 3; do
  reply=$(body | runuser -l openclaw -c "$send" 2>&1)
  rc=$?
  [ "$rc" = 0 ] && break
  say "attempt $attempt failed (rc $rc): ${reply:0:300}"
  [ "$attempt" = 3 ] || sleep 20
done
[ -z "$RUN" ] || printf '%s\n' "$reply" | as_b tee "$RUN/mail.json" > /dev/null

id=""
if [ "$rc" = 0 ]; then
  id=$(printf '%s' "$reply" | python3 -c '
import json, sys
def find(o):
    if isinstance(o, dict):
        for k in ("messageId", "message_id", "id"):
            if isinstance(o.get(k), str) and o[k]:
                return o[k]
        for v in o.values():
            r = find(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v)
            if r:
                return r
    return ""
text = sys.stdin.read()
try:
    print(find(json.loads(text[text.find("{"):])))
except ValueError:
    print("")
')
fi
if [ "$rc" = 0 ] && [ -n "$id" ]; then
  record "mail: sent $verdict id=$id subject=\"$subject\""
  say "sent $verdict id=$id"
  if [ "$verdict" = GREEN ]; then as_b touch "$STATE/heartbeat-$today"; fi
  exit 0
fi
record "mail: FAILED rc=$rc id=${id:-none} subject=\"$subject\" reply=${reply:0:200}"
say "FAILED rc=$rc id=${id:-none}: ${reply:0:300}"
exit 1
