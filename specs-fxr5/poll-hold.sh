#!/usr/bin/env bash
# Poll the box's landing-batch GPU hold every 60 s for at most 90 min; exit 0 at the first "down".
cd "$(dirname "$0")/.." || exit 64
export BLOOMERY_REMOTE='~/repo/bloomery-fxr5'
log=specs-fxr5/out/hold-poll.log
for i in $(seq 1 90); do
  ans=$(BLOOMERY_BOX_READONLY=1 ./tools/box.sh 'test -e /root/bloomery-batch.gpuhold && echo up || echo down' 2>&1 | tail -1)
  echo "$(date -u +%FT%TZ) poll $i: $ans" >> "$log"
  [ "$ans" = down ] && { echo "DOWN at poll $i" >> "$log"; exit 0; }
  sleep 60
done
echo "STILL UP after 90 polls" >> "$log"
exit 1
