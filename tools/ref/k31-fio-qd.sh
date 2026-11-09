#!/usr/bin/env bash
# fio 2 MiB random O_DIRECT reads over the K3.1 shards at queue 3/6/12, each drive alone and both together (docs/cards/k31-fio-qd.card).
# Read-only. Prints one row a point: config, drive, bandwidth MB/s, IOPS, mean latency ms.
set -u
A=/models/Kimi-K3.1-Pro-IQ1_S; B=/mirror/Kimi-K3.1-Pro-IQ1_S
files() { local d=$1; local l=""; for i in 1 2 3 4 5 6 7 8; do l="$l${l:+:}$d/Kimi-K3.1-Pro-IQ1_S-0000$i-of-00031.gguf"; done; echo "$l"; }
run() { # name depth [A] [B]
  local name=$1 qd=$2; shift 2; local jobs=""
  for d in "$@"; do jobs="$jobs --name=$(basename $d) --filename=$(files $d)"; done
  fio --readonly --rw=randread --bs=2M --direct=1 --ioengine=libaio --iodepth="$qd" --time_based --runtime=20 --group_reporting=0 \
      --output-format=json $jobs 2>/dev/null | python3 -c "
import json,sys; j=json.load(sys.stdin)
for job in j['jobs']:
    r=job['read']; print(f\"[fio] {sys.argv[1]} qd={sys.argv[2]} {job['jobname']}: {r['bw']/1024:.0f} MB/s iops={r['iops']:.0f} lat_mean={r['lat_ns']['mean']/1e6:.2f} ms\")
" "$name" "$qd"
}
echo "[fio] $(fio --version); $(date -Iseconds); max_sectors_kb nvme0=$(cat /sys/block/nvme0n1/queue/max_sectors_kb) nvme1=$(cat /sys/block/nvme1n1/queue/max_sectors_kb)"
for qd in 3 6 12; do run nvme0-alone $qd $A; run nvme1-alone $qd $B; run both $qd $A $B; done
