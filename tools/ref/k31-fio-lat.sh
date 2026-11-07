#!/usr/bin/env bash
# Random O_DIRECT read latency of the two drives at 512 KiB / 1 MiB / 2 MiB and queue depth 1 / 4 / 8, fio on the first K3.1
# shard of each drive (read-only), 20 s a cell; plus a 2 MiB queue-4 cell with hugepage buffers when huge pages are free.
set -u
F0=/models/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-00001-of-00031.gguf
F1=/mirror/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-00001-of-00031.gguf
run() { # name file bs qd [extra]
  local name=$1 f=$2 bs=$3 qd=$4; shift 4
  fio --name="$name" --filename="$f" --rw=randread --direct=1 --ioengine=libaio --bs="$bs" --iodepth="$qd" --numjobs=1 \
      --time_based --runtime=20 --readonly --norandommap --randrepeat=0 --output-format=json "$@" 2>/dev/null \
  | python3 -I -c "
import json,sys
j=json.load(sys.stdin)['jobs'][0]['read']
c=j['clat_ns']; p=c.get('percentile',{})
print('%-22s bs=%-5s qd=%s  MB/s=%6.0f  clat_mean=%7.0fus  p99=%7.0fus' % ('$name', '$bs', '$qd', j['bw_bytes']/1e6, c['mean']/1e3, p.get('99.000000',0)/1e3))
"
}
echo "max_sectors_kb nvme0=$(cat /sys/block/nvme0n1/queue/max_sectors_kb) nvme1=$(cat /sys/block/nvme1n1/queue/max_sectors_kb); HugePages_Free=$(grep HugePages_Free /proc/meminfo | awk '{print $2}')"
for bs in 512k 1m 2m; do for qd in 1 4 8; do
  run nvme0-malloc "$F0" $bs $qd
  run nvme1-malloc "$F1" $bs $qd
done; done
if [ "$(grep HugePages_Free /proc/meminfo | awk '{print $2}')" -ge 8 ]; then
  run nvme0-mmaphuge "$F0" 2m 4 --iomem=mmaphuge
  run nvme1-mmaphuge "$F1" 2m 4 --iomem=mmaphuge
else
  echo "hugepage cell skipped: no free huge pages"
fi
