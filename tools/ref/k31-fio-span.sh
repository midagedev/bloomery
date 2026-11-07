#!/usr/bin/env bash
# docs/cards/k31-fio-span.card: 1 MiB random O_DIRECT reads over all 31 shards of each drive at queue 4 and 8, 20 s a cell.
set -u
F0=$(ls /models/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-000*-of-00031.gguf | paste -sd:)
F1=$(ls /mirror/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-000*-of-00031.gguf | paste -sd:)
run() { local name=$1 f=$2 qd=$3
  fio --name="$name" --filename="$f" --rw=randread --direct=1 --ioengine=libaio --bs=1m --iodepth="$qd" --numjobs=1 \
      --time_based --runtime=20 --readonly --norandommap --randrepeat=0 --file_service_type=random --output-format=json 2>/dev/null \
  | python3 -I -c "
import json,sys
j=json.load(sys.stdin)['jobs'][0]['read']; c=j['clat_ns']; p=c.get('percentile',{})
print('%-14s bs=1m qd=%s  MB/s=%6.0f  clat_mean=%7.0fus  p99=%7.0fus' % ('$name', '$qd', j['bw_bytes']/1e6, c['mean']/1e3, p.get('99.000000',0)/1e3))
"
}
echo "files nvme0=$(echo "$F0" | tr ':' '\n' | wc -l) nvme1=$(echo "$F1" | tr ':' '\n' | wc -l)"
for qd in 4 8; do run nvme0-31shards "$F0" $qd; run nvme1-31shards "$F1" $qd; done
