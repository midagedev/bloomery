#!/usr/bin/env bash
# docs/cards/k31-fio-align.card: the span cells with 4 KiB-aligned random offsets.
set -u
F0=$(ls /models/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-000*-of-00031.gguf | paste -sd:)
F1=$(ls /mirror/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-000*-of-00031.gguf | paste -sd:)
run() { local name=$1 f=$2 bs=$3 qd=$4
  fio --name="$name" --filename="$f" --rw=randread --direct=1 --ioengine=libaio --bs="$bs" --blockalign=4k --iodepth="$qd" --numjobs=1 \
      --time_based --runtime=20 --readonly --norandommap --randrepeat=0 --file_service_type=random --output-format=json 2>/dev/null \
  | python3 -I -c "
import json,sys
j=json.load(sys.stdin)['jobs'][0]['read']; c=j['clat_ns']; p=c.get('percentile',{})
print('%-18s bs=%s qd=%s  MB/s=%6.0f  clat_mean=%7.0fus  p99=%7.0fus' % ('$name', '$bs', '$qd', j['bw_bytes']/1e6, c['mean']/1e3, p.get('99.000000',0)/1e3))
"
}
for qd in 4 8; do run nvme0-align4k "$F0" 1m $qd; run nvme1-align4k "$F1" 1m $qd; done
run nvme0-align4k "$F0" 2m 4; run nvme1-align4k "$F1" 2m 4
