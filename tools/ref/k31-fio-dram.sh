#!/usr/bin/env bash
# docs/cards/k31-fio-dram.card. The dd pids are captured at start and killed by pid after.
set -u
F0=$(ls /models/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-000*-of-00031.gguf | paste -sd:)
F1=$(ls /mirror/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-000*-of-00031.gguf | paste -sd:)
run() { local name=$1 f=$2 eng=$3 bs=$4 qd=$5 jobs=$6
  fio --name="$name" --filename="$f" --rw=randread --direct=1 --ioengine="$eng" --bs="$bs" --blockalign=4k --iodepth="$qd" --numjobs="$jobs" --group_reporting \
      --time_based --runtime=20 --readonly --norandommap --randrepeat=0 --file_service_type=random --output-format=json 2>/dev/null \
  | python3 -I -c "
import json,sys
j=json.load(sys.stdin)['jobs'][0]['read']; c=j['clat_ns']; p=c.get('percentile',{})
print('%-26s %s bs=%s qd=%s jobs=%s  MB/s=%6.0f  clat_mean=%7.0fus  p99=%7.0fus' % ('$name', '$eng', '$bs', '$qd', '$jobs', j['bw_bytes']/1e6, c['mean']/1e3, p.get('99.000000',0)/1e3))
"
}
run nvme0-psync16 "$F0" psync 2m 1 16
run nvme1-psync16 "$F1" psync 2m 1 16
run nvme0-alone "$F0" libaio 1m 4 1
PIDS=()
for i in $(seq 1 32); do dd if=/dev/zero of=/dev/null bs=256M status=none & PIDS+=($!); done
sleep 3
run nvme0-under-32dd "$F0" libaio 1m 4 1
run nvme0-psync16-under-32dd "$F0" psync 2m 1 16
kill "${PIDS[@]}" 2>/dev/null; wait 2>/dev/null; echo "dd killed: ${#PIDS[@]}"
