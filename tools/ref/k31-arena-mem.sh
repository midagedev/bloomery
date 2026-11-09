#!/usr/bin/env bash
# docs/cards/k31-arena-mem.card: one fio cell (16 jobs x qd1, 2 MiB random O_DIRECT reads), then the warm-arena request under
# BLOOMERY_NVTIER_MEM=pinned and =anon. Inside a lease-hold.
set -u
F0=/models/Kimi-K3.1-Pro-IQ1_S/Kimi-K3.1-Pro-IQ1_S-00001-of-00031.gguf
fio --name=nvme0-16xqd1 --filename="$F0" --rw=randread --direct=1 --ioengine=libaio --bs=2m --iodepth=1 --numjobs=16 --group_reporting \
    --time_based --runtime=20 --readonly --norandommap --randrepeat=0 --output-format=json 2>/dev/null \
| python3 -I -c "
import json,sys
j=json.load(sys.stdin)['jobs'][0]['read']; c=j['clat_ns']; p=c.get('percentile',{})
print('fio 16xqd1 2m nvme0: MB/s=%.0f clat_mean=%.0fus p99=%.0fus' % (j['bw_bytes']/1e6, c['mean']/1e3, p.get('99.000000',0)/1e3))
"
for kind in pinned anon; do
  O=/root/k31-arena-mem-$kind
  K31_PROBE_N=512 K31_PROBE_ENV="BLOOMERY_HOST_ROOM=232G BLOOMERY_NVTIER_MEM=$kind" K31_PROBE_OUT=$O bash tools/ref/k31-nvme-probe.sh > $O.log 2>&1
  echo "== $kind rc=$?"; grep -h "^plan" $O/summary.txt | grep -o "arena=[0-9]*"; grep -h "tg_tok_s\|proc_io\|^decode" $O/summary.txt
  python3 -I /root/iostat-quarters.py $O/iostat.log "$(grep -o 'request_start=.*' $O/summary.txt | cut -d= -f2)" \
    "$(grep -o '"prompt_ms": [0-9.]*' $O/summary.txt | cut -d' ' -f2)" "$(grep -o '"predicted_ms": [0-9.]*' $O/summary.txt | cut -d' ' -f2)" 2>&1 | sed 's/at 256 positions/at 128 positions/'
done
