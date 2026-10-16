#!/usr/bin/env bash
# G1 prefill nsys sitting (perf, not committed): 19863679's generate_qwen3moe on a 4096-id LCG prompt, A6000.
set -euo pipefail
O=$HOME/boltspeed; mkdir -p $O
M=/models/nemotron-3.5-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
B=/home/box/repo/bloomery-perf-g1b-a6000/target/release/generate_qwen3moe
. tools/ref/lease.sh
T=$(lcg_prompt 4096)
nsys profile -t cuda --cuda-graph-trace=node -f true -o $O/g1-p4096 \
  env -u BLOOMERY_REF_MODEL BLOOMERY_REF_MODEL=$M $B --tokens "$T" -n 2 --ctx 4352 --time > $O/g1-p4096.gen.log 2>&1
nsys export --type sqlite -f true -o $O/g1-p4096.sqlite $O/g1-p4096.nsys-rep >/dev/null
python3 tools/ref/nsys-table.py $O/g1-p4096.sqlite --sort sum > $O/g1-p4096.table.txt
grep -E "time (prompt|step)|load|pass" $O/g1-p4096.gen.log | tail -6
head -45 $O/g1-p4096.table.txt
