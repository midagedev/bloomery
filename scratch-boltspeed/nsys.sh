#!/usr/bin/env bash
# Bolt speed nsys sitting (perf, not committed): ours decode d6 and llama.cpp pp512, under lease-hold.
set -euo pipefail
O=$HOME/boltspeed; mkdir -p $O
M=/models/nemotron-3.5-bolt/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
nsys profile -t cuda --cuda-graph-trace=node -f true -o $O/ours-d6 \
  env -u BLOOMERY_REF_MODEL BLOOMERY_REF_MODEL=$M target/release/generate_qwen3moe --tokens 100000,1000,2000,3000,4000,5000 -n 64 --ctx 256 --time > $O/ours-d6.gen.log 2>&1
nsys export --type sqlite -f true -o $O/ours-d6.sqlite $O/ours-d6.nsys-rep >/dev/null
python3 tools/ref/nsys-table.py $O/ours-d6.sqlite --sort sum > $O/ours-d6.table.txt
nsys profile -t cuda --cuda-graph-trace=node -f true -o $O/lcpp-pp512 \
  /home/user/llama.cpp-mainline-bolt/build-cuda/bin/llama-bench -m $M -p 512 -n 0 -r 1 -ngl 99 -fa on > $O/lcpp-pp512.bench.log 2>&1
nsys export --type sqlite -f true -o $O/lcpp-pp512.sqlite $O/lcpp-pp512.nsys-rep >/dev/null
python3 tools/ref/nsys-table.py $O/lcpp-pp512.sqlite --steps 1 --sort sum > $O/lcpp-pp512.table.txt
grep -E "time (prompt|step)|load|capture" $O/ours-d6.gen.log | tail -5
head -40 $O/ours-d6.table.txt; echo ====; head -40 $O/lcpp-pp512.table.txt; tail -3 $O/lcpp-pp512.bench.log
