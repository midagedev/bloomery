cd /workspace/repo/bloomery-perf-iq2opt
R=/tmp/claude-1000/-workspace-repo-bloomery-perf/13de5b33-6693-464c-b35e-172a11e4e6f0/scratchpad/runs
export BLOOMERY_BOX_QUEUE=1 BLOOMERY_REMOTE=~/repo/bloomery-perf-ncmoe BLOOMERY_CARD=3090 BLOOMERY_MODEL=nemotron_h_moe
export BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/bolt-lcpp-ncmoe2-3090.card BLOOMERY_TIMING_GPU=GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1 BLOOMERY_AB_ROUNDS=2'
for K in 8 24 40 48; do
  timeout 1800 ./tools/box.sh "export LCPP_GPU_FLAGS='-ngl 99 -fa on --n-cpu-moe $K' && bash tools/ref/depth-qwen3moe.sh lcpp:6 lcpppp:512" > $R/ncmoe$K-3090.log 2>&1; echo "ncmoe$K rc=$?"
done
