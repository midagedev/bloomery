cd /workspace/repo/bloomery-perf
R=/tmp/claude-1000/-workspace-repo-bloomery-perf/13de5b33-6693-464c-b35e-172a11e4e6f0/scratchpad/runs
export BLOOMERY_BOX_QUEUE=1 BLOOMERY_MODEL=nemotron_h_moe
BLOOMERY_CARD=a6000 BLOOMERY_REMOTE=~/repo/bloomery-perf-nsys2 timeout 5400 ./tools/box.sh 'bash tools/ref/lease-hold.sh --card docs/cards/bolt-g1-nsys-a6000.card -- bash scratch-boltspeed/nsys-g1.sh' > $R/nsys-g1b.log 2>&1; echo "a6000 rc=$?"
BLOOMERY_CARD=3090 BLOOMERY_REMOTE=~/repo/bloomery-perf-nsys3090 BLOOMERY_BOX_ENV='BLOOMERY_TIMING_GPU=GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1' timeout 5400 ./tools/box.sh 'bash tools/ref/lease-hold.sh --card docs/cards/bolt-g1-nsys-3090.card -- bash scratch-boltspeed/nsys-g1-3090.sh' > $R/nsys-g1-3090.log 2>&1; echo "3090 rc=$?"
