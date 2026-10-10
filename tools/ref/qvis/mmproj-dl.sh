#!/usr/bin/env bash
# Box: fetch the two BF16 projectors the Qwen image-input sets are dumped with, each at the repository revision named
# here, into /models/mmproj, detached at idle IO priority and bounded at 1 h; then check each file's size and sha256
# against the LFS oid of the revision's tree listing.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'bash tools/ref/qvis/mmproj-dl.sh'
#   BLOOMERY_BOX_READONLY=1 tools/box.sh 'cat /models/mmproj/qvis-dl.rc; cat /models/mmproj/qvis-dl.log'
#
# The llama.cpp sibling rule (`find_best_sibling`) picks the BF16 file of each default `-hf` repository; the F16 copies
# that are already under /models/mmproj stay. The lmstudio directory carries the organization, because unsloth's
# repository of the same name holds its own files in /models/mmproj/Qwen3.6-35B-A3B-GGUF.
set -u
D=/models/mmproj
mkdir -p "$D"
[ -f "$D/qvis-dl.pid" ] && kill -0 "$(cat "$D/qvis-dl.pid")" 2> /dev/null && { echo "already running pid $(cat "$D/qvis-dl.pid")"; exit 0; }
rm -f "$D/qvis-dl.rc"
# <repo> <revision> <file> <directory under $D> <bytes> <sha256>
SPECS=(
  "lmstudio-community/Qwen3.6-35B-A3B-GGUF 68a34855558af61cbef0324d31f411be8a506b08 mmproj-Qwen3.6-35B-A3B-BF16.gguf lmstudio-community--Qwen3.6-35B-A3B-GGUF 902822016 e5c205cec2fd28f66c3895e4040021ab994b860323c3db8531640305ff49b322"
  "unsloth/Qwen3.8-Flash-Next-GGUF 766911a6b7369840a91dbcd95f9f997acaab6cd6 mmproj-BF16.gguf Qwen3.8-Flash-Next-GGUF 907542944 2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355"
)
script=$(
  cat << 'EOF'
set -o pipefail
rc=0
export HF_TOKEN_PATH=/home/user/.cache/huggingface/token
for spec in "$@"; do
  set -- $spec
  out=$D/$4
  mkdir -p "$out"
  timeout --kill-after=30 3600 ionice -c3 nice -n 19 /home/user/ft/bin/hf download "$1" "$3" --revision "$2" --local-dir "$out" || { rc=$?; echo "download of $1 $3 failed ($rc)"; continue; }
  got=$(stat -c %s "$out/$3")
  sum=$(sha256sum "$out/$3" | cut -d' ' -f1)
  if [ "$got" != "$5" ] || [ "$sum" != "$6" ]; then
    echo "MISMATCH $out/$3: $got bytes sha256 $sum, want $5 $6"
    rc=3
  else
    echo "ok $out/$3 $got bytes sha256 $sum"
  fi
done
echo "$rc" > $D/qvis-dl.rc
EOF
)
export D
nohup setsid bash -c "$script" _ "${SPECS[@]}" > "$D/qvis-dl.log" 2>&1 < /dev/null &
echo $! > "$D/qvis-dl.pid"
sleep 3
echo "pid $(cat "$D/qvis-dl.pid")"
