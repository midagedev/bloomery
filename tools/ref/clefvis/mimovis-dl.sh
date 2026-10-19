#!/usr/bin/env bash
# Box: fetch the BF16 projector the MiMo-V2.6-Flash image-input set is dumped with, at the repository revision named here, into
# /models/mmproj, detached at idle IO priority and bounded at 1 h; then check the file's size and sha256 against the LFS oid of the
# revision's tree listing.
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'bash tools/ref/clefvis/mimovis-dl.sh'
#   BLOOMERY_BOX_READONLY=1 tools/box.sh 'cat /models/mmproj/mimovis-dl.rc; cat /models/mmproj/mimovis-dl.log'
#
# The text file in /models is the same repository's MXFP4 split set. The directory carries the organization for the reason
# qvis/mmproj-dl.sh's does: another publisher's repository of the same name may hold its own files in /models/mmproj.
set -u
D=/models/mmproj
mkdir -p "$D"
[ -f "$D/mimovis-dl.pid" ] && kill -0 "$(cat "$D/mimovis-dl.pid")" 2> /dev/null && { echo "already running pid $(cat "$D/mimovis-dl.pid")"; exit 0; }
rm -f "$D/mimovis-dl.rc"
# <repo> <revision> <file> <directory under $D> <bytes> <sha256>
SPECS=(
  "ggml-org/MiMo-V2.6-Flash-MOPD-GGUF e6d4bced739310a721eefad4c247ec7e0247c89d mmproj-MiMo-V2.6-Flash-MOPD-BF16.gguf ggml-org--MiMo-V2.6-Flash-MOPD-GGUF 2748509792 9b7d98d8afe2a6ae25635ecc490fb37883f912bf1d5648d1334fdc703e4a565e"
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
echo "$rc" > $D/mimovis-dl.rc
EOF
)
export D
nohup setsid bash -c "$script" _ "${SPECS[@]}" > "$D/mimovis-dl.log" 2>&1 < /dev/null &
echo $! > "$D/mimovis-dl.pid"
sleep 3
echo "pid $(cat "$D/mimovis-dl.pid")"
