# shellcheck shell=bash
# The stub tests' cards (depth-qwen3moe-stub.sh, depth-ds41-stub.sh, depth-glm5next-stub.sh): sourced with
# T set to the stub tree, it writes the tree's cards.sh — two made-up UUIDs, STUB_GPU_A6000 and
# STUB_GPU_3090, never the box's — and, unless STUB_CARDS_FILE_ONLY is set, three stubs into $T/bin. It
# runs nothing else.
#   nvidia-smi         the one-card queries answer as they always did in the stubs (any card's name is
#                      the A6000's, no compute process, `stub` fields); a query that names a card by -i
#                      and asks the two-card fields answers for that card: its name, power limit (the
#                      3090's STUB_3090_LIMIT, default 250.00), enforced limit and PCI address.
#                      STUB_3090_GONE=1: every query of the 3090 fails as a card off the bus does (rc 15).
#                      STUB_BUSY_ONCE=<3090|a6000>: that card's first compute-apps query in the two-card
#                      guard's form (with gpu_uuid) shows a process.
#   journalctl         the kernel journal: the Xid lines the stub llama-bench left in $TMPDIR/stub-xid,
#                      whatever the options; STUB_JOURNAL_FAIL=1 fails it.
#   stub-bench-cards   sourced by a stub llama-bench once its options are read and `label` is set: under a
#                      CUDA_VISIBLE_DEVICES of two cards, ggml_cuda_init's device lines on stderr (one
#                      device when STUB_SEE_ONE names the label); an Xid 79 line of the 3090 into
#                      $TMPDIR/stub-xid when STUB_BENCH_XID names it (the first time only). Under one card
#                      it prints nothing, so the one-card checks read what they always read.
STUB_GPU_A6000=GPU-00000000-0000-0000-0000-000000000000
STUB_GPU_3090=GPU-11111111-1111-1111-1111-111111111111
cat > "$T/tools/ref/cards.sh" << CARDSFILE
# shellcheck shell=bash
# The stub tree's cards: made-up UUIDs that the stub nvidia-smi answers to.
GPU_3090=$STUB_GPU_3090
GPU_A6000=$STUB_GPU_A6000
CARDSFILE
[ -z "${STUB_CARDS_FILE_ONLY:-}" ] || return 0
cat > "$T/bin/nvidia-smi" << 'SMI'
#!/usr/bin/env bash
id='' q=''
for ((a = 1; a <= $#; a++)); do
  case ${!a} in
    -i) b=$((a + 1)); id=${!b} ;;
    --query-gpu=*) q=${!a#--query-gpu=} ;;
    --query-compute-apps=*) q=apps ;;
  esac
done
case $id in
  GPU-11111111-*) card=3090 name='NVIDIA GeForce RTX 3090 (stub)' lim=${STUB_3090_LIMIT:-250.00} bus=00000000:41:00.0 ;;
  GPU-00000000-*) card=a6000 name='NVIDIA RTX A6000 (stub)' lim=300.00 bus=00000000:61:00.0 ;;
  *) card='' ;;
esac
if [ "$card" = 3090 ] && [ -n "${STUB_3090_GONE:-}" ]; then
  echo "Unable to determine the device handle for GPU0000:41:00.0: GPU is lost.  Reboot the system to recover this GPU" >&2
  exit 15
fi
case $q in
  apps)
    m=${TMPDIR:-/tmp}/stub-once-busy
    case $* in *gpu_uuid*) guard=1 ;; *) guard='' ;; esac
    if [ -n "$guard" ] && [ -n "$card" ] && [ "${STUB_BUSY_ONCE:-}" = "$card" ] && [ ! -e "$m" ]; then
      touch "$m"
      echo "$id, 4242, 100 MiB"
    fi
    ;;
  name) echo "NVIDIA RTX A6000 (stub)" ;;
  name,power.limit,enforced.power.limit,pci.bus_id) echo "$name, $lim, $lim, $bus" ;;
  name,power.limit,enforced.power.limit,clocks.max.sm) echo "$name, $lim W, $lim W, 2100 MHz" ;;
  *) echo "stub, stub, stub, stub, stub, stub, stub, stub" ;;
esac
SMI
cat > "$T/bin/journalctl" << 'JCTL'
#!/usr/bin/env bash
[ -z "${STUB_JOURNAL_FAIL:-}" ] || { echo "journalctl: the stub fails" >&2; exit 1; }
cat "${TMPDIR:-/tmp}/stub-xid" 2> /dev/null
exit 0
JCTL
cat > "$T/bin/stub-bench-cards" << 'CARDS'
# shellcheck shell=bash
case ${CUDA_VISIBLE_DEVICES:-} in
  *,*)
    if [ "${STUB_SEE_ONE:-}" = "$label" ]; then
      echo "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 48539 MiB):" >&2
      echo "  Device 0: NVIDIA RTX A6000 (stub), compute capability 8.6, VMM: yes, VRAM: 48539 MiB" >&2
    else
      echo "ggml_cuda_init: found 2 CUDA devices (Total VRAM: 72663 MiB):" >&2
      echo "  Device 0: NVIDIA RTX A6000 (stub), compute capability 8.6, VMM: yes, VRAM: 48539 MiB" >&2
      echo "  Device 1: NVIDIA GeForce RTX 3090 (stub), compute capability 8.6, VMM: yes, VRAM: 24124 MiB" >&2
    fi
    ;;
esac
if [ "${STUB_BENCH_XID:-}" = "$label" ] && [ ! -e "${TMPDIR:-/tmp}/stub-once-xid" ]; then
  touch "${TMPDIR:-/tmp}/stub-once-xid"
  echo "1790428806.570394 ws kernel: NVRM: Xid (PCI:0000:41:00): 79, pid='<unknown>', name=<unknown>, GPU has fallen off the bus." >> "${TMPDIR:-/tmp}/stub-xid"
fi
CARDS
chmod +x "$T/bin/nvidia-smi" "$T/bin/journalctl"
