#!/usr/bin/env bash
# inkling2-kld.sh — KL divergence of Inkling 2 Mini files against one base file, on the family's oracle tree
# (llama.cpp PR #32695, tools/ref/models/inkling2.sh), CPU only. A development instrument, not a gate and not a
# timing runner: it takes no lease and prints no number meant as a speed.
#
#   BLOOMERY_MODEL=inkling2 tools/box.sh 'bash tools/ref/inkling2-kld.sh base  <base.gguf>  <tag>'
#   BLOOMERY_MODEL=inkling2 tools/box.sh 'bash tools/ref/inkling2-kld.sh score <cand.gguf>  <tag> <base tag>'
#
# Two trees, one logits file: ggml-org's BF16 is arch `inkling2_loop`, which only llama.cpp master reads, and the UD files
# are arch `inkling2`, which only the oracle tree (PR #32695 with the scale patch) reads. `base` of the BF16 file runs with
# INK2_PPL_BIN set to a master build's llama-perplexity (CPU only: -DGGML_CUDA=OFF), `score` of a UD file with the default
# (the oracle tree's); the base file is the same bytes to both.
#
# `base` writes the logits of the base file over the fixed text set, one llama-perplexity run a domain, into
# $INK2_KLD_DIR/<tag>.<domain>.kld; `score` runs <cand.gguf> over the same chunks against that base (llama.cpp's
# --kl-divergence) and prints one summary line a domain and a pooled one:
#
#   kld tag=<tag> base=<base tag> domain=<d> chunks=<n> positions=<p> mean_kld=<v> err=<±> same_top=<%> ppl_ratio=<v>
#
# The text set (domains, in the order they run): en = wikitext-2 raw test (the head of /home/user/eval/wiki.test.raw),
# ko = a Korean wiki sample (the lead's eval-ko.txt, built by evaltext/build_eval.py from articles the calibration
# did not use), code = Rust sources (eval-code.txt). `INK2_KLD_CHUNKS` (default 12) chunks of `INK2_KLD_CTX`
# (default 512) tokens each, scored on the second half of every chunk (255 positions a chunk), so 3,060 positions a
# domain at the defaults. A base file is 2 * n_vocab bytes a position, 1.2 GB a domain at those.
#
# Every run is CPU only (CUDA_VISIBLE_DEVICES hidden, -ngl 0, INK2_KLD_THREADS threads, default 32) and bounded
# (bounded_bg, tools/gate-bound.sh: INK2_KLD_BOUND seconds a run, default 1500); a run that fails leaves its log as
# <tag>.<domain>.log and the script ends nonzero. The pid of the run in progress is in $INK2_KLD_DIR/<tag>.pid.
set -euo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
# shellcheck source=tools/gate-bound.sh
source "$HERE/../gate-bound.sh" || exit 70

mode=${1:?usage: inkling2-kld.sh base|score <gguf> <tag> [<base tag>]}
model=${2:?missing the gguf}
tag=${3:?missing the tag}
base_tag=${4:-}

dir=${INK2_KLD_DIR:-/root/bloomery-data/ink2q/kld}
bin=${INK2_PPL_BIN:-/root/bloomery-data/ink2q/lcpp/bin/llama-perplexity}
chunks=${INK2_KLD_CHUNKS:-12}
ctx=${INK2_KLD_CTX:-512}
threads=${INK2_KLD_THREADS:-32}
bound=${INK2_KLD_BOUND:-1500}
domains=${INK2_KLD_DOMAINS:-"en ko code"}

declare -A text=(
  [en]=/home/user/eval/wiki.test.raw
  [ko]=/root/bloomery-data/evaltext/eval-ko.txt
  [code]=/root/bloomery-data/evaltext/eval-code.txt
)

[ -x "$bin" ] || { echo "inkling2-kld: $bin is not built (cmake --target llama-perplexity in the oracle tree's source)" >&2; exit 66; }
[ -f "$model" ] || { echo "inkling2-kld: no such file: $model" >&2; exit 66; }
case $mode in
  base) ;;
  score) [ -n "$base_tag" ] || { echo "inkling2-kld: score needs the base's tag" >&2; exit 64; } ;;
  *) echo "inkling2-kld: mode is base or score" >&2; exit 64 ;;
esac
mkdir -p "$dir"
export CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH="$(dirname "$bin")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for d in $domains; do
  src=${text[$d]:?unknown domain $d}
  [ -f "$src" ] || { echo "inkling2-kld: missing text $src" >&2; exit 66; }
  log=$dir/$tag.$d.log
  common=(-m "$model" -f "$src" -c "$ctx" -b "$ctx" -ub "$ctx" --chunks "$chunks" -t "$threads" -ngl 0 --no-warmup)
  if [ "$mode" = base ]; then
    out=$dir/$tag.$d.kld
    [ ! -e "$out" ] || { echo "inkling2-kld: $out exists" >&2; exit 64; }
    BG_OUT=$log bounded_bg "$bound" "$bin" "${common[@]}" --kl-divergence-base "$out.part"
    echo "$BG_PID" >"$dir/$tag.pid"
    bounded_bg_wait || { echo "inkling2-kld: $d base run failed (rc $?), see $log" >&2; exit 1; }
    mv "$out.part" "$out"
    grep -a "Final estimate" "$log" | sed "s/^/ppl tag=$tag domain=$d /" || true
  else
    ref=$dir/$base_tag.$d.kld
    [ -f "$ref" ] || { echo "inkling2-kld: no base $ref" >&2; exit 66; }
    BG_OUT=$log bounded_bg "$bound" "$bin" "${common[@]}" --kl-divergence-base "$ref" --kl-divergence
    echo "$BG_PID" >"$dir/$tag.pid"
    bounded_bg_wait || { echo "inkling2-kld: $d score run failed (rc $?), see $log" >&2; exit 1; }
    python3 - "$log" "$tag" "$base_tag" "$d" "$chunks" "$ctx" <<'PY'
import re, sys
log, tag, base, dom, chunks, ctx = sys.argv[1:7]
t = open(log, errors="replace").read()
def num(p, g=1):
    m = re.search(p, t)
    return m.group(g) if m else "NA"
pos = int(chunks) * (int(ctx) - 1 - int(ctx) // 2)
print(f"kld tag={tag} base={base} domain={dom} chunks={chunks} positions={pos} "
      f"mean_kld={num(r'Mean\s+KLD:\s+([-\d.]+)')} err={num(r'Mean\s+KLD:\s+[-\d.]+\s+±\s+([\d.]+)')} "
      f"median_kld={num(r'Median\s+KLD:\s+([-\d.]+)')} p99_kld={num(r'99\.0%\s+KLD:\s+([-\d.]+)')} "
      f"same_top={num(r'Same top p:\s+([\d.]+)')} ppl_q={num(r'Mean PPL\(Q\)\s+:\s+([\d.]+)')} "
      f"ppl_base={num(r'Mean PPL\(base\)\s+:\s+([\d.]+)')}")
PY
  fi
done
