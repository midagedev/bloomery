# shellcheck shell=bash
# llama.cpp's own placement for the depth runners' mainline arms (depth-ds41.sh, depth-qwen3moe.sh):
# llama-bench's fit in place of the profile's hand-set -ngl / --n-cpu-moe. Sourced after the profile;
# it defines functions and one default, and exports nothing.
#
# The fit arms: lcppfit:<D> (decode, the lcpp:<D> arm's command line) and lcppppfit[<U>]:<P> (prefill,
# the lcpppp[<U>]:<P> arm's), each at LCPP_GPU_FLAGS with every placement option removed and
# `-fitt LCPP_FIT_TARGET -v` added.
#
# What llama-bench does with it (tools/llama-bench/llama-bench.cpp, mainline 53ed051ce and the V4.1
# branch 5210c7c5e alike; line numbers mainline's, the branch's are 4 lower, its usage text's 1):
#   -fitt, --fit-target <MiB>  the free memory to leave per device (1049). Default 0, and 0 is off:
#                  the fit runs only when fitt or fitc differs from the default 0 (2290-2291).
#   -fitc, --fit-ctx <n>       the smallest context the fit may shrink to (1058); llama-bench always
#                  passes n_ctx = n_prompt + n_gen + n_depth (2313), and common_fit_params changes
#                  the context only when it is 0 (common/fit.cpp:198, 455), so here it only turns the
#                  fit on. Not passed.
#   The fit replaces the placement: it sets n_gpu_layers to the default (-1), tensor_split to a
#   zeroed buffer and the buffer-type overrides to its own list before it runs (2304-2307), so -ngl,
#   --n-cpu-moe, -ts and -ot on the command line are dropped without a word; every context option
#   (-fa, -nopo, -ub, -b, -ctk) stays and is part of what the fit measures. llama-bench ignores the
#   fit's status (2315): on a failure it loads at n_gpu_layers -1 with no overrides, and the only trace
#   is common_fit_params' warning on stderr (common/fit.cpp:895, 898; log.cpp:113-116).
#   What it chose is printed nowhere but in the loader's lines under -v: the fit's own trace is
#   LOG_TRC, level 4, above the verbosity llama-bench leaves (3, LOG_DEFAULT_LLAMA), the table's ngl
#   column is the command line's, and tools/fit-params (llama-fit-params) is built in neither tree.
#   Without -v llama-bench silences every log (2246-2247).
#   On one card the fit keeps every layer's dense part on the card, then gives whole layers their
#   experts front to back (common/fit.cpp:750), then tries part of one more (the up, then the gate
#   projection: 826-856); the remaining layers' experts go to the host in the CPU buffer type
#   (657, 577). So its host experts are the last layers', where --n-cpu-moe K's are the first K.
# LCPP_FIT_TARGET is 1024, the margin llama.cpp's `-fit on` leaves in llama-cli and llama-server
# (common/common.h:481, fit_params_target); a caller's value is kept, as for the profile's flags.
#
# The row. -v prints every model load's lines: the fit's measuring loads (no_alloc) first, then the
# real one. lcpp_fit_col reads the last load's `offloaded` line, its `model buffer size` lines and its
# `buffer type overridden` lines into the row's `fit` column; fewer than two loads (the fit never ran)
# or common_fit_params' failure warning fails the arm by name.
# The cost of -v inside the timed window [derived, not measured]: ggml-cuda prints `CUDA Graph id N
# reused` once per CUDA graph compute (ggml-cuda.cu:2607), one line a step with the model on the card,
# at most one per CUDA split a step on V4.1 (about 2 a layer, 80 a step, with experts on the host), each
# an unbuffered write to the runner's pipe of a few µs: under 0.1 % of a Qwen3 step (~5 ms) and under
# 0.4 % of a V4.1 step (~100 ms). The other debug lines of the decode path are load-time or commented
# out (src/llama-context.cpp:1406); the scheduler's per-node dump needs GGML_SCHED_DEBUG.
: "${LCPP_FIT_TARGET:=1024}"

# lcpp_fit_eng <engine>: true for a fit arm's engine: lcppfit, lcppppfit, lcppppfit<U>.
lcpp_fit_eng() { case $1 in lcppfit | lcppppfit | lcppppfit[1-9]*) return 0 ;; *) return 1 ;; esac; }

# lcpp_fit_flags <flags>: the fit arm's flags, into FIT_FLAGS: <flags> without -ngl, --n-gpu-layers,
# -ncmoe, --n-cpu-moe, -ts, --tensor-split, -ot, --override-tensor and their values, then -fitt
# LCPP_FIT_TARGET -v; the removed words into FIT_DROPPED. llama-bench reads a `--` option with `_` as
# `-` (parse_cmd_params), so both spellings are removed. Flags that already carry a fit option are
# refused (1, FIT_WHY): a second value would be a second test in one process.
lcpp_fit_flags() {
  local -a words
  local i w n
  FIT_FLAGS='' FIT_DROPPED='' FIT_WHY=''
  read -r -a words <<< "$1"
  i=0
  while [ "$i" -lt ${#words[@]} ]; do
    w=${words[$i]}
    n=$w
    case $n in --*) n=$(echo "$n" | tr _ -) ;; esac
    case $n in
      -fitt | --fit-target | -fitc | --fit-ctx)
        FIT_WHY="the profile's flags already carry $w ($1); the fit arm would add a second value"
        return 1
        ;;
      -ngl | --n-gpu-layers | -ncmoe | --n-cpu-moe | -ts | --tensor-split | -ot | --override-tensor)
        FIT_DROPPED="${FIT_DROPPED:+$FIT_DROPPED }$w ${words[$((i + 1))]:-}"
        i=$((i + 2))
        continue
        ;;
    esac
    FIT_FLAGS="${FIT_FLAGS:+$FIT_FLAGS }$w"
    i=$((i + 1))
  done
  FIT_FLAGS="${FIT_FLAGS:+$FIT_FLAGS }-fitt $LCPP_FIT_TARGET -v"
}

# lcpp_fit_probe <llama-bench>: true when the binary's own --help lists --fit-target; else 1 and
# FIT_WHY. The binary loads its backends before it parses (llama-bench.cpp:2232), so it runs with
# CUDA_VISIBLE_DEVICES=-1: ggml_cuda_init returns on the failed device count (ggml-cuda.cu:221-228)
# and no card is touched.
lcpp_fit_probe() {
  local out rc=0
  out=$(CUDA_VISIBLE_DEVICES=-1 timeout --kill-after=10 60 "$1" --help 2> /dev/null) || rc=$?
  if [ "$rc" != 0 ]; then
    FIT_WHY="$1 --help exited $rc"
    return 1
  fi
  case $out in *--fit-target*) return 0 ;; esac
  FIT_WHY="$1 --help lists no -fitt/--fit-target: this llama-bench has no fit"
  return 1
}

# lcpp_fit_col <output>: a fit arm's llama-bench output (its stderr) into FIT_COL, ` | fit <summary>`,
# and FIT_LINES, the last load's offloaded and buffer lines; 1 and FIT_WHY when the fit failed or did
# not run. The summary: `offloaded N/M, overridden <buft>:<count>… in blk <lo>-<hi> (blk <lo>: <k>),
# buffers <name> <MiB>… MiB` (`overridden none` without overrides; a buffer that appears in several
# lines, one per shard, is summed).
lcpp_fit_col() {
  local sum loads line
  FIT_COL='' FIT_LINES='' FIT_WHY=''
  line=$(grep -E 'common_fit_params: (failed to fit params|encountered an error while trying to fit params)' <<< "$1" | head -n 1)
  if [ -n "$line" ]; then
    FIT_WHY="llama-bench's fit failed, and llama-bench loads without it: $line"
    return 1
  fi
  sum=$(awk '
    /loaded meta data with / { loads++; ngl = ""; nb = 0; no = 0; lo = -1; hi = -1; lines = ""; split("", bsz); split("", bname); split("", obn); split("", ocnt); split("", perl) }
    /load_tensors: offloaded [0-9]+\/[0-9]+ layers to GPU/ { for (i = 1; i <= NF; i++) if ($i == "offloaded") ngl = $(i + 1); lines = lines $0 "\n" }
    / model buffer size = / {
      for (i = 1; i <= NF; i++) if ($i == "model" && $(i + 1) == "buffer") { b = $(i - 1); v = $(i + 4) }
      if (!(b in bsz)) { nb++; bname[nb] = b }
      bsz[b] += v; lines = lines $0 "\n"
    }
    /^tensor .* buffer type overridden to / {
      t = $NF
      if (!(t in ocnt)) { no++; obn[no] = t }
      ocnt[t]++
      if (match($2, /^blk\.[0-9]+\./)) {
        l = substr($2, 5, RLENGTH - 5) + 0
        perl[l]++
        if (lo < 0 || l < lo) lo = l
        if (hi < 0 || l > hi) hi = l
      }
    }
    END {
      if (ngl == "") { printf "%d\t\n", loads; exit }
      s = "offloaded " ngl ", overridden "
      if (no == 0) s = s "none"
      for (i = 1; i <= no; i++) s = s (i > 1 ? " " : "") obn[i] ":" ocnt[obn[i]]
      if (lo >= 0) s = s " in blk " lo "-" hi " (blk " lo ": " perl[lo] ")"
      s = s ", buffers"
      for (i = 1; i <= nb; i++) s = s " " bname[i] " " sprintf("%.2f", bsz[bname[i]])
      s = s " MiB"
      printf "%d\t%s\n%s", loads, s, lines
    }' <<< "$1")
  line=${sum%%$'\n'*}
  loads=${line%%$'\t'*}
  if [ "${loads:-0}" -lt 2 ]; then
    FIT_WHY="llama-bench's fit did not run: ${loads:-0} model load(s) in its -v output, want the fit's measuring load and the real one"
    return 1
  fi
  if [ -z "${line#*$'\t'}" ]; then
    FIT_WHY="the last model load in llama-bench's -v output has no 'offloaded N/M layers' line"
    return 1
  fi
  FIT_COL=" | fit ${line#*$'\t'}"
  FIT_LINES=$(sed 1d <<< "$sum")
}

# `bash tools/ref/lcpp-fit.sh --self-test`: the flags, the probe against stub binaries and the column
# on fixture output (just check-recipes runs it on the Mac; the runners' use of it is
# depth-ds41-stub.sh's, on the box).
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = --self-test ]; then
  fails=0
  check() {
    if [ "$2" = "$3" ]; then echo "ok $1"; else echo "FAIL $1: got [$2], want [$3]"; fails=$((fails + 1)); fi
  }
  command -v timeout > /dev/null || timeout() { shift 2; "$@"; }
  lcpp_fit_flags '-ngl 999 --n-cpu-moe 33 -fa on -t 32 -nopo 1'
  check flags-v41 "$FIT_FLAGS|$FIT_DROPPED" "-fa on -t 32 -nopo 1 -fitt 1024 -v|-ngl 999 --n-cpu-moe 33"
  lcpp_fit_flags '-ngl 99 -fa on'
  check flags-q36 "$FIT_FLAGS|$FIT_DROPPED" "-fa on -fitt 1024 -v|-ngl 99"
  lcpp_fit_flags '-ot blk\.1\.=CPU --n_cpu_moe 3 -ts 1,1 -ncmoe 2 --n-gpu-layers 9 --override-tensor x=CPU --tensor-split 3 -fa on'
  check flags-all "$FIT_FLAGS" "-fa on -fitt 1024 -v"
  lcpp_fit_flags '-ngl 99 -fitt 512' && r=0 || r=1
  check flags-refuse "$r|${FIT_WHY%%;*}" "1|the profile's flags already carry -fitt (-ngl 99 -fitt 512)"
  LCPP_FIT_TARGET=2048 lcpp_fit_flags '-fa on'
  check flags-target "$FIT_FLAGS" "-fa on -fitt 2048 -v"
  e=''
  for x in lcppfit lcppppfit lcppppfit4096 lcpp lcpppp lcpppp2048 lcpp33 lcppfitx; do lcpp_fit_eng "$x" && e+="$x "; done
  check eng "$e" "lcppfit lcppppfit lcppppfit4096 "
  t=$(mktemp -d "${TMPDIR:-/tmp}/lcpp-fit-self-test.XXXXXX")
  # shellcheck disable=SC2016 # the stub expands it when it runs
  printf '#!/usr/bin/env bash\necho "cuda=$CUDA_VISIBLE_DEVICES" > "%s/env"\necho "  -fitt, --fit-target <MiB>   fit"\n' "$t" > "$t/fit"
  printf '#!/usr/bin/env bash\necho "  -ngl, --n-gpu-layers <n>"\n' > "$t/nofit"
  printf '#!/usr/bin/env bash\nexit 3\n' > "$t/broken"
  chmod +x "$t/fit" "$t/nofit" "$t/broken"
  lcpp_fit_probe "$t/fit" && r=0 || r=1
  check probe-fit "$r $(cat "$t/env")" "0 cuda=-1"
  lcpp_fit_probe "$t/nofit" && r=0 || r=1
  check probe-nofit "$r ${FIT_WHY#"$t/nofit "}" "1 --help lists no -fitt/--fit-target: this llama-bench has no fit"
  lcpp_fit_probe "$t/broken" && r=0 || r=1
  check probe-broken "$r ${FIT_WHY#"$t/broken "}" "1 --help exited 3"
  rm -rf "$t"
  out='llama_model_loader: loaded meta data with 50 key-value pairs and 900 tensors from m.gguf (version GGUF V3 (latest))
load_tensors: offloaded 41/41 layers to GPU
load_tensors:        CUDA0 model buffer size = 90000.00 MiB
llama_model_loader: loaded meta data with 50 key-value pairs and 900 tensors from m.gguf (version GGUF V3 (latest))
tensor blk.6.ffn_gate_inp.weight (1 MiB f32) buffer type overridden to CPU
tensor blk.6.ffn_gate_exps.weight (2048 MiB q3_K) buffer type overridden to CPU
tensor blk.6.ffn_down_exps.weight (2048 MiB q4_K) buffer type overridden to CPU
tensor blk.7.ffn_up_exps.weight (2048 MiB q3_K) buffer type overridden to CPU
tensor blk.39.ffn_up_exps.weight (2048 MiB q3_K) buffer type overridden to CPU
load_tensors: offloaded 41/41 layers to GPU
load_tensors:        CUDA0 model buffer size = 41000.50 MiB
load_tensors:   CPU_Mapped model buffer size = 40000.00 MiB
load_tensors:   CPU_Mapped model buffer size = 30000.25 MiB
llama_context: n_ctx = 256'
  lcpp_fit_col "$out" && r=0 || r=1
  check col "$r|$FIT_COL" "0| | fit offloaded 41/41, overridden CPU:5 in blk 6-39 (blk 6: 3), buffers CUDA0 41000.50 CPU_Mapped 70000.25 MiB"
  check col-lines "$(echo "$FIT_LINES" | wc -l | tr -d ' ') $(echo "$FIT_LINES" | head -n 1)" "4 load_tensors: offloaded 41/41 layers to GPU"
  lcpp_fit_col "$(printf 'llama_model_loader: loaded meta data with 1 key-value pairs\nload_tensors: offloaded 49/49 layers to GPU\nllama_model_loader: loaded meta data with 1 key-value pairs\nload_tensors: offloaded 49/49 layers to GPU\nload_tensors:        CUDA0 model buffer size = 20000.00 MiB\nload_tensors:   CPU_Mapped model buffer size =   300.00 MiB\n')" && r=0 || r=1
  check col-none "$r|$FIT_COL" "0| | fit offloaded 49/49, overridden none, buffers CUDA0 20000.00 CPU_Mapped 300.00 MiB"
  lcpp_fit_col "$(echo "$out" | sed -n '4,$p')" && r=0 || r=1
  check col-one-load "$r|$FIT_WHY" "1|llama-bench's fit did not run: 1 model load(s) in its -v output, want the fit's measuring load and the real one"
  lcpp_fit_col "common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
$out" && r=0 || r=1
  check col-failed "$r|$FIT_WHY" "1|llama-bench's fit failed, and llama-bench loads without it: common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort"
  lcpp_fit_col "$(echo "$out" | sed -n '1,4p')" && r=0 || r=1
  check col-no-offload "$r|$FIT_WHY" "1|the last model load in llama-bench's -v output has no 'offloaded N/M layers' line"
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($fails failures)"
  [ "$fails" = 0 ]
  exit
fi
