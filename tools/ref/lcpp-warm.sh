# shellcheck shell=bash
# The warm llama.cpp arms: llama-server loaded once, fed the token ids our arm feeds, once as a discarded
# warm-up request and then as the timed request. Sourced (depth-ds41.sh, depth-qwen3moe.sh; the GLM runner
# can take the same functions); it defines functions and one default, and runs nothing.
#
# Why a server and not llama-bench. llama-bench draws new std::rand() ids for its tests and has no option
# for a fixed prompt (-p N, -r, --no-warmup only), so on a model whose rows are read by id — V4.1's lazy
# engram tables, Qwen3.8's per-layer embeddings — every bench row meets rows it has not read before, each
# through a page fault inside its clock. llama-server's /completion takes the prompt as a token array: the
# same ids twice, the first request discarded, the second timed with its rows in the page cache.
#
# The functions, their inputs and outputs (every output a global, SRV_WHY set on a failure):
#   lcpp_srv_bin                   the llama-server of the profile's tree: LCPPSRV, else the llama-server
#                                  beside LCPPBIN (the bench and the server of one build link one libllama)
#   lcpp_srv_flags <flags> [fit]   the profile's llama-bench flags in llama-server's spellings, into
#                                  SRV_FLAGS; 1 and SRV_WHY on a flag the table below does not know. With
#                                  `fit`: the fit twin's (lcpp-fit.sh's lcpp_fit_flags first: the placement
#                                  options dropped, then -fitt T -v, here `-fit on -fitt T -v`); without it
#                                  `-fit off`, the hand-set placement kept as given
#   lcpp_srv_ctx <prompt ids> <n_predict>  the -c: prompt + n_predict + 1 rounded up to 256
#   lcpp_srv_cmd <server> <model> <ctx> <flag words...>  into SRV_CMD, the server's whole command line:
#                                  the flags, then LCPP_SRV_FIXED, -c, --host 127.0.0.1 --port 0
#   lcpp_srv_probe <server>        true when <server> --help (no card: CUDA_VISIBLE_DEVICES=-1) lists every
#                                  option word of SRV_CMD; else 1 and SRV_WHY
#   lcpp_srv_start <bound> <log> [NAME=VALUE...]  starts `timeout --kill-after=10 <bound> env <NAME=VALUE...>
#                                  SRV_CMD` with its stdout and stderr in <log>, and waits (at most <bound> s)
#                                  for its `listening on http://127.0.0.1:<port>` line — printed after the
#                                  model loaded — and a 200 from /health. SRV_PID (the timeout's pid, the only
#                                  one signalled), SRV_PORT; 1 with SRV_WHY and SRV_RC when it exited first
#                                  (its rc) or never listened (124)
#   lcpp_srv_request <ids> <n_predict> <bound>  one POST /completion: the comma-separated ids as the prompt,
#                                  greedy (temperature 0, top_k 1), ignore_eos, cache_prompt off,
#                                  return_tokens; pgmajfault read just before and after (SRV_F0, SRV_F1). Into
#                                  SRV_PROMPT_N, SRV_PROMPT_MS, SRV_PROMPT_TPS, SRV_PRED_N, SRV_PRED_MS,
#                                  SRV_PRED_TPS, SRV_CACHE_N, SRV_TOKENS (the generated ids, comma-separated).
#                                  1 with SRV_WHY when the request fails, answers other than 200, has no
#                                  timings, or prompt_n is not the ids' count, cache_n is not 0, or
#                                  predicted_n is not n_predict
#   lcpp_srv_arm <ids> <n_predict> <bound>  the warm-up request, then the timed one: WARM_TPS (the warm-up's
#                                  rate, the arm's kind of rate), WARM_FAULTS (pgmajfault across it),
#                                  WARM_TOKENS, SRV_SAME (lcpp_srv_same of the warm-up's continuation and the
#                                  timed one's), and the timed request's SRV_* above. 1 with SRV_WHY (`warm-up:
#                                  …` or `timed: …`). Leaves the server up: a caller's retry is one more
#                                  lcpp_srv_request, and the caller stops it
#   lcpp_srv_same <a> <b>          `same` for two equal comma lists of ids, else `differs at <i>`: greedy
#                                  decoding repeats its continuation, so the warm-up read the timed rows
#   lcpp_srv_stop                  TERM to SRV_PID, which passes it to the server; KILL to its children and
#                                  to it 30 s later if it is still up; SRV_EXIT its status
#   lcpp_srv_build <log>, lcpp_srv_device <log>  the server's `build:` line and the card it opened
#   lcpp_srv_majflt                /proc/vmstat's pgmajfault (the stub tests redefine it)
# A caller runs: lcpp_srv_flags, lcpp_srv_cmd, (before its lease) lcpp_srv_probe, then per arm
# lcpp_srv_start, lcpp_srv_arm, [lcpp_srv_request], lcpp_srv_stop — and stops the server on every path
# after a start, the failed ones included. The depth runners' own arms are built on these by the runner
# side below (srv_eng, srv_pp, srv_ub, srv_cmd_of, srv_ids, srv_check_arm, srv_preflight, srv_tree,
# srv_dry_cmd, srv_config, srv_arm, srv_take, srv_row); the GLM runner's srv_run is the same sequence
# without the warm-up and can move onto the functions above.
#
# The rates, against llama-bench's (tools/server/server-common.h server_slot_stats, the V4.1 branch 5210c7c
# and mainline 53ed051ce alike):
#   decode  predicted_per_second = (predicted_n - 1) / (t_gen_last - t_prompt_last): the decode steps over
#           their own wall. The first generated token comes from the prompt's logits, and its sample closes
#           the prompt's clock, so it is the prompt's (n_gen_steps: "the first token is free"). llama-bench's
#           tg is N one-token decodes over their wall: the same per-step rate, over N - 1 steps here, at
#           positions D + 1 .. D + N - 1 where bench's are D .. D + N - 1. Each step here also samples (top_k
#           1 over the logits) and runs the slot loop's bookkeeping; bench feeds a random id and samples
#           nothing.
#   prompt  prompt_per_second = prompt_n / (t_prompt_last - t_start): from the slot's first prompt batch to
#           the first sample, which synchronizes the context. llama-bench's pp is P ids through llama_decode
#           in batches of -b and ubatches of -ub, then one synchronize: the same work, plus here the batch
#           assembly per ubatch and the one sample. The server's defaults are bench's (-ub 512 -b 2048).
#   Neither clock holds the HTTP exchange or the tokenization: those are before t_start and after the last
#   token. The timed window W for the cold tag is prompt_ms + predicted_ms; the fault count around the
#   request covers that window and the exchange around it (an upper bound, as the other arms' counts).
#
# The flag table (llama-bench spelling -> llama-server's, common/arg.cpp in both trees): -ngl, -ncmoe,
# --n-cpu-moe, -fa, -t, -ub, -b, -lzm, -ts, -ot, -ctk, -ctv (and their long forms) are one spelling in both;
# -nopo 1 is --no-op-offload and -nopo 0 --op-offload; -fitt T is `-fit on -fitt T` and -v is -v. A value
# with a comma in a flag that takes one value is several bench tests and refused, as is every other word.
# V4.1's translated flags plus -fit off are the profile's LCPP_CLI_FLAGS word for word (the self-test).
# Unchanged defaults both engines share: mmap, the f16 K/V cache, -tb = -t (common.cpp: n_threads_batch
# follows n_threads when unset; llama-bench sets both to -t).
# What the server needs that the bench has no word for, LCPP_SRV_FIXED and the arm's own:
#   -np 1         one slot, as the bench's one sequence (-1 is automatic: four slots and a unified cache)
#   -ctxcp 0      no context checkpoints: on a model with SWA or a recurrent cache the server copies the
#                 cache's state to the host at prompt batches (server-context.cpp do_checkpoint), inside the
#                 prompt clock; the bench does not
#   --cache-ram 0 no host prompt cache: at slot selection the server saves the slot's state to RAM and
#                 loads a cached one (prompt_save/prompt_load), outside the clock, but it is a copy of the
#                 whole state per request and memory the page cache loses
#   -fit off      llama-server fits by default (`-fit on`, 1024 MiB), moving the arguments not given; the
#                 hand-set arm keeps its placement as given, the fit twin passes -fit on explicitly
#   -c C          the prompt ids + n_predict + 1, rounded up to 256: the bench's n_ctx is D + N (P for a pp
#                 test) padded to 256 under flash attention, and the server stops a slot whose next token
#                 would not fit, so a pp arm of P = 4096 has 4352 here against the bench's 4096 (256 more
#                 positions of cache; V4.1: 40 KiB a position [derived, the profile's KV note])
#   --host 127.0.0.1 --port 0   an ephemeral port the kernel picks at bind; the server logs it
# The fit twin's column needs -v: common_fit_params demotes its measuring loads' log lines to debug
# (common/fit.cpp:49), so lcpp_fit_col counts two loads only at the debug level. -v also prints the slot's
# debug lines, a few a generated token through common_log's worker thread; their cost inside the timed
# windows is a few microseconds a token against 20-50 ms steps [derived, not measured], lcpp-fit.sh's
# estimate for the bench's -v.
#
# Stopping. The server runs under `timeout <bound>`, one bound for its load, the warm-up, the timed request
# and a retry; SRV_PID is the timeout's pid, started here. TERM goes to it (timeout passes it on); a server
# still up 30 s later gets KILL through its parent's pid and then the parent. Never by name.
: "${LCPP_SRV_FIXED:=-np 1 -ctxcp 0 --cache-ram 0}"

lcpp_srv_bin() { echo "${LCPPSRV:-${LCPPBIN%/*}/llama-server}"; }

lcpp_srv_flags() {
  local -a words
  local i w n v flags=$1
  SRV_FLAGS='' SRV_WHY=''
  if [ "${2:-}" = fit ]; then
    lcpp_fit_flags "$flags" || { SRV_WHY=$FIT_WHY; return 1; }
    flags=$FIT_FLAGS
  fi
  read -r -a words <<< "$flags"
  i=0
  while [ "$i" -lt ${#words[@]} ]; do
    w=${words[$i]} v=${words[$((i + 1))]:-}
    n=$w
    case $n in --*) n=$(echo "$n" | tr _ -) ;; esac
    case $n in
      -ngl | --n-gpu-layers | -ncmoe | --n-cpu-moe | -fa | --flash-attn | -t | --threads | -ub | --ubatch-size | -b | --batch-size | -lzm | --lazy-mode | -ctk | --cache-type-k | -ctv | --cache-type-v)
        case $v in '' | -* | *,*) SRV_WHY="$w '$v': one value, not a list of bench tests ($1)"; return 1 ;; esac
        SRV_FLAGS+=" $w $v"
        i=$((i + 2))
        ;;
      -ts | --tensor-split | -ot | --override-tensor)
        [ -n "$v" ] || { SRV_WHY="$w has no value ($1)"; return 1; }
        SRV_FLAGS+=" $w $v"
        i=$((i + 2))
        ;;
      -nopo | --no-op-offload)
        case $v in
          1) SRV_FLAGS+=" --no-op-offload" ;;
          0) SRV_FLAGS+=" --op-offload" ;;
          *) SRV_WHY="$w '$v': 0 or 1 ($1)"; return 1 ;;
        esac
        i=$((i + 2))
        ;;
      -fitt | --fit-target)
        case $v in '' | *[!0-9]*) SRV_WHY="$w '$v': one MiB value ($1)"; return 1 ;; esac
        SRV_FLAGS+=" -fit on -fitt $v"
        i=$((i + 2))
        ;;
      -v | --verbose)
        SRV_FLAGS+=" -v"
        i=$((i + 1))
        ;;
      *)
        SRV_WHY="'$w' has no llama-server spelling in lcpp-warm.sh's table ($1)"
        return 1
        ;;
    esac
  done
  [ "${2:-}" = fit ] || SRV_FLAGS+=" -fit off"
  SRV_FLAGS=${SRV_FLAGS# }
}

lcpp_srv_ctx() { echo $((($1 + $2 + 1 + 255) / 256 * 256)); }

lcpp_srv_cmd() {
  local server=$1 model=$2 ctx=$3
  shift 3
  # shellcheck disable=SC2206 # LCPP_SRV_FIXED is one string of words
  SRV_CMD=("$server" -m "$model" "$@" $LCPP_SRV_FIXED -c "$ctx" --host 127.0.0.1 --port 0)
}

lcpp_srv_probe() {
  local out rc=0 w missing=''
  out=$(CUDA_VISIBLE_DEVICES=-1 timeout --kill-after=10 60 "$1" --help 2> /dev/null) || rc=$?
  if [ "$rc" != 0 ]; then
    SRV_WHY="$1 --help exited $rc"
    return 1
  fi
  for w in "${SRV_CMD[@]:1}"; do
    case $w in
      -[a-z]* | --[a-z]*)
        grep -qE -- "(^|[ ,])$w([ ,=]|$)" <<< "$out" || missing+=" $w"
        ;;
    esac
  done
  if [ -n "$missing" ]; then
    SRV_WHY="$1 --help lists no$missing"
    return 1
  fi
}

lcpp_srv_majflt() { awk '$1 == "pgmajfault" { print $2 }' /proc/vmstat; }

lcpp_srv_start() {
  local bound=$1 log=$2 k
  shift 2
  SRV_PID='' SRV_PORT='' SRV_WHY='' SRV_RC=0
  timeout --kill-after=10 "$bound" env "$@" "${SRV_CMD[@]}" > "$log" 2>&1 &
  SRV_PID=$!
  for ((k = 0; k < 2 * bound; k++)); do
    [ -n "$SRV_PORT" ] || SRV_PORT=$(sed -n 's|.*listening on http://127\.0\.0\.1:\([0-9][0-9]*\)$|\1|p' "$log" | head -n 1)
    if [ -n "$SRV_PORT" ] && curl -sf -o /dev/null --max-time 10 "http://127.0.0.1:$SRV_PORT/health"; then
      return 0
    fi
    if ! kill -0 "$SRV_PID" 2> /dev/null; then
      wait "$SRV_PID"
      SRV_RC=$?
      [ "$SRV_RC" != 0 ] || SRV_RC=70
      SRV_WHY="llama-server exited $SRV_RC before it answered /health"
      return 1
    fi
    sleep 0.5
  done
  SRV_RC=124
  SRV_WHY="llama-server did not answer /health within ${bound} s"
  return 1
}

lcpp_srv_request() {
  local ids=$1 np=$2 bound=$3 body resp code rc=0 t
  SRV_WHY='' SRV_PROMPT_N='' SRV_PROMPT_MS='' SRV_PROMPT_TPS='' SRV_PRED_N='' SRV_PRED_MS='' SRV_PRED_TPS=''
  SRV_CACHE_N='' SRV_TOKENS=''
  body=$(mktemp "${TMPDIR:-/tmp}/lcpp-warm-body.XXXXXX") || return 2
  resp=$(mktemp "${TMPDIR:-/tmp}/lcpp-warm-resp.XXXXXX") || { rm -f "$body"; return 2; }
  python3 -c '
import json, sys
ids = [int(t) for t in sys.stdin.read().split(",") if t.strip()]
print(json.dumps({"prompt": ids, "n_predict": int(sys.argv[1]), "temperature": 0, "top_k": 1,
                  "ignore_eos": True, "cache_prompt": False, "return_tokens": True, "stream": False}))' "$np" <<< "$ids" > "$body"
  SRV_F0=$(lcpp_srv_majflt)
  code=$(curl -sS --max-time "$bound" -o "$resp" -w '%{http_code}' -H 'Content-Type: application/json' \
    --data-binary @"$body" "http://127.0.0.1:$SRV_PORT/completion" 2>&1) || rc=$?
  SRV_F1=$(lcpp_srv_majflt)
  if [ "$rc" != 0 ] || [ "$code" != 200 ]; then
    SRV_WHY="POST /completion: curl rc $rc, HTTP ${code:0:200}: $(head -c 300 "$resp" 2> /dev/null)"
    rm -f "$body" "$resp"
    return 1
  fi
  t=$(python3 -c '
import json, sys
r = json.load(open(sys.argv[1]))
t = r["timings"]
print(t["prompt_n"], t["prompt_ms"], t["prompt_per_second"], t["predicted_n"], t["predicted_ms"],
      t["predicted_per_second"], t.get("cache_n", 0), ",".join(str(x) for x in r.get("tokens", [])) or "-")' "$resp" 2> /dev/null) || {
    SRV_WHY="no timings in the /completion response: $(head -c 300 "$resp")"
    rm -f "$body" "$resp"
    return 1
  }
  rm -f "$body" "$resp"
  read -r SRV_PROMPT_N SRV_PROMPT_MS SRV_PROMPT_TPS SRV_PRED_N SRV_PRED_MS SRV_PRED_TPS SRV_CACHE_N SRV_TOKENS <<< "$t"
  local n
  n=$(tr ',' '\n' <<< "$ids" | grep -c .)
  if [ "$SRV_PROMPT_N" != "$n" ]; then
    SRV_WHY="prompt_n $SRV_PROMPT_N, not the $n ids sent"
  elif [ "$SRV_CACHE_N" != 0 ]; then
    SRV_WHY="cache_n $SRV_CACHE_N: the prompt was not processed whole"
  elif [ "$SRV_PRED_N" != "$np" ]; then
    SRV_WHY="predicted_n $SRV_PRED_N, not $np"
  fi
  [ -z "$SRV_WHY" ]
}

lcpp_srv_arm() {
  local ids=$1 np=$2 bound=$3
  lcpp_srv_request "$ids" "$np" "$bound" || { SRV_WHY="warm-up: $SRV_WHY"; return 1; }
  WARM_FAULTS=$((SRV_F1 - SRV_F0)) WARM_TOKENS=$SRV_TOKENS
  if [ "$np" = 1 ]; then WARM_TPS=$SRV_PROMPT_TPS; else WARM_TPS=$SRV_PRED_TPS; fi
  lcpp_srv_request "$ids" "$np" "$bound" || { SRV_WHY="timed: $SRV_WHY"; return 1; }
  SRV_SAME=$(lcpp_srv_same "$WARM_TOKENS" "$SRV_TOKENS")
}

lcpp_srv_same() {
  local k
  local -a a b
  if [ "$1" = "$2" ]; then echo same; return; fi
  IFS=, read -r -a a <<< "$1"
  IFS=, read -r -a b <<< "$2"
  for ((k = 0; k < ${#a[@]} || k < ${#b[@]}; k++)); do [ "${a[$k]:-}" = "${b[$k]:-}" ] || break; done
  echo "differs at $k"
}

lcpp_srv_stop() {
  local k
  [ -n "${SRV_PID:-}" ] || return 0
  kill "$SRV_PID" 2> /dev/null
  for ((k = 0; k < 150; k++)); do kill -0 "$SRV_PID" 2> /dev/null || break; sleep 0.2; done
  if kill -0 "$SRV_PID" 2> /dev/null; then
    echo "[server] still up 30 s after TERM; KILL to the children of $SRV_PID, then $SRV_PID" >&2
    pkill -KILL -P "$SRV_PID"
    kill -KILL "$SRV_PID" 2> /dev/null
  fi
  wait "$SRV_PID" 2> /dev/null
  SRV_EXIT=$?
  SRV_PID=''
}

lcpp_srv_build() { sed -n 's/^build: //p' "$1" | head -n 1; }
lcpp_srv_device() { sed -n 's/^ *Device 0: \([^,]*\),.*/\1/p' "$1" | head -n 1; }

# The runner side, shared by depth-ds41.sh and depth-qwen3moe.sh: their server arms' engines, command lines,
# checks, dry line, run and row. The runner has, before calling them: its arm arrays A_ENG, A_DEP, A_LABEL,
# A_IDS (lcg, or a corpus name whose ids are A_TOK's), N, BOUND, MODEL, LCPP, LCPP_GPU_FLAGS, CARD_NAME,
# ARM_FAIL_STEM, SRVBIN (srv_preflight sets it), sums and pp_sums; the functions witness, ref_witness,
# lcg_prompt, majflt_now, cold_check, cold_verdict, counted, arm_fail, count_row and lcpp_fit_col; and
#   srv_tail <wall s>   the end of a server row after its device column, in that runner's field order
#   srv_after <round> <label> <key>   (optional) a guard after the server stopped
# A server arm's engine (lcppsrv, lcppsrvfit, lcppsrvpp[fit][<U>]): srv_pp says whether it times the prompt,
# srv_ub its ubatch lever.
srv_eng() { case $1 in lcppsrv | lcppsrvfit | lcppsrvpp | lcppsrvpp[1-9]* | lcppsrvppfit | lcppsrvppfit[1-9]*) return 0 ;; *) return 1 ;; esac; }
srv_pp() { case $1 in lcppsrvpp*) return 0 ;; *) return 1 ;; esac; }
srv_ub() { case $1 in lcppsrvpp*) local u=${1#lcppsrvpp}; echo "${u#fit}" ;; esac; }
# srv_cmd_of <i>: server arm <i>'s command line into SRV_CMD, its n_predict into SRV_NP and
# the batch sizes its prompt row names into SRV_BATCH: LCPP_GPU_FLAGS (with the ubatch lever) in the
# server's spellings, at the fit for a fit twin.
srv_cmd_of() {
  local eng=${A_ENG[$1]} flags=$LCPP_GPU_FLAGS ub
  local -a words
  SRV_NP=$N SRV_BATCH="ub 512 b 2048 (llama-server defaults)"
  if srv_pp "$eng"; then
    SRV_NP=1 ub=$(srv_ub "$eng")
    if [ -n "$ub" ]; then
      flags="$flags -ub $ub -b $((ub > 2048 ? ub : 2048))"
      SRV_BATCH="ub $ub b $((ub > 2048 ? ub : 2048)) (the arm's lever)"
    fi
  fi
  case $eng in *fit*) lcpp_srv_flags "$flags" fit ;; *) lcpp_srv_flags "$flags" ;; esac || return 1
  read -r -a words <<< "$SRV_FLAGS"
  lcpp_srv_cmd "$SRVBIN" "$MODEL" "$(lcpp_srv_ctx "${A_DEP[$1]}" "$SRV_NP")" "${words[@]}"
}
# srv_ids <i>: the ids server arm <i> sends, comma-separated: lcg_prompt D, or its corpus's first P ids.
srv_ids() { if [ "${A_IDS[$1]}" = lcg ]; then lcg_prompt "${A_DEP[$1]}"; else echo "${A_TOK[$1]}"; fi; }
# srv_check_arm <arm>: the pre-lease checks of one server arm's engine and flags: its ubatch lever against
# flags that already set a batch size, and every word of LCPP_GPU_FLAGS in the translation table. 1 with
# SRV_WHY.
srv_check_arm() {
  local eng=${1%%:*} ub
  ub=$(srv_ub "$eng")
  case $ub in *[!0-9]*) SRV_WHY="'$eng' is lcppsrv, lcppsrvfit, lcppsrvpp[fit][<U>]"; return 1 ;; esac
  if [ -n "$ub" ]; then
    case " $LCPP_GPU_FLAGS " in
      *" -ub "* | *" --ubatch-size "* | *" -b "* | *" --batch-size "*)
        SRV_WHY="the profile's flags already set the batch sizes ($LCPP_GPU_FLAGS); the ubatch lever would add a second value"
        return 1
        ;;
    esac
  fi
  case $eng in *fit*) lcpp_srv_flags "$LCPP_GPU_FLAGS" fit ;; *) lcpp_srv_flags "$LCPP_GPU_FLAGS" ;; esac
}
# srv_preflight <runner>: before the lease, when a server arm is given: SRVBIN, which must exist, curl, and
# every flag of every server arm's command line in the server's --help (run with no card); exits 2 or 64
# by name.
srv_preflight() {
  local i
  SRVBIN=$(lcpp_srv_bin)
  [ -x "$SRVBIN" ] || { echo "$1: the lcppsrv arms: no llama-server at $SRVBIN (build it in the tree: cmake --build build --target llama-server)" >&2; exit 2; }
  command -v curl > /dev/null || { echo "$1: the lcppsrv arms need curl" >&2; exit 2; }
  for i in "${!ARMS[@]}"; do
    [ "${A_KIND[$i]}" = srv ] || continue
    srv_cmd_of "$i"
    lcpp_srv_probe "$SRVBIN" || { echo "$1: arm '${ARMS[$i]}': $SRV_WHY" >&2; exit 64; }
  done
}
# srv_tree <tree line>: the server's tree line with the sha256 of the library its launcher runs.
srv_tree() {
  local impl=${SRVBIN%/*}/libllama-server-impl.so
  if [ -f "$impl" ]; then echo "$1 impl=libllama-server-impl.so sha256=$(sha256sum "$impl" | cut -c1-12)"; else echo "$1"; fi
}
# srv_dry_cmd <i>: server arm <i>'s command line as a dry run prints it.
srv_dry_cmd() {
  srv_cmd_of "$1"
  echo "timeout --kill-after=10 $BOUND ${SRV_CMD[*]}   # row label '${A_LABEL[$1]}', ids=${A_IDS[$1]} (${A_DEP[$1]} ids): one POST /completion discarded, then the same timed, n_predict $SRV_NP, greedy, ignore_eos, cache_prompt off$(srv_pp "${A_ENG[$1]}" && echo ", $SRV_BATCH")"
}
# srv_config: the [config] lines of the server arms.
srv_config() {
  lcpp_srv_flags "$LCPP_GPU_FLAGS"
  echo "[config] lcppsrv: $SRVBIN $SRV_FLAGS $LCPP_SRV_FIXED -c <ids + n_predict + 1, rounded up to 256> (lcpp-warm.sh): a discarded POST /completion of the arm's ids, then the same timed; decode rows predicted_per_second at n_predict $N, prompt rows prompt_per_second at n_predict 1 (lcppsrvpp<U>: -ub U -b max(U, 2048))"
  if lcpp_srv_flags "$LCPP_GPU_FLAGS" fit 2> /dev/null; then echo "[config] lcppsrvfit: $SRVBIN $SRV_FLAGS $LCPP_SRV_FIXED (the server's fit places the model)"; fi
}
# One server arm: its llama-server, the warm-up request and the timed one between the
# witness blocks, then the row (srv_row) or a FAIL row; the server stopped on every path. Under the warm
# rows a timed request the cold tag marks is sent once more to the same server before it stops, and the
# rows print after it: the first as COLD, then the retry's row or its FAIL row rc=cold. majflt counts from
# the server's start, timed around the timed request; the wall runs from the start to that request's end.
# srv_arm <index> <round>
srv_arm() {
  local i=$1 r=$2 eng=${A_ENG[$1]} dep=${A_DEP[$1]} label=${A_LABEL[$1]} key ids log t0 f0 rc='' why='' build dev win
  FIT_COL=''
  if srv_pp "$eng"; then key=p=$dep; else key=d=$dep; fi
  srv_cmd_of "$i"
  ids=$(srv_ids "$i")
  log=$(mktemp "${TMPDIR:-/tmp}/$ARM_FAIL_STEM-server.XXXXXX") || exit 2
  witness "pre r$r $label $key"
  ref_witness
  t0=$(date +%s) f0=$(majflt_now)
  SRV_TRIES=0 SRV_RETRY_WHY=''
  if ! lcpp_srv_start "$BOUND" "$log"; then
    rc=$SRV_RC why=$SRV_WHY
  elif ! lcpp_srv_arm "$ids" "$SRV_NP" "$BOUND"; then
    rc=0 why=$SRV_WHY
  else
    srv_take 0 "$t0" "$f0"
    win=$(awk -v p="$SRV_PROMPT_MS" -v g="$SRV_PRED_MS" 'BEGIN { printf "%.4f", (p + g) / 1e3 }')
    cold_check "$((SRV_F1 - SRV_F0))" "$win"
    if [ "$WARM_ROWS" = 1 ] && counted && [ -n "$COLD_TAG" ]; then
      if lcpp_srv_request "$ids" "$SRV_NP" "$BOUND"; then
        SRV_SAME="$(lcpp_srv_same "$WARM_TOKENS" "$SRV_TOKENS") (the retry)"
        srv_take 1 "$t0" "$f0"
      else
        SRV_RETRY_WHY="retry: $SRV_WHY"
      fi
    fi
  fi
  lcpp_srv_stop
  witness "post r$r $label $key"
  ! declare -F srv_after > /dev/null || srv_after "$r" "$label" "$key"
  build=$(lcpp_srv_build "$log") dev=$(lcpp_srv_device "$log")
  # A fit twin's loader lines say what the server's fit chose; a fit that failed or never ran is a FAIL row,
  # whatever the server measured after it (it loads without the fit, as llama-bench does).
  if [ -n "$rc" ]; then
    arm_fail "$r" "$label" "$key" "$rc" "$why" "$(cat "$log")"
  elif [[ $eng == *fit* ]] && ! lcpp_fit_col "$(cat "$log")"; then
    arm_fail "$r" "$label" "$key" 0 "$FIT_WHY" "$(cat "$log")"
  else
    [ -z "$FIT_COL" ] || echo "$FIT_LINES" | sed "s/^/    $label fit /"
    COLD_TRY=0
    srv_row "$i" "$r" 0 "$build" "$dev"
    if [ "$COLD_QUEUED" = 1 ]; then
      ROW_TAG=ROW COLD_TRY=1
      if [ "$SRV_TRIES" = 2 ]; then
        srv_row "$i" "$r" 1 "$build" "$dev"
      else
        arm_fail "$r" "$label" "$key" 0 "$SRV_RETRY_WHY" "$(cat "$log")"
      fi
      COLD_TRY=0
    fi
  fi
  rm -f "$log"
}
# srv_take <try> <t0> <f0>: the request just made, as try <try> of srv_row: its timings, its fault counts
# (whole from <f0>, timed around it), the wall from <t0>, the continuation.
srv_take() {
  SRV_T_PTPS[$1]=$SRV_PROMPT_TPS SRV_T_PN[$1]=$SRV_PROMPT_N SRV_T_PMS[$1]=$SRV_PROMPT_MS
  SRV_T_GTPS[$1]=$SRV_PRED_TPS SRV_T_GMS[$1]=$SRV_PRED_MS SRV_T_SAME[$1]=$SRV_SAME
  SRV_T_TIMED[$1]=$((SRV_F1 - SRV_F0)) SRV_T_WHOLE[$1]=$((SRV_F1 - $3)) SRV_T_WALL[$1]=$(($(date +%s) - $2))
  SRV_TRIES=$(($1 + 1))
}
# srv_row <index> <round> <try> <build> <device>: a server arm's row from try <try> (srv_take), and its
# sums; under the warm rows cold_verdict makes it a COLD row or, on the retry, its FAIL row.
srv_row() {
  local i=$1 r=$2 k=$3 eng=${A_ENG[$1]} dep=${A_DEP[$1]} label=${A_LABEL[$1]} v win tags col unit=tok/s key
  if srv_pp "$eng"; then key=p=$dep unit='tok/s(pp)' v=${SRV_T_PTPS[$k]}; else key=d=$dep v=${SRV_T_GTPS[$k]}; fi
  v=$(awk -v x="$v" 'BEGIN { printf "%.2f", x }')
  win=$(awk -v p="${SRV_T_PMS[$k]}" -v g="${SRV_T_GMS[$k]}" 'BEGIN { printf "%.4f", (p + g) / 1e3 }')
  cold_check "${SRV_T_TIMED[$k]}" "$win"
  MAJ_COL=" | majflt ${SRV_T_WHOLE[$k]} (timed ${SRV_T_TIMED[$k]}; ≤ $MAJ_BOUND % of W ${win} s)"
  cold_verdict "$r" "$label" "$key" "${SRV_T_TIMED[$k]}" "$win" || return 0
  tags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
  col=" | llama-server ids=${A_IDS[$i]}: warm-up $(awk -v x="$WARM_TPS" 'BEGIN { printf "%.2f", x }') $unit majflt $WARM_FAULTS, continuation ${SRV_T_SAME[$k]}"
  if srv_pp "$eng"; then
    echo "$ROW_TAG r$r $label p=$dep n=0 | tok/s(pp) $v @ n=0, prompt $dep, $CARD_NAME$FIT_COL$col | $SRV_BATCH | build ${4:-?} | device ${5:-?}$(srv_tail "${SRV_T_WALL[$k]}")"
    counted || return 0
    pp_sums+=("$label|$dep|$r|$v|$tags")
  else
    echo "$ROW_TAG r$r $label d=$dep n=$N | tok/s $v @ n=$N, depth $dep, $CARD_NAME$FIT_COL$col | prompt_n ${SRV_T_PN[$k]} prompt tok/s $(awk -v x="${SRV_T_PTPS[$k]}" 'BEGIN { printf "%.2f", x }') | build ${4:-?} | device ${5:-?}$(srv_tail "${SRV_T_WALL[$k]}")"
    counted || return 0
    sums+=("$label|$dep|$r|$v||$tags")
  fi
  count_row
}


# `bash tools/ref/lcpp-warm.sh --self-test`: the flag table against V4.1's LCPP_CLI_FLAGS, its refusals,
# the fit twin's flags, the context, the probe against stub binaries, and a stub server's start, requests,
# checks and stop (python3 and curl; just check-recipes runs it on the Mac, where a stub server binds a
# port on 127.0.0.1). The runners' use of it is depth-ds41-stub.sh's and depth-qwen3moe-stub.sh's.
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = --self-test ]; then
  fails=0
  check() {
    if [ "$2" = "$3" ]; then echo "ok $1"; else echo "FAIL $1: got [$2], want [$3]"; fails=$((fails + 1)); fi
  }
  command -v timeout > /dev/null || timeout() { shift 2; exec "$@"; }
  # shellcheck source=tools/ref/lcpp-fit.sh
  source "${BASH_SOURCE[0]%/*}/lcpp-fit.sh"
  lcpp_srv_flags '-ngl 999 --n-cpu-moe 33 -fa on -t 32 -nopo 1'
  check flags-v41 "$SRV_FLAGS" "-ngl 999 --n-cpu-moe 33 -fa on -t 32 --no-op-offload -fit off"
  lcpp_srv_flags '-ngl 99 -fa on -lzm off -ncmoe 26 -t 32'
  check flags-q38 "$SRV_FLAGS" "-ngl 99 -fa on -lzm off -ncmoe 26 -t 32 -fit off"
  lcpp_srv_flags '-ngl 99 -fa on -lzm off -ncmoe 26 -t 32' fit
  check flags-fit "$SRV_FLAGS" "-fa on -lzm off -t 32 -fit on -fitt 1024 -v"
  lcpp_srv_flags '-ngl 999 --n-cpu-moe 33 -fa on -t 32 -nopo 1' fit
  check flags-fit-v41 "$SRV_FLAGS" "-fa on -t 32 --no-op-offload -fit on -fitt 1024 -v"
  lcpp_srv_flags '-ngl 99 -nopo 0 -ub 4096 -b 4096 -ot blk\.1\.=CPU'
  check flags-rest "$SRV_FLAGS" "-ngl 99 --op-offload -ub 4096 -b 4096 -ot blk\.1\.=CPU -fit off"
  lcpp_srv_flags '-ngl 99 -t 16,32' && r=0 || r=1
  check refuse-list "$r|$SRV_WHY" "1|-t '16,32': one value, not a list of bench tests (-ngl 99 -t 16,32)"
  lcpp_srv_flags '-ngl 99 -mmp 0' && r=0 || r=1
  check refuse-unknown "$r|$SRV_WHY" "1|'-mmp' has no llama-server spelling in lcpp-warm.sh's table (-ngl 99 -mmp 0)"
  lcpp_srv_flags '-ngl 99 -nopo 2' && r=0 || r=1
  check refuse-nopo "$r" 1
  lcpp_srv_flags '-ngl 99 -fitt 512' fit && r=0 || r=1
  check refuse-fit-twice "$r|${SRV_WHY%%;*}" "1|the profile's flags already carry -fitt (-ngl 99 -fitt 512)"
  check ctx-decode "$(lcpp_srv_ctx 4096 96)" 4352
  check ctx-pp "$(lcpp_srv_ctx 4096 1)" 4352
  check ctx-small "$(lcpp_srv_ctx 6 96)" 256
  check ctx-edge "$(lcpp_srv_ctx 160 96)" 512
  lcpp_srv_cmd /x/llama-server /m.gguf 256 -ngl 99 -fit off
  check cmd "${SRV_CMD[*]}" "/x/llama-server -m /m.gguf -ngl 99 -fit off -np 1 -ctxcp 0 --cache-ram 0 -c 256 --host 127.0.0.1 --port 0"
  LCPPBIN=/t/build/bin/llama-bench
  check bin "$(lcpp_srv_bin)" /t/build/bin/llama-server
  check bin-own "$(LCPPSRV=/s/llama-server lcpp_srv_bin)" /s/llama-server
  t=$(mktemp -d "${TMPDIR:-/tmp}/lcpp-warm-self-test.XXXXXX")
  # The stub server: --help lists the flags; otherwise it binds 127.0.0.1 at a kernel-picked port, prints
  # the listening line and answers /health and /completion with timings. STUB_SRV_BADN answers with one
  # predicted token too few, STUB_SRV_CACHED with cache_n 3, STUB_SRV_EXIT exits 5 before it listens.
  cat > "$t/llama-server" << 'EOF'
#!/usr/bin/env python3
import http.server, json, os, signal, sys
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))  # llama-server's own handler: a clean exit
if "--help" in sys.argv:
    for f in ("-m", "-ngl", "-fa", "-t", "-fit", "-np", "-ctxcp", "--cache-ram", "-c", "--host", "--port", "--no-op-offload"):
        print(f"{f}, --x   stub")
    sys.exit(0)
if os.environ.get("STUB_SRV_EXIT"):
    sys.exit(5)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.reply(200 if self.path == "/health" else 404, {"status": "ok"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n, p = len(req["prompt"]), req["n_predict"]
        if os.environ.get("STUB_SRV_BADN"): p -= 1
        self.reply(200, {"tokens": [1000 + i for i in range(p)], "timings": {
            "cache_n": 3 if os.environ.get("STUB_SRV_CACHED") else 0, "prompt_n": n, "prompt_ms": 100.0,
            "prompt_per_second": n * 10.0, "predicted_n": p, "predicted_ms": 50.0 * max(p - 1, 0),
            "predicted_per_second": 20.0 if p > 1 else 0.0}})
s = http.server.HTTPServer(("127.0.0.1", 0), H)
print(f"main: listening on http://127.0.0.1:{s.server_address[1]}", flush=True)
s.serve_forever()
EOF
  chmod +x "$t/llama-server"
  lcpp_srv_cmd "$t/llama-server" /m.gguf 256 -ngl 99 -fit off
  lcpp_srv_probe "$t/llama-server" && r=0 || r=1
  check probe "$r" 0
  lcpp_srv_cmd "$t/llama-server" /m.gguf 256 -ngl 99 -lzm off -fit off
  lcpp_srv_probe "$t/llama-server" && r=0 || r=1
  check probe-missing "$r|${SRV_WHY#"$t/llama-server "}" "1|--help lists no -lzm"
  lcpp_srv_cmd "$t/llama-server" /m.gguf 256 -ngl 99 -fit off
  lcpp_srv_majflt() { echo 7; }
  if lcpp_srv_start 20 "$t/log"; then
    lcpp_srv_arm 1,2,3,4,5,6 4 20 && r=0 || r=1
    check arm "$r|$WARM_TPS|$WARM_FAULTS|$SRV_SAME|$SRV_PROMPT_N|$SRV_PRED_N|$SRV_PRED_TPS|$SRV_CACHE_N|$SRV_TOKENS" \
      "0|20.0|0|same|6|4|20.0|0|1000,1001,1002,1003"
    lcpp_srv_request 1,2,3 1 20 && r=0 || r=1
    check pp "$r|$SRV_PROMPT_TPS|$SRV_PRED_N" "0|30.0|1"
    check same "$(lcpp_srv_same 1,2,3 1,2,3)|$(lcpp_srv_same 1,2,3 1,5,3)|$(lcpp_srv_same 1,2 1,2,3)" "same|differs at 1|differs at 2"
    lcpp_srv_stop
    check stopped "$(kill -0 "$SRV_PID" 2> /dev/null && echo up || echo down)" down
  else
    check start "$SRV_WHY" ""
  fi
  lcpp_srv_start 20 "$t/log" STUB_SRV_BADN=1 && {
    lcpp_srv_arm 1,2,3 4 20 && r=0 || r=1
    check badn "$r|$SRV_WHY" "1|warm-up: predicted_n 3, not 4"
    lcpp_srv_stop
  }
  lcpp_srv_start 20 "$t/log" STUB_SRV_CACHED=1 && {
    lcpp_srv_request 1,2,3 4 20 && r=0 || r=1
    check cached "$r|$SRV_WHY" "1|cache_n 3: the prompt was not processed whole"
    lcpp_srv_stop
  }
  lcpp_srv_start 20 "$t/log" STUB_SRV_EXIT=1 && r=0 || r=1
  check exits "$r|$SRV_RC|$SRV_WHY" "1|5|llama-server exited 5 before it answered /health"
  rm -rf "$t"
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($fails failures)"
  [ "$fails" = 0 ]
  exit
fi
