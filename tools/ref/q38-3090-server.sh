#!/usr/bin/env bash
# A lone-3090 Qwen3.8 server at the serving defaults, driven through two multi-turn chats and read off its own log: the
# functional run (no lease, no timing) that the discovery search's refutation calls M4 (specs/leader/discover/refute-report.md
# section 4). It answers what the server's prompt path does on the card the README's 3090 rows never ran it on: the plan's card
# experts with the draft, the expert stream's ring half and lane rate, the residency word, and what a prompt call with a decode
# history restores at its end (`call stream end ... restored= end_us=`, body38.rs stream_end).
#
#   BLOOMERY_MODEL=qwen4exp tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41
#     --release --bin bloomery-serve-qwen38 && bash tools/ref/q38-3090-server.sh $HOME/q38-3090-server/<stamp>'
#   bash tools/ref/q38-3090-server.sh <out dir> [--ctx-size N] [--p1 P] [--p2 P]
#   bash tools/ref/q38-3090-server.sh --self-test        (a stub tree and a stub server; runs on the Mac)
#
# What it runs. tools/box.sh's default pin leaves the 3090 the one visible card, so `bloomery-serve-qwen38` with --place unset
# takes the common rule's answer for one card (plan a on that card, serve_seats/qwen38.rs): adaptive residency mid-p0-s1, the
# expert stream split, the MTP draft under its cost-chosen width. Nothing is set but the server's own non-lever flags: --host
# 127.0.0.1, --port 0 (a free port, read from the `listening on` record), --ctx-size N (default 8192: one slot of one request's
# context, the release pass's height). The script refuses, by name and before the load: a CUDA_VISIBLE_DEVICES that is not one
# card or whose card is not an RTX 3090, and any lever set in its environment, read from the server's own table (`--levers`
# prints one line a lever, `set <value>` for a lever the environment sets, and the binary refuses by name a value or a
# BLOOMERY_ name no registry row takes).
# After the load it checks the server's own word: the `plan` record's card= must name the 3090 (a bare CUDA index follows CUDA's
# fastest-first order, which nvidia-smi's does not), else the server is stopped and the run fails.
#
# The gate lock. The server runs under tools/gpu-gate.sh (the owner of the 3090's lock and of the run's bound; here 1200 s
# unless BLOOMERY_GATE_BOUND), started in the background with its pid captured at spawn (`$!`, written to <out>/gate.pid). The
# gate runner is a wrapper: a TERM to its pid would leave the server running, so the script reads the server's own pid from
# /props (`engine.server_pid`, written to <out>/server.pid) and sends TERM to that, then waits on the gate pid. A lock not
# taken in 300 s is contention (exit 75, the gate runner stopped); a server that does not listen in 600 s after the lock is
# a failed load (exit 1, with its pid unread: the bound ends it).
#
# The chats. /v1/chat/completions, greedy, thinking off (`chat_template_kwargs.enable_thinking` false), cache_prompt on. Chat A:
# a short first turn (about 30 ids, under STREAM_FLOOR 32, wide38.rs:1819: stream_begin, body38.rs:3201-3234, returns before it
# streams unless the split's own walk gate is lower; a first turn that does stream prints kept=1 restored=0, which this run does
# not read), then a second turn that resends the conversation with the answer and a passage of about P1 tokens (default 512): a
# prompt call with a decode history, the case this run reads (a fresh server's call at P tokens is the depth runner's bench arm,
# M5). Chat B the same with P2 (default 4096). Then one short request, so the width chooser's `mtp width` record, which a server
# prints at the slot's next prompt call, reaches the log for chat B's second turn. The passage is the first K ids of the profile's
# prose corpus ($BLOOMERY_DATA/qwen4exp/corpus-prose.ids) through /detokenize, K adjusted by /tokenize to land on P tokens: the
# turn's own prompt_tokens and cached_tokens are printed, so the actual new tokens are on the page. Each request's new lines of
# the server log go to <out>/log-<name>.txt, and the records the lead reads (`call stream end`, `xstream end`, `mtp width`,
# `cache reuse`, `mtp keep`) are printed under it.
#
# Reading it. The load: the `plan` record's card_experts (with the draft's bytes out of the plan), `load draft=`, the
# `xstream=split xstream_half=<N> ... xstream_lane_gbs=<G>` word, the `residency` record. Expected [derived, refutation M4]: half
# N 60..100, lane G 19..21.4 GB/s, residency mid-p0-s1. The second turns: restored 1,700..3,000 at P 512 and end_us 0.35..0.6 s
# (the restore's copies at ~20 GB/s, ~3.1 MB an expert, drained every 4 copies, swap.rs:3211-3270), `kept=0`. restored 800 or
# under at P 512 puts prefill-1 (restore under the walk) at +15 % or less.
set -uo pipefail

die() {
  local rc=$1
  shift
  echo "q38-3090-server.sh: $*" >&2
  exit "$rc"
}

# q38srv_levers <server binary>: the levers its own table reports as set in this environment, one line, space separated; rc 1
# with the binary's refusal on stdout when it refuses the environment (a value its kind does not take, an unknown BLOOMERY_ name).
q38srv_levers() {
  local bin=$1 table
  table=$("$bin" --levers 2>&1) || {
    echo "$table"
    return 1
  }
  printf '%s\n' "$table" | awk '$4 == "set" { printf "%s ", $1 }' | sed 's/ $//'
}

# q38srv_card: the one visible card's name, rc 0 only when CUDA_VISIBLE_DEVICES names exactly one card and it is an RTX 3090.
q38srv_card() {
  local vis=${CUDA_VISIBLE_DEVICES:-} name
  [ -n "$vis" ] || return 1
  case $vis in *,*) return 1 ;; esac
  name=$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader 2> /dev/null |
    awk -F', ' -v v="$vis" '$1 == v || $2 == v { print $3; n++ } END { exit n == 1 ? 0 : 1 }') || return 1
  echo "$name"
  case $name in *3090*) return 0 ;; *) return 1 ;; esac
}

# q38srv_drive <base url> <serve.log> <out dir> <corpus ids file> <p1> <p2> <flush s>: the chats, and the records of each request;
# `flush` seconds after each answer before its log lines are read (the server's stderr is a pipe to a file).
q38srv_drive() {
  python3 - "$@" << 'PY'
import json
import os
import sys
import time
import urllib.request

base, log, out, corpus, p1, p2, flush = sys.argv[1:8]
p1, p2, flush = int(p1), int(p2), float(flush)
HEADS = ("call stream end", "xstream end", "mtp width", "cache reuse", "mtp keep")


def call(path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def passage(ids, want):
    k, n = want, 0
    for _ in range(5):
        if k > len(ids):
            raise SystemExit(f"q38-3090-server.sh: the corpus holds {len(ids)} ids, a passage of {want} tokens needs more")
        text = call("/detokenize", {"tokens": ids[:k]})["content"]
        n = len(call("/tokenize", {"content": text, "add_special": False})["tokens"])
        if abs(n - want) <= max(4, want // 100):
            return text, n
        k = max(8, k + want - n)
    raise SystemExit(f"q38-3090-server.sh: no passage of {want} tokens from the corpus (the last try {n})")


def records(chunk):
    return [ln for ln in chunk.splitlines() if any(ln.startswith(h + " ") for h in HEADS)]


def send(name, messages, max_tokens, log_at):
    t0 = time.time()
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    resp = call("/v1/chat/completions", body)
    wall = time.time() - t0
    time.sleep(flush)
    with open(log, "rb") as f:
        f.seek(log_at)
        chunk = f.read().decode("utf-8", "replace")
    with open(os.path.join(out, f"log-{name}.txt"), "w") as f:
        f.write(chunk)
    with open(os.path.join(out, f"resp-{name}.json"), "w") as f:
        json.dump(resp, f)
    usage = resp.get("usage", {})
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", "?")
    text = resp["choices"][0]["message"]["content"]
    print(f"== {name}: prompt_tokens={usage.get('prompt_tokens', '?')} cached_tokens={cached} "
          f"completion_tokens={usage.get('completion_tokens', '?')} wall_s={wall:.2f}")
    for ln in records(chunk):
        print("   " + ln[:400])
    return text, os.path.getsize(log)


ids = [int(x) for x in open(corpus).read().split()[: max(p1, p2) * 2]]
at = os.path.getsize(log)
Q1 = "In two sentences, what does a mixture-of-experts layer do?"
for tag, want in (("A", p1), ("B", p2)):
    text, n = passage(ids, want)
    conv = [{"role": "user", "content": Q1 if tag == "A" else Q1 + " Keep it short."}]
    a1, at = send(f"{tag}1", conv, 64, at)
    conv += [{"role": "assistant", "content": a1},
             {"role": "user", "content": text + "\n\nIn one sentence, what is this passage about?"}]
    print(f"   {tag}2 passage {n} tokens (asked {want})")
    _, at = send(f"{tag}2", conv, 48, at)
send("F", [{"role": "user", "content": "Say OK."}], 8, at)
print("== read: a prompt call with a decode history ends `call stream end ... kept=0 restored=N end_us=U`")
PY
}

# q38srv_stop <out dir> <gate pid>: TERM to the server's own pid (<out>/server.pid, from /props), the gate runner waited on, KILL
# to the server after 60 s; the runner's rc into <out>/gate.rc. Safe to call twice.
q38srv_stop() {
  local out=$1 gate=$2 pid i rc
  [ ! -e "$out/gate.rc" ] || return 0
  pid=$(cat "$out/server.pid" 2> /dev/null || true)
  if [ -n "$pid" ]; then kill -TERM "$pid" 2> /dev/null || true; else kill -TERM "$gate" 2> /dev/null || true; fi
  i=0
  while kill -0 "$gate" 2> /dev/null; do
    if [ "$i" -ge 60 ]; then
      if [ -n "$pid" ]; then kill -KILL "$pid" 2> /dev/null || true; fi
      kill -KILL "$gate" 2> /dev/null || true
      break
    fi
    sleep "${Q38SRV_POLL:-1}"
    i=$((i + 1))
  done
  rc=0
  wait "$gate" 2> /dev/null || rc=$?
  echo "$rc" > "$out/gate.rc"
}

Q_OUT='' Q_GATE=''
q38srv_main() {
  local out=${1:-} ctx=8192 p1=512 p2=4096 corpus tree card lev gate_rc addr pid i waited
  [ -n "$out" ] || die 64 "usage: q38-3090-server.sh <out dir> [--ctx-size N] [--p1 P] [--p2 P] | --self-test"
  shift
  while [ $# -gt 0 ]; do
    case $1 in
      --ctx-size | --p1 | --p2)
        [ $# -ge 2 ] || die 64 "$1 takes a whole number"
        case $2 in '' | *[!0-9]* | 0) die 64 "$1 takes a whole number above 0, got '$2'" ;; esac
        case $1 in --ctx-size) ctx=$2 ;; --p1) p1=$2 ;; *) p2=$2 ;; esac
        shift 2
        ;;
      *) die 64 "unknown argument '$1'" ;;
    esac
  done
  case $out in /*) ;; *) die 64 "the out dir is an absolute path outside the synced tree (rsync --delete removes what it does not carry), got '$out'" ;; esac
  tree=$PWD
  case $out/ in "$tree"/*) die 64 "the out dir $out is inside the synced tree $tree" ;; esac
  [ ! -e "$out" ] || die 64 "the out dir $out exists: a run writes a new one"
  card=$(q38srv_card) || die 64 "the one visible card must be an RTX 3090 (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}', its card '${card:-none}')"
  [ -x target/release/bloomery-serve-qwen38 ] || die 66 "no target/release/bloomery-serve-qwen38: the recipe builds it before this script runs"
  lev=$(q38srv_levers target/release/bloomery-serve-qwen38) || die 64 "the server refuses this environment: $lev"
  [ -z "$lev" ] || die 64 "lever(s) set in the environment: $lev — the run is the serving defaults, every lever unset"
  [ -n "${BLOOMERY_DATA:-}" ] || die 64 "BLOOMERY_DATA is unset (tools/box.sh exports it): the prose corpus is \$BLOOMERY_DATA/qwen4exp/corpus-prose.ids"
  corpus=$BLOOMERY_DATA/qwen4exp/corpus-prose.ids
  [ -r "$corpus" ] || die 66 "no prose corpus at $corpus"
  mkdir -p "$out" || die 73 "cannot create $out"
  { echo "card: $card (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"; env | grep '^BLOOMERY_' | sort; } > "$out/env.txt"
  BLOOMERY_GATE_BOUND=${BLOOMERY_GATE_BOUND:-1200} bash tools/gpu-gate.sh bloomery-serve-qwen38 --host 127.0.0.1 --port 0 --ctx-size "$ctx" > "$out/serve.log" 2>&1 &
  Q_OUT=$out Q_GATE=$!
  echo "$Q_GATE" > "$out/gate.pid"
  trap 'q38srv_stop "$Q_OUT" "$Q_GATE"' EXIT
  waited=0
  until grep -q '^gpu-gate.sh: bloomery-serve-qwen38 on ' "$out/serve.log"; do
    kill -0 "$Q_GATE" 2> /dev/null || { wait "$Q_GATE"; die 1 "the gate runner ended before it held the lock (rc $?): $(tail -n 3 "$out/serve.log" | tr '\n' ' ')"; }
    [ "$waited" -lt 300 ] || die 75 "the 3090's gate lock was not taken in 300 s: contention, not a failure"
    sleep "${Q38SRV_POLL:-2}"
    waited=$((waited + 2))
  done
  waited=0
  addr=
  while [ -z "$addr" ]; do
    addr=$(sed -n 's/.* listening on http:\/\/\([^ ]*\).*/\1/p' "$out/serve.log" | head -n 1)
    [ -z "$addr" ] || break
    kill -0 "$Q_GATE" 2> /dev/null || { wait "$Q_GATE"; die 1 "the server ended before it listened (rc $?): $(tail -n 3 "$out/serve.log" | tr '\n' ' ')"; }
    [ "$waited" -lt 600 ] || die 1 "the server did not listen in 600 s after the lock (its pid is unread: the gate bound ends it)"
    sleep "${Q38SRV_POLL:-2}"
    waited=$((waited + 2))
  done
  i=0
  until pid=$(curl -sf "http://$addr/props" | python3 -c 'import json,sys; print(json.load(sys.stdin)["engine"]["server_pid"])' 2> /dev/null) && [ -n "$pid" ]; do
    [ "$i" -lt 60 ] || die 1 "no engine.server_pid from http://$addr/props"
    sleep "${Q38SRV_POLL:-1}"
    i=$((i + 1))
  done
  echo "$pid" > "$out/server.pid"
  grep -Eq '^plan .* card=[^ ]*3090' "$out/serve.log" || die 1 "the server's plan record names another card than the 3090: $(grep -m 1 '^plan ' "$out/serve.log" | cut -c1-200)"
  echo "listening on $addr, card $card, ctx-size $ctx, server pid $pid, gate pid $Q_GATE"
  q38srv_drive "http://$addr" "$out/serve.log" "$out" "$corpus" "$p1" "$p2" "${Q38SRV_FLUSH:-1.0}" || die 1 "the chats failed (the server log is $out/serve.log)"
  echo "== load (the server's own records, $out/serve.log)"
  grep -E '^(place unset|parallel |plan |residency |load |xstream=|draft keep|cache |bloomery-serve-qwen38: )' "$out/serve.log" | cut -c1-400
  q38srv_stop "$out" "$Q_GATE"
  trap - EXIT
  gate_rc=$(cat "$out/gate.rc" 2> /dev/null || echo '?')
  echo "gate runner rc $gate_rc (143 is the TERM this script sent the server)"
  case $gate_rc in 0 | 143) exit 0 ;; *) exit 1 ;; esac
}

# --self-test: a stub tree (a stub gate runner, a stub server and its --levers table, a stub nvidia-smi, a tiny corpus) and this
# script run against it, on the Mac as on the box. The stub server writes the records a real one would at each chat request.
q38srv_selftest() {
  local self t fails=0 rc out served p
  self=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
  t=$(mktemp -d "${TMPDIR:-/tmp}/q38srv.XXXXXX") || die 73 "no temp dir"
  mkdir -p "$t/tree/tools" "$t/tree/target/release" "$t/bin" "$t/data/qwen4exp"
  seq 1000 9999 > "$t/data/qwen4exp/corpus-prose.ids"
  printf '%s\n' '#!/bin/sh' 'echo "0, GPU-3090-STUB, NVIDIA GeForce RTX 3090"' 'echo "1, GPU-A6000-STUB, NVIDIA RTX A6000"' > "$t/bin/nvidia-smi"
  # The server's `--levers` table: a line a lever, `set <value>` in the value column when the environment sets it.
  cat > "$t/tree/target/release/bloomery-serve-qwen38" << 'SH'
#!/bin/sh
[ "$1" = --levers ] || exit 2
if [ -n "${BLOOMERY_DRAFT+x}" ]; then
  echo "BLOOMERY_DRAFT             M mode    set $BLOOMERY_DRAFT                parsed"
else
  echo "BLOOMERY_DRAFT             M mode    unset: the plain path            parsed"
fi
echo "BLOOMERY_RESIDENCY         C setting unset: follows the plan          parsed"
echo "BLOOMERY_XSTREAM           A arm     unset: split under a             parsed"
if [ -n "${STUB_LEVERS_REFUSE+x}" ]; then echo "bloomery-levers: $STUB_LEVERS_REFUSE names no registry row" >&2; exit 2; fi
SH
  cat > "$t/tree/tools/gpu-gate.sh" << 'SH'
#!/bin/bash
name=$1
shift
echo "gpu-gate.sh: $name on the 3090 (asked 3090)" >&2
python3 "$(dirname "$0")/stub_server.py" "$@"
exit $?
SH
  cat > "$t/tree/tools/stub_server.py" << 'PY'
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

seen = []
state = {"calls": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/props":
            return self.send({"engine": {"server_pid": os.getpid()}})
        self.send({"status": "ok"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        b = json.loads(self.rfile.read(n).decode())
        if self.path == "/tokenize":
            return self.send({"tokens": list(range(len(b["content"].split())))})
        if self.path == "/detokenize":
            return self.send({"content": " ".join(f"w{i}" for i in b["tokens"])})
        words = len(json.dumps(b["messages"]).split())
        prev = seen[-1] if seen else 0
        state["calls"] += 1
        history = state["calls"] > 1
        if words > 30:
            sys.stderr.write(f"call stream end picks=40 admitted=900 bytes=2800000000 pick_us=1 backlog_us=0 "
                             f"kept={0 if history else 1} restored={2100 if history else 0} end_us={420000 if history else 0}\n")
        if state["calls"] == 5:
            sys.stderr.write("mtp width windows=10 kept=1,2,3,4 widths=0,0,0,10 e=2.400 gate=4 costs=-,-,-,40 a=0.8 closed=0\n")
        sys.stderr.flush()
        seen.append(words)
        self.send({"choices": [{"message": {"role": "assistant", "content": "ok"}}],
                   "usage": {"prompt_tokens": words, "completion_tokens": 3,
                             "prompt_tokens_details": {"cached_tokens": min(prev, words)}}})


srv = HTTPServer(("127.0.0.1", 0), H)
sys.stderr.write(f"plan place=a card={os.environ.get('STUB_PLAN_CARD', 'RTX3090')} card_experts=3336\n")
sys.stderr.write("xstream=split xstream_half=77 xstream_staging=64 xstream_lane_gbs=20.10 xstream_m_min=2494 (unset: stub)\n")
sys.stderr.write(f"bloomery-serve-qwen38: place=a ctx=8192 slots=1 slot_ctx=8192 listening on http://127.0.0.1:{srv.server_address[1]}\n")
sys.stderr.flush()
srv.serve_forever()
PY
  chmod +x "$t/bin/nvidia-smi" "$t/tree/target/release/bloomery-serve-qwen38" "$t/tree/tools/gpu-gate.sh"
  # st_run <label> <expected rc> <pattern in the output> [NAME=VALUE ...] -- <args>: this script in the stub tree.
  st_run() {
    local label=$1 want=$2 pat=$3
    shift 3
    local -a envs=()
    while [ "$1" != -- ]; do
      envs+=("$1")
      shift
    done
    shift
    out=$(cd "$t/tree" && env PATH="$t/bin:$PATH" CUDA_VISIBLE_DEVICES=GPU-3090-STUB BLOOMERY_DATA="$t/data" Q38SRV_POLL=0.2 Q38SRV_FLUSH=0.2 ${envs[@]+"${envs[@]}"} bash "$self" "$@" 2>&1)
    rc=$?
    if [ "$rc" != "$want" ] || ! printf '%s\n' "$out" | grep -qE -- "$pat"; then
      echo "FAIL $label: rc $rc (want $want), pattern '$pat' in:" >&2
      printf '%s\n' "$out" | tail -n 12 >&2
      fails=$((fails + 1))
    else
      echo "ok   $label"
    fi
  }
  st_run happy 0 'B2: prompt_tokens=' -- "$t/out1" --p1 40 --p2 120
  # st_has <label> <pattern>: the happy run's output holds the pattern.
  st_has() {
    if printf '%s\n' "$out" | grep -qE -- "$2"; then echo "ok   $1"; else
      echo "FAIL $1: no '$2' in the happy run's output" >&2
      fails=$((fails + 1))
    fi
  }
  served=$out
  out=$served st_has second-turn-restored 'call stream end .*kept=0 restored=2100 end_us=420000'
  out=$served st_has width-record 'mtp width windows=10'
  out=$served st_has load-records 'xstream=split xstream_half=77 .*xstream_lane_gbs=20.10'
  out=$served st_has passage-tokens 'B2 passage 120 tokens'
  st_run plan-card 1 'plan record names another card than the 3090' STUB_PLAN_CARD=A6000 -- "$t/out12" --p1 40 --p2 120
  st_run lever-set 64 'lever\(s\) set in the environment: BLOOMERY_DRAFT' BLOOMERY_DRAFT=mtp -- "$t/out6"
  st_run unknown-name 64 'STUB_UNREGISTERED names no registry row' STUB_LEVERS_REFUSE=STUB_UNREGISTERED -- "$t/out7"
  st_run two-cards 64 'must be an RTX 3090' CUDA_VISIBLE_DEVICES=GPU-3090-STUB,GPU-A6000-STUB -- "$t/out8"
  st_run a6000-visible 64 'must be an RTX 3090' CUDA_VISIBLE_DEVICES=GPU-A6000-STUB -- "$t/out9"
  st_run relative-out 64 'absolute path' -- out10
  st_run existing-out 64 'exists' -- "$t/out1"
  st_run bad-number 64 'whole number' -- "$t/out11" --p1 x
  # Every served run stopped its server: the pid file names a process that is gone, and the gate rc is the TERM's.
  p=$(cat "$t/out1/server.pid" 2> /dev/null || echo 0)
  if [ "$p" = 0 ] || kill -0 "$p" 2> /dev/null; then
    echo "FAIL out1: its server (pid $p) is still up or was never recorded" >&2
    if [ "$p" != 0 ]; then kill -KILL "$p" 2> /dev/null || true; fi
    fails=$((fails + 1))
  elif [ "$(cat "$t/out1/gate.rc" 2> /dev/null)" != 143 ]; then
    echo "FAIL out1: the gate runner's rc is '$(cat "$t/out1/gate.rc" 2> /dev/null)', not the 143 of the TERM" >&2
    fails=$((fails + 1))
  else
    echo "ok   server stopped, gate rc 143"
  fi
  rm -rf "$t"
  [ "$fails" = 0 ] || die 1 "self-test: $fails failed"
  echo "q38-3090-server.sh: self-test ok"
}

if [ "${1:-}" = --self-test ]; then
  q38srv_selftest
  exit $?
fi
q38srv_main "$@"
