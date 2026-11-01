# bloomery 작업 목록. 모든 타깃은 맥에서 치고 박스에서 돈다(tools/box.sh가 rsync한다).
# 맥은 arm64라 CPU 커널이 아예 빌드되지 않는다. 로컬 cargo로 게이트를 돌리려 하지 말 것.
#
# 레시피는 절대 `cd ~/repo/bloomery`를 쓰지 않는다. box.sh가 이미 $REMOTE로 들어가고 그 값은
# 워크트리 이름에서 유도된다 — 레시피가 다시 cd하면 워크트리 라운드가 메인 트리를 재게 된다
# (2026-09-19 ffn 라운드가 보고: "gate-ffn은 ~/repo/bloomery로 가는데 이 워크트리는
# ~/repo/bloomery-ffn으로 rsync된다. 둘 다 박스에 있어서 엉뚱한 트리를 시험한다").

# The card check a timed recipe runs first in its box command, before any build (tools/ref/card-precheck.sh).
precheck := 'bash tools/ref/card-precheck.sh'

# The shared target selector also supplies the tools that inspect each build.
arch := `bash tools/arch.sh`
export BLOOMERY_ARCH := arch

# A recipe that builds a shipped bin (tools/release/build.sh's BINS) builds it with that file's
# feature list, spelled out: the gates of a shipped bin run the binary the release packs, a recipe parameter in a
# `--features` hides the crates from the gate selection and the ledger key, and a bin takes one feature set across
# the recipes (tools/recipes.py's bin check). A gate bin that needs other features goes on a cargo line of its own.
# `just check-recipes` fails a recipe by name.

# serve-qwen38's placement (a the A6000, gate the 3090) and the card it puts
# in box.sh's view.
serve_place := env_var_or_default('SERVE_PLACE', 'a')
serve_card := if serve_place == 'gate' { '3090' } else { 'a6000' }

default:
    @just --list

# `--features gpu`(R26): 피처 뒤에 본체가 숨은 바이너리는 그 피처를 켜야 검사된다 —
# gpu-gates 바이너리의 GPU 본체가 끄면 통째로 안 보인다.
# 빠른 루프: 타입 검사만, 커널은 안 만든다. 의존을 고친 뒤 cargo가 박스 쪽 Cargo.lock을 고쳐 쓰면 lock-back.sh가
# 그것을 이 트리로 가져온다 — box.sh는 한 방향으로만 싣는다.
check:
    ./tools/box.sh 'cargo check --workspace --all-targets --features gpu,bloomery-metal/pipeline-scan,bloomery-metal/encode-trace,bloomery-gpu-gates/deepseek41,bloomery-gpu-gates/vision,bloomery-gpu-gates/glm5next,bloomery-gpu-gates/mimo2,bloomery-gpu-gates/solar2,bloomery-gpu-gates/minimaxm3,bloomery-gpu-gates/flint,bloomery-gpu-gates/gemma-embedding2,bloomery-gpu-gates/gemma4a,bloomery-gpu-gates/clef,bloomery-gpu-gates/hf,bloomery-gpu-gates/glm5next-hf,bloomery-gpu-glm5next/hf,bloomery-decision/safetensors'
    ./tools/lock-back.sh

# check와 같은 이유로 `--features gpu`. 이 피처를 켠 것이 기준 계기다(R26). V4.1 op 게이트의 피처
# `deepseek41`도 켠다 — 그 바이너리들은 이 피처가 있어야 컴파일된다. clippy는 디바이스 코드를 만들지 않으므로
# V4.1 커널의 컴파일러 결함은 여기가 아니라 op 게이트 빌드에서 드러난다.
# lint. 에러 0이 계약이고 경고 수는 RESULTS/AGENTS에 적힌 기준선과 비교한다.
lint:
    ./tools/box.sh 'cargo clippy --workspace --all-targets --features gpu,bloomery-metal/pipeline-scan,bloomery-metal/encode-trace,bloomery-gpu-gates/deepseek41,bloomery-gpu-gates/vision,bloomery-gpu-gates/glm5next,bloomery-gpu-gates/mimo2,bloomery-gpu-gates/solar2,bloomery-gpu-gates/minimaxm3,bloomery-gpu-gates/flint,bloomery-gpu-gates/gemma-embedding2,bloomery-gpu-gates/gemma4a,bloomery-gpu-gates/clef,bloomery-gpu-gates/hf,bloomery-gpu-gates/glm5next-hf,bloomery-gpu-glm5next/hf,bloomery-decision/safetensors'

# The build without `vision` of the bins that refuse an image request by name — serve_seats/ds41.rs:444 and glm.rs:825 (--mmproj),
# decide.rs:545 (the Qwen3.5 seat's encoder), gate_glm5next_e2e.rs:5146 (--only media-seat). One bin takes one feature set (tools/recipes.py's
# bin check), and every recipe that builds these bins builds them with `vision`, so only this check compiles the
# `cfg(not(feature = "vision"))` branches. A cargo check writes no executable, so it replaces no bin another lane runs. mac-combos runs this shape.
check-no-vision:
    ./tools/box.sh 'cargo check -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --bin bloomery-serve --bin bloomery-serve-ds41 --bin gate_glm5next_e2e'

# fmt는 맥에서 돈다. box.sh의 rsync가 단방향이라 박스에서 포맷하면 결과가 돌아오지
# 않고 다음 명령에 덮여 사라진다(2026-09-19에 그렇게 한 번 날렸다). cargo fmt는 컴파일을
# 하지 않고 파싱만 하므로 arm64 맥에서 정상 동작한다 — AGENTS.md의 "맥에서 게이트 금지"는
# 빌드가 필요한 것에 대한 규칙이고, 판정은 박스의 fmt-check가 한다. rustfmt는 rust-toolchain.toml이 고정한
# 나이틀리의 것이다 — PATH의 Homebrew rustfmt는 다른 판이다. 명령은 fmt-check의 것에서 `-- --check`를 뺀 것이고
# 툴체인을 고르는 것은 tools/mac-check.sh다.
fmt:
    ./tools/mac-check.sh fmt

fmt-check:
    ./tools/box.sh 'cargo fmt --all -- --check'

# 정적 계층을 맥에서 돈다: check·lint 레시피의 박스 명령에 `--target x86_64-unknown-linux-gnu`를 붙인 교차
# 검사(링크 없음), fmt-check의 명령은 고정 툴체인의 rustfmt로. 테스트와 게이트는 여기서 돌지 않는다.
# 환경은 레포 밖 ~/opt/bloomery-mac-env.sh 하나다 — 준비물과 만드는 법은 tools/mac-check.sh 머리말.
# check 다음에 mac-combos를 돈다: 박스 창 전에 라운드가 도는 이 한 줄이 게이트 레시피의 빌드 모양까지 본다.
mac-check:
    ./tools/mac-check.sh check
    ./tools/mac-check.sh combos

# check 레시피는 피처 한 벌만 빌드하므로 `--features gpu` 단독이나 피처 없는 lib 테스트 빌드의 컴파일 오류는
# 여기서만 잡힌다. 목록의 주인은 tools/recipes.py combos다.
# 게이트·빌드 레시피가 컴파일하는 (패키지, 모드, 피처) 모양마다 `cargo check`를 같은 교차 검사로 한 번씩 돈다.
mac-combos:
    ./tools/mac-check.sh combos

# lint의 `^warning:` 수를 AGENTS.md의 기준값과 비교한다. 넘으면 빨강.
mac-lint:
    ./tools/mac-check.sh lint

# fmt-check를 맥에서, 고정 나이틀리의 rustfmt로 돈다.
mac-fmt-check:
    ./tools/mac-check.sh fmt-check

# 순수 크레이트(tools/recipes.py pure-crates가 크레이트 그래프와 소스로 고른 것)의 `cargo test -p`를 맥에서 네이티브로 돈다.
# 개발 루프의 증거이고, 착륙의 증거는 박스의 기록이다.
mac-test:
    ./tools/mac-check.sh test

# The Metal backend's gate, on an Apple Silicon Mac: the device contract's clauses on this Mac's Metal device, run by
# tools/mac-gate.sh (the Metal lock, the bound, the refusals). crates/metal links no cuda-oxide, so plain cargo builds it.
# The crate's lib tests (the qwen35moe session's dispatch counts) and its bins' unit tests first, then the device
# gate, then the qwen35moe kernels' gate (the qwen35moe fixture under $BLOOMERY_FIXTURE_ROOT, or --fixture), then that
# gate's mutants, each of which its clause must catch. A lab-* recipe, not a gate-* one: the box has no Metal device, so no landing batch
# selects it. The mutants step runs every registered instance's mutants in one process under one bound; 2026-10-27 it
# took 945 s on an M1 Pro (16-core GPU) at two instances and 220 mutants, over the 900 s default, so it runs at 1800.
lab-metal:
    cargo test -p bloomery-metal --lib && cargo test -p bloomery-metal --features pipeline-scan --bin metal_scan && cargo test -p bloomery-metal --bin metal_attention_time --bin metal_gemm_time --bin generate_metal && cargo build --release -p bloomery-metal --bin gate_metal_device --bin gate_metal_qwen35moe_kernels && ./tools/mac-gate.sh gate_metal_device && ./tools/mac-gate.sh gate_metal_qwen35moe_kernels && BLOOMERY_GATE_BOUND=1800 ./tools/mac-gate.sh gate_metal_qwen35moe_kernels --mutants-all

# The Metal session's e2e gate on an Apple Silicon Mac, run by tools/mac-gate.sh over six rows (`--row`): the qwen35moe
# MLX split, then the qwen3moe MLX split (attention layers only, no shared expert, no recurrent store; the clauses that
# hold a delta contract print SKIP with their reason), then the qwen38d MLX split (Qwen3.8-27B's fixture: a dense FFN in
# every layer, so the clause that walks a router's picks prints SKIP), then the lfm3 checkpoint folder (LFM3-1.7B's
# MLX 4-bit folder, `$BLOOMERY_FIXTURE_ROOT/lfm3-mlx`: a short conv beside the attention layers, a sandwich residual, a
# dense FFN and a tied head; no `prepare` map reads it, so the folder is the model, and the clause that tells a folder
# from its split prints SKIP), then the llama checkpoint folder (Llama-3.1-8B-Instruct's MLX 4-bit folder cut to four
# layers, `$BLOOMERY_FIXTURE_ROOT/llama-mlx`: grouped-query attention with no head norm and NORM rope pairs, a dense
# SwiGLU and a head of its own; the folder is the model there too), then the qwen3 checkpoint folder (Qwen3-8B's MLX
# 4-bit folder cut to four layers, `$BLOOMERY_FIXTURE_ROOT/qwen3-mlx`: grouped-query attention of 32 heads over 8 key
# heads of 128 values with a q and a k head norm and the NEOX rope, a dense SwiGLU in every layer and a head of its
# own; the clauses of a delta layer or a router print SKIP). A split row opens
# `$BLOOMERY_FIXTURE_ROOT/<family>-mlxsplit`: the token's encoded count, the fault path of a bad token id, the
# generation core the CLI runs (ids, stops, sink), the
# profile step (its picks and logits equal the plain step's, its group tags covering the count's groups, its weight
# bytes equal the file's own tensors), the MLX folder opened as it is, which must tell what its prepared split tells
# (m_mlx_folder; before the gate, model-hf's hw_a_ tests hold the two files' tensor bytes equal), and the first-step
# logits against the committed mlx-lm dump (m_logits, and for the lfm3 row m_logits_deep over its deep pin, past the
# sliding window); then the gate's kernel mutants of the qwen35moe row, of the qwen38d row, of the lfm3 row, of the
# llama row and of the qwen3 row, each of which its clause must catch; then the generate_metal CLI's smoke on the qwen35moe split (and, for
# the ids and the text prompt, on the qwen3moe and qwen38d splits). The prompt ids through --tokens must print the ids
# pinned here, a text prompt must decode to non-empty stdout, and the decode rate and the profile's decode_ms must count the loop's passes (none for a one-token
# run, 7 for eight tokens). The GGUF fixture smoke left with the ik clauses.
# PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified
# A lab-* recipe: the box has no Metal device, so no landing batch selects it.
lab-metal-e2e:
    #!/usr/bin/env bash
    set -euo pipefail
    die() { echo "lab-metal-e2e: $*" >&2; exit 1; }
    # A step that fails ends the recipe with its own exit code, named by its command line.
    run() { "$@" || { rc=$?; echo "lab-metal-e2e: '$*' failed (rc $rc)" >&2; exit "$rc"; }; }
    run cargo build --release -p bloomery-metal --bin gate_metal_qwen35moe_e2e --bin generate_metal --bin bloomery-serve-metal
    # model-hf's hw_a_ tests are #[ignore]d for the fixture root's MLX folders and their splits, which only this recipe has.
    cargo test -p bloomery-model-hf --lib -- --ignored hw_a_
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e --row qwen3moe
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e --row qwen38d
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e --row lfm3
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e --row llama
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e --row qwen3
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e --mutants-all
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e --row qwen38d --mutants-all
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e --row lfm3 --mutants-all
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e --row llama --mutants-all
    run ./tools/mac-gate.sh gate_metal_qwen35moe_e2e --row qwen3 --mutants-all
    # The smoke opens the MLX split's first shard, the same file the e2e gate opens.
    shard() {
        local dir="${BLOOMERY_FIXTURE_ROOT:-/models/fixtures}/$1-mlxsplit" found=()
        shopt -s nullglob
        found=("$dir"/*-00001-of-*.gguf)
        shopt -u nullglob
        [ "${#found[@]}" -eq 1 ] || die "$1-mlxsplit has ${#found[@]} first shards under $dir"
        printf '%s\n' "${found[0]}"
    }
    fx="$(shard qwen35moe)"
    want="95971,52281,135771,76423,11251,224948,16570,68738"
    ids="$(BLOOMERY_REF_MODEL="$fx" ./target/release/generate_metal --tokens 760,6511,314 -n 8 --print-ids)" || die "generate_metal smoke: the run failed (rc $?)"
    [ "$ids" = "$want" ] || die "generate_metal smoke: ids $ids; $want is expected"
    # The qwen3moe split's prompt is 785,6722,315: the CLI picks the library of the file's instance.
    fx3="$(shard qwen3moe)"
    want3="54792,120829,120829,120829,120829,54792,54792,14152"
    ids3="$(BLOOMERY_REF_MODEL="$fx3" ./target/release/generate_metal --tokens 785,6722,315 -n 8 --print-ids)" || die "generate_metal qwen3moe smoke: the run failed (rc $?)"
    [ "$ids3" = "$want3" ] || die "generate_metal qwen3moe smoke: ids $ids3; $want3 is expected"
    out3="$(BLOOMERY_REF_MODEL="$fx3" ./target/release/generate_metal --prompt "The quick brown fox" -n 8)" || die "generate_metal qwen3moe text smoke: the run failed (rc $?)"
    [ -n "$out3" ] || die "generate_metal qwen3moe text smoke: empty stdout"
    # The qwen38d split's dense FFN runs the prompt as dense GEMMs: the same prompt ids as the qwen35moe split's (one vocabulary).
    # Four of the 27B's layers pick the space token every step: the smoke holds the CLI path, m_logits holds the numbers.
    fx38="$(shard qwen38d)"
    want38="220,220,220,220,220,220,220,220"
    ids38="$(BLOOMERY_REF_MODEL="$fx38" ./target/release/generate_metal --tokens 760,6511,314 -n 8 --print-ids)" || die "generate_metal qwen38d smoke: the run failed (rc $?)"
    [ "$ids38" = "$want38" ] || die "generate_metal qwen38d smoke: ids $ids38; $want38 is expected"
    out38="$(BLOOMERY_REF_MODEL="$fx38" ./target/release/generate_metal --prompt "The quick brown fox" -n 8)" || die "generate_metal qwen38d text smoke: the run failed (rc $?)"
    [ -n "$out38" ] || die "generate_metal qwen38d text smoke: empty stdout"
    out="$(BLOOMERY_REF_MODEL="$fx" ./target/release/generate_metal --prompt "The quick brown fox" -n 8)" || die "generate_metal text smoke: the run failed (rc $?)"
    [ -n "$out" ] || die "generate_metal text smoke: empty stdout"
    one="$(BLOOMERY_REF_MODEL="$fx" ./target/release/generate_metal --tokens 760,6511,314 -n 1 --print-ids --profile 2 2>&1 >/dev/null)" || die "generate_metal rate smoke: the one-token run failed (rc $?)"
    { grep -qx 'decode_passes=0' <<<"$one" && grep -qx 'decode_tok_s=none' <<<"$one" && grep -q '^prof total .* decode_ms=none$' <<<"$one"; } || die "generate_metal rate smoke: a one-token run must print decode_passes=0, decode_tok_s=none and the profile's decode_ms=none"
    many="$(BLOOMERY_REF_MODEL="$fx" ./target/release/generate_metal --tokens 760,6511,314 -n 8 --print-ids --profile 2 2>&1 >/dev/null)" || die "generate_metal rate smoke: the eight-token run failed (rc $?)"
    tr ' ' '\n' <<<"$many" | awk -F= '$1=="decode_passes"{p=$2} $1=="decode_s"{s=$2} $1=="decode_tok_s"{r=$2} $1=="decode_ms"{m=$2} END{ok = p==7 && s>0 && r/(p/s)>0.97 && r/(p/s)<1.03 && m/(s/p*1e3)>0.97 && m/(s/p*1e3)<1.03; exit !ok}' || die "generate_metal rate smoke: an 8-token run must count 7 passes, and decode_tok_s and the profile's decode_ms must be per pass"

# The Metal backend's move proof, on an Apple Silicon Mac (`just ptx-compare`'s analogue; AGENTS.md's change-class table, the move row): a
# change that should leave a model's speed alone leaves the program alone. BASE (a commit, read through `git archive`) and this tree (or
# `--head REV`, or `--here DIR`) each build `metal_scan` and `generate_metal` with the `pipeline-scan` and `encode-trace` features, then on
# the qwen35moe fixture ($BLOOMERY_FIXTURE_ROOT): `metal_scan` builds every compute pipeline the engine builds from both GEMM programs,
# serializes each into a binary archive and hashes it (what the device compiler makes of each kernel, one row per pipeline), and
# `generate_metal --encode-trace` writes every dispatch, binding, `setBytes` value, barrier and commit/wait/event at depth ~0 and at depth 600.
# Prints both tables (`metal_scan --compare`, `--compare-trace`: the first differing dispatch). Exit 0 only when every pipeline hash, both
# traces and the generated ids are identical; 3 something existing moved or a trace differs, 4 a pipeline was removed, 5 both, 6 only
# pipelines were added, 2 a side refused by name (no instrument and none applies), 1 a build or run failed. A side older than the commit
# that added the instrument gets it laid over. Runs no gate and takes no lock; the owner is tools/metal-compare.sh.
[arg("HEAD", long="head")]
[arg("HERE", long="here")]
[arg("OUT", long="out")]
metal-compare BASE HEAD='' HERE='' OUT='':
    ./tools/metal-compare.sh {{BASE}} {{ if HEAD != '' { '--head "' + HEAD + '"' } else { '' } }} {{ if HERE != '' { '--here "' + HERE + '"' } else { '' } }} {{ if OUT != '' { '--out "' + OUT + '"' } else { '' } }}

# clippy on the Metal backend's macOS body, which the static tier's x86_64-linux cross check compiles as an empty crate.
mac-metal-lint:
    cargo clippy -p bloomery-device -p bloomery-metal --all-targets --features bloomery-metal/pipeline-scan,bloomery-metal/encode-trace

# 라운드 루프의 맥 정적 계층 한 줄(≤180 s 목표): fmt-check와 check 스크립트 여덟은 병렬 레인, cargo 레인은 lint(clippy가
# type-check를 겸한다 — check를 따로 돌리면 멤버 크레이트를 두 번 짓는다) 다음 BASE 범위 combos. BASE를 주면 그 base
# 이후 바뀐 파일을 입력에 갖는 모양만 돌고 초록이 레저(~/.cache/bloomery/mac-static-ledger.tsv)에 남는다; 인자 없이 전체.
# 주인은 tools/mac-static.sh다.
mac-static *ARGS:
    ./tools/mac-static.sh {{ARGS}}

# `just mac-static BASE` for this tree on the second Mac (idle, 10 cores), its output and exit code back here: a round runs
# `just remote-static BASE` instead of `just mac-static BASE` when this Mac's load decides the wall of the latter. The second
# Mac runs at most two at once, a third waits for a slot. Setup tools/mac-remote/SETUP.md; the owner is tools/mac-remote.sh.
remote-static *ARGS:
    ./tools/mac-remote.sh {{ARGS}}

# Every pool Mac's run slots and track locks (live or dead); `just mac-locks --clean` removes the dead ones
# (no live remote run and a heartbeat older than LOCK_DEAD, 600 s). The owner is tools/mac-remote.sh.
mac-locks *ARGS:
    ./tools/mac-remote.sh --locks {{ARGS}}

# An off-box check for a contributor: fmt, the pure crates' tests, the Python tool self-tests and every tools/check-*.sh
# (T0, any OS with the pinned toolchain), plus the host-nightly units on x86_64 or aarch64 Linux with no card (T1, the
# default on x86_64).
# `just contrib-check --tier t0|t1`. One line per unit, then a DONE line; docs/contrib/local-check.md. The owner is tools/contrib-check.sh.
contrib-check *ARGS:
    ./tools/contrib-check.sh {{ARGS}}

# Before a push: Shingler's reviewer on this machine, on the committed HEAD against origin/main, under your own Claude
# login (`--full` adds the validator, `--minutes N` bounds a role, default 10). Exit 0 CLEAN, 1 MUST_FIX, 75 no verdict.
# A base behind origin/main is refused by name (64) before the reviewer runs. A CLEAN run prints one pull-request body
# line with the full HEAD sha. Nothing is posted; the pull request's own review stays the review of record. The owner
# is tools/review/local.py.
pre-review *ARGS:
    python3 tools/review/local.py {{ARGS}}

# 레시피 자체의 점검(맥, grep뿐). 게이트 줄의 `||`는 종료 코드를 삼킨다 — tools/check-recipes.sh 머리말.
check-recipes:
    ./tools/check-recipes.sh

# The gate-* recipes that read a file changed since the merge-base with BASE (default main), or in A..B (from merge-base(A, B) to B, as `git diff A...B`) — on the Mac, no build, runs nothing (tools/affected-gates.sh).
# The class stage is on: a gate of an F family (tools/families.tsv) stays only when its family's files, its recipe or its own targets changed,
# and each one left out is printed with its reason; `--no-class` keeps every gate the graph reaches.
# `--narrow --scan BASE_LOG NEW_LOG …` (ptx-scan logs of the base and the change): the gates that run the changed host path.
affected BASE='main' *ARGS:
    ./tools/affected-gates.sh {{BASE}} {{ARGS}}

# BASE..HEAD narrowed by ptx-scans it takes itself: derives the bins whose scans cover every kernel carrier (tools/recipes.py
# scan-set), builds and scans them with `--no-jit` at both commits in two persistent box dirs (~/repo/bloomery-narrow-base,
# ~/repo/bloomery-narrow-head: a second run builds incrementally), and prints `just affected --narrow` over those pairs,
# its `recipes:` line included. HEAD defaults to this tree's HEAD commit, read through `git archive` (uncommitted edits are
# not scanned); `--worktree` instead scans this tree's current content (uncommitted and untracked files too) as the head side,
# for a round that may not commit: BASE must be an ancestor of HEAD and no HEAD is given. It runs no gate and takes no lease,
# no hold, no card and no gate lock. `--out DIR` keeps the scan logs and the list (default
# ~/.cache/bloomery/narrow/<tree>/<base8>-<head8>, <base8>-worktree); `--dry-run` prints the scan set and touches no box.
# The owner is tools/narrow-scan.sh.
[arg("OUT", long="out")]
[arg("DRY", long="dry-run", value="--dry-run")]
[arg("TREE", long="worktree", value="--worktree")]
narrow BASE HEAD='HEAD' OUT='' DRY='' TREE='':
    ./tools/narrow-scan.sh {{BASE}} {{ if HEAD != 'HEAD' { HEAD } else { '' } }} {{TREE}} {{DRY}} {{ if OUT != '' { '--out "' + OUT + '"' } else { '' } }}

# The proof that a kernel move is bit-identical, in one command (AGENTS.md's change-class table): `just narrow`'s two --no-jit scans
# BASE..HEAD (`--worktree` for this tree's current content; same dirs, lock and base-side cache), with each bin's PTX modules
# brought back, then `ptx-canon.py --scans` per bin and `ptx-canon.py BASE.ptx HEAD.ptx ENTRY` for every entry whose md5
# moved. Prints one line per moved entry and a summary (`canon-check: N bins, M entries: i identical, r reordered-only, d differs,
# o only-on-one-side; rows equal|moved: …`); exit 0 only when no entry differs, none is on one side only, every swapped pair is
# independent and every compared row column is equal, 1 otherwise, 2 on a refusal named (a missing module, a failed scan, a
# usage error), 75 on contention. Runs no gate and takes no lease, no hold, no card and no gate lock. `--out DIR` keeps the scan
# logs, the modules and DIR/canon/ (the diffs; default ~/.cache/bloomery/canon/<tree>/<base8>-<head8>, <base8>-worktree);
# `--dry-run` prints the scan set and touches no box. The owner is tools/canon-check.sh (`--from DIR` re-judges a kept dir).
[arg("OUT", long="out")]
[arg("DRY", long="dry-run", value="--dry-run")]
[arg("TREE", long="worktree", value="--worktree")]
canon-check BASE HEAD='HEAD' OUT='' DRY='' TREE='':
    ./tools/canon-check.sh {{BASE}} {{ if HEAD != 'HEAD' { HEAD } else { '' } }} {{TREE}} {{DRY}} {{ if OUT != '' { '--out "' + OUT + '"' } else { '' } }}

# The add and move classes' ptx-scan proof (AGENTS.md's change-class table) in one command: BIN scanned with the driver JIT at BASE (a
# commit, read through `git archive` into a box dir of its own, ~/repo/bloomery-<track>-ptxcmp-base) and at this tree (as `just ptx-scan`
# syncs it), and one table: entries on each side, added, removed, and the existing entries whose any column or md5 moved. Exit 0 when
# nothing existing moved (the head holds every base entry, equal, and maybe more: `added` says which), 3 an entry changed, 4 removed,
# 5 both, 2 a log or pair refused by name, 1 a side failed, 75 the box is held. The base side is cached under target/ptx-compare/ by
# BASE, BIN, the features and the scanner: `--base-only` runs it at a round's open, the run without it at the end; `--no-cache` scans
# the base again. A gate-batch item: `'ptx-compare:BASE generate_ds41 --features gpu,deepseek41'`.
# The add and move proof of two JIT scans as one table (owner: tools/ptx-compare.sh).
[arg("FEATURES", long="features")]
[arg("BASEONLY", long="base-only", value="--base-only")]
[arg("NOCACHE", long="no-cache", value="--no-cache")]
ptx-compare BASE BIN FEATURES='gpu' BASEONLY='' NOCACHE='':
    @bash tools/scan-args.sh ptx {{BIN}}
    ./tools/ptx-compare.sh {{BASE}} {{BIN}} --features {{FEATURES}} {{BASEONLY}} {{NOCACHE}}

# The host CPU flags of an oxide build: .cargo/config.toml's list per triple, what tools/oxide-host.sh makes of it, and the generated
# .cargo/cuda-oxide.toml — on the Mac, no build.
check-rustflags:
    ./tools/check-rustflags.sh

# 아키텍처 축 점검(맥, grep뿐) — docs/arch-split.md 「검사」. 넷 다 엄격하다.
check-arch:
    bash tools/check-arch.sh

# Defaults twins (Mac, text only): a lever a gate pins off its default needs a clause that runs the default, or a dated
# reason in tools/defaults-twins.tsv (tools/check-defaults.py's header has the rules).
check-defaults:
    bash tools/check-defaults.sh

# Model-name ratchet (Mac, text only): the code lines of common code that name a model, held two-sided per (file, family)
# by tools/model-tax.tsv, and the tables held to the rows BASE had, so a count cannot be raised by editing them
# (tools/model_tax.py's header has the rules).
check-model-tax BASE='origin/main':
    bash tools/check-model-tax.sh check --base {{BASE}}

# The docs a start by name is read from, held to the code they repeat (Mac, text only): the README's Use table and docs/models.md's
# Recommended file against the model catalog (tools/catalog.tsv), and docs/agents.md's exit table, event names, stop causes and loading
# phases against crates/serve (tools/check-catalog-docs.py has each copy and its owner).
check-catalog-docs:
    bash tools/check-catalog-docs.sh

# 주석 규약(AGENTS.md Conventions): 엔진 크레이트 src/ 주석에 이슈 번호·날짜 금지, 예외는 `PIN(날짜):`.
check-comments:
    ./tools/check-comments.sh

# 레버 읽기 점검(맥, 텍스트뿐): crates/*/src에서 레버 레지스트리 밖에서 환경 변수를 읽는 곳은 그 읽기를 옮길
# 라운드와 함께 tools/levers-direct.txt에 한 줄로 올라 있어야 한다 — 규칙은 tools/check-levers.sh 머리말.
check-levers:
    ./tools/check-levers.sh

# The per-crate unsafe ratchet (Mac, grep only, no build): the unsafe lines under crates/<dir>/{src,tests,benches}
# against the pins in tools/unsafe-ratchet.txt; the counting rule and exit codes are tools/check-unsafe.sh's header.
check-unsafe:
    bash tools/check-unsafe.sh

# The engine thread's bounded waits on the card (Mac, text only, no build): each function tools/check-waits.sh
# lists waits with host::await_done before any blocking copy and never calls .synchronize(); rule in its header.
check-waits:
    ./tools/check-waits.sh

# The bounded-child owner's static guard (Mac, text only, no build): a runner under tools/ starts a child through
# tools/gate-bound.sh and never under a `timeout` of its own; the rule and its list are in tools/check-bounded.sh.
check-bounded:
    bash tools/check-bounded.sh

# The one-bundle-load ratchet (Mac, grep only, no build): a `#[cuda_module]` family's raw `load` outside the
# shared_module! slot — the rule and exit codes are tools/check-loads.sh's header.
check-loads:
    ./tools/check-loads.sh

# The ptx-shapes name check (Mac, text only, no build): every row of tools/ref/ptx-shapes.tsv names a kernel its binary's device
# crates declare, and every kernel they declare has a row — gate-ptx-spill's `stale` and `unpinned` halves without the box. Rule,
# macro family, refusals and gaps are tools/check-ptx-names.py's header.
check-ptx-names:
    ./tools/check-ptx-names.sh

# `[group('v41-load')]`: 바이너리가 V4.1 전체를 적재하는(body::open — 호스트 세트 populate) 레시피. tools/gate-batch.sh가 한 레인(A)에 차례로 세운다.
# GPU 게이트 바이너리는 tools/gpu-gate.sh 아래에서 카드마다 하나인 게이트 락에 줄을 선다(빌드는 병렬, 실행만 직렬).
# 게이트 하나가 모델을 12 GB 올리므로 둘이 같은 카드에 겹치면 OOM이다. 시간 러너의 임대(/root/bloomery-cpu.lock)와는
# 다른 락이다.
#
# GPU 커널 게이트 P1–P3: 트랙마다 자기 바이너리 하나(crates/gpu-gates/src/bin/gate_pN.rs).
# 참조는 bloomery_gpu_gates(gguf 디퀀트 + f64 내적), 밴드는 KERNEL_BAND = 1e-5. 정확성 실행이고 측정이 아니다.
gate-gpu-p1:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p1 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p1'

gate-gpu-p2:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p2 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p2'

gate-gpu-p3:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features gpu --bin gate_p3 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p3 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p3'

gate-gpu-p4:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p4 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p4'

gate-gpu-p5:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p5 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p5'

gate-gpu-p6:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p6 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p6'

gate-gpu-p9:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p9 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p9'

# B6: q4k_gemv_sel — 슬롯마다 그 expert 하나로 돈 q4k_gemv와 비트 동일, 범위 밖 id는 슬롯을 건드리지 않는다.
gate-gpu-q4k-sel:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_q4k_sel && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_q4k_sel'

# 모델 없는 K-quant expert 계열(bloomery_gpu::kquant): Q5_K `_sel` down과 gate·up(act 런타임 인자)을 합성 스택에서
# f64 참조의 밴드, 슬롯·m열·폴트·act 절로 보고, 같은 워크의 Q4_K 형제가 q4k_gemv_sel과 비트 동일한지 본다. relu² 전문가 카드 레그의
# 양자화기 둘(열 복제와 relu² 접힘)이 Lightning 형상에서 호스트 규칙과 비트 동일한지도 본다(절 13). 모델 파일 없음.
gate-gpu-kquant:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_kquant && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_kquant'

# 두 단계 후보 선택(bloomery_gpu::cand, V4.1 후보 마스크의 커널 넷)을 모델 없이 합성 점수로 호스트 규칙과 비트 대조한다: 블록 키,
# 참조 select_candidate_blocks의 선택(핀, 동점은 낮은 블록, ±0 동점), 경계 이하 무작업, 압축·히스토그램·카운트 뷰, remap, 폴트, 캡처 재생.
gate-gpu-cand:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_cand && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_cand'

# Qwen3.8의 gated-residual hyper-connection(bloomery_gpu::hc_gated)을 모델 없이 파일 모양(4×2560, rank 320)의 합성
# 입력으로 본다: f64 규칙의 밴드, 단계마다 카드 규칙과 비트 동일, 8열과 1열, combine·init, 이름 붙은 거부, 폴트, 로컬 depot.
gate-gpu-hc-gated:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_hc_gated && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_hc_gated'

# The Mamba-2 decode step (bloomery_gpu::mamba2: conv ring, bias, SiLU, scan at state 128, D, z gate, grouped norm — one fused launch) against the
# host twin model::mamba2::step_token, with no model file, at the 4B widths: bit for bit where the device library is not asked for a value libm
# gives differently, within the derived band where it is; state 96/256 refused by name, the non-finite-y fault, eager = captured replay.
gate-gpu-mamba2:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_mamba2 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_mamba2'

# The chunked SSD prompt scan (bloomery_gpu::mamba2_ssd: one launch per pass, f16 tensor-core products, f32 sums) against an f64 twin of the
# scan rule and beside the engine's f32 pass, at the Bolt widths with no model file: every y and the final state within the derived f16 bound
# (1 to 2048 tokens, two passes in a row), bit-identical reruns, the non-finite fault and its reach, named refusals, no local depot.
gate-gpu-mamba2-ssd:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_mamba2_ssd && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_mamba2_ssd'

# NVFP4 dense gemv (bloomery_gpu::mxfp4, Nvfp4Kernels) at Bolt's ssm_in / ssm_out shapes, synthetic tensors loaded through
# Weights::load_where and run through site::kgemv: every output written, within the derived band of the host dequant
# times the tensor's scale, bit for bit the host rule; the load's refusals by name. The routed pair (bloomery_gpu::mxfp4_sel)
# at Qwen3.8's widths on stacks with a scale an expert, loaded on a plan row's card experts: the same clauses, ids past the
# stack raised by name. `gate_nvfp4 --bench-kernels` prices the dense gemv and the routed pair.
gate-gpu-nvfp4:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_nvfp4 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_nvfp4'

# MXFP8 dense gemv and wide tile (bloomery_gpu::q8f32, Gemm32Kernels) at attn_q_b / attn_output / ssm_f_b shapes of a 4,096-wide model,
# synthetic tensors loaded through Weights::load_where: every output written, within the derived band of the host dequant, bit for
# bit the host walk; the Q8_0 site gemv and the F32 tile bit for bit their rules; the load's refusals by name.
# `gate_mxfp8 --bench-kernels` prices the gemv beside Q8_0.
gate-gpu-mxfp8:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features gpu --bin gate_mxfp8 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_mxfp8 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_mxfp8'

# BF16 weights in their words (bloomery_gpu::q8f32's bf16 gemv, row list and tile, the head, q38's embedding lookup) on
# synthetic tensors loaded through the placed path: every output written, within the derived band, bit for bit the F32
# rule over the widened values and the F32 site's outputs; the Q8_0 head and embedding bit for bit their rules.
gate-gpu-bf16:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features gpu --bin gate_bf16 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_bf16 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_bf16'

# nemotron_h (Nemotron 3 Nano 4B Q4_K_M) end-to-end gate, one card: the captured step's entries against the layer kinds the header derives
# (the scans, the squared ReLUs, the group-5 flash passes, the appends, the Q5_0 embedding row; no group-8 flash entry, no host node), eager =
# captured replay bit for bit over 16 tokens, reset rewinds the ids and the logits bits, the scan state carries (layer 0 nonzero, the last
# token's logits differ from the token alone), the fault word clean. A step that raises the fault word fails the clause naming it. Against
# mainline's last-position logits (ref-nemotron-logits, `just dump-ref-nemotron`): the body's argmax is the set's on every prompt, and
# each prompt's cosine keeps the floor with the same top-5 ids; the max |delta| and cosine print per prompt. A set the box lacks fails by name.
# Real tier only: the set is the real file's and the Nano has no fixture twin of it, so the fixture tier defers the gate by name.
gate-gpu-nemotron-e2e:
    BLOOMERY_MODEL=nemotron_h ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-nemotron-e2e && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_nemotron_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_nemotron_e2e'

# nemotron_h_moe (Nemotron 3.5 Bolt) end-to-end gate, one card, on the fixture tier's four layers (or the real file's eighty-eight
# under the real tier): the captured step's entries against the layer kinds the header derives (the group-16 attention, the MoE
# layers' routers at their own counts and the four iq2_xxs_gemv_sel a MoE layer), eager = replay, reset rewinds, the scan state
# carries; in the fixture tier mainline's set (fx_ref-nemotron-moe-logits): the cosine floor, the argmax above the band and
# each MoE layer's kept experts on every prompt.
gate-gpu-nemotron-h-moe-e2e:
    BLOOMERY_MODEL=nemotron_h_moe ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_nemotron_h_moe_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_nemotron_h_moe_e2e'

# Muse Spark Open's end-to-end gate (gate_musespark_e2e's doc). Fixture tier: the whole decode chain of the fixture's four
# layers (pattern [T,F,T,T], so layers 0→1 and 1→2 are the SWA→full transitions) against mainline's node dumps
# (BLOOMERY_TIER=fixture just dump-ref-musespark writes fx_ref_musespark_{a,b,c,e} first): the structure clause's entries, eager =
# replay, reset rewinds, the K/V rows carry, a NaN router lane raises the fault word, moe_is_its_rule, then every compared node
# at every layer and position within the bands, set e's positions past the 2,048 window, the greedy continuation's argmax above
# the band, and the placed arms (card, mid, host: the card arm bit-identical to the whole-card load, layer 0 of the first token
# near it on the other two). Real tier: the real file as the placed load of the gate plan (the 3090's bytes on the card in
# view, nearly every routed expert on the host tier: ~204e9 host bytes, alone), the same own-contract clauses, the three oracle
# clauses against ref_musespark_{a,b,c,e} (just dump-ref-musespark) and the host-only arm. Wall [derived]: 8-10 min.
[group('solo-real')]
gate-gpu-musespark-e2e:
    BLOOMERY_MODEL=muse-spark ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_musespark_e2e && BLOOMERY_GATE_BOUND=1800 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_musespark_e2e'

# Muse Spark Open's prompt batch gate (gate_musespark_prefill's doc): the batch against the one-token steps over prompt lengths
# spanning the 2048 window, in the logits row, layer 0's output and the stores, then a step after the batch; passes of 7 rows
# against the load's bit for bit; every layer alone on the steps' own input within the band. Fixture tier: the whole-card and
# the half-budget placed arms; real tier: the gate plan, alone (~204e9 host bytes). Wall [derived]: ~15 min fixture, 26-32 min
# real.
[group('solo-real')]
gate-gpu-musespark-prefill:
    BLOOMERY_MODEL=muse-spark ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_musespark_prefill && BLOOMERY_GATE_BOUND=2400 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_musespark_prefill'

# Muse Spark Open on plan (b′) (gate_musespark_prefill's doc, its two-card arm, `--two-card`): the A6000 the stage card, the 3090
# the expert tier, the host the rest, under one card budget. The load's tier on the 3090 as planned, 16 steps eager = replay, the
# first token's layers and a two-pass prompt's last row within 1e-3 of the union on the A6000 alone (opened first and released),
# then the common prompt clauses on the two cards with the tier serving their batches; then the seat's gate (gate_musespark_serve)
# with both cards in view: `--place bp` plans the tier card, `--place` unset takes bp exactly when its plan puts the family's
# break-even (845 experts) on the tier, and the server loads and answers.
# Fixture tier: half the experts on each card. Real tier: ~2 GiB of experts on the stage, the 3090's room on the tier, the rest on
# the host, alone. Wall [derived]: the prefill gate's one arm plus two loads and the seat's gate, ~10 min fixture, 30-40 min real.
[group('solo')]
[group('v41-load')]
gate-gpu-musespark-twocard:
    BLOOMERY_MODEL=muse-spark BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_musespark_prefill && BLOOMERY_GATE_BOUND=2400 bash tools/gpu-gate.sh gate_musespark_prefill --two-card && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_musespark_serve && D=target/musespark-serve-twocard && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_BOUND=1800 bash tools/gpu-gate.sh gate_musespark_serve --dir $D'

# Muse Spark Open's load gate (gate_musespark_load's doc): the body's placed load of the gate plan proven against its plan —
# every expert in one place and the stack bytes the file's, each segment on the card resident at the plan's buffers and nothing
# else, the routed stacks' first n_l experts and sampled norms the file's bytes, the slot map the ids' prefix, the load inside
# the plan. Fixture tier: the fixture's card under a half-experts budget. Real tier: ~204e9 host bytes, alone.
[group('solo-real')]
gate-gpu-musespark-load:
    BLOOMERY_MODEL=muse-spark ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_musespark_load && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_musespark_load'

# The Muse Spark seat of bloomery-serve (gate_musespark_serve's doc): the seat plans one slot, refuses a second slot and a tier card, runs the common
# residency machine and plans the expert stream's ring by the common rule, loads the file at the context it was asked for and answers one chat turn
# with its reasoning split out of the recipient protocol and its turn ending at the model's own end token (the last two chat clauses are the real
# tier's, Tag::FileBound). Loads the whole model once, under the big-load
# lock the V4.1 loads take. Predicted wall [derived: gate-gpu-minimaxm3-serve's fixture rows, 50 to 53 s]: about 1 min on the fixture; the real file
# about 3 min (load 40 to 110 s, a 175-id prompt as steps at ~78 ms, the answer); the bound is 1800 s.
[group('solo-real')]
[group('v41-load')]
gate-gpu-musespark-serve:
    BLOOMERY_MODEL=muse-spark ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_musespark_serve && D=target/musespark-serve-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_BOUND=1800 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_musespark_serve --dir $D'


# Solar-Open2-250B end-to-end gate (gate_solar2_e2e's doc): the 36 KDA layers (beta = 2 sigmoid, softplus decay) and 12 NoPE gated GQA
# layers with their 320-expert routed blocks, one prompt a step, against the PAI tree's CPU dump of the same file (refset family
# pai-solar2, `just dump-ref-solar2`): every layer's output at the last position within the relative ceiling, in the real tier the 40 top
# logprobs of each of four prompts within the band derived from the family's sigma with the argmax PAI's or a named tie, and PAI's
# 64 greedy tokens of three prompts teacher-forced with the clear-margin disagreements within the pin derived from their margins. The
# fixture tier runs the self-consistency clauses and the layer outputs against the `fx_` sets and leaves the logprob and greedy
# clauses to the real tier by name. Loads the whole model once: alone in a batch in the real tier, under the big-load lock.
[group('solo-real')]
[group('v41-load')]
gate-gpu-solar2-e2e:
    BLOOMERY_MODEL=solar-open2 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features solar2 --release --bin gate_solar2_e2e && BLOOMERY_GATE_BOUND=1200 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_solar2_e2e'

# Solar-Open2-250B prefill gate (gate_solar2_prefill's doc): the prompt batch is the steps within the error model's band — the last
# logits row and the step after a prompt by z against the logit sigma, every layer's output under the two-implementation ceilings,
# the stores' list — at P = 1, 7, 512 and 513 ids; the group clauses are bit for bit. Every clause is a self-consistency clause, so
# the fixture tier runs them all on the fixture's four layers.
[group('solo-real')]
[group('v41-load')]
gate-gpu-solar2-prefill:
    BLOOMERY_MODEL=solar-open2 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features solar2 --release --bin gate_solar2_prefill && BLOOMERY_GATE_BOUND=1200 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_solar2_prefill'

# Solar-Open2-250B on plan (b′) (gate_solar2_twocard's doc): the stage on the A6000 and the expert tier on the 3090 against the same
# plan with every tier segment moved onto the stage card (the A6000 alone, no tier code), both planned under one card budget the gate
# sets, on a fixed walk whose routing is checked to reach every tier layer: the step leg's tokens and logits rows, the prompt call's
# token and logits row, the group leg's call at groups of two against groups of one, and the stage graph's node count. Every clause is
# a self-consistency clause, so the fixture tier runs them all (the fixture's header budget puts about half of each card-eligible
# layer's experts on a card). Both cards in view (BLOOMERY_CARD=both: both gate locks, alone in a batch), two loads of the model in
# turn under the big-load lock the V4.1 loads take. Predicted wall [derived: gate-gpu-solar2-e2e's loads measured 9 to 28 s warm, a
# decode step about 41 ms; glm5next-twocard at the fixture tier is 49 s median]: two loads of 10 to 60 s, the step leg's 80 steps about
# 3 s, the call leg 512 ids and the group leg 2 x 1024 ids a few seconds each: 40 s to 3 min warm after a warm build; the bound is
# 1200 s.
[group('solo')]
[group('v41-load')]
gate-gpu-solar2-twocard:
    BLOOMERY_MODEL=solar-open2 BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features solar2 --release --bin gate_solar2_twocard && BLOOMERY_GATE_BOUND=1200 bash tools/gpu-gate.sh gate_solar2_twocard'

# The Solar-Open2 seat of `bloomery-serve` (gate_solar2_serve's doc): `--parallel 2` refused by name before any plan, one server at
# --ctx 4096 loading the file and listening (its `load` and `listening` records name the architecture, one slot, the file's layers and
# the context), and one chat turn at temperature 0 with a reasoning budget of 16 answered with reasoning and content both, split at the
# model's own `<|think:start|>` / `<|think:end|>` (the seat reads the pair from the model's description), neither tag left in either.
# Loads the whole model once, under the big-load lock the V4.1 loads take. Predicted wall [derived: gate-gpu-solar2-e2e's loads measured
# 9 to 28 s warm, a decode step about 41 ms]: one `--plan` refusal of about 1 s, the load 9 to 28 s warm (up to about 3 min cold, 151.8 GB
# at 1 to 3 GB/s from the drive), the chat's prompt call about 2 s and its 64 steps about 2.6 s: 15 to 40 s warm after a warm build;
# the bound is 600 s. Measured on the A6000, the page cache warm: 24 s on a start whose GPU code was cached (the load record 16.9 s,
# listening at 18.1 s), 49 s on the first start of a new binary (the load record 22.9 s, listening at 39.8 s: the 15.7 s outside the
# load record, against 1.2 s on the second start, is not decomposed; the start note names a first start's GPU code compile); the lock
# waits are the gate runner's and are in neither.
[group('solo-real')]
[group('v41-load')]
gate-gpu-solar2-serve:
    BLOOMERY_MODEL=solar-open2 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_solar2_serve && D=target/solar2-serve-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_BOUND=600 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_solar2_serve --dir $D'

# The nemotron seat of bloomery-serve on Nemotron 3 Nano 4B Q4_K_M (gate_nemotron_serve's doc): `--plan` with no flag names its
# default context (the ladder's rung, held by the ctx line and the rung above refused) and exits before any load, `--parallel 2` is
# refused by name (the body holds one sequence); one server with no flag listens at the context its ctx line named, answers
# /completion at temperature 0 with the ids NemotronHModel gives stepped directly on the same ids (prose, and the chat turn the
# server's template renders), serves that turn on /v1/chat/completions (usage counts its ids), resends an edited prompt as a
# fresh run that keeps nothing and an extended one that keeps the held positions. One card.
gate-gpu-nemotron-serve:
    BLOOMERY_MODEL=nemotron_h ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_nemotron_serve && D=target/nemotron-serve-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_nemotron_serve --dir $D'

# The same seat gate on the moe body (gate_nemotron_serve's doc, every clause one path over both
# bodies): the Bolt fixture under the fixture tier (nemotron_h_moe, the fixture's four layers, any
# card), the arch word and the layer count read from the file, the session's prompt call the body's
# batched `GpuModel::prefill` (several ids; one id is a step). One card.
gate-gpu-nemotron-h-moe-serve:
    BLOOMERY_TIER=fixture BLOOMERY_MODEL=nemotron_h_moe ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_nemotron_serve && D=target/nemotron-h-moe-serve-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_nemotron_serve --dir $D'

# GLM-5.3-Flash 라우터(288개 중 8개, sigmoid, ×2.5 — `router::glm5next`)를 모델 없이 합성 입력으로 호스트 규칙과 맞춘다.
# 규칙마다 절이 하나다: logits·ids·weights·batch는 비트, scores는 sigmoid 밴드, ties·guard·fault·graph는 명시한 값.
gate-gpu-glm-router:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_glm5next_router && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_router'

# IQ2_XXS·IQ2_XS·IQ3_XXS·IQ4_XS·Q2_K 행 코어(bloomery_gpu::iq): ref-synth 합성 행(gate-1-1이 덤프; IQ2_XXS는 시드 블록)을 q8_1
# 8열과 곱해 호스트 규칙과 비트 동일한지, ggml 디퀀트 × f32 기준 대비 유도 한계와 핀 안인지 본다. K = 4096과 꼬리가 남는
# K = 2304 두 형상, IQ2_XXS는 Bolt 공유 전문가(4096 → 5376, 5376 → 4096)도.
gate-gpu-iq:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_iq && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_iq'

# i-quant expert select entries (bloomery_gpu::iq_sel) over the file's own block word stream, on synthetic stacks
# (ref-synth + seeded IQ4_NL and IQ2_XXS): the IQ2_XS, IQ3_XXS and IQ4_XS gate·ups, the IQ4_NL down `_sel32`, the IQ2_XXS,
# IQ2_XS, IQ3_XXS and IQ4_XS downs over Q8Act (GLM UD-Q2_K_XL's) and IQ2_XXS's routed up and down at Nemotron 3.5 Bolt's shapes.
# Bit-equal to the host rule and the plane kernels, the f64 band and pins, the silu rule band, faults, NaN d, a wide
# activation, capture and named refusals. The IQ4_NL down `_sel32` also runs at Lightning's two shapes (K = 2688 and 1856, ten
# slots on a repeated column) for the relu² card leg's ungated up.
gate-gpu-iq-sel *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_iq_sel && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_iq_sel {{ARGS}}'

# 행 사상 Q6_K gemv(bloomery_gpu::q6k_ids): MTP 드래프트가 빌리는 Q6_K `output` 머리. 합성 70,000행 "어휘"(K = 2,560,
# UD-Q3_K_XL 머리의 형상)를 고정 시드로 짓고, m ∈ {1,2,4,8} × (1·7·65,536 정렬 id 맵 + 맵 없음)의 매 로짓을
# enqueue_gemv_q6k의 모은 행 스택과 비트 대조하고, ggml 디퀀트 × f32의 f64 합 대비 유도 밴드(PIN), id 범위 밖의 이름 붙은
# 폴트와 NaN 행, 캡처 재생, 런처의 이름 붙은 거부를 본다. 모델 파일 없음.
gate-gpu-q6k-ids:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_q6k_ids && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_q6k_ids'

# 종단 게이트: 27층 + lm_head + argmax 전체 사슬을 ik CUDA greedy(33프롬프트 × 32스텝)와 대조하고,
# 그래프 재생이 즉시 실행과 토큰 단위로 같은지 본다(정확성 실행, 측정 아님). 이름이 pN이 아닌 이유는
# gate_e2e.rs 머리글 — P9·P10은 커널 꾸러미로 이미 쓰이고 있다.
gate-gpu-e2e *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_e2e {{ARGS}}'

# 하이브리드 MoE 경계 게이트: V2-Lite를 n_l 32와 0(n_l 밖 expert는 호스트)으로 올려 전부 카드인 모델과 대조한다.
# 캡처 노드 수, eager = 재생, 층별 카드 슬롯 비트 동일·호스트 합 밴드, argmax 뒤집힘 밴드, 겹침 레버는 순서만 바꾸는지.
[group('pool')]
gate-gpu-hybrid *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_hybrid && bash tools/gpu-gate.sh gate_hybrid {{ARGS}}'

# 적응형 residency 기계 게이트(모델 파일 없음, 합성 스택): 같은 이력 = 같은 값(복사 시점과 무관), 정적 재배치와 비트 동일,
# 늦은 복사는 스트림이 기다림, 호스트·카드 맵이 한 id를 두 번 또는 0번 서비스하지 않음, reset은 시드로, 호스트 비상주
# 희생자는 이름으로 거부, 고정 시드는 움직이지 않음(adaptres §4 c1–c5·c7).
[group('pool')]
gate-gpu-swap *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_swap && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_swap {{ARGS}}'

# The expert stream's gate (no model file, a synthetic source): the stream is the rule's set cut at the ring's half in
# rank order, its rows and exclusion set say so, the engine stream reads a slot only once its copy landed, a half's next
# stream waits for the half's read, an unread stream refuses the call's end, a stuck lane is a named error within its
# deadline, a ring with no room, or capped at no bytes, is refused by its own name.
gate-gpu-xstream *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_xstream && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_xstream {{ARGS}}'

# 양자화 모델의 정확한 수학(가중치는 정확히 역양자화, 활성·어텐션은 f64, q8·f16 반올림 없음)으로 교사 강제 자의
# 한 프롬프트를 푼다: 스텝마다 정확 top1·마진·ik 토큰 격차. 두 엔진(우리, ik)이 갈리는 자리의 심판이다.
# CPU 64스레드를 쓰므로 기계 전역 임대를 잡는다. 예: `just exact-ref 12 --steps 23`, `--kv f16`은 캐시 f16 반올림 팔.
exact-ref ID *ARGS:
    ./tools/box.sh '{{precheck}} docs/cards/exact-ref.card && cargo build --release -p bloomery-gpu-gates --bin exact_ref && bash tools/ref/lease-hold.sh --card docs/cards/exact-ref.card -- ./target/release/exact_ref --prompt-id {{ID}} {{ARGS}}'

# gate-gpu-e2e 교사 강제 팔의 참값 파일: prompts.tsv의 프롬프트마다 exact_ref --emit(32스텝 전부)을 돌려
# $BLOOMERY_DATA/exact-forced-32.tsv 하나로 모은다. 머리 주석에 모델 경로, 강제 토큰 파일(greedy-ik-cuda-32.tsv)의
# sha256, exact_ref 빌드 커밋(맥 워크트리의 git describe — 박스 트리에는 .git이 없다)과 실행한 바이너리의 sha256.
# 임대는 프롬프트마다 따로 잡는다(전체 약 50분 — 다른 라운드가 그 사이에 끼어들 수 있게). 실행하는 바이너리는
# 조각 디렉터리로 복사한 사본이라 도중의 다른 빌드가 바꾸지 못한다. 최종 파일은 .tmp.<pid>에 쓰고 rename한다.
# 강제 토큰 파일을 다시 만들었으면(greedy-ref-cuda) 이것도 다시 — 게이트가 sha256으로 낡은 참값을 거부한다.
build-exact-forced:
    #!/usr/bin/env bash
    set -euo pipefail
    commit=$(git describe --always --dirty --abbrev=12)
    script=$(cat <<'EOF'
    set -euo pipefail
    {{precheck}} docs/cards/exact-ref.card
    cargo build --release -p bloomery-gpu-gates --bin exact_ref
    out="$BLOOMERY_DATA/exact-forced-32.tsv"
    forced="$BLOOMERY_DATA/greedy-ik-cuda-32.tsv"
    ids=$(grep -v -e '^#' -e '^$' tools/ref/prompts.tsv | cut -f1)
    parts="$out.parts.$$"
    mkdir -p "$parts"
    cp target/release/exact_ref "$parts/exact_ref"
    for id in $ids; do
      bash tools/ref/lease-hold.sh --card docs/cards/exact-ref.card -- "$parts/exact_ref" --prompt-id "$id" --emit "$parts/p$id.tsv" > "$parts/p$id.log"
      echo "exact-forced p$id rows=$(grep -vc '^#' "$parts/p$id.tsv")"
    done
    heads=$(for id in $ids; do head -n 1 "$parts/p$id.tsv"; done | sort -u)
    [ "$(printf '%s\n' "$heads" | wc -l)" = 1 ] || { echo "build-exact-forced: parts disagree: $heads" >&2; exit 1; }
    tmp="$out.tmp.$$"
    {
      echo "$heads forced=greedy-ik-cuda-32.tsv forced_sha256=$(sha256sum "$forced" | cut -d' ' -f1) exact_ref_commit=$COMMIT exact_ref_sha256=$(sha256sum "$parts/exact_ref" | cut -d' ' -f1)"
      printf '#id\tstep\texact_top1\texact_top2\texact_margin\n'
      for id in $ids; do grep -v '^#' "$parts/p$id.tsv"; done
    } > "$tmp"
    mv "$tmp" "$out"
    rm -rf "$parts"
    echo "build-exact-forced: wrote $out rows=$(grep -vc '^#' "$out")"
    EOF
    )
    ./tools/box.sh "COMMIT=$commit; $script"

# 한 강제 위치에서 우리 GPU 기본 패스(텐서 코어 flash)의 층·탭별 상대 거리를 세 심판 팔에 대해 한 번에 찍는다: f64(정확),
# ours(우리 활성 양자화 규칙만 흉내 낸 f64 사슬 — K-quant 입력 128값 블록), ik(ik 규칙 — 전부 32값 블록).
# exact_ref가 세 팔을 CPU 임대 아래 차례로 덤프하고(프롬프트당 팔 하나 약 1.5분), forced_probe가 그 셋과 대조한다.
# 덤프와 exact_ref 로그는 박스 target/exact-taps/ID-STEP/. 예: `just exact-taps 12 23`.
exact-taps ID STEP:
    ./tools/box.sh '{{precheck}} docs/cards/exact-ref.card && cargo build --release -p bloomery-gpu-gates --bin exact_ref && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin forced_probe && D=target/exact-taps/{{ID}}-{{STEP}} && mkdir -p $D && bash tools/ref/lease-hold.sh --card docs/cards/exact-ref.card -- ./target/release/exact_ref --prompt-id {{ID}} --steps {{STEP}} --act f64 --dump $D/f64 > $D/f64.log && bash tools/ref/lease-hold.sh --card docs/cards/exact-ref.card -- ./target/release/exact_ref --prompt-id {{ID}} --steps {{STEP}} --act ours --dump $D/ours > $D/ours.log && bash tools/ref/lease-hold.sh --card docs/cards/exact-ref.card -- ./target/release/exact_ref --prompt-id {{ID}} --steps {{STEP}} --act ik --dump $D/ik > $D/ik.log && grep -H "top5" $D/f64.log $D/ours.log $D/ik.log && bash tools/gpu-gate.sh forced_probe --prompt-id {{ID}} --step {{STEP}} --dump $D/engine --against $D/f64,$D/ours,$D/ik'

# 얇은 끝-끝 디코드 CLI(greedy, 토큰 하나씩, 프리필 커널 없음). `--time` 없이 토큰만 찍는 것은
# 평범한 실행이고, `--time`은 측정이라 임대가 필요하다 — time-gpu-generate가 그쪽이다.
generate *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate && ./target/release/generate {{ARGS}}'

# generate의 스텝당 ms(리드 전용): 기계 전역 임대 아래, 증인 블록 전후. 기록이 되는 것은 graph 모드의
# 푸터이고 eager 모드는 호스트 제출 경로의 값이다.
time-gpu-generate *ARGS:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate && bash tools/ref/time-gate.sh generate {{ARGS}} --time'

# P0b: 블록 0 FFN 융합 스파이크 — 융합 4런치가 op 8런치와 비트 동일한지(정확성 실행).
gate-gpu-p0b *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p0b && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p0b {{ARGS}}'

# P8a 블록 0 스텝(ctx_max 64에서 21노드 — fuse1 뒤; 한 구간보다 큰 캐시에서는 flash_merge가 붙어 22)의 재생 µs — 임대·증인, 리드 전용.
time-gpu-p8:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 && bash tools/ref/time-gate.sh gate_p8'

# P8 op별 프로파일(리드 전용): 스텝을 op 단위로 동기 분해한 µs 표(ops=그래프 노드 수와 동일) + sync_floor
# 보정 + refresh_params 호스트 시간 + 같은 프로세스의 그래프 재생 µs + 어텐션/FFN 분할. 임대·증인은 time-gate.sh 소유.
prof-gpu-p8:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 && bash tools/ref/time-gate.sh gate_p8 --profile'

# 커널 런치당 실제 비용(리드 전용): op별 표가 답할 수 없는 것 — 한 행의 net_us가 디바이스 시간인지
# 프로파일 자신의 제출 경로인지 — 를 두 갈래로 가른다. eager는 N런치 뒤 동기화 하나(호스트 제출과
# 디바이스 시간 중 큰 쪽), graph는 같은 N런치를 그래프 하나로 캡처해 재생(스텝의 그래프 노드가 실제로
# 무는 값). 빈 커널 touch가 둘의 바닥이고, f32 gemv는 행 수 셋에서 같은 프로세스로 잰다. 이어서 그룹 int8 GEMM의
# 팔들을 `gate_gemm --bench-kernels`로 잰다(ARGS, 예를 들어 `--bench-arm`은 이쪽으로 간다). 임대·증인은 time-gate.sh 소유.
bench-gpu-kernels *ARGS:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 --bin gate_gemm && bash tools/ref/time-gate.sh gate_p8 --bench-kernels && bash tools/ref/time-gate.sh gate_gemm --bench-kernels {{ARGS}}'

# The grouped GEMM's counters (ncu, A6000, under the lease, lead-only): gate_gemm --bench-kernels --bench-arm ARM (default
# gemm_q4k_moe_t4096: 128 experts of 768 x 2048 Q4_K, top-8, T = 4096) runs that one arm, and ncu takes 16 of its eager
# gemm_q4k launches after the 64-launch warm-up burst; the form and its levers are in the header of tools/ref/ncu-gpu.sh.
# Expected on main [derived, https://github.com/midagedev/rig-log/blob/main/docs/bloomery/research/q3next-design-report.md sections 3 and 6; not numbers of record]: L1TEX
# 55-65 % and the top unit, about 1.29e8 LSU data-pipe wavefronts a launch, issue-active 35-40 %, tensor 22-26 %, DRAM
# 10-15 %. Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built and the runner prints its command line.
ncu-gpu-gemm ARM='gemm_q4k_moe_t4096':
    ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_gemm; fi && BLOOMERY_NCU_FORM=gemm BLOOMERY_NCU_GEMM_ARM={{ARM}} bash tools/ref/ncu-gpu.sh'

# The V4.1 prompt projections' counters (ncu, A6000, under the lease, lead-only): one launch each of the joined qkv, q_b,
# wo_a heads and wo_b at m = 8 in the middle full chunk of a layer >= 2 of a P-token prompt (default 512). The launch skip
# comes from the newest nsys-gpu-ds41-prefill trace of the same P (run that first; it must be of this binary),
# and the profiled launches' names, grids, blocks and column counts are checked before the summary: per launch the
# block-step cycles and each unit's demand in cycles. The header of tools/ref/ncu-gpu.sh has the form. Under
# BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built and the runner prints its command line.
ncu-gpu-ds41-pp P='512':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41; fi && BLOOMERY_NCU_FORM=ds41pp BLOOMERY_NCU_PROMPT={{P}} bash tools/ref/ncu-gpu.sh'

# The Qwen3 prompt's counters (ncu, A6000, under the lease, lead-only): one KERNEL launch (gqa_prefill_flash, the one the
# form's table holds) in layer LAYER of a P-token prompt, in the process of depth-qwen3moe.sh's `<P>` arm (-n 1), at the
# card's own clock. The launch skip is derived from the code (tools/ref/q3pp.py plan: 24 at P = 4096) and the profiled
# launch is proved before the summary: the run's load and plan lines, the name, grid (2,048 x 128 at P = 4096) and the
# tensor pipe's HMMA count, which depends on the ubatch's rows and position (34,078,720). Then the clock, the step's
# cycles and each unit's demand beside them, and the stall composition. BLOOMERY_NCU_BIN=<absolute path> profiles
# another tree's generate_qwen3moe instead (nothing is built). Expected on main for gqa_prefill_flash at P = 4096 [derived,
# https://github.com/midagedev/rig-log/blob/main/docs/bloomery/research/q3tail-design-report.md 3.2 and 4; not numbers of record]: SM clock 1.46-1.55 GHz, tensor pipe 56-60 %,
# issue 0.36-0.40 a scheduler, 3,400-3,620 SM cycles a block-step. The header of tools/ref/ncu-gpu.sh has the form.
# Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built and the runner prints the derivation and its command line.
ncu-gpu-qwen3-pp P='4096' LAYER='24' KERNEL='gqa_prefill_flash':
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ] && [ -z "${BLOOMERY_NCU_BIN:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi && BLOOMERY_NCU_FORM=q3pp BLOOMERY_NCU_PROMPT={{P}} BLOOMERY_NCU_LAYER={{LAYER}} BLOOMERY_NCU_KERNEL={{KERNEL}} bash tools/ref/ncu-gpu.sh'

# 캡처된 그래프의 노드당 값, 비용 모형의 `c_node`(리드 전용): cnode_probe --time이 빈 `touch` 노드 784개와 504개짜리
# 그래프를 잡아, 그래프마다 노드 수와 재생 한 번의 기록을 먼저 확인한 뒤 `touch_us_min`·`touch_us_mean`·
# `touch_us_per_node`를 찍는다. 정확성만 볼 때는 `bash tools/gpu-gate.sh cnode_probe --check`. 임대·증인·A6000 고정은
# time-gate.sh가 쥔다.
time-gpu-cnode:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin cnode_probe && bash tools/ref/time-gate.sh cnode_probe'

# 호스트 → 카드 링크(리드 전용): `nvidia-smi topo -m`과 카드마다 PCI 주소·sysfs 경로·링크를 찍고, A6000에서 1 GiB pinned와
# pageable(https://github.com/midagedev/rig-log/blob/main/docs/bloomery/facts.md의 교정값), 1.6 GB pinned 복사 48회 연속(복사마다 GB/s의 퍼짐, 복사 중에 읽은 링크)을 잰다.
# BLOOMERY_CARD=both면 3090 단독과 두 카드 동시 복사(카드별 율과 합)도 잰다. 임대·증인·Xid 수는 tools/ref/h2d-pcie.sh가
# 쥐고, 카드는 docs/cards/pcie-a6000.card다. 예: `BLOOMERY_CARD=both BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/pcie-a6000.card' just time-gpu-h2d`.
time-gpu-h2d:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin h2d_probe && bash tools/ref/h2d-pcie.sh'

# 호스트 → 카드 복사가 호스트 티어의 DRAM 읽기에서 뺏는 몫(리드 전용): bench_v41_host --time을 단독으로, h2d_probe의
# pageable-loop 옆에서, staged-loop(스트리밍 설계의 pinned 링) 옆에서 라운드마다 번갈아 돌린다. 복사는 벤치가 비워 둔 코어에
# 고정하고, 라운드마다 두 율과 k = (U_alone − U_with) / C_with를 찍는다. ARGS는 tools/ref/dma-dram-share.sh로 간다(--arms는
# 벤치의 팔을 그대로 넘기고 --rounds, --seconds, --chunk, --fill-threads …). 카드는 docs/cards/dma-dram-share.card다.
# 예: `BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/dma-dram-share.card' just time-dma-dram --arms engine:6`.
time-dma-dram *ARGS:
    BLOOMERY_MODEL=${BLOOMERY_MODEL:-deepseek41} ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-model --bin bench_v41_host && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin h2d_probe && bash tools/ref/dma-dram-share.sh {{ARGS}}'

# 유휴 상태 A/B(리드 전용): 가장 깊은 cpuidle 상태(이 박스의 C2, 깨는 데 18 µs)를 모든 CPU에서 켠 팔과 끈 팔을
# 한 임대 안에서 번갈아 잰다. 명령은 A6000에 고정돼 라운드마다 팔 순서를 바꿔 돌고, 원래 값은 모든 종료 경로에서
# 되돌린다. 예: `just cstate-ab 6 env BLOOMERY_HYBRID_NL=32 target/release/generate -n 64 --time`.
cstate-ab ROUNDS *CMD:
    ./tools/box.sh '{{precheck}} && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate && bash tools/ref/cstate-ab.sh {{ROUNDS}} {{CMD}}'

# P8: 조립된 디코드 스텝(블록 0부터)을 덤프와 대조 — 엔진 자신의 탭.
gate-gpu-p8 *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p8 {{ARGS}}'

# MoE 융합(P8 준비): 전문가 여섯의 gate·up·swiglu 한 런치 + 결합 한 런치가 op 경로와 비트 동일한지(정확성 실행).
gate-gpu-moe:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_moe_fused && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_moe_fused'

# V4.1 라우터와 전문가(B4 J·K): 라우터, 라우팅·공유 전문가의 클램프 SwiGLU, combine을 우리 규칙과 V4.1 ik 덤프
# (5토큰 세트와 디코드 스텝 세트)에 층마다 대조한다. 클램프는 합성 입력으로 한계 너머까지 따로 본다.
gate-gpu-ds41-moe:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-ds41-moe && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_moe && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_moe'

# V4.1 MoE 조각(B5 1단계, chain G1): step4·d1n 세트의 40층 전부에서 라우터·카드 expert·공유 expert·호스트 티어 합류·combine·
# HC_POST(+접기)를 한 프로세스 안에서 돈다 — op 경로와 비트 동일, 라우터 id는 근접 동률 빼고 정확, combine·스트림·접기는 op 밴드에서
# 옮겨 온 반올림 한계 안의 예측 차이로 덤프와 대조, 재생 = eager(층마다 호스트가 서비스), 종류별 노드 수.
gate-gpu-ds41-chain-ffn:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-ds41-chain-ffn && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_chain_ffn && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_chain_ffn'

# P7 뒤 절반: 블록·종단 게이트 하네스의 자기 검증(호스트 전용 — 두 오라클 사이의 알려진 거리를 재현해야 한다).
gate-gpu-block:
    ./tools/box.sh 'cargo build --release -p bloomery-gpu-gates --bin gate_block && bash tools/host-gate.sh gate_block'

# P10: 가중치 상주 — 모델 파일의 모든 텐서를 커널이 먹는 디바이스 형식으로 올린다(정확성 실행, 측정 아님).
gate-gpu-p10:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p10 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p10'

# V4.1 적재 게이트 ②: 배치 계획이 카드마다 두는 세그먼트를, 이름으로 찾은 카드에 올리고 계획과 바이트 단위로 대조한다.
# 할당기 반올림도 계획의 항과 바이트까지 같아야 한다. 정확성 실행이지 측정이 아니다. 계획 a는 --plan a.
# 착륙 묶음이 고르지 않는 opt-in이다(gate-* 레시피가 아니라 `just affected`에 잡히지 않는다). 호스트 세트의 populate·lock
# 경로나 두 카드 staging이 바뀌면 이름으로 돌린다 — 계획 (b)가 트리에서 유일한 두 카드 적재다. `--plan bp`는 계획 (b′)
# (A6000은 계획 (a), 3090은 전문가 층)이고, 드래프트 예약을 위해 프로필의 DSPARK_MODEL을 `--draft`로 넘긴다.
[group('solo')]
[group('v41-load')]
stage-gpu-load-v41 *ARGS='--plan b':
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/real-only.sh stage-gpu-load-v41 && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_load_v41 && __d=$(. tools/ref/ref-paths.sh && dspark_model) && case " {{ARGS}} " in *" --plan bp "*) set -- --draft "$__d" ;; *) set -- ;; esac && bash tools/gpu-gate.sh gate_load_v41 {{ARGS}} "$@"'

# 같은 게이트 ②의 호스트 절반: 계획 (b)의 호스트 세그먼트 전부(유도 약 196 GB)를 샤드 매핑째 잠근다. 리드 전용이고,
# RAM을 크게 쓰는 트랙이 없을 때만 돈다.
# stage-gpu-load-v41처럼 opt-in이고, 같은 변경에서 이름으로 돌린다.
[group('solo')]
[group('v41-load')]
stage-gpu-load-v41-lock:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/real-only.sh stage-gpu-load-v41-lock && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_load_v41 && BLOOMERY_HOST_LOCK=1 bash tools/gpu-gate.sh gate_load_v41 --plan b --lock'

# qdot 커널률(리드 전용): ik 자신의 x4 커널과 Rust qdot-rate를 같은 CPU 임대 안에서 한 코어에 고정해 번갈아 돈다.
# ik 하네스는 build-ref가 짓는다. 인자는 라운드 수(기본 3).
measure-qdot-rate *ARGS:
    ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-qdot --bin qdot-rate && bash tools/ref/qdot-rate.sh {{ARGS}}'

# The qdot MT streaming ceiling (lead-only): qdot-rate-mt inside the machine-wide CPU lease, with
# the witness blocks around it. The shapes are the binary's SHAPES table, keyed by (qtype, k);
# ARGS go to the binary (--granularity, --only TYPE[,TYPE...], --iq1s-generic for the IQ1_S
# pre-hand-kernel arm).
measure-qdot-rate-mt *ARGS:
    ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-qdot --bin qdot-rate-mt && . tools/ref/ref-paths.sh && . tools/ref/lease.sh && WITNESS=(head loadavg pressure-cpu pressure-io gpus lock-holder cpu cpu-freq) && lease_take && { witness pre || exit $?; rc=0; lease_bounded 3600 target/release/qdot-rate-mt {{ARGS}} || rc=$?; witness post || exit $?; lease_release; exit $rc; }'

# V4.1 호스트 expert 다리의 벤치(bench_v41_host) — 정확성 실행이다. 실제 V4.1 파일을 mmap해 층마다 엔진의 디스패치
# (gate+up 묶음 하나, swiglu를 품은 down 하나)를 돌리고, 층 여덟의 expert마다 gate·up·swiglu·down 여섯 행을 같은 바이트의
# f64 참조와 대조한다(밴드 1e-5). 층마다 샤드 지도와 작업 집합의 상주율도 찍는다. 모델은 Mac 쪽 BLOOMERY_MODEL이고
# 없으면 deepseek41이다. BLOOMERY_MODEL=qwen4exp면 Qwen3.8의 routed expert를 같은 벤치로 연다.
bench-cpu-v41-host-check:
    BLOOMERY_MODEL=${BLOOMERY_MODEL:-deepseek41} ./tools/box.sh 'bash tools/ref/real-only.sh bench-cpu-v41-host-check && cargo build --release -p bloomery-model --bin bench_v41_host && bash tools/host-gate.sh bench_v41_host --check'

# 같은 벤치의 시간(리드 전용): 스레드 8·16·24·30·32마다 팔 넷(엔진 모양 n_host 6·5·3, 행렬별 디스패치 6)의 토큰당 ms·GB/s,
# 디스패치 수, 상주율. CPU 임대·증인·낡은 바이너리 거부는 host-rate.sh가 쥔다. 조용한 틈에 돈다 — 빌드가 도는 동안 잰
# 수는 증인이 받지 않는다. 모델은 위 레시피와 같이 BLOOMERY_MODEL이고 없으면 deepseek41이다.
time-cpu-v41-host *ARGS:
    BLOOMERY_MODEL=${BLOOMERY_MODEL:-deepseek41} ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-model --bin bench_v41_host && bash tools/ref/host-rate.sh {{ARGS}}'

# The NVMe tier's reader (lead-only): probe_nvread over Qwen3.8's routed experts, cold, at k = 1, 2, 3 experts a
# batch from distinct layers and over one layer's range from an id, each read by the page-cache path (WILLNEED, then
# POPULATE_READ split across threads) and by O_DIRECT preads of the same runs, each batch dropped and checked cold by
# mincore before and resident after. The rows' header is the probe's, the lease, the witness blocks with the drive's
# read counters, the drive's queue and the reconcile line are tools/ref/nvtier-read.sh's; the card is
# docs/cards/nvtier-read.card. ARGS go to the runner (--rounds, --seed, --range-from, then the probe's own: --batches,
# --threads, ...). Example: `BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/nvtier-read.card' just time-nvread`.
time-nvread *ARGS:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-gpu-gates --bin probe_nvread && bash tools/ref/nvtier-read.sh {{ARGS}}'

# 2026-09-20 사고(q_nope2 무한루크가 gate-mt를 매달아 병렬 에이전트 둘을 '무활동'으로 죽임)의
# 보강. 이 트랙 원격 디렉터리 아래 실행 파일을 물고 있는 고아 프로세스를 찾아 죽인다.
# 고르는 축은 /proc/<pid>/exe이지 cmdline이 아니다 — `pgrep -f "<dir>/target"`은 그 패턴을
# 자기 argv에 든 셸(스캔하는 셸 자신, ssh 핸들러)도 같이 고른다. 자신과 조상 pid는 접두
# 비교 전에 제외한다. 죽이는 것은 TERM → 5초 → KILL. 목록만 보려면 `just box-gc --dry-run`.
# 이 트랙 원격 디렉터리 아래 실행 파일을 문 고아 프로세스를 죽인다. 병렬 트랙 시작·끝에 한 번씩.
# 시팅의 가드를 지나간다(BLOOMERY_BOX_READONLY=1): 아무것도 빌드하지 않고, 찾는 고아가 임대를 쥔 그 프로세스일 수 있다.
# 읽기는 아무것도 동기화하지 않으므로 스크립트를 stdin으로 보낸다: 이 트리의 box-gc.sh가 원격 디렉터리에서 있는 그대로 돈다.
# 한 번도 동기화하지 않은 트랙에는 원격 디렉터리가 없다(box.sh의 66). 거기서 돈 것이 없으니 거둘 것도 없다.
box-gc *ARGS='--kill':
    #!/usr/bin/env bash
    set -uo pipefail
    BLOOMERY_BOX_READONLY=1 ./tools/box.sh 'bash -s -- {{ARGS}}' < tools/box-gc.sh
    rc=$?
    if [ "$rc" = 66 ]; then echo "box-gc: this track has no remote directory on the box (box.sh rc 66): nothing ran there, nothing to collect"; rc=0; fi
    exit "$rc"

# Dry run of the reference-set dataset's publisher on the box: every `fx_` set's family, files, bytes and sha256 under $BLOOMERY_DATA,
# the sets left out and why. Writes into /tmp on the box only, uploads nothing, opens no card, builds nothing; it syncs the script, so it
# waits for a sitting like any box command (tools/refset-publish.sh).
refset-publish-dry:
    ./tools/box.sh 'bash tools/refset-publish.sh --dry-run --out /tmp/refset-publish-dry'

# The hold on this side: `just box-hold up <owner> [reason]` makes ~/.cache/bloomery/box-hold/<owner>, `just box-hold down <owner>` removes it. While
# it is up, a tools/box.sh call that is not a read (BLOOMERY_BOX_READONLY=1) and not that owner's (BLOOMERY_HOLD_OWNER) ends 75 before it syncs or
# sends anything. The rule is tools/box.sh's header; this recipe reaches no box.
box-hold ACTION OWNER *REASON:
    ./tools/box.sh --hold {{quote(ACTION)}} {{quote(OWNER)}} {{quote(REASON)}}

# 박스에 남은 트랙 디렉터리를 로컬 워크트리와 대조한다. 인자 없이 목록, `just box-tracks --remove NAME…`로 준 이름만 삭제(그때도 stale일 때).
box-tracks *ARGS:
    ./tools/box-tracks.sh {{ARGS}}

# 이 Mac의 워크트리마다 크기와 착륙 상태를 보인다. 인자 없이 목록, `just mac-disk clean|retire NAME…`로 준 이름만 정리(retire는 LANDED·PRUNABLE일 때만).
mac-disk *ARGS:
    python3 tools/mac-disk.py {{ARGS}}

# A round's worktree on this Mac (tools/wt.sh): added at BASE (default HEAD), main's CLAUDE.local.md copied in, target/ seeded from the nearest warm tree.
wt-new NAME BASE='':
    bash tools/wt.sh new {{NAME}} {{BASE}}

# Retire a round's worktree through tools/mac-disk.py retire (LANDED or PRUNABLE only), refused while a batch or static run is in flight.
wt-retire NAME:
    bash tools/wt.sh retire {{NAME}}

# 1-4의 첫 tok/s. 같은 임대 안에서 ik를 같은 파일·같은 조건으로 한 번 더 잰다.
# The prebuilt Linux release (tools/release/build.sh's header: the checks it runs on the binaries before it packs):
# built on the box from a committed tree, with this commit as the binaries' --version. The tarball lands in the box
# remote dir's target/release-dist/. The box's platform picks the archive: the x86-64 one on the box, the GB10 one on a
# GB10 (run there with BLOOMERY_BOX_LOCAL=1: box.sh's remote guard reads /root's lease); BLOOMERY_ARCH naming the RTX 50
# series' arch builds its x86-64 archive on the box, beside the first (tools/release/build.sh's header names the archs).
# Before the box is touched, tools/release/carry.sh refuses a commit that lacks a fix tools/release/carry.tsv lists
# (here, where .git is).
release-build VERSION:
    test -z "$(git status --porcelain --untracked-files=no)" || { echo "release-build: the tree has uncommitted changes" >&2; exit 1; }
    ./tools/release/carry.sh
    ./tools/box.sh "bash tools/release/build.sh {{VERSION}} $(git rev-parse --short=12 HEAD)"

# The release sitting (tools/release/sitting.sh's header: the cells file, the order, the hold, the verdict): every cell of the
# release's table timed and checked on one binary inside one hold. CELLS is a cells file on the box (or under the synced tree);
# ARGS are the script's: `--binary portable --tarball target/release-dist/bloomery-<v>-linux-x86_64-cuda-sm86.tar.gz` for the
# release's verdict (the tarball's own bloomery-serve), `--binary tree` for an iteration's (this recipe builds the tree's
# bloomery-serve first, as depth-serve does), `--dry-run`, `--keep-going`. BLOOMERY_LEASE_CARD (BLOOMERY_BOX_ENV) names the
# sitting's prediction card; BLOOMERY_HOLD_OWNER (default rel) owns the box-side hold and passes box.sh's guard. The verdict
# lands beside the tarball (<tarball>.sitting; a tree stage's is target/release-dist/<version>-<tree>-tree.sitting), where
# tools/release/publish.sh reads it.
release-sitting VERSION CELLS *ARGS:
    ./tools/box.sh '{{precheck}} && case " {{ARGS}} " in *" --binary tree "*) cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve;; esac && bash tools/release/sitting.sh --version {{VERSION}} --cells {{CELLS}} --hold-owner {{env_var_or_default("BLOOMERY_HOLD_OWNER", "rel")}} {{ARGS}}'

build-decode:
    ./tools/box.sh 'cargo build --release -p bloomery-model --bin bloomery-decode'

measure-decode: build-decode
    ./tools/box.sh 'bash tools/ref/decode-measure.sh'

# 스레드 스윕. 같은 임대 안에서 1/8/16/32를 연달아 돌린다 — https://github.com/midagedev/rig-log/blob/main/docs/bloomery/plan.md의 "스레드 수는
# 재서 정한다"가 이것이고, 답은 이 티어가 대역폭에 닿는 지점이다.
measure-sweep: build-decode
    ./tools/box.sh 'NOCACHE=0 SWEEP="1 8 16 32 64" bash tools/ref/decode-measure.sh'

# 시간 귀속. 러너가 임대와 증인을 소유한다 — 프로파일 표도 측정이고, 옆에서 빌드
# 하나만 돌아도 site 간 비율이 흔들린다. 레벨 1(배분)과 2(단계)를 연달아 찍는다.
# 같은 임대 안 A/B: `just ab-decode bloomery-<track> ...` (각 트리는 미리 build-decode).
# 인자는 박스 `~/repo/` 아래 **디렉터리 이름**이다 — 맥 절대경로를 줘도 basename으로 바꾼다.
# 맥 셸의 BLOOMERY_AB_ROUNDS·BLOOMERY_AB_ENVS·BLOOMERY_AB_IK는 ssh를 그냥 넘지 않으므로 여기서
# 원격 명령줄 앞에 K=V로 실어 보낸다. BLOOMERY_AB_ENVS는 작은따옴표로 싸서 넘기므로 값 안의
# 작은따옴표 하나가 그 인용을 깬다 — `K=V;K2=V2` 형태(이 변수의 문법 전부)는 안전하다.
#   BLOOMERY_AB_ROUNDS=6 BLOOMERY_AB_ENVS="BLOOMERY_SPIN=0" just ab-decode bloomery-foo
# BLOOMERY_AB_DEPTH=d 는 깊이 d 의 프롬프트를 먼저 프리필한다(러너 머리말) — 어텐션 레버는 깊은 행에서만 보인다.
ab-decode *DIRS: build-decode
    #!/usr/bin/env bash
    set -euo pipefail
    names=
    for d in {{DIRS}}; do names="$names $(basename "$d")"; done
    ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS} ${BLOOMERY_AB_DEPTH:+BLOOMERY_AB_DEPTH=$BLOOMERY_AB_DEPTH} ${BLOOMERY_AB_ENVS:+BLOOMERY_AB_ENVS='$BLOOMERY_AB_ENVS'} ${BLOOMERY_AB_IK:+BLOOMERY_AB_IK=$BLOOMERY_AB_IK} bash tools/ref/ab-decode.sh$names"

# The model load's wall, one start per arm per round under the lease, each stopped at its `load phases` record (lead-only):
# the arms are trees under ~/repo/ on the box whose generate_ds41 was built there (as time-gpu-ds41 builds it), the first
# the base; the model file is evicted from the page cache before each timed start, `--warm` leaves it. tools/ref/load-ab.sh's
# header has the rows and the paired effects. The card and the rounds go through BLOOMERY_BOX_ENV:
#   BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/cflra-load.card BLOOMERY_AB_ROUNDS=3' just load-ab bloomery-base bloomery-new
load-ab *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && bash tools/ref/load-ab.sh {{ARGS}}'

# The Mac shell's BLOOMERY_PROFILE_DEPTH and BLOOMERY_PROFILE_LEVELS ride in front of the remote command, as in
# ab-decode; lever env goes through BLOOMERY_BOX_ENV:
#   BLOOMERY_BOX_ENV="BLOOMERY_FLASH_SEGMENTS=8" BLOOMERY_PROFILE_DEPTH=1024 just measure-profile
measure-profile: build-decode
    ./tools/box.sh "${BLOOMERY_PROFILE_DEPTH:+BLOOMERY_PROFILE_DEPTH=$BLOOMERY_PROFILE_DEPTH} ${BLOOMERY_PROFILE_LEVELS:+BLOOMERY_PROFILE_LEVELS='$BLOOMERY_PROFILE_LEVELS'} bash tools/ref/profile-measure.sh"

# 디코드 스텝의 스레드별 perf 표(리드 전용): 임대·증인 아래, 프리필이 끝난 뒤 pid에 붙어 cpu-clock으로 뜬다.
# `perf report --tid`는 메인 스레드의 dot 커널을 통째로 빼고 스핀만 보여 줬으므로(perf 6.8) `perf script`로 집계한다.
# PERF_SECS(기본 2.0)·PERF_TOP·PERF_ANNOTATE·PERF_CALLERS·PERF_KEEP.
perf-decode *ARGS: build-decode
    ./tools/box.sh 'bash tools/ref/perf-decode.sh {{ARGS}}'

# 참조 하네스(ggml에 링크하는 C++). 진실값과 기준 속도의 출처다.
# build-qdot-ref.sh는 ik의 커널 테이블까지 링크하는 x4 하네스 열을 짓고 참조 하네스 다섯을
# **실행까지** 한다 — gate-qdot의 hw 테스트 일곱이 읽는 $BLOOMERY_DATA/ref/의 덤프
# (*-ik-dot.txt 다섯과 q5_K의 to_float 행)가 그 산출물이다. q5_K는 V2-Lite에 없어서 V4.1 첫
# 샤드를 읽는다. 나머지 다섯(*_rate)은 빌드만 한다: 실행은 측정이다.
build-ref:
    ./tools/box.sh 'bash tools/ref/build-qdot-ref.sh'

# The IQ4_XS reference alone: its own ik tree at the commit of tools/ref/iq4xs-ik.pin (made once by
# tools/ref/ik-pin-tree.sh, CPU only), iq4xs_ref linked to it, and the dump with the meta that names the commit.
# gate-qdot's two IQ4_XS clauses read them; no other dump is touched.
build-ref-iq4xs:
    ./tools/box.sh 'bash tools/ref/build-qdot-ref.sh iq4xs_ref'

# The V4.1 file's r8 sidecar (model::r8file): builds r8conv, then tools/ref/r8-sidecar.sh converts and
# verifies it under the CPU lease — 155.7 GB written beside the source on /models. A tool run, not a
# gate; the user decides when. ARGS: `verify` checks an existing sidecar again without converting.
r8-sidecar *ARGS:
    ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-model --bin r8conv && bash tools/ref/r8-sidecar.sh {{ARGS}}'

# 의존성 감사. cuda-oxide가 rev로 고정돼 있는지가 핵심이다.
deny:
    ./tools/box.sh 'cargo deny check'

# 오라클 계측기. 1-2부터의 게이트가 읽는 참조 텐서를 $BLOOMERY_DATA/ref/에 만든다.
# ik 빌드가 바뀌면 다시 돌린다 — 참조는 그 빌드의 출력이다.
build-ref-dump:
    ./tools/box.sh 'bash tools/ref/build-dump.sh'

dump-ref:
    ./tools/box.sh 'bash tools/ref/dump.sh'

# GPU 엔진의 오라클: 같은 계측기를 -ngl 99로, $BLOOMERY_DATA/ref_cuda/에. CPU 참조는 건드리지 않는다.
dump-ref-cuda:
    ./tools/box.sh 'BLOOMERY_REF_BACKEND=cuda bash tools/ref/dump.sh'

# V4.1 오라클(B0c): 같은 계측기를 deepseek41 프로필로 돌려 $BLOOMERY_DATA/ref_deepseek41/에 쓴다. CPU 백엔드만
# 쓴다. 프로필의 --defer-experts가 파일 세트 전체의 populate를 건너뛰어 그래프가 닿는 것만 읽는다(page cache가
# 찬 상태에서 1.27 GB; 콜드 읽기량은 재지 않았다). VARIANT(step4, d1, d2와 접미사 -unfused, -every-node)는
# 조용한 프리필 뒤 디코드 한 스텝을 자기 세트에 덤프한다 — models/deepseek41.sh.
dump-ref-v41 *VARIANT:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/dump.sh {{VARIANT}}'

# qwen3moe(Qwen3-30B-A3B-2507) 오라클: 같은 계측기를 qwen3moe 프로필로 돌려 $BLOOMERY_DATA/ref_qwen3moe/에 쓴다(CPU,
# 5토큰). VARIANT(step4, d1k, d4k와 접미사 -every-node)는 조용한 프리필 뒤 디코드 한 스텝을 자기 세트에 덤프한다 —
# models/qwen3moe.sh. d1k·d4k는 $BLOOMERY_DATA/qwen3moe/corpus-prose.ids의 앞 1,025·4,097개 id를 읽는다(sha256 고정).
dump-ref-qwen3moe *VARIANT:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/ref/dump.sh {{VARIANT}}'

# 같은 5토큰의 ik CUDA 덤프(-ngl 99, box env가 핀한 3090)를 $BLOOMERY_DATA/ref_cuda_qwen3moe/에. 가중치 17.5 GiB가 카드에
# 올라가므로 3090에 다른 프로세스가 없을 때 친다.
dump-ref-qwen3moe-cuda:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'BLOOMERY_REF_BACKEND=cuda bash tools/ref/dump.sh'

# Qwen3.6-35B-A3B(qwen35moe) 오라클: ik가 lmstudio Q4_K_M 파일을 CPU로 돌린 노드 덤프(5토큰 배치 세트).
# VARIANT를 주면 조용한 프리필 뒤 디코드 한 스텝을 제 세트로 뜬다(step4, d1k, u1s4, u1s5와 -every-node —
# tools/ref/models/qwen35moe.sh). CPU 임대 아래서 돌므로 카드(docs/cards/q35oracle-dump.card)가 있어야 한다.
dump-ref-qwen35moe *VARIANT:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'bash tools/ref/dump.sh {{VARIANT}}'

# The Qwen3.5 dense (qwen35, Clef's backbone) hidden-state oracle: hidden_ref linked against llama.cpp mainline's build
# (tools/ref/build-hidden.sh), then llama.cpp's result_norm of every position of the first 64, 600 and 4,096 prose ids
# into $BLOOMERY_DATA/ref_qwen35_hidden_p{64,600,4096}/ (the 27B Q4_K_M file; refset family hidden-qwen35) and
# $BLOOMERY_DATA/ref_qwen35_flashq8_hidden_p{64,600,4096}/ (Clef-Flash at bartowski's Q8_0; hidden-qwen35-flash-q8) and
# $BLOOMERY_DATA/ref_qwen35_lev_hidden_p{64,600,4096}/ (lev at ggml-org's Q4_K_M, the 4B geometry with a tied head; hidden-qwen35-lev),
# tools/ref/hidden.sh. Each dump puts its whole file on the card box.sh puts in view; no lease (a functional oracle). `--cpu-twin` writes each
# set's CPU twin into <set>.cpu/ instead (no card, 16 threads): the oracle's own floor the gate's bands come from.
build-ref-hidden:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'bash tools/ref/build-hidden.sh'

dump-hidden-qwen35 *SETS:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'bash tools/ref/hidden.sh {{SETS}}'

# The dense Qwen3.8-27B (qwen38d: general.architecture qwen35, lmstudio's Q4_K_M) logits oracle's binary: dense_logits_ref
# (tools/ref/dense_logits_ref.cpp) linked against llama.cpp mainline's build in the qwen35 profile's tree, which
# tools/ref/emb2/oracle-tree.sh proves is the pinned commit's build (nothing is built in the tree); into target/release of the
# box's directory, where tools/gpu-gate.sh runs it (tools/ref/build-dense-logits.sh).
build-dense-logits-ref:
    BLOOMERY_MODEL=qwen38d ./tools/box.sh 'bash tools/ref/build-dense-logits.sh'

# The qwen38d logits oracle sets (tools/ref/dense-logits.sh): mainline's logits after the first 64 and 4,096 prose ids and after
# each of the 32 ids that follow them (teacher forced), into $BLOOMERY_DATA/ref_qwen38d_logits_p{64,4096}/ (refset family
# logits-qwen38d). Each dump puts the whole file on the card (the A6000 unless BLOOMERY_CARD says otherwise; its gate lock); no
# lease (a functional oracle). `--cpu-twin` writes each set's CPU twin into <set>.cpu/ instead (no card, 16 threads): the oracle's
# own floor a gate's band is derived from (`python3 tools/ref/dense-logits.py floor <set>`).
dump-logits-qwen38d *SETS:
    case " {{SETS}} " in *" --cpu-twin "*) card=3090 ;; *) card=${BLOOMERY_CARD:-a6000} ;; esac; BLOOMERY_CARD=$card BLOOMERY_MODEL=qwen38d ./tools/box.sh 'bash tools/ref/dense-logits.sh {{SETS}}'

# The qwen38nv logits oracle sets (tools/ref/dense-logits.sh, profile tools/ref/models/qwen38nv.sh): mainline's logits on the GGUF twin
# of the Qwen3.8-27B NVFP4 fixture cut after the first 64 and 4,096 prose ids and after each of the 32 ids that follow them (teacher
# forced), into $BLOOMERY_DATA/ref_qwen38nv_logits_p{64,4096}/ (refset family logits-qwen38nv). The same dense_logits_ref binary as
# build-dense-logits-ref, and the twin's size and sha256 pinned in tools/ref/dense-logits.py TARGETS first. Each dump puts the whole
# file on the card (the A6000 unless BLOOMERY_CARD says otherwise; its gate lock); no lease (a functional oracle). `--cpu-twin`
# writes each set's CPU floor into <set>.cpu/ instead (no card, 16 threads), what the fixture gate's band is derived from
# (`python3 tools/ref/dense-logits.py floor <set>`).
dump-logits-qwen38nv *SETS:
    case " {{SETS}} " in *" --cpu-twin "*) card=3090 ;; *) card=${BLOOMERY_CARD:-a6000} ;; esac; BLOOMERY_CARD=$card BLOOMERY_MODEL=qwen38nv ./tools/box.sh 'bash tools/ref/dense-logits.sh {{SETS}}'

# Nanbeige4.3-33B (nanbeige43: general.architecture llama, bartowski's Q4_K_M) logits oracle's binary: the same dense_logits_ref as
# build-dense-logits-ref (one source, one mainline tree and pin; the device arch comes from the writer's target), built under this profile
# (tools/ref/models/nanbeige43.sh) into target/release of the box's directory (tools/ref/build-dense-logits.sh).
build-dense-logits-ref-nanbeige43:
    BLOOMERY_MODEL=nanbeige43 ./tools/box.sh 'bash tools/ref/build-dense-logits.sh'

# The nanbeige43 logits oracle sets (tools/ref/dense-logits.sh): mainline's logits after the first 64 and 4,096 prose ids of this
# vocabulary ($BLOOMERY_DATA/nanbeige43/corpus-prose.ids) and after each of the 32 ids that follow them (teacher forced), into
# $BLOOMERY_DATA/ref_nanbeige43_logits_p{64,4096}/ (refset family logits-nanbeige43). Each dump puts the whole file on the card (the
# A6000 unless BLOOMERY_CARD says otherwise; its gate lock); no lease (a functional oracle). `--cpu-twin` writes each set's CPU twin
# into <set>.cpu/ instead (no card, 16 threads): the oracle's own floor a gate's band is derived from
# (`python3 tools/ref/dense-logits.py floor <set>`).
dump-logits-nanbeige43 *SETS:
    case " {{SETS}} " in *" --cpu-twin "*) card=3090 ;; *) card=${BLOOMERY_CARD:-a6000} ;; esac; BLOOMERY_CARD=$card BLOOMERY_MODEL=nanbeige43 ./tools/box.sh 'bash tools/ref/dense-logits.sh {{SETS}}'

# Clef-Flash image input, the oracle's binary: dump_mtmd (tools/ref/clefvis/dump_mtmd.cpp) linked against llama.cpp mainline's libmtmd in the qwen35 profile's tree (tools/ref/clefvis/build-clefvis.sh).
build-clefvis-ref:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'bash tools/ref/clefvis/build-clefvis.sh'

# Clef-Flash image input, the oracle sets (tools/ref/clefvis/clefvis.sh): mainline mtmd's preprocess (ref_clefvis_preproc), tower taps (ref_clefvis_taps), hidden states of three prompts (ref_clefvis_hidden_c*, with the prose control ref_clefvis_prose_c* and the bf16-rows control ref_clefvis_bf16rows_c*) into $BLOOMERY_DATA; SETS names base sets, `--cpu-twin` writes the CPU twins (<set>.cpu) and `--ids` the prompts' ids by the release's tokenizer (both without a card), `--ref-d` set D: the release's answers on the A6000. The card runs go through the A6000's gate lock; no lease.
dump-ref-clefvis *ARGS:
    case " {{ARGS}} " in *" --ids "*|*" --cpu-twin "*) card=3090 ;; *) card=${BLOOMERY_CARD:-a6000} ;; esac; BLOOMERY_CARD=$card BLOOMERY_MODEL=qwen35 ./tools/box.sh 'bash tools/ref/clefvis/clefvis.sh {{ARGS}}'

# Qwen3.6 / Qwen3.8 image input, the oracle's tree: llama.cpp mainline at 36a73916e in /home/user/llama.cpp-36a73916 (tools/ref/qvis/build-lcpp-qvis.sh; detached, pid, log and rc under /root/lcpp-qvis), beside Clef's 53ed051ce tree.
build-qvis-lcpp:
    ./tools/box.sh 'setsid -f bash tools/ref/qvis/build-lcpp-qvis.sh </dev/null >/dev/null 2>&1'

# MiniCPM-V 4.7 image input, the oracle's tree: llama.cpp mainline at 358a420e in /home/user/llama.cpp-358a420e, the same script as the Qwen tree (tools/ref/qvis/build-lcpp-qvis.sh) with the pin, tree and sentinel directory by environment; detached, pid, log and rc under /root/lcpp-mc47. Mainline gained `minicpmv4_7` at 69f201a20; the Qwen tree (36a73916e) has none.
build-mc47-lcpp:
    BLOOMERY_BOX_ENV='LCPP_BUILD_PIN=358a420ec4ed59b6b0be08b1d1f71dd478b99a99 LCPP_BUILD_TREE=/home/user/llama.cpp-358a420e LCPP_BUILD_DIR=/root/lcpp-mc47' ./tools/box.sh 'setsid -f bash tools/ref/qvis/build-lcpp-qvis.sh </dev/null >/dev/null 2>&1'

# Qwen3.6 / Qwen3.8 / GLM-5.3-Flash image input, the oracle's binary: dump_mtmd linked against that tree's libmtmd and libllama-common, with its chat mode (tools/ref/clefvis/build-clefvis.sh). One binary, $BLOOMERY_DATA/bin/dump_mtmd_qvis, serves all three seats; CLEFVIS_OUT=<dir> (in BLOOMERY_BOX_ENV) builds into another directory for a test.
build-qvis-ref:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'bash tools/ref/clefvis/build-clefvis.sh'

# Qwen3.6 (MODEL qwen35moe) or Qwen3.8 (qwen4exp) image input, the oracle sets (tools/ref/clefvis/clefvis.sh, tools/ref/qvis/cases.tsv): the tower's output end (ref_<arch>_vis_taps), hidden states of three chat prompts (…_hidden_c*, with the prose control …_prose_c* and the bf16-rows control …_bf16rows_c*) and 32 greedy decode steps after two of them (…_decode_c*) into $BLOOMERY_DATA; SETS names base sets, `--cpu-twin` writes the CPU twins (<set>.cpu) and `--ids` the chat ids of every request by llama-server's own path (…_chatids_*, both without a card). The card runs go through the card's gate lock (CLEFVIS_BOUND is the whole call's bound); no lease.
dump-ref-qvis MODEL *ARGS:
    case " {{ARGS}} " in *" --ids "*|*" --cpu-twin "*) card=3090 ;; *) card=${BLOOMERY_CARD:-a6000} ;; esac; BLOOMERY_CARD=$card BLOOMERY_MODEL={{MODEL}} ./tools/box.sh 'bash tools/ref/clefvis/clefvis.sh {{ARGS}}'

# MiniCPM-V 4.7 image input, the oracle's binary: dump_mtmd (tower, hidden and chat arms) linked against the 358a420e tree's libmtmd and libllama-common (tools/ref/clefvis/build-clefvis.sh, the minicpmv47 row of tools/ref/clefvis/profile.sh) into $BLOOMERY_DATA/bin/dump_mtmd_mc47; CLEFVIS_OUT=<dir> (in BLOOMERY_BOX_ENV) builds into another directory for a test.
build-mc47-ref:
    BLOOMERY_MODEL=minicpmv47 ./tools/box.sh 'bash tools/ref/clefvis/build-clefvis.sh'

# MiniCPM-V 4.7 image input, the oracle sets (tools/ref/clefvis/clefvis.sh, the minicpmv47 seat of profile.sh, images tools/ref/clefvis/images-minicpmv47.tsv, cases cases-minicpmv47.tsv, requests requests-minicpmv47.jsonl): the tiles' preprocess (ref_minicpmv47_vis_preproc) and the tower's graph nodes of every tile (ref_minicpmv47_vis_taps; a row <image>/t<k>/<node>, k = 0 the overview, 1.. the slices), the hidden states of three chat prompts (ref_minicpmv47_vis_hidden_mc1..3; result_norm of every position and the four position lanes) and, with `--ids`, the ids llama-server's chat path gives each request (ref_minicpmv47_vis_chatids_mc1..3, the ids files the hidden sets read) into $BLOOMERY_DATA; SETS names base sets, `--cpu-twin` writes the CPU twins (<set>.cpu, no card). The card run goes through the card's gate lock (CLEFVIS_BOUND is the whole call's bound); no lease.
dump-ref-mcvis *ARGS:
    case " {{ARGS}} " in *" --ids "*|*" --cpu-twin "*) card=3090 ;; *) card=${BLOOMERY_CARD:-a6000} ;; esac; BLOOMERY_CARD=$card BLOOMERY_MODEL=minicpmv47 ./tools/box.sh 'bash tools/ref/clefvis/clefvis.sh {{ARGS}}'

# EmbeddingGemma 2 (gemma-embedding2, its tower gemma4v), the oracle's tree: llama.cpp mainline at 6868a52cb in /home/user/llama.cpp-6868a52c, the same script as the Qwen tree (tools/ref/qvis/build-lcpp-qvis.sh) with the pin, tree, sentinel directory and targets by environment (llama-server and llama-tokenize besides the libraries); detached, pid, log and rc under /root/lcpp-emb2.
build-emb2-lcpp:
    BLOOMERY_BOX_ENV='LCPP_BUILD_PIN=6868a52cb337450658006ad2f9281c2eb63f1666 LCPP_BUILD_TREE=/home/user/llama.cpp-6868a52c LCPP_BUILD_DIR=/root/lcpp-emb2 LCPP_BUILD_TARGETS=llama,mtmd,llama-common,llama-server,llama-tokenize' ./tools/box.sh '[ "$(id -u)" = 0 ] || { echo "build-emb2-lcpp: this recipe requires root on the box (/root/lcpp-emb2 and /home/user); run build-lcpp-qvis.sh directly with writable LCPP_BUILD_DIR and LCPP_BUILD_TREE on a local machine" >&2; exit 2; }; setsid -f bash tools/ref/qvis/build-lcpp-qvis.sh </dev/null >/dev/null 2>&1'

# EmbeddingGemma 2, the oracle's binary: dump_emb2 (tools/ref/emb2/dump_emb2.cpp) linked against that tree's libmtmd and libllama (tools/ref/clefvis/build-clefvis.sh, the gemma-embedding2 row of tools/ref/clefvis/profile.sh) into $BLOOMERY_DATA/bin/dump_emb2; CLEFVIS_OUT=<dir> (in BLOOMERY_BOX_ENV) builds into another directory for a test.
build-emb2-ref: (build-embedding-ref "gemma-embedding2")

# EmbeddingGemma v1 links the common dumper against the same mainline tree as v2.
build-emb1-ref: (build-embedding-ref "gemma-embedding")

# The model profile owns the files and oracle tree; both encoders share this build.
build-embedding-ref MODEL:
    BLOOMERY_MODEL={{MODEL}} ./tools/box.sh 'bash tools/ref/clefvis/build-clefvis.sh'

# EmbeddingGemma 2, the oracle sets (tools/ref/emb2/emb2.sh; refset families in crates/refset/src/arch/gemma_embedding2) into $BLOOMERY_DATA: every text case's token rows, pooled vector and layer taps (ref_gemma_embedding2_text), the image cases' tower taps and rows (ref_gemma_embedding2_tower), llama-server's /v1/embeddings vectors of both (ref_gemma_embedding2_server), and the CPU twins of all three (.cpu). The text cases and their ids are the tokenizer oracle's gemma4 set: `just dump-ref-tokenizer gemma4` first. STEPS names some of text, tower, server, text.cpu, tower.cpu and server.cpu (none: all six). The card steps go through the card's gate lock (EMB2_BOUND is the call's and each step's bound); no lease.
dump-ref-emb2 *STEPS: (dump-ref-embedding "gemma-embedding2" STEPS)

# Gemma4a host mel from synthetic decoded PCM, using the existing pinned mainline tree.
dump-ref-emb2-mel:
    ./tools/box.sh 'bash tools/ref/emb2/mel.sh'

# Gemma4a soft rows from the pinned audio projector, using the E9 mel set.
dump-ref-emb2-audio:
    ./tools/box.sh 'bash tools/ref/emb2/audio.sh'

# Audio tower: bounded mel chunks, Conformer soft rows and clean/tapped equality.
gate-gpu-gemma4a:
    ./tools/box.sh 'bash tools/gate.sh --gpu --oxide --release -p bloomery-gpu-gemma4a --lib -- --include-ignored'
    ./tools/box.sh 'bash tools/gate.sh --oxide --release -p bloomery-gpu-gates --features gemma4a --bin gate_gemma4a'
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gemma4a --release --bin gate_gemma4a && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemma4a'

# Host-only audio preprocessing: mel bits, chunk boundaries and tower token counts.
gate-emb2-mel:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-vision --lib --test gemma4a -- --include-ignored --nocapture'

# EmbeddingGemma v1: text, server and text.cpu sets; no image tower.
dump-ref-emb1 *STEPS: (dump-ref-embedding "gemma-embedding" STEPS)

# The common EmbeddingGemma writer, selected by model architecture.
dump-ref-embedding MODEL *STEPS:
    BLOOMERY_MODEL={{MODEL}} ./tools/box.sh 'bash tools/ref/emb2/emb2.sh {{STEPS}}'

# GLM-5.3-Flash image input (mtmd `glm5v`), the oracle sets on the fixture tier (tools/ref/clefvis/clefvis.sh, the glm5next seat of profile.sh): the preprocess (fx_ref_glm5next_vis_preproc) and the tower's output nodes (…_taps), hidden states of three chat prompts (…_hidden_c*) and 32 greedy decode steps after two of them (…_decode_c*), the chat ids (…_chatids_*, with --ids), into $BLOOMERY_DATA. The text model is the fixture's copy in mainline's spelling, /models/fixtures-mainline/glm5next (tools/ref/gguf-respell.py); SETS names base sets, `--cpu-twin` writes the CPU twins and `--ids` the chat ids (both without a card). The card runs go through the card's gate lock; no lease.
dump-ref-glmv *ARGS:
    case " {{ARGS}} " in *" --ids "*|*" --cpu-twin "*) card=3090 ;; *) card=${BLOOMERY_CARD:-a6000} ;; esac; BLOOMERY_TIER=fixture BLOOMERY_CARD=$card BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/clefvis/clefvis.sh {{ARGS}}'

# Nemotron 3.5 Bolt image input, the oracle's binary: dump_mtmd (the tower arm) linked against the mainline tree cd0a5b6a1 (the nemotron_h_moe profile's, a CPU build) in /home/user/llama.cpp-mainline-bolt (tools/ref/clefvis/build-clefvis.sh, the nemotron_h_moe row of tools/ref/clefvis/profile.sh) into $BLOOMERY_DATA/bin/dump_mtmd_l5, the binary every seat of that tree shares; CLEFVIS_OUT=<dir> (in BLOOMERY_BOX_ENV) builds into another directory for a test, and a run of the sets then names it the same way.
build-bolt-ref:
    BLOOMERY_MODEL=nemotron_h_moe ./tools/box.sh 'bash tools/ref/clefvis/build-clefvis.sh'

# Nemotron 3.5 Bolt image input (mtmd `nemotron_omni`, the RADIO tower), the oracle sets (tools/ref/clefvis/clefvis.sh, the nemotron_h_moe seat of profile.sh, images tools/ref/clefvis/images-bolt.tsv; refset families in crates/refset/src/arch/nemotron_h_moe/vis.rs): the preprocess (ref_nemotron_h_moe_vis_preproc) and the tower's per-block nodes (ref_nemotron_h_moe_vis_taps) into $BLOOMERY_DATA. The oracle tree has no CUDA backend, so a set is the CPU run (`--cpu`, the default of this recipe): no card lock, no twin, no lease; the 24 GB text file is read for its vocabulary only.
dump-ref-boltvis *ARGS:
    BLOOMERY_MODEL=nemotron_h_moe ./tools/box.sh "bash tools/ref/clefvis/clefvis.sh --cpu {{ARGS}}"

# GLM-5.3-Flash(glm5next) 오라클: 같은 덤프 도구를 glm5next 프로필로, ik를 CPU로 돌려 5토큰 배치 세트를
# $BLOOMERY_DATA/ref_glm5next/에 뜬다. VARIANT(step4, d1k, d3kdsa, d16kdsa와 -every-node 접미사)를 주면 조용한 프리필 뒤 디코드 한 스텝을
# 제 세트로 뜬다 — models/glm5next.sh. d1k는 $BLOOMERY_DATA/glm5next/corpus-prose.ids의 첫 1,025개 id를 읽는다(sha256 핀).
# 덤프마다 199.7 GB 샤드 집합 전체를 CPU 임대 아래서 올리므로 카드가 있어야 한다
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/glmref-dump.card').
dump-ref-glm5next *VARIANT:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/real-only.sh dump-ref-glm5next && bash tools/ref/dump.sh {{VARIANT}}'

# 같은 5개 id를 ik의 CUDA 백엔드로 $BLOOMERY_DATA/ref_cuda_glm5next/에, TF32는 끈다(NVIDIA_TF32_OVERRIDE=0: 없으면 이 모델에서
# ik CUDA의 top-1 일치가 0.896이었다, rig-log 2026-09-15). 이대로는 돌 수 없다: dump.sh의 -ngl 99가 199.7 GB 파일을 카드 한 장에
# 올리므로 CUDA 세트에는 배치 계획이 먼저 필요하다. GPU 소비자는 아직 없다.
dump-ref-glm5next-cuda:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/real-only.sh dump-ref-glm5next-cuda && NVIDIA_TF32_OVERRIDE=0 BLOOMERY_REF_BACKEND=cuda bash tools/ref/dump.sh'

# The GLM HF map oracle (tools/ref/glm5next-hf-map.py) into $BLOOMERY_DATA/ref-glm5next-hf-map, by HTTP range reads of
# nvidia/GLM-5.3-Flash-NVFP4 and unsloth's GLM-5.3-Flash UD-Q4_K_XL at the revisions the family pins: the checkpoint's shard
# headers and small F32/BF16 tensors, the GGUF's tensor table and the same tensors. About 30 MB; no lease, no card.
# `just gate-glm5next-hf-map` reads it. The writer refuses an existing set directory.
dump-ref-glm5next-hf-map:
    ./tools/box.sh 'bash tools/ref/real-only.sh dump-ref-glm5next-hf-map && python3 tools/ref/glm5next-hf-map.py "$BLOOMERY_DATA/ref-glm5next-hf-map"'

# The Qwen3.8 NVFP4 oracle (tools/ref/qwen4exp-nvfp4-oracle.py, refset family hf-qwen4exp-nvfp4) into
# $BLOOMERY_DATA/hf-qwen4exp-nvfp4/: the forward of nvidia/Qwen3.8-Flash-Next-NVFP4 at the family's revision in transformers'
# qwen4_exp, f32 on the CPU, with the weights the prepared split holds, on the batch set's ids and on IDS, a file of the 3,001
# ids of ref_qwen4exp_d3k (its `# tokens` line). One decoder layer is resident at a time (about 13 GiB of the folder read a
# layer); no card. QWEN4EXP_NVFP4_PY (in BLOOMERY_BOX_ENV) names a Python with torch, numpy, safetensors and transformers at the
# script's TRANSFORMERS_REV. `just gate-gpu-qwen4exp-nvfp4` reads the sets.
dump-ref-qwen4exp-nvfp4 IDS *ARGS:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/real-only.sh dump-ref-qwen4exp-nvfp4 && "${QWEN4EXP_NVFP4_PY:-python3}" tools/ref/qwen4exp-nvfp4-oracle.py --model /models/nvidia__Qwen3.8-Flash-Next-NVFP4/fc694b54fb0174e0913e6adf86691ef85a4ead47 --out "$BLOOMERY_DATA/hf-qwen4exp-nvfp4" --prose {{IDS}} {{ARGS}}'

# ik's node dump of a family's fixture file (glm5next, qwen4exp, qwen35moe, deepseek41), into $BLOOMERY_DATA/fx_<the real twin's set>
# (refset's fixture families): the same dumper and profile as the family's real recipe, the fixture first shard as the model,
# the fixture file's `# fixture` line in the header (refset-check --fixture-line), and in the long variants (d1k, d3k, d3kdsa,
# d16kdsa) ik's routes for the prefill's positions (dump_ref --prefill-routes). No VARIANT is the batch set; a VARIANT is the
# real recipe's (step4, step4-every-node, d1k, d3k, d3kdsa, d16kdsa; deepseek41: step4-every-node, d1n-every-node,
# d1-every-node, d2-every-node). BLOOMERY_TIER=fixture is required (in the real tier: exit 64; the real sets have their own
# recipes). Under the CPU lease and a card (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/fxoracle-dump.card'); DUMP_DRY=1 in
# that variable prints the dump and stops before the lease. Under BLOOMERY_FIXTURE_VARIANT=<tag> (tools/fixture-variants.tsv; not the
# VARIANT argument) the model is that fixture variant's file and the sets are fx_<tag>_<the real twin's set>, beside the base's.
dump-ref-fixture FAMILY *VARIANT:
    BLOOMERY_MODEL={{FAMILY}} ./tools/box.sh '[ "${BLOOMERY_TIER:-real}" = fixture ] || { echo "dump-ref-fixture: this recipe needs BLOOMERY_TIER=fixture, got ${BLOOMERY_TIER:-real}; it dumps the fixture files, and the real sets have their own recipes (exit 64)" >&2; exit 64; } && bash tools/ref/dump.sh {{VARIANT}}'

# The fixture variant TAG of FAMILY's fixture (tools/fixture-variants.tsv: glm5next iq4xs): the fixture requantized by ik's
# llama-quantize to the tag's type map, every other tensor bit for bit, into /models/fixture-variants/TAG/<fixture dir>/
# (tools/ref/fixture-requant.sh has the steps and the refusals; `fixture variant-check` holds the written file to its source;
# an existing output is refused). The variant is then run with BLOOMERY_FIXTURE_VARIANT=TAG under BLOOMERY_TIER=fixture, and its
# oracle sets are dumped by the recipes below under the same two variables (their sets carry the tag: fx_TAG_...).
fixture-requant FAMILY TAG:
    env -u BLOOMERY_FIXTURE_VARIANT BLOOMERY_MODEL={{FAMILY}} BLOOMERY_TIER=fixture ./tools/box.sh 'bash tools/ref/fixture-requant.sh {{FAMILY}} {{TAG}}'

# The MTP draft oracles' dumper, GLM-5.3-Flash's and Qwen3.8-Flash-Next's: dump_mtp linked against the ik tree that
# carries the glm5next MTP graph on upstream's qwen4exp one (the glm5next profile's GLM_MTP_IK at GLM_MTP_SHA, checked
# clean); `cmake --build` of its libllama and libcommon first (nothing to do when current). Under the CPU lease:
# BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/glmmtpref-dump.card'.
build-ref-dump-mtp:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/build-dump-mtp.sh'

# Every node ik's MTP context computes while the target decodes 64 positions after the first 64 prose ids, one draft
# token a round, into $BLOOMERY_DATA/ref-mtp/prose64_n64_k1/ (refset family mtp-glm5next). ik on the CPU, CUDA hidden,
# under the CPU lease and the same card; the set is installed from staging only with its `# complete` trailer.
dump-ref-mtp-glm5next:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/dump-mtp.sh'

# Qwen3.8-Flash-Next MTP draft oracle: the same dump_mtp (built by `just build-ref-dump-mtp`) under the qwen4exp profile,
# the shared draft file beside the target (-md): every node ik's MTP context computes while the target decodes 64
# positions after the first 64 prose ids, one draft token a round, into $BLOOMERY_DATA/ref-mtp/qwen4exp_prose64_n64_k1/
# (refset family mtp-qwen4exp). ik on the CPU, CUDA hidden, under the CPU lease and its card
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/mtp-qwen4exp-dump.card'); installed from staging only with its
# `# complete` trailer.
dump-ref-mtp-qwen4exp:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/dump-mtp.sh'

# The MTP draft oracle of a family's fixture file (glm5next, qwen4exp), into $BLOOMERY_DATA/ref-mtp/fx_<the real twin's set>:
# dump-mtp.sh with the fixture first shard as the target, the fixture file's `# fixture` line in the header and, for qwen4exp, the
# draft file beside the fixture target (GLM's draft is the target itself). Needs dump_mtp built from this tree
# (`just build-ref-dump-mtp`). BLOOMERY_TIER=fixture is required (in the real tier: exit 64; `just dump-ref-mtp-glm5next` and
# `dump-ref-mtp-qwen4exp` are the real sets'). Under the CPU lease and a card, as every dump; DUMP_DRY=1 in BLOOMERY_BOX_ENV
# prints the dump and stops before the lease. Under BLOOMERY_FIXTURE_VARIANT=<tag> the target is that fixture variant's file and the
# set is ref-mtp/fx_<tag>_<the real twin's set>.
dump-ref-mtp-fixture FAMILY:
    BLOOMERY_MODEL={{FAMILY}} ./tools/box.sh '[ "${BLOOMERY_TIER:-real}" = fixture ] || { echo "dump-ref-mtp-fixture: this recipe needs BLOOMERY_TIER=fixture, got ${BLOOMERY_TIER:-real}; it dumps the fixture files, and the real sets have their own recipes (exit 64)" >&2; exit 64; } && bash tools/ref/dump-mtp.sh'

# Qwen3.8-Flash-Next(qwen4exp) 오라클: 같은 덤프 도구를 qwen4exp 프로필로, ik를 CPU로 돌려 5토큰 배치 세트를
# $BLOOMERY_DATA/ref_qwen4exp/에 뜬다. VARIANT(step4, d1k, d3k와 -every-node 접미사)를 주면 조용한 프리필 뒤 디코드 한 스텝을
# 제 세트로 뜬다 — models/qwen4exp.sh. d1k와 d3k는 $BLOOMERY_DATA/qwen4exp/corpus-prose.ids의 첫 1,025개와 3,001개 id를
# 읽는다(sha256 핀). d3k의 스텝은 위치 3,000이라 QSA 층이 인덱서가 고른 셀만 본다.
# 덤프마다 111.3 GB 샤드 집합 전체를 CPU 임대 아래서 올리므로 카드가 있어야 한다
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/q38ref-dump.card').
dump-ref-qwen4exp *VARIANT:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/dump.sh {{VARIANT}}'

# The MiMo-V2.6-Flash (mimo2) oracle tree: /home/user/ik-mimo2, ik 043ced9a (the fused attn_qkv load) with the
# one shared-oracle commit upstream lacks cherry-picked, checked by patch-id and configured as that tree, built under
# the CPU lease, then dump_ref linked against it into $BLOOMERY_DATA/bin-mimo2 (tools/ref/build-ik-mimo2.sh; it
# writes the tree's PROVENANCE). BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/ikmimo2-build.card'.
build-ref-dump-mimo2:
    BLOOMERY_MODEL=mimo2 ./tools/box.sh 'bash tools/ref/build-ik-mimo2.sh'

# The mimo2 oracle: the same dumper under the mimo2 profile, ik on the CPU, the 5-id batch set into
# $BLOOMERY_DATA/ref_mimo2/. A VARIANT (step4, d1k, d4096, each with an -every-node suffix) dumps one decode
# step after a quiet prefill into a set of its own (models/mimo2.sh); d1k and d4096 read the first 1,025 and
# 4,097 ids of $BLOOMERY_DATA/mimo2/corpus-prose.ids (sha256 pin): d1k is the first depth past the 128-id
# sliding window, d4096 runs the nine full-attention layers over a deep context. Each dump pages in the whole
# 167.4 GB split set under the CPU lease, so it needs a card
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/mimoref-dump.card'). The profile's tree is the mimo2
# oracle tree and its dump_ref the one in bin-mimo2 (build-ref-dump-mimo2).
dump-ref-mimo2 *VARIANT:
    BLOOMERY_MODEL=mimo2 ./tools/box.sh 'bash tools/ref/dump.sh {{VARIANT}}'

# The same 5 ids on ik's CUDA backend into $BLOOMERY_DATA/ref_cuda_mimo2/. Not runnable as it stands: dump.sh's
# -ngl 99 puts the 167.4 GB file on one card, so a CUDA set needs a placement first. No GPU consumer yet.
dump-ref-mimo2-cuda:
    BLOOMERY_MODEL=mimo2 ./tools/box.sh 'BLOOMERY_REF_BACKEND=cuda bash tools/ref/dump.sh'

# ref_io.h's option parser, token reader and raw writer (tools/ref/ref_io_test.cpp). No card, no llama.cpp.
test-ref-io:
    g++ -std=c++17 -Wall -o "${TMPDIR:-/tmp}/ref_io_test" tools/ref/ref_io_test.cpp && "${TMPDIR:-/tmp}/ref_io_test"

# Nemotron 3 Nano 4B last-position logits, mainline on the CPU (ik has no nemotron_h). nemotron_ref
# links the existing mainline build (tools/ref/build-nemotron-ref.sh, run under either nemotron profile; it does not rebuild it) and
# writes $BLOOMERY_DATA/ref-nemotron-logits (refset family ref-nemotron-logits). CUDA stays hidden.
# No card: the lane is none.
build-ref-nemotron:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=90 BLOOMERY_MODEL=nemotron_h ./tools/box.sh 'bash tools/ref/build-nemotron-ref.sh'

dump-ref-nemotron:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=600 BLOOMERY_MODEL=nemotron_h ./tools/box.sh 'bash tools/ref/dump-nemotron.sh'

# Mainline llama.cpp's chat layer as an oracle: tools/ref/chat_parse_ref.cpp renders one user turn through a chat
# template and parses a generation as llama-server parses it (content, reasoning, tool calls). It links mainline's
# existing build (tools/ref/build-chat-parse-ref.sh) into $BLOOMERY_DATA/bin/chat_parse_ref. No card: the lane is none.
build-ref-chat-parse:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=60 ./tools/box.sh 'bash tools/ref/build-chat-parse-ref.sh'

# Nemotron 3.5 Bolt (nemotron_h_moe) last-position logits: the same scripts as the Nano's under the nemotron_h_moe profile, which names its
# own mainline tree (/home/user/llama.cpp-mainline-bolt, cd0a5b6a1) and so its own binary (nemotron_ref_moe). The 22 GB file on the CPU
# into $BLOOMERY_DATA/ref-nemotron-moe-logits (refset family ref-nemotron-moe-logits). `BLOOMERY_TIER=fixture just dump-ref-nemotron-moe`
# dumps the same prompts on the Bolt fixture into $BLOOMERY_DATA/fx_ref-nemotron-moe-logits (fx-ref-nemotron-moe-logits), with each
# MoE layer's last-position experts.
build-ref-nemotron-moe:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=90 BLOOMERY_MODEL=nemotron_h_moe ./tools/box.sh 'bash tools/ref/build-nemotron-ref.sh'

dump-ref-nemotron-moe:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=312 BLOOMERY_MODEL=nemotron_h_moe ./tools/box.sh 'bash tools/ref/dump-nemotron.sh'

# LFM3-1.7B oracle: llama.cpp mainline at 19cce6919, the commit that added `lfm3` (ik has none), in a tree of its own,
# /home/user/llama.cpp-19cce691: the same script as the Qwen tree (tools/ref/qvis/build-lcpp-qvis.sh) with the pin, tree, sentinel
# directory and targets by environment, so the md5s the nemotron_h logits family pins stay where they are. Detached, pid, log and rc
# under /root/lcpp-lfm3; the rc file is the end (0 built). The pin and tree are the lfm3 profile's (tools/ref/models/lfm3.sh).
build-lfm3-lcpp:
    BLOOMERY_BOX_ENV='LCPP_BUILD_PIN=19cce6919992fdffd92e0ac152caaf1e2b8f254e LCPP_BUILD_TREE=/home/user/llama.cpp-19cce691 LCPP_BUILD_DIR=/root/lcpp-lfm3 LCPP_BUILD_TARGETS=llama,mtmd,llama-common,llama-tokenize,llama-bench' ./tools/box.sh '[ "$(id -u)" = 0 ] || { echo "build-lfm3-lcpp: this recipe requires root on the box (/root/lcpp-lfm3 and /home/user); run build-lcpp-qvis.sh directly with writable LCPP_BUILD_DIR and LCPP_BUILD_TREE on a local machine" >&2; exit 2; }; setsid -f bash tools/ref/qvis/build-lcpp-qvis.sh </dev/null >/dev/null 2>&1'

# LFM3-1.7B's three GGUF files (Q4_K_M, Q8_0, BF16) at the profile's Hugging Face revision into /models/LFM3-1.7B-GGUF, checked against
# the table's bytes and sha256, and the REVISION file beside them (tools/ref/fetch-lfm3.sh). A file that matches stays. The queue's
# 300 s is derived (6.41 GB at an assumed 100 MB/s plus the sha256 pass); the box's link is not measured.
fetch-lfm3:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=300 BLOOMERY_MODEL=lfm3 ./tools/box.sh 'bash tools/ref/fetch-lfm3.sh'

# LFM3-1.7B last-position logits, mainline on the CPU: nemotron_ref (tools/ref/nemotron_ref.cpp) linked against the lfm3 tree's
# build (tools/ref/build-nemotron-ref.sh under the lfm3 profile; it does not rebuild the tree) into $BLOOMERY_DATA/bin/nemotron_ref_lfm3,
# and its `build` line is the two md5s crates/refset/src/arch/lfm3 pins. Run after build-lfm3-lcpp's rc is 0. No card: the lane is none.
build-ref-lfm3:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=90 BLOOMERY_MODEL=lfm3 ./tools/box.sh 'bash tools/ref/build-nemotron-ref.sh'

# The three sets of the three files, one after the other (tools/ref/dump-lfm3.sh over tools/ref/dump-nemotron.sh): ten prompts each,
# into $BLOOMERY_DATA/ref-lfm3-logits (Q4_K_M), ref-lfm3-q8-logits and ref-lfm3-bf16-logits (refset families of those names). CUDA
# stays hidden. The queue's 600 s is derived, not measured: 6,499 prompt tokens a file at about 2.9 GFLOP a token (the header's
# shapes), over an assumed 80 prompt tokens/s for the Q4_K_M file at -t 8 and slower for Q8_0 and BF16; that rate is the one term
# the run measures.
dump-ref-lfm3:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=600 BLOOMERY_MODEL=lfm3 ./tools/box.sh 'bash tools/ref/dump-lfm3.sh'

# Solar-Open2-250B oracle: solar2_dump linked against the existing PAI build (llama.cpp 6ea215d17 + prometheusAIR's solar_open2 patch,
# /home/user/llama.cpp-solar2), which this does not rebuild (tools/ref/solar2-dump-build.sh); the binary and its build record land in
# $BLOOMERY_DATA/bin.
build-ref-solar2:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=60 BLOOMERY_MODEL=solar-open2 ./tools/box.sh 'bash tools/ref/solar2-dump-build.sh'

# The Solar-Open2 oracle sets (refset family pai-solar2; tools/ref/solar2-dump.sh has the layout): the PAI tree on the CPU over the
# four prompts of tools/ref/data/solar2 from one load of the 151.8 GB Q4_K_M file, into $BLOOMERY_DATA/ref_solar2_{a,b,c,d}: each layer's
# taps for a, each layer output for b and c, the last logits row for all four, the top-40 logprobs and 64 greedy tokens. Predicted wall
# 2-4 min, under the CPU lease and its card (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/solar2-dump.card'); DUMP_DRY=1 in
# that variable prints the dump and stops before the lease. `BLOOMERY_TIER=fixture just dump-ref-solar2` dumps the same prompts on the
# fixture into $BLOOMERY_DATA/fx_ref_solar2_{a,b,c,d}.
dump-ref-solar2:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=240 BLOOMERY_MODEL=solar-open2 ./tools/box.sh 'bash tools/ref/solar2-dump.sh'

# MiniMax-M3-UD-IQ4_XS oracle: minimaxm3_dump (the shared tools/ref/cb_dump.cpp, the Solar-Open2 dumper's source) linked against the
# existing mainline build (llama.cpp cd0a5b6a1, CPU, no patch, /home/user/llama.cpp-mainline-bolt, whose tracked files must be clean),
# which this does not rebuild (tools/ref/minimaxm3-dump-build.sh); the binary and its build record land in $BLOOMERY_DATA/bin.
build-ref-minimaxm3:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=60 BLOOMERY_MODEL=minimax-m3 ./tools/box.sh 'bash tools/ref/minimaxm3-dump-build.sh'

# The MiniMax-M3 oracle sets (refset family lcpp-minimaxm3; tools/ref/minimaxm3-dump.sh has the layout and the taps): mainline on the
# CPU over the five prompts of tools/ref/data/minimaxm3 from one load of the 208.5 GB UD-IQ4_XS file, into
# $BLOOMERY_DATA/ref_minimaxm3_{a,b,c,d,e}: each layer's taps for a, each layer output and the router's expert ids for b and c, the last
# logits row for d, the sparse attention's block scores and selection for e (the only context past 2,048 positions), the top-40 logprobs
# and 64 greedy tokens for a, b and c. Predicted wall 3-10 min, under the CPU lease and its card
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/minimaxm3-dump.card'); DUMP_DRY=1 in that variable prints the dump and stops before
# the lease. The file has no fixture: BLOOMERY_TIER=fixture is refused by name. The profile's own model is the M3.1 checkpoint folder, so the
# recipe names the GGUF's first shard as BLOOMERY_REF_MODEL; this route is kept as code and is unverified for M3.1 (`dump-ref-minimaxm31`).
dump-ref-minimaxm3:
    BLOOMERY_REF_MODEL=/models/MiniMax-M3-UD-IQ4_XS/MiniMax-M3-UD-IQ4_XS-00001-of-00006.gguf BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=330 BLOOMERY_MODEL=minimax-m3 ./tools/box.sh 'bash tools/ref/minimaxm3-dump.sh'

# The MiniMax-M3.1-Flash oracle sets (refset family tf-minimaxm31; tools/ref/mm31-dump.sh has the layout and the phases): transformers' own
# MiniMaxM3VL decoder layers in f32 on the CPU over the vendor's checkpoint folder (tools/ref/mm31/dump_tf.py: one layer resident at a time,
# the MXFP8 and NVFP4 matrices dequantized on the fly and checked bit for bit against their libraries), over the five prompts of
# tools/ref/data/minimaxm3, into $BLOOMERY_DATA/ref_minimaxm31_{a,b,c,d,e,e_perhead} (e under the vendor's selection rule, e_perhead under the card
# kernel's); BLOOMERY_TIER=fixture dumps the fixture cut's sets as fx_ref_minimaxm31_*. PHASE is `prefill` (the layer-major pass), `greedy` (the
# token-major continuation of a, b and c, after which every set is installed; a job cut at its bound continues with MM31_RESUME=1 in
# BLOOMERY_BOX_ENV), `all` (both in one job: the cut's) or `selftest` (the dequantization check alone). Each job takes the CPU lease and its card
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/minimaxm31-dump.card', the cut's docs/cards/minimaxm31-dump-fixture.card); DUMP_DRY=1 in that
# variable prints the plan and stops before the lease. Predicted wall [derived: the card]: prefill 900 s, greedy 2100 s, the rest 600 s.
dump-ref-minimaxm31 PHASE:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S={{ if PHASE == 'prefill' { '900' } else if PHASE == 'greedy' { '2100' } else { '600' } }} BLOOMERY_MODEL=minimax-m3 ./tools/box.sh 'bash tools/ref/mm31-dump.sh {{PHASE}}'

# Muse-Spark-Open oracle: musespark_dump (the shared tools/ref/cb_dump.cpp) linked against the existing mainline
# build (llama.cpp ba3c05f71, PR 30616's merge, CPU, no patch, /home/user/llama.cpp-muse, whose tracked files
# must be clean), which this does not rebuild (tools/ref/musespark-dump-build.sh); the binary and its build
# record land in $BLOOMERY_DATA/bin.
build-ref-musespark:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=60 BLOOMERY_MODEL=muse-spark ./tools/box.sh 'bash tools/ref/musespark-dump-build.sh'

# The Muse-Spark-Open oracle sets (refset family lcpp-musespark; tools/ref/musespark-dump.sh has the layout and
# the taps): mainline on the CPU over the five prompts of tools/ref/data/musespark from one load of the
# 225.6 GB UD-IQ2_XXS file, into $BLOOMERY_DATA/ref_musespark_{a,b,c,d,e}: every per-layer tap for a, each
# layer output and the router's expert ids for b and c, the last logits row for d, the past-the-window
# context e (2,560 ids against the 2,048 window), the top-40 logprobs and greedy tokens. The prompt ids are
# the muse tree's llama-tokenize on the file (the script's header has the command); they are taken once and
# kept in tools/ref/data/musespark. Predicted wall 10-15 min [derived: the script's header], under the CPU
# lease and its card (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/musespark-dump.card'); DUMP_DRY=1 in
# that variable prints the dump and stops before the lease. `BLOOMERY_TIER=fixture just dump-ref-musespark`
# dumps the same prompts on the fixture into $BLOOMERY_DATA/fx_ref_musespark_{a,b,c,d,e}.
dump-ref-musespark:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=900 BLOOMERY_MODEL=muse-spark ./tools/box.sh 'bash tools/ref/musespark-dump.sh'

# DSpark draft 오라클: ik `db517b69`에 `ik-dsv41-draft.py`만 얹은 트리(`/home/user/ik-dspark-draft`)를 짓고
# dump_draft를 그 트리의 libllama·libggml에 링크한다. ik 빌드는 CPU 임대 아래서 돈다.
build-ref-dump-draft:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/build-dump-draft.sh'

# The V4.1 candidate-mask oracle: ik ed27bf7e (its separate V4.1 graph, V41_SEPARATE) with #2507's fix cherry-picked, at
# /home/user/ik-cand, and a dump_ref linked against it in $BLOOMERY_DATA/bin-cand; the ik build runs under the CPU lease
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/candref-build.card'). `just dump-ref-v41 d1c-unfused-every-node` and the
# `-sep` variants dump from it (tools/ref/models/deepseek41.sh).
build-ref-ik-cand:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/build-ik-cand.sh'

# draft를 켠 greedy 디코드에서 draft 컨텍스트의 노드 전부를 $BLOOMERY_DATA/ref-draft/<세트>/에 쓴다(코드 코퍼스 64 + 32 위치,
# 블록 폭 3). 3090 게이트 락 + CPU 임대; 세트는 staging 뒤 `# complete` 트레일러가 있을 때만 설치된다.
dump-ref-draft:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/dump-draft.sh'

# 오라클 세트의 정수 사본 검사(B0c): 정수 텐서마다 무손실 사본이 있고, f32 파일이 그 값의 RNE인가.
# 인자는 세트 디렉터리(기본 $BLOOMERY_DATA/ref). V2-Lite의 ref는 v1이라 사본이 없어 빨강이다 — just gate에 넣지 않는다.
check-int-twins *ARGS:
    ./tools/box.sh 'python3 tools/ref/check-int-twins.py {{ARGS}}'

# V4.1 라우터 추적(B10): 토큰 스트림을 프리필로 흘려, 층마다 토큰별로 고른 expert 여섯을 $BLOOMERY_DATA/router/<이름>/에
# 쓴다. CPU 임대 안에서 돌고 30분에서 끊는다. CORPUS가 `oracle`이면 오라클의 다섯 토큰으로 돌려 덤프의 id와 정수로
# 맞춰 본다. 인자는 러너 머리글 참조(--name, --chunk, --max-tokens, --top-k-only).
trace-router CORPUS *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/router-trace.sh {{CORPUS}} {{ARGS}}'

# GLM-5.3-Flash의 라우터 추적: trace-router와 같고, 코퍼스는 $BLOOMERY_DATA/glm5next/corpus-<이름>.ids(GLM 토크나이저),
# 세트 이름은 glm5next-<이름>이다(V4.1 세트와 한 디렉터리를 쓴다).
trace-router-glm5next CORPUS *ARGS:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/router-trace.sh {{CORPUS}} {{ARGS}}'

# Qwen3.8-Flash-Next의 라우터 추적: trace-router와 같고, 코퍼스는 $BLOOMERY_DATA/qwen4exp/corpus-<이름>.ids(Qwen3.8 토크나이저),
# 세트 이름은 qwen4exp-<이름>이다.
trace-router-qwen4exp CORPUS *ARGS:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/router-trace.sh {{CORPUS}} {{ARGS}}'

# MiMo-V2.6-Flash의 라우터 추적: trace-router와 같고, 코퍼스는 $BLOOMERY_DATA/mimo2/corpus-<이름>.ids(MiMo 토크나이저),
# 세트 이름은 mimo2-<이름>이다. 0번 층이 dense라 세트는 1–47층을 담는다(router-coverage.py는 --layers 1-47).
# 마지막 층의 attention이 출력 행만 남기고 FFN이 도므로 프로파일(models/mimo2.sh)이 --all-logits를 붙인다.
trace-router-mimo2 CORPUS *ARGS:
    BLOOMERY_MODEL=mimo2 ./tools/box.sh 'bash tools/ref/router-trace.sh {{CORPUS}} {{ARGS}}'

# ik 트리 하나의 wikitext-2 퍼플렉서티(c2048, 4청크, CPU만)를 서빙하는 V4.1 파일로 CPU 임대 아래서 잰다.
# 두 트리를 연달아 돌리면 포트의 A/B다. ARGS: --chunks N.
# 그 밖의 ARGS: --ctx N, --batch N, --ubatch N, --kld-base(KLD 기준 파일을 쓴다), --kld <기준 태그>(그 파일에 대한 KLD) — 러너 머리글.
ik-ppl TREE TAG *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/ik-ppl.sh {{TREE}} {{TAG}} {{ARGS}}'

# 1단계 서브블록 게이트. 각 라운드가 자기 것 하나만 소유한다. gate-ops는 bloomery-model 라이브러리의 단위 시험도
# 같이 돈다(--lib, --include-ignored) — 그것을 도는 레시피가 따로 없다.
gate-ops:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model --lib --test ops -- --include-ignored --nocapture'

# The V2-Lite sub-block gates that run in the debug profile (their debug_assert!s are coverage), one cargo call that runs every
# target even when an earlier one is red: the MLA attention (attn), the dense FFN and the shared expert (ffn), the MoE block
# (moe) and the output head (head), each against the oracle's staged tensors.
gate-ds2-debug:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model --no-fail-fast --test attn --test ffn --test moe --test head -- --ignored --nocapture'

# The V2-Lite assembled-model gates, one cargo call in the release profile that runs every target even when an earlier one is red:
# the 27 blocks end to end in a debug build would take minutes, and Rust does not reassociate f32, so the numbers do not depend
# on the profile.
# forward: 1-4 조립 게이트.
# kv: 1-5 KV 캐시 게이트: 캐시가 있는 경로와 없는 경로의 로짓이 비트 동일한가.
# derived: Derived 게이트: 토큰과 무관한 wk_b Q8_0 재양자화를 로드시 한 번으로 옮겼다 —
# 사전 계산이 값을 바꾸지 않는다. 블록은 참조 구현(테스트 안의 예전 두 루프)과
# 바이트 동일, step(명시적 Derived)과 forward(래퍼)의 로짓은 비트 동일.
# prompts: 1-4 판정 게이트: 프롬프트 32개의 argmax를 ik와 대조한다. just argmax-ref가 먼저다.
gate-ds2-release:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --no-fail-fast --test forward --test kv --test derived --test prompts -- --ignored --nocapture'

# The CPU engine's host gates in the release profile: one cargo call for the three that take libtest's default threads (alloc, mt,
# r8file; --include-ignored because most of r8file's tests are not ignored), then profile's own call, whose --test-threads=1 keeps
# the gate body's set_var from racing its child helper tests (the test file's SAFETY comment). Both calls run, and the recipe is
# red when either is.
# alloc: 할당 게이트: 정상 상태 디코드 스텝 하나가 할당자를 몇 번 부르는가(전 스레드 합).
# 속도를 지키는 게이트는 없지만 이 원인 하나는 정수로 환원된다 — 메인 스레드가
# 할당·0 채우기를 하는 동안 워커 전부가 논다.
# mt: 행 병렬 게이트: 스레드 수(1/3/32)가 로짓을 한 비트도 안 바꾸는가. BLOOMERY_THREADS는
# 프로세스당 한 번 읽히므로 자식 프로세스 재실행으로 덤프를 뽑아 바이트 비교한다
# (tests/mt.rs의 패턴 설명 참조).
# r8file: r8 사이드카(model::r8file) 게이트: gguf::write로 쓴 합성 두 샤드 원본에서 변환, Sidecar::open, verify가
# 통과하고, 원본과의 식별 차이와 변환 거부는 모두 이름으로 거부된다. r8conv 바이너리는 변환부터 검증까지
# 끝까지 돈다. 호스트 티어 적재는 원본이 달라진 사이드카를, 호스트 층은 down까지 담은 사이드카를 이름으로
# 거부한다. 모델 파일은 읽지 않는다.
# profile: 1-5 프로파일러 게이트: 계측이 로짓을 한 비트도 안 바꾸고, 스텝 시간의 80% 이상을 커버하며,
# 세 site가 전부 살아 있는가.
gate-model-host:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --no-fail-fast --test alloc --test mt --test r8file -- --include-ignored --nocapture; multi=$?; bash tools/gate.sh --release -p bloomery-model --test profile -- --ignored --nocapture --test-threads=1; single=$?; echo "gate-model-host rc: multi $multi single $single"; [ "$multi" = 0 ] && [ "$single" = 0 ]'

# B1a 배치 게이트: V4.1 샤드 아홉의 헤더만으로 텐서마다 역할·장치·형식·상주 바이트를 정하고, 설계 §5의 두 안
# ((a) A6000 + DDR4, (b) A6000 0–19층 + 3090 20–39층)이 설계의 핀과 같은지 본다. 텐서 바이트·GPU·임대 없이 초 단위다.
# BLOOMERY_PLACEMENT_TABLE=1이면 텐서별 표도 찍는다. (b′) 시험은 DSpark 드래프트 헤더에서 3090 예약을 읽으므로
# 프로필의 DSPARK_MODEL을 내보낸다.
gate-placement:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-placement && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gate.sh --release -p bloomery-model --test placement -- --include-ignored --nocapture && bash tools/gate.sh --release -p bloomery-placement --lib -- --include-ignored --nocapture'

# B5 메타 게이트. 먼저 층 표 단위 시험을 돌린다: 일부러 깨뜨린 층 표 일곱 개를 ik 적재기의 검사 넷과 우리 거부 둘이
# 각각 깨진 그 층에서 거부해야 한다. 이어서 V4.1의 하이퍼파라미터·층 종류·텐서 이름(arch/deepseek41/{hparams,names}.rs)을
# ik가 같은 파일에서 읽은 값(적재 출력과 헤더), ik가 지은 그래프(오라클 매니페스트 둘의 노드), 파일의 텐서 목록과
# 대조한다. 헤더와 매니페스트만 읽으므로 몇 초면 끝난다.
gate-ds41-meta:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-ds41-meta && bash tools/gate.sh --release -p bloomery-model --lib -- arch::deepseek41 arch::tests --nocapture && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gate.sh --release -p bloomery-model --test ds41_meta -- --ignored --nocapture'

# A published V4.1 set opens as it is: the header's facts under any publisher's spelling, the Engram hash constants as keys or
# tensors, and the Engram tables in the file a header names for them. Headers only, seconds. The pins are the 1017 Q3_K_M set's,
# so BLOOMERY_V41_MODEL=<first shard> opens another publisher's set of the model (ISTA-DASLab's 2.00 bpw shards state the
# same facts), and BLOOMERY_V41_ENGRAM names the companion file when it is not beside the model.
gate-ds41-open:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-ds41-open && bash tools/gate.sh --release -p bloomery-model --test ds41_open -- --ignored --nocapture'

# GLM HF names/dims and converted F32/BF16 values against the map oracle $BLOOMERY_DATA/ref-glm5next-hf-map
# (`just dump-ref-glm5next-hf-map`). The set opens through its refset family (crates/refset/src/arch/glm5next), which
# refuses by name a set of another checkpoint, GGUF or writer, with a file that is not the one written, or never finished.
# No fixture-tier conversion.
gate-glm5next-hf-map:
    ./tools/box.sh 'bash tools/ref/real-only.sh gate-glm5next-hf-map && bash tools/gate.sh -p bloomery-model-hf --test glm5next_map -- --ignored hw_glm_map_matches_the_unsloth_gguf --nocapture'

# qwen3moe 메타 게이트: 하이퍼파라미터 거부 단위 시험(합성 헤더) 뒤, arch/qwen3moe가 파일에서 읽은 값을 ik 그래프
# (ref_qwen3moe 매니페스트의 노드 형상·op)와 헤더에 대조하고, 텐서 전부가 역할과 허용 타입을 갖는지 본다. 헤더만, 초 단위.
gate-qwen3moe-meta:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-qwen3moe-meta && bash tools/gate.sh --release -p bloomery-model --lib -- arch::qwen3moe --nocapture && bash tools/gate.sh --release -p bloomery-model --test qwen3moe_meta -- --ignored --nocapture'

# Muse Spark Open's header gate: the family's unit tests, then the reader on the first shard of the UD-IQ2_XXS file
# (headers only, seconds; the muse-spark profile names it, BLOOMERY_REF_MODEL moves it). No fixture-tier conversion: the
# header test pins the real file's description (its first shard's bytes, 62 layers, hidden 8192), and the fixture tier's
# plan clause is gate-gpu-musespark-load's plan_places_every_expert.
gate-musespark-meta:
    BLOOMERY_MODEL=muse-spark ./tools/box.sh 'bash tools/ref/real-only.sh gate-musespark-meta && bash tools/gate.sh --release -p bloomery-model --lib -- arch::musespark --nocapture && bash tools/gate.sh --release -p bloomery-model --test musespark_meta -- --ignored --nocapture'

# qwen35moe(Qwen3.6-35B-A3B) 헤더 게이트: 리더가 읽은 파일의 서술과 커버리지 검사의 목록(트리가 아직 못 돌리는 인스턴스, 층별).
gate-qwen35moe-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test qwen35moe_meta -- --ignored --nocapture'

# llama(Nanbeige4.3-33B) header gate: the unit tests of the synthetic headers, then what the reader resolves from the real file
# (the description line by line), what the model needs of a program, and what the coverage check lists for it. Headers only: seconds.
gate-llama-meta:
    BLOOMERY_MODEL=nanbeige43 ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::llama --nocapture && bash tools/gate.sh --release -p bloomery-model --test llama_meta -- --ignored --nocapture'

# lfm3 (LFM3-1.7B Q4_K_M) header gate: the reader's refusals on synthetic headers (a missing key, a wrong conv width, a wrong layer
# pattern, a rope factor table of the wrong length or type, a missing or unclaimed tensor), the roles and the store bytes of each
# layer kind, then what the reader resolves from the real file (the description line by line, the coverage list naming each part
# no program runs, the rope factors' 32 x 1 and 32 x 1e30). Headers only, seconds. No program runs the description yet, so the
# coverage list is printed and its new parts pinned, not empty. No fixture-tier conversion: the file is the fixture (`self`).
gate-lfm3-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::lfm3 arch::classify --nocapture && bash tools/gate.sh --release -p bloomery-model --test lfm3_meta -- --ignored --nocapture'

# Cache append safety contracts and model-less plane/ring writer refusals.
gate-gpu-rope-writers:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu --release --doc rope_neox::RopeNeoxKernels && bash tools/gate.sh --oxide -p bloomery-gpu-gates --features gpu,mimo2 --release --lib --bin gate_qwen35moe_attn --bin gate_mimo2_flash -- --include-ignored writers --nocapture'

# qwen3moe 커널 게이트 여섯을 한 프로세스(`gate_qwen3moe_kernels`)에서 차례로 돈다(ik CPU 덤프 네 세트: 5토큰 프리필, 깊이
# 4·1,024·4,096의 디코드 스텝). 게이트는 그 바이너리의 모듈(`shared/qwen3moe_<이름>_gate.rs`)이고, 하나가 빨강이어도 나머지는
# 돈다. ARGS에 게이트 이름(rope router down head flash experts)을 주면 그것만 돈다. `rope`는 헤드별 QK RMS 노름과 NEOX 로프·K/V
# 캐시 쓰기(한 런치)를 두 절로, `router`는 소프트맥스 라우터 128/8과 재정규화, 그리고 ubatch 라우터를, `down`은 다운 `_sel`
# (Q6_K, Q4_K)을, `head`는 argmax를 접은 Q6_K 헤드를, `flash`는 GQA 플래시 디코드 두 패스(스칼라·텐서 코어)를, `experts`는
# 체인 커널 셋(Q4_K 임베딩 행(dequant_row·ik inp_embd와 비트 동일), expert gate·up·SwiGLU `_sel`(ffn_moe_gate_par 대조),
# combine(routed_out 대조))을 잰다. 레시피 하나가 바이너리 하나를 돌고, 그 바이너리가 이 게이트들을 모듈로 품는다.
gate-gpu-qwen3moe-kernels *ARGS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen3moe-kernels && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_kernels && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3moe_kernels {{ARGS}}'

# Qwen3.6-35B-A3B MoE 블록: 257행 게이트 라우터(전문가 256 + 공유 전문가 게이트 행), 조인된 257-전문가 스택, K=512 down `_sel`,
# 아홉 슬롯 FFN과 ubatch 경로를 호스트 규칙과 ik 밴드에 맞춘다. ARGS는 `--case <이름>[,…]`.
gate-gpu-qwen35moe-moe *ARGS:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen35moe_moe && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35moe_moe {{ARGS}}'

# Qwen3.6-35B-A3B 전 체인 게이트(한 장): 노드 수 핀(디코드 520, 패스 537 + 4m), 디코드 다섯 스텝 = 다섯 행 패스 = eager
# = 리셋 뒤 반복(logits·저장소 비트 동일), 층별 teacher-forced 탭(GDN·어텐션·MoE)과 자유 주행 l_out을 ik 배치 세트에,
# step4·d1k 세트는 ik의 프리필 상태(cache_s·cache_k·cache_v)를 실어 한 스텝을 댄다.
[group('pool')]
gate-gpu-qwen35moe-e2e:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen35moe_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35moe_e2e'

# Qwen3.6's placed load on the common card leg's types (`gate_qwen35moe_e2e --common-placed-only`, clause (o9)): the file planned
# with no card expert (the host tier's chain) and at (o)'s budget, the card running every layer's experts by the leg its types name
# (`accepts::qwen35_leg`). Asserted: the planner's counted arena is the load's, the common layers hold card experts, the prompt
# call is the fifth step's bits, each layer's output after five steps stays within the host tier's arithmetic band, and a wide
# unit's logits stand within the band of the host tier's and of the same load's one-token steps. Runs on the fixture variants whose
# routed stacks are a type the dispatch's launches do not run (tools/fixture-variants.tsv); the base fixture runs the dispatch's.
[group('pool')]
gate-gpu-qwen35moe-leg:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen35moe_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35moe_e2e --common-placed-only'

# The Qwen3.6 verify gate (one card): a load opened to verify drafted rows (`open_verifying`: 8 stamped lanes a delta layer)
# commits every `accepted` in 1..=m of a verify of m = 4 and 8 rows with no re-feed and no copy, and the 16 steps after it
# give the plain load's ids and last logits bit for bit (graph and eager, a chain of commits too); a lookup draft over
# `Session<Body35>` emits the plain run's 64 ids with partial commits; the plain load's store bytes, plan layout bytes and
# graph node counts equal their header derivation and the verifying load adds exactly the lanes (420 MiB + 960 B of
# stamps); a cut to a checkpoint after a commit; the refusals by name. The real Qwen3.6 file.
[group('pool')]
gate-gpu-qwen35moe-verify:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen35moe-verify && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen35moe_verify && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35moe_verify'

# Qwen3.6's own next-token layer (the unsloth UD-Q4_K_M file that carries `blk.40`, named by BLOOMERY_MTP_MODEL and pinned by
# its general.name and size) as a draft through Body35's layer path: the refusals by name at open (the plain file,
# BLOOMERY_REF_MODEL, with no next-token layer; a media load),
# the draft's resident bytes against the plan's figure with the bf16 widening named, the target's token_embd and output
# read in place at their addresses and copied nowhere, the input join at one stream against the host's rounding band and
# its refused stream counts, the draft's token equal to the host argmax of its logits, the refusals on the live draft, and
# a printed smoke of how often the draft's token is the target's own next token over a 64-id prompt. The real file only.
[group('pool')]
gate-gpu-qwen35moe-mtp:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen35moe-mtp && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen35moe_mtp && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35moe_mtp'

# The qwen3 serve seat's served drafts: BLOOMERY_DRAFT=dspark on Qwen3.6, on its two fine-tunes (Sage-A1.5 and
# Ornith-1.5, which the RedHat draft fits by their shapes, not by their names) and on MiniCPM-V 4.7 (its
# full-vocabulary draft, borrowing the target's Q6_K head and embedding rows), each target greedy with and
# without the draft on the same prompt, the chat surface under the draft, BLOOMERY_DRAFT=lookup on Qwen3.6,
# and the cross-target, placed, --parallel 2 and qwen3moe refusals. Real files only (the drafts and every
# target). The card is the A6000: the DSpark draft's load holds the whole target on it, and Qwen3.6's
# 29 GB does not fit the 24 GB card (the seat then refuses the draft by name).
gate-gpu-qwen3-dspark:
    BLOOMERY_MODEL=qwen35moe BLOOMERY_CARD=a6000 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen3-dspark && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_qwen3_dspark && D=target/qwen3-dspark-gate && rm -rf $D && mkdir -p $D && Q36=$(sed -n "s/^MODEL=\${BLOOMERY_REF_MODEL:-\(.*\)}$/\1/p" tools/ref/models/qwen35moe.sh) && test -n "$Q36" && MINI=$(sed -n "s/^MODEL=\${BLOOMERY_REF_MODEL:-\(.*\)}$/\1/p" tools/ref/models/minicpmv47.sh) && test -n "$MINI" && Q30=$(sed -n "s/^MODEL=\${BLOOMERY_REF_MODEL:-\(.*\)}$/\1/p" tools/ref/models/qwen3moe.sh) && test -n "$Q30" && bash tools/gpu-gate.sh gate_qwen3_dspark --qwen "$Q36" --qwen-draft /models/Qwen3.6-35B-A3B/dspark-redhat --sage /models/Sage-A1.5/Sage-A1.5-Q4_K_M.gguf --ornith /models/Ornith-1.5-35B-A3B/Ornith-1.5-35B-Q4_K_M.gguf --minicpm "$MINI" --minicpm-draft /models/MiniCPM-V-4.7-35B-A3B/dspark --qwen3moe "$Q30" --dir $D'

# Qwen3.6-35B-A3B's DSpark draft, RedHatAI's speculator (an HF directory on the box, `--draft DIR` names another): the dense layers on the card against a host f64 computation of the same layers from the file's bf16
# weights (a block of 8 rows over 3 and over 2,100 context rows, the window 2,048), the captured pass replaying to the
# eager bits, and the loader's card bytes. Real file only (the draft belongs to the Q4_K_M target); the gate lock.
gate-gpu-qwen35-dspark:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen35-dspark && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu,dspark-hf --release --bin gate_qwen35_dspark && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35_dspark'

# The same bin on the MiniCPM-V 4.7 35B-A3B text model (a qwen35moe file with its own vocabulary and a declared, stripped next-token
# layer; profile minicpmv47). ik refuses the file as it is and no ik set was dumped from it, so the gate takes its text path: the
# clauses that read no set (node-count pins, decode = pass = eager = reset bit for bit, the gemv arm, the wide plan, refusals,
# checkpoints, resident slots, the q8_0 cache arm) on the file, the first 32 greedy ids of three prompts (English, Korean, code)
# against ik's on a header-corrected copy, and a `SKIPPED (<clause>)` line for every clause that reads a set. Real file only (no
# fixture of it); the gate lock.
[group('pool')]
gate-gpu-minicpm-text:
    BLOOMERY_MODEL=minicpmv47 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-minicpm-text && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen35moe_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35moe_e2e'

# The Clef backbone (Qwen3.5 dense, qwen35) gate: the tensor-core decode flash refused at group 6 before any upload,
# then the prompt call's final-norm hidden states (Tail::Hidden) of the first 64, 600 and 4,096 prose ids, each from a
# reset as one GEMM ubatch (the 64 also as eight passes: the gemv arm), against llama.cpp mainline's result_norm on the
# same file (`just dump-hidden-qwen35`), every position within the derived band: the 27B Q4_K_M file (16.5 GB), then
# Clef-Flash at bartowski's Q8_0 (9.5 GB: every site through the launch its type picks), then lev at Q4_K_M (3 GB, the
# 4B geometry with its head tied to the token embedding). Fits either card.
gate-gpu-clef-hidden:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_clef_hidden && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_clef_hidden'

# The dense Qwen3.8-27B generation gate (qwen35, either card). Real tier, lmstudio's Q4_K_M (16.5 GB): the load against the header
# derivation without blk.64, the prompt call and forced decode steps' logits rows against mainline's (`just dump-logits-qwen38d`)
# within the bands the oracle's own CPU twin gives, graph = eager = pass, reset, the checkpoint cut with its re-fed tail, and two
# resident slots in one pass against each alone. Fixture tier, the NVFP4 cut folder (tools/fixture-cuts/qwen38nv.json) and its
# GGUF twin: ours on the twin against mainline's logits rows (`just dump-logits-qwen38nv`) within the bands the activation rule's
# derived term and the oracle's own CPU floor give, and ours on the folder equal bit for bit to ours on the twin. The gate lock;
# no lease.
gate-gpu-qwen38d:
    BLOOMERY_MODEL=qwen38d ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu,checkpoint --release --bin gate_qwen35_dense && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35_dense'

# The dense Nanbeige4.3-33B generation gate (llama, bartowski's Q4_K_M, either card, both tiers): the same clauses as
# gate-gpu-qwen38d over the Body35 chain on a trunk of attention layers alone (52 layers of 56/8 heads of 128, NORM rope, no head
# norm): the load against the header derivation, the prompt call and forced decode steps' logits rows against mainline's
# (`just dump-logits-nanbeige43`) within the bands the oracle's own CPU twin gives, graph = eager = pass, reset, the cut to any
# position with its re-fed tail (no layer holds a store to copy, so no checkpoint), and two resident slots in one pass against
# each alone. The group of 7 query heads a KV head runs the tensor-core flash only. The gate lock; no lease. The whole-fit verdict's
# cache term is the description layout's sum (`KvLayout::trunk_bytes`); the `kv_planes::` unit tests run first (host only, no card).
gate-gpu-llama-dense:
    BLOOMERY_MODEL=nanbeige43 ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features gpu --lib -- kv_planes:: && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_llama_dense && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_llama_dense'

# The qwen3 seat of bloomery-serve on Nanbeige4.3-33B's Q4_K_M, a `llama` file, with no seat word: `--plan` picks the seat by the
# program that runs the file's description (and `--model ds41` is refused naming `--model qwen3`); one server at ctx 4096 prints the
# qwen3 seat's load and listening records naming `llama`, renders the file's ChatML with one leading <|im_start|> and no added BOS
# (the raw /tokenize door adds it), splits the reply at the <think> span the generation prompt opens (thinking off: no span), renders
# a tool request's block with the file's template (the chat door counts the ids of that rendering) and parses a scripted XML call
# through a scripted engine behind the same template, and answers /completion at temperature 0 with the ids generate_qwen3moe --last-step prints for the
# same ids (run once the server has stopped, the file does not fit a card twice). One card; the gate lock.
gate-gpu-llama-serve:
    BLOOMERY_MODEL=nanbeige43 ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_llama_serve && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe && D=target/llama-serve-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_llama_serve --dir $D'

# Body35's image rows on Clef-Flash at bartowski's Q8_0 (9.5 GB, either card), against the clefvis sets (set B's tower rows
# as the spans' rows; set C's three prompts, their CPU twins and the C'' runs): the unit rows and the text rows a media call
# leaves (a span a unit boundary cuts, at the cut units and at one), set C's positions against ours and the rotation table
# they gather, the table against the host rebuild, a text request on the media load against the text load's (fresh and after
# an image call), a media call on a capturing stream, a call that starts inside an image, the cut back to an image's start
# and a cut inside one, the sequence's record saved and restored, a pass of an image slot and a text slot, a parked sequence
# resumed, a step graph captured at load and replayed after an image call, and the hidden band of set C per region. Real
# file only (the clefvis sets are dumped from it); the gate lock.
gate-gpu-qwen35-media:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen35-media && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features vision --release --bin gate_qwen35_media && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35_media'

# The same bin on the qwen35moe profile's seat: Body35's image rows on Qwen3.6-35B-A3B at lmstudio's Q4_K_M (21.2 GB, whole
# on one card, refused by name when the whole-fit verdict does not take it) against the qwen35moe-vis sets (mainline's mtmd on
# the same file): set B's tower rows as the spans' rows; set C's three prompts, their CPU twins and the C'' runs for the unit
# rows, set C's positions against ours and the rotation table they gather, the text request on the media load against the text
# load's, a step graph captured at load and replayed after an image call, and the hidden band per region; set F's two decodes
# (32 greedy steps after c1 and c2): the positions the steps read against F's n_past, the ids, each step's hidden against F's
# band, and the steps' logits against the same ids through passes of two rows. Not repeated here: Body35's slot and cache
# clauses (the dense seat closes them). Real file only; the gate lock.
gate-gpu-qwen35moe-media:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen35moe-media && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features vision --release --bin gate_qwen35_media && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35_media'

# Qwen3.8's image input (gate_qwen38_media's doc): Body38's media call on the ubatch walk (`enable_media`, the media load's
# own rotation table) on the UD-Q4_K_XL file, its routed experts on the card's id prefix and the host beyond it, against
# the qwen4exp-vis sets (mainline's mtmd on the same file, `just dump-ref-qvis qwen4exp`): (A) the unit rows of an image
# call (the widened bf16 rows of set B at image positions, a no-span call's rows elsewhere, a span a unit boundary cuts
# read in each unit), (B) set C's positions against ours and the table they gather, (C) the encoder file's identity
# against the text file through the seat's own check (Qwen3.6's projector refused by name, Qwen3.8's accepted), (D) the
# text request on the media load against the text load's, the call-start refusal inside a held image, the cut back to an
# image's start and its refusal inside one, a media call while capturing, the text load's refusals and a step graph
# captured at load and replayed after an image call, (E) set F's two decodes (ids on the prefix its CPU twin agrees on, the
# positions the steps read), (F) the tower's rows through the seat's resident encoder against set B within the tower
# gate's band, and (G) one served answer (`bloomery-serve --model qwen38 --mmproj`: set E's prompt ids, set F's ids or a
# near tie, and the 400 of a server with no --mmproj). The plan runs on the gate card's bytes less the tower's and the
# rotation tables'. Real file only (the sets are dumped from it). A chain member (v41-load, not solo): the host set (the most
# of the file) loads under the big-load lock four times, the text load, the media load and the two servers one after the
# other, and the gate's own plans take the workstation's plan room (PLAN_ROOM), so a neighbour's load moves none of them;
# the two servers plan on the host's reading, which no admitted neighbour (a lane item beside the chain) lowers to the need.
[group('v41-load')]
gate-gpu-qwen38-media:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen38-media && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_qwen38_media && D=target/qwen38-media-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_BOUND=2400 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen38_media --dir $D'

# The decide seat of bloomery-serve (gate_clef_serve's doc): Clef-Flash at bartowski's Q5_K_M (6.9 GB, the
# `qwen35` file of rev d7f376ea) with the release's head; the six refusals (a qwen35 file with no head, --model
# decide with no head, --model qwen3 with --head, a head config no row knows, a head narrower than the backbone,
# a GGUF of llama.cpp's Clef layout with no decision head) exit before any backbone load, then one
# server answers the suite's first request on its row's route only, speaks llama.cpp's /v1/systemone wire (the
# answer's model is the server's, /v1/models, the engine object, an image with no encoder a 501 with llama.cpp's
# hint), the same body again; then the same quantization in llama.cpp's Clef layout (bartowski's current upload,
# 6.98 GB, the head inside the GGUF in Q8_0; fetched into /root/models/clef-flash/) serves the request with no head
# flag, and its answers equal the same file's with the release's head (the one difference: the head's dtype); then
# the image input on Clef-Flash's Q8_0 file with its bf16 encoder (/root/models/clef-flash/mmproj-*-bf16.gguf): the
# text control with no encoder, the encoder's `vision` record, set D's six requests (the clefvis oracle's, real file
# only: $BLOOMERY_DATA/clefvis/ref) answered in the release's shape and top options, the text bodies unmoved by the
# media load, the image refusals, and the first image request again; ARGS `--hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M`
# adds the clauses that start the seat from the hub with no projector flag (it fetches the repo's own projector and
# serves images) and with --no-mmproj (it serves none): a network read. Either card.
gate-gpu-clef-serve *ARGS:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_clef_serve && D=target/clef-serve-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_clef_serve --model /models/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf --head /models/clef-flash/hf/joint_head.safetensors --clef-model /root/models/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf --image-model /models/clef-flash/Cloudflare_clef-flash-Q8_0.gguf --mmproj /root/models/clef-flash/mmproj-Cloudflare_clef-flash-bf16.gguf --dir $D {{ARGS}}'

# The decide seat of bloomery-serve on lev (gate_lev_serve's doc): ggml-org/lev-GGUF's Q4_K_M (3.0 GB, a `qwen35`
# file with `qwen35.decision.type = lev`, fetched into /root/models/lev-4b/) with no head flag; five refusals before
# any backbone load (--head beside a lev file, --model decide --head beside one, a GGUF whose decision type is kev,
# a qwen35 file with no decision keys, --help's table), then one server answers the 14 requests of tools/ref/lev/
# against llama.cpp mainline's server (`just dump-lev`): the same body shape and top options, usage the sum of the
# prompts' ids; ARGS `--hf ggml-org/lev-GGUF:Q4_K_M` adds the clause that serves the file from the hub, no head flag, and
# answers the first request as the `-m` server does (a network read). Either card.
gate-gpu-lev-serve *ARGS:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_lev_serve && D=target/lev-serve-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_lev_serve --model /root/models/lev-4b/lev-Q4_K_M.gguf --head /models/clef-flash/hf/joint_head.safetensors --ref $BLOOMERY_DATA/lev/4b/ref --dir $D {{ARGS}}'

# clef_hidden on the box (ARGS as its doc: --model, --ids, --out, ...): a qwen35 file's prompt-only pass, every
# position's final-norm hidden state to a file and a `clef hidden` record with the call's functional wall.
clef-hidden *ARGS:
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin clef_hidden && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh clef_hidden {{ARGS}}'

# Qwen3.8-Flash-Next (qwen4exp) whole-program gate on the 3090's plan (either card, shared/gate_card.rs): the step's
# node count, graph = eager = one pass (logits and every store bit for bit), reset clears, every layer's streams on the
# batch set within the borrowed band with the router's flips named, the step after each set's prompt (4, 1,024 and
# 3,000 positions; D3K's fed by ubatches), D1K's prompt by passes = by steps, windows over D3K's selector region by
# passes = by steps, and the refusals. Loads the whole host set: alone in a batch, under the big-load lock.
# solo-real, reason r1 (the whole host set under the big-load lock; not both cards, no host-memory pin): the fixture tier balances it.
[group('solo-real')]
[group('v41-load')]
gate-gpu-qwen4exp-e2e:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen4exp_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen4exp_e2e'

# Qwen3.8 UD-Q3_K_XL's i-quant routed experts (IQ3_XXS and IQ4_XS gate/up, IQ4_NL down) through the threaded host leg
# on the 3090's plan (either card, shared/gate_card.rs): the leg's per-expert gate, up, combine and down and its union
# call's tile equal an independent recompute from the file's rows by qdot's dot bit for bit on a sampled set of layers
# and host-served experts; the 512-position prompt's ubatch walk lists host slots on every layer holding an i-quant
# stack, the load naming the three types on path=fused; and the same prompt and 8 decode steps at the pool's default
# width and at width 1 (two processes, each loading the file, BLOOMERY_POISON=1) leave the same ids and logits bit for
# bit. Loads the 90 GB file twice: alone in a batch, under the big-load lock.
# Stays solo (r1) in the real tier only (real-only: its file is UD-Q3_K_XL's i-quants, which the Qwen3.8 fixture (Q4) does not hold; the fixture tier defers it).
[group('solo')]
[group('v41-load')]
gate-gpu-qwen4exp-iqleg:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen4exp-iqleg && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen4exp_iqleg && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen4exp_iqleg'

# Qwen3.8-Flash-Next from the NVIDIA NVFP4 checkpoint (the split `prepare` writes from nvidia/Qwen3.8-Flash-Next-NVFP4) on
# one card whose plan holds every routed expert (`--place a`; a DGX Spark's GB10): the load puts no expert on the host; the
# batch set's eager steps hold every layer's streams within the e2e gate's band of the checkpoint's own forward (refset
# hf-qwen4exp-nvfp4, the sets of the split's dense form: BF16 as stored, or the Q8_0 of BLOOMERY_PREPARE_Q8=1), the
# router's flips named, and its last argmax; the batch set by one ubatch and D3K's step after its prefill by ubatches hold
# the oracle's argmax (D3K's a named tie allowed); the distances to the other form's sets are recorded. The same plan
# under a card budget 8 GiB short splits each layer's experts between the card and the host, and its eager steps hold
# the same band; under a small `mid` residency, every layer's card set reseeded to its highest ids, the machine stages
# those host experts onto the card, every part of each (an NVFP4 stack's codes, block scales and tensor scale) the bytes a
# static load uploads, and the eager steps hold the band again. The bin's unit
# test, which holds the choice of sets to the split's dense form, runs first. BLOOMERY_REF_MODEL moves the split from
# the model folder below, including to prepare's stored-form output, or names the checkpoint folder itself, which the
# load reads as stored (no prepare). Needs BLOOMERY_CARD_ALL_EXPERTS: only a card of 128 GB or more holds the plan, so the
# maintainers' box defers it by name (the fixture holds no NVFP4 stack either).
gate-gpu-qwen4exp-nvfp4:
    BLOOMERY_REF_MODEL="${BLOOMERY_REF_MODEL:-/models/Qwen3.8-Flash-Next-NVFP4/Qwen3.8-Flash-Next-NVFP4-00001-of-00004.gguf}" BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/needs-env.sh gate-gpu-qwen4exp-nvfp4 BLOOMERY_CARD_ALL_EXPERTS && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features gpu,checkpoint --bin gate_qwen4exp_nvfp4 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu,checkpoint --release --bin gate_qwen4exp_nvfp4 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen4exp_nvfp4'

# Qwen3.8's MTP draft program on the 3090 beside the target: first, before any walk, the draft's load — its weights,
# store and row map hold its plan's bytes, its token_embd and output are the target's own buffers at their addresses,
# and Mtp38::open's refusals — then every graph of ik's MTP draft set (mtp-qwen4exp) replayed teacher-forced — eh_proj,
# l_out and the head's logits within their derived bands, the router's flips named, the argmax ik's where its margin
# clears the band — the row-list head against the full head, the target's streams paired with ik's hidden rows, the
# captured walks = the eager ones, and the refusals and the NaN fault. Loads the whole host set: alone in a batch, under
# the big-load lock.
# solo-real, reason r1 (the whole host set under the big-load lock; not both cards, no host-memory pin): the fixture tier balances it.
[group('solo-real')]
[group('v41-load')]
gate-gpu-qwen4exp-mtp:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen4exp_mtp && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen4exp_mtp'

# Qwen3.8's shared draft store: each walk detaches its written pages, and a reset releases
# shared mappings before whole-state and prefix put-backs. No reference set is needed.
[group('v41-load')]
gate-gpu-qwen4exp-mtp-share:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gate.sh --gpu --oxide -p bloomery-gpu-gates --release --features gpu --bin gate_qwen4exp_mtp -- --ignored hw_mtp_shared_store --nocapture'

# Qwen3.8's MTP input boundary on a checkpoint folder (BLOOMERY_QWEN38_FOLDER names a Qwen3.8-Flash-Next folder with its MTP layer,
# such as FP4-Local): MTP reads only target hidden rows the last layer completed, in both prompt walks, and refuses the rows of a
# tapped call that failed past its launch; the draft's own row runs only at the refreshed walk's next position and is refused after
# a state adoption, reset, store walk, slot exchange or cut; the drafted greedy ids equal the plain ones. Then the serve seat's closing
# call (hw_close_fold): a prompt's last call that runs its last id as its last row, cut anywhere or kept by a cache, gives the one
# call's first token and its bits (and the later steps' unless that id completes a delta chunk, which the next step folds), its end
# mark before that id. Then the serve seat's prompt quantum (hw_quantum_cuts): a prompt cut at the multiples of
# CHECKPOINT_EVERY, its first and last calls shorter than a ubatch walk's fewest ids, gives the one call's first token, bits,
# steps and checkpoints. The whole model on one card, one test at a time: without BLOOMERY_QWEN38_FOLDER a batch
# defers it by name (needs-env).
[group('v41-load')]
gate-gpu-qwen4exp-mtp-input:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/needs-env.sh gate-gpu-qwen4exp-mtp-input BLOOMERY_QWEN38_FOLDER && export BLOOMERY_GATE_V41_LOAD=1 && BLOOMERY_REF_MODEL="$BLOOMERY_QWEN38_FOLDER" bash tools/gate.sh --gpu --oxide -p bloomery-gpu-gates --release --features gpu,checkpoint --bin gate_qwen4exp_mtp -- --ignored hw_mtp_input_validity hw_close_fold hw_quantum_cuts --nocapture --test-threads 1'

# 그룹 int8 텐서코어 GEMM(bloomery_gpu::gemm): 실파일 Qwen3 gate(Q4_K)·down(Q4_K, Q6_K) 스택과 V4.1 routed gate(Q3_K)
# 하나, 같은 형상의 합성 스택(+ Q5_K), T ∈ {1,15,16,17,64,511,512,4096} × 라우팅 넷을 f64 참조의 유도 밴드에 대고 잰다.
# fault·거부·그래프 재생 포함. ARGS는 `--case <부분문자열>` 필터. `--bench-kernels`는 판정 대신 런치 값을 잰다
# (리드 전용, `bench-gpu-kernels`). relu² 접힌 32값 블록 양자화기(`g32_relu_sqr`)는 relu_sqr 다음 quantize_gemm32와 비트 동일.
gate-gpu-gemm *ARGS:
    ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-gemm && bash tools/gate.sh --oxide --release -p bloomery-gpu-gates --features gpu --bin gate_gemm -- workers_must_complete kquant_contract_keeps_a_subnormal_step_exact nan_bits_equal_takes_the_length_and_any_nan'
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-gemm && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_gemm && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemm {{ARGS}}'

# 선형 어텐션(Gated DeltaNet) 카드 커널 셋 — conv+준비, delta 스텝, 게이트 노름 — 을 호스트 규칙과 비트로 맞춘다.
# 모델 없이 Qwen3.6-35B-A3B 형상의 합성 입력으로 돌고, fault 사이트, 한 토큰의 그래프, 깊이 4,096의 f64 drift 보고를 함께 찍는다.
# ARGS `--case kda_ik`는 그 대신 KDA 층마다 conv·준비·delta·게이트 노름을 ik GLM 배치 세트(ref_glm5next)의 입력에 대 본다.
gate-gpu-linear *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_linear && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_linear {{ARGS}}'

# Qwen3.6-35B-A3B full-attention 카드 커널 — rope-256(q/k 노름 + 앞 64차원 NEOX), 디코드 flash 256(두 패스),
# 프리필 flash 256, 게이트 접은 o-proj 양자화기 — 을 모델 없이 16/2×256, n_rot 64, θ 1e7 합성 입력으로 호스트 규칙에 맞추고,
# ik 배치 세트의 층 3 탭 넷과 비교한다.
gate-gpu-qwen35moe-attn:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen35moe_attn && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen35moe_attn'

# EmbeddingGemma 2 global 층의 카드 커널 둘, rope-512(q/k 노름, 512차원 전체 NEOX, f16 추가)와 프리필 flash 512(p4,
# V 반쪽 둘, 32키 타일)를 모델 없이 합성 입력(rope 4/1·8/2, flash 4/1·16/2 헤드×512, θ 1e6, 스케일 1)으로 호스트 규칙과 f64 정확 어텐션 band에 맞춘다.
gate-gpu-gemma-embedding2-head512:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_gemma_embedding2_head512 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemma_embedding2_head512'

# MiniMax-M3's attention head kernel (head 128, the first 64 values turned: norm, partial NEOX rope, f16 append), model-less at
# the model's shapes (64/4 and 8/2 heads, θ 5e6): bit for bit the shared host transcription, the cache-position and overflow faults.
gate-gpu-minimaxm3-rope:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_minimaxm3_rope && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_minimaxm3_rope'

# MiniMax-M3's sparse attention (MSA), model-less at the model's shapes (4 index heads, blocks of 128, 16 kept, 64/4 heads of 128):
# the index heads, the block selection bit for bit against its host twin at the counts across the block boundaries up to 32k, the tie
# rule, the forced own block, a NaN key; the selected flash against the dense pass at every block kept and within the f64 band past
# it; the flash's faults.
gate-gpu-minimaxm3-msa:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features minimaxm3 --release --bin gate_minimaxm3_msa && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_minimaxm3_msa'

# MiniMax-M3.1-Flash end-to-end gate (gate_minimaxm3_e2e's doc): the whole decode chain read from the checkpoint folder in its stored
# formats (MXFP8 projections, NVFP4 routed experts, BF16 embedding and head: every layer a grouped-query attention layer with a per-head
# q/k norm and a partial rope, the sparse layers attending the blocks their indexer picks, the dense lead, the routed blocks with a shared
# expert), one prompt a step, against transformers' f32 dump of the same folder (refset family tf-minimaxm31, `just dump-ref-minimaxm31`;
# the one context past 2,048 positions is read under the card kernel's per-head selection rule). The real tier holds every layer's output
# at the last position within the relative ceiling (the sets that hold it) and the greedy tokens of three prompts teacher-forced with the
# clear-margin disagreements within the pin derived from their margins; the top-logprob clause is deferred by name. The fixture tier runs the
# fixture cut (`fixture cut`: the folder's layers 0, 3 and 59 at routed width 512, /models/fixtures/minimaxm31) against its own dump
# (`BLOOMERY_TIER=fixture just dump-ref-minimaxm31 all`: fx_ref_minimaxm31_*) with the self-consistency clauses and the layer outputs, and leaves
# the other clauses to the real tier by name. The M3 GGUF route (llama.cpp's dump of MiniMax-M3-UD-IQ4_XS, family lcpp-minimaxm3) is code only and
# unverified. Loads the whole model once and feeds about 4,500 single-token steps [derived: the five prompts' 4,308 ids and the teacher-forced greedy's 192]: alone in the real tier, under the big-load lock. `--plan` (host
# only: the gate plan's experts a layer and its bytes by term), `--smoke` (load, a short greedy decode, no pins) and `--route-trace DIR SET` (the
# host tier's route trace of prompt a, b or c into a new DIR, to hold against the set's `ffn_moe_topk-*.logical.i32`) reach the same binary
# through ARGS.
[group('solo-real')]
[group('v41-load')]
gate-gpu-minimaxm3-e2e *ARGS:
    BLOOMERY_MODEL=minimax-m3 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features minimaxm3,checkpoint --release --bin gate_minimaxm3_e2e && BLOOMERY_GATE_BOUND=1800 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_minimaxm3_e2e {{ARGS}}'

# MiniMax-M3.1-Flash prefill gate (gate_minimaxm3_prefill's doc): the prompt batch against the one-token steps over the four prompt lengths,
# in the logits row, the layer outputs and the stores, on the checkpoint folder in the real tier and on the fixture cut in the fixture tier.
# Every clause is a self-consistency clause, so the fixture tier runs them all. Loads the whole model once, under the big-load lock.
[group('solo-real')]
[group('v41-load')]
gate-gpu-minimaxm3-prefill:
    BLOOMERY_MODEL=minimax-m3 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features minimaxm3,checkpoint --release --bin gate_minimaxm3_prefill && BLOOMERY_GATE_BOUND=1800 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_minimaxm3_prefill'

# The MiniMax-M3.1 seat of bloomery-serve (gate_minimaxm3_serve's doc): the seat opens the checkpoint folder, serves two slots in turn,
# refuses a placement that hangs an expert tier card, loads at the context it was asked for and answers one chat turn with its
# reasoning split at the model's own tags and its turn ending at the model's own end token. The fixture tier runs on the cut folder,
# where the chat clauses are file-bound and left to the real tier. Loads the whole model once, under the big-load lock the V4.1 loads
# take. Predicted wall [derived: the folder's load about 25 s cold, the chat's prompt call and 64 steps about 2 min]: about 3 min; the
# bound is 1800 s.
[group('solo-real')]
[group('v41-load')]
gate-gpu-minimaxm3-serve:
    BLOOMERY_MODEL=minimax-m3 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_minimaxm3_serve && D=target/minimaxm3-serve-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_BOUND=1800 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_minimaxm3_serve --dir $D'

# MiniMax-M3.1-Flash on plan (b′) (gate_minimaxm3_prefill's doc, its `--arm bp`): the stage on the A6000 and the expert tier on the 3090,
# both planned under one card budget the gate sets, against the same plan with every tier segment on the stage card (the A6000 alone, no
# tier code): the load's tier experts by layer the plan's, a walk's decode ids and logits rows bit for bit the reference's and its eager
# steps the captured graph's, every tier layer sent a slot; then the common prefill clauses on the two-card load, which must send the tier
# their batches; then the serve gate with both cards in view (`place_bp_follows_the_cards`). Real tier only: the fixture cut's three layers
# leave the tier no expert at its budget. Both cards in view (BLOOMERY_CARD=both: both gate locks, alone in a batch), three loads of the
# folder in turn under the big-load lock. Predicted wall [derived: the folder's load measured about 135 s, the prefill gate's clauses about
# 2 min, the serve gate about 3 min]: about 12 min warm after a warm build; the bounds are 2400 s and 1800 s.
[group('solo')]
[group('v41-load')]
gate-gpu-minimaxm3-twocard:
    BLOOMERY_MODEL=minimax-m3 BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-minimaxm3-twocard && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features minimaxm3,checkpoint --release --bin gate_minimaxm3_prefill && BLOOMERY_GATE_BOUND=2400 bash tools/gpu-gate.sh gate_minimaxm3_prefill --arm bp && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_minimaxm3_serve && D=target/minimaxm3-serve-twocard && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_BOUND=1800 bash tools/gpu-gate.sh gate_minimaxm3_serve --dir $D'

# Nemotron-H MoE (Lightning) with its 32-value experts on the card (Relu2Leg for the step, the 32-value GEMM for the
# prompt): host, card (the 3090's room, split) and whole (the A6000's room, box.sh's pick) arms near the host path [derived: 8..12 min,
# mostly the build; three 25.5 GB loads]. The nemotron_h_moe profile's file is Bolt's (IQ2_XXS stacks); the gate runs
# Lightning's UD-Q4_K_XL, which BLOOMERY_REF_MODEL moves the profile to.
gate-gpu-nemotron-h-moe-card32:
    BLOOMERY_REF_MODEL="${BLOOMERY_REF_MODEL:-/models/Nemotron-3.5-Lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf}" BLOOMERY_MODEL=nemotron_h_moe BLOOMERY_CARD=a6000 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-nemotron-h-moe-card32 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_nemotron_h_moe_card32 && bash tools/gpu-gate.sh gate_nemotron_h_moe_card32'

# Nemotron-H MoE (Lightning) read from its ModelOpt checkpoint folder (NVFP4 routed experts, the dense projections
# made at load): host (every expert on the host, read from the folder in place), split (the whole plan's card bytes less
# half its routed experts', so about half stay on the host) and whole (the A6000's room) arms near the host path, the prompt
# path near its steps, and the logits' cosine against the UD-Q4_K_XL twin as a diagnostic [derived, not measured: the
# four loads decide the wall, about 10..20 min; no gate-times row yet]. BLOOMERY_REF_MODEL moves the folder,
# LIGHTNING_TWIN (in BLOOMERY_BOX_ENV) the twin.
gate-gpu-nemotron-h-moe-folder:
    BLOOMERY_REF_MODEL="${BLOOMERY_REF_MODEL:-/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}" BLOOMERY_MODEL=nemotron_h_moe BLOOMERY_CARD=a6000 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-nemotron-h-moe-folder && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu,checkpoint --release --bin gate_nemotron_h_moe_folder && BLOOMERY_GATE_BOUND=1800 bash tools/gpu-gate.sh gate_nemotron_h_moe_folder --twin "${LIGHTNING_TWIN:-/models/Nemotron-3.5-Lightning/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf}"'

# MiMo-V2 plan (b') on the A6000 and the 3090 (the 3090 an expert tier): the load, the walk equal to the union bit for
# bit, eager equal to replay, the batch equal to the union and sent [derived: about 8..12 min warm]; the bound is
# 2400 s.
[group('solo')]
[group('v41-load')]
gate-gpu-mimo2-twocard:
    BLOOMERY_MODEL=mimo2 BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-mimo2-twocard && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features mimo2 --release --bin gate_mimo2_e2e && BLOOMERY_GATE_BOUND=2400 bash tools/gpu-gate.sh gate_mimo2_e2e --two-card'

# Head-256 pack-one prefill, whole inputs and symmetric windows, against the f64 attention band.
gate-gpu-gemma-embedding-head256:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_gemma_embedding_head256 && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemma_embedding_head256'

# EmbeddingGemma 2 beside Qwen3-30B-A3B Q4_K_M: server vectors, placement cost, pool reserve,
# greedy tokens with embedding calls in flight, and named refusals; the runner holds one card.
[group('pool')]
gate-gpu-gemma-embedding2-serve *ARGS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-gemma-embedding2-serve && bash tools/gate.sh -p bloomery-gpu-gates --features hf --test sentence_prompts && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_gemma_embedding2_serve && D=target/embedding2-serve-gate-$(date +%s)-$$ && mkdir -p "$D" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemma_embedding2_serve --dir "$D" {{ARGS}} && if [ -z "{{ARGS}}" ]; then BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemma_embedding2_serve --dir "$D/audio" --audio; fi'

# EmbeddingGemma 2's encoder (gpu-gemma-embedding2) end to end on the Q8_0 file the sets are dumped from
# (refset's gemma_embedding2::MODEL, held to its sha256; the sets' `# model` line names the same path): every
# text case of mainline's card set (just dump-ref-emb2) through one pass, its taps within the CPU twin's band,
# a rerun bit for bit, and the refusals by name. The
# key bounds first hold every key of the symmetric mask, independently of the tap bands.
# Both text encoders use the same pass and serving gates, with their own pinned sets.
gate-gpu-gemma-embedding-e2e:
    BLOOMERY_MODEL=gemma-embedding ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gemma-embedding2,vision,deepseek41 --release --bin gate_gemma_embedding2_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemma_embedding2_e2e --v1'

gate-gpu-gemma-embedding-serve:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-gemma-embedding-serve && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_gemma_embedding2_serve && D=target/embedding-serve-gate-$(date +%s)-$$ && mkdir -p "$D" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemma_embedding2_serve --v1 --dir "$D"'

gate-gpu-gemma-embedding2-e2e *ARGS:
    ./tools/box.sh 'bash tools/gate.sh --gpu --oxide --release -p bloomery-gpu-gemma-embedding2 --lib -- --include-ignored'
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::gemma_embedding2::spec::tests::key_bounds_are_the_symmetric_mask --exact'
    ./tools/box.sh 'bash tools/gate.sh --oxide --release -p bloomery-gpu-gates --features gemma-embedding2,vision,deepseek41 --bin gate_gemma_embedding2_e2e'
    BLOOMERY_MODEL=gemma-embedding2 ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gemma-embedding2,vision,deepseek41 --release --bin gate_gemma_embedding2_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemma_embedding2_e2e {{ARGS}}'

# Gemma4 vision's mainline card and CPU sets, including every block's rows.
# Every layer and projected row is held to its CPU/card twin distance.
gate-gpu-gemma4v-tower:
    BLOOMERY_MODEL=gemma-embedding2 ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features vision --release --bin gate_gemma4v_tower && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gemma4v_tower'

# GLM-5.3-Flash MLA 층의 카드 부분 — 잠재 RMS + f16 추가, 인덱서 키 LayerNorm + [키; 게이트] 추가(`latent`),
# 창 0·싱크 −∞·스케일 1/16으로 다시 쓰는 V4.1 어텐션 — 을 모델 없이 합성 입력으로 호스트 규칙에 맞춘다.
# 추가는 캐시 비트, 어텐션은 오차 모형에서 끌어낸 밴드, 싱크 −∞는 분할 부분합의 무싱크 접기와 비트로 본다.
gate-gpu-glm-mla:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_glm_mla && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm_mla'

# GLM-5.3-Flash's k-pool selector on the card (`latent::index_pool`, `kpool::kpool_score`, `qsa::qsa_topk_high`, the
# V4.1 attention over the list): synthetic clauses against the host rules and the exact rule's bands, then every
# latent layer teacher-forced on ik's --dsa step sets (refset `ik-glm5next-dsa`, `just dump-ref-glm5next d3kdsa`
# and `d16kdsa`). Reads no weight but the file's `indexer_compressor_ape`. The two --dsa sets are ik's: Oracle clauses, deferred
# to the real tier by name in the fixture tier; the synthetic clauses run on either file.
gate-gpu-glm-sel:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_glm_sel && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm_sel'

# GLM-5.3-Flash's prompt projections on the tensor-core GEMM (`gpu-glm5next/src/gemm.rs`, no engine caller yet): a KDA
# layer's seven and a latent layer's stack over 512 fixed-seed columns, on the file's own weights (only those tensors
# resident, about 0.3 GB), every output bit for bit the host transcription of `gemm_q8_0p`'s contract
# (`gpu-gates/src/gemm32.rs`); one quantize per input; the refusals by name. Every clause is a self-consistency clause,
# so the fixture tier runs them all on the fixture's layers.
gate-gpu-glm5next-gemm:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_gemm && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_gemm'

# MiMo-V2's attention kernels: the rope-only append and the K192/V128 scalar flash with the window and the sinks
# (A1-A2, F0-F9 on seeded data; G1-G2 on ik's ref_mimo2* sets, each opened through the ik-mimo2 family check against
# the mimo2 profile's model file).
gate-gpu-mimo2-flash *ARGS:
    BLOOMERY_MODEL=mimo2 ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_mimo2_flash && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_mimo2_flash {{ARGS}}'

# Flint's head-128 flash with a window and sinks, and the no-rope q/k RMS append.
# Model-free: random inputs against an f64 reference. No model file.
gate-gpu-flint-attn:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_flint_attn && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_flint_attn'

# Flint's floor-normalized top-six router: optional selection bias, rows and captured replay.
gate-gpu-flint-router:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_flint_router && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_flint_router'

# LFM3's head-128 flash at a group of four query heads a key head (window 256 and none, plane and ring, no sinks), against an f64
# reference and bit for bit against the group of eight. Model-free: seeded inputs. No model file.
gate-gpu-lfm3-kernels:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_lfm3_kernels && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_lfm3_kernels'

# Full-file eager/captured, resident-slot and ring-retention consumer transitions.
gate-gpu-flint-lib:
    ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-flint-lib && bash tools/gate.sh --gpu --oxide --release -p bloomery-gpu-flint --lib -- --include-ignored --exact tests::hw_full_file_capture_slots_reset_and_ring_boundary'

# Real two-card top-six prompt/decode, union reference, capture and tier fault recovery.
[group('solo')]
[group('v41-load')]
gate-gpu-flint-twocard:
    BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-flint-twocard && export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gate.sh --gpu --oxide --release -p bloomery-gpu-flint --lib -- --include-ignored --exact two_card_tests::hw_two_card_full_file_union_capture_publication_fault_reset'

# Muse Spark's 256-expert top-8 sigmoid router (`sigmoid_256_8`). Model-free like the Flint one,
# plus the bias clause: the selection follows the biased scores, the weights the unbiased ones.
gate-gpu-muse-router:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_muse_router && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_muse_router'

# qwen3moe 전 체인 게이트(3090 한 장): 노드 수 핀, 층별 teacher-forced·자유 주행 l_out 대조, greedy(ik-greedy-qwen3moe의
# 파일), graph = eager. 텐서 코어 플래시(엔진 경로)로 한 번 돈다. 스칼라 패스는 (u)의 자로 eager 안에서만 돈다.
[group('pool')]
gate-gpu-qwen3moe-e2e:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen3moe-e2e && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3moe_e2e'

# qwen3moe routed types under the --place-unset defaults: for every type the host's routed kernels run, a crafted
# two-layer header planned by the CLI's and the serve seat's defaults. A refusal, or card experts of a type the card
# table does not run, fails by name. The synthetic layout with MXFP8 attention also loads whole and placed;
# no external model file is needed, so the fixture tier runs it, under the default profile: no family fixture's card budget applies.
gate-gpu-qwen3moe-types:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_types && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3moe_types'

# routed card layers vs the card leg, over the census stubs: every family's (qwen3moe, qwen35moe, qwen4exp, glm5-next,
# mimo2, deepseek41) --place-unset production defaults (the CLI's and the serve seat's, per family) plan each set's
# header-only stub at derived card rooms (the floor of a CardBudgetFloor refusal, then floor + f·E), and every layer a
# plan gives card experts must be one leg_runs runs for the file's own architecture; one set per family is retyped
# through the sweep triples (127 headers: every T×T×T of five types plus the two gate≠up mixed triples, the
# block-mis-dividing ones skipped by name; V4.1 also its exact leg and the Q6_K down) and planned the same way. Where a
# family's loader reads a plan-time stack table (Qwen3.8's swap38::stacks, MiMo's mimo2::swap::stacks) it must take
# every plan L1 passes. Plan only: no weight loaded, no card opened, no model file opened (the census manifest names the
# stubs, by default tools/census/stubs.tsv under /models), so the fixture tier runs it as the real tier does. The gate
# runs on the 3090 (gpu-gate.sh's default pin, lane A in a batch): the arms plan on the census they see, and an `any`
# run on the A6000 would plan its bytes instead of the gate machine's. The recipe's ARGS carry the bin's
# --manifest/--root.
gate-gpu-routed-legs *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,mimo2 --release --bin gate_routed_legs && bash tools/gpu-gate.sh gate_routed_legs {{ARGS}}'

# ik CPU의 greedy 연속(프롬프트 0–7, 32토큰)을 $BLOOMERY_DATA/qwen3moe/greedy/에. 프롬프트마다 CPU 임대를 잡는다.
ik-greedy-qwen3moe:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'for p in 0 1 2 3 4 5 6 7; do GEN=32 PROMPT=$p bash tools/ref/ik-greedy.sh || exit $?; done'

# ik KLD 기준 파일 TAG에 대한 qwen3moe 체인의 NLL·KLD·top-1(출력만, 판정 없음).
run-qwen3moe-ppl TAG:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_e2e && BLOOMERY_GATE_BOUND=1680 bash tools/gpu-gate.sh gate_qwen3moe_e2e --ppl {{TAG}}'

# ik-ppl을 qwen3moe 파일에(MODEL을 프로필 값으로 넘긴다). ARGS는 ik-ppl.sh 머리글.
ik-ppl-qwen3moe TREE TAG *ARGS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'MODEL=$BLOOMERY_REF_MODEL bash tools/ref/ik-ppl.sh {{TREE}} {{TAG}} {{ARGS}}'

# qwen3moe 생성 CLI(3090): --prompt 텍스트 또는 --tokens id 목록, -n, --ctx, --mode, --time.
gen-qwen3moe *ARGS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe && bash tools/gpu-gate.sh generate_qwen3moe {{ARGS}}'

# V4.1 호스트 expert 티어 게이트(B5). moe::Meta가 파일의 expert 384개를 Hparams와 같게 읽는지 확인한 뒤, 5토큰 세트의
# 모든 층에서 ik의 라우팅을 주입해 호스트 티어가 낸 routed 부분합을 ik의 ffn_moe_out과 도출한 밴드 안에서 대조한다
# (플립은 따로 세고, 층마다 세트가 닿은 클램프 히트 수를 찍는다 — 어느 층이 닿는지는 파일의 것이다). 클램프 자체는
# 층마다 합성 입력으로 한계 너머까지 따로 본다: 토큰 0의 라우팅을 그대로 두고 입력 행을 2의 거듭제곱으로 키워
# silu(g) > L, u > L, u < −L을 모두 넘긴 뒤 combine이 qdot::swiglu_clamp와 비트 동일하고 f64 서술 안인지 확인한다.
# 세트가 라우팅한 expert만 읽는다. 끝으로 크레이트 doctest —
# ops::ShardTensor의 compile_fail(다른 샤드를 가리키는 핸들은 만들 수 없다).
gate-ds41-host:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test ds41_host -- --ignored --nocapture && bash tools/gate.sh --release -p bloomery-model --doc'

# 호스트 합집합 호출 게이트: moe::HostLayer::experts_union_into가 토큰 열 k = 1·2·3·6에서 열마다
# experts_into(그 열, 그 열의 목록)와 비트까지 같은지를 V4.1 라우팅 층 40개 전부(ref_deepseek41의 라우팅)에서
# 지연 팔 둘로 재고, 파일 하나를 받는 진입은 V2-Lite 블록 1에서, 이름 붙은 거부 여섯은 각각 확인한다.
# r8 배치: gate·up을 r8 사이드카에서 행-레인 타일로 읽는 층이 파일 행을 읽는 층과 열마다 비트까지 같다
# (합성 층 하나, 모델 파일 없이 박스 CPU의 AVX2에서).
# --test-threads=1: 두 테스트가 프로세스 전역 지연 레버(ops::set_defer_quant)를 뒤집는다.
gate-union:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test union -- --ignored --nocapture --test-threads=1'

# The gate fixture generator (model::fixture, bin fixture of crates/model-hf) on the V4.1 spec (model::arch::deepseek41::fixture) against
# the real file's header and the real DSpark draft's: the plan (the map's nine layers, the metadata overrides, the engine
# reading the header-only files as the source's layer kinds), determinism, every sampled block's scales and RMS, small
# subset files written, opened and verified end to end (under 1 GB each, removed), and the named refusals. The Qwen3.8 spec
# (model::arch::qwen35moe::fixture) against the Qwen3.8 file's header and the shared MTP draft's, at the profile's paths:
# the four-layer plan read back by the engine's hparams at interval 2, the planned card budget (the written file's plan
# under it holds half the experts on every layer the card loads), and the ik-silent traps refused by name. The Qwen3.6 spec
# (model::arch::qwen35moe::fixture::qwen35moe, the lmstudio Q4_K_M file) against its header: the four-layer plan read back by
# the engine's hparams at interval 2 (the shared expert's gate at 1/sqrt(2048), the split keys of the one-file source, a draft
# refused by name), the planned card budget (the written file's plan under it holds 128 of the 256 experts on every layer)
# and the source traps its rules refuse by name (no interval, another width, an uncovered key); its rules and its map's layer
# kinds are also the model lib's tests (gate-ops). The V4.1 spec's
# ff 512 (the plan, the draft's ff, the routed stacks and the r8 sidecar's stacks), its planned budget (half the experts
# at the gate machine, ctx 32768) and its two router biases that differ between experts (the family's header check refuses
# zeros, equal and a constant shift); the GLM-5.3-Flash spec (model::arch::glm5next::fixture) against the GLM file's
# header: the seven-layer plan read back by the engine's hparams (kinds, dense prefix, NextN believed), the planned budget
# under the NextN load (half the experts on each card-eligible layer) and the traps its reader and the kernels' constants
# fall into, refused by name; the Nemotron 3 Nano 4B spec (model::arch::nemotron_h::fixture) against the Nemotron file's
# header: the five-layer plan read back by the engine's hparams (kinds, every stem and type pair of the source), the written
# fixture (about 0.93 GB) verified against it, and the maps that lose a kind or a pair refused by name; the Nemotron 3.5 Bolt
# spec (model::arch::nemotron_h_moe::fixture) against the Bolt file's header: the four-layer plan read back by the engine's
# hparams (kinds, the two MoE layers' expert shapes, expert_count the map's largest router, every stem and type pair of the
# source), the written fixture (about 2.16 GB) verified against it, and the maps that lose a kind or the two shapes refused
# by name; the Solar-Open2-250B spec (model::arch::solar2::fixture) against the Solar file's header: the four-layer plan read
# back by the engine's hparams (kinds, KV heads, every stem and type pair of the source), the planned budget (half the
# experts on each card-eligible layer, none on the Q6_K-down layers), and on a small synthetic source the written fixture's
# constants and the maps that lose a mixer or a pair refused by name; the Qwen3-30B-A3B spec (model::arch::qwen3moe::fixture)
# against the Instruct-2507 Q4_K_M file's header: the four-layer plan read back by the engine's hparams (every stem and type
# pair of the source, the Q6_K layers kept), the written fixture (about 1.95 GB) verified against it with a flipped byte
# refused by name, and the maps that lose a pair refused by name; and a toy family's r8 sidecar written, checked and
# refused when absent, other or flipped.
# Reads headers only; writes only under this tree's target/tmp (the Nemotron clause writes its one 0.93 GB file there, the
# Qwen3-30B-A3B clause its one 1.95 GB file).
gate-fixture:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-fixture && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gate.sh --release -p bloomery-model-hf --test fixture --bin fixture -- --include-ignored --nocapture --test-threads=1'

# Every fixture written under /models/fixtures (BLOOMERY_FIXTURE_ROOT) holds the card budget its header records: for each
# family with a planned budget, the plan of the written file under it is half the experts on every card-eligible layer
# (model::fixture::check_budget, the re-plan `fixture verify` runs, here from the written header alone: no source file, no
# bytes, no card). A plan term that moves after a fixture was written (a table the card holds, a reserve) leaves the
# recorded number a granule short and every gate that plans under the header sees a layer one expert off; this names the
# family and the counts. Either tier: it reads the files as they are.
gate-fixture-budget:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model-hf --bin fixture -- --ignored hw_the_written_fixtures_hold_their_recorded_budget --nocapture'

# The architecture registry (crates/model/src/arch/registry.rs): every architecture string is in one row and every `Arch`
# in one row, a fixture row declares its row's architecture, the draft rules are the three rows that have one,
# tools/arch-registry.tsv (which tools/hf-arch-check.py reads) is the registry's print, and each family's written fixture
# (the nine GGUF ones under /models/fixtures, BLOOMERY_FIXTURE_ROOT; a cut folder is read by its own reader) reads through the registry exactly as its family's own reader
# does. Headers only, no card. Either tier: it reads the fixtures as they are.
gate-arch-registry:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::registry --include-ignored --nocapture'

# 1-5 스레드 풀 게이트: 상주 워커 풀의 분할 전수·커버리지·반복 호출·패닉 전파.
# hw_ 토폴로지 테스트는 #[ignore]라 --include-ignored로 같이 돈다. The pool's own unit tests (--lib: the L3 size
# reading, the CCD map's pin condition) run with it.
gate-threads:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-threads --lib --test pool --test helper -- --include-ignored --nocapture'

# The OS-call owner (crates/os, host only): each call's Linux body runs here; its macOS bodies and their named
# refusals run on a Mac through `just mac-test`.
gate-os:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-os --lib -- --nocapture'

# The device contract's pure arithmetic and its compile-fail doc tests: window ranges and split_at, Elems against Bytes, the fault word's
# rules on a host model of the word. No card.
gate-device:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-device'

# The record core (crates/record, host only): the one owner of a line's syntax, its schema and its reader.
gate-record:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-record --lib'

# The device contract's clauses on the card (bloomery_device::conformance, the set gate_metal_device runs on a Mac): buffers,
# windows and casts, copies in stream order, the shared pages a kernel sees, events and a held stream, the fault word against
# the contract's host model, the census. No model file.
gate-gpu-device:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_gpu_device && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_gpu_device'

# The chain plans' lib tests (crates/chain, host only): a layer's kind and the refusals that name what no kernel runs.
gate-chain:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-chain --lib -- --nocapture'

# 레버 레지스트리 게이트(crates/levers, 호스트만): 모든 행이 형식에 맞는가, 종류가 받지 않는 값과 은퇴한 이름을
# 이름을 대고 거부하는가, 제자리 행과 tools/levers-direct.txt가 서로 맞는가. 레지스트리를 마크다운 표로 찍는다.
[group('canary')]
gate-levers:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-levers --lib -- --nocapture'

# 샘플러 게이트: greedy·top-k=1이 엔진 argmax와 같고, top-p·min-p·반복 벌점이 참조의 정의대로 자르고,
# 추첨 빈도가 소프트맥스를 따르며, 같은 시드는 같은 토큰을 내고, 첫 호출 뒤 할당이 0인가. 박스 자원 불필요.
gate-sampler:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-sampler --test sampler -- --nocapture'

# The generation loop (crates/runtime, host only): the stop rule, the lookup draft, and the drafts'
# pick/commit arithmetic against a mock target — plain = draft tokens, a rejected row taken back,
# only kept tokens in the lookup's context, a failed call not run again.
[group('canary')]
gate-runtime:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-runtime --lib -- --nocapture'

# The model description crate (crates/models, host only): the shape selector's rows — every model
# the tree runs selects its router and flash instance, a shape between or outside the rows is
# refused by name, no two rows overlap — and the shipped MTP head list's miss rate on each held-out
# language set (tests/head_coverage.rs: under the break-even, held-out text no input of the build).
[group('canary')]
gate-models:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-models --lib --test head_coverage -- --nocapture'

# The session crate's host tests (crates/app, a device-linked crate, so cargo oxide test): a fault
# read at a step, a group end or a call end is one SessionError::Fault. They open no card (host group).
[group('canary')]
[group('host')]
gate-app:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-app --release --lib'

# The Qwen planner's host-only default-context refusal and fit contract. Opens no card.
[group('host')]
gate-qwen3-plan:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe'

# 2단계 qdot 게이트: Q3_K×Q8_K 융합 커널이 현재 경로(dequant+roundtrip+f32)와
# 1e-5 안팎에서 일치하고, 정확해(f64)에 더 가깝고, 스칼라 폴백과 비트 동일인가.
# 순수 게이트(rejects_unaligned_k)는 #[ignore]가 아니라 --include-ignored로 같이 돈다.
# `--test generic` holds the generic host dot (qdot::generic) to the hand kernels' bits and the type rows' activation
# forms to the kernel list.
gate-qdot:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-qdot --test qdot --test generic -- --include-ignored --nocapture'

# The quant gate: every gguf::QROWS row against llama.cpp mainline's quant oracle ($BLOOMERY_DATA/ref-quant, `just dump-ref-quant`):
# (a) the sizes the harness wrote, (b) dequant_row bit-equal to to_float, (c) the host activation encoder bit-equal to from_float
# of ggml's vec_dot_type where the row's host_act is ggml's, (d) qdot's scalar and AVX2 dots against vec_dot, bit-equal for the
# types listed in BIT_EQUAL and within the band derived in crates/qdot/tests/quant_rows.rs for the rest. The set's table of ggml's
# types is held both ways: a QROWS row ggml's table does not name, and a type it names with no QROWS row, are red by name; a type
# the table states skipped (no to_float, no vec_dot) must be a row with no decoder and no CPU matmul. `just gate-quant q4_K` takes
# one type's rows (ggml's type name, comma-separated for several) through QUANT_ONLY; a name that is neither a row nor a type of
# the table ends the test with a named error. No model file, no card.
gate-quant TYPE='':
    ./tools/box.sh 'export QUANT_ONLY={{TYPE}} && bash tools/gate.sh --release -p bloomery-qdot --test quant_rows -- --include-ignored --nocapture'

# B3 engram 게이트: V4.1의 NVMe 테이블에서 매핑으로 읽은 행이 같은 오프셋의 pread와
# 바이트 동일이고(선행 읽기를 걸어도 같고), 행 간격이 텐서를 정확히 타일링하는가.
# 정확성 실행이라 임대는 필요 없다 — 속도는 engram-rate가 임대 안에서 잰다. --lib은 prefetch의 helper 배치
# 단위 테스트(한 cpu 마스크를 물려받은 float helper가 마스크를 넓히는가)를 같은 게이트에 넣는다.
gate-engram:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-engram --lib --test engram -- --include-ignored --nocapture'

# The tokenizer gate: our ids equal `llama-tokenize`'s on every engram corpus text and every case of
# crates/tokenizer/tests/cases.txt, in both parse modes, and the reference's ids decode back to the text, for
# the seven vocabularies the engine runs: V4.1 (deepseek-v3), Qwen3-MoE (qwen2), GLM-5.3-Flash (glm4),
# Qwen3.8-Flash-Next (qwen35), Nemotron 3 Nano 4B (pixtral), MiniMax-M3 (minimax-m2) and Muse-Spark-Open
# (gpt-4o). It only reads the sets: each opens through its refset family
# (crates/refset/src/arch/tokenizer.rs), which refuses by name a set written from another reference tree (commit or patch), from
# another vocabulary file or other cases, with a text that is not the one dumped, or never finished.
# `just dump-ref-tokenizer` writes them. EmbeddingGemma v1's llama SentencePiece cases also compare with
# BOS/EOS, in both parse modes, against the pinned mainline set and its extra cases.
# The Llama-3 BPE vector test reads llama.cpp mainline 6868a52cb's models/ggml-vocab-llama-bpe.gguf with its .inp and
# .out, copied into $BLOOMERY_DATA/vocab-llama-bpe/.
gate-tokenizer:
    ./tools/box.sh 'BLOOMERY_LLAMA_BPE_GGUF=$BLOOMERY_DATA/vocab-llama-bpe/ggml-vocab-llama-bpe.gguf bash tools/gate.sh --release -p bloomery-tokenizer --lib --test tokenizer -- --include-ignored --nocapture'
# The tokenizer oracle's sets, once per change of the reference trees, the vocabulary files, cases.txt or the
# docs/ the Korean text is gathered from: `llama-tokenize`'s ids per vocabulary into $BLOOMERY_DATA/ref-tokenizer-<v>/
# (crates/tokenizer/tools/oracle.sh; VOCAB is v41, qwen3moe, glm5next, qwen38, nemotron, minimaxm3, musespark, gemma4, llama or
# all, every one but gemma4 and llama). No lease, no GPU. gemma4 (EmbeddingGemma 2) and llama (v1) use mainline 6868a52cb
# (`just build-emb2-lcpp`), with their profile's EMBED_CASES after cases.txt. The EmbeddingGemma oracle sets read those ids.
# Every vocabulary is held to the tree it is read by: before the dump the tree's commit, its patch and its build are checked against
# the pin crates/tokenizer/tools/oracle.sh names (tools/ref/emb2/oracle-tree.sh), and the manifest's `# build` line states it
# (`<commit>`, or `<commit>+<patch>` for the two ik side trees) for the refset family to pin, which any rebuild of that tree keeps.
# A re-dump on a moved tree changes that pin in oracle.sh and crates/refset/src/arch/tokenizer.rs: the gate names which.
# llama's corpora go in rows of at most 1 MB, its tokenizer being quadratic in a text.
# V4.1's reference is a side tree (/home/user/ik-tilde: ik main plus the one line that puts `~` in the S class,
# ik#2528) and Qwen3-MoE, GLM-5.3-Flash and Qwen3.8 use /home/user/ik-tokref (the ik-idxkey commit plus the
# unicode_tolower fix, ik#2520). Nemotron uses /home/user/llama.cpp-19cce691: ik-tokref has no nemotron_h
# architecture, and mainline reads its `pixtral` pre as TEKKEN from e197cef45 on. MiniMax-M3 uses /home/user/llama.cpp-solar2
# (mainline 6ea215d17 plus the Solar-Open2 patch), whose lenient UTF-8 decoder the crate follows.
# Pass --cases-only to dump the fixed texts without the large corpus snapshots.
dump-ref-tokenizer VOCAB='all' *ARGS:
    ./tools/box.sh 'timeout --kill-after=10 900 bash crates/tokenizer/tools/oracle.sh {{VOCAB}} {{ARGS}}'
# Embedding pooling, request parsing and response bytes against the pinned llama.cpp fixtures, no card.
[group('canary')]
gate-embed:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-embed --lib --test request --test oracle -- --nocapture'

# Clef's and lev's host gates (crates/decision, host only), one cargo call in the release profile at the feature set the Clef head
# reads its weights with: the lib's tests, then both test targets, every one run even when an earlier one is red.
# clef: Clef's request encoder, joint schema head and SystemOne answer against the
# release's own Python (tools/ref/clef_ref.py's files under $BLOOMERY_DATA/clef/flash/ref): ids and spans
# exact, f32 logits inside the derived band of the f64 referee, the body's shape and top options.
# lev: lev's label readout against llama.cpp mainline's server (tools/ref/lev_ref.py's files
# under $BLOOMERY_DATA/lev/4b/ref): the label list and every prompt's ids of the 14 requests equal the server's, the
# answer math by hand-computed cases (temperatures, buckets, the variants' softmax average, the rating scale), and the
# policy (option order, prompt shape, refusals).
gate-decision:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-decision --features safetensors --no-fail-fast --lib --test clef --test lev -- --include-ignored --nocapture'
# lev's oracle: the patched mainline server answering tools/ref/lev/{suite,edge}.jsonl on the lev file, its prompts' ids
# and bodies into $BLOOMERY_DATA/lev/4b/ref (tools/ref/lev_ref.py; the server is built by tools/ref/build-lcpp-lev.sh,
# the file fetched into /root/models/lev-4b/). A functional dump, no lease: the server runs through tools/gpu-gate.sh
# (tools/ref/lev-dump.sh), under a card's gate lock and its bound, on the card the runner picks. OUT is the directory
# written (another one shows the dump is the same without replacing the gates' files).
dump-lev OUT='$BLOOMERY_DATA/lev/4b/ref':
    BLOOMERY_MODEL=qwen35 ./tools/box.sh 'mkdir -p {{OUT}} && bash tools/ref/lev-dump.sh --suite tools/ref/lev/suite.jsonl --edge tools/ref/lev/edge.jsonl --out {{OUT}}'
# HTTP 서버 게이트(모의 엔진): llama-server JSON 형태, SSE 프레이밍, 정지 규칙(정지 id 목록 전부), V4.1·GLM 채팅 템플릿 렌더링,
# GLM 도구 호출 파서(glmxml), Qwen3의 헤르메스 호출·모델이 여는 생각 범위(hermes), 연결 상한·유휴 연결 종료·컨텍스트 끝의 정지(limits). 박스 자원 불필요.
# Anthropic's Messages API on the chat path (anthropic). Qwen3.6's and Qwen3.8's XML tool calls, and every seat
# template's tool-call markup mapped to its parser (qwenxml). A tool call's streamed fragments in every markup
# (callstream). The repetition, frequency and presence penalties as llama-server takes them (penalty). A sampled
# request's drafted passes against its plain steps (sampdraft).
# A chat clip and an embeddings clip against a mock decoder (src/media, src/api, src/arch/gemma4v): both part spellings, the
# plan before the text, the placeholders paired by id and order, an embeddings input's images and audio charged before its clips
# are planned, every named refusal, `/props` `modalities.video`. The decoder's own subprocess rules (src/video): an ffprobe past
# its limit and an ffmpeg that does not end after its last frame are killed and named; a decode is taken only when ffmpeg ended well.
gate-serve:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-serve --lib --test serve --test dsml --test glmxml --test hermes --test qwenxml --test callstream --test limits --test anthropic --test penalty --test sampdraft --test contract -- --include-ignored --nocapture'

# The server's video decode against a real ffmpeg (crates/serve/tests/video.rs): a lossless clip whose every frame is one colour
# naming its index, probed (its frame count counted by decoding where the container states none), planned by a tick rule
# and a linspace rule and decoded at the plan's indices at one and four threads, the same frames; a clip with audio longer than
# its video probes the video's frames; a decode past the file's end and a file that is no video as named errors; a two-stream
# clip decodes the stream the probe measured; a clip with one damaged packet, its header intact, is a named error with no
# frame handed on; a 60 s 1080p H.264 clip's plan arrives whole and in order. Needs ffmpeg and
# ffprobe 5.1 or newer on PATH (the gate fails by name without them). No card, no lease.
gate-serve-video:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-serve --test video -- --include-ignored --nocapture'

# The template engine (crates/jinja, host only): its unit tests, the HF environment's whitespace control, scoping,
# macros, filters and tests, Python's `str()` and slicing rules, and the V4.1 template rendered by hand. The engine's
# jinja2-parity cases are gate-serve's. No network, no card.
[group('canary')]
gate-jinja:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-jinja --lib -- --nocapture'

# engram IO 실험실의 시험(crates/engram-lab, 엔진 사용처 없음): 컨텍스트 창의 슬롯 순서, 행 캐시의 LRU를 스택
# 거리 모의와 대조, 캐시가 내주는 바이트. 이름이 gate-가 아니라 lab-이라 `just affected`가 엔진 착륙에서 고르지
# 않는다. V4.1 분할 파일과 engram-corpus 스트림이 있어야 돈다.
lab-engram:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-engram-lab --test lab -- --include-ignored --nocapture'

# The parser fuzz crate (fuzz/, excluded from the workspace): its own tests, then each target for 60 s on its
# seeds, as fuzz/README.md runs them (tools/fuzz-lab.sh). The name is lab-, not gate-, so `just affected` never
# selects it, and a finding is a bug to fix, not a landing's red. Needs `cargo install cargo-fuzz --locked` on
# the box (the pinned toolchain and a C++ compiler it has); derived wall ~6 min of fuzzing (six 60 s targets) plus
# the test build and one sanitizer build, cold. `just deny` does not scan fuzz/, so fuzz/Cargo.lock is outside
# its advisory and license check; libfuzzer-sys is NCSA, which deny.toml does not allow, and nothing ships from
# fuzz/.
lab-fuzz:
    ./tools/box.sh 'bash tools/fuzz-lab.sh'

# engram-rate 바이너리. 측정은 tools/ref/engram-rate.sh가 임대 안에서 돌린다 —
# 이 레시피는 빌드만 한다(러너가 낡은 바이너리를 재는 것을 막는 단계).
build-engram:
    ./tools/box.sh 'cargo build --release -p bloomery-engram-lab --bin engram-rate'

# engram 토큰당 비용 표(기본 팔 여덟 — 캐시 팔은 --ids와 함께 --arms로 부를 때만 돈다). 기계 전역 임대를
# 잡으므로 리드가 조용한 시점에 친다.
measure-engram *ARGS: build-engram
    ./tools/box.sh 'bash tools/ref/engram-rate.sh {{ARGS}}'

# engram-reuse가 읽을 토큰 스트림. 참조 엔진의 토크나이저가 vocab만 읽는다 —
# 측정이 아니라서 임대도, 조용한 기계도 필요 없다(초 단위).
engram-corpus *ARGS:
    ./tools/box.sh 'bash tools/ref/engram-corpus.sh {{ARGS}}'

# ik의 답(프롬프트 32개의 greedy 다음 토큰). 오라클과 달리 파일 하나만 쓰고
# $BLOOMERY_DATA/ref는 건드리지 않는다 — argmax.sh가 끝에서 매니페스트 해시로 확인한다.
build-argmax:
    ./tools/box.sh 'bash tools/ref/build-argmax.sh'

argmax-ref:
    ./tools/box.sh 'bash tools/ref/argmax.sh'

# kv-clear / warmup 재현자. ik가 문맥의 n_eval이 0인 동안 들어온 "BOS 한 토큰" 디코드를
# 워밍업 그래프로 지어 전문가를 전부(64개) 돌린다 — 그 그래프가 쓴 KV가 그 시퀀스 전체를
# 바꾼다. 팔마다 그 술어의 접속사 하나씩만 뒤집는다. 산출물은 스크래치로만 간다.
build-kvclear:
    ./tools/box.sh 'bash tools/ref/build-kvclear.sh'

kvclear-probe *ARGS:
    ./tools/box.sh 'source tools/ref/ref-paths.sh && CUDA_VISIBLE_DEVICES= /root/bloomery-scratch/ikclear/bin/kvclear_probe -m "$MODEL" --prompts tools/ref/prompts.tsv -ngl 0 -c 512 -t 32 {{ARGS}}'

# Stage 1-1 gate: the dequantization oracle's sets against the gguf crate's hw tests: types, tensors and rows of
# the V2-Lite file and of the V4.1 file's first shard, and the synthetic rows of the types no model file on the box
# holds (q2_K, iq2_xs, iq3_xxs, iq4_xs), each bit for bit against ggml's `to_float`; --include-ignored brings the
# split reader's and the inventory's plain tests along. The sets open through their refset families
# (crates/refset/src/arch/dequant.rs), which refuse by name a set written by another harness or ggml build, from
# another model file, with a file that is not the one dumped, or never finished; `just dump-ref-dequant` writes
# them. gen-iq-tables.py --check re-extracts the i-quant codebooks (iq_tables.rs) from ik's headers and compares
# them with the committed file byte for byte.
gate-1-1:
    ./tools/box.sh 'source tools/ref/ref-paths.sh && python3 tools/ref/gen-iq-tables.py --check --ik "$IK" && bash tools/gate.sh -p bloomery-gguf -- --include-ignored --nocapture'
# The dequantization oracle's sets, once per change of the ggml build, dequant_ref.cpp or the model files: builds
# the harness (tools/ref/build-dequant.sh) and dumps $BLOOMERY_DATA/ref-dequant-v2lite (V2-Lite),
# ref-dequant-v41<suffix> (the V4.1 file) and ref-synth (`--synthetic`), each with its DEQUANT.tsv
# (tools/ref/dump-dequant.sh). No lease, no GPU.
dump-ref-dequant:
    ./tools/box.sh 'bash tools/ref/dump-dequant.sh'

# The quant oracle's harness (tools/ref/quant_oracle.cpp) linked against llama.cpp mainline's libggml-base and libggml-cpu in the
# Qwen image-input oracle's tree (`just build-qvis-lcpp`; checked at its pin, never rebuilt here) into
# $BLOOMERY_DATA/bin/quant_oracle (tools/ref/quant-oracle.sh). No lease, no card.
build-quant-oracle:
    ./tools/box.sh 'bash tools/ref/quant-oracle.sh build'

# The quant oracle's set, once per change of quant_oracle.cpp or the mainline tree: for each ggml weight type and row length,
# seeded weight rows (quantized by ggml and random-code rows), their to_float, activation rows in the type's vec_dot_type and
# every vec_dot, under $BLOOMERY_DATA/ref-quant with its QUANT.tsv. The set opens through its refset family
# (crates/refset/src/arch/quant.rs), which refuses by name a set of another harness or mainline commit, of a model file, with a
# file that is not the one dumped, or never finished. Builds the harness first. No lease, no card; `just gate-refset` reads it.
dump-ref-quant:
    ./tools/box.sh 'bash tools/ref/quant-oracle.sh dump'

# 커밋 전에 치는 것. 측정은 포함하지 않는다(조용한 기계가 필요하다).
# The static checks come from check-scripts, not from a name here. The other deps are one a line, so two additions do not touch one line.
gate: \
    check-scripts \
    fmt-check \
    lint \
    gate-1-1 \
    gate-vision \
    gate-gpu-gates-lib \
    gate-gpu-lib \
    gate-sampler \
    gate-levers

# `just gate`'s static checks (Mac, text only): every tools/check-*.sh but the ones --skip names, in parallel (at most one a cpu),
# each one's output in its own log, one line a check and every red check's log whole; the exit is the first red check's. The list is
# the files, so a new check script joins `just gate` and `just smoke` by existing, with no line to add its name to
# (tools/mac-static.sh --checks has the derivation and the arguments each check takes). emb2, oracle-tree and records have no
# check-* recipe and `just gate` does not run them; delete a --skip to add one.
check-scripts:
    bash tools/mac-static.sh --checks --skip emb2 --skip oracle-tree --skip records

# The smoke tier (https://github.com/midagedev/rig-log/blob/main/docs/bloomery/gates-plan.md 3.1): the static checks (check-scripts, check,
# fmt-check, lint), the V4.1 decode step, the V4.1 prompt batch at
# P = 512 alone, V2-Lite end to end — a subset run of unchanged gates, never the landing batch. Two lanes
# through tools/gate-batch.sh (its header: the list, lanes, logs, DONE line); ARGS go to it (--dry-run, --out DIR,
# --lanes 1). It refuses to start while the timing lease is held.
smoke *ARGS:
    ./tools/gate-batch.sh --smoke {{ARGS}}

# The weekly tier and the rotation: every `weekly-*` recipe (a gate taken out of the landing batch; `just affected`
# selects only `gate-*`, and names a weekly recipe when a file one of its trigger rows in tools/gate-paths.tsv matches
# changes), then every gate the fixture tier defers (real-only, real-file, no-fixture, needs-env) that has no green row
# at its current key and none newer than 7 days (tools/rotation.py due), and every gate of an F family likewise, in one sitting through
# tools/gate-batch.sh, on the lead's ledger. `just affected` never selects the rotation.
# ARGS go to the batch (--dry-run, --out DIR, --rerun).
weekly *ARGS:
    mkdir -p target && python3 tools/rotation.py due --with-weekly --out target/weekly.list && ./tools/gate-batch.sh --ledger --list target/weekly.list {{ARGS}}

# The rotation's due list alone, as a `--list` file for tools/gate-batch.sh: the gates no landing runs and those of an F family, stalest
# family first, each with its predicted wall. It reads the box once, for the keys.
# ARGS go to tools/rotation.py due (--days N, --classes FILE).
rotation-due *ARGS:
    python3 tools/rotation.py due {{ARGS}}

# The release batch's `--list`: every S0 gate, every gate of an R family, the class-staged `affected` since TAG without the gates of
# the F families (the rotation holds those; each is named on stderr with its last green date) and the rotation's due list, one recipe a
# line; the counts per part (and with --wall the predicted wall) on stderr.
release-list TAG *ARGS:
    python3 tools/recipes.py release-list {{TAG}} {{ARGS}}

# B0b V4.1 인벤토리: 분할 GGUF의 헤더만 읽어 텐서 표를 뽑는다(임대 불필요, 텐서 바이트 미접촉).
# 표는 박스의 /tmp에 쓰고 scp로 회수한다 — 박스 작업 트리에 쓰면 다음 box.sh의 rsync --delete가 지운다.
# 회수한 표는 OUT(기본 target/v41-inventory.md)에 놓인다. 문서 사본(https://github.com/midagedev/rig-log/blob/main/docs/bloomery/v41-inventory.md)은 rig-log docs/bloomery/에 있고
# 이 트리에 쓰지 않는다.
inventory-v41 OUT='target/v41-inventory.md':
    ./tools/box.sh 'cargo run --release -p bloomery-gguf --bin gguf-inventory -- --markdown /tmp/v41-inventory.md "$BLOOMERY_V41_DIR"/*-of-*.gguf'
    mkdir -p "$(dirname {{OUT}})"
    scp "${BLOOMERY_BOX:-ws}:/tmp/v41-inventory.md" {{OUT}}

# GPU 헤드(P8의 head 조각): result_norm → lm_head(Q6_K) → argmax를 그래프 하나로 잡아
# 덤프의 마지막 토큰과 대조(정확성 실행, 핀된 헤드 밴드 안).
gate-gpu-head:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_head_gpu && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_head_gpu'

# m열 커널(q8_0·q8_0 heads·K-quant q3_K·q4_K·q5_K·q6_K·블록 대각 heads·공유 expert gate·up·헤드 argmax): m열 런치의 열 c가
# 그 열 하나를 m = 1로 쏜 결과와 비트 동일한지를 파일의 형식으로 고른 V4.1 실제 행과 합성 K(≤ 8192)에서 m = 1..8 전부
# 본다 — k토큰 스텝이 k개 순차 스텝과 비트 동일하려면 이것이 서야 한다.
gate-gpu-mcol:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-mcol && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_mcol && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_mcol'

# bloomery-gpu's lib tests compile without a lock, then tools/gate.sh --gpu runs the compiled
# test executable under tools/gpu-gate.sh's shared card lock and runtime bound. The hw_ tests
# (graph.rs' capture, replay, host flags and node kinds; checkpoint.rs; host/step.rs' boundary) use device 0.
# A test that holds a stream shut (checkpoint.rs' Hold) or parks one on a wait it opens only after a driver call (host/tier.rs' parking_work) is named
# hw_held_ and runs in a second pass, one at a time: a device-wide sync in a parallel test (any cuMemFree) takes the
# driver's lock while it waits for that stream, and the call that would open it queues behind it. hold::require_serial_pass
# refuses by name anywhere else. Then the lib tests of bloomery-verbs, the RDMA link's verbs, which open no NIC. just gate
# runs it.
[group('canary')]
gate-gpu-lib:
    ./tools/box.sh 'bash tools/gate.sh --gpu --oxide -p bloomery-gpu --release --lib -- --include-ignored --skip hw_held_ && RUST_TEST_THREADS=1 bash tools/gate.sh --gpu --oxide -p bloomery-gpu --release --lib -- --include-ignored hw_held_ && bash tools/gate.sh --release -p bloomery-verbs --lib'

# bloomery-gpu-gates 라이브러리의 단위 시험(호스트 전용, 카드·게이트 락 없음). ptx.rs의 컨테이너 판독 계약 —
# 8의 배수 길이 페이로드의 마지막 본문, 번들 경계, 파서가 거부한 섹션은 빈 표가 아니라 오류 — 이 여기서 돈다.
# gpu 피처 없이 빌드하므로 디바이스 크레이트를 컴파일하지 않고, 그래서 평범한 cargo다. just gate가 부른다.
# probe_nvread's helper tests (batch split, quantiles, aligned spans, the seeded layer draw) run here too: the bin takes no feature.
# The lib runs with the `hf` feature, which adds the model file's fetch (model_file::fetch) and its tests.
[group('canary')]
gate-gpu-gates-lib:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-gpu-gates --features hf --lib && bash tools/gate.sh --release -p bloomery-gpu-gates --bin probe_nvread'

# V4.1 오라클 세트 ref_deepseek41의 모든 행과 디코드 스텝 세트의 헤더를 하니스가 읽는지 본다. 호스트 전용이라 카드도
# 게이트 락도 쓰지 않는다. 시험 둘의 출력이 섞이지 않게 한 스레드로 돈다.
gate-ds41-oracle:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-gpu-gates --lib -- --ignored hw_ds41_oracle --nocapture --test-threads=1'

# The --hf resolver (crates/hf, host only): the repo string, set selection from captured API listings (a split-file
# repo, mmproj and imatrix left out, zero and two matches refused), cache paths, the model named once, and the fetch
# through curl on file:// URLs (a resume from a truncated file, a refetch-free second start, size and digest refusals),
# and the offline start (a refused connection lets the one verified cached set stand in; zero or two, or an HTTP error
# from a local port, are refused). The pre-tokenizer rows' reading (`minimax_encode`: every row of the vocabulary table opens in the
# tokenizer as Hugging Face's `tokenizers` reads it, or is refused by name) runs with it. No network, no card.
[group('canary')]
gate-hf:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-hf --lib --test minimax_encode -- --nocapture'

# The Qwen3.8-Flash-Next vocabulary the folder reader derives (crates/hf, host only; BLOOMERY_QWEN38_HF names nvidia/Qwen3.8-Flash-Next-NVFP4's
# folder, of which it reads the tokenizer files and config.json): the digest of every key `hf::vocab::keys` derives is the one the reader gave
# before the pre-tokenizer rows, and the pre-tokenizer is still `qwen35`. No card.
gate-hf-vocab-qwen38:
    ./tools/box.sh 'bash tools/ref/needs-env.sh gate-hf-vocab-qwen38 BLOOMERY_QWEN38_HF && bash tools/gate.sh --release -p bloomery-hf --lib hw_the_qwen38_vocabulary_is_unchanged -- --ignored --nocapture'

# MiniMax-M3.1-Flash's vocabulary from its checkpoint folder (crates/hf, host only; BLOOMERY_MM31_HF names the folder, of which it reads the
# tokenizer files, config.json and the chat template): the vocabulary `hf::vocab::keys` derives is MiniMax-M3's GGUF vocabulary (the
# `minimax-m2` pre-tokenizer, tokens, token types and merges at their digests, the bos, eos, padding and unknown ids and the stop list; id
# 16474 is the one difference), and the tokenizer serving builds from it encodes the 15,621 texts of the pinned corpus to the ids Hugging
# Face's `tokenizers` gives. No card.
gate-hf-minimax:
    ./tools/box.sh 'bash tools/ref/needs-env.sh gate-hf-minimax BLOOMERY_MM31_HF && bash tools/gate.sh --release -p bloomery-hf --lib hw_the_minimax_folder_derives_the_m3_gguf_vocabulary -- --ignored --nocapture && bash tools/gate.sh --release -p bloomery-hf --test minimax_encode -- --ignored --nocapture'

# The shared mapped checkpoint reader's dtype, span, shard and conversion contracts. No model or card.
[group('canary')]
gate-safetensors:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-safetensors --lib -- --nocapture'

# The shared checkpoint preparer's shard and tokenizer checks, without model files.
gate-prepare:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model-hf --bin prepare -- --nocapture'

# The served tokenizer against the pinned HF corpus; BLOOMERY_QWEN38_HF names its checkpoint folder.
gate-prepare-corpus:
    ./tools/box.sh 'bash tools/ref/needs-env.sh gate-prepare-corpus BLOOMERY_QWEN38_HF && bash tools/gate.sh --release -p bloomery-model-hf --bin prepare hw_the_served_tokenizer_encodes_the_corpus_as_hf -- --ignored --nocapture'

# What the model description does with the hub (crates/model-hf, host only): the fixture fetch's header reads by range
# into sparse files, the refusal of a header that outruns its listing or a short range, and GLM-5.3-Flash's checkpoint
# map and tensor-parallel cuts on synthetic headers and the hf crate's checkpoint fixtures. No network, no card. The
# `fixture` bin's tests run in gate-fixture, the map against the real checkpoint in gate-glm5next-hf-map.
[group('canary')]
gate-model-hf:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-model-hf --lib -- --nocapture'

# MiniMax-M3.1-Flash's checkpoint map on a real folder (crates/model-hf, host only; BLOOMERY_MM31_FOLDER names the vendor's folder or its
# fixture cut): headers and tokenizer, no tensor bytes. Every tensor the folder stores is read, skipped (the vision tower, the projector) or a
# companion; the M3 reader reads the header the map writes and classifies every tensor; the counts and bytes are those the map's tables derive
# (the real folder: 60 layers, 89,632 stored tensors, 1,359 planned, 248,369,266,176 text bytes; the cut: 3 layers). No card.
gate-minimax31-folder:
    ./tools/box.sh 'bash tools/ref/needs-env.sh gate-minimax31-folder BLOOMERY_MM31_FOLDER && bash tools/gate.sh -p bloomery-model-hf --lib -- --ignored hw_the_minimax_folder_opens_as_the_m3_reader_reads_it --nocapture'

# Reference sets (crates/refset, host only): the readers' unit tests, then every family's sets in place
# against the family table — each complete, dumped from the file the tree runs, of the family's ik build.
# The draft set's family needs the profile's DSPARK_MODEL, exported as the dspark recipes do.
gate-refset:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-refset && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gate.sh --release -p bloomery-refset --lib -- --include-ignored --nocapture'

# Which reference sets are stale, one line each: with no arguments every family's sets in place, else
# `<family> <path>...` reads the sets at the paths as sets of that family (host only, reads only).
refset-check *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh refset-check && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && cargo build --release -p bloomery-refset --bin refset-check && bash tools/host-gate.sh refset-check {{ARGS}}'

# The type census's header stubs: from tools/census/stubs.tsv (`python3 tools/type-census.py pin` writes it), the first
# header_bytes bytes of each real file are fetched from the hub at the row's revision, checked against header_sha256,
# written at ROOT/<org>/<repo>/<file> as a sparse file of file_bytes, and read back and checked again. ROOT is a
# directory on the box that does not exist or is empty (e.g. /models/census-stubs-<revision tag>): a rebuild goes to a
# new one. It reads the hub only: no card, no lock, not timed.
census-stubs ROOT:
    ./tools/box.sh 'python3 tools/type-census.py build "{{ROOT}}" tools/census/stubs.tsv'

# V4.1 호스트 스텝 계획: 디코드 걷기 단위 시험(위치 2,101개)을 돈 뒤, V4.1 오라클 세트(배치 세트와 디코드 스텝 세트)의
# 그래프 입력 전부를 계획과 비트 단위로 대조한다. 어떤 검사도 맡지 않은 입력은 빨강이다. 호스트 전용(카드·게이트 락 없음).
gate-ds41-plan:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-ds41-plan && bash tools/gate.sh --release -p bloomery-model --lib -- arch::deepseek41::plan --nocapture && cargo build --release -p bloomery-gpu-gates --bin gate_deepseek41_plan && bash tools/host-gate.sh gate_deepseek41_plan'

# V4.1 어텐션 쪽 게이트 아홉을 한 프로세스(`gate_deepseek41_kernels`)에서 차례로 돈다. 게이트는 그 바이너리의 모듈
# (`shared/ds41_<이름>_gate.rs`)이고, 하나가 빨강이어도 나머지는 돈다. ARGS에 게이트 이름(hc rope comp index engram woa attn
# chain_glue chain_attn)을 주면 그것만 돈다. 게이트마다 V4.1 파일의 메타데이터를 읽고 ik 덤프 세트를 읽으므로 모델을 고정한다.
# - hc: V4.1 하이퍼커넥션(B4): 사슬 커널(RMS·분할 K hc_fn gemv·HC_PRE), HC_POST와 접기를 우리 규칙과 V4.1 ik 덤프에 서브층마다
#   대조한다. engram 층(1·14)의 분리 짝(HC_POST 단독 → 접기 단독)이 융합 런치와 비트 동일한지도 본다.
# - rope: V4.1 rope 계열(머리 꼬리 rope, ROPE_BACK, 잠재 K/V의 norm·rope·f16 링 기록)을 5토큰 세트와 디코드 스텝 세트 전부에
#   대조한다. rope 사이트는 ik와 비트 동일, K/V 행은 유도한 밴드 안이어야 한다.
# - comp: V4.1 압축기와 인덱스 키(상태 gemv, DS4_COMP 풀링, norm, 꼬리 rope, f16 캐시 행, 링 persist, 인덱스 키의
#   norm·rope·Hadamard)를 5토큰 세트와 디코드 스텝 세트의 모든 소스 층에서 대조한다.
# - index: V4.1 인덱서(op F: 쿼리·가중치·점수·top-k 목록)를 d1·d2 세트와 그 unfused 쌍의 내는 층 전부에서 ik 덤프와 대조한다.
#   d1n의 항등 목록과 그 위의 attention을 접두 경로와 비트 대조하고, 16,384·32,768행의 합성 깊이 케이스(심어 둔 동점 포함),
#   재실행 비트 동일과 캡처된 재생까지 본다. 목록은 행 오름차순, 동점은 낮은 행.
# - engram: V4.1 engram 게이트: 키 norm, 그리고 쿼리 norm·f64 점곱·부호 붙은 제곱근 시그모이드·스트림 갱신을 5토큰 세트와 디코드
#   스텝 세트의 engram 층(1·14)마다 우리 규칙과 ik 규칙 시뮬, 덤프에 대조한다. 토큰마다 gemv → 키 norm → 게이트 사슬도 본다.
#   규칙은 파일의 형식이 고른다: engram_wkv가 Q8_0이면 q8_0_gemv와 ik의 q8_2 규칙, Q3_K(공개 파일)이면 q8_1 → q3k_gemv와
#   ik의 q8_K × Q3_K 점곱(engram_kv와 비트 동일), 행 테이블과 gain 둘도 Q8_0/bf16 또는 Q3_K. 그 밖의 형식은 텐서 이름을 대며 거부한다.
# - woa: V4.1 attn_output_a(블록 대각 여덟 그룹)를 q8_0_gemv_heads 한 번의 발사로, attn_output_b는 q8_0_gemv로 돌려 5토큰 세트와
#   디코드 스텝 세트의 층마다 우리 규칙(비트 동일), ik 규칙 시뮬(덤프와 비트 동일), 덤프(값마다 유도한 밴드)에 대조한다.
# - attn: V4.1 어텐션(B4): 세트 일곱(5토큰 배치와 디코드 스텝 여섯)의 모든 층을 접두 키와 선택 행 두 경로로 본다. ik 규칙
#   시뮬 대 덤프, 커널 대 우리 f32 규칙과 덤프, 그래프 재생, 깊이 케이스. sink를 읽으려고 V4.1 모델 파일을 연다.
# - chain_glue: The V4.1 glue piece (chain G1) on the step4 and d1n sets: the host engram row ids against the dump's int rows,
#   the embedding broadcast bit for bit, each engram site's step within the band b4engram's pins propagate, and
#   the head end (hc_out bit for bit, the logits against their predicted gap, the argmax); one captured graph
#   replayed per set against the eager run, with its node count. The weights load through the engine's role
#   formats, so a Q3_K engram_wkv (the public file) runs q8_1 + q3k_gemv, pinned within KERNEL_BAND of the exact
#   dot and against ik's Q3_K dot bit for bit; the node count follows the format.
# - chain_attn: V4.1 attention 조각(B5 1단계, chain G1): step4·d1n 세트의 40층 전부에 덤프의 입력을 주입하고, 스트림·접기·링·압축기
#   캐시를 유도한 편차(σ 전파, z ≤ 9) 안에서 ik와 대조한다. 스텝이 안 쓴 슬롯은 비트 동일, 재생 = eager, 층 종류별 노드 수.
gate-gpu-ds41-kernels *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-ds41-kernels && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_kernels && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_kernels {{ARGS}}'

# V4.1 device-linked lib tests: rope tables match ggml even at positions outside the oracle set.
# Include ignored hw_ checks for pre-activation gate/up errors under the common GPU card lock.
[group('canary')]
gate-gpu-ds41-lib:
    ./tools/box.sh 'bash tools/gate.sh --gpu --oxide -p bloomery-gpu-deepseek41 --release --lib -- --include-ignored'

# V4.1 서버 엔진 결합(gpu-gates의 bind, deepseek41 기능 뒤에 있어 gate-gpu-gates-lib가 빌드하지 않는다)과 모든 서버 좌석이 나눠 쓰는
# 비전 좌석(vision_seat: 플래그, 투영기↔텍스트 검사, 인코더 자리와 reserve, 표의 위치당 바이트, `vision` 레코드)과 임베딩 모델의
# 사이드 좌석(side_seat: 플래그)의 호스트 단위 시험:
# 샘플링 스텝이 엔진의 로짓 버퍼 하나를 빌려 쓰고, 길이가 다른 호출자 버퍼는 이름 붙은 오류다. 카드·게이트 락 없음.
# With `checkpoint` on, model_file::fetch's admission test runs too: a checkpoint build's `--hf` takes a checkpoint
# repo, and the side seat, which opens GGUF only, refuses it before any shard download. The start's agent half
# (start::, behind the same feature: the class an error of the start path names, the plan line a seat's notes make, the
# early listener's hand-over) runs here, since gate-gpu-gates-lib builds without it.
[group('host')]
gate-ds41-bind:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41,hf --lib -- vision_seat:: && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41,hf --lib -- side_seat:: && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41,checkpoint --lib -- model_file::fetch::tests::checkpoint_support_is_checked_before_download --nocapture && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41,hf --lib -- bind:: && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41,hf --lib -- start::'

# The Qwen3.8 serve seat's feed rules (bloomery-serve-qwen38's own unit tests, behind the deepseek41 feature like bind):
# the route trace takes only the step feed, no MTP draft, one slot and no residency word but off; the step feed takes
# no MTP draft and no such word; a step-fed prompt is never cut at a message start; the one levers list the server and its
# gate parse (shared/qwen38_levers.rs) holds exactly the levers the seat reads; the default slot count's floor fallback
# answers the NVMe tier floor's refusal and a unified pool's PoolBudget, and no other; a checkpoint folder's plan holds
# its least part reads on the NVMe tier arena's floor, and on a unified pool an unset context fits under them while a
# set one is refused. The shared serve client
# holds the reply evidence; serve_config holds the settings file's overrides and named refusals. The V4.1 soak's
# readers of the same replies refuse a served answer they cannot read (soak_ds41_serve's own unit tests), and the
# decide seat gate reads set D's grids and spans whole or refuses them by name (gate_clef_serve's own unit tests).
# The seat's `mtp width` record and the `mtp summary` it shares carry the width chooser's
# state in their pinned shape (record::tests::width_records). Host only, no card or gate lock.
[group('host')]
gate-q38-serve-rules:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41,gemma-embedding2,hf --lib -- serve_client:: && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41,gemma-embedding2,hf --lib -- serve_config:: && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41,gemma-embedding2,hf --lib -- record::tests::width_records && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41,gemma-embedding2,hf --bin bloomery-serve-qwen38 && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41 --bin soak_ds41_serve && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --bin gate_clef_serve'

# The shared draft's prompt boundaries and per-slot costs in the other serve targets that include it, and the Raon seat's
# unset draft rule: a request drafts while its think span is open, up to the tag that closes it. The launcher's checks
# before the fetch (`tests/serve_preflight.rs`) run on builds without and with `vision`. Host only, no card or gate lock.
[group('host')]
gate-serve-draft-rules:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features glm5next,clef,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --bin bloomery-serve --bin bloomery-serve-ds41 --test serve_preflight && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --test serve_preflight'

# The V4.1 binaries' `--place` word (`generate::Place`, behind the deepseek41 feature like bind): each alias and its card
# list (`a6000+3090` is `bp`) planning the machine its plan function makes, a list word planning its own cards, and every
# refusal by name before any plan. The GLM generator's TP plan holds rank order and bytes at N = 1, 2 and 4. Every
# --place word llama-server's flags make (llama_place) parses at the seats and resolves on the census.
# Host only, no card or gate lock.
[group('host')]
gate-ds41-place:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41 --lib -- generate:: && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41 --lib -- llama_place::seat_words && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41 --lib -- llama_launch:: && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features glm5next-hf --bin generate_glm5next'

# ik의 KLD 기준 파일(ik-ppl --kld-base)이 그것을 쓴 실행과 맞는지 본다: 헤더의 ctx·청크 수와 파일 크기가 실행의 결과
# 줄과 같고, 기록마다 ik 양자화기가 쓸 수 있는 모양이며, 파일에서 다시 잰 PPL이 실행이 찍은 PPL과 양자화 밴드 안에서
# 같다. 기본은 kldbase-c2048x4와 kldbase-c512x16이다. 어휘 수를 모델 파일에서 읽으므로 V4.1 프로필로 돈다. 호스트
# 전용이라 카드도 게이트 락도 쓰지 않는다. FAIL-first는 박스 명령 안에서 BLOOMERY_KLD_FILE=<사본 경로>를 준다 — 그
# 파일 하나만 읽고, 파일 이름의 stem이 가리키는 실행의 로그와 맞춘다.
gate-ds41-kld:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-ds41-kld && bash tools/gate.sh --release -p bloomery-gpu-gates --lib -- --ignored hw_ds41_kld --nocapture'

# A PLE site's two kernels (Qwen3.8's layer 1, bloomery_gpu::ple) on synthetic inputs, no model: the gate bit for bit
# against its host rule and within a derived per-value bound of the exact f64 values; the dilated conv bit for bit
# against its host rule; then the m-column, ring-wrap, rollback, reset, fault and refusal clauses.
gate-gpu-ple:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_ple && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_ple'

# 조립된 V4.1 스텝(B5 1단계, pos < 512·선택 없음)을 게이트 배치로 엔진 입구를 거쳐 돌린다. --structure: 캡처된 스텝의
# 커널·메모리 연산 배치 수가 조각들이 예측한 수와 같고, step4 세트의 상태를 주입한 재생이 eager 실행과 비트까지 같다.
# --sets: step4·d1n 세트에서 층마다 덤프의 상태를 주입해 eager 스텝 하나 — engram 행 id와 라우터 id는 정확히(근접 동률은
# 면제), 서브층마다 스트림은 전파 밴드 안, result_output의 argmax. 3090, 게이트 락.
# --select(2단계): d1(top_k 64)·d2 세트, 인덱서 층마다 선택이 실제로 일어나는 자리 — 리스트가 그 층 점수의 정확한 top-k인지,
# 인덱서의 쿼리·가중치가 포락선 안인지, ik 리스트와의 차이가 동률 밴드 안뿐인지; 리스트를 바꿔 끼웠을 때 어텐션이 받는
# 영향은 포락선에 합쳐 잰다.
# --skew-structure, --skew-sets, --skew-api (C4 ktok-skew, shared/ds41_skew.rs), on the same load after those, from a
# reset: tokens t and t+1 as two rows one layer apart on one stream against two one-token steps, bit for bit.
# --skew-structure: the captured nodes exactly twice the step's by kind and kernel name, the batch order, each row's
# shadow table. --skew-sets: after step4 and d1 are injected, both rows' logits, streams, folds and lists and every
# cache, compressor state and history equal the steps in turn, eager and replayed, and after a rollback (t+1 refused)
# one step of another token equals the steps. --skew-api: the engine's entry (step_pair, rollback), graph and eager,
# then the deep cuts. One load.
# --feed, alone after --slots, on a load at the default levers: a file whose batch-read tensors are all Q3_K is fed by the
# prompt batch with no refusal; any other file is fed as steps, `Body::batch_refusal` names a tensor that decided it, and its
# prompt call is the same ids stepped one by one (the last token and the snapshot). Under BLOOMERY_FIXTURE_VARIANT=istadense
# (ISTA's dense types) every clause above reads that file's own oracle sets, `fx_istadense_...`.
[group('v41-load')]
gate-gpu-ds41-step *ARGS='--structure --sets --select --skew-structure --skew-sets --skew-api':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_step && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_step {{ARGS}} && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_step --slots && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_step --feed'

# V4.1 전문가 층(host/tier.rs, twoeng R3)의 루프백: 스테이지와 층을 한 카드(3090, 게이트 배치)의 Gpu 둘에 올린다.
# The gate plan puts each layer's id prefix [0, n_l) on the card; the tier plan moves its last 23 ids (n_l−23 … n_l) to
# device 1 (the tier), the (b′) shape. 기준은 게이트 계획 자체(합집합이 한 카드에)다. 두 적재는
# 차례로, 기준 먼저. --union(T1): 산문 프롬프트 16위치를 한 스텝씩, 탐욕 32스텝, 쌍 4번(그래프),
# 프롬프트와 4스텝(eager)이 기준과 argmax·logits 비트까지 같고 스테이지 그래프 노드 수가 같다; 전제: 층이 가진 층마다
# 라우팅된 슬롯이 한 번 이상 갔다. --fault(T2): 층 카드 폴트가 스텝의 오류·독, 리셋 뒤 기준과 같다. --lost(T3): 층
# 스트림을 호스트 플래그 뒤에 묶으면 데드라인 안에 카드 상실 이름, CardLost 독, 다음 스텝·리셋 거절. --two(T7): 두 장치에
# 같은 전문가가 있는 계획은 업로드 전에 이름으로 거절. 프롬프트 배치(tierbatch, twoeng R4): --batch(B7) P = 64 배치 하나가
# 같은 적재에서 스텝으로 먹인 것과 logits 밴드(logits_band::walk: clear top-1 같음, max·rms 핀) 안이다; --batch2(B7 그룹) P = 1024
# 배치 둘을 한 그룹으로 돌린 것이 기준 적재의 배치 호출과 같은 밴드 안이고(층 전문가는 tile, 기준은 GEMM) 그룹이 실제로 둘이었다; --bfault(B2) 층 카드 폴트를 올린 뒤의 프롬프트 호출이 같은 폴트로 끝나고 독,
# 리셋 뒤 스텝과 같은 밴드 안이다; --blost(B3) 층 스트림을 묶은 채 프롬프트 호출이 데드라인 안에 카드 상실 이름과 CardLost 독으로 끝난다.
# --bfeat(B8) 따로 한 번 더 적재해, 적재가 배치 버퍼를 만든 뒤 드래프트 특징 탭을 붙이고 prepare_prefill을 다시 부른다(--place bp가
# 드래프트를 여는 순서). 같은 id의 프롬프트 호출이 드래프트 창으로 넘긴 특징이 전 위치를
# 넘긴 호출의 것과 위치·값 비트까지 같고, 전 위치 호출의 특징은 P = 513 스텝의 특징과 FEATURE_REL_PIN 안이다; 전제: 배치 둘이 한 그룹, 넓은 호출은 두 배치 모두에서 행을 넘겼다. 드래프트 헤더는 프로필의 DSPARK_MODEL.
# B7 feeds 64 positions and B8 513 (two batches): the batch call's depth against the steps is gate-gpu-ds41-prefill's oracle, so
# these keep only what the tier loopback load shows. The lost-card clauses --lost (T3), --blost (B3) and --bfirst (B3f) are not
# in the landing run: weekly-gpu-ds41-lost runs them. The 3090's plan on either card (shared/gate_card.rs), gate lock.
[group('v41-load')]
gate-gpu-ds41-tier *ARGS='--union --fault --two --batch --batch2 --bfault --bfeat':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_tier && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_tier {{ARGS}}'

# V4.1 plan (b′) on both cards (--place bp): the stage on the A6000, the expert tier and the DSpark draft on the 3090.
# The reference is the same plan with every tier set joined to the stage card's (one card, no tier code); --loopback, by
# name only, adds the plan with the tier on the A6000 beside the stage (two Gpus on one card). The draft is on the 3090
# in every load; the plans are made under a card budget so the A6000 holds both sets; the stage keeps each layer's id
# prefix and the tier the next ids. --union: the prose prompt (32 ids) step by step, 48 greedy steps, then the prompt
# through the draft and 16 DSpark passes: the stepped part's tokens and logits rows bit for bit the reference's, the call and
# passes within the logits band (logits_band::walk; the tier's experts run tiles, the reference's the GEMM) and their kept
# counts equal over the walked passes, the stage graph's node count the reference's; precondition: every tier layer sent a routed slot, the passes kept and rejected a proposal.
# --lost: the tier's stream held behind a host flag, the step named as a lost card within the deadline, no token,
# CardLost, the next step refused; not in the landing run: weekly-gpu-ds41-lost runs it. Two loads, one after the
# other. Both cards (BLOOMERY_CARD=both: both gate locks), alone in a batch.
# Group solo, reason r2 (both cards: the plan (b′) holds the stage on the A6000 and the tier and the draft on the 3090 at once): a
# fixture-tier run keeps it, so it stays `solo`, not `solo-real`.
[group('solo')]
[group('v41-load')]
gate-gpu-ds41-twocard *ARGS='--union':
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_twocard && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gpu-gate.sh gate_deepseek41_twocard {{ARGS}}'

# The lost-card clauses: gate_deepseek41_tier --bfirst --blost --lost (B3f, B3 and T3: the loopback's tier stream held
# behind a host flag before a fresh load's first prompt call, before a prompt call, before a step, on the 3090, each on
# a load of its own) and gate_deepseek41_twocard --lost (before a step on plan (b′), both cards, one load): the call or
# the step fails within the go deadline and its grace naming the lost card, the host tier is poisoned as a lost card
# (CardLost), the next call or step is refused by that poison and a reset by name. Then the tier gate's T1, T2, B7, B2 and
# the same three lost-card clauses again over BLOOMERY_TIER_LINK=loopback (two pages on one card and the host pump that
# carries each go and done between them, host/link/pump.rs): every clause as over the shared page. Weekly: `just weekly`
# runs it, and `just affected` names it when a file its triggers in tools/gate-paths.tsv match changes. Alone in a batch.
# Group solo, reason r2 (the twocard leg holds both cards): a fixture-tier run keeps it, so it stays `solo`, not `solo-real`.
[group('solo')]
[group('v41-load')]
weekly-gpu-ds41-lost:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_tier && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gpu-gate.sh gate_deepseek41_tier --bfirst --blost --lost && BLOOMERY_TIER_LINK=loopback bash tools/gpu-gate.sh gate_deepseek41_tier --union --fault --batch --batch2 --bfault && BLOOMERY_TIER_LINK=loopback bash tools/gpu-gate.sh gate_deepseek41_tier --bfirst --blost --lost'
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_twocard && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/gpu-gate.sh gate_deepseek41_twocard --lost'

# 상주 교체의 r8 → Q3_K 풀기(`ds41_r8_q3k_groups`, 그룹 하나를 블록 하나가 제자리에서): 무작위 r8 워드를 V4.1 파트
# 모양(2304행 × 20 슈퍼블록)과 홀수 폭 작은 모양으로 풀어 참조 커널 `ds41_r8_q3k`와 `qdot::unpack_q3k_r8`의 바이트와
# 같고 파트 밖 보호 워드를 건드리지 않으며 창이 잡은 컨텍스트 핸들을 돌려주는지, 잘못된 파트(16바이트 경계 밖, 21 슈퍼블록,
# 반 그룹)는 이름 붙은 오류인지 본다.
# 모델 파일 없음.
gate-gpu-ds41-unpack:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_unpack && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_unpack'

# V4.1 host streaming in a prompt call (BLOOMERY_HOSTSTREAM, set on in the gate) over the residency at mid-p40-s1, plan
# (b′), groups of 2, the host set locked, both cards, one load. First the residency's own clauses (shared/ds41_residency.rs,
# streaming off): the host-set refusal at load (a host one byte short of the churn pool, named, before the load), then on
# the load c6 (a DSpark pair pass keeping one row folds what one step folds), c1 (the prose history twice, same tokens and
# logits, flips landed), the admitted experts' slots byte for byte a static load's (the r8 unpack), the host set's
# residency answers, the tier entries unmoved, the passes' kinds and kept rows (the prompt call one pass, 0 kept), c3
# (the copy stream held 1 s, same history), the serve seat's reset leaving the residency, c7 (a reset back to the seed,
# no host byte released). Then, streaming on: s2 (a call that streams, then the same call on the placement it left admits
# nothing and gives the same argmax and logits), s1 (a 1,536-id call and 8 steps free and with the copy stream held 1 s:
# the same picks per group and layer, tokens and logits), s3 (4 prose and 4 code windows of 512 ids, streaming off and
# on: where they first differ, on's top-1 margin below 1.5). Both cards, alone in a batch.
# Group solo, reason r2 (both cards: plan (b′) on the A6000 and the 3090 at once): a fixture-tier run keeps it, so it stays `solo`.
[group('solo')]
[group('v41-load')]
gate-gpu-ds41-callstream *ARGS:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_ds41_callstream && bash tools/gpu-gate.sh gate_ds41_callstream {{ARGS}}'

# The same residency clauses on plan (a), the serving plan: every layer and the head on the A6000, no tier card
# (gate_ds41_callstream --place a --only residency; the tier clause asks that the host map holds no tier entry), and
# the static clause, whose load of the dumped card table follows the residency load's teardown, then the serving
# default's load: the word the V4.1 seat runs with BLOOMERY_RESIDENCY unset on this plan (mid-p0-s1 where the host takes
# its churn pool, else the first P whose pool fits), its pinned count and c1. The A6000 alone, the host set locked, one
# load at a time, alone in a batch.
# Group solo-real, reason r1 (its host set is locked under the big-load lock: the whole file on the real tier, ~15 GB on a fixture)
# on one card, the A6000 by box.sh's pick (`BLOOMERY_CARD=a6000`): alone in the real tier, and in the fixture tier fixed to lane B.
[group('solo-real')]
[group('v41-load')]
gate-gpu-ds41-residency-a:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=a6000 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_ds41_callstream && bash tools/gpu-gate.sh gate_ds41_callstream --place a --only residency'

# V4.1's attach of the NVMe expert tier (shared/nvtier_family.rs, whose clauses a second family calls): a paged arm
# against an all-RAM arm on the same file, plan (a), the residency off, the r8 pair off — the paged plan's arena is
# the given slots, the tier built and covering every paged layer, one prompt call and greedy steps bit for bit the
# all-RAM arm's, the tier's card leg unattached, the room under the floor, the arena under its floor and the r8
# sidecar on a paged plan each refused by name, and the unset residency `off` with why `paged`. Each arm's room and
# arena reach the plan through PlanInputs, so no environment crosses the tiers. The A6000 alone, one load at a time.
# Group solo-real, reason r1 (its host set is the whole file on the real tier, ~15 GB on a fixture) on one card, the A6000 by
# box.sh's pick (`BLOOMERY_CARD=a6000`): alone in the real tier, and in the fixture tier fixed to lane B.
[group('solo-real')]
[group('v41-load')]
gate-gpu-ds41-nvtier:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=a6000 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41 --bin gate_ds41_nvtier && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_ds41_nvtier && bash tools/gpu-gate.sh gate_ds41_nvtier'

# V4.1 long runs on the gate placement, three arms on one load, the host set populated and locked
# (BLOOMERY_HOST_LOCK=1 unless the environment says otherwise: populated pages alone are reclaimed under
# another process's reads). --faults first, the process's first steps after the load: 32 greedy graph
# steps after a reset, each step's faults outside the engram helper read around it; from step 2 on no
# major fault and at most the pinned minor ones. Then, every position through the finite probe before the
# engine steps it: --free, prompt row 7 then up to 330 greedy tokens; --trigger, prompt row 0 plus the 311
# fed ids whose last position selects six layer-34 experts that all score 0, then 16 greedy tokens. Red on
# a fault over the pin, a non-finite stream at any seam, a run of 8 equal generated tokens, an eager argmax
# that is not the engine's token, or a first difference with ik's greedy ids where our margin is not below
# 1.5. Solo: its fault pin is a host-memory count another lane's model load moves. ARGS go after the three
# arms (`-n N`). FAIL-first for the faults arm: BLOOMERY_BOX_ENV="BLOOMERY_HOST_POPULATE=0 BLOOMERY_HOST_LOCK=0".
# The 3090's plan on either card (shared/gate_card.rs), gate lock.
[group('solo')]
[group('v41-load')]
gate-gpu-ds41-long *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-ds41-long && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_long && BLOOMERY_HOST_LOCK=${BLOOMERY_HOST_LOCK:-1} BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_long --faults --free --trigger {{ARGS}}'

# The candidate mask past the 16,384 positions where it first selects, on the live model at the serving context (gate
# placement, 3090), two loads one after the other. G2, gate_deepseek41_long --candidates: 16,448 prose ids through the
# prompt batch, then 32 eager steps with the attention piece's candidate tap armed; at every step layer 20's kept blocks
# are the reference's selection over its own scores and each consumer's compaction and list hold to them, and some list
# the mask changed. G2b, gate_deepseek41_prefill --cand: positions 16,300-16,500 as one batch call and as 200 steps
# from the same keep point, every ring, compressor state, compressed row, index key and the last logits bit for bit.
# The landing batch runs no model-load prefill past 16,384: the wiring at depth lands on the chain_attn gate of
# gate-gpu-ds41-kernels (synthetic G2s), and below the bound every candidate launch is a no-op, so the batch-set indexing
# past it is proved here alone. Weekly: `just weekly` runs it, and `just affected` names it when a file its triggers in
# tools/gate-paths.tsv match changes. Alone in a batch.
[group('solo')]
[group('v41-load')]
weekly-gpu-ds41-cand:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh weekly-gpu-ds41-cand && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_long --bin gate_deepseek41_prefill && BLOOMERY_HOST_LOCK=${BLOOMERY_HOST_LOCK:-1} BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_long --candidates && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_prefill --cand'

# ik가 V4.1 파일로 prompts.tsv의 행 PROMPT를 greedy로 잇는다(CPU, 디코드마다 토큰 하나, CPU 임대 안) — long 게이트
# --free의 참조, $BLOOMERY_DATA/greedy-ds41/. 행 0은 세 토큰 만에 EOS라 긴 비교는 행 7. 러너 머리말 참조.
ik-greedy-ds41 PROMPT='0':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'PROMPT={{PROMPT}} bash tools/ref/ik-greedy.sh'

# G4: ik의 KLD 기준 파일 TAG($BLOOMERY_DATA/ikppl/TAG.kld)에 대한 우리 NLL·KLD·top-1 — 청크마다 리셋하고 id마다 엔진
# 스텝 하나. 512 × 16청크는 약 8,200스텝이라 한도를 28분으로 올린다. 빨강은 Δ_PPL > +1.5 %.
[group('v41-load')]
run-ds41-ppl TAG:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh run-ds41-ppl && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_step && BLOOMERY_GATE_BOUND=1680 bash tools/gpu-gate.sh gate_deepseek41_step --ppl {{TAG}}'

# V4.1 decode CLI, functional run (no timing): generate_ds41 feeds the prompt one real step per id, then greedy
# -n tokens, and prints them. The recipe loads the step gate's placement on the 3090 (--place gate), so its tokens
# are comparable with gate-gpu-ds41-long's free arm; later arguments override it (a flag given twice takes its last value) —
# `BLOOMERY_CARD=a6000 just gen-ds41 --place a …` loads the serving placement (a) on the A6000. 3090, gate lock.
[group('v41-load')]
gen-ds41 *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gen-ds41 && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 && bash tools/gpu-gate.sh generate_ds41 --place gate {{ARGS}}'

# The V4.1 binaries' record schemas (crates/gpu-gates/src/record.rs) into tools/bloomery/schema/, which record.rs's
# checked_in_schemas_are_current holds to the binaries', and the engine's plans the flow model reads
# (generate_ds41 --plan, placement (a), P 128/256/384/512/1536/4096/16384, CED on and off) into tools/flow/plans/:
# P 1536 is weekly-gpu-ds41-flowcounts' three-batch arm, P 16384 the prefill headline's prompt. Loads nothing onto a card;
# runs with the A6000 in view, since --place a plans the stage on the largest visible card. Each plan runs at the plan
# room (tools/ref/plan-room.sh: BLOOMERY_HOST_ROOM pinned); a plan that moves with the room turns the refresh red.
records-refresh:
    #!/usr/bin/env bash
    set -euo pipefail
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=a6000 ./tools/box.sh 'bash tools/ref/real-only.sh records-refresh && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 --bin bloomery-chat --bin gate_deepseek41_prefill >&2 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41,gemma-embedding2,hf --release --bin bloomery-serve-qwen38 >&2 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41,vision,gemma-embedding2,hf --release --bin bloomery-serve-ds41 >&2 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,vision,hf,glm5next-hf --release --bin generate_glm5next >&2 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe --bin clef_hidden >&2 && cargo build --release -p bloomery-gpu-gates --bin probe_nvread >&2 && for b in generate_ds41 bloomery-chat bloomery-serve-ds41 bloomery-serve-qwen38 gate_deepseek41_prefill generate_glm5next generate_qwen3moe clef_hidden probe_nvread; do target/release/$b --records-schema; done && for P in 128 256 384 512 1536 4096 16384; do for c in on off; do echo "#> tools/flow/plans/ds41-p$P-ced-$c.rec generate_ds41 --plan --depth $P --place a under BLOOMERY_CED=$c at the plan room" && bash tools/ref/plan-room.sh $P $c; done; done' | python3 tools/bloomery/records.py refresh

# The flow model's queue entries held to the engine's (3090, placement gate): generate_ds41 -n 2 under
# BLOOMERY_STEP_STATS=1 prints its counter (`stat prefill front`, `stat prefill lb`), at --depth 512 once at the default
# group and once under BLOOMERY_PREFILL_GROUP=1, and at --depth 1536 at the default group: three configs, which the
# group lever of 2 runs as one group (a lone last batch joins the group before it), so the model's group wrap and its
# fold are held too. The two default-group configs are one --arm list of one load and one process — 512/2, 1536/2,
# then 512/2 again: the third arm repeats the first, so a state that leaks across the session's clear between arms
# shows — and before each arm after the first generate_ds41 asserts the model came back (position 0, not poisoned,
# each card's free bytes). The group arm keeps its own process: the group lever is load-level on V4.1. Each is a
# tools/gpu-gate.sh run whose log goes to a file; then tools/flow/ds41_prefill.py --counts on each log, which compares
# it layer-batch by layer-batch with the model's config for each arm's own `call plan` (PG2 or PG1) over the engine's
# plan of that P (tools/flow/plans/, `just records-refresh`). Red unless both run rc 0, the arm log prints three
# `arm <k>: counts: equal` lines (one an arm) and its repeat verdict `repeat of arm 0: equal`, the group log its one
# `counts: equal`, and the arm run's stderr two `state back at arm <i>: ok` lines (one a checked boundary). Two loads;
# generate_ds41 names the arm list's order first (`arms in order:`). The arm run's stderr goes to a file and is printed
# whole after it, its rc kept; logs and --counts output in target/flowcounts-gate/.
# Weekly: `just weekly` runs it, and `just affected` names it when a file its triggers in tools/gate-paths.tsv match
# changes.
[group('v41-load')]
weekly-gpu-ds41-flowcounts:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh weekly-gpu-ds41-flowcounts && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 && D=target/flowcounts-gate && rm -rf $D && mkdir -p $D && { BLOOMERY_STEP_STATS=1 bash tools/gpu-gate.sh generate_ds41 --place gate --arm 512/2 --arm 1536/2 --arm 512/2 > $D/g-arms.log 2> $D/g-arms.err; ga=$?; cat $D/g-arms.err; (exit $ga); } && BLOOMERY_STEP_STATS=1 BLOOMERY_PREFILL_GROUP=1 bash tools/gpu-gate.sh generate_ds41 --place gate --depth 512 -n 2 > $D/g1.log && { python3 tools/flow/ds41_prefill.py --counts $D/g-arms.log > $D/g-arms.counts; ra=$?; python3 tools/flow/ds41_prefill.py --counts $D/g1.log > $D/g1.counts; r1=$?; cat $D/g-arms.counts $D/g1.counts; echo "--counts rc: the --arm list (512, 1536, 512 again) $ra, BLOOMERY_PREFILL_GROUP=1 $r1"; [ "$ra" = 0 ] && [ "$r1" = 0 ] && [ "$(grep -c "^arm [0-9]*: counts: equal$" $D/g-arms.counts)" = 3 ] && grep -q "^repeat of arm 0: equal$" $D/g-arms.counts && grep -qx "counts: equal" $D/g1.counts && [ "$(grep -c "^state back at arm [0-9]*: ok$" $D/g-arms.err)" = 2 ] && echo "the state back: ok before arms 1 and 2" && echo "weekly-gpu-ds41-flowcounts: PASS"; }'

# The served draft, bit for bit (3090, placement gate): generate_ds41 under BLOOMERY_DRAFT=lookup must emit the plain
# run's tokens. One prompt, both arms, -n 64: the first 128 ids of the code corpus. Red unless the two `tokens` lines are
# identical and the draft arm both kept and rejected a proposal (0 < accepts < proposals in its draft summary).
# The draft arm runs at BLOOMERY_MTP_WIDTH=fixed: the gate pins the draft's tokens, and the width chooser's `cost` mode
# decides on host wall time, so under load it verifies only its warm-up passes.
# Two loads, each its own tools/gpu-gate.sh run: the gate compares the CLI's two arms as shipped; logs in target/draft-gate/.
[group('v41-load')]
gate-gpu-ds41-draft:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-ds41-draft && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 && D=target/draft-gate && rm -rf $D && mkdir -p $D && C=$(head -n 128 "$BLOOMERY_DATA/engram/corpus-code.ids" | paste -sd, -) && bash tools/gpu-gate.sh generate_ds41 --place gate --tokens "$C" -n 64 > $D/code-plain.log && BLOOMERY_MTP_WIDTH=fixed BLOOMERY_DRAFT=lookup bash tools/gpu-gate.sh generate_ds41 --place gate --tokens "$C" -n 64 > $D/code-draft.log && grep -h "^draft summary " $D/code-draft.log && grep "^tokens " $D/code-plain.log > $D/code-plain.tok && grep "^tokens " $D/code-draft.log > $D/code-draft.tok && echo "code plain $(cat $D/code-plain.tok)" && echo "code draft $(cat $D/code-draft.tok)" && cmp $D/code-plain.tok $D/code-draft.tok && echo "code: tokens identical" && S=$(grep "^draft summary " $D/code-draft.log) && P=$(echo "$S" | sed -n "s/.*proposals=\([0-9]*\).*/\1/p") && A=$(echo "$S" | sed -n "s/.* accepts=\([0-9]*\).*/\1/p") && [ "$A" -gt 0 ] && [ "$A" -lt "$P" ] && echo "code: accepts $A of $P proposals: both branches ran" && echo "gate-gpu-ds41-draft: PASS"'

# The DSpark loop (3090, placement gate). gate_deepseek41_dsloop: the target's feature tap of the draft's target_layers
# (header only) — node counts equal the step's predicted nodes plus the taps, each mean right after the MoE join before
# its tapped layer, and its features equal, bit for bit, the host mean of the eagerly read streams for the eager step,
# the graph step and both rows of a pair pass. Then generate_ds41 prompt row 0, -n 64, plain and BLOOMERY_DRAFT=dspark,
# both under the same card budget so the 8.5 GB draft fits beside the target on the 3090, the plain run planning the
# draft's seat (--seat-dspark) so both runs plan the same card experts: red unless the tokens lines
# are identical and the draft summary's proposals equal its passes (the draft arm runs at BLOOMERY_MTP_WIDTH=fixed: the
# gate pins the draft's tokens on every pass, not the width chooser, whose `cost` mode decides on host wall time) and
# its accepts are neither 0 nor every proposal. Logs in target/dsloop-gate/.
# Then gate_deepseek41_dsloop --draft-load reads the draft arm's `load draft=dspark` record (no card, no gate lock; the
# arm ran on a gate card under its lock, where no other process runs): the card's loss around the load past the draft's
# counted buffers, taken - (resident - working), must be within the formula's scratch allowance, SCRATCH_ALLOWANCE
# (64 MiB). The runtime refuses on the load's own count and never on taken, which another process's allocation would
# move; the captured graphs and kernel modules are held by this clause.
# Fixture tier (BLOOMERY_TIER=fixture): every step runs on the one card the runner takes, generate_ds41 at --place a under
# the fixture's header budget (the one box.sh exports, the draft beside the target on that card), and the draft's accept
# rate, which the fixture's random draft does not have, is left to the real tier by a deferred line.
# Lane A (fixed 3090) in both tiers: the real tier's generate_ds41 calls are --place gate, which takes the 3090 on the box.
[group('v41-load')]
gate-gpu-ds41-dspark-loop:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 --bin gate_deepseek41_dsloop && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_dsloop --structure --tap && D=target/dsloop-gate && rm -rf $D && mkdir -p $D && if [ "${BLOOMERY_TIER:-real}" = fixture ]; then BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh generate_ds41 --place a --prompt-id 0 --ctx 4096 -n 64 --seat-dspark > $D/plain.log && BLOOMERY_MTP_WIDTH=fixed BLOOMERY_DRAFT=dspark BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh generate_ds41 --place a --prompt-id 0 --ctx 4096 -n 64 > $D/dspark.log; else export BLOOMERY_CARD_BUDGET=13G && bash tools/gpu-gate.sh generate_ds41 --place gate --prompt-id 0 --ctx 4096 -n 64 --seat-dspark > $D/plain.log && BLOOMERY_MTP_WIDTH=fixed BLOOMERY_DRAFT=dspark bash tools/gpu-gate.sh generate_ds41 --place gate --prompt-id 0 --ctx 4096 -n 64 > $D/dspark.log; fi && grep -h "^load draft=\|^draft summary " $D/dspark.log && target/release/gate_deepseek41_dsloop --draft-load $D/dspark.log && grep "^tokens " $D/plain.log > $D/plain.tok && grep "^tokens " $D/dspark.log > $D/dspark.tok && echo "plain $(cat $D/plain.tok)" && echo "dspark $(cat $D/dspark.tok)" && cmp $D/plain.tok $D/dspark.tok && echo "tokens identical" && S=$(grep "^draft summary " $D/dspark.log) && P=$(echo "$S" | sed -n "s/.*proposals=\([0-9]*\).*/\1/p") && A=$(echo "$S" | sed -n "s/.* accepts=\([0-9]*\).*/\1/p") && N=$(echo "$S" | sed -n "s/.* passes=\([0-9]*\).*/\1/p") && [ -n "$N" ] && [ "$P" = "$N" ] && echo "proposals $P = passes $N: every pass verified a proposal" && if [ "${BLOOMERY_TIER:-real}" = fixture ]; then echo "deferred(real) file-bound: accepts $A of $P proposals (the real draft accept rate)"; else [ "$A" -gt 0 ] && [ "$A" -lt "$P" ] && echo "accepts $A of $P proposals: both branches ran"; fi && echo "gate-gpu-ds41-dspark-loop: PASS"'

# V4.1 배치 프리필(`body::prefill`)을 게이트 배치에서: corpus-prose.ids 앞부분으로 3001스텝 디코드 오라클을 한 번 뜨고(DSpark
# 드래프트의 특징 탭 포함), P ∈ {1, 127, 128, 129, 511, 512, 513, 2300}과 분할 700+400·1800+1000·300+2700마다 모든 층의 창 링·
# 섀도 행·압축 행·인덱스 키·압축기 상태, 탭 행, logits가 P번 스텝과 비트 동일해야 한다. The 3090's plan on either card, gate lock. `--cases a,b`·
# `--no-split`은 FAIL-first용 부분 실행, `--seams P`는 판정이 아니라 첫 어긋난 이음매를 찾는 로케이터.
[group('v41-load')]
gate-gpu-ds41-prefill *ARGS='':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_prefill && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_prefill {{ARGS}}'

# V4.1 텍스트 쪽 이미지 주입(`body::prefill_media`, `Session::prompt_media`)을 게이트 배치에서: 공식 비전 오라클 세트
# (deepseek41v, `grad-448`)의 aligner 행과 mmproj 구분자 행으로 스팬을 만들어 세 배치(프롬프트 한가운데, T_MAX 경계를
# 걸침, 프롬프트를 닫음)에 먹이고 조항 (i)–(vi)을 본다: 미디어 자리 임베딩=행 넓힌 값·텍스트 자리=스팬 없는 호출(비트),
# 라우터 id=카드 자체 점수의 호스트 select_n(bias_vl/bias, 커버리지 단언 포함), engram 층의 스트림 항등(−0 예외),
# 분할 불변(한 호출=스팬 앞 텍스트에서 끊은 두 호출, 상태 해시+24 디코드), 이미지로 끝나는 프롬프트 뒤 창의 dead와
# 스냅숏/바이트 재개, 세 거부의 이름. Clause (vii) holds the serve seat's media call (two different images of the set in
# one prompt, `grad-448` and the non-square `pad-600x451`; `vision_seat::span_rows_of` + `Encoded::prompt` over `ds41_media`'s
# splice and builder, no tower) to the body's: same argmax, bit-identical last hidden row, and the builder's spans equal the
# gate's own; its control: the same ids with no spans differ from the media call. The 3090's plan on either card
# (shared/gate_card.rs), gate lock.
[group('v41-load')]
gate-gpu-ds41-media *ARGS='':
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_ds41_media && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_ds41_media {{ARGS}}'

# Text in, text out (3090, placement gate). One bloomery-chat load runs four --then runs, ordered
# seed 7 a (row 0, --seed 7), row 0 --greedy, row 7 --greedy, seed 7 b: the two seed-7 runs bracket the
# greedy ones, so a state carried from any run into a later one, or a sampler reused instead of one a
# run, turns "seed 7: ids identical" red. Prompt rows 0 and 7: a --greedy run must tokenize to the
# row's ids (llama-tokenize's); row 0 must stop at the end-of-generation id (stop=eog); on row 7 its ids
# must be the start of generate_ds41 --tokens <those ids> -n 16, and it must run to stop=length, so that
# leg compares every id; the streamed text must equal the vocabulary's one-shot decode
# (text_consistent=true). The chat's stderr splits one file a run at each `chat:` record (run k's the
# lines after run k-1's `chat:` line, through its own), every check on its run's file; exactly four
# `chat:` records, and exactly three `state back at run <i>: ok` lines (one a checked boundary). Then
# generate_ds41 on row 7's ids as its own process, the gate comparing the two binaries. Two loads. The chat
# names its runs' order first (`runs in order:`); that line, the state-back lines, a failed run's own lines and
# gpu-gate.sh's `waited N s` for a lock, which a batch runner takes off the item's time, are copied from its
# stderr into the recipe's output whatever its rc, and the rc kept. Logs in target/chat-gate/.
[group('v41-load')]
gate-gpu-ds41-chat:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-ds41-chat && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 --bin bloomery-chat && D=target/chat-gate && rm -rf $D && mkdir -p $D && R0=$BLOOMERY_DATA/greedy-ds41/prompt0.tsv && R7=$BLOOMERY_DATA/greedy-ds41/prompt7.tsv && T0=$(grep -v "^#" $R0 | head -n 1 | cut -f2) && T7=$(grep -v "^#" $R7 | head -n 1 | cut -f2) && echo "[$(grep -v "^#" $R0 | head -n 1 | cut -f3)]" | sed "s/,/, /g" > $D/p0-row.ids && echo "[$(grep -v "^#" $R7 | head -n 1 | cut -f3)]" | sed "s/,/, /g" > $D/p7-row.ids && { bash tools/gpu-gate.sh bloomery-chat --place gate --seed 7 --prompt "$T0" -n 16 --then --greedy --prompt "$T0" -n 16 --then --greedy --prompt "$T7" -n 16 --then --seed 7 --prompt "$T0" -n 16 > $D/chat.out 2> $D/chat.err; rc=$?; sed -n "/^gpu-gate\.sh: waited /p; /^runs in order: /p; /^state back at /p; /^bloomery-chat: /p; /^GPU GATE /p" $D/chat.err; (exit $rc); } && awk -v d=$D "{ print > sprintf(\"%s/run%d.err\", d, n) } /^chat: /{ n++ }" $D/chat.err && I7=$(grep -v "^#" $R7 | head -n 1 | cut -f3) && bash tools/gpu-gate.sh generate_ds41 --place gate --tokens "$I7" -n 16 > $D/p7-gen.log && sed -n "s/^prompt_ids //p" $D/run1.err > $D/p0-chat.ids && sed -n "s/^ids //p" $D/run1.err > $D/p0-chat.tok && sed -n "s/^prompt_ids //p" $D/run2.err > $D/p7-chat.ids && sed -n "s/^ids //p" $D/run2.err > $D/p7-chat.tok && sed -n "s/^tokens //p" $D/p7-gen.log > $D/p7-gen.tok && sed -n "s/^ids //p" $D/run0.err > $D/seed7a.tok && sed -n "s/^ids //p" $D/run3.err > $D/seed7b.tok && echo "p0 row ids     $(cat $D/p0-row.ids)" && echo "p0 chat ids    $(cat $D/p0-chat.ids)" && echo "p0 chat greedy $(cat $D/p0-chat.tok)" && echo "p7 row ids     $(cat $D/p7-row.ids)" && echo "p7 chat ids    $(cat $D/p7-chat.ids)" && echo "p7 chat greedy $(cat $D/p7-chat.tok)" && echo "p7 generate    $(cat $D/p7-gen.tok)" && echo "seed 7 a $(cat $D/seed7a.tok)" && echo "seed 7 b $(cat $D/seed7b.tok)" && echo "chat texts: $(cat $D/chat.out)" && cmp $D/p0-row.ids $D/p0-chat.ids && echo "p0 prompt ids: identical to the row" && test -s $D/p0-chat.tok && grep -q "^chat: .* stop=eog$" $D/run1.err && echo "p0: stopped at the end-of-generation id" && grep -qx "text_consistent=true" $D/run1.err && echo "p0 text: consistent with the decode" && cmp $D/p7-row.ids $D/p7-chat.ids && echo "p7 prompt ids: identical to the row" && test -s $D/p7-chat.tok && case "$(tr -d "[] " < $D/p7-gen.tok)," in "$(tr -d "[] " < $D/p7-chat.tok),"*) echo "p7 greedy ids: a prefix of generate_ds41 (the whole of it unless stop=eog)" ;; *) false ;; esac && grep -qx "text_consistent=true" $D/run2.err && echo "p7 text: consistent with the decode" && grep -q "^chat: .* stop=length$" $D/run2.err && echo "p7: ran to -n, every id compared" && grep -qx "text_consistent=true" $D/run0.err && grep -qx "text_consistent=true" $D/run3.err && test -s $D/seed7a.tok && cmp $D/seed7a.tok $D/seed7b.tok && echo "seed 7: ids identical" && [ "$(grep -c "^chat: " $D/chat.err)" = 4 ] && echo "chat: four runs on one load" && [ "$(grep -c "^state back at run [0-9]*: ok$" $D/chat.err)" = 3 ] && echo "the state back: ok before runs 1, 2 and 3" && echo "gate-gpu-ds41-chat: PASS"'

# The HTTP server on the V4.1 engine (3090, placement gate). Prompt row 0: generate_ds41 --tokens <the row's ids> -n 16
# under its own gate-lock hold, then gate_ds41_serve under another: it starts bloomery-serve-ds41 --port 0 --place gate,
# and checks /completion's ids at temperature 0 against generate_ds41's (all 16, or a prefix ending in the
# end-of-generation id), a chat turn streamed and not streamed (same content, [DONE] last), /tokenize of the row's text
# against the row's ids, the same /completion again after those (reset leaves nothing behind), and /props' engine
# object (name and version, the file's header facts, each device's bytes = the plan's, the KV bytes); then kills the
# server it spawned and waits for it. Then the same server under BLOOMERY_DRAFT=dspark with the draft on the A6000
# (BLOOMERY_DSPARK_CARD=A6000, the profile's DSPARK_MODEL): gate_ds41_serve --plain holds its greedy ids to the plain
# server's and generate_ds41's, and checks /props' draft object and row, the draft counts in timings and /metrics, the
# sampled and ignore_eos requests served by plain steps with no draft counted, and prefix reuse under the draft
# (a continuation, a cut, a resume) against
# cache_prompt: false. Then the image clauses (gate_ds41_serve --mmproj: the server built with the vision feature, the
# same build gate-gpu-vision and gate-gpu-ds41-media's encoder file): /props vision on, the encoder's seat and bytes on
# its `vision` record (the off-plan card under the gate placement), the bar-chart scene's values read from a data URL, the
# same image's span kept through the prompt cache, the reserve path under --place bp (the plan's bytes carry the
# encoder's reserve), and the plain server's no-vision no-reserve /props. Then plan (b′) with the draft (--place bp: the
# A6000 plan (a), the 3090 the expert tier and the draft): generate_ds41 --place bp under the draft, then
# gate_ds41_serve --place bp holds the server's greedy ids to it
# (both under BLOOMERY_RESIDENCY=off: the fixed placement, whose card and host rounding do not move with a flip's timing) and
# checks /props (the A6000, the tier card with the plan's tier bytes and the draft's class, the host) and the draft counts
# in timings. Both cards in view (BLOOMERY_CARD=both: gpu-gate.sh takes both cards' gate locks, and the batch runs it
# alone). Eight loads; logs and the raw stream in target/serve-gate/, the draft run's in target/serve-gate/draft/, the
# image runs' in target/serve-gate/{mmproj,mmproj-bp,no-mmproj}/ plus the unset-ctx bp --plan run
# (target/serve-gate/mmproj-ctx, before the load), the two-card run's in target/serve-gate/bp/. Weekly:
# `just weekly` runs it, and `just affected` names it when a file its
# triggers in tools/gate-paths.tsv match changes.
[group('solo')]
[group('v41-load')]
weekly-gpu-ds41-serve:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/real-only.sh weekly-gpu-ds41-serve && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41,vision,gemma-embedding2,hf --release --bin bloomery-serve-ds41 --bin gate_ds41_serve && D=target/serve-gate && rm -rf $D && mkdir -p $D && R=$BLOOMERY_DATA/greedy-ds41/prompt0.tsv && T=$(grep -v "^#" $R | head -n 1 | cut -f2) && I=$(grep -v "^#" $R | head -n 1 | cut -f3) && bash tools/gpu-gate.sh generate_ds41 --place gate --tokens "$I" -n 16 > $D/gen.log && bash tools/gpu-gate.sh gate_ds41_serve --gen $D/gen.log --prompt "$T" --ids "$I" --dir $D && bash tools/gpu-gate.sh gate_ds41_serve --mmproj --gen $D/gen.log --prompt "$T" --ids "$I" --dir $D && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_DRAFT=dspark BLOOMERY_DSPARK_CARD=A6000 bash tools/gpu-gate.sh gate_ds41_serve --gen $D/gen.log --prompt "$T" --ids "$I" --dir $D/draft --plain $D && mkdir -p $D/bp && BLOOMERY_RESIDENCY=off BLOOMERY_DRAFT=dspark bash tools/gpu-gate.sh generate_ds41 --place bp --tokens "$I" -n 16 > $D/bp/gen.log && BLOOMERY_RESIDENCY=off BLOOMERY_DRAFT=dspark bash tools/gpu-gate.sh gate_ds41_serve --place bp --gen $D/bp/gen.log --prompt "$T" --ids "$I" --dir $D/bp'

# nvtier R2의 게이트: 페이지드 계획(방 27 GiB — 분할 다이얼이 아레나를 준다)의 NVMe 전문가 티어 램 아레나를
# 상주 전 계획과 같은 파일에서 적재해, 설계 §7의 조항들을 본다 — 비트 동일성(512 프롬프트 + 96 탐욕 스텝),
# 아레나의 계수(미스·채움·축출 > 0, 상주 바이트 ≤ 예산, 버퍼드 읽기 0), --audit의 슬롯 대 pread memcmp,
# 이름 붙은 거부(바닥, 직독 탐침, 한 슬롯 못 되는 예산), 그리고 레지던시 조항(풀 0의 mid, unresident·faulting 0).
# sittings 조항은 split 아래 페이지드 계획을 두 번 적재해(두 번째는 호스트 메모리 복사 부하 옆) 레인 탐침이 달라도
# 픽·토큰·로짓이 비트 동일한지 본다.
# 아레나가 모델 매핑을 결코 advise하지 않음을 조항이 지킨다(gate_nvtier.rs의 advice_disjoint). 2026-10-08에
# 페이지드 티어의 두 이름 붙은 거부(HOST_LOCK 병행, r8 상주 사본)를 뺐다 — 커버리지 변경, 게이트 머리말에 근거.
# 계획은 gate_card의 3090 바이트로 서므로 어느 카드든 같은 핀이 선다 — 카드를 고정할 사실이 없어 any.
# Real-only (no fixture conversion): under the fixture tier box.sh exports the qwen4exp fixture's BLOOMERY_CARD_BUDGET, and
# gate_nvtier refuses it as a lever it does not act on. A chain member (v41-load, not solo): the all-RAM arm's plan takes the
# workstation's plan room (PLAN_ROOM), the paged arms and the grant children set their rooms, so a neighbour's load moves no plan.
[group('v41-load')]
gate-gpu-nvtier:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-nvtier && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_nvtier && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_nvtier'

# Qwen3.8's adaptive expert residency on one card (BLOOMERY_RESIDENCY set in the gate at mid-p<P>-s1, P from the plan's
# card experts): the churn pool's host refusal at load, a verify keeping its counted rows only, the same history twice
# with flips landed, every admitted slot byte for byte a static load's, the passes' kinds and kept rows (the prompt one
# pass keeping 0, each step 1, a verify its accepted rows), and a reset back to the seed. Plan (a) on the card in view
# (the gate plans `a` from the census, under the largest budget whose plan leaves every card layer the residency's experts
# and the host the churn pool: a unified card's pool holds both). Loads the whole host set under the big-load lock: a chain
# member (v41-load, not solo) on any card (no fact pins it), its plan room the workstation's own (PLAN_ROOM, which a unified
# pool caps), so a neighbour's load moves no plan. Real-only: no fixture tier yet (the router double, R4); it opens the
# real file whatever the tier names.
[group('v41-load')]
gate-gpu-qwen38-residency:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen38-residency && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen38_residency && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen38_residency'

# Qwen3.8 plan (b′) on both cards (--place bp): the stage on the stage card, the expert tier on the tier card (the real
# tier plans the A6000's and the 3090's bytes on the two cards in view, the fixture tier the cards as they report), its
# decode walks (a step and the 2-, 3- and 4-row verifies) and its ubatch prompt walk serving the tier. The reference is
# the same plan with every tier set joined to the stage card's (one card, no tier code), the plans made under a card
# budget so the stage card holds both sets. --union: 32 prose ids step by step, greedy steps, six verifies, greedy steps, then a 600-id
# prompt call and 8 greedy steps, every token and logits row bit for bit the reference's; the stage graphs' nodes the
# reference's less one a tier layer, the tier's graphs eight a tier layer and the fault copy; preconditions: every tier
# layer sent a routed slot, the prompt's tier services one a tier layer. --residency: the tier load under mid-p<P>-s1,
# the history twice the same with flips landed, its ids the residency-off run's or parted at a near tie, the streamed
# 600-id prompt call by the same rule, the tier's set the live map's tier rows, no tier expert's bytes in the host set.
# --prompt4k: plan (b′) and its union at 4,352 positions, a 4,096-id prompt call (the full ubatch window) and 8 greedy
# steps bit for bit. --lost: the tier's stream held behind a host flag, the step named as a lost card, CardLost, the
# next step refused. Up to five loads, one after the other. Both cards (BLOOMERY_CARD=both: both gate locks), alone in a
# batch.
# solo, reason r2 (both cards, both gate locks; no host-memory pin): stays solo in the fixture tier, which runs it in lane X in seconds.
[group('solo')]
[group('v41-load')]
gate-gpu-qwen38-twocard *ARGS='--union --residency --prompt4k --slots':
    BLOOMERY_MODEL=qwen4exp BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen38_twocard --bin generate_qwen3moe && bash tools/gpu-gate.sh gate_qwen38_twocard {{ARGS}}'

# Qwen3-30B and Qwen3.6 on plan (b′) (gate_qwen3moe_twocard's doc): the stage on the A6000 and one expert tier on the
# 3090 against the same plan with every tier segment joined onto the stage card. The real tier's card budget is the
# gate's, and it is what puts experts on the tier. Step tokens and logits rows, bit for bit the joined reference. The
# Qwen3-30B arm is real-file only: the fixture tier checks that file's header and runs the Qwen3.6 arm. Both cards
# (BLOOMERY_CARD=both: both gate locks), two loads of one file then two of the other, under the big-load lock the V4.1
# loads take. The bound is the solar2 two-card recipe's, copied with that recipe's shape. Both
# profiles' model paths are resolved inside the box command: the qwen35moe one through the box's
# own profile, the qwen3moe one with the box's profile exports unset so it reads its own profile.
[group('solo')]
[group('v41-load')]
gate-gpu-qwen3moe-twocard:
    BLOOMERY_MODEL=qwen35moe BLOOMERY_CARD=both ./tools/box.sh "q3=\$(env -u BLOOMERY_REF_MODEL -u BLOOMERY_REF_MODEL_PROFILE BLOOMERY_MODEL=qwen3moe bash -c '. tools/ref/ref-paths.sh && printf %s \"\$MODEL\"') && q35=\$(. tools/ref/ref-paths.sh && printf %s \"\$MODEL\") && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen3moe_twocard && BLOOMERY_GATE_BOUND=1200 bash tools/gpu-gate.sh gate_qwen3moe_twocard \"\$q3\" \"\$q35\""

# Qwen3.8's plan (b′) with its expert tier on a tier node (docs/contrib/dgx-spark.md 9.5 items 5 and 6): node 0, a child
# of the gate with BLOOMERY_TIER_NODES set, and node 1, tier_server, over loopback on this machine's card, the tier's page
# one /dev/shm file both map. Node 1's census read over the channel (node_cards, the load's own reader) is the plan's tier
# card. The clauses cover that census; refused opens for swapped experts and six invalid bounds; a refused open of
# a copy whose tier row holds other bytes; a refused symlink bind with its target unchanged after disconnect; the
# history (32 prose ids step by step, greedy steps, six verifies, greedy steps) bit for bit against the same plan in
# one process, the stage graph nodes equal and the tier graphs nine a tier layer plus the fault copy; stopped node 1;
# and node 0 killed or stopped. The history also kills node 1: node 0's next step names the lost node, CardLost, and
# refuses the next step and a reset. Stopped node 1, continued after node 0 fails, ends naming the loss. Node 0 killed
# or stopped mid-step: node 1 releases its own counters and go words, drains and exits naming the loss; stopped node 0,
# continued, fails its held step naming the release and the lost card. Each budget at most 16 GiB, sized for both
# processes on one card: on the box the A6000 by box.sh's pick (`BLOOMERY_CARD=a6000`), as two 16 GiB budgets pass the
# 3090's 24 GB. The binary's cleanup unit tests run through tools/gate.sh before the card gate.
[group('v41-load')]
gate-gpu-qwen38-nodes *ARGS='':
    BLOOMERY_MODEL=qwen4exp BLOOMERY_CARD=a6000 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features gpu --bin gate_qwen38_nodes && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features gpu --bin gate_qwen38_nodes -- --ignored hw_census_clause_requires_the_remote_census && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_qwen38_nodes --bin tier_server && bash tools/gpu-gate.sh gate_qwen38_nodes {{ARGS}}'

# A refused replay on a separate tier node resets to its fresh history. Requires BLOOMERY_TIER_NODES
# and BLOOMERY_TIER_LINK=rdma, with tier_server already serving the same model on that node. Without
# BLOOMERY_TIER_NODES a batch defers it by name (needs-env); a run by hand exits 67 before a build.
# Set it through BLOOMERY_BOX_ENV='BLOOMERY_TIER_NODES=192.0.2.10:9000'.
[group('v41-load')]
gate-gpu-qwen38-nodes-remote:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'bash tools/ref/needs-env.sh gate-gpu-qwen38-nodes-remote BLOOMERY_TIER_NODES && export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features gpu --bin gate_qwen38_nodes -- --ignored hw_a_refused_remote_replay_resets_to_its_fresh_history --nocapture'

# The HTTP server on the Qwen3.8 engine (placement gate: the 3090's bytes on the card in view). The prompt is the profile's five-id text
# ("The capital of France is") and its ids the profile's REF_TOKENS — ik's llama-tokenize on this model's
# vocabulary, never our own tokenizer (the derivation command is the profile comment's) — and five is at most the
# eight the gate needs: below Prompt38::GEMM_FROM both engines feed the prompt by bit-for-bit passes, so the
# server's greedy ids are generate_qwen3moe's; at nine or more the server runs the last id as the ubatch walk's last
# row as the CLI does, a delta chunk it completes left to the next step to fold. generate_qwen3moe --tokens <those ids> -n 16 under its own gate-lock hold,
# then gate_qwen38_serve as two processes under another, the arms one after another. --arm plain first: it starts
# bloomery-serve-qwen38 --port 0 --place gate with BLOOMERY_DRAFT=off BLOOMERY_RESIDENCY=off, and checks /props'
# engine object against its own plan of the file (architecture qwen4exp, each device's bytes = the plan's, the KV
# bytes), /completion's ids at temperature 0 against generate_qwen3moe's (all 16, or a prefix ending in the
# end-of-generation id), the same /completion once more after a chat turn, the served context in /props, and the
# plain cache clause; then its own servers one at a time: the ctx clause's flagless, refused and one-slot servers
# (they stop before the load), the slots pair (the gate's own --parallel 2 shape, which also holds the drafted
# off-slot and late-request clauses, and the seat under its own defaults but BLOOMERY_DRAFT=mtp, which also holds
# the residency and width-chooser wiring and the reset route) and the sampled rounds one. --arm drafted second: the main server
# under BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=off (the fixed window), /props' draft object against the gate's
# drafted plan, the greedy pass's ids and its draft counts, the continued requests (each keeps the held prefix,
# one `mtp prompt` record a prompt call past it, the plain ids), the sampled ones (a sampled /completion drafts;
# the same at top_k 1 gives this server's greedy ids) and the cache clause's drafted half. The gate's header names
# the clauses other tests hold. Logs and the raw stream in target/q38-serve-gate/{plain,drafted}/ (the gen log at
# its root).
# The build takes the deepseek41 feature, not the qwen family's plain gpu:
# the server surface it links (gpu-gates' bind, serve_client; the serve and sampler crates) sits behind
# that feature today — `just affected` and the recipes.py pins scope it so.
# Stays solo, reason: the host set (tools/gate-batch-resources.tsv) loads under the big-load lock, and the seat's unset residency pick reads MemAvailable against the churn pool, so a neighbour's load moves it.
# Never solo-real: the generate call and the two main servers pass `--place gate` (the 3090's bytes on the smallest card with the room for them: the 3090 on the box), the unset-place servers plan on the one card in view, and the gate plans on the card each server resolves (no shared/gate_card.rs); its gpu-gate.sh calls name no card (the 3090's lock on the box), and solo-real balances on `any`.
[group('solo')]
[group('v41-load')]
gate-gpu-qwen38-serve:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && . tools/ref/ref-paths.sh && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41,gemma-embedding2,hf --release --bin bloomery-serve-qwen38 --bin gate_qwen38_serve && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe && D=target/q38-serve-gate && rm -rf $D && mkdir -p $D && T="The capital of France is" && I=$REF_TOKENS && echo "prompt: $T" && echo "ids: $I" && bash tools/gpu-gate.sh generate_qwen3moe --place gate --tokens "$I" -n 16 > $D/gen.log && bash tools/gpu-gate.sh gate_qwen38_serve --arm plain --gen $D/gen.log --prompt "$T" --ids "$I" --dir $D/plain && bash tools/gpu-gate.sh gate_qwen38_serve --arm drafted --gen $D/gen.log --prompt "$T" --ids "$I" --dir $D/drafted && bash tools/gpu-gate.sh gate_qwen38_serve --arm lookup --gen $D/gen.log --prompt "$T" --ids "$I" --dir $D/lookup'

# The qwen3 seat of bloomery-serve (--model qwen3 -m <file>) against generate_qwen3moe, on the qwen3moe profile's file
# (Qwen3-30B-A3B, the MODEL box.sh exports under BLOOMERY_MODEL=qwen3moe), the qwen35moe profile's (Qwen3.6-35B-A3B,
# that profile's default MODEL, read from tools/ref/models/qwen35moe.sh) and the qwen38d profile's (the dense Qwen3.8-27B,
# read from tools/ref/models/qwen38d.sh): per file, one server, then one CLI, one process
# at a time under one gate-lock hold. The server's /completion of a chat turn rendered by its own template (past the
# eight ids a pass takes) gives generate_qwen3moe --tokens <those ids> -n 64 --last-step's ids, and its
# /v1/chat/completions of the same turn is those ids (gate_qwen3_serve's header has the clauses). Logs in
# target/qwen3-serve-gate/<n>/. The build takes glm5next: bloomery-serve links every seat's device bundle. --images adds
# the image clauses (the `vision` build, the family's bf16 projector and the oracle sets of ref_qwen35moe_vis_chatids_* and
# _decode_*): the qwen3moe file's --mmproj refusal and, on the Qwen3.6 file, the seat with its encoder.
[group('pool')]
gate-gpu-qwen3-serve:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen3-serve && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_qwen3_serve && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe && D=target/qwen3-serve-gate && rm -rf $D && mkdir -p $D && Q36=$(sed -n "s/^MODEL=\${BLOOMERY_REF_MODEL:-\(.*\)}$/\1/p" tools/ref/models/qwen35moe.sh) && test -n "$Q36" && Q38D=$(sed -n "s/^MODEL=\${BLOOMERY_REF_MODEL:-\(.*\)}$/\1/p" tools/ref/models/qwen38d.sh) && test -n "$Q38D" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3_serve --model "$BLOOMERY_REF_MODEL" --model "$Q36" --model "$Q38D" --dir $D --images'

# bloomery-serve-qwen38 on the box, for a person to attach a client to (toktape records from it): the A6000 by
# default (SERVE_PLACE=gate for the 3090), the port 8080 unless SERVE_PORT, a four-hour gate bound unless
# BLOOMERY_GATE_BOUND, further ARGS pass through (--alias, --ctx-size, --chat-template-file, and the Qwen3.8 levers
# through the environment). gpu-gate.sh takes the card's gate lock, so no other gate lands on the card it serves
# from, and bounds the run; the address is on the server's stderr.
serve-qwen38 *ARGS:
    BLOOMERY_MODEL=qwen4exp BLOOMERY_CARD={{serve_card}} ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41,gemma-embedding2,hf --release --bin bloomery-serve-qwen38 && BLOOMERY_GATE_BOUND=${BLOOMERY_GATE_BOUND:-14400} bash tools/gpu-gate.sh bloomery-serve-qwen38 --port ${SERVE_PORT:-8080} --place {{serve_place}} {{ARGS}}'

# The same CLI's per-step ms (lead-only): placement (a) on the A6000 under the machine-wide lease, witness blocks
# around it (tools/ref/time-gate.sh). Example: `just time-gpu-ds41 --depth 6 -n 96`.
time-gpu-ds41 *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41 && bash tools/ref/time-gate.sh generate_ds41 {{ARGS}} --time'

# ik's V4.1 decode on the first 512 ids of corpus-<CORPUS>.ids, plain or with the DSpark draft, under the lease on
# the A6000 (lead-only): the ik twin of `just time-gpu-ds41 --tokens <those ids> -n N`. tools/ref/ik-draft.sh's header
# has the prompt round-trip checks, ik's command line and the summary line. Example: `just time-ik-draft prose 96 dspark`.
time-ik-draft CORPUS N ARM="dspark":
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-tokenizer --bin bloomery-tokenize && bash tools/ref/ik-draft.sh {{CORPUS}} {{N}} {{ARM}}'

# mainline llama.cpp(V4.1 포트, 프로필의 LCPP)로 같은 corpus-<CORPUS>.ids 첫 512 id 프롬프트를 A6000에서 임대 안에 디코드한다(리드 전용):
# `just time-gpu-ds41 --tokens <그 id> -n N`의 mainline 쌍둥이, llama-completion에 LCPP_CLI_FLAGS. 프롬프트 왕복 검사 둘과 요약 줄은
# tools/ref/ik-draft.sh의 lcpp 팔 그대로다. 예: `just time-lcpp-prompt prose 96`.
time-lcpp-prompt CORPUS N:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && cargo build --release -p bloomery-tokenizer --bin bloomery-tokenize && bash tools/ref/ik-draft.sh {{CORPUS}} {{N}} lcpp'

# V4.1 decode by depth, three engines in one lease on the A6000 (lead-only): our arms `<D>` (generate_ds41 --depth D,
# placement (a)), ik's `ik:<D>` (llama-bench -gp D,96 at the profile's IK_GPU_FLAGS) and mainline's `lcpp:<D>`
# (llama-bench -d D at LCPP_GPU_FLAGS; `lcpp<K>:<D>` sweeps --n-cpu-moe K), alternated, rounds rotated, with per-depth
# ours/reference ratios. tools/ref/depth-ds41.sh's header has the arms, the placement difference and the environment levers.
# BLOOMERY_BOX_ENV='BLOOMERY_AB_ORDER=blocks' runs the arms in engine blocks instead (ours, each corpus, each bin:,
# lcpp, ik), each block's rounds together with its arms rotated inside it, and one discarded DISCARD r0 process first
# (a reference block's arm with the most token draws at its largest --n-cpu-moe, so its engram rows are warm; ours'
# longest prompt); the preheat is then off unless BLOOMERY_PREHEAT=1. The recipe passes BLOOMERY_AB_ROUNDS alone from
# the Mac's environment: every other runner variable goes through BLOOMERY_BOX_ENV.
# Prefill: every `<D>` row also carries pp_tok/s from generate_ds41's `time prompt` row (the D fed steps), and
# `ikpp:<P>`/`lcpppp:<P>` run llama-bench -p P -n 0 at the same flags (`ikpp<U>`/`lcpppp<U>`: -ub U), with a per-P table.
# `prose:<P>[@NAME=VALUE,...]` is ours on the first P ids of corpus-prose.ids (--tokens), in prose-only tables;
# `code:<P>` the same on corpus-code.ids. BLOOMERY_GEN_PLACE=a|gate picks our arms' --place (gate: the 3090 as
# BLOOMERY_TIMING_GPU); under the default order each reference arm's host set is preheated first (BLOOMERY_PREHEAT=0:
# off), rows carry majflt with the measured window's count (lcpp arms through llama-bench --progress) and [cold], and a
# failed arm is a FAIL row the runner goes past (rc 1 at the end). An ours arm under --place a or bp with no
# BLOOMERY_RESIDENCY runs the residency and prompt streaming (the lever's placement default; `@BLOOMERY_RESIDENCY=off`
# is the fixed placement's arm), and the runner holds each arm to what its binary's `residency host` record says.
# With no ours arm generate_ds41 is not built, nor under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 (the command lines, no lease);
# no arms means the runner's default (6 ik:6), which builds.
depth-gpu-ds41 *ARMS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { ours=; [ -n "{{ARMS}}" ] || ours=1; for a in {{ARMS}}; do case $a in prose:* | code:*) ours=1 ;; *:*) ;; *) ours=1 ;; esac; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41; fi; } && bash tools/ref/depth-ds41.sh {{ARMS}}'

# Qwen3-30B-A3B 전 카드 디코드를 깊이별로(A6000, 한 임대, 리드 전용): 우리 `<D>`(프롬프트를 실제 D스텝으로 먹임), ik `ik:<D>`(-gp D,96, 프로필
# 플래그)·`ikdef:<D>`(llama-bench 기본값), mainline `lcpp:<D>`(-d D)를 바퀴마다 순서를 돌려 번갈아 재고, 깊이마다 ours/각 참조 비율을 찍는다.
# 팔·플래그 근거·뺀 플래그는 tools/ref/depth-qwen3moe.sh 머리에. 우리 팔이 없으면 generate_qwen3moe를 빌드하지 않는다; 인자 없으면 6 ik:6 lcpp:6.
# mistral.rs arms: `mrs:<D>` (mistralrs bench --depth D, PagedAttention pool at the cache height) and `mrspa0:<D>` (--paged-attn off).
# Prefill: every `<D>` row also carries pp_tok/s from generate_qwen3moe's `time prompt` row, and `ikpp:<P>`/`lcpppp:<P>`
# (llama-bench -p P -n 0; `ikpp<U>`/`lcpppp<U>`: -ub U) and `mrspp:<P>` (mistralrs bench --prompt-len P) with a per-P table.
# `prose:<P>[@NAME=VALUE,...]`는 우리 팔이다. 이 프로필의 $BLOOMERY_DATA/qwen3moe/corpus-prose.ids에서 앞 P개 id를
# --tokens로 넣는다(행 라벨 ours@prose). prose 팔은 같은 P의 prose 팔하고만, 자기 표에서 비교한다.
# Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 the runner prints the command lines and exits before the lease, and nothing is built.
depth-gpu-qwen3moe *ARMS:
    BLOOMERY_MODEL="${BLOOMERY_MODEL:-qwen3moe}" ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { ours=; [ -n "{{ARMS}}" ] || ours=1; for a in {{ARMS}}; do case $a in prose:*) ours=1 ;; *:*) ;; *) ours=1 ;; esac; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi; } && bash tools/ref/depth-qwen3moe.sh {{ARMS}}'

# Qwen3.6-35B-A3B(qwen35moe) 같은 표: depth-gpu-qwen3moe의 러너·팔·표를 qwen35moe 프로필로 돈다. 우리 팔은 같은
# generate_qwen3moe가 파일 헤더에서 아키텍처를 읽어 Body35로 연다.
# Its prompt runs as its `--prefill` default (auto: ubatches for nine ids or more, `kind=` in the row names the path).
# `<D>@prefill=<auto|pass|gemm>` (and `prose:<P>@prefill=…`) runs one arm's prompt at that `--prefill` in a load of its own:
# `512 512@prefill=pass` is the same-binary A/B of the ubatch prompt against its eight-position passes (header: Prefill path).
# `prose:<P>`는 depth-gpu-qwen3moe의 prose 팔과 같다. 이 프로필의
# $BLOOMERY_DATA/qwen35moe/corpus-prose.ids 앞 P개 id를 --tokens로 넣고(행 라벨 ours@prose), prose 표에서만 비교한다.
# 참조 트리와 플래그는 tools/ref/models/qwen35moe.sh. BLOOMERY_DRY=1이면 명령줄만, 빌드 없음.
depth-gpu-qwen35moe *ARMS:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { ours=; [ -n "{{ARMS}}" ] || ours=1; for a in {{ARMS}}; do case $a in prose:*) ours=1 ;; *:*) ;; *) ours=1 ;; esac; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi; } && bash tools/ref/depth-qwen3moe.sh {{ARMS}}'

# Qwen3.8-Flash-Next (qwen4exp), the same table: depth-gpu-qwen3moe's runner, arms and tables under the qwen4exp
# profile. Our arm is generate_qwen3moe opening Body38 on the A6000, each layer's routed expert prefix on the card as
# its budget holds (BLOOMERY_QWEN38_EXPERTS unset is `card`; `@BLOOMERY_QWEN38_EXPERTS=host` puts every routed expert
# on the host tier), its prompt as its `--prefill` default runs it (`kind=` in the row); the reference is mainline
# llama.cpp, hand-set -ncmoe and fit (models/qwen4exp.sh).
# `prose:<P>[@NAME=VALUE,...]` is ours on the first P ids of $BLOOMERY_DATA/qwen4exp/corpus-prose.ids (--tokens), row
# label ours@prose, compared only with prose arms of the same P in prose's own tables — the lcg prompt's pseudo-random
# ids route heavily skewed, so a lever tuned on it can mean nothing on real text. Each arm pages the ~111 GB host set
# in when it is cold. BLOOMERY_DRY=1 prints the command lines only, nothing built.
depth-gpu-qwen4exp *ARMS:
    BLOOMERY_MODEL=qwen4exp ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { ours=; [ -n "{{ARMS}}" ] || ours=1; for a in {{ARMS}}; do case $a in prose:*) ours=1 ;; *:*) ;; *) ours=1 ;; esac; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi; } && bash tools/ref/depth-qwen3moe.sh {{ARMS}}'

# Nemotron 3 Nano 4B (nemotron_h), the same table: depth-gpu-qwen3moe's runner and tables under the nemotron_h
# profile (tools/ref/models/nemotron_h.sh). Our arm is the same generate_qwen3moe, which reads the architecture from
# the file's header and runs the whole model on one card, its prompt by the body's prompt call (`kind=gemm`, passes of
# up to 512 ids) or, under `<D>@prefill=pass`, as steps (`kind=step`): `512 512@prefill=pass` is the same-binary A/B of
# the two. `--seed-depth` is refused, so a depth D arm feeds D ids. The references are mainline llama.cpp's arms (lcpp:,
# lcpppp:); ik and mistral.rs arms are refused by name. BLOOMERY_DRY=1 prints the command lines and builds nothing. A
# BLOOMERY_MODEL other than nemotron_h is refused by name (exit 64), not overridden: the Bolt profile runs through
# depth-gpu-qwen3moe.
depth-gpu-nemotron *ARMS:
    case "${BLOOMERY_MODEL:-nemotron_h}" in nemotron_h) ;; *) echo "depth-gpu-nemotron: BLOOMERY_MODEL=$BLOOMERY_MODEL: this recipe times the nemotron_h profile; another profile runs through depth-gpu-qwen3moe (BLOOMERY_MODEL=$BLOOMERY_MODEL just depth-gpu-qwen3moe ...)" >&2; exit 64 ;; esac && BLOOMERY_MODEL=nemotron_h just depth-gpu-qwen3moe {{ARMS}}

# GLM-5.3-Flash 디코드를 깊이별로, 프리필을 길이별로(A6000, 한 임대, 리드 전용):
# ours `<D>` (the first D prose ids, run as the binary's `--prefill` default, batches; `kind=` in the row names the feed),
# llama.cpp PR 두 가지 `lcpp27752:<D>`·`lcpp27754:<D>`(-d D)와 `…pp[<U>]:<P>`,
# llama-server 한 요청으로 재는 `…srv:<D>`·`…mtp:<D>`(MTP 초안, 서버가 찍는 draft acceptance 줄을 행에 그대로 싣는다),
# exllamav3 `exl3:<D>`·`exl3pp:<P>`(perf.py, EXL3 4.05 bpw — 다른 양자화라 비율 표에 넣지 않는다). GGUF 팔은 바퀴마다 순서를
# 돌리고 exllamav3 팔은 그 뒤 한 덩어리로 돈다. 팔·플래그 근거는 tools/ref/depth-glm5next.sh와 models/glm5next.sh 머리에.
# 우리 팔이 없거나 BLOOMERY_BOX_ENV=BLOOMERY_DRY=1이면 generate_glm5next를 빌드하지 않는다; 인자 없으면 512 lcpp27754:512.
depth-gpu-glm5next *ARMS:
    BLOOMERY_MODEL=glm5next ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { ours=; [ -n "{{ARMS}}" ] || ours=1; for a in {{ARMS}}; do case $a in *:*) ;; *) ours=1 ;; esac; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,vision,hf,glm5next-hf --release --bin generate_glm5next; fi; } && bash tools/ref/depth-glm5next.sh {{ARMS}}'

# Any model's decode and prompt rate through `bloomery-serve`, one fresh server an arm and a round (A6000, or this
# machine's card under tools/box.sh's local mode; one lease, lead-only): `<tree>:<D>` is ~/repo/<tree>'s bloomery-serve
# on the first D ids of the model's prose corpus, `env=K=V,K=V:<tree>:<D>`
# sets levers for that arm's server only, the arms rotate every round (BLOOMERY_AB_ROUNDS, default 4). Rows: tg and pp tok/s
# from the server's own `timings`, the load seconds and the major faults; then per arm the mean and, against the first arm of
# its D, the difference with its 95 % interval. BLOOMERY_MODEL names the profile (tools/ref/models/<name>.sh) and is required.
# The context, n_predict and port ride BLOOMERY_BOX_ENV as DEPTH_SERVE_CTX, DEPTH_SERVE_N and DEPTH_SERVE_PORT (defaults: the
# serving default's context, 128, 18180; a set context must hold D + n_predict - 1); the header of tools/ref/depth-serve.sh has the arms, the rates and the
# refusals. The binary requires the features of every seat (`required-features` of the bin), whatever the model: an arm that
# names this tree builds it here, an arm of another tree needs that tree's own build of this line. Under
# BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built and no lease is taken: the plan and each binary's state.
depth-serve *ARMS:
    @[ -n "${BLOOMERY_MODEL:-}" ] || { echo "depth-serve: BLOOMERY_MODEL is unset: name the model profile (tools/ref/models/<name>.sh), e.g. BLOOMERY_MODEL=mimo2 just depth-serve <tree>:<D>" >&2; exit 64; }
    ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && { here=$(basename "$PWD"); ours=; for a in {{ARMS}}; do t=${a#env=*:}; [ "${t%%:*}" != "$here" ] || ours=1; done; if [ -n "$ours" ] && [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve; fi; } && bash tools/ref/depth-serve.sh {{ARMS}}'

# The schemajump against Clef-Flash sitting: the Clef decide seat and a Qwen3.6 seat, both bloomery-serve, one lease on
# the A6000, the arms alternating request by request (the design's §4; the cards are docs/cards/sj-*.card). Two windows
# of one lease: window 1 is Clef + the Qwen3.6 jump arm, window 2 is Clef + the same binary with the jump
# arm's lever off (the step arm; the runner names the lever), the Clef arm repeated in both as the anchor. The runner is
# tools/ref/sj-vs-clef.sh (its header has the windows, the requests and the tables); the requests are the SystemOne
# suites' converted to the Qwen seat's response_format, and the tables are the client wall (p50, p95 per suite and
# per bucket) beside the server's own split, the agreement against the official BF16 rows and against the Clef arm
# served, and the jump arm against the step arm. The knobs (the two model files, the ports, the rounds) ride
# BLOOMERY_BOX_ENV as SJ_CLEF_MODEL, SJ_QWEN_MODEL, SJ_CLEF_PORT, SJ_QWEN_PORT, SJ_ROUNDS and SJ_ROUNDS_LONG (the bounds
# of the servers and the request run, SJ_SERVER_BOUND and SJ_RUN_BOUND, are in the runner's header); the
# card is BLOOMERY_LEASE_CARD (docs/cards/sj-vs-clef-a6000.card for the baseline sitting, sj-jump-vs-step-a6000.card
# for the ab one) and the lease's card check reads the rounds from BLOOMERY_AB_ROUNDS. Under
# BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built, no lease is taken: the plan, with the card pick. Both servers run
# on the timing card (the A6000; BLOOMERY_TIMING_GPU names another, tools/ref/timing-card.sh): the runner pins it, waits out
# a compute process on it before each window, and reads it and the servers' logs after. SJ_WINDOWS names other windows
# (`ds41:on` and `ds41:off`: the V4.1-Flash server alone with the jump lever on or off, on the cards and placement
# bloomery-serve-ds41 picks, with SJ_DS41_MODEL; this recipe builds that binary when the windows name it) and SJ_SUITES
# other suite files (`tools/ref/clef/suite-hard.jsonl`, scored against its `.gold.jsonl`); the header of the runner has both.
sj-vs-clef:
    ./tools/box.sh "${BLOOMERY_AB_ROUNDS:+export BLOOMERY_AB_ROUNDS=$BLOOMERY_AB_ROUNDS && }"'{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then case " $(printf %s "${SJ_WINDOWS:-qwen:on qwen:off}" | tr , " ") " in *" qwen:"*) cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve;; esac && case " $(printf %s "${SJ_WINDOWS:-}" | tr , " ") " in *" ds41:"*) cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41,vision,gemma-embedding2,hf --release --bin bloomery-serve-ds41;; esac; fi && bash tools/ref/sj-vs-clef.sh'

# Qwen3-30B-A3B decode-step kernel timeline (nsys, A6000, under the lease, lead-only): generate_qwen3moe's seed form at each
# depth (default 6 4096), one prefill pass + BLOOMERY_NSYS_N - 1 replays, the cache height depth-qwen3moe.sh uses. The
# header of tools/ref/nsys-gpu.sh has the boundary and the windows. Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built.
nsys-gpu-qwen3moe *DEPTHS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi && BLOOMERY_NSYS_DEPTHS="{{DEPTHS}}" bash tools/ref/nsys-gpu.sh'

# Qwen3.6-35B-A3B (qwen35moe) decode-step kernel timeline (nsys, A6000, under the lease, lead-only): the seed form of
# nsys-gpu-qwen3moe on the qwen35moe profile, generate_qwen3moe opening the file as Body35. A depth entry is `D` or
# `D@model=<absolute path>` for another file of the family (the profile's own is the Q4_K_M); each profile takes its
# boundary entry from its file's token_embd type, so one lease can profile the Q4_K_M and the UD-Q4_K_XL. The header of
# tools/ref/nsys-gpu.sh has the boundary and the windows. Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built and no
# lease is taken.
nsys-gpu-qwen35moe *DEPTHS:
    BLOOMERY_MODEL=qwen35moe ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi && BLOOMERY_NSYS_DEPTHS="{{DEPTHS}}" bash tools/ref/nsys-gpu.sh'

# Qwen3-30B-A3B prefill kernel table (nsys, A6000, under the lease, lead-only): generate_qwen3moe's timed prompt (the
# depth runner's `<P>` arm) at each length P (default 4096, one ubatch at the default ubatch size), then
# BLOOMERY_NSYS_N - 1 replays; the table is the prompt's window, its units and the head, per kernel, with the grouped
# GEMM summed. The header of tools/ref/nsys-gpu.sh has the boundary. Expected at P = 4096 on main [derived,
# https://github.com/midagedev/rig-log/blob/main/docs/bloomery/research/q3next-design-report.md; not numbers of record]: gemm_q4k + gemm_q6k about 62 % of the prompt,
# softmax_router_logits 0.36-0.54 ms a launch. Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built.
nsys-gpu-qwen3moe-prefill *PROMPTS:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin generate_qwen3moe; fi && BLOOMERY_NSYS_FORM=prefill BLOOMERY_NSYS_DEPTHS="{{PROMPTS}}" bash tools/ref/nsys-gpu.sh'

# V4.1 디코드 스텝의 커널 타임라인(nsys, A6000, 임대 안, 리드 전용): 깊이마다 커널 합 대 호스트 합류 빈틈. 인자는 깊이 목록
# 또는 말뭉치 팔(prose:<P>, code:<P> — depth-ds41.sh의 말뭉치 팔이 먹이는 같은 id). BLOOMERY_GEN_PLACE(a|gate|bp)와
# BLOOMERY_TIMING_CARDS=a6000+3090(bp의 두 카드 모드), BLOOMERY_RESIDENCY 등 BLOOMERY_BOX_ENV로 실은 레버가
# 프로파일 대상에 그대로 전해지고 [levers] 줄로 찍힌다; 레지던시가 켜진 실행은 엔진 스트림 옆 복사 스트림 표
# (tools/ref/ds41copy.py)를 더한다. BLOOMERY_BOX_ENV=BLOOMERY_DRY=1이면 빌드도 임대도 없이 명령줄만 찍는다.
nsys-gpu-ds41 *DEPTHS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41; fi && bash tools/ref/nsys-ds41.sh {{DEPTHS}}'

# V4.1 prompt batch timeline (nsys, A6000, under the lease, lead-only): generate_ds41 --depth P -n 2 --mode graph --time at
# each prompt length P (default 512), the window from the prompt's first kernel to the first replay, cut into layer-batches
# at ds41_ffn_places and the joins against the run's stat lines; per layer-batch the route window, the union gap and the
# post, the kernel terms of one layer-batch (BLOOMERY_NSYS_LAYER, default 2) and of layers 2-39, the card's idle time and the
# launch queue from the CUDA API trace. The
# header of tools/ref/nsys-ds41.sh and tools/ref/ds41pp.py have the cut. Under BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 nothing is built.
nsys-gpu-ds41-prefill *PROMPTS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh '{{precheck}} && if [ -z "${BLOOMERY_DRY:-}" ]; then cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41; fi && BLOOMERY_NSYS_FORM=prefill bash tools/ref/nsys-ds41.sh {{PROMPTS}}'

# ik KLD 기준 파일 둘을 위치마다 비교한다(P = A, Q = B, 태그는 $BLOOMERY_DATA/ikppl 아래, 호스트만).
# ARGS: --ubatch N(여러 번 줄 수 있다), --ik <ik-ppl --kld 태그>(ik가 찍은 요약과 밴드 안에서 맞는지).
kld-diff A B *ARGS:
    ./tools/box.sh 'cargo build --release -p bloomery-gpu-gates --bin kld_diff && bash tools/host-gate.sh kld_diff "$BLOOMERY_DATA/ikppl/{{A}}.kld" "$BLOOMERY_DATA/ikppl/{{B}}.kld" {{ARGS}}'

# V4.1 본체를 게이트 배치(모든 층과 헤드, 예산 안에서 가장 큰 expert 접두)대로 엔진 입구를 거쳐 두 번 올린다
# (the 3090's plan on either card, shared/gate_card.rs).
# 세그먼트는 계획의 바이트대로, 슬롯 맵은 그 접두대로 올라가야 하고(카드 사본과 호스트 티어 사본이 같아야 한다), 층마다
# 상태는 KvLayout의 바이트와 같아야 한다. 위치 4·301·1025의 스텝 이미지를 되읽어 계획의 정수·RopeTable의 표와 맞춘다.
# 체인은 캡처돼야 하고(노드 수는 step 게이트의 몫) 합성 깊이는 거부돼야 한다. 두 번째 적재는 첫 번째와 같은 양을 가져가고,
# 드롭은 컨텍스트의 첫 캡처가 쥔 몫만 빼고 전부 돌려줘야 한다. 측정이 아니라 정확성 실행이다(파일 중 카드 몫을 두 번 읽는다).
# Then two loads more, the host tier reading the source (BLOOMERY_R8=off: no sidecar byte in its host set) and then the
# r8 sidecar, each fed 16 prose ids and a greedy step: the same argmax and logits bit for bit. The source load's populate
# reads the routed gates and ups from the file, and the r8 load, last, leaves the sidecar's pages cached.
[group('v41-load')]
gate-ds41-load:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_deepseek41_load && CUDA_MODULE_LOADING=EAGER BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_deepseek41_load'

# ik의 CUDA 답(프롬프트 33개의 다음 토큰): GPU 엔진 종단 게이트의 참조. 카드 선택과 오프로드
# 깊이는 dump.sh와 같다(박스 env의 3090 핀, -ngl 99). ik의 CUDA는 ubatch 하나에 9토큰 이상이
# 들어가면 쓰레기를 낸다(upstream의 MMQ 경로, mainline이 양자화한 파일에서만). 그래서 argmax.sh가 --step-prefill(M=1 경로)로 먹인다 —
# 사정과 되돌리는 법(BLOOMERY_REF_BATCH_PREFILL=1)은 argmax.sh 머리글. 8 GB 모델을 1분 안에
# 올렸다 내리는 정확성 실행이라 임대는 필요 없다. 카드에 다른 컴퓨트 프로세스가 있으면
# 끝나기를 기다린다 — 죽이지 않는다.
argmax-ref-cuda:
    ./tools/box.sh 'BLOOMERY_REF_BACKEND=cuda bash tools/ref/argmax.sh'

# 위의 greedy 연속(프롬프트 다음 토큰부터 32스텝, argmax_ref --gen): gen_ids·gen_margins
# 열을 가진 greedy-ik-cuda-32.tsv를 쓴다. 종단 게이트(gpu-gates::prompts)가 읽는 참조다.
greedy-ref-cuda:
    ./tools/box.sh 'BLOOMERY_REF_BACKEND=cuda BLOOMERY_REF_GEN=32 bash tools/ref/argmax.sh'

# P8b: 조립된 MoE 층(층 1) 스텝을 덤프와 대조 — 어텐션 절반 + 라우터·전문가 여섯·공유 전문가·결합.
# 입력은 오라클의 l_out-0, KV는 오라클의 kv_cache-1 앞 다섯 행. 밴드는 핀하지 않고 표만 찍는다.
gate-gpu-p8b *ARGS:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_p8b && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_p8b {{ARGS}}'

# PTX 스캔(계측기, 게이트 아님): 게이트 바이너리가 싣고 있는 디바이스 코드의 엔트리별
# 디포·로컬 왕복·블록 폭 표. 디포를 가진 엔트리가 먼저 나온다. 단언은 gate_p5/gate_p4가 한다 — 머리글 참조.
# 바이너리의 cargo 피처는 `--features`로 준다(기본 gpu). V4.1 게이트 바이너리는 `--features deepseek41`.
# The jit_regs and jit_local columns are the values the driver JIT actually assigned; the ld.global.scalar, ld.global.v2 and
# ld.global.v4 columns follow them. oxart_jit loads the modules onto the card to read them, so a full scan runs under the
# gate lock (tools/gpu-gate.sh).
# `--no-jit`: the driver JIT is not built or run, so the scan takes no card lock and waits on no landing batch's GPU hold.
# The banner names `jit=skipped(no-jit)` and both JIT columns read `skipped(no-jit)`. The md5 block and the ptxas columns
# are as in a full scan: that proves a move (`just affected --narrow` reads such a pair as identical or refuses it by name).
# The add and occupancy classes and gate-ptx-spill need the full scan. `-- --no-jit` among the words does the same.
# BIN 뒤의 말은 ptx-scan.sh의 엔트리 부분 문자열 하나뿐이다. 자리로 준 피처(`gpu,deepseek41`, 피처 이름)나 플래그는
# 빌드 전에 거절한다(tools/scan-args.sh, 세 스캔 레시피 공용) — ARGS로 흘러 기본 피처로 빌드되면 트랙의 바이너리를
# 호스트 전용으로 덮어쓴다.
[arg("FEATURES", long="features")]
[arg("NOJIT", long="no-jit", value="--no-jit")]
ptx-scan BIN FEATURES='gpu' NOJIT='' *ARGS:
    @bash tools/scan-args.sh ptx {{BIN}} {{NOJIT}} {{ARGS}}
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features {{FEATURES}} --release --bin {{BIN}} {{ if (NOJIT + " " + ARGS) =~ "(^| )--no-jit( |$)" { "" } else { "--bin oxart_jit" } }} && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx && bash tools/ptx-scan.sh {{NOJIT}} {{BIN}} {{ARGS}}'

# PTX 스필 래칫 — 빌드 시점 게이트. generate_ds41(deepseek41)의 ptx-scan 표에서 엔트리마다
# spill(ptxas 스필 저장 바이트)·jit_local(드라이버 JIT의 스레드당 로컬 바이트)을 tools/ref/ptx-shapes.tsv와 tools/ref/ptx-shapes/의 표 중 스캔의 arch와 cc에 맞는 표의 핀과 대조한다.
# 핀 위든 아래든 다른 값, 핀 없는 새 엔트리, 스캔에서 사라진 핀 행이 전부 빨강이다 — 스필은 비트를 안 바꾸고 속도만
# 바꿔 비트 게이트가 전부 지나가므로(ds41_attn_seg_sel의 8 B 스필을 눈으로 찾았다), 0이 아닌 핀은 PIN 줄로 사유를 단다.
# The second binary, gate_qwen3vl_tower (--features vision), is scanned for the `vis_` entries of the image tower crate only
# (bloomery-gpu-vision, V4.1's six, the Qwen3-VL tower's five and the glm5v tower's three); the bloomery-gpu entries of the same bundle are pinned by
# generate_ds41. One binary covers bloomery-gpu because gate_e2e, the other binary that links it, carries the same bundle: it links
# no device crate generate_ds41 does not, with the same features on each, and declares no kernel of its own
# (`tools/check-ptx-names.py` holds that, from the manifests).
# [group('canary')]: tools/gate-batch.sh runs the canary recipes first in their lanes and stops the batch at the first red one (its
# header's «Canary»); gate-ptx-spill and the lib-test gates of the crates carry it, the gates a half-applied change turns red first.
[group('canary')]
gate-ptx-spill:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu,deepseek41 --release --bin generate_ds41 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features vision --release --bin gate_qwen3vl_tower --bin oxart_jit && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx && bash tools/ptx-spill-check.sh tools/ref/ptx-shapes.tsv tools/ref/ptx-shapes/ generate_ds41 gate_qwen3vl_tower:vis_'

# 인자: `<엔트리> n,t,…`(분기 결정 한 줄을 따라간 경로), `<엔트리> list`(목록).
# SASS 스캔(ptx-scan의 짝, 계측기): 첫 대기 전에 발행된 전역 로드 수를 루프마다, 그리고 한 경로를 따라 센다.
# 자리로 준 피처와 sass-scan.sh가 받지 않는 플래그는 ptx-scan처럼 빌드 전에 거절한다(tools/scan-args.sh).
[arg("FEATURES", long="features")]
sass-scan BIN FEATURES='gpu' *ARGS:
    @bash tools/scan-args.sh sass {{BIN}} {{ARGS}}
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features {{FEATURES}} --release --bin {{BIN}} && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx && bash tools/sass-scan.sh {{BIN}} {{ARGS}}'

# 공유 로드·스필 스캔(계측기, 게이트 아님): 엔트리마다 PTX 본문의 ld.shared(폭별)·ld.global·st.shared·shfl·fma·
# cvt.f16 수와, ptxas가 만든 SASS의 STL·LDL·LDS(폭별) 수를 한 줄에 센다. ptx-scan의 spill 열(저장+적재 바이트)이
# 어느 쪽에서 왔는지, 공유 로드가 벡터로 합쳐졌는지를 명령 한 줄로 가른다. 카드가 필요 없다(ptxas·cuobjdump만).
# 자리로 준 피처는 ptx-scan처럼 빌드 전에 거절한다(tools/scan-args.sh). 예: `just lds-scan gate_hc_gated hc_gated_up_mix`.
[arg("FEATURES", long="features")]
lds-scan BIN FEATURES='gpu' *ARGS:
    @bash tools/scan-args.sh lds {{BIN}} {{ARGS}}
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features {{FEATURES}} --release --bin {{BIN}} && cargo build --release -p bloomery-gpu-gates --bin oxart_ptx && bash tools/lds-scan.sh {{BIN}} {{ARGS}}'

# 판정 비교(맥에서 돈다): 두 트리에서 같은 레시피를 돌리고, 빌드 줄·시간·pid·스레드 id를 가린 뒤 출력을 비교한다.
verdict-diff *ARGS:
    python3 tools/verdict-diff.py {{ARGS}}

# 바퀴마다 팔 순서를 돌리고, 임대는 기존 러너가 잡는다. `run --dry-run`은 계획만 찍는다.
# GPU A/B(리드 전용): (트리, env) 팔을 시간 레시피로 번갈아 돌리고 팔별 평균·SD·첫 팔 대비 차와 그 구간을 낸다.
gpu-ab *ARGS:
    python3 tools/gpu-ab.py {{ARGS}}

# DSpark 드래프트 파일 읽기: MXFP4 디코드 단위 시험(스케일 바이트 256개 × 코드 16개 전부)을 돈 뒤, 엄격한 리더로
# 드래프트를 열어 hparams·텐서 표·바이트 타일링을 보고, mxfp4_ref 덤프(just build-ref)와 비트 대조하고, 전문가 스케일
# 바이트를 전부 훑는다. 바이트 표를 찍는다. 대상 쪽 텐서(token_embd·output)를 읽으려고 V4.1 프로필로 돈다. 호스트 전용.
gate-dspark-read:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-dspark-read && bash tools/gate.sh --release -p bloomery-gguf --lib -- quant --nocapture && cargo build --release -p bloomery-gpu-gates --bin gate_dspark_read && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && bash tools/host-gate.sh gate_dspark_read'

# DSpark 드래프트의 두 커널(f32 가중치 HC_PRE, bf16 Markov 헤드 + argmax)을 카드에서 호스트 규칙과 비트 단위로 대조한다.
# dsref 세트(code64_n32_w3) 블록 0에 대한 ik와의 거리는 찍기만 한다(진단, 핀 아님).
gate-gpu-dspark-hc:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-dspark-hc && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features deepseek41 --bin gate_dspark_hc && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_dspark_hc && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_dspark_hc'

# DSpark 드래프트의 routed MoE(MXFP4 expert gate·up·SwiGLU와 down+결합, 128개 중 3개 라우터)를 카드에서 호스트 규칙과
# 비트 단위로, f32 참조와는 q8_1 한계와 핀으로 대조한다. dsref 세트 블록 0 층 0에 대한 ik와의 거리는 찍기만 한다.
gate-gpu-dspark-experts:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-dspark-experts && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_dspark_experts && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_dspark_experts'

# DSpark 드래프트를 3090에 올리고(텐서별 카드 형식, 그룹별 바이트 = 인벤토리, cuMemGetInfo 증감), 특징→KV 그래프를
# dsref 세트 블록 0–3에 대조한다: main_x와 층마다 링 행이 ik의 q8_1 활성값 규칙이 허용하는 유도 한계 안.
gate-gpu-dspark-kv:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-dspark-kv && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_dspark_kv && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_dspark_kv'

# DSpark 드래프트의 블록 패스와 헤드를 ik가 떨군 dsref 세트와 대조한다. ik 규칙 arm(블록 인과 마스크, SwiGLU clamp 없음 —
# ik 드래프트가 도는 방식)은 블록 0–2의 모든 탭을 유도 밴드로 판정하고, 엔진이 도는 기준 규칙(model.py) arm은 두 규칙이
# 갈리지 않는 행만 판정한다. 블록마다 draft_argmax, w=5 비트 동일, 캡처 노드 수 = 커널 목록(패스 73 + 8w, 링 append 8). DraftBody가
# 로드 때 잡은 그래프는 블록 0–2·폭 1–5에서 eager와 비트 동일, 세트 전체에서 id와 링이 eager 쌍둥이와 같다. 수락 수는 찍기만 한다.
gate-gpu-dspark-graph:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-dspark-graph && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41 --release --bin gate_dspark_graph && __s=$(. tools/ref/ref-paths.sh && dspark_model) && export BLOOMERY_DSPARK_MODEL="$__s" && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_dspark_graph'

# DSpark Markov 헤드 단독 초안(E6)의 오프라인 수락률 — 시간을 재지 않는 CPU 실행이고 임대를 잡지 않는다. 먼저 self-test,
# 다음 코퍼스 스트림마다 앞 50k토큰의 acc/positions·top-2·top-4와 E5 markov1 재계산, 이전 토큰별 argmax 표의 해시와
# 위치별 경로와의 불일치 수(0이어야 한다)를 찍는다. 목표는 코퍼스 텍스트 자체다. 인자는 bin에 그대로 간다(--tokens N).
markov-accept *ARGS:
    BLOOMERY_MODEL=deepseek41 ./tools/box.sh 'bash tools/ref/real-only.sh markov-accept && cargo build --release -p bloomery-model --bin markov-accept && bash tools/host-gate.sh markov-accept --self-test && S=$(. tools/ref/ref-paths.sh && dspark_model) && D=$BLOOMERY_DATA/engram && bash tools/host-gate.sh markov-accept "$S" $D/corpus-code.ids $D/corpus-prose.ids $D/corpus-prose-all.ids $D/corpus-korean.ids $D/corpus-threads.ids {{ARGS}}'

# V4.1 비전 오라클(V0): 공식 체크포인트의 image_processor.py·vision.py를 박스 torch로 3090에서(게이트 락 아래)
# tools/ref/vision/images/에 돌려 $BLOOMERY_DATA/ref-vision/deepseek41v/에 쓴다. 두 번 돌려 바이트 동일할 때만 설치하고,
# MANIFEST에 리비전·샤드 sha256·torch·Pillow·카드·mmproj를 적는다.
dump-ref-vision:
    ./tools/box.sh 'bash tools/ref/vision/dump-vision.sh'

# The V4.1 vision fork comparison (tools/ref/vision/): smalinin's llama.cpp fork at its pin, built under
# ~/repo/bloomery-visref-fork on the box with its row-injection harness visref_fork (build-fork.sh, CPU only, ~4 min).
build-visref-fork:
    ./tools/box.sh 'bash tools/ref/vision/build-fork.sh'

# The fork comparison's set: the official encoder's rows of tools/ref/vision/scenes/ (dump-vision.sh, set
# deepseek41v-scenes), then the fork's greedy answers and logits on each scene and two text controls per scene
# (visref.sh) into $BLOOMERY_DATA/ref-visref/deepseek41-scenes, family visref-deepseek41. Not timed, no lease; the
# fork runs under the card's gate lock and the V4.1 load lock.
dump-ref-visref:
    ./tools/box.sh 'BLOOMERY_VISION_SET=deepseek41v-scenes VISION_IMAGES=tools/ref/vision/scenes bash tools/ref/vision/dump-vision.sh && bash tools/ref/vision/visref.sh'

# JPEG decoder fixtures: dump-jpeg.py on the box encodes crops of tools/ref/vision/images/ with the system Pillow and
# decodes each file the way the reference does (its header lists the files); the set comes back whole into
# crates/vision/tests/fixtures/jpeg/, which the image tests read. Another libjpeg writes other bytes: commit the set whole.
dump-ref-jpeg:
    #!/usr/bin/env bash
    set -euo pipefail
    out=/tmp/$(basename "$PWD")-dump-jpeg
    ./tools/box.sh "rm -rf $out && /home/user/ft/bin/python3 tools/ref/vision/dump-jpeg.py --out $out"
    got=$(mktemp -d) && chmod 755 "$got"
    BLOOMERY_BOX_READONLY=1 ./tools/box.sh "tar -C $out -cf - ." | tar -C "$got" -xf -
    grep -q '^# complete' "$got/MANIFEST.tsv"
    rm -rf crates/vision/tests/fixtures/jpeg
    mkdir -p crates/vision/tests/fixtures
    mv "$got" crates/vision/tests/fixtures/jpeg
    ls -l crates/vision/tests/fixtures/jpeg

# V4.1 비전 호스트 게이트(V1): 리사이즈 플랜과 패드 기하, 패드된 u8 이미지와 bf16 패치(리샘플 유무 모두 비트 동일),
# 스팬 id·타입을 그 세트와 대조하고, mmproj 헤더의 hparams와 모든 텐서의 이름을 검사한다.
# It also runs tests/jpeg.rs: the JPEG decoder against Pillow on the fixtures of dump-ref-jpeg, within the pinned frontier.
# The video planner (src/video.rs) on synthetic probes: numpy's linspace rounding, the tick counts, overflow, the tail, the stamps,
# the budget's order (size first, then frames) and the named refusals; the temporal patch cut's frame order (src/preprocess.rs).
gate-vision:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-vision --lib --test vision --test jpeg -- --include-ignored --nocapture'

# V4.1 이미지 인코더 카드 게이트(V2): 전체 사슬(ViT 32블록 + aligner, bf16 텐서코어 GEMM, 비인과 어텐션, 2D RoPE)을
# 비전 오라클 세트의 모든 이미지에 돌려 공식 vision.py와 탭별로 대조한다 — embed·blk0·forced 탭은 절대 핀, 자유 실행 탭은
# 참조의 자기 민감도 행(MANIFEST `# sensitivity`)의 K배 안; 블록 0은 crates/gpu-vision의 호스트 규칙과 연산별로 대조. 3090, 게이트 락.
# grad-448 이미지가 없는 세트는 이름을 대고 거부한다; 그 단위 시험이 tools/gate.sh로 먼저 돈다.
gate-gpu-vision:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features vision --bin gate_vision_encoder && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features vision --release --bin gate_vision_encoder && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_vision_encoder'

# The Qwen3-VL image tower on the card (the `qwen3vl_merger` mmproj of Clef-Flash, Qwen3.6 and Qwen3.8): 27 blocks (heads of 72,
# LayerNorm, tanh GELU, the learned position table resized to the grid, 2D RoPE) and the 2x2 merger against every tap of the Clef
# set B (ref_clefvis_taps: mainline mtmd's graph nodes, with the CPU twin). A tap is held to K times the larger of the reference's
# rounding floor, its CPU twin's distance and the network's sensitivity to 1-ulp input noise; a one-step tap to K times the floor.
# Block 0 and the merger are held to their host rules op by op (position rows, norms and RoPE bit for bit, GEMMs inside the
# accumulation-order bound, attention inside its f32 bound, the tanh GELU at a pinned count). Also the card bytes (weights
# 924,123,136 B, scratch 570,949,632 B at 4096 tokens), 249 launches an image, and reuse after the largest image bit for bit. The
# Qwen3.6 and Qwen3.8 arms end in a named deferred line while their family's set is not in the tree (a deferral is not a pass).
# A set without the full-tap image is refused by name; the binary's unit test of that runs through tools/gate.sh first.
# Real file only; the gate lock.
gate-gpu-qwen3vl-tower:
    ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-qwen3vl-tower && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features vision --bin gate_qwen3vl_tower && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features vision --release --bin gate_qwen3vl_tower && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_qwen3vl_tower'

# The MiniCPM-V image tower on the card (the mmproj of MiniCPM-V 4.7: the Qwen3-VL block set plus a vit_merger of windowed attention and a
# 2x2 mean-residual down-sample, and a 2x2 MLP merger). The two kernels it adds, each against its host rule on seeded synthetic inputs:
# the windowed attention (16 heads of 72 inside consecutive windows of rows, every value inside the f32 bound of attn_bound around the
# exact attention of its own window, the count over it pinned at 0) and the group mean (bf16 in and out, ((p0 + p1) + p2) + p3 then
# x 0.25 then one rounding, bit for bit, with a control that the inputs exercise the sum's order), swept over 4, 64 and 1024 rows;
# the named refusal of a row count that is not a whole number of windows. Then the tower on the real mmproj: the card bytes (weights
# 1,101,559,552 B, scratch per patch of a tile), 230 launches a tile, the narrowed tensor; the vit_merger and the final merger
# teacher-forced against their f32 host twins on a real image's own taps (the erf GELU inside its libm bound); three images (448x448,
# a sliced 1000x700, 200x200) with finite rows, the rows each grid gives and reuse after the largest bit for bit. Then the tower against
# mainline's own nodes (the card's set ref_minicpmv47_vis_taps with its CPU twin and ref_minicpmv47_vis_preproc, under $BLOOMERY_DATA; the
# mmproj's sets, refused by name when dumped for another file): each tile of its three images fed mainline's own pixels, eleven taps from
# the patch embedding to the final merger's rows and the embd rows (merger rows then the tile's separator rows, which equal mainline's
# bit for bit) in a band of K_MARGIN x sqrt(N) times the larger of the twin's distance, the bf16 floor and the nudge sensitivity; our plan
# and canvas equal the set's. Real file only; the gate lock.
gate-gpu-minicpmv-tower:
    ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-minicpmv-tower && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features vision --release --bin gate_minicpmv_tower && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_minicpmv_tower'

# The image-tower crate's host tests (crates/gpu-vision, a device-linked crate, so cargo oxide test): the MiniCPM-V tower's merge-order
# rows and position buckets, the host twins of its two mergers, and its card bytes held equal to the plan's figure. They open no card
# (host group).
[group('canary')]
[group('host')]
gate-gpu-vision-lib:
    ./tools/box.sh 'bash tools/gate.sh --oxide -p bloomery-gpu-vision --release --lib'

# The image-tower crate's `hw_` tests, which open a card (the lib recipe above runs the rest): the Gemma4 tower's fused 3x3 pooling
# mean against the gather followed by the group mean, bit for bit, and the tower's held scratch bytes against its card figure. The
# shared GPU runner owns card selection and the runtime lock.
gate-gpu-vision-hw:
    ./tools/box.sh 'bash tools/gate.sh --gpu --oxide -p bloomery-gpu-vision --release --lib -- --ignored'

# The glm5v image tower on the card (the mmproj of GLM-5.3-Flash: llama.cpp's `glm4v` graph with a per-head q/k norm, a clamped
# SwiGLU and an erf GELU; 24 blocks of heads of 64, a 2x2 convolution merger) against the glm5next vision seat's tap set B
# (fx_ref_glm5next_vis_taps, with its CPU twin: mainline mtmd's `vit_out`, `after_fc_proj`, `after_ffn_proj` and `embd` of nine
# images). A tap is held to K times the larger of the reference's rounding floor, its CPU twin's distance and the network's
# sensitivity to 1-ulp input noise; the merger and projector MLP are held teacher-forced on the reference's `vit_out`. Block 0 and
# the merger are held to their host rules op by op (norms, per-head norms and RoPE bit for bit, GEMMs inside the
# accumulation-order bound, the merger's convolution against a host convolution of the file's weight, attention inside its f32
# bound, the clamped SwiGLU at pinned counts and the erf GELU inside its libm bound, on sweeps and in the chain). Also the card bytes (weights
# 1,128,026,112 B, scratch 2,016,768,000 B at 8000 tokens), 273 launches an image, and reuse after the largest image bit for bit.
# Fixture-only (tools/ref/fixture-only.sh): no real GLM-5.3-Flash vision encoder file or real-tier set exists yet, so a real-tier batch
# defers the recipe by name; run by hand on the real tier, the bin ends by name with 0 clauses run, never green on nothing. Drop the
# call when a real-tier set exists.
gate-gpu-glm5v-tower:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/fixture-only.sh gate-gpu-glm5v-tower && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features vision --release --bin gate_glm5v_tower && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5v_tower'

# The RADIO image tower on the card (the `nemotron_omni` mmproj of Nemotron 3.5 Bolt: 32 blocks of heads of 80, LayerNorm, erf GELU, a
# learned 128x128 position table resized half-pixel, ten class and register rows that attend and are dropped, a 2x2 pixel-shuffle
# projector with an RMS norm and a ReLU-squared MLP) against the Bolt vision seat's sets (ref_nemotron_h_moe_vis_preproc and
# ref_nemotron_h_moe_vis_taps: mainline mtmd's graph nodes and `embd` of nine images; the oracle is a CPU build, so there is no card run
# and no CPU twin). The host preprocess is held to set A bit for bit; a tap is held to K times the larger of the reference's
# rounding floor and the network's sensitivity to 1-ulp input noise; block 0, block 1 and the projector are held teacher-forced on the
# reference's own inputs. Block 0 and the projector are held to their host rules op by op (norms and epilogues bit for bit, GEMMs inside
# the accumulation-order bound, attention inside its f32 bound, the erf GELU inside its libm bound, ReLU-squared bit for bit, the
# position rows against the table and the rule). Also the card bytes (weights 1,723,781,120 B, scratch 593,251,328 B at 3,328 tokens),
# 230 launches an image, and reuse after the largest image bit for bit. Real file only; the gate lock.
gate-gpu-radio-tower:
    ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-radio-tower && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features vision --release --bin gate_radio_tower && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_radio_tower'

# glm5next (GLM-5.3-Flash) header gate: the reader's refusals on synthetic headers, then its description of the
# file and the coverage check's list. Headers only, seconds; needs all six shards.
gate-glm5next-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::glm5next --nocapture && bash tools/gate.sh --release -p bloomery-model --test glm5next_meta -- --ignored --nocapture'

# mimo2 (MiMo-V2.6-Flash) header gate: first a scalar KV-head count applies to every layer, and synthetic headers are
# refused by name (short per-layer arrays, the two QKV forms, wrong dimensions); then the RL file's header is pinned: the
# per-layer KV-head and window arrays, the two head widths (key 192, value 128), the window, sinks and value scale
# per layer, the rope base per layer kind, the sigmoid noaux_tc router, the coverage list and the KV values per
# layer. A doctored real shard 0 with one KV array entry cut is refused by name, and so is the MTP-only file. Its
# lib run holds the program's unit tests too (`arch::mimo2::program`: the step's node and memory-operation counts, each
# layer's attention arguments and each block's from its description, a layer or a block the kernels do not run
# refused by name), which need no card. Headers only, seconds.
gate-mimo2-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::mimo2 --nocapture && bash tools/gate.sh --release -p bloomery-model --test mimo2_meta -- --ignored --nocapture'

# solar-open2 (Solar-Open2-250B) header gate: the reader's refusals on synthetic headers (an unread dense lead, a KDA layer
# with the GQA gate projection, `ssm_a` in ik's layout, a router or expert tensor of other dims), then the description of the
# Q4_K_M and IQ4_XS files: the GQA and KDA layer lines, the keys left unread, the tensors by role and type and the coverage
# list. The lib half is what runs the synthetic refusals, the common noaux block and the shared role walk that MiniMax-M3 reads
# through too; the test half is the files under /models. Headers only, seconds.
gate-solar-open2-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::solar2 arch::noaux arch::classify arch::minimaxm3 --nocapture && bash tools/gate.sh --release -p bloomery-model --test solar_open2_meta -- --ignored --nocapture'

# nemotron_h (Nemotron 3 Nano 4B) header gate: the reader's refusals on synthetic headers (a Mamba
# layer read as attention, nemotron_h_moe, a missing ssm key, a rope need on an attention layer, a
# state width other than 128), then the reader's description of the Q4_K_M file. Headers only,
# seconds. The lib half is what runs the synthetic refusals; the test half is the file.
gate-nemotron-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::nemotron_h --nocapture && bash tools/gate.sh --release -p bloomery-model --test nemotron_meta -- --ignored --nocapture'

# nemotron_h_moe (Nemotron 3.5 Bolt 74B-A9B) header gate: the reader's refusals on synthetic headers (a missing
# moe_latent_size, a per-layer array of the wrong length, a routed stack or bias whose expert dim is not its layer's
# router's, routed arrays off their layer's kind, expert_count not the largest router's, a MoE key missing or of a
# value the description cannot hold, a projection scale that is not one f32) and the coverage items of a latent MoE
# layer; the plan's three places for the routed experts (Experts::Card, Placed, Host) on the fixture layout (a card that
# holds every expert plans Placed as Card, a small card keeps each layer's id prefix and hosts the rest, the Host arm keeps
# none on the card, a context of no position, a tier card and a second card are refused under every arm); then the
# reader's description of the 24GB GGUF at the box's path (the test reads one fixed path) and, on its real header, the
# host tier's plans: whether the 3090 holds every routed expert, a partial plan, the 12 GiB room and all-host, each
# layer's prefix on the card and the rest on the host. Headers only, seconds.
gate-nemotron-h-moe-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::nemotron_h_moe --nocapture && bash tools/gate.sh --release -p bloomery-model --test nemotron_h_moe_meta -- --ignored --nocapture'

# The Bolt file's routed experts on the host tier (arch::nemotron_h_moe::host, the relu² form of the common union): every
# layer of the 24GB GGUF at the box's path through `host::layer` (a MoE layer builds as relu² at its own expert count and
# width, a Mamba-2 or attention layer as none, a layer past the model refused by name), then the file's narrowest and widest
# MoE layer through one union scratch made for the widest, at one column and nine, equal to experts_into bit for bit under
# both deferral arms. Reads the stacks of the experts the columns list: seconds.
gate-nemotron-h-moe-host:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --test nemotron_h_moe_host -- --ignored --nocapture'

# gemma-embedding2 (EmbeddingGemma 2) header gate: the reader's refusals on synthetic headers (a full
# layer at the sliding head, a missing tensor, another pooling, causal attention, a key head count
# that does not divide the heads, an odd rope, another tokenizer, a missing key) and the coverage
# items of an encoder, then the reader's description of the Q8_0 file at the box's path
# (BLOOMERY_BOX_ENV='BLOOMERY_MODEL=<path>' names another). Headers only, seconds.
gate-gemma-embedding2-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::gemma_embedding2 --nocapture && bash tools/gate.sh --release -p bloomery-model --test gemma_embedding2_meta -- --ignored --nocapture'

# gemma-embedding (EmbeddingGemma) header gate: architecture-derived bidirectional
# attention and the six-layer window pattern, the dense head and named refusals on
# synthetic headers, then the Q8_0 header. BLOOMERY_BOX_ENV can override BLOOMERY_MODEL.
gate-gemma-embedding-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::gemma_embedding:: --nocapture && bash tools/gate.sh --release -p bloomery-model --test gemma_embedding_meta -- --ignored --nocapture'

# Flint header-only reader and coverage refusals. BLOOMERY_MODEL names the pinned
# UD-Q4_K_XL GGUF or a prefix through its padded header; no payload or GPU is read. No profile
# holds the file, so a batch defers the recipe by name (needs-env) until BLOOMERY_BOX_ENV or the
# item's env sets BLOOMERY_MODEL to it; the reader's unit tests also run in gate-ops.
gate-flint-meta:
    ./tools/box.sh 'bash tools/ref/needs-env.sh gate-flint-meta BLOOMERY_MODEL && bash tools/gate.sh --release -p bloomery-model --lib -- arch::flint --nocapture && bash tools/gate.sh --release -p bloomery-model --test flint_meta -- --ignored --nocapture'

# mimo2 (MiMo-V2.6-Flash) end-to-end: the program loaded once by the gate placement, the 3090's bytes on either card
# (every routed expert on the host tier, the host set derived at about 161.5 GB), against ik's sets (refset `ik-mimo2`,
# dumped from the MOPD file): (s) the step's node count and its memory operations, the layer kinds, each layer's
# attention arguments against its description's and the file's pins, the stores' bytes and no routed expert on a card;
# (p) graph = eager bit for bit on the batch set; (c) the batch set's last argmax ik's or a named tie, each layer's
# output distance printed; (t) the argmax after each step set's prompt (the two 4-token sets, then the 1,024- and
# 4,096-position ones) ik's or a named tie. There is no fixture file for it yet, so it is a weekly (selected by its
# trigger rows in tools/gate-paths.tsv and its own text) and a fixture-tier run is refused by name (66). `--only
# s|p|c|t` runs one clause on the load, `--only r` the residency machine's clauses on a load of their own, and
# `--step-sets short|long` takes (t)'s sets by length. Loads the whole host
# set: alone in a batch, and under the big-load lock the V4.1 loads take.
[group('solo')]
[group('v41-load')]
weekly-gpu-mimo2-e2e *ARGS:
    BLOOMERY_MODEL=mimo2 ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features mimo2 --release --bin gate_mimo2_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_mimo2_e2e {{ARGS}}'

# glm5next (GLM-5.3-Flash) end-to-end: the program loaded once by the gate placement on the card the runner puts in view
# (the 3090's bytes; every routed expert on the host tier, the host set of about 185 GB populated), against ik's sets
# (refset `ik-glm5next`): the step's node count, graph = eager bit for bit, every layer's streams on the batch set within
# the derived band, and the argmax after each step set's prompt — here the two 4-token step sets (`--step-sets short`);
# the 1,024- and 3,070-position sets are weekly-gpu-glm5next-e2e-long's. Right after the load, before those, the card
# experts (shared/glm5next_card.rs): the slot map holds each layer's id prefix [0, n_l) in ascending order, only on layers
# whose stacks the card reads; at slots 0, n/2 and n-1 each stack's `_sel` and the gate·up are that expert's own upload
# from the file, bit for bit; the card copy of the map is the host map at its row offsets. In the fixture tier the real
# file's clauses that read ik's sets (free, forced, the step sets) are Oracle clauses, deferred to the real tier by name,
# and ik's sets dumped on the fixture file (`fx_ref_glm5next*`) take their place: free, ik-routed (ik's picks and weights
# planted), forced and the two 4-token step sets, each a fixture-oracle clause the gate declares, held to bands counted
# for the fixture's random weights, an argmax as a tie band; every other clause runs on the fixture. Loads the whole
# model: alone in a batch in the real tier, under the big-load lock the V4.1 loads take; balanced over the cards in the
# fixture tier, which takes no lock. The slots arm's (sdm) holds the NextN draft over a media call: at a span's positions
# it reads the span's row, the row the target consumed there, in place of the image token's embedding (feed B, the
# reference's: llama.cpp's MTP draft adds the target's vision row at an image position).
# PIN(2026-10-09): its own bound, 1200 s. The measured median is 773 s (train028b/028c/028f's times rows), past 0.75 x
# the default 900 the batch's budget refusal names; the gate's arms admit no smaller split (`--only` takes one arm and
# `--only slots` runs the stagger's loads too, so main, pp and slots are three items). A split pays no cold load per
# item: the cold first load, 50.7 s, is the batch's first GLM process's alone (train029a's 3090 log, inside its main
# arm of 316 s); a later process's first load there took 18.2-24.9 s (residency, stagger) against 16.0-18.9 s for a warm
# load inside one process, so the two added items cost at most 2 x (24.9 - 16.0) = 17.8 s [derived] of load plus their
# process starts, and the chain, which runs one item at a time under the V4.1 load lock, takes nothing back: one item
# stays. The slots arm's (sd), (s3) and (sdm) run on the harness's load, 16.4 s of load less (the same log), so about
# 773 - 16.4 = 757 s [derived], still past 675 s. 1200 keeps the kill at 1.55 x the median while the budget check reads
# 900 s, the default bound itself.
[group('solo-real')]
[group('v41-load')]
gate-gpu-glm5next-e2e:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,vision --release --bin gate_glm5next_e2e && BLOOMERY_GATE_BOUND=1200 BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_e2e --step-sets short'

# The same gate's drafted slots pass (`--only stagger-draft`, (h1)-(h4)) on a NextN load of its own at SLOT_CTX, its plan
# counting two slots: each slot's verify rows, two a slot, as one pass of four (`GpuModel::verify_slots` and
# `commit_slots`) against each slot's drafted run alone, in graph and eager mode; forced kept rows; the refusals while a
# pass waits for its commit and of the passes a NextN load does not run, a pass that keeps every row
# (`GpuModel::step_slots`) among them; the captured four-row pass against the verify pair's. The gate's run with no
# `--only` leaves these clauses out. Loads the whole model once: alone in a batch in the real tier, under the big-load
# lock the V4.1 loads take; balanced over the cards in the fixture tier, which takes no lock.
[group('solo-real')]
[group('v41-load')]
gate-gpu-glm5next-stagger:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,vision --release --bin gate_glm5next_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_e2e --only stagger-draft'

# GLM-5.3-Flash's prompt batch takes image rows (`gpu-glm5next/src/media.rs`, `prefill_media`): the crate's host tests
# (the position rule, the image-token check, the bf16 rows widened into the four streams), then (im) of
# gate_glm5next_e2e (`--only media`, `shared/glm5next_media.rs`): the fixture's prompts c1-c3 fed mainline's own tower
# rows of the image-input sets (`fx_ref_glm5next_vis_*`, `just dump-ref-glmv`), the last position's `result_norm` held
# within the CPU twin's distance of the CUDA run and nearer it than the text-only control, and the media refusals by
# name with the model unmoved. Fixture-only (tools/ref/fixture-only.sh): no real GLM-5.3-Flash vision encoder file or real-tier set
# exists yet, so a real-tier batch defers the recipe by name; run by hand on the real tier, the bin fails by name before any load (0
# clauses). Drop the call when a real-tier set exists. Loads the fixture once, on either card.
gate-gpu-glm5next-media:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/fixture-only.sh gate-gpu-glm5next-media && bash tools/gate.sh --oxide -p bloomery-gpu-glm5next --release --lib && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,vision --release --bin gate_glm5next_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_e2e --only media'

# The tensor-parallel GLM-5.3-Flash body (`gpu-glm5next/src/tp`): the crate's tests with its `hf` feature (each rank's
# part of a tensor, the head's merge, and a checkpoint folder's rank parts joined into the whole on the card), then
# gate_glm5next_tp, one, two and four ranks on the one visible card, each a context of its own,
# their parts carried by a loopback: one rank against the one-card body, every rank alike and within the fixture's bands
# of one rank, a rerun bit for bit, and each run against ik's batch set (the bin's doc lists the clauses). The fixture's
# card budget is unset: the one-card body (n1) compares against holds every expert its plan admits on the card, and the
# bin refuses the lever. Fixture-only (tools/ref/fixture-only.sh): the real file's ranks are loaded across nodes, not on
# one card.
gate-gpu-glm5next-tp:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'bash tools/ref/fixture-only.sh gate-gpu-glm5next-tp && bash tools/gate.sh --gpu --oxide -p bloomery-gpu-glm5next --release --features hf --lib -- --include-ignored tp:: && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features glm5next --bin gate_glm5next_tp -- --include-ignored && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_tp && unset BLOOMERY_CARD_BUDGET && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_tp'

# The serve seat's media call of the same fixture (`bloomery-serve --mmproj` on a GLM file), gate_glm5next_e2e's `--only media-seat`
# ((sm), `shared/glm5next_media.rs`'s `seat`, built with the `vision` feature the other recipes leave out): the encoder file of
# the fixture's image sets, seated as the seat seats it and loaded as its resident tower; two procedural images of known grids
# encoded on it and spliced through the seat's own steps (`Resident::encode`, `Encoded::prompt`), held to
# `prefill_media` called with spans put together from the images' sizes, the same argmax and a bit-identical last hidden row; a
# control without the rows that lies away from it; and the tower's refusal by name of a grid past its cap. No oracle: the
# numbers' quality is the tower's gate's and (im)'s. Fixture tier only: on the real tier the bin fails by name before any load
# (0 clauses). Loads the fixture once, on either card.
gate-gpu-glm5next-media-seat:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,vision --release --bin gate_glm5next_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_e2e --only media-seat'

# The same gate's long step sets (`--only main --step-sets long`): the load at CTX and its clauses, with the 1,024-position
# set and the 3,070-position `--dsa` set, the only comparisons with ik past 4 positions and the only end-to-end run past
# the dense limit. Weekly: `just weekly` runs it, and `just affected` names it when a file its triggers in
# tools/gate-paths.tsv match changes. In the fixture tier the same load runs ik's sets dumped on the fixture file
# (`fx_ref_glm5next_d1k`, `fx_ref_glm5next_d3kdsa`) as fixture-oracle clauses, the prefill and the step planted with ik's
# routes and held to bands counted for the fixture's random weights: those two clauses are all this run adds to
# gate-gpu-glm5next-e2e's main arm, whose other clauses it repeats.
[group('solo')]
[group('v41-load')]
weekly-gpu-glm5next-e2e-long:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,vision --release --bin gate_glm5next_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_e2e --only main --step-sets long'

# glm5next (GLM-5.3-Flash) NextN draft: the target loaded twice by its NextN plan on the card the runner puts in view
# (the 3090's bytes in the real tier), without the next-token layer and with it (`Body::open_placed_nextn`), against ik's MTP
# draft set (refset `mtp-glm5next`): the set's MTP input (no position mask), the walk's refusals, every graph of the set
# replayed from ik's inputs with each block's proposal the set's draft, the two loads' plain runs bit for bit, and the
# drafted windows (`MtpDraft<Body>`) the plain run's ids with a rejected and an accepted row, the prompt in batches and by
# steps. In the fixture tier the clauses that read ik's set are Oracle clauses and the ones that need the file's trained
# draft (a window that accepts a proposal, two slots keeping different counts) are file-bound premises, all deferred to
# the real tier by name; the plain-run, drafted-ids, state-byte and slot clauses run on the fixture. Loads the whole model
# twice: alone in a batch in the real tier, under the big-load lock the V4.1 loads take; balanced over the cards in the
# fixture tier, which takes no lock.
[group('solo-real')]
[group('v41-load')]
gate-gpu-glm5next-mtp:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_mtp && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_mtp'

# glm5next (GLM-5.3-Flash)'s attach of the NVMe expert tier (shared/nvtier_family.rs, whose clauses V4.1's gate calls first): a
# paged arm against an all-RAM arm on the same file on the gate placement, the residency off — the paged plan's arena is the
# given slots (sized from the file: at most 64, fewer where the need leaves less above the floor), the tier built and covering
# every paged layer, one prompt call and greedy steps on the host leg's paged arm bit for bit the all-RAM arm's, the card
# leg's paged arm (a pinned arena: on an integrated device by default, else a child of the gate at BLOOMERY_NVTIER_MEM=pinned)
# claiming on the step port only the ids the leg held and replaying its eager steps bit for bit, the
# room under the floor and the arena under its floor each refused by name, the NextN plan refused at the room the draft
# layer's routed experts leave under the floor and planned above it, and the unset residency as the plan's numbers say,
# never why `paged`.
# Each arm's room and arena reach the plan through PlanInputs, so no environment crosses the tiers; the one variable the gate
# sets is its arm child's BLOOMERY_NVTIER_MEM. The fixture tier runs on the plain fixture under the least card budget its own
# plan accepts (derived in the gate off the plan's CardBudgetFloor: no routed expert on the card, so every routed layer's
# experts are the host's and the tier pages them). Group solo-real, reason r1 (the real tier loads the whole file three
# times, one load at a time), balanced over the cards in the fixture tier, under the big-load lock the V4.1 loads take.
[group('solo-real')]
[group('v41-load')]
gate-gpu-glm5next-nvtier:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features glm5next --bin gate_glm5next_nvtier && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_nvtier && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_nvtier'

# glm5next (GLM-5.3-Flash) on plan (b′): the stage on the A6000, the expert tier on the 3090 (`app::arch::glm5next::open_pair`
# at --place bp, two KDA lanes, under a card budget on both cards: 16 GiB in the real tier, the header's in the fixture
# tier), against a reference load of the same plan whose
# A6000 holds the stage's and the tier's experts with no tier: the step leg (32 prompt steps, 48 greedy), the pair leg (16
# verifies of two rows, a third rejected) and the call leg (one 512-position prompt batch, 4 steps after) bit for bit the
# reference's, tokens, kept counts and logits rows; the stage's step graph of the reference's node count; every tier layer
# sent a routed slot by each leg, the call's batch served by the tier. Two loads, one after the other. Then two more
# runs, each under its own bound: --residency (the plain and the NextN load under mid-p0-s1 beside the tier: the same
# history twice with flips landed, and no tier expert moved — the host map's tier entries, the tier's set and the stage
# card's copy of the map), and --nextn (a NextN plan with a next-token expert on the tier refused by name before any
# upload, then the drafted ids and windows on the two cards against the one-card NextN load of the union, the walk on
# the stage card). Two loads each. Then --records: generate_glm5next's load record names the A6000 alone under --place a
# and the A6000 then the 3090 (with the tier's experts) under bp, one child process a placement. Both cards
# (BLOOMERY_CARD=both: both gate locks), alone in a batch, in both tiers: a tier is two cards by construction. In the
# fixture tier the clauses that need the file's trained weights (a pair that accepts a proposal, flips landed and an
# expert admitted, the plan with no budget leaving the tier experts, the unset rule's bp at the real file's tier size)
# are file-bound premises deferred to the real tier by name; every bit-for-bit clause runs on the fixture.
[group('solo')]
[group('v41-load')]
gate-gpu-glm5next-twocard:
    BLOOMERY_MODEL=glm5next BLOOMERY_CARD=both ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,vision,hf,glm5next-hf --release --bin gate_glm5next_twocard --bin generate_glm5next && bash tools/gpu-gate.sh gate_glm5next_twocard && bash tools/gpu-gate.sh gate_glm5next_twocard --residency && bash tools/gpu-gate.sh gate_glm5next_twocard --nextn && bash tools/gpu-gate.sh gate_glm5next_twocard --records'

# GLM's adaptive expert residency on the gate placement (BLOOMERY_RESIDENCY set in the gate at mid-p<P>-s1, P half the
# plan's least card slots a layer): the churn pool's host refusal at load, the batch prompt one pass keeping 0 and each
# step 1, every admitted slot byte for byte the file's on a Q4_K layer and on the Q5_K gate/up layer (per-layer parts),
# the steps prompt refused by name, the same history twice with flips landed, and a reset back to the seed. Every clause
# runs on the fixture but the one that needs the file's routing skew (flips landed and an expert admitted), a file-bound
# premise deferred to the real tier by name. Loads the whole host set: the e2e gates' batching rule.
[group('solo-real')]
[group('v41-load')]
gate-gpu-glm5next-residency:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next --release --bin gate_glm5next_residency && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_glm5next_residency'

# The GLM seat of bloomery-serve (--model glm) on the gate placement, against generate_glm5next on the same card
# (gate_glm5next_serve's header has the clauses). Two processes, each under its own gate-lock hold and bound, the loads
# one after the other: --arm plain first runs the seat with both levers unset and --plan at --place a and gate (the
# unset rule's records, no load), then holds the served greedy ids of one chat turn's prompt to the plain CLI's under
# BLOOMERY_DRAFT=off BLOOMERY_RESIDENCY=off and under BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=off (three loads); --arm
# drafted (BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=mid-p0-s1, the clip's levers) holds the draft's records and counts, the
# residency's records, its reset and the same ids after it, and the ids against the CLI under the same levers through
# the first landed flip (two loads). --arm plain ends with the seat as a user starts it (no --place, --ctx or
# --parallel, no lever set: bp on the two cards), holding the unset rules' words at the load (draft mtp, residency
# mid-p0-s1, pinned 0) and a chat turn served. Logs in target/glm-serve-gate/{plain,plain/mtp,plain/unset,drafted}/.
# Loads the whole host set: alone in a batch, under the big-load lock. Both cards in view: the --plan arms' a and bp
# plan on the largest card and the next-largest; the loads stay on the gate placement's card (--place gate).
# Real-only: the gate states its clauses against the real file's numbers (the unset rule's bp pick, BP_TIER_EXPERTS and
# BREAK_EVEN; a draft that accepts; flips landed under the drafted arm's residency; the slots' kept-row bounds) and has no
# fixture-tier split yet, so the fixture tier defers the recipe to the real tier by name (tools/ref/real-only.sh).
[group('solo')]
[group('v41-load')]
gate-gpu-glm5next-serve:
    BLOOMERY_MODEL=glm5next BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-glm5next-serve && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,hf,glm5next-hf --release --bin generate_glm5next && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_glm5next_serve && D=target/glm-serve-gate && rm -rf $D && mkdir -p $D && bash tools/gpu-gate.sh gate_glm5next_serve --arm plain --dir $D/plain && bash tools/gpu-gate.sh gate_glm5next_serve --arm drafted --dir $D/drafted'

# The GLM seat's image input beside its draft (gate_glm5next_serve --arm media): one server of --parallel 2 under
# BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=off with --mmproj, the GLM-5.3 projector, then an image chat on one slot and a
# text chat on the other. (m1) the `vision` record names the card the gate placement leaves off (the A6000), seat=off-plan;
# (m2) the image chat is a 200 whose timings carry draft counts (the draft's media prompt call keeps its slot's draft on)
# and the seat prints no `mtp prompt` record naming a skipped why for it; (m3) the text chat is a 200 that drafts; (m4) no
# `mtp prompt` record of the scenario names a skipped why. Both chats print draft_n_accepted / draft_n. Logs in
# target/glm-media-gate/media/. One load of the whole host set: alone in a batch, under the
# big-load lock. Real-only: the gate states its clauses against the real file and the real projector, so the fixture tier
# defers the recipe by name (tools/ref/real-only.sh); a weekly, selected by its trigger rows in tools/gate-paths.tsv and
# its own text, and not part of gate-gpu-glm5next-serve. BLOOMERY_CARD=both: both gate locks, the tower on the A6000.
[group('solo')]
[group('v41-load')]
weekly-gpu-glm5next-media-serve:
    BLOOMERY_MODEL=glm5next BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/real-only.sh weekly-gpu-glm5next-media-serve && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_glm5next_serve && D=target/glm-media-gate && rm -rf $D && mkdir -p $D && bash tools/gpu-gate.sh gate_glm5next_serve --arm media --dir $D --mmproj /models/mmproj/unsloth--GLM-5.3-Flash-GGUF/mmproj-BF16.gguf'

# The shipped `bloomery-serve` started once per image flag on the GLM-5.3 fixture, then by catalog name and through the agent contract
# (gate_serve_start's header has the clauses): (s1) `-m <fixture>` with the serving defaults and one text chat; (s2) `--place a --mmproj
# <unsloth's mmproj-BF16.gguf>`, the tower's `vision` record and one image chat (free-448x448.png) answered with its 16 x 16 + 2 positions;
# (s3) `--hf bloomery-fixture/glm5next:fixture` offline from a cache of symlinks to the fixture and the projector, each beside a `.verified`
# marker holding the file's sha256; (s4) no server stderr of the run holds "does not print this kind" or "panicked"; (s5) `--chat-template-file`
# with the fixture's own template behind a marker: its call-carrying conversation renders the marker and then exactly what (s1) rendered;
# (s6) the need hook (`BLOOMERY_NEED_HOOK`): exit 75 ends the start before it listens with the hook's input the `need` record, exit 0
# lets it listen; (s7) `--model <catalog name>` offline from a cache laid out as the catalog row's repo: the `pick`, `bound` and `ready`
# events in order, a chat, stdout holding events only, and a SIGTERM ending it with exit 0 and a `stopped` event; (s8) `--model <name>
# --dry-run --json`: one document whose pick is the cached row, and a `start` that replays the pick by name (the same row, revision and
# context: its `--plan --json` equals the pick and the fixture file's); (s9) the loading window of (s7): `/health` is 503 `loading` from
# the `bound` event until the first 200, none before `ready` is printed and none after; (s10) the exit table on the refusals a start
# meets: an unknown name 64, a path given as a name 64 with `-m` in `next`, an embedding file 65, a card budget below the floor 80, a bad
# lever 78, a held port 83 found before the load, each ending in an `error` event of its code. The launcher's checks (`tests/serve_preflight.rs`) run beside it: the normal inputs reaching the fetch, and a seat word alone starting by name and refusing offline as `network`. Each of (s1) to (s3) is preceded by a `--plan`
# run of its arguments. It holds the image-flag starts at the fixture tier, which the real-tier seat gates do not. The bloomery-serve built is the release's
# (--features glm5next,clef,vision,mimo2,solar2). Both cards in view
# (BLOOMERY_CARD=both: both gate locks, alone in a batch): with `--place a` the other card is off the plan, so the tower sits there and no
# fixture card budget has to hold its reserve; one card in view would drop the repo's projector or refuse the reserve. Fixture-only
# (tools/ref/fixture-only.sh): the real tier would hash the real shards and start the real model three times, so a real-tier batch
# defers the recipe by name. Predicted wall [derived from gate-times.tsv: glm5next-twocard at the fixture tier is 49 s median over eight
# loads, qwen38-serve 89 s over ten-plus spawns]: four starts of about 8 s, three `--plan` runs of about 2 s, a warm build, the sha256 of the
# 8.6 GB set and the 1.2 GB projector in threads beside the first two starts: 53 to 98 s warm; (s7) to (s10) add about 25 s [derived: (s7)
# one start of about 8 s, the pick's `--plan` child of about 2 s and the stop, 11 s; (s8) three `--plan` runs and a dry run, 9 s; (s10) six
# refusals, the no-fit one with a pick's child, 6 s]; the bound is the runner's 900 s.
[group('solo')]
gate-gpu-serve-start-glm:
    BLOOMERY_MODEL=glm5next BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/fixture-only.sh gate-gpu-serve-start-glm && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_serve_start && D=target/serve-start-gate/glm && rm -rf $D && mkdir -p $D && bash tools/gpu-gate.sh gate_serve_start --family glm --dir $D'
    ./tools/box.sh 'bash tools/ref/fixture-only.sh gate-gpu-serve-start-glm && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --test serve_preflight -- --ignored --exact hw_a_device_list_alias_reaches_the_selected_seat'
    ./tools/box.sh 'bash tools/ref/fixture-only.sh gate-gpu-serve-start-glm && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --test serve_preflight -- --ignored --exact hw_preflight_normal_inputs_reach_the_fetch'
    ./tools/box.sh 'bash tools/ref/fixture-only.sh gate-gpu-serve-start-glm && bash tools/gate.sh --oxide -p bloomery-gpu-gates --release --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --test serve_preflight -- --ignored --exact hw_a_seat_word_alone_starts_by_name_and_refuses_offline'

# The same gate on the V4.1 fixture, clauses (s1) to (s10) (the file's architecture picks the seat; (s1) to (s3) read its plan from each
# start's own log, and (s8)'s `--plan --json` runs end through `start::plan_exit`). The projector is the `# mmproj` line of the V4.1 vision set
# ($BLOOMERY_DATA/ref-vision/deepseek41v, `just dump-ref-vision`), the file the V4.1 image gates load. The serving defaults are the seat's own
# (context 32768, two slots), planned under the fixture's card budget. A fourth start takes `--chat-template-file` (the fixture's own template
# behind a marker): its call-carrying conversation renders the marker and then exactly what the plain start rendered, so the seat's respell of
# V4's DSML spelling reached the file's text. (s7) to (s10) start by catalog name, read `--dry-run --json` and refuse as the exit table says
# (the GLM recipe above lists them).
# Predicted wall [derived, as above]: four starts of about 12 s over the 15 GB host set, the sha256 of the set and the projector in threads
# beside the first two starts, a warm build: 72 to 132 s warm; (s7) to (s10) add about 30 s [derived: the starts of about 12 s and the
# `--plan` runs of about 3 s over the same set, 14 s + 10 s + 6 s]; the bound is the runner's 900 s.
[group('solo')]
gate-gpu-serve-start-ds41:
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/fixture-only.sh gate-gpu-serve-start-ds41 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,vision,mimo2,solar2,minimaxm3,flint,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_serve_start && D=target/serve-start-gate/ds41 && rm -rf $D && mkdir -p $D && bash tools/gpu-gate.sh gate_serve_start --family ds41 --dir $D'

# The MiMo seat of `bloomery-serve` (gate_mimo2_serve): the refusals before any load (--plan's default context, the one
# slot, the tier card), one server at --ctx 4160 --parallel 1 holding its load and listening records, the three prompts'
# ids (the batch set's, a prose, the chat turn rendered by the server's template), the chat answer inside those ids, the
# edit and extension prefix clauses (the seat keeps any held position), and then the same file opened in process at the
# same context answering the server's ids bit for bit, the batch prompt's first id ik's or a named tie. Logs in
# target/mimo2-serve-gate/<server>/. Loads the whole host set twice (the server, then the session): alone in a batch, and
# under the big-load lock the V4.1 loads take. Real-only: there is no fixture file for it yet, so a fixture-tier run is
# refused by name (66) and the recipe is a weekly, selected by its trigger rows in tools/gate-paths.tsv and its own text.
# The bound is 28 min (the default 900 s is under two cold loads). Both cards in view (BLOOMERY_CARD=both: both gate
# locks, alone in a batch): the process runs the plan and refusal clauses, `--place bp`'s tier-card refusal included,
# then the rest in a child of itself that sees the A6000 alone, where the id clause's gate plan runs on one card; the
# gate's module doc has the census rules.
[group('solo')]
[group('v41-load')]
weekly-gpu-mimo2-serve:
    BLOOMERY_MODEL=mimo2 BLOOMERY_CARD=both ./tools/box.sh 'bash tools/ref/real-only.sh weekly-gpu-mimo2-serve && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,solar2,minimaxm3,flint,vision,gemma-embedding2,hf,checkpoint --release --bin bloomery-serve --bin gate_mimo2_serve && D=target/mimo2-serve-gate && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_BOUND=1680 bash tools/gpu-gate.sh gate_mimo2_serve --dir $D'

# glm5next decode CLI, functional run (no timing): generate_glm5next feeds --tokens one step per id, then greedy -n
# tokens. The gate placement on the 3090 unless --place a (and BLOOMERY_CARD=a6000). 3090, gate lock.
[group('v41-load')]
gen-glm5next *ARGS:
    BLOOMERY_MODEL=glm5next ./tools/box.sh 'export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features glm5next,clef,mimo2,vision,hf,glm5next-hf --release --bin generate_glm5next && bash tools/gpu-gate.sh generate_glm5next --place gate {{ARGS}}'

# qwen4exp(Qwen3.8-Flash-Next) 헤더 게이트: qwen35moe 리더의 변형 행이 합성 헤더에서 이름으로 거절하는 것들, 그다음 파일의
# 서술·역할별 텐서·커버리지 검사의 목록과 헤더에서 세운 계획(카드·호스트·캐시 바이트). 헤더만, 초 단위. 마지막 호출은
# 호스트 티어가 48층을 모두 서빙하는지와 네 층의 단계별 밴드다(네 층에서 expert 30개의 행을 읽는다, 초 단위). 샤드 넷이 다 있어야 한다.
# 그 사이에 MTP 초안 파일 둘(shared와 아닌 것)을 타깃 헤더에 대어 읽은 서술을 고정한다. 앞 호출이 빨개도 뒤 호출은 모두 돌고,
# 종료 코드는 어느 하나라도 0이 아니면 0이 아니다.
# The drafted plan's card bound on a synthetic file (qwen4exp_mtp_reserve, no file read) runs after the lib tests.
gate-qwen4exp-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::qwen35moe arch::tests::a_qwen4exp --nocapture; lib=$?; bash tools/gate.sh --release -p bloomery-model --test qwen4exp_mtp_reserve -- --nocapture; reserve=$?; bash tools/gate.sh --release -p bloomery-model --test qwen4exp_meta -- --ignored --nocapture; meta=$?; bash tools/gate.sh --release -p bloomery-model --test qwen4exp_mtp_meta -- --ignored --nocapture; mtp=$?; bash tools/gate.sh --release -p bloomery-model --test qwen4exp_host -- --ignored --nocapture; host=$?; bash tools/gate.sh --release -p bloomery-model --test qwen3_bp_plan -- --nocapture; bp=$?; echo "gate-qwen4exp-meta rc: lib $lib reserve $reserve meta $meta mtp $mtp host $host bp $bp"; [ "$lib" = 0 ] && [ "$reserve" = 0 ] && [ "$meta" = 0 ] && [ "$mtp" = 0 ] && [ "$host" = 0 ] && [ "$bp" = 0 ]'

# The chat route trace (D2's input, not a gate, lead-only, in the lead's A6000 window): one bloomery-serve-ds41
# process (one load, plan (a) on the A6000) with BLOOMERY_ROUTE_TRACE=OUT and the step feed runs every prompt row of
# PROMPTS in file order, greedy (/apply-template, /tokenize, /completion at temperature 0, n_predict 256,
# cache_prompt false), and the engine writes every position's routed ids per layer into OUT, a new directory, as a
# router set with the slot kinds and a call row per prompt call; the driver seals it once the server has exited
# (route_trace.py seal) and joins contexts.tsv (one row per request). OUT is an absolute path or target/…: a path
# elsewhere in the remote tree is refused by name, since the next sync (rsync --delete) removes it. tools/ref/route-trace-chat.py's header has the contract. Predicted wall about 13-15 min [derived]. It
# takes neither the A6000 gate lock nor the V4.1 load lock (the driver is not a target/release binary
# tools/gpu-gate.sh can run): run it inside a hold. The server runs under the driver's own bound (--bound, 1800 s),
# and every request under an HTTP timeout, so the driver ends when the server does.
route-trace-chat OUT PROMPTS='tools/ref/data/d2-prompts-ko.tsv,tools/ref/data/d2-prompts-en-code.tsv':
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=a6000 ./tools/box.sh 'bash tools/ref/real-only.sh route-trace-chat && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41,vision,gemma-embedding2,hf --release --bin bloomery-serve-ds41 && python3 tools/ref/route-trace-chat.py run --server target/release/bloomery-serve-ds41 --out {{OUT}} --prompts {{PROMPTS}}'

# The chat route trace of Qwen3.8 (D2's input for the second family, not a gate, lead-only, in the lead's A6000 window):
# route-trace-chat with one bloomery-serve-qwen38 process (one load, plan (a) on the A6000) instead of the V4.1 server.
# The server takes BLOOMERY_ROUTE_TRACE=OUT and BLOOMERY_PREFILL=steps from the driver (the trace records the step feed),
# serves one slot with the draft and the residency off under the trace, and the driver, its prompts, its join and its
# seal are route-trace-chat's (tools/ref/route-trace-chat.py's header has the contract; OUT is an absolute path or
# target/…). Wall: at most 48 × 256 generated and about 4,300 prompt ids, 16.6k positions, each a step. ~~About 5-6 min
# of requests at 18.9 ms a step (the plain step on plan (a) with the residency off, prose:512, docs/cards/q38res-ab.card)
# [derived]~~ missed: the three-prompt run took 800 positions in 55.7 s, 101, 77 and 30 ms a position per request, falling
# (a cold host set is the likely cause; majflt was not read), and the load 102 s from spawn to listening. The 48 prompts
# take about 8 min at the last request's 30 ms and 19 min at the run's 70 ms mean, plus the build [derived from that one
# run]. It takes neither the A6000 gate lock nor the V4.1 load lock (the driver is not a target/release binary
# tools/gpu-gate.sh can run): run it inside a hold. The server runs under the driver's own bound (--bound, 1800 s), and every request under an
# HTTP timeout, so the driver ends when the server does.
route-trace-chat-q38 OUT PROMPTS='tools/ref/data/d2-prompts-ko.tsv,tools/ref/data/d2-prompts-en-code.tsv':
    BLOOMERY_MODEL=qwen4exp BLOOMERY_CARD=a6000 ./tools/box.sh 'bash tools/ref/real-only.sh route-trace-chat-q38 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41,gemma-embedding2,hf --release --bin bloomery-serve-qwen38 && python3 tools/ref/route-trace-chat.py run --server target/release/bloomery-serve-qwen38 --out {{OUT}} --prompts {{PROMPTS}}'

# 서버 soak(M2, 리드 전용, 게이트 아님): bloomery-serve-ds41을 A6000에 배치 (a)로 띄우고, 시드를 고정한 요청 묶음을
# MINUTES분 보낸다. 30초마다 표본을 떠서 메모리 누수를 판정한다. 상한은 MINUTES분에 900초를 더한 값이다.
[group('solo')]
[group('v41-load')]
soak-ds41 MINUTES='30':
    BLOOMERY_MODEL=deepseek41 BLOOMERY_CARD=a6000 ./tools/box.sh 'bash tools/ref/real-only.sh soak-ds41 && export BLOOMERY_GATE_V41_LOAD=1 && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features deepseek41,vision,gemma-embedding2,hf --release --bin bloomery-serve-ds41 --bin soak_ds41_serve && D=target/soak && rm -rf $D && mkdir -p $D && BLOOMERY_GATE_BOUND=$(( {{MINUTES}} * 60 + 900 )) bash tools/gpu-gate.sh soak_ds41_serve --minutes {{MINUTES}} --dir $D'

# --- raon-ref (Raon-1-mini-preview oracle) -------------------------------------------------------------------------------------------

# Raon-1-mini-preview oracle: raon1_dump (the shared tools/ref/cb_dump.cpp) linked against KRAFTON's llama.cpp branch raon-1-mini-preview
# at fb895662d (CPU, no patch, /home/user/llama.cpp-raon, whose tracked files must be clean), which this does not rebuild: the tree is
# built once by tools/ref/raon1-tree.sh (a background job; its header has the command). tools/ref/raon1-dump-build.sh is the call; the
# binary and its build record land in $BLOOMERY_DATA/bin.
build-ref-raon1:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=60 BLOOMERY_MODEL=raon1 ./tools/box.sh 'bash tools/ref/raon1-dump-build.sh'

# The Raon-1-mini-preview oracle sets (refset family lcpp-raon1; tools/ref/raon1-dump.sh has the layout and the taps): the vendor branch
# on the CPU over the four prompts of tools/ref/data/raon1 from one load of the Q4_K_M file, into $BLOOMERY_DATA/ref_raon1_{a,b,c,d}: each
# layer's taps for a, each layer output and the router's expert ids for b and c, the last logits row for d, the top-40 logprobs and 32
# greedy tokens for a, b and c. Predicted wall 1-5 min, under the CPU lease and its card
# (BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/raon1-dump.card'); DUMP_DRY=1 in that variable prints the dump and stops before the
# lease. The file has no fixture yet: BLOOMERY_TIER=fixture is refused by name.
dump-ref-raon1:
    BLOOMERY_Q_CARD=none BLOOMERY_Q_PRED_S=240 BLOOMERY_MODEL=raon1 ./tools/box.sh 'bash tools/ref/raon1-dump.sh'

# --- end raon-ref ---

# --- raon-body (Raon-1-mini chain) ---------------------------------------------------------------------------------------------------

# Raon-1-mini end-to-end gate (gate_raon1_e2e's doc): the whole step on one card (RDA and gated latent layers, Raon_mHC's four streams,
# the dense and routed SiTU blocks), one prompt id a step, against the vendor branch's sets (refset family lcpp-raon1, `just
# dump-ref-raon1`): on each of the English, Korean and Rust sets the argmax after the prompt and the 32-token greedy continuation id for
# id; then, in the verifying load, the rows pass, verifies, commits and taps against the steps bit for bit, and with the DFlash2 draft
# beside the file the drafted greedy run is the plain one id for id. The file has no fixture yet, so the gate runs in the real tier
# only.
gate-gpu-raon1-e2e:
    BLOOMERY_MODEL=raon1 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-raon1-e2e && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_raon1_e2e && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_raon1_e2e'

# --- end raon-body ---

# --- dfl-wire (Raon-1's DFlash2 draft) -----------------------------------------------------------------------------------------------

# Raon-1's DFlash2 draft on one card (gate_raon1_dflash's doc): the draft file beside the Q4_K_M target loads; V4.1's DSpark draft (the
# deepseek41 profile's DSPARK_MODEL, read in a shell of its own so the raon1 profile stays the command's) is refused by name; and on set
# a the first block after the prompt, fed the vendor's own taps, proposes the host oracle's 7 ids. The target model is not loaded: the
# draft borrows its file's output and embedding rows. The file has no fixture yet, so the gate runs in the real tier only.
gate-gpu-raon1-dflash:
    BLOOMERY_MODEL=raon1 ./tools/box.sh 'bash tools/ref/real-only.sh gate-gpu-raon1-dflash && cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_raon1_dflash && __v=$(env -u BLOOMERY_REF_MODEL_PROFILE -u BLOOMERY_REF_MODEL -u BLOOMERY_DSPARK_MODEL BLOOMERY_MODEL=deepseek41 bash -c ". tools/ref/ref-paths.sh && dspark_model") && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_raon1_dflash --v41-draft "$__v"'

# --- end dfl-wire ---

# --- raon-attn (Raon-1-mini attention layers) ------------------------------------------------------------------------------------------

# Raon-1's RDA step (conv prep, write gate, KDA delta) and gated latent attention (`raon1::mla::attend`) on the card at the model's
# shapes against the HF model in f64 (gate_raon1_attn's doc), seeded inputs, no model file; each with an in-gate separation.
gate-gpu-raon1-attn:
    ./tools/box.sh 'cargo oxide build --arch {{arch}} -- -p bloomery-gpu-gates --features gpu --release --bin gate_raon1_attn && BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh gate_raon1_attn'

# --- end raon-attn ---

# --- raon-model ---
# raon1 (Raon-1-mini-preview) header gate: the reader's refusals on synthetic headers (a latent layer read as delta, a head
# count of 2, a missing or off-width latent row, the hyper-connection mix of another width, a tensor of the other kind), the
# common KDA, noaux and hybrid-plan owners it reads through, then the description of the Q4_K_M file: the delta and latent
# layer lines, the keys left unread, the tensors by role and type, the coverage list and the stores of the layers. The lib
# half is what runs the synthetic refusals; the test half is the file under /models. Headers only, seconds.
gate-raon1-meta:
    ./tools/box.sh 'bash tools/gate.sh --release -p bloomery-model --lib -- arch::raon1 arch::kda arch::noaux arch::classify arch::place_hybrid arch::solar2 arch::glm5next --nocapture && bash tools/gate.sh --release -p bloomery-model --test raon1_meta -- --ignored --nocapture'

# The raon vocabulary: our ids equal the reference tree's `llama-tokenize` on every case of crates/tokenizer/tests/cases.txt and
# cases-raon.txt, in both parse modes (the pins are tests/raon-ids.tsv, written by crates/tokenizer/tools/raon-oracle.sh from the
# krafton-ai llama.cpp tree whose pre-tokenizer `raon` and NFC normalizer exist there only), and the header's two end tokens end
# generation. The vocabulary only: seconds.
gate-raon1-tokenizer:
    ./tools/box.sh 'BLOOMERY_LLAMA_BPE_GGUF=$BLOOMERY_DATA/vocab-llama-bpe/ggml-vocab-llama-bpe.gguf bash tools/gate.sh --release -p bloomery-tokenizer --lib --test raon -- --include-ignored --nocapture'

# --- end raon-model ---
