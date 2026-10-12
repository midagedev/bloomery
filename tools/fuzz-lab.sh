#!/usr/bin/env bash
# The parser fuzz crate (fuzz/, excluded from the workspace), on the box, from the directory box.sh entered:
# its own tests, then each fuzz target for 60 s on its seeds, as fuzz/README.md runs them. `just lab-fuzz`.
#   ./tools/box.sh 'bash tools/fuzz-lab.sh'
#
# Needs cargo-fuzz on the box (`cargo install cargo-fuzz --locked`), the pinned nightly and a C++ compiler.
# The tests go through tools/gate.sh, which owns their bound and exit code. The call is spelled here, not in
# the justfile: tools/recipes.py reads a recipe's `cargo test` call with no `-p` as the whole workspace's, and
# `--manifest-path fuzz/Cargo.toml` names another one.
#
# Exit code: the first failing step's own (gate.sh's, a fuzz finding's, 124 or 137 a bound).
set -euo pipefail
cd "${BASH_SOURCE[0]%/*}/.."
bash tools/gate.sh --manifest-path fuzz/Cargo.toml
timeout --kill-after=10 1500 cargo fuzz build
for target in jinja_parse jinja_render glmxml qwenxml dsml hermes; do
  timeout --kill-after=10 180 cargo fuzz run "$target" "fuzz/corpus/$target" -- \
    -max_total_time=60 -max_len=4096 -dict=fuzz/parsers.dict
done
