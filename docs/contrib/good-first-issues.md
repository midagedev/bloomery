# Good first issues

Ten open rows taken from the open-work board (`docs/plan-triage.md`, mixed Korean and English) and **re-checked
against the code at `6b630e5e` on 2026-10-10**. The board is a notebook and many of its rows have since been closed, so
every row below cites the code as it is now, not the board's line numbers. Sizes are the board's (XS, S, M, L); where my
reading differs the row says so. "Verify" is a command you can run on your own machine.

Rows 1 to 7 need nothing but your machine. Rows 8 and 9 can be written on one, but landing them needs the maintainers'
box (`crates/gpu-gates`: the maintainers run the box gates and `just ptx-scan --no-jit`). Row 10 is a decision to raise
first. Take a row by opening an issue that names it. The rules in [`CONTRIBUTING.md`](../../CONTRIBUTING.md) apply: a
test that fails first, no relaxed gate, say what you ran.

Setup for the serve rows: the pinned nightly from `rust-toolchain.toml`; `cargo test --release -p bloomery-serve` needs
no card and no model (the pure crates, `python3 tools/recipes.py pure-crates`; `bloomery-serve` has a mock engine,
`crates/serve/src/mock.rs`).

## The serve API (mock engine, no model)

### 1. `n_predict` below -1 is silently clamped (XS)
- **What:** `n_predict: if n_predict < 0 { -1 } else { n_predict }` turns -2 into "no limit" without a word, against the
  rule that undefined input fails by name (`AGENTS.md`, Conventions). The board says llama-server answers 400; that is
  `[unverified]`, so read llama.cpp's `tools/server` and write the rule in your issue before the change.
- **Files:** `crates/serve/src/api.rs:1380-1384,1434`; a test beside the token-limit test at `api.rs:4205`.
- **Verify:** `cargo test --release -p bloomery-serve`.

### 2. `/tokenize` reads a missing or non-string `content` as "" (XS)
- **What:** `b.get("content").and_then(Value::as_str).unwrap_or("")` accepts a number or an object as the empty string,
  while its sibling fields go through `get_b`, which refuses another type by name.
- **Files:** `crates/serve/src/api.rs:1995`; model the test on the sampling-field test at `api.rs:3896`.
- **Verify:** `cargo test --release -p bloomery-serve`.

### 3. The sampling fields the server cannot honour are ignored, not refused (M on the board; refusing them is S)
- **What:** `typical_p`, `mirostat`, `dry_*`, `xtc_*`, `top_n_sigma` and `dynatemp_*` are never read from a request, so
  a client that sets them gets a plain sample with no warning. The refusal list has the pattern (`n_probs`, `grammar`,
  `logprobs`, `logit_bias` are refused by name); add rows that refuse a non-neutral value. The neutral values are the
  ones the server already reports (`api.rs:1522-1537`). Use llama-server's field names and its
  `tools/server/tests/unit/test_*.py` cases as the oracle (`docs/plan-triage.md`, "llama-server parity").
- **Files:** `crates/serve/src/api.rs:1337-1370` (`refused`), tests in the same file.
- **Verify:** `cargo test --release -p bloomery-serve`.

### 4. A test copy of `mock::Hold` (S)
- **What:** `crates/serve/tests/sampdraft.rs:390` defines `Hold` (enter, reached, open) and a `Held` engine wrapper; the
  crate's own `mock::Hold` (`crates/serve/src/mock.rs:638`) has the same fields and methods, and `DraftMock::holding`
  already blocks `prefill`. Delete the copy and use the mock's (about 60 lines). Read both before you start: the test's
  `reached(n)` uses a fixed `BOUND` where the mock's takes the bound as an argument.
- **Files:** `crates/serve/tests/sampdraft.rs:390-440`, `crates/serve/src/mock.rs:631-690`.
- **Verify:** `cargo test --release -p bloomery-serve --test sampdraft` passes before and after.

### 5. cargo-fuzz targets for the parsers that read untrusted text (S)
- **What:** the tree has no fuzz target (no `cargo-fuzz` or `libfuzzer` in any manifest). Start with the public surface:
  `jinja::Template::parse` and `render` (`crates/jinja/src/lib.rs:77,105`), and the streaming tool-call scanners
  `push` and `finish` (`crates/serve/src/glmxml.rs:178,210`, `qwenxml.rs:279,330`, and `dsml`, `hermes`). The HTTP
  readers `read_request` (`http.rs:114`) and `read_chunked` (`http.rs:187`) are private, so say in the issue whether
  you propose a feature-gated re-export. Put the crate under `fuzz/` and add it to the workspace `exclude` list
  (`Cargo.toml:5`, as the compiler-bug reproducer is).
- **Files:** a new `fuzz/` crate, `Cargo.toml`.
- **Verify:** `cargo +nightly fuzz run <target> -- -max_total_time=60` on Linux or macOS; report any crash with its
  input. A target is not done until you have shown it finds a planted defect.

## Tools (Python and shell, any machine)

### 6. `lcpp-warm.sh --self-test` leaves its Python stub server running (S)
- **What:** the board records 21 stub servers (`lcpp-warm-self-test.*/llama-server`) alive on a Mac after the self-test,
  the oldest a day and a half old (an observation from 2026-10-09; I did not reproduce it). The script has no `trap`
  (`grep -n trap tools/ref/lcpp-warm.sh` prints nothing). Kill the stub on exit by its captured pid (never by pattern)
  and add a no-leftover check to the self-test (`tools/ref/lcpp-warm.sh:724-938`).
- **Files:** `tools/ref/lcpp-warm.sh`.
- **Verify:** `bash tools/ref/lcpp-warm.sh --self-test`, then list the stub processes; make it fail first by removing
  your trap.

### 7. `recipes.py` does not see a one-line attributed `mod x;` (S)
- **What:** the module-tree reader's `_MOD` pattern (`tools/recipes.py:876`) matches `mod x;` only at the start of a line,
  and `#[path = "..."]` only on a line of its own (`:953`). A `#[cfg(test)] mod x;` or `#[path = "x.rs"] mod x;` on
  one line is invisible to the gate-selection graph. No such line exists in the tree today (I grepped `crates/`), so
  this is a latent gap: fix the pattern and add a self-test case on a synthetic tree.
- **Files:** `tools/recipes.py` (the self-test is in the same file).
- **Verify:** `python3 tools/recipes.py --self-test`; `bash tools/check-recipes.sh`.

## Code in `crates/gpu-gates` and the workspace (type-checked on a Mac, judged on the box)

For rows 8 and 9 off-box verification is the static tier only: `just mac-static` on a Mac, which type-checks the Linux
build (`docs/BUILD.md:299`). Whether a Linux contributor can type-check device crates without `cargo oxide` is
`[unverified]`. The maintainers run the gates and `ptx-scan` for the change.

### 8. The device-name token `.replace(' ', "_")` is copied seven times (XS)
- **What:** one rule (a card name made file-safe) is spelled at seven sites. Give it one helper in the gates lib.
- **Files:** `crates/gpu-gates/src/generate.rs:590`, `bin/gate_ds41_serve.rs:1823,1824`, `bin/gate_glm5next_twocard.rs:1462`,
  `bin/gate_deepseek41_rope.rs:377`, `bin/oxart_jit.rs:63`, `bin/shared/serve_seats/ds41.rs:1044`.
- **Verify:** `just mac-static`; the strings the callers print are unchanged.

### 9. Dead stub `main` functions behind `required-features` (S-M, mechanical)
- **What:** 99 bins in `crates/gpu-gates/src/bin/` carry `#[cfg(not(feature = "..."))] fn main()` that exits 2 (my count;
  the board said 58). A bin whose `[[bin]]` declares `required-features` can never build without the feature, so its stub
  is dead (`gate_q4k_sel` does: `crates/gpu-gates/Cargo.toml:433-435`, stub `gate_q4k_sel.rs:46`). Check each bin's
  `[[bin]]` before you delete its stub, and leave the ones that lack the declaration alone.
- **Files:** `crates/gpu-gates/src/bin/*.rs`, `crates/gpu-gates/Cargo.toml`.
- **Verify:** `just mac-static` and `bash tools/check-recipes.sh` (it refuses a recipe that builds a
  bin without its `required-features`, so no recipe can reach a stub). A deletion's proof (`ptx-scan` equal to the base) is the maintainers' run.

## A decision first

### 10. `overflow-checks` for the packages that parse untrusted input (XS)
- **What:** the root `Cargo.toml` has no `[profile.*]` table. The board proposes `overflow-checks = true` per package
  for `bloomery-serve`, `-jinja`, `-hf`, not profile-wide, because overflow checks in device MIR break device lowering.
  Raise it in an issue first: no gate guards speed, so the maintainers pick the packages, and the tokenizer is on the
  prompt path.
- **Files:** `Cargo.toml` (`[profile.release.package.<crate>]`).
- **Verify:** `cargo test --release -p bloomery-serve -p bloomery-jinja -p bloomery-hf` passes with the setting on.

## Rows I checked and left out

Rows that were closed when I read the code are not here. If you find a board row that looks stale, say so in an issue;
the board's owner (the lead) removes landed rows.
