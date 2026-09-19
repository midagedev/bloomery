# What a high-quality, agent-friendly Rust repo looks like in 2026

Research survey for `mulle` — read before writing `AGENTS.md` and setting up tooling.
Scope: a Rust + CUDA-Rust (NVlabs/cuda-oxide) LLM inference engine; hand-written AVX2
intrinsics on the host, `#[kernel]`/`#[cuda_module]` device code, ggml as numeric oracle,
benchmarks that need a quiet machine.

**Spec correction (verified against the tree): the task brief says "four standalone binary
crates". There are three** (`crates/oxide-ice-unroll`, `crates/q3k-cpu`, `crates/q3k-gemv`),
each with its own `[workspace]` stanza and `rust-toolchain.toml` (all pin
`nightly-2026-08-28`), no root workspace, no tests, no lint config, no CI.
Recommendations below are written for the three crates that exist.

Ground rules followed: every non-obvious claim carries a URL; claims I could not source
are marked `[unsourced]`. Primary sources preferred; secondary sources are labeled as such.
Local verifications were run on the Mac (stable `cargo 1.96.1` / `rustc 1.96.1` /
`clippy 0.1.96`) — the box's pinned nightly was not touched.

## 1. Workspace hygiene

### What a single workspace buys

| Feature | What it does | Primary source |
|---|---|---|
| `cargo test --workspace` / `--workspace --all-targets` | one command runs every member's tests | [Cargo book, workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html) |
| Shared lockfile | one `Cargo.lock` at the root; today each standalone crate resolves independently, so `q3k-gemv` and `oxide-ice-unroll` can silently build against different `cuda-oxide` git revisions | same |
| `[workspace.dependencies]` + `dep.workspace = true` | one place to pin versions (notably the `cuda-oxide` git rev and `cuda-core` version) | [Cargo book, specification of dependencies](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html) |
| `[workspace.lints]` + `[lints] workspace = true` | lint contract in one place; members inherit (see §2). Cargo even warns when a package *doesn't* inherit while workspace lints exist (`missing_lints_inheritance`, warn-by-default) | [Cargo book, lints](http://doc.rust-lang.org/nightly/cargo/reference/lints.html) |
| Shared `target/` + resolver v2 | faster incremental builds, one feature-unification regime | [Cargo book, resolver](https://doc.rust-lang.org/cargo/reference/resolver.html) |

### What breaks, and the documented pattern

Two separate concerns, often conflated:

1. **Different toolchain per member.** rustup resolves the toolchain by walking *up* from
   the current directory and taking the closest `rust-toolchain.toml`
   ([rustup book, overrides](https://rust-lang.github.io/rustup/overrides.html)).
   So nested toolchain files technically take effect per-cwd — but any workspace-wide
   command run from the root (`cargo test --workspace`) uses the *root* toolchain for
   *all* members. A member cannot pin its own toolchain for workspace-wide runs.
   The documented escape hatch is `exclude` in the root `[workspace]` table, leaving the
   odd member as a standalone crate with its own `rust-toolchain.toml`
   ([Cargo book, workspaces](https://doc.rust-lang.org/cargo/reference/workspaces.html)).
   This repo does not need it today: all three crates already pin the same
   `nightly-2026-08-28`.
2. **Third-party cargo subcommand (`cargo oxide build --arch sm_86`).** There *is* a
   documented pattern for exactly this: cargo-oxide ships a **passthrough mode for normal
   Cargo workspaces** — `cargo oxide build -- -p my_app`, with `--device-codegen-crate`
   owner filters (hyphens normalized to underscores) so the CUDA backend emits device
   artifacts only for the listed crates while host code for every crate still goes
   through LLVM, plus `cargo oxide test -- -p my_app` for tests
   ([cargo-oxide README](https://raw.githubusercontent.com/NVlabs/cuda-oxide/main/crates/cargo-oxide/README.md)).
   What breaks is running *plain* `cargo build/test --workspace`: device crates using
   `#[kernel]`/`#[cuda_module]` need the codegen backend, so plain cargo is not a valid
   way to build them. The rule for `AGENTS.md` is therefore: **device crates are built
   only via `cargo oxide`, never via bare `cargo`** — same class of rule as the repo's
   existing "builds run on the box via `tools/box.sh`".

### Recommended `Cargo.toml` shape (verified below)

Root `Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = ["crates/*"]

[workspace.dependencies]
cuda-device = { git = "https://github.com/NVlabs/cuda-oxide.git", rev = "<pin>" }
cuda-host = { git = "https://github.com/NVlabs/cuda-oxide.git", rev = "<pin>" }
cuda-core = "0.3.1"
libc = "0.2"

[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "deny"

[workspace.lints.clippy]
undocumented_unsafe_blocks = "deny"
missing_safety_doc = "deny"
```

Each member (`crates/q3k-cpu/Cargo.toml`, etc.):

```toml
[package]
name = "q3k-cpu"
version = "0.1.0"
edition = "2024"

[lints]
workspace = true

[dependencies]
libc.workspace = true
```

Notes: pin the `cuda-oxide` git dependency with `rev =` (today it floats — see §6,
`cargo deny`); keep one root `rust-toolchain.toml`, delete the three per-crate copies
(they are byte-identical today). If a future member ever needs a different toolchain or
must stay invisible to `cargo oxide`, `exclude` it rather than breaking the workspace.

Verification (this machine, stable cargo 1.96.1) — the stanza shape above is accepted:

```
$ cargo metadata --format-version 1 --no-deps 2>/dev/null | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print('workspace metadata OK:', ...)"
workspace metadata OK: ['a']
```

(full throwaway tree under `/tmp/lintcheck`: root `[workspace.lints.rust/clippy]` +
member `[lints] workspace = true`).

## 2. Lints as a contract

Exact syntax is as in §1. `[workspace.lints]` tables are named `[workspace.lints.rust]`,
`[workspace.lints.clippy]`, `[workspace.lints.cargo]`, `[workspace.lints.rustdoc]`; members
opt in with `[lints] workspace = true` (a member-local `[lints.x]` table *replaces*,
not merges — a crate that writes its own `[lints.clippy]` silently stops inheriting;
secondary source, field report:
[cairn#120](https://github.com/kage1020/cairn/issues/120)).
Stabilized in Cargo 1.74 (2023); the `missing_lints_inheritance` cargo lint that
catches non-inheriting members is warn-by-default
([Cargo book](http://doc.rust-lang.org/nightly/cargo/reference/lints.html)).
`[unsourced]`: the 1.74 stabilization version is from memory — recheck before citing
upstream.

### Status of the specific lints (all verified on clippy 0.1.96 unless noted)

| Lint | Level / group | Since | Note |
|---|---|---|---|
| `unsafe_op_in_unsafe_fn` (rustc) | warn-by-default **in edition 2024** (part of `rust-2024-compatibility`); deny it | [Edition guide](https://doc.rust-lang.org/edition-guide/rust-2024/unsafe-op-in-unsafe-fn.html) | this repo is already edition 2024, so `deny` is pure win; `cargo fix --edition` migrates |
| `clippy::undocumented_unsafe_blocks` | **allow by default — verified locally** (no warning on bare `unsafe{}` without opt-in) | config keys below | must be explicitly denied; this is the `// SAFETY:` enforcer |
| `clippy::missing_safety_doc` | clippy `style`, **warn by default — verified locally** (`warning: unsafe function's docs are missing a # Safety section`) | added 1.39.0, [clippy book](https://rust-lang.github.io/rust-clippy/master/index.html#missing_safety_doc) | deny it |
| `clippy::multiple_unsafe_ops_per_block` | `restriction`, allow, added 1.69.0 | [clippy book](https://rust-lang.github.io/rust-clippy/master/index.html#multiple_unsafe_ops_per_block) | pairs with the above: one op per block, each justified |
| `clippy::not_unsafe_ptr_arg_deref` | **deny by default — verified locally** | observed on clippy 0.1.96 | free; safe fns taking raw pointers get flagged |
| `clippy::missing_errors_doc` / `missing_panics_doc` (pedantic) | allow | — | consider; panics-doc matters for index helpers |

`undocumented_unsafe_blocks` semantics, from `cargo clippy --explain` on this machine:
it requires a `// SAFETY:` comment on the line(s) immediately preceding the `unsafe`
block ("with nothing appearing in between"); tunable via `accept-comment-above-statement`
and `accept-comment-above-attributes` in `clippy.toml`. So the `// SAFETY:` convention
**is lint-enforced** once the lint is denied — the answer to the brief's parenthetical
is yes, since the lint exists (exact stabilization version
`[unsourced]`, pre-1.70s by the changelog references
([1](https://github.com/rust-lang/rust-clippy/issues/15755))).

### Which groups to deny in a numerics/`unsafe`-heavy crate

Recommend `deny` on: `unsafe_op_in_unsafe_fn`, `undocumented_unsafe_blocks`,
`missing_safety_doc`, plus rustc `missing_docs` at least `warn`.
Do **not** blanket-deny `clippy::pedantic`/`nursery`. The cast lints specifically —
`cast_possible_truncation`, `cast_sign_loss`, `cast_possible_wrap`, `cast_precision_loss`
— fire on the `as` conversions that SIMD decode code is made of (every `u8 as i8 as i32`
in `q3k-cpu`'s field loop). Community practice is to keep them at `warn` with narrow
`#[allow]`s on a named conversion helper, or to `deny(clippy::checked_conversions)` and
route intentional truncations through one helper (secondary source, consulting writeup:
[corrode.dev](https://corrode.dev/blog/pitfalls-of-safe-rust/)).
Known-noisy-in-SIMD list to keep at `warn` (judgement, informed by the lint docs):
`cast_possible_truncation`, `cast_sign_loss`, `cast_precision_loss`,
`similar_names` (lanes named `q3l`/`q3h`/`sc0..sc3` *should* look alike),
`too_many_lines` / `too_many_arguments` (kernel launch shapes),
`missing_panics_doc` at warn only. Other surfaces in this repo: none yet (binaries only,
no lib/TUI/server split) — revisit when the loader becomes a library.

## 3. `unsafe` discipline

What serious performance crates do, with citable instances of each practice:

1. **Deny `unsafe_op_in_unsafe_fn` + `undocumented_unsafe_blocks`, one op per block.**
   Status and mechanics are in §2. The `// SAFETY:` comment must name the *specific
   precondition* (allocation, bounds, alignment, validity), not restate the code.
   `multiple_unsafe_ops_per_block` (restriction, allow-by-default) is the "one op per
   block" half. This repo's `dot_row`/`field_dot` already carry `# Safety` doc sections
   and per-site `// SAFETY:` comments — the lints turn that existing habit into a
   compile-time contract.
2. **`# Safety` docs on every `pub unsafe fn` via `missing_safety_doc = deny`.**
   Warn-by-default since clippy 1.39.0, verified on this machine (§2). Note it is a
   *clippy* lint, not rustc: plain `rustc` stays silent on undocumented `pub unsafe fn`
   (verified: `--crate-type=lib` build, no warning), so CI must run clippy or the
   contract is void.
3. **The `checked`/`debug` swap for `get_unchecked`.** The published-crate instance is
   TiKV's `unchecked-index` crate: `pub unsafe fn get_unchecked` checks
   `v.assert_indexable_with(&index)` under `#[cfg(debug_assertions)]` and calls the raw
   `get_unchecked` in release
   ([source](https://tikv.github.io/doc/src/unchecked_index/lib.rs.html)).
   std itself does the same pairing (`debug_assert!` + `get_unchecked` + `// SAFETY:`,
   e.g. [core/hash/sip.rs](https://doc.rust-lang.org/1.74.0/src/core/hash/sip.rs.html)).
   Recommended shape for this repo's hot indexing:

   ```rust
   #[inline(always)]
   unsafe fn idx(buf: &[f32], i: usize) -> f32 {
       // SAFETY: caller guarantees i < buf.len() (CCD-partitioned ranges).
       #[cfg(any(debug_assertions, feature = "checked-index"))]
       { assert!(i < buf.len()); }
       unsafe { *buf.get_unchecked(i) }
   }
   ```

   with `checked-index = []` in `[features]` so `cargo oxide test --features checked-index`
   (and miri, §6) exercise the bounds while release stays unchecked. Tests then prove
   the property "no `get_unchecked` is reachable out of bounds" instead of hoping it.
4. **Contested point — `debug_assert!` is not soundness.** A 2026 fix in
   `zenjxl-decoder` explicitly *replaced* `debug_assert!`-only bounds checks with real
   `assert!` ("provably sound" commit, [imazen/zenjxl-decode](https://github.com/imazen/zenjxl-decoder/commit/7a4e506c533f441466ad566fa1e7a5e18060482c)):
   if the bound cannot be derived from construction (as opposed to merely *expected*
   from partitioning), the check must survive into release. Rule of thumb for the 73
   elided checks: partition-derived bounds get the `checked-index` treatment; anything
   derived from file input (GGUF offsets, block counts) gets an unconditional `assert!`
   or, better, a newtype (§5) that makes the invalid value unconstructible.

## 4. Testing numeric kernels against an oracle

Contract here: "matches ggml within a relative-error band" (1e-4 f32 / 1e-2 q8_1).

| Concern | Current practice (2026) | Source |
|---|---|---|
| Property-based testing | **`proptest` is the default** (1.11.0, 2026-03-24, shrinking + composable strategies). `quickcheck` was **revived to 1.0+ / 1.1.0 on 2026-02-10** (repo pushed 2026-04-03, not archived) — lighter and QuickCheck-faithful, but weaker shrinking. Versions from [crates.io](https://crates.io/) API, fetched 2026-09-19. Use proptest for quantized-block arbitraries; the choice is contested only at the margins, and every recent Rust testing guide standardizes on proptest (secondary: [adocpdf design](https://github.com/joaoleal/adocpdf/blob/HEAD/openspec/changes/archive/2026-08-16-behavioural-testing/design.md), [adv-testing skill](https://github.com/kiefbc/dotfiles/blob/HEAD/claude/.claude/skills/advanced-testing/references/rust.md)) |
| Float comparison | **`approx` (0.6.0-rc2, 2026-02-05, actively developed) over `float-cmp` (0.10.0, 2024-09-20, quiet)** — versions from crates.io API. But: the repo contract is a *relative-error band*, and `approx::relative_eq!` expresses exactly that; for the ggml-oracle harness a 5-line hand-rolled `max_rel_err` (which `q3k-gemv` already computes inline) is clearer than either crate and has no tolerance semantics to misread. Use `approx` in unit tests, keep the hand-rolled band in the oracle harness |
| Golden/snapshot (PTX, dequantized row) | **`insta` 1.48.0** (crates.io). Add as dev-dependency with a serializer feature and assert on the string form: `cargo add --dev insta --features yaml` ([quickstart](https://insta.rs/docs/quickstart/)). PTX via `assert_snapshot!(ptx_string)`; a dequantized row via `assert_yaml_snapshot!`/`assert_debug_snapshot!`. Review workflow is `cargo insta test` / `cargo insta review`; compile `insta`+`similar` with `opt-level = 3` under `[profile.dev.package]` per the same page. Exact CI gate flags `[unsourced]` — check `cargo insta --help` on the box before writing the CI job |
| Same test vs scalar + SIMD paths | gate the scalar reference behind a feature or `cfg`, run the identical oracle assertion against both in one test binary (e.g. `#[cfg(feature = "checked-index")]` scalar path vs AVX2 path vs ggml values). No crate needed — this is a test-organization pattern, and inventing a harness crate for it would be over-engineering |
| Keeping GPU tests out of the fast loop | three composable mechanisms: (a) `#[ignore]` on GPU/hardware tests (`cargo test` semantics, unchanged); (b) nextest **`default-filter`** so the default profile skips them and CI opts in — exact snippet from [nextest selecting docs](https://github.com/nextest-rs/nextest/blob/HEAD/site/src/docs/selecting.md): `default-filter = 'not package(special-tests)'` under `[profile.default]`, `default-filter = 'all()'` under `[profile.ci]` in `.config/nextest.toml`, override with `--ignore-default-filter`; (c) `cargo oxide test -- -p <gpu-crate>` passthrough so device tests run through the backend ([cargo-oxide README](https://raw.githubusercontent.com/NVlabs/cuda-oxide/main/crates/cargo-oxide/README.md)). Also relevant: per-test `slow-timeout` + `threads-required` (heavy-test mutual exclusion — the machine-lock problem in §6) at [nextest config](https://nexte.st/docs/configuration/) |

## 5. Making illegal states unrepresentable

Idiom in 2026: **plain newtype + `derive_more` for boilerplate; `nutype` only when a
*validation invariant* must hold.** `derive_more` 2.1.1 / `nutype` 0.8.0-beta.2
(crates.io, 2026-09-19). `nutype`'s value is `validate(...)` + sanitizers on
construction ([nutype repo](https://github.com/robertream/nutype),
[docs](https://docs.rs/crate/nutype_macros/latest)); row indices, block counts, and byte
offsets need no validation — they need *distinction* (`Row != Col != ByteOff`), which is
`struct RowIdx(pub usize)` (or private field + `From`/`TryFrom`) with `derive_more`
for `From/Into/Display/AsRef` so the newtype path stays frictionless
(secondary: [rust-optimize skill](https://github.com/dekobon/git-remote-object-store/blob/HEAD/.claude/skills/rust-optimize/SKILL.md),
[state-space-minimization](https://github.com/dkubb/skills/blob/HEAD/skills/state-space-minimization/references/languages/rust.md)).
`nutype` here would add a proc-macro dependency for zero benefit — say no.

Zero-cost claim, proven rather than asserted. On this machine
(`rustc 1.96.1 -O`, aarch64):

```
$ rustc --edition 2021 -O --emit=asm --crate-type=lib t.rs -o t.s
$ grep -n "^_raw_sum\|^_newtype_sum\|^__" t.s
__ZN1t11newtype_sum17h071168f3ccefebc7E:        # one body emitted
__ZN1t7raw_sum17h8093ec3e53e37028E = __ZN1t11newtype_sum17h071168f3ccefebc7E
```

`raw_sum(buf: &[f32], idx: &[usize])` vs `newtype_sum(buf: &[f32], idx: &[RowIdx])`
(looping `get_unchecked`) produced **one** machine-code body with `raw_sum` aliased to
`newtype_sum` — stronger than "identical loops": the newtype abstraction compiled away
entirely, including inside `get_unchecked`. (Throwaway source `/tmp/newtype/t.rs`;
caveat: aarch64 host, not the box's Zen 3 — re-run on the box is one command and worth
doing once for the record, since the hot path there is AVX2.)

## 6. The tool belt

Versions from the crates.io API, 2026-09-19. Verdicts are for *this* repo specifically.

| Tool | What it does | Applies here? | Minimal adoption |
|---|---|---|---|
| `cargo nextest` 0.9.145 | test runner: per-test timeouts, retries, filterset DSL, sharding; much faster/louder CI than `cargo test` | **Yes.** The `.config/nextest.toml` `default-filter` + `slow-timeout` + `threads-required` from §4 is the GPU-test story. Caveat (secondary source, community skills): nextest does not run doctests — CI still needs `cargo test --doc` ([example](https://github.com/laurigates/cleanscope/blob/HEAD/.claude/skills/cargo-nextest/SKILL.md)) | `cargo install cargo-nextest --locked`, add `.config/nextest.toml` as in §4 |
| `cargo deny` 0.20.2 | dependency auditing: advisories (RustSec), licenses, bans (duplicate versions, wildcards), sources | **Yes — more than usual.** Two members depend on `cuda-oxide` via floating git URLs today; deny's `[sources]` + `[bans]` + `advisories` is how that stops drifting. `cargo deny init && cargo deny check` ([README](https://raw.githubusercontent.com/EmbarkStudios/cargo-deny/main/README.md), [book](https://embarkstudios.github.io/cargo-deny/)). Minimal `deny.toml`: `[advisories] vulnerability = "deny"`, `[licenses] allow = ["MIT", "Apache-2.0"]`, `[bans] multiple-versions = "warn"`, `[sources] unknown-git = "warn"` (field shape conventional — secondary: [vykar skill](https://github.com/borgbase/vykar/blob/HEAD/.agents/skills/architecture-review/SKILL.md)) | `deny.toml` at root + CI job |
| `cargo miri` | UB detector (OOB, uninit, aliasing, leaks) by interpreting tests | **Partially — host scalar code only.** Miri supports "very few AVX512 intrinsics at the moment" ([README](https://raw.githubusercontent.com/rust-lang/miri/master/README.md)) — AVX2 intrinsics coverage is likewise incomplete, and it fundamentally cannot see `cuda-oxide` device code (interpreter, no FFI/platform shims; GPU kernels need the codegen backend). Use it on the scalar reference path + `checked-index` feature, with SIMD behind `cfg` gates, not on `dot_row` itself. `cargo miri test -p <scalar-tests>`; Miri documents a [nextest integration](https://raw.githubusercontent.com/rust-lang/miri/master/README.md) |
| `criterion` 0.8.2 vs `divan` 0.1.21 | benchmark harnesses | **criterion, bluntly.** Criterion is actively maintained again and gives statistical rigor + HTML reports for the ggml-comparison contract; divan compiles faster with a nicer API and is what large workspaces (uutils: 22 packages) consolidate on ([uutils commit](https://github.com/uutils/coreutils/commit/91930760ae285a26acf86bf33cabd74b1f9501af)), but rigor matters more here than iteration speed. Either way: **neither tool provides the machine-wide quiet lock** this repo's numbers depend on (`docs/quiet-machine.md` protocol + witness rows stay mandatory); wrap the harness invocation in the lock script, and never benchmark with `#[test]` (secondary: [rust-testing skill](https://github.com/fernandezbaptiste/agentkit/blob/HEAD/skills/rust-testing/SKILL.md)) |
| `cargo machete` 0.9.2 / `udeps` | unused-dependency detection | **machete, not udeps.** machete runs on stable Rust; udeps needs nightly `-Z` flags (field evidence: [cdf batch notes](https://github.com/z3z1ma/cdf/blob/HEAD/.10x/evidence/2026-07-09-p2-e2-g1-b4-batch.md)). This repo pins nightly anyway, but machete has fewer moving parts and a per-crate `ignored` escape hatch (`[package.metadata.cargo-machete]`, secondary: [rust-quality-gates skill](https://github.com/caliluke/skills/blob/HEAD/skills/rust-quality-gates/SKILL.md)). `cargo machete` in CI; low value before the workspace exists, real value after |
| `cargo xtask` vs `just` 1.58.0 | task runners | **`just` only; no xtask.** Current synthesis across the ecosystem: most tasks belong in `just`; `xtask` only when a task needs Rust code (codegen, complex release flows) — canonical shape a single `publish = false` member (secondary: [toolchain-and-workspace](https://github.com/zzci/skills/blob/HEAD/skills/pma-rust/references/toolchain-and-workspace.md)); live hybrids keep `just` as the canonical layer over a slim xtask ([lodestone](https://github.com/matteopolak/lodestone/commit/d1060c42dce48091a9db9d0b8f79f274b5f4da83)). This repo's automation is shell-shaped (`tools/box.sh` rsync, quiet-machine witnesses) — a `justfile` fronting those scripts is the whole need; an xtask crate would be a second, untested task runner ([cautionary case](https://github.com/praxiomlabs/rust-mssql-driver/commit/72e2ff996f3125ee8db6f2db4f11c0d23fd5fa22)) |

Not worth adopting: `udeps` (above); a workspace-wide coverage gate (llvm-cov on benchmark-shaped
binaries measures the harness, not the kernels — `[unsourced]` judgement); anything that
parses PTX as text in CI beyond an `insta` snapshot (brittle across toolkit upgrades).

## 7. `AGENTS.md` as an actual artifact

The format: open spec, **no mandated sections** — "AGENTS.md is just standard Markdown.
Use any headings you like" ([agents.md](https://agents.md/)). Stewarded since 2025 by the
Agentic AI Foundation under the Linux Foundation, adopted by 60k+ projects
(multiple secondary summaries agree, e.g.
[prompt-shelf](https://github.com/vivo-lab-inc/the-prompt-shelf/blob/HEAD/src/content/blog/agents-md-codex-setup-guide-2026.md),
[authoring skill](https://github.com/agentparadise/agentic-primitives/blob/HEAD/./plugins/meta/skills/authoring-agents-md/SKILL.md)).
It complements README (humans) rather than replacing it.

Three real `AGENTS.md` files in non-trivial Rust repos (fetched raw 2026-09-19):

**1. nushell/nushell — 21 lines** ([raw](https://raw.githubusercontent.com/nushell/nushell/main/AGENTS.md)).
Sections: "Testing and code style rules", "Issue and PR Guidelines". Mostly prohibitions
+ pointers (`See rust_style.md, FAQ.md, HOWTOS.md`). Commands: none spelled out, but
workspace/semver rules inline. Excerpt:

> - Never use `.unwrap()` except in tests - always handle errors with `ShellError` or `ParseError`
> - No panicking on user input, no nightly features, no GPL deps (MIT, Apache License 2.0, CC0 only)
> - Dependencies: use workspace dependencies, exact semver `"1.2.3"`, no git dependenciess in PRs

**2. uutils/coreutils — 44 lines** ([raw](https://raw.githubusercontent.com/uutils/coreutils/main/AGENTS.md)).
Sections numbered: 1. Never read or copy GNU code (license trap), 2. A PR needs tests,
3. Keep the PR description short, 4. Read the docs already in this repo. Explicitly a
*pointer*, not a duplicate: "they are the actual rules, this file is only a pointer".
No build commands (those live in `DEVELOPMENT.md`). Excerpt:

> Writing this file is not a statement for or against using AI coding agents. People are
> using them on this repository either way, so this file exists to make sure that when
> they do, the rules of the project are actually followed. … Answers to the reviewers
> should be done by a human, not a agent.

**3. sxyazi/yazi — 53 lines** ([raw](https://raw.githubusercontent.com/sxyazi/yazi/main/AGENTS.md)).
Sections: Rules, Project, Style, Code Changes, Validation. Style-heavy (naming, module
conventions, Lua boundary rules) with a `Validation` block of exact commands:

```sh
cargo check -p <package>
cargo test -p <package>
cargo clippy -p <package>
rustfmt +nightly **/*.rs
```

plus "Use `cargo check` instead of `cargo build` unless artifacts are needed."

Evidence synthesis for a repo like this one: **short (20–55 lines), prohibitions first,
commands included or pointed at, never a README duplicate.** The highest-value lines in
all three are the ones that prevent *irreversible or project-specific* mistakes
(GPL contamination, unwrap policy, nightly policy) — for mulle the analogues are: build
only via `cargo oxide` on the box through `tools/box.sh`, never plain `cargo build` for
device crates; numbers measured-only with witness rows; Korean prose / English code
comments (CLAUDE.md convention).

`AGENTS.md` vs `CLAUDE.md`: Claude Code reads `CLAUDE.md`, not `AGENTS.md` — explicit in
[Anthropic's memory docs](https://code.claude.com/docs/en/memory) per contemporaneous
writeups ([gist, checked 2026-06-03](https://gist.github.com/yurukusa/d36197848911f025add142abefcde685),
[prompt-shelf](https://github.com/vivo-lab-inc/the-prompt-shelf/blob/HEAD/src/content/blog/does-claude-code-support-agents-md-2026.md));
the most-upvoted feature request (anthropics/claude-code#6235, filed 2025-08-21) asked
for `AGENTS.md` support, and late-2025/2026 press reported a fallback (read `AGENTS.md`
only when no `CLAUDE.md` exists,
[runtimewire](https://runtimewire.com/article/claude-code-adds-agents-md-support),
[cryptobriefing](https://cryptobriefing.com/anthropic-claude-code-agents-md-support))
— treat the fallback as version-dependent and **do not rely on it** `[unsourced:
fallback shipping status not verified on this machine]`. Projects that carry both use
one canonical file plus a thin bridge: `AGENTS.md` canonical with a 1-line `CLAUDE.md`
`@AGENTS.md` import, or the reverse symlink (secondary:
[cc-templates](https://github.com/fuzzykala/cc-templates),
[rugwiroparfait/agents.md](https://github.com/rugwiroparfait/agents.md)).
Recommendation: keep `CLAUDE.md` (Korean, existing, richer) canonical for Claude,
write `AGENTS.md` in English as the portable subset (build/test commands, prohibitions,
pointers), and add a one-line `@AGENTS.md` reference so neither drifts.

## 8. What a 2026 Rust reviewer would call out (short)

1. **FMA contraction across the host/device boundary.** cuda-oxide exposes `--no-fmad`
   (`CUDA_OXIDE_NO_FMA=1`) because implicit multiply-add contraction changes numerics
   ([README](https://raw.githubusercontent.com/NVlabs/cuda-oxide/main/crates/cargo-oxide/README.md)).
   The ggml-oracle band must state the FMA policy on both sides, or a 1e-4 failure is
   undebuggable. Same for host `-C target-cpu=znver3` RUSTFLAGS: tribal knowledge today,
   belongs in `.cargo/config.toml` + `AGENTS.md`.
2. **`--unchecked-indexing` exists as a cuda-oxide build flag** (elides device bounds
   checks; "UB on OOB", same README). A delegated round *will* discover it while chasing
   GB/s. Decide now, in writing, whether device code may use it — and if yes, the
   `checked-index`-equivalent gate for device code is Compute Sanitizer
   (`cargo oxide sanitize`, same README), not hope.
3. **PTX is arch-pinned output.** `--arch sm_86` vs auto-detect vs `--materialize-cubin`
   produce different artifacts; an `insta` PTX snapshot without the arch in its name or
   assertion context will false-fail on the first machine change. Name snapshots
   `..._sm86.snap`.
4. **Vendored-dependency provenance.** `tools/ref/` links ggml C++; the ik fork's
   `block_q8_K` layout (with its ik-only `sum` field) is already load-bearing in
   `q3k-cpu`. Pin the fork rev next to the layout comment or the next upstream rebase
   silently moves the struct under the code.
5. **No `#[test]` benchmarks, no uncaptured quiet-machine state.** Covered in §6; the
   reviewer version: any number in `RESULTS.md` without a witness row is a rumor.

## What I would do first, in order

1. Create the root workspace with `workspace.dependencies` (pin the `cuda-oxide` git `rev`) and `workspace.lints`, converting members to `[lints] workspace = true`.
2. Deny `unsafe_op_in_unsafe_fn`, `undocumented_unsafe_blocks`, and `missing_safety_doc` in `workspace.lints` and fix the fallout once.
3. Add a `checked-index` feature swapping `get_unchecked` for asserted indexing, and run the ggml-oracle comparisons under it.
4. Put oracle assertions (relative-error band) and proptest block arbitraries in `tests/` for each crate, with GPU/hardware tests `#[ignore]`d behind a nextest `default-filter`.
5. Snapshot one PTX output and one dequantized row with `insta` so codegen drift fails loudly.
6. Add `deny.toml`, a `justfile` fronting `tools/box.sh` + the quiet-machine lock, and CI running clippy/nextest/machete/deny.
7. Write `AGENTS.md` (English, ≤60 lines, commands + prohibitions + pointers) with a one-line bridge from `CLAUDE.md`.

---

*Trap docs checked: `CLAUDE.md`, `docs/plan.md`, `crates/q3k-cpu/src/main.rs`,
`crates/q3k-gemv/src/main.rs`, all three `Cargo.toml` + `rust-toolchain.toml` files.
Applicable items: nightly pin (all three), `cargo oxide` build driver, ggml-oracle
contract, quiet-machine measurement rule, Korean-prose/English-code convention —
each drove a section above. No existing helper/mapping/constant table was found to
reuse (the crates share no code today — itself a §1 argument); no other surface
(web/TUI/CLI/server) exists to cross-check; no user-facing copy was written; no test
assertions were changed; no files outside `docs/research/rust-quality-2026.md` were
touched.*


