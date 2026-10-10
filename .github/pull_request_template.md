## What and why

<!-- What changes, in 2–3 sentences, and the issue it closes (`Closes #N`). -->

## Code shape

The review reads these first. A box left unticked needs one line of why.

- [ ] **No second copy.** Logic two models (or two call sites) share has one common owner; I lifted it there instead of
      copying it (`AGENTS.md` "Common code first", `docs/rust-quality.md` R14). Where I found an existing copy, I
      named it below.
- [ ] **Model code holds only architecture facts.** A model's own files carry only what its architecture forces
      (tensor shapes, the attention kind, the quant). A common owner gained no per-model branch, flag or trait method.
- [ ] **Dead code is gone.** Nothing I made unused is left behind (functions, flags, gate clauses, docs).
- [ ] **No silent failure.** Undefined input is a named error or panic, never a defined-looking output.
- [ ] **Comments state what is true now.** No dates, issue numbers or how a bug was found (`tools/check-comments.sh`).
- [ ] **No relaxed gate.** No band, tolerance or lint moved to make a check pass; a re-pin carries `PIN(YYYY-MM-DD):`
      and its reason.
- [ ] `cargo fmt` and `cargo clippy` are clean on what I touched; the lint count did not go up.

Duplication found or removed (`path:line`), or "none":

## What I ran

<!-- Paste the commands and their real output (the summary lines). Say which you could not run and why: most gates
need the maintainers' workstation, and a maintainer runs those (CONTRIBUTING.md, "How a pull request is verified"). -->

## Hardware (if the change is for a machine)

<!-- `--version`, card, driver, CPU, RAM, and what you ran on it. Your numbers carry their conditions. -->

## AI help

<!-- Say whether AI helped and with which parts. You have read the diff and can explain it. -->
