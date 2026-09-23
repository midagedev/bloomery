# Contributing

Thank you for looking. Issues and pull requests are welcome. Three rules matter more than the rest.

**The gates are the contract.** Every subsystem has a `just gate-*` recipe that exits non-zero on failure, and a change is done when the gates it touches pass. Do not relax a band, a tolerance or a lint to make a gate pass. If a threshold must move, move it with a dated `PIN(YYYY-MM-DD):` comment and the reason. A new gate should first be shown to fail on the defect it guards.

**Measurements only through the runners.** Timed numbers come from the runners in `tools/ref/` (`measure.sh`, `cpu-measure.sh`, `depth-ds41.sh` and their `just` recipes). They take a machine-wide lease, wait for an idle GPU and write a witness block around every timed region. A number taken by hand next to a running build is not comparable, and a pull request should not quote one.

**No number without its conditions.** Write `tok/s @ n=96, depth 4096, RTX A6000`, not `29 tok/s`. Say which placement, which prompt and whether instrumentation was on. A derived number says [derived]. A number that turns out wrong is struck through and corrected in place, not silently edited.

Also:

- Code comments and anything that goes upstream are in English. Comments state what is true now; history belongs in the commit message.
- Code shape follows `docs/rust-quality.md`.
- Never build a device crate with plain `cargo`; use `cargo oxide build --arch sm_86` (see `docs/BUILD.md`).
- The full working contract is `AGENTS.md`.
