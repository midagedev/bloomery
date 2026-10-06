# Contributing

Thank you for looking. Issues and pull requests are welcome. Three rules matter more than the rest.

**The gates are the contract.** Every subsystem has a `just gate-*` recipe that exits non-zero on failure, and a change is done when the gates it touches pass. Do not relax a band, a tolerance or a lint to make a gate pass. If a threshold must move, move it with a dated `PIN(YYYY-MM-DD):` comment and the reason. A new gate should first be shown to fail on the defect it guards.

**Measurements only through the runners.** Timed numbers come from the runners in `tools/ref/` (`decode-measure.sh`, `depth-gpu.sh`, `depth-ds41.sh` and their `just` recipes). They take a machine-wide lease, wait for an idle GPU and write a witness block around every timed region. A number taken by hand next to a running build is not comparable, and a pull request should not quote one.

**No number without its conditions.** Write `tok/s @ n=96, depth 4096, RTX A6000`, not `29 tok/s`. Say which placement, which prompt and whether instrumentation was on. A derived number says [derived]. A number that turns out wrong is struck through and corrected in place, not silently edited.

**AI-assisted pull requests are welcome.** Use whatever tools you like; much of bloomery itself was written with AI
help. Point your agent at `AGENTS.md` first: it is the working contract, written to be followed by an agent. Three
things we ask:

- Say in the pull request that AI helped, and with which parts.
- You answer for the change: you have read the diff and can explain it.
- Paste the real output of the gates you ran. Do not paste output you did not run, or numbers that did not come from
  the runners.

**Pull requests for your own hardware are especially welcome.** bloomery has run on very little: one RTX A6000 48 GB,
one RTX 3090 24 GB, an RTX 3060 12 GB under WSL2, a 32-core AVX2 CPU (Zen 3) with 256 GB of RAM, and Linux. Other
cards, other memory sizes, other CPUs (AVX-512, fewer cores, less RAM), two cards of the same kind, and native Windows
are untested. If bloomery fails or runs slowly on your machine, a fix is worth more to us than almost anything else.

- Run the gates your change touches on your machine, and paste their output with the machine (`--version`, the card,
  the driver, the CPU and the RAM).
- We run the same gates on our cards before merging, so a change for your hardware does not break the cards we have.
- Make the new path follow from what the machine reports (the card, its memory, the CPU's features) rather than a
  flag only you would set, where the hardware allows.
- A speed claim comes from the runners in `tools/ref/`, with the conditions written out as above. We cannot repeat it
  on hardware we do not have, so say how you measured it.

Also:

- Code comments and anything that goes upstream are in English. Comments state what is true now; history belongs in the commit message.
- Code shape follows `docs/rust-quality.md`.
- Never build a device crate with plain `cargo`; use `cargo oxide build --arch sm_86` (see `docs/BUILD.md`).
- The full working contract is `AGENTS.md`.
