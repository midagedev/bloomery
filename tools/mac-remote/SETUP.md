# The second Mac for `tools/mac-remote.sh`

What is installed on it, and the commands that made each piece. Commands only; the host is the one line of
`~/.config/bloomery/mac-remote` (`user@host`) on the Mac that runs the tool, and is not in this repository.

The account also runs CI runners, so nothing below edits a shell profile or touches `~/.local/bin`: the runner
sets its own `PATH` (`~/opt/bloomery-bin:~/.cargo/bin:/usr/bin:…`) in every remote script. Present already on a
stock macOS: `git`, `rsync`, `curl`, the Xcode command line tools (`/Library/Developer/CommandLineTools/usr/lib/libclang.dylib`).

Run each step on the second Mac (`ssh "$(cat ~/.config/bloomery/mac-remote)" bash -s < step.sh`). `tools/mac-remote.sh`
checks every piece in one preflight and names a missing one with its step number (rc 69).

## 1. Python (3.11 or newer: tools/recipes.py imports `tomllib`; the system `/usr/bin/python3` is 3.9), with numpy and Pillow

check-recipes runs the Python tools' self-tests, and two of them (`tools/ref/window-union.py`, `tools/ref/router-residency.py`)
fail without numpy; the lead's Mac has numpy 2.5.1 and Pillow 12.2.0, so the venv has the same two.

    mkdir -p ~/opt/bloomery-bin ~/opt/bloomery-setup && cd ~/opt/bloomery-setup
    curl -sSfL -o uv.tgz https://github.com/astral-sh/uv/releases/download/0.13.0/uv-aarch64-apple-darwin.tar.gz
    tar -xzf uv.tgz && cp uv-aarch64-apple-darwin/uv ~/opt/bloomery-bin/uv
    export UV_PYTHON_INSTALL_DIR=$HOME/opt/bloomery-py UV_PYTHON_BIN_DIR=$HOME/opt/bloomery-bin
    ~/opt/bloomery-bin/uv python install 3.14
    ~/opt/bloomery-bin/uv venv --python 3.14 ~/opt/bloomery-venv
    ~/opt/bloomery-bin/uv pip install --python ~/opt/bloomery-venv/bin/python numpy==2.5.1 pillow==12.2.0
    # a wrapper, not a symlink (a venv's python3 must be started by its own path to find the venv), and relative to
    # itself, not to $HOME (recipes.py's self-test runs tools under a fake HOME)
    printf '#!/bin/sh\nexec "$(dirname "$0")/../bloomery-venv/bin/python3" "$@"\n' > ~/opt/bloomery-bin/python3
    chmod +x ~/opt/bloomery-bin/python3

Installed: uv 0.13.0, Python 3.14.8 (the lead's Mac runs 3.14.7), numpy 2.5.1, Pillow 12.2.0.

## 2. just, at the version of the lead's Mac (recipes.py parses `just --dump --dump-format json`)

    curl -sSfL -o just.tgz https://github.com/casey/just/releases/download/1.58.0/just-1.58.0-aarch64-apple-darwin.tar.gz
    tar -xzf just.tgz just && cp just ~/opt/bloomery-bin/just

Installed: just 1.58.0. The runner refuses (69) a remote `just --version` that differs from the local one.

## 3. Rust: rustup without profile edits, then every component rust-toolchain.toml pins and the cross-check target

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o rustup-init.sh
    sh rustup-init.sh -y --default-toolchain none --no-modify-path
    ~/.cargo/bin/rustup toolchain install nightly-2026-08-28 --profile minimal \
      -c rust-src,rustc-dev,rust-analyzer,clippy,rustfmt,llvm-tools -t x86_64-unknown-linux-gnu

Installed: rustup 1.29.1, `rustc 1.100.0-nightly (e457a7b0d 2026-08-27)`. A new `rust-toolchain.toml` channel needs
the same `toolchain install` line with the new date. The pinned components are all taken up front so a plain `cargo`
through the rustup proxy has nothing to add during a run.

## 4. The cross check's environment, copied byte for byte from the lead's Mac (the shape keys do not hash these headers)

    # on the lead's Mac
    rsync -a ~/opt/cuda-13.3 ~/opt/linux-sysroot ~/opt/bloomery-mac-env.sh "$(cat ~/.config/bloomery/mac-remote):opt/"
    rsync -anci ~/opt/cuda-13.3 ~/opt/linux-sysroot ~/opt/bloomery-mac-env.sh "$(cat ~/.config/bloomery/mac-remote):opt/"   # no lines: identical

(The recipe in the header of `tools/mac-check.sh` takes the same two trees from the box, `ws:/usr/local/cuda-13.3/include`
and `ws:/usr/include`; a second Mac that must not drift from the first copies from it.) When `~/opt` on the lead's Mac
changes, run the first command again.

## Layout and upkeep

    ~/repo/bloomery-mr/<track>/          one clone a track (the worktree directory's name); target/ is its own
    ~/repo/bloomery-mr/.lock-<track>/    the track's lock: owner (host/pid), run-pid while mac-static runs
    ~/repo/bloomery-mr/.slot-<i>         the run slots (SLOTS in tools/mac-remote.sh, 2): a run holds one with `lockf`
                                         (/usr/bin/lockf takes flock(2) through O_EXLOCK, so the kernel frees the slot when
                                         its holder dies) from `just mac-static`'s start to its end; `.slot-<i>.holder` names
                                         the track and start time of the holder, which a run prints at its start and, while it waits, every 5 min
    ~/.cache/bloomery/mac-static-ledger.tsv   the remote's own ledger (shared by the tracks, not seeded)

A track's first run is a cold build; the next ones are warm. A retired track's clone and target/ are removed by hand
(`rm -rf ~/repo/bloomery-mr/<track>`); nothing prunes them.

## Walls (idle second Mac, 2026-10-11, `tools/mac-remote.sh 347a86e5` for the tree at dc49ae45, warm `target/`)

    one run alone         507 s, 509 s   (setup 8-11 s, run 497 s)
    two runs at once      533 s, 548 s   (+5-8 %; one of them lint-cold: lint 72 s, setup 24 s)
    one run, both slots held for 63 s    571 s = 63 s of waiter lines naming the holders, then the usual 497 s + setup 8 s

The wall is one block of `check-recipes`: the gate-batch self-test (`tools/gate-batch.sh --self-test`, serial; 424 checks at dc49ae45, and the count and its wall grow with every case added since)
took 474 s of the 494 s `check-recipes` wall at load 1.5 (a timed copy of check-recipes.sh); the next block took 84 s, the
prelude 20 s, the other blocks run in parallel under those. `just mac-static` on any Mac meets the same block (derived, not
measured on the lead's Mac), so the second Mac removes the load effect, not the 474 s.
