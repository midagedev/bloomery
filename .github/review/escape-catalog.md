# Review escape catalog

Patterns that passed review here and turned out to be defects. The review bot reads this file before it decides; when
a diff matches a pattern, the pattern's verdict applies as written. The catalog grows instead of the prompt.

Rules:
- An entry needs a confirmed defect in this project (a fix commit or a red gate), not a guess.
- Do not add what a lint, `cargo fmt`, the CI check or a `tools/check-*.sh` script already catches, nor what a model
  knows without being told.
- Format: name, signal (how it looks in a diff), verdict, the incident. At most 10 lines each.
- When a check script starts catching a pattern, delete the entry and leave one line naming the script.

## C1. Exit code read after a command substitution or a pipe

- **Signal:** `echo "$(cmd) rc=$?"`, `x=$(cmd); echo $?` inside another substitution, `cmd | tail`, `cmd | tee log`
  without `set -o pipefail` or `${PIPESTATUS[0]}`, in a script that reports pass or fail.
- **Verdict:** must fix (kind A). The reported code is the last command's (`basename`, `tail`), so a red run reads
  green.
- **Incident:** 2026-10-05, a day of check-script results that were `basename`'s exit code; again 2026-10-12 in a box
  runner; a `merge --ff-only | tail` that hid a failed merge (2026-10-04).

## C2. Unquoted heredoc in a script that writes a script or a spec

- **Signal:** `cat <<EOF` (not `<<'EOF'`) whose body holds backticks, `$(…)` or `$var` meant literally.
- **Verdict:** must fix (kind A). The shell runs the backticks while writing the file.
- **Incident:** 2026-09-28 and 2026-10-05, backticks in a heredoc body ran a gate while a file was being written.

## C3. Process matching that can hit its own shell

- **Signal:** `pgrep -f <pattern>`, `pkill -f <pattern>`, a scan of `/proc/*/cmdline`, or `until ! pgrep …` used to
  wait for or stop a job.
- **Verdict:** must fix (kind A). The pattern is in the caller's own command line, so the wait never ends or the
  kill takes down the caller (once the SSH session). Wait on a pid captured with `$!`; signal only pids recorded at
  start.
- **Incident:** 2026-09-22 (a `/proc` cmdline scan stopped its own SSH session), 2026-10-10 (a cleanup `pkill` killed
  its own shell).

## C4. A shell function named after the command it calls

- **Signal:** `md5() { … md5 … }`, `sort() { … sort … }`.
- **Verdict:** must fix (kind A). The function calls itself until the process limit.
- **Incident:** 2026-10-11, a helper `md5` forked 5,300 processes.
