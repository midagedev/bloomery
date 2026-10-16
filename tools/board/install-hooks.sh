#!/bin/bash
# Merge tools/board/claude-hooks.json into the user's Claude settings (the board's event hooks, tools/board/hook.py).
#   install-hooks.sh [--settings FILE] [--uninstall]
# Idempotent: our entries are the ones whose command names tools/board/hook.py; they are replaced, never doubled, and a
# run that changes nothing writes nothing. A changed file is backed up first as FILE.bak-<epoch>. @REPO@ in the fragment
# becomes this checkout (run it from the main checkout).
set -euo pipefail
settings=$HOME/.claude/settings.json
mode=install
while [ $# -gt 0 ]; do
    case $1 in
    --settings) settings=${2:?--settings takes a file}; shift 2 ;;
    --uninstall) mode=uninstall; shift ;;
    *) echo "usage: install-hooks.sh [--settings FILE] [--uninstall]" >&2; exit 64 ;;
    esac
done
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
[ -f "$settings" ] || { mkdir -p "$(dirname "$settings")"; echo '{}' >"$settings"; }
python3 - "$settings" "$here/claude-hooks.json" "$repo" "$mode" <<'PY'
import json, os, shutil, sys, time
path, frag_path, repo, mode = sys.argv[1:5]
MARK = "tools/board/hook.py"
with open(path, encoding="utf-8") as fh:
    cur = json.load(fh)
if not isinstance(cur, dict):
    sys.exit(f"{path}: not a JSON object")
new = json.loads(json.dumps(cur))
hooks = new.setdefault("hooks", {})
def ours(group):
    return any(MARK in h.get("command", "") for h in group.get("hooks", []))
for ev in list(hooks):  # drop every earlier copy of ours, wherever it sits
    hooks[ev] = [g for g in hooks[ev] if not ours(g)]
    if not hooks[ev]:
        del hooks[ev]
if mode == "install":
    with open(frag_path, encoding="utf-8") as fh:
        frag = json.loads(fh.read().replace("@REPO@", repo))
    for ev, groups in frag["hooks"].items():
        hooks.setdefault(ev, []).extend(groups)
if not hooks:
    del new["hooks"]
if new == cur:
    print(f"{path}: unchanged")
    sys.exit(0)
bak = f"{path}.bak-{int(time.time())}-{os.getpid()}"
shutil.copy2(path, bak)
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(new, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
os.replace(tmp, path)
print(f"{path}: {mode}ed (backup {bak})")
PY
