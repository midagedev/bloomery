# shellcheck shell=bash
# Source provenance for timing witnesses. Callers must check tree_line's status before inserting
# its output into another command or an array: these runners do not enable errexit.
# tree_line <binary> <tree> [HEAD digits=9] [no-Git policy=required|box]
# The box policy is depth-ds41's synced-tree path: without Git, only the runner's own tree may
# use BLOOMERY_GIT_COMMIT, and its dirty count remains unknown, never a clean count.
tree_line() {
  local bin=$1 tree=$2 digits=${3:-9} policy=${4:-required} sha head dirty status rc
  sha=$(sha256sum "$bin" 2> /dev/null | cut -c1-12)
  if head=$(git -c safe.directory="$tree" -C "$tree" rev-parse --short="$digits" HEAD 2> /dev/null); then
    :
  elif [ "$policy" = box ]; then
    head='?' dirty='?'
    if [ -n "${BLOOMERY_GIT_COMMIT:-}" ] && [ "$(cd "$tree" 2> /dev/null && pwd -P)" = "$(pwd -P)" ]; then
      head="$BLOOMERY_GIT_COMMIT(box.sh)"
    fi
    printf '%s sha256=%s head=%s dirty_files=%s\n' "$bin" "${sha:-?}" "$head" "$dirty"
    return 0
  else
    head='?'
  fi
  # Keep git's exit separate from the line count; pipefail alone cannot stop a non-errexit caller.
  # Optional locks stay off so a root runner never rewrites another user's index.
  status=$(GIT_OPTIONAL_LOCKS=0 git -c safe.directory="$tree" -C "$tree" status --porcelain --untracked-files=no 2> /dev/null) || {
    rc=$?
    printf 'source-inspection: git status failed for %s (exit %s)\n' "$tree" "$rc" >&2
    return 2
  }
  dirty=0
  [ -z "$status" ] || dirty=$(printf '%s\n' "$status" | wc -l | tr -d ' ')
  printf '%s sha256=%s head=%s dirty_files=%s\n' "$bin" "${sha:-?}" "$head" "$dirty"
}
