#!/usr/bin/env bash
dspark_path() {
  local d=${1:-} why=${2:-}
  if [ -z "$d" ]; then
    echo "${why:-ref-paths.sh: DSPARK_MODEL is empty (exit 66)}" >&2
    exit 66
  fi
  printf '%s' "$d"
}
