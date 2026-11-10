#!/usr/bin/env bash
# shellcheck shell=bash
# The kimi-k3-full profile: Kimi K3 at the real file's widths and counts, the profile whose GGUF fixture is the full-shape
# cut (crates/model/src/arch/registry.rs PROFILE_FIXTURES, the kimik3full row of tools/fixture-dirs.tsv). The mini's
# dispatch classes are not the real file's (tools/kcover/unreached.tsv); this cut's are. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=kimi-k3-full; never executed, and it exports nothing. Every name but MODEL_NAME is the kimi-k3 profile's: the
# real tier's file is that profile's file.
__full_dir=${BASH_SOURCE[0]%/*}
[ "$__full_dir" != "${BASH_SOURCE[0]}" ] || __full_dir=.
# shellcheck source=tools/ref/models/kimi-k3.sh
source "$__full_dir/kimi-k3.sh"
unset __full_dir
# SC2034: read by the file that sources this one. MODEL is the base profile's, spelled here so an item's key
# selects its directory's manifest rows from this file too.
# shellcheck disable=SC2034
MODEL_NAME=kimi-k3-full
MODEL=${BLOOMERY_REF_MODEL:-/models/Kimi-K3-UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}
