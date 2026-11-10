#!/usr/bin/env bash
# shellcheck shell=bash
# The muse-spark-full profile: Muse Spark at the real file's widths and counts, the profile whose GGUF fixture is the full-shape
# cut (crates/model/src/arch/registry.rs PROFILE_FIXTURES, the musesparkfull row of tools/fixture-dirs.tsv). The mini's
# dispatch classes are not the real file's (tools/kcover/unreached.tsv); this cut's are. Sourced by ref-paths.sh when
# BLOOMERY_MODEL=muse-spark-full; never executed, and it exports nothing. Every name but MODEL_NAME is the muse-spark profile's: the
# real tier's file is that profile's file.
__full_dir=${BASH_SOURCE[0]%/*}
[ "$__full_dir" != "${BASH_SOURCE[0]}" ] || __full_dir=.
# shellcheck source=tools/ref/models/muse-spark.sh
source "$__full_dir/muse-spark.sh"
unset __full_dir
# SC2034: read by the file that sources this one. MODEL is the base profile's, spelled here so an item's key
# selects its directory's manifest rows from this file too.
# shellcheck disable=SC2034
MODEL_NAME=muse-spark-full
MODEL=${BLOOMERY_REF_MODEL:-/data/moved/models/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf}
