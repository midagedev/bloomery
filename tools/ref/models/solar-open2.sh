#!/usr/bin/env bash
# shellcheck shell=bash
# The solar-open2 (Solar-Open2-250B, Q4_K_M) profile: what the reference harness needs that is a property of the
# model. Sourced by ref-paths.sh when BLOOMERY_MODEL=solar-open2; never executed, and it exports nothing.
#
# The name is the GGUF general.architecture value, so `grep solar-open2` finds this profile, the registry row and the
# oracle family (crates/refset/src/arch/solar2).
#
#   MODEL   the first shard of the four-shard Q4_K_M split set (llamaraspberryrabbit/Solar-Open2-250B-GGUF, 151.8 GB).
#           BLOOMERY_REF_MODEL moves it; a caller's own MODEL= is ignored. The path is the family's identity
#           (refset::arch::solar2::MODEL): a set names its file's full path, so the file is not moved or deleted before
#           its sets are re-dumped.
#   PAI     prometheusAIR's solar_open2 patch on llama.cpp 6ea215d17 (/home/user/llama.cpp-solar2, built CPU+CUDA): the
#           only faithful implementation of the architecture (beta = 2 sigmoid), so the oracle and the public baseline
#           (rig-log 2026-10-20: no mainline arch, no PR). ik has no solar-open2; the default IK is only so IKBIN can be formed.
#
# The dump (tools/ref/solar2-dump.sh) hides CUDA and takes the CPU lease.
#
# SC2034: every name here is read by the file that sources this one.
# shellcheck disable=SC2034
MODEL_NAME=solar-open2
: "${IK:=/home/user/ik-idxkey}"
: "${PAI:=/home/user/llama.cpp-solar2}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Solar-Open2-250B-Q4_K_M/Solar-Open2-250B-Q4_K_M-00001-of-00004.gguf}
