#!/usr/bin/env bash
# shellcheck shell=bash
# The seat the image-input oracle runs for: what clefvis.sh and build-clefvis.sh need that is a property of the seat, not of
# the machine. Sourced after tools/ref/ref-paths.sh, which has picked the model profile (BLOOMERY_MODEL); never executed,
# and it exports nothing. A seat is a family of the Qwen3-VL tower (mtmd `qwen3vl_merger`) and its text model:
#
#   qwen35     Clef-Flash         sets ref_clefvis_*             mainline 53ed051ce (the model profile's LCPP)
#   qwen35moe  Qwen3.6-35B-A3B    sets ref_qwen35moe_vis_*       mainline 36a73916e (QVIS_LCPP)
#   qwen4exp   Qwen3.8-Flash-Next sets ref_qwen4exp_vis_*        mainline 36a73916e (QVIS_LCPP)
#
# The two Qwen seats share the mainline tree and so the binary; the Clef seat keeps its own tree and binary, so its sets'
# bytes do not move with the Qwen seats' pin. The values below are the ones crates/refset names for the same seat
# (arch/qwen35/clefvis.rs, arch/qwen35moe/vis.rs, arch/qwen4exp/vis.rs): the family checks a set's `# model`, `# mmproj`,
# `# build` lines against them.
#
#   FAMILY          clef | qvis: which arms the seat has. clef: set A (preproc), the release's ids (clef_ref.py), set D;
#                   qvis: the chat ids (E) and the decode steps (F), the tower tapped to its output end
#   SETPFX          the sets' common name
#   ORACLE          the mainline tree the sets are dumped from, its build/bin holding libmtmd, libllama and libggml
#   BIN             the dump_mtmd binary of that tree
#   MMPROJ, MMPROJ_SHA256, TEXT, PROSE_IDS, PROSE_SHA256   the files the sets are dumped from and their digests
#   CASES, REQUESTS the cases (case, images, decode steps) and the chat requests the ids come from (qvis)
#   REF             where a case's ids file is
#   NCMOE_A6000, NCMOE_3090   the experts of this many blocks stay in host memory on that card (0: the model fits)
#   ROWS_FROM       dump_mtmd hidden --rows-from: where result_norm is read. embd: the context's embeddings (Clef, Qwen3.6);
#                   node: the graph node through cb_eval (Qwen3.8: qwen4exp's n_embd_out is hc * n_embd, so mainline's embedding
#                   extraction reads past result_norm). A node-read set says `# rows_from node`; asking for a node splits the
#                   scheduler's graph, so its rows differ from the embeddings' by rounding
#   THREADS         the CPU threads of a card run's host-side experts and of the twin
#
# SC2034: every name here is read by the files that source this one.
# shellcheck disable=SC2034
: "${QVIS_LCPP:=/home/user/llama.cpp-36a73916}"
DATA=${BLOOMERY_DATA:-/root/bloomery-data}
NCMOE_A6000=0 NCMOE_3090=0 ROWS_FROM=embd THREADS=16 CASES=tools/ref/clefvis/cases.tsv REQUESTS=
case ${MODEL_NAME:-} in
  qwen35)
    FAMILY=clef SETPFX=ref_clefvis ORACLE=$LCPP BIN=$DATA/bin/dump_mtmd
    MMPROJ=/root/models/clef-flash/mmproj-Cloudflare_clef-flash-bf16.gguf
    MMPROJ_SHA256=3c45b34aee6f353a0d41d6b96ba712a498a17a82f0a6152bf68a5021f9652c0f
    TEXT=$FLASH_Q8
    REF=${CLEFVIS_REF:-$DATA/clefvis/ref}
    ;;
  qwen35moe)
    FAMILY=qvis SETPFX=ref_qwen35moe_vis ORACLE=$QVIS_LCPP BIN=$DATA/bin/dump_mtmd_qvis
    MMPROJ=/models/mmproj/lmstudio-community--Qwen3.6-35B-A3B-GGUF/mmproj-Qwen3.6-35B-A3B-BF16.gguf
    MMPROJ_SHA256=e5c205cec2fd28f66c3895e4040021ab994b860323c3db8531640305ff49b322
    TEXT=/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-Q4_K_M.gguf
    PROSE_IDS=$DATA/qwen35moe/corpus-prose.ids
    PROSE_SHA256=dca5f89b2903f9ffd2f4e20eec18a9fdf3561bcb531a721a114e4fab68e925be
    CASES=tools/ref/qvis/cases.tsv REQUESTS=tools/ref/qvis/requests.jsonl
    REF=${QVIS_REF:-$DATA/qvis/qwen35moe}
    # the 21.2 GB file whole on a 48 GB card; on the 24 GB card with the tower and a 4,096 context the experts of 4 blocks (0.49 GB each) stay on the host
    NCMOE_A6000=${QVIS_NCMOE_A6000:-0} NCMOE_3090=${QVIS_NCMOE_3090:-4}
    THREADS=16
    ;;
  qwen4exp)
    ROWS_FROM=node FAMILY=qvis SETPFX=ref_qwen4exp_vis ORACLE=$QVIS_LCPP BIN=$DATA/bin/dump_mtmd_qvis
    MMPROJ=/models/mmproj/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf
    MMPROJ_SHA256=2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355
    TEXT=/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
    PROSE_IDS=$DATA/qwen4exp/corpus-prose.ids
    PROSE_SHA256=dca5f89b2903f9ffd2f4e20eec18a9fdf3561bcb531a721a114e4fab68e925be
    CASES=tools/ref/qvis/cases.tsv REQUESTS=tools/ref/qvis/requests.jsonl
    REF=${QVIS_REF:-$DATA/qvis/qwen4exp}
    # the 111 GB file does not fit a card: the experts of the first blocks stay on the host (tools/ref/models/qwen4exp.sh
    # sizes llama-bench 26 and 43 for a 512-token context; the dump's 4,096 context and tower want a block or two more)
    NCMOE_A6000=${QVIS_NCMOE_A6000:-28} NCMOE_3090=${QVIS_NCMOE_3090:-45}
    THREADS=24
    ;;
  *)
    echo "profile.sh: no image-input seat for BLOOMERY_MODEL='${MODEL_NAME:-}' (qwen35, qwen35moe or qwen4exp)" >&2
    exit 2
    ;;
esac
