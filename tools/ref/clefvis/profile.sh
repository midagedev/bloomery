#!/usr/bin/env bash
# shellcheck shell=bash
# The seat the image-input oracle runs for: what clefvis.sh and build-clefvis.sh need that is a property of the seat, not of
# the machine. Sourced after tools/ref/ref-paths.sh, which has picked the model profile (BLOOMERY_MODEL); never executed,
# and it exports nothing. A seat is a family of the Qwen3-VL tower (mtmd `qwen3vl_merger`) and its text model, of the GLM
# tower (`glm5v`) and its, or of the MiniCPM-V 4.7 tower (`minicpmv4_7`) and its:
#
#   qwen35     Clef-Flash         sets ref_clefvis_*             mainline 53ed051ce (the model profile's LCPP)
#   qwen35moe  Qwen3.6-35B-A3B    sets ref_qwen35moe_vis_*       mainline 36a73916e (QVIS_LCPP)
#   qwen4exp   Qwen3.8-Flash-Next sets ref_qwen4exp_vis_*        mainline 36a73916e (QVIS_LCPP)
#   glm5next   GLM-5.3-Flash      sets fx_ref_glm5next_vis_*     mainline 36a73916e (QVIS_LCPP), the fixture tier only
#   minicpmv47 MiniCPM-V 4.7      sets ref_minicpmv47_vis_*      mainline 358a420ec (MC47_LCPP), the tower arm only
#
# The row is picked by BLOOMERY_MODEL, the model profile's name: MiniCPM-V 4.7's text profile (models/minicpmv47.sh) is a
# qwen35moe file and says MODEL_NAME=qwen35moe, which is Qwen3.6's row. The Qwen seats and the GLM seat share the mainline tree
# and so the binary; the Clef seat keeps its own tree and binary, so its sets' bytes do not move with the Qwen seats' pin, and
# MiniCPM's tree (mainline gained `minicpmv4_7` at 69f201a20, after 36a73916e) has its own, dump_mtmd_mc47. The values below are the ones crates/refset names for the same seat
# (arch/qwen35/clefvis.rs, arch/qwen35moe/vis.rs, arch/qwen4exp/vis.rs): the family checks a set's `# model`, `# mmproj`,
# `# build` lines against them.
#
#   FAMILY          clef | qvis | mcvis: which arms the seat has. clef: set A (preproc), the release's ids (clef_ref.py), set D;
#                   qvis: the chat ids (E) and the decode steps (F), the tower tapped to its output end; mcvis: the tower only,
#                   sets A and B per tile (dump_mtmd's canvas seat), no ids, no prompt arms
#   PREPROC         1: the tower arm writes set A (preproc, the card run only): Clef and the GLM seat; 0: the Qwen seats
#   ARMS            the prompt kinds a case dumps: hidden (C), prose (C', with PROSE_IDS), bf16rows (C''); the GLM seat's is hidden
#   IMAGES_EXTRA    a file of more image rows (images.tsv's format) the tower arm takes after images.tsv: the GLM seat's cap case
#   IMAGES_LIST     the file of image rows the tower arm takes (images.tsv's format; default images.tsv): MiniCPM's three images
#   SEAT            the projector the seat is for, passed to dump_mtmd as --seat: it refuses another projector's mmproj by name
#                   (empty: the seat is whatever the mmproj's projector says, as the Qwen and GLM seats have always run)
#   IMAGE_PAD_ID    the id of the text model's image token, passed to dump_mtmd as --image-pad-id (empty: its Qwen default)
#   RESPELL_SOURCE  the fixture's first shard the text model is a respelled copy of (tools/ref/gguf-respell.py): the set's
#                   `# fixture` line is that file's, and dump_mtmd refuses a copy whose bloomery.respell.source names another
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
# MiniCPM-V 4.7's tree: mainline at 358a420e (just build-mc47-lcpp), the first one this seat's sets were dumped from
: "${MC47_LCPP:=/home/user/llama.cpp-358a420e}"
DATA=${BLOOMERY_DATA:-/root/bloomery-data}
# CLEFVIS_OUT moves the binary's directory, as build-clefvis.sh reads it (a scratch build under test).
BINDIR=${CLEFVIS_OUT:-$DATA/bin}
NCMOE_A6000=0 NCMOE_3090=0 ROWS_FROM=embd THREADS=16 CASES=tools/ref/clefvis/cases.tsv REQUESTS=
PREPROC=0 ARMS="hidden prose bf16rows" IMAGES_EXTRA="" IMAGE_PAD_ID="" RESPELL_SOURCE="" SEAT="" IMAGES_LIST=tools/ref/clefvis/images.tsv
case ${BLOOMERY_MODEL:-} in
  qwen35)
    FAMILY=clef PREPROC=1 SETPFX=ref_clefvis ORACLE=$LCPP BIN=$BINDIR/dump_mtmd
    MMPROJ=/root/models/clef-flash/mmproj-Cloudflare_clef-flash-bf16.gguf
    MMPROJ_SHA256=3c45b34aee6f353a0d41d6b96ba712a498a17a82f0a6152bf68a5021f9652c0f
    TEXT=$FLASH_Q8
    REF=${CLEFVIS_REF:-$DATA/clefvis/ref}
    ;;
  qwen35moe)
    FAMILY=qvis SETPFX=ref_qwen35moe_vis ORACLE=$QVIS_LCPP BIN=$BINDIR/dump_mtmd_qvis
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
    ROWS_FROM=node FAMILY=qvis SETPFX=ref_qwen4exp_vis ORACLE=$QVIS_LCPP BIN=$BINDIR/dump_mtmd_qvis
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
  glm5next)
    # GLM-5.3-Flash on the fixture tier: the text model is the fixture's copy in mainline's spelling `glm5-next`
    # (tools/ref/gguf-respell.py: only the header differs), read vocab-only by the tower arm; sets carry the fx_ prefix
    # (ref-paths.sh SET_PREFIX). The real tier waits for its file's spelling and offload plan (a release-batch round).
    fixture_dump_tier clefvis.sh
    [ -n "$SET_PREFIX" ] || { echo "profile.sh: the glm5next image-input seat runs on the fixture tier (BLOOMERY_TIER=fixture); the real file is a later round (exit 66)" >&2; exit 66; }
    FAMILY=qvis PREPROC=1 ARMS=hidden SETPFX=${SET_PREFIX}ref_glm5next_vis ORACLE=$QVIS_LCPP BIN=$BINDIR/dump_mtmd_qvis
    MMPROJ=/models/mmproj/unsloth--GLM-5.3-Flash-GGUF/mmproj-BF16.gguf
    MMPROJ_SHA256=82b7adb44411dbb298b1c689567804e5ed0790d80f308ed8b5b6b8ac0527d2ac
    TEXT=${QVIS_FIXTURE_MAINLINE_ROOT:-/models/fixtures-mainline}/glm5next/$(basename "$MODEL")
    RESPELL_SOURCE=$MODEL
    IMAGE_PAD_ID=154854
    IMAGES_EXTRA=tools/ref/clefvis/images-glm5v.tsv
    CASES=tools/ref/qvis/cases.tsv REQUESTS=tools/ref/qvis/requests.jsonl
    REF=${QVIS_REF:-$DATA/qvis/${SET_PREFIX}glm5next}
    # the 8 GB fixture fits a card whole with the tower and a 4,096 context: no experts stay on the host
    THREADS=16
    ;;
  minicpmv47)
    # the text model is a qwen35moe file with its own vocabulary (models/minicpmv47.sh); only the tower runs: it loads the
    # vocabulary, never the weights, so no experts stay on the host
    FAMILY=mcvis PREPROC=1 ARMS="" SEAT=minicpmv4_7 SETPFX=ref_minicpmv47_vis ORACLE=$MC47_LCPP BIN=$BINDIR/dump_mtmd_mc47
    MMPROJ=/models/MiniCPM-V-4.7-35B-A3B/mmproj-model-f16.gguf
    MMPROJ_SHA256=bff3b688e14c300eca273092c37c50a0ddfcc2054e079af1a0480edeed5d20f1
    TEXT=/models/MiniCPM-V-4.7-35B-A3B/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
    IMAGES_LIST=tools/ref/clefvis/images-minicpmv47.tsv
    CASES=/dev/null
    THREADS=16
    ;;
  *)
    echo "profile.sh: no image-input seat for BLOOMERY_MODEL='${BLOOMERY_MODEL:-}' (qwen35, qwen35moe, qwen4exp, glm5next or minicpmv47)" >&2
    exit 2
    ;;
esac
