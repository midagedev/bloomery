#!/usr/bin/env bash
# shellcheck shell=bash
# The nanbeige43 (Nanbeige4.3-33B, dense) profile: what the reference engine and the harnesses need that is a property of
# the *model*, not of the machine. Sourced by ref-paths.sh when BLOOMERY_MODEL=nanbeige43; never executed, and it exports
# nothing. The file's GGUF general.architecture is llama (the profile's name picks the file, the architecture picks the
# engine), so `grep llama` finds the metadata keys (llama.*) and the engine's reader, and `grep nanbeige43` this profile, its
# oracle sets and their refset family (crates/refset/src/arch/llama/dense.rs).
#
#   MODEL           bartowski's Nanbeige4.3-33B Q4_K_M (rev 9beb68421fdf7be095093a6888325fa2c83ddd38, 20,788,825,216 bytes,
#                   sha256 2cef7985...: tools/ref/dense-logits.py TARGETS holds the identity the writer checks before it opens
#                   the file). BLOOMERY_REF_MODEL moves it. The file runs on either card in both tiers, so the fixture tier's
#                   row is `self` (tools/ref/ref-paths.sh; tools/fixture-dirs.tsv is keyed by this profile's name)
#   LCPP            the mainline llama.cpp tree that is the numeric oracle, at LCPP_PIN (src/models/llama.cpp, LLM_ARCH_LLAMA,
#                   rope type NORM). dense_logits_ref (tools/ref/dense_logits_ref.cpp) links against its build/bin libraries
#                   (tools/ref/build-dense-logits.sh), which tools/ref/emb2/oracle-tree.sh proves are the commit's build.
#                   The tree is the qwen35 profile's; IK is the default the scripts that expand IKBIN beside every profile
#                   take, no ik oracle exists for this profile
#   PROSE_IDS       the prose corpus under this vocabulary: the tokenizer oracle's prose text
#                   ($BLOOMERY_DATA/tokenizer-qwen3moe/prose.txt, md5 c6bb074439479420faffa33257424c00, the text the qwen35
#                   profile's corpus is made from) through this file's mainline llama-tokenize, one id per line:
#                     llama-tokenize -m $MODEL -f <prose.txt> --ids --no-parse-special --no-bos --no-escape --log-disable
#                   with the brackets and commas of the `[a, b, c]` print turned into lines. --no-bos because the file sets
#                   add_bos_token (BOS is <|im_start|>, 166100): the ids are the text's own, as the Nemotron tokenizer set's are,
#                   and a prompt cut from them starts with the text. --no-escape because llama-tokenize turns `\n` in a -f text
#                   into a newline otherwise. 83,482 ids; PROSE_SHA256 is the check
#   LCPPBIN         mainline llama-bench, the depth runner's reference arms (lcpp:, lcpppp:)
#   LCPP_GPU_FLAGS  the whole model on the card the run holds, flash attention on
#   LOGITS_TARGET   tools/ref/dense-logits.py TARGETS' name for MODEL: its size, sha256, vocabulary and architecture
#   LOGITS_SETS     the oracle sets, `<set>:<P>`: the first P ids of PROSE_IDS are the prompt and the LOGITS_FORCED ids after
#                   them the forced continuation (teacher forced on the corpus), each row mainline's logits after the prompt
#                   or after a forced id (tools/ref/dense-logits.sh)
#   LOGITS_CTX/UBATCH  the context every set is produced at (P = 4,096 and 32 forced ids fit; the card engine's
#                   context is 4,352 too) and llama.cpp's default ubatch
#
# SC2034: every name here is read by the file that sources this one, which shellcheck does not see from this file alone.
# shellcheck disable=SC2034
MODEL_NAME=nanbeige43
: "${IK:=/home/user/ik-idxkey}"
MODEL=${BLOOMERY_REF_MODEL:-/models/Nanbeige4.3-33B/Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf}
: "${LCPP:=/home/user/llama.cpp-mainline}"
LCPP_PIN=53ed051ce5e8193652e449f43216ca3859454f49
: "${LCPPBIN:=$LCPP/build/bin/llama-bench}"
: "${LCPP_GPU_FLAGS:=-ngl 99 -fa on}"
PROSE_IDS=${BLOOMERY_DATA:-/root/bloomery-data}/nanbeige43/corpus-prose.ids
PROSE_SHA256=cd3a1e34bd40830d5c4314d4fa65cf051e380b2b4fce262f013f8fd134fbc013
LOGITS_TARGET=nanbeige43-33b
LOGITS_FORCED=32
LOGITS_CTX=4352
LOGITS_UBATCH=512
LOGITS_SETS=(ref_nanbeige43_logits_p64:64 ref_nanbeige43_logits_p4096:4096)
