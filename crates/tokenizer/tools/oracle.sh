#!/usr/bin/env bash
# oracle.sh — reference token ids for the tokenizer gate, written next to the exact text they came
# from. The writer behind `just dump-ref-tokenizer`; `just gate-tokenizer` only reads the sets,
# through refset's tokenizer families (crates/refset/src/arch/tokenizer.rs), which refuse a set that
# this script did not finish or that another reference tree (commit or patch) or vocabulary wrote.
#
# Not a measurement: no lease, no timing, no GPU (CUDA_VISIBLE_DEVICES is emptied; llama-tokenize
# loads the vocabulary only). The text sets are engram-corpus.sh's, gathered the same way from the
# same trees, so the gate covers what the engram rounds tokenized:
#
#   code       $IK/src/*.cpp
#   prose      $IK/docs/**/*.md and $IK/README.md
#   prose-all  every Markdown file under $IK
#   threads    $IK/github-data/**/*.md, byte order
#   korean     this tree's docs/**/*.md that are at least 20 % Hangul, byte order
#
# plus the hand-picked cases in crates/tokenizer/tests/cases.txt (one per line, printf %b
# escapes; an empty line is the empty string; a line starting with '#' is a comment).
#
# Each text is tokenized twice: with special-token parsing (llama-tokenize's default) and with
# --no-parse-special (what engram-corpus.sh used). Output, under $BLOOMERY_DATA/<set>/:
#
#   <name>.txt           the text, exactly as the reference read it
#   <name>.ids           one id per line, special tokens parsed
#   <name>.nps.ids       one id per line, --no-parse-special
#   cases/<nn>.{txt,ids,nps.ids}
#   MANIFEST.tsv         the executable and the libllama.so it loads (path, md5), the tree they were
#                        built from (`# build`: its commit, and its patch where it has one), the
#                        vocabulary file, the cases file (path, md5), one row per text (the corpus
#                        names, then cases/<nn>: bytes, text md5, id counts), and the `# complete`
#                        trailer
#
# The vocabularies, each with its reference tree and its set (`profile` below owns the table;
# crates/refset/src/arch/tokenizer.rs pins the same trees, each as the commit it is built at, with the hash of its patch where
# it has one):
#
#   v41       V4.1's first shard                  ik-tilde  (`~` is a symbol, as in mainline and HF)  ref-tokenizer-v41
#   qwen3moe  the Qwen3-MoE file                  ik-tokref (the unicode_tolower fix)                  ref-tokenizer-qwen3moe
#   glm5next  GLM-5.3-Flash's first shard         ik-tokref                                            ref-tokenizer-glm5next
#   qwen38    Qwen3.8-Flash-Next's first shard    ik-tokref                                            ref-tokenizer-qwen38
#   nemotron  Nemotron 3 Nano 4B                  llama.cpp-19cce691 (pixtral; --no-bos, below)        ref-tokenizer-nemotron
#   minimaxm3 MiniMax-M3-UD-IQ4_XS's first shard  llama.cpp-solar2 (minimax-m2; --no-bos, below)       ref-tokenizer-minimaxm3
#   musespark Muse-Spark-Open's first shard       llama.cpp-muse (gpt-4o; --no-bos, below)             ref-tokenizer-musespark
#   mistrallarge4 Mistral Large 4's first shard   llama.cpp-ml4 (tekken-num3; --no-bos, below)         ref-tokenizer-mistrallarge4
#   gemma4    EmbeddingGemma 2's Q8_0 file        mainline 6868a52cb (the gemma-embedding2 profile's  ref-tokenizer-gemma4
#                                                 LCPP and MODEL, tools/ref/models/gemma-embedding2.sh)
#   llama     EmbeddingGemma v1's Q8_0 file      the same mainline pin, gemma-embedding profile       ref-tokenizer-llama
#   nanbeige43  Nanbeige4.3-33B's Q4_K_M file     llama.cpp-mainline at the nanbeige43 profile's LCPP  ref-tokenizer-nanbeige43
#                                                 and LCPP_PIN (tools/ref/models/nanbeige43.sh)
#
# Every vocabulary's reference is a tree's build, not one machine's md5s: before anything is read, the tree is checked against
# the pin the profile names (tools/ref/emb2/oracle-tree.sh owns the check), and the manifest then states `# build <commit>` or
# `# build <commit>+<patch>`, which the refset family pins. The commit is the first 9 digits of the tree's upstream commit
# (BUILD_PIN); the patch is `oracle_patch_hash`, the 12 digits that name the tree's tracked files against that commit, for the
# trees that carry local commits (ik-tilde, ik-tokref). A mainline tree has none: its HEAD is the commit, with no tracked change,
# and llama-tokenize's own `--version` names it. Every build is also held to be newer than every source of the tree (the commit
# does not say which sources the library holds); ik's llama-tokenize has no `--version`, so its age is all there is, and V4.1's
# `~` probe (below) shows the patch's effect in the binary. The executable's and the library's md5s
# stay in the manifest (tools/ref/emb2/emb2.sh holds the ids to this machine's binary) but change with every rebuild, so no
# family pins them. A tree that moved is a new pin, read off the tree: `source tools/ref/emb2/oracle-tree.sh; oracle_patch_hash
# <tree> <commit>`.
#
# llama's corpora are cut into rows of at most CHUNK bytes, named <corpus>.<nnn> (a cut at a line end: `split -C`). That vocabulary's
# SPM tokenizer partitions a text at its user-defined special tokens, the newline runs among them, and walks the fragment list once
# per match, so the time of one text grows with the square of its size; the corpora whole outran the recipe's bound. The other
# vocabularies keep one row per corpus, which the tokenizer gate's corpus test names. nanbeige43 is SentencePiece too, but its
# user-defined tokens are four (<think> </think> <tool_call> </tool_call>) and a newline is a byte-fallback piece, so a text has
# few matches to walk and its time is linear in the text: one row per corpus (set CHUNK in its profile to cut it).
#
# A vocabulary can have cases of its own after the shared ones (EXTRA, `# extra cases` in the manifest, rows numbered on
# from the shared cases): gemma4's crates/tokenizer/tests/cases-gemma4.txt, the boundaries of its own vocabulary and the
# texts the EmbeddingGemma 2 oracle sets are dumped from (tools/ref/emb2/emb2.sh reads their ids here).
#
# The sets' directories are `ref-tokenizer-*`, not the `tokenizer*` of the format this script wrote
# before it wrote a trailer: a checkout that still runs that format's gate rewrites those and cannot
# touch these.
#
# Every text's wall time is printed (`oracle: <name> ... s`), the quickest read of where a long run spends it.
#
# Every vocabulary reads the same texts: the default profile's ik tree and this tree's docs/. `all`
# takes one snapshot of docs/ per vocabulary, in turn, all but gemma4 and llama, which run by name on a machine with their
# mainline tree.
#
# Usage: bash crates/tokenizer/tools/oracle.sh v41|qwen3moe|glm5next|qwen38|nemotron|minimaxm3|musespark|mistrallarge4|nanbeige43|gemma4|llama|all [--cases-only]
# --cases-only writes the fixed cases without reading the corpus trees.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
CASES=$ROOT/crates/tokenizer/tests/cases.txt

# manifest_head — MANIFEST.tsv's lines before the rows: the executable, the library, the build, the vocabulary, the text
# tree and the cases files. Each md5 is read into a variable first (file_md5), so a failed read ends the script.
manifest_head() {
  local tok_md5 lib_md5 cases_md5 extra_md5=
  tok_md5=$(file_md5 "$TOKENIZE")
  lib_md5=$(file_md5 "$LIBLLAMA")
  cases_md5=$(file_md5 "$CASES")
  [ -z "$EXTRA" ] || extra_md5=$(file_md5 "$EXTRA")
  printf '# tokenizer\t%s\t%s\n' "$TOKENIZE" "$tok_md5"
  printf '# libllama\t%s\t%s\n' "$LIBLLAMA" "$lib_md5"
  [ -z "$BUILD_ID" ] || printf '# build\t%s\n' "$BUILD_ID"
  printf '# vocabulary\t%s\n' "$SHARD"
  printf '# text tree\t%s\n' "$IK"
  printf '# cases\t%s\t%s\n' "$CASES" "$cases_md5"
  [ -z "$EXTRA" ] || printf '# extra cases\t%s\t%s\n' "$EXTRA" "$extra_md5"
  printf 'set\tbytes\ttext_md5\tids\tnps_ids\n'
}

# row <name> — the text <name>'s manifest row: bytes, text md5 and the two id counts, each read into a variable first.
row() {
  local bytes sum ids nps
  bytes=$(wc -c < "$OUT.new/$1.txt")
  sum=$(file_md5 "$OUT.new/$1.txt")
  ids=$(wc -l < "$OUT.new/$1.ids")
  nps=$(wc -l < "$OUT.new/$1.nps.ids")
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$bytes" "$sum" "$ids" "$nps"
}

# `bash crates/tokenizer/tools/oracle.sh --self-test`: manifest_head and row on stand-in files, with md5sum and wc
# stand-ins first on PATH (tools/ref/read-stubs.sh). Whole files give their lines; a digest or count read that fails ends
# the function, under the script's set -e, before it prints the line. No box, no llama-tokenize, no corpus.
if [ "${1:-}" = --self-test ]; then
  set +e
  t=$(mktemp -d "${TMPDIR:-/tmp}/oracle-test.XXXXXX") || exit 70
  trap 'rm -rf "$t"' EXIT
  # shellcheck source=tools/ref/ref-paths.sh
  source "$ROOT/tools/ref/ref-paths.sh"
  # shellcheck source=tools/ref/read-stubs.sh
  source "$ROOT/tools/ref/read-stubs.sh"
  mkdir "$t/bin" "$t/set.new"
  read_stubs "$t/bin"
  PATH=$t/bin:$PATH
  TOKENIZE=$t/llama-tokenize LIBLLAMA=$t/libllama.so CASES=$t/cases.txt EXTRA= SHARD=$t/vocab.gguf IK=$t/ik BUILD_PIN= BUILD_ID=
  OUT=$t/set bad=0 d=0123456789abcdef0123456789abcdef tab=$'\t'
  # case_run <function> <file> <text>: <file> holds <text>, then the function (row's name x) under set -e; its status and output.
  case_run() {
    printf %s "$3" > "$2"
    got=$(set -e; "$1" x)
    echo "$?|$got"
  }
  echo t > "$TOKENIZE"
  echo l > "$LIBLLAMA"
  echo c > "$CASES"
  echo x > "$OUT.new/x.ids"
  echo x > "$OUT.new/x.nps.ids"
  r=$(case_run manifest_head "$OUT.new/x.txt" x)
  check head "${r%%$'\n'*}" "0|# tokenizer${tab}$TOKENIZE${tab}$d"
  check row "$(case_run row "$OUT.new/x.txt" x)" "0|x${tab}7${tab}$d${tab}7${tab}7"
  r=$(case_run manifest_head "$LIBLLAMA" fail-md5)
  check head-md5 "$r" "1|"
  echo l > "$LIBLLAMA"
  check row-md5 "$(case_run row "$OUT.new/x.txt" fail-md5)" "1|"
  printf %s x > "$OUT.new/x.txt"
  check row-wc "$(case_run row "$OUT.new/x.ids" fail-wc)" "1|"
  echo "oracle self-test: $([ "$bad" = 0 ] && echo ok || echo FAIL) ($bad failures)"
  [ "$bad" = 0 ]
  exit
fi

case $#:${2:-} in
  1: | 2:--cases-only) ;;
  *) echo "usage: oracle.sh VOCAB [--cases-only]" >&2; exit 64 ;;
esac

case ${1:-} in
  v41 | qwen3moe | glm5next | qwen38 | nemotron | minimaxm3 | musespark | mistrallarge4 | nanbeige43 | gemma4 | llama) ;;
  all)
    for v in v41 qwen3moe glm5next qwen38 nemotron musespark mistrallarge4 nanbeige43; do
      echo "oracle: $v"
      bash "${BASH_SOURCE[0]}" "$v" "${@:2}"
    done
    exit 0
    ;;
  *) echo "usage: oracle.sh v41|qwen3moe|glm5next|qwen38|nemotron|minimaxm3|musespark|mistrallarge4|nanbeige43|gemma4|llama|all" >&2; exit 64 ;;
esac
VOCAB_NAME=$1

# IK (the text trees) and BLOOMERY_DATA default as in every runner here. The texts are the default
# profile's: another profile's IK tree would give this vocabulary other texts than its siblings.
# shellcheck source=tools/ref/ref-paths.sh
source "$ROOT/tools/ref/ref-paths.sh"
# shellcheck source=tools/ref/emb2/oracle-tree.sh
source "$ROOT/tools/ref/emb2/oracle-tree.sh"
[ "$BLOOMERY_MODEL" = deepseek2 ] || {
  echo "oracle: the texts are the default profile's (deepseek2); BLOOMERY_MODEL=$BLOOMERY_MODEL would take another IK tree" >&2
  exit 64
}

# profile <name>: TOKENIZE (the reference executable), SHARD (the vocabulary file), SET (the
# directory under $BLOOMERY_DATA) and EXTRA (the vocabulary's own cases file, or none) of one
# vocabulary. REF_FLAGS holds the flags that make a reference tree's llama-tokenize read the text as
# ik's does: `--no-bos` where the file would prepend a BOS (the gate compares the text's own ids),
# `--no-escape` where it would turn a `\n` in the text into a newline.
EXTRA=
ORACLE_TREE='' BUILD_PIN='' BUILD_PATCH='' CHUNK=0
TREE_FLAGS=()

# The reference trees. ORACLE_TREE is the directory, BUILD_PIN the upstream commit it is built at (full hash), BUILD_PATCH the
# hash of the local patch on that commit (empty: the tree is the commit), TREE_FLAGS the flags oracle_tree takes for it.
# PIN(2026-10-18): each tree by its commit and patch, not by the md5s of one machine's binary and library.
# ik-tokref: the ik-idxkey commit db517b690 plus the unicode_tolower fix.
tree_tokref() {
  ORACLE_TREE=/home/user/ik-tokref
  BUILD_PIN=db517b690a8c0865fd8cf51565ae6b4a7295eb01 BUILD_PATCH=29cabc43ed4d
  TREE_FLAGS=(--unversioned)
}

# llama.cpp-mainline: mainline at 53ed051ce, no patch; the commit the qwen35 families (crates/refset/src/arch/qwen35) name.
# nanbeige43's tree. Its `pixtral` pre is still LLAMA3, so nemotron's is the later tree below.
tree_mainline() {
  ORACLE_TREE=/home/user/llama.cpp-mainline
  BUILD_PIN=53ed051ce5e8193652e449f43216ca3859454f49 BUILD_PATCH=
  TREE_FLAGS=()
}

profile() {
  REF_FLAGS=()
  case $1 in
    v41)
      # The V4.1 file set is the deepseek41 profile's choice, exported by tools/box.sh.
      local dir=${BLOOMERY_V41_DIR:?BLOOMERY_V41_DIR unset — run through tools/box.sh, which exports it from the deepseek41 profile}
      # ik-tilde: ik main at dad2cb3e5 plus one commit, the `~` in the S class.
      ORACLE_TREE=/home/user/ik-tilde
      BUILD_PIN=dad2cb3e55138cbdd7df988c66c0c3c54f6d34ad BUILD_PATCH=0e498d9dab2a
      TREE_FLAGS=(--unversioned)
      SHARD=$(find "$dir" -maxdepth 1 -name '*-00001-of-*.gguf' | sort | head -1)
      [ -n "$SHARD" ] || { echo "oracle: no first shard under $dir" >&2; exit 66; }
      SET=ref-tokenizer-v41
      ;;
    qwen3moe)
      tree_tokref
      SHARD=${BLOOMERY_QWEN3MOE_VOCAB:-/models/Qwen3-30B-A3B/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}
      SET=ref-tokenizer-qwen3moe
      ;;
    glm5next)
      tree_tokref
      SHARD=/models/GLM-5.3-Flash-UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
      SET=ref-tokenizer-glm5next
      ;;
    qwen38)
      tree_tokref
      SHARD=/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
      SET=ref-tokenizer-qwen38
      ;;
    nemotron)
      # ik-tokref knows the pre name and refuses the file: its loader has no
      # nemotron_h architecture. This tree's binary loads it. llama.cpp-19cce691: mainline at 19cce6919, no patch (the lfm3
      # logits oracle's tree, tools/ref/models/lfm3.sh). Its `pixtral` pre is the TEKKEN regex with ignore_merges, which
      # `Pre::Tekken` follows; mainline 53ed051ce reads the name as LLAMA3.
      ORACLE_TREE=/home/user/llama.cpp-19cce691
      BUILD_PIN=19cce6919992fdffd92e0ac152caaf1e2b8f254e BUILD_PATCH=
      TREE_FLAGS=()
      SHARD=/models/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf
      SET=ref-tokenizer-nemotron
      # The file sets add_bos_token, so llama-tokenize prepends BOS. The gate's
      # shared comparison encodes with add_special false (the text's own ids,
      # which have to decode back to the text), and --no-bos is that call:
      # tokenize.cpp passes it through as add_special. Mainline's
      # llama-tokenize also processes escapes in a -f text by default
      # (tokenize.cpp `params.escape`), so a `\n` in the code corpus would
      # reach the tokenizer as a newline; --no-escape keeps the text's bytes.
      REF_FLAGS=(--no-bos --no-escape)
      ;;
    minimaxm3)
      # llama.cpp-solar2: mainline at 6ea215d17 plus the Solar-Open2 patch (crates/refset/src/arch/solar2). Its `minimax-m2` pre
      # is the GPT4O regex list with one expression, which unicode.cpp has no hand-written splitter for, and its UTF-8 decoder
      # is the lenient one this crate's decode_lenient follows (an overlong form passes through): the mainline tree the
      # MiniMax-M3 node dumps are read by (cd0a5b6a1) rejects one, byte by byte, as U+FFFD.
      ORACLE_TREE=/home/user/llama.cpp-solar2
      # Its llama-tokenize reports `version: <n> (<commit>)`, not the `commit <hash>` the tree check reads: the age check is all there is.
      BUILD_PIN=6ea215d171fd31df943bf1ac8227129f2b963160 BUILD_PATCH=1616f7f3d842
      TREE_FLAGS=(--unversioned)
      SHARD=/models/MiniMax-M3-UD-IQ4_XS/MiniMax-M3-UD-IQ4_XS-00001-of-00006.gguf
      SET=ref-tokenizer-minimaxm3
      # Nemotron's flags, a mainline llama-tokenize too: the gate compares the text's own ids, and a `\n` in a text is two bytes.
      REF_FLAGS=(--no-bos --no-escape)
      ;;
    musespark)
      # llama.cpp-muse: mainline at PR 30616's merge (the musespark node-dump tree, crates/refset/src/arch/musespark), no patch.
      # Its `gpt-4o` pre is GPT4O's one regex under its own name — this tree's spelling of the `minimax-m2` splitter.
      # Its UTF-8 decoder is the strict one (unicode.cpp rejects an overlong form and a codepoint above U+10FFFF, byte by
      # byte as U+FFFD), which `Pre::strict_utf8` gives the `gpt-4o` row; the other trees above accept an overlong form.
      ORACLE_TREE=/home/user/llama.cpp-muse
      BUILD_PIN=ba3c05f71da85cd0439e84ba42b0478b24d5db16 BUILD_PATCH=
      TREE_FLAGS=()
      SHARD=/models/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-00001-of-00006.gguf
      SET=ref-tokenizer-musespark
      # Nemotron's flags, a mainline llama-tokenize too: the file sets add_bos_token, and the gate compares the text's own ids.
      REF_FLAGS=(--no-bos --no-escape)
      ;;
    mistrallarge4)
      # llama.cpp-ml4: mainline at PR 31376's merge (the mistrallarge4 node-dump tree, crates/refset/src/arch/mistrallarge4), no
      # patch. Its `tekken-num3` pre is the Tekken regex with digit runs grouped by three from the right, and its UTF-8 decoder
      # is the strict one the `gpt-4o` row of the Muse tree has (`Pre::strict_utf8`).
      ORACLE_TREE=/home/user/llama.cpp-ml4
      BUILD_PIN=78feec062d50ec9e8940244cac918ac4480b60cb BUILD_PATCH=
      TREE_FLAGS=()
      # The files live on the root drive; /models/Mistral-Large-4-1T-A52B-UD-IQ1_S is a symlink to their directory.
      SHARD=/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf
      SET=ref-tokenizer-mistrallarge4
      # The chat template's special tokens, digit runs 1..10 and the other cases of this vocabulary after the shared ones.
      EXTRA=$ROOT/crates/tokenizer/tests/cases-mistrallarge4.txt
      # The file sets add_bos_token (BOS is <s>), and the gate compares the text's own ids, as Muse's does.
      REF_FLAGS=(--no-bos --no-escape)
      ;;
    nanbeige43)
      # Mainline's tree (53ed051ce). The file is the nanbeige43 profile's own MODEL, not the one tools/box.sh resolved for the
      # default profile.
      tree_mainline
      # shellcheck disable=SC2016 # the child shell expands it
      SHARD=$(env -u BLOOMERY_REF_MODEL bash -c 'source "$1" && printf "%s" "$MODEL"' _ "$ROOT/tools/ref/models/nanbeige43.sh")
      SET=ref-tokenizer-nanbeige43
      EXTRA=$ROOT/crates/tokenizer/tests/cases-nanbeige43.txt
      # The file sets add_bos_token (BOS is <|im_start|>), and llama-tokenize's ids keep it: the set holds the ids a
      # /tokenize with add_special gives, and the text's own ids are the ones after that first id (no EOS: add_eos_token is
      # false). The EmbeddingGemma sets are read the same way. --no-escape keeps the text's bytes, as for nemotron.
      REF_FLAGS=(--no-escape)
      ;;
    gemma4 | llama)
      # The profile's own MODEL, not the file tools/box.sh resolved for the default profile.
      local emb2 model_profile=gemma-embedding2
      [ "$1" != llama ] || model_profile=gemma-embedding
      # shellcheck disable=SC2016 # the child shell expands them
      emb2=$(env -u BLOOMERY_REF_MODEL bash -c 'source "$1" && printf "%s\t%s\t%s\t%s" "$LCPP" "$MODEL" "${EMBED_CASES:?the profile names EMBED_CASES}" "${LCPP_PIN:?the profile names LCPP_PIN}"' _ \
        "$ROOT/tools/ref/models/$model_profile.sh")
      ORACLE_TREE=${emb2%%$'\t'*}
      emb2=${emb2#*$'\t'}
      SHARD=${emb2%%$'\t'*}
      emb2=${emb2#*$'\t'}
      SET=ref-tokenizer-$1
      EXTRA=$ROOT/crates/tokenizer/tests/${emb2%%$'\t'*}
      BUILD_PIN=${emb2#*$'\t'}
      TREE_FLAGS=(--built-by 'just build-emb2-lcpp')
      # The SPM partition walk is quadratic in a text's size (header): llama's corpora go in rows of at most 1 MB.
      [ "$1" != llama ] || CHUNK=1000000
      # No --no-bos: the EmbeddingGemma sets run these ids as the model's
      # input, BOS and EOS included, as llama-server's /tokenize with
      # add_special gives them (tools/ref/emb2/server_vectors.py).
      REF_FLAGS=(--no-escape)
      ;;
  esac
  TOKENIZE=$ORACLE_TREE/build/bin/llama-tokenize
}
profile "$VOCAB_NAME"

[ -x "$TOKENIZE" ] || { echo "oracle: no llama-tokenize at $TOKENIZE" >&2; exit 66; }
[ -f "$SHARD" ] || { echo "oracle: no vocabulary file at $SHARD" >&2; exit 66; }
# ik's binaries export libllama.so; the mainline build nemotron uses exports libllama.so.0.
LIBLLAMA=$(ldd "$TOKENIZE" | awk '$1 == "libllama.so" || $1 == "libllama.so.0" { print $3; exit }')
[ -f "$LIBLLAMA" ] || { echo "oracle: $TOKENIZE loads no libllama.so (ldd names '$LIBLLAMA')" >&2; exit 66; }
oracle_tree "${TREE_FLAGS[@]}" ${BUILD_PATCH:+--patch "$BUILD_PATCH"} oracle "$ORACLE_TREE" "$BUILD_PIN" llama-tokenize
BUILD_ID=${BUILD_PIN:0:9}${BUILD_PATCH:++$BUILD_PATCH}
if [ "$VOCAB_NAME" = v41 ]; then
  # The reference's `~` is a symbol, so `~/` is one word and one id; a tree whose `~` is in neither
  # P nor S returns [96, 17].
  probe=$(CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$SHARD" -p '~/' --ids --log-disable)
  [ "$probe" = '[71520]' ] || {
    echo "oracle: $TOKENIZE tokenizes '~/' as $probe on $SHARD, not [71520]: its \`~\` is not a symbol" >&2
    exit 65
  }
fi

OUT=$BLOOMERY_DATA/$SET
rm -rf "$OUT.new"
mkdir -p "$OUT.new/cases"

# gather <name> — engram-corpus.sh's gather(), same finds, same sort orders.
gather() {
  case $1 in
    code)
      find "$IK/src" -maxdepth 1 -name '*.cpp' | sort | xargs -d '\n' cat -- ;;
    prose)
      { find "$IK/docs" -name '*.md' | sort | xargs -d '\n' cat --; cat -- "$IK/README.md"; } ;;
    prose-all)
      find "$IK" -name '*.md' -not -path '*/.git/*' | sort | xargs -d '\n' cat -- ;;
    threads)
      find "$IK/github-data" -name '*.md' | LC_ALL=C sort | xargs -d '\n' cat -- ;;
    korean)
      find "$ROOT/docs" -name '*.md' | LC_ALL=C sort | python3 -c '
import sys
for path in sys.stdin.read().splitlines():
    t = open(path, encoding="utf-8").read()
    if sum("가" <= c <= "힣" for c in t) >= 0.2 * max(1, len(t)):
        print(path)
' | xargs -d '\n' cat -- ;;
  esac
}

# ids <text file> <out prefix> — both parse modes, one id per line, and the wall of each on stderr.
ids() {
  local t0 t1 t2
  t0=$(date +%s.%N)
  one "$1" "$2.ids"
  t1=$(date +%s.%N)
  one "$1" "$2.nps.ids" --no-parse-special
  t2=$(date +%s.%N)
  awk -v n="${2##*/}" -v b="$(wc -c < "$1")" -v a="$t0" -v c="$t1" -v d="$t2" \
    'BEGIN { printf "oracle: %s %d bytes: parse %.1f s, no-parse %.1f s\n", n, b, c - a, d - c }' >&2
}

# one <text file> <out file> [flag] — llama-tokenize --ids, reshaped to one id per line.
one() {

  CUDA_VISIBLE_DEVICES= "$TOKENIZE" -m "$SHARD" -f "$1" --ids --log-disable "${REF_FLAGS[@]}" "${@:3}" > "$2.raw"
  tr -d '[] ' < "$2.raw" | tr ',' '\n' | { grep -v '^$' || true; } > "$2"
  rm -f "$2.raw"
}

manifest_head > "$OUT.new/MANIFEST.tsv"

if [ "${2:-}" != --cases-only ]; then
  for name in code prose prose-all threads korean; do
    if [ "$CHUNK" -gt 0 ]; then
      gather "$name" | split -C "$CHUNK" -d -a 3 --additional-suffix=.txt - "$OUT.new/$name."
      pieces=("$OUT.new/$name".[0-9][0-9][0-9].txt)
      [ -e "${pieces[0]}" ] || { echo "oracle: the corpus $name is empty" >&2; exit 65; }
      for piece in "${pieces[@]}"; do
        piece=${piece##*/}
        piece=${piece%.txt}
        ids "$OUT.new/$piece.txt" "$OUT.new/$piece"
        row "$piece" >> "$OUT.new/MANIFEST.tsv"
      done
    else
      gather "$name" > "$OUT.new/$name.txt"
      ids "$OUT.new/$name.txt" "$OUT.new/$name"
      row "$name" >> "$OUT.new/MANIFEST.tsv"
    fi
  done
fi

n=0
for cases in "$CASES" ${EXTRA:+"$EXTRA"}; do
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in '#'*) continue ;; esac
    n=$((n + 1))
    name=$(printf 'cases/%02d' "$n")
    printf '%b' "$line" > "$OUT.new/$name.txt"
    ids "$OUT.new/$name.txt" "$OUT.new/$name"
    row "$name" >> "$OUT.new/MANIFEST.tsv"
  done < "$cases"
done
printf '# complete\t%s\n' "$n" >> "$OUT.new/MANIFEST.tsv"

rm -rf "$OUT"
mv "$OUT.new" "$OUT"
cat "$OUT/MANIFEST.tsv"
