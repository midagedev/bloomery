#!/usr/bin/env python3
"""vocab-only-gguf.py SRC DST — DST is SRC's metadata (every key but `split.*`) with no tensors.

llama-tokenize loads the whole split of a sharded file before it reads the vocabulary, so the first shard
of a model whose later shards are missing or still arriving cannot be tokenized with. DST is the same
tokenizer under one file the reference can load: the oracle runs on it with `MODEL=DST`. The script reads
DST back and fails unless every key but `split.*` is equal to SRC's.

gguf-py is the reference tree's own (`GGUF_PY`, default the box's llama.cpp-ml4):

    python3 crates/tokenizer/tools/vocab-only-gguf.py <shard 1> /tmp/vocab-only.gguf
    MODEL=/tmp/vocab-only.gguf bash crates/tokenizer/tools/kimik2-oracle.sh > crates/tokenizer/tests/kimik2-ids.tsv
"""
import os
import sys

sys.path.insert(0, os.environ.get("GGUF_PY", "/home/user/llama.cpp-ml4/gguf-py"))
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402


def kept(key):
    return not (key.startswith("GGUF.") or key.startswith("split."))


def main(src, dst):
    r = GGUFReader(src)
    if len(r.tensors) != 0:
        sys.exit("vocab-only-gguf: %s has %d tensors; it is a file with tensors, not a first shard" % (src, len(r.tensors)))
    w = GGUFWriter(dst, r.fields["general.architecture"].contents())
    n = 0
    for key, f in r.fields.items():
        if key == "general.architecture" or not kept(key):
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == GGUFValueType.ARRAY else None
        w.add_key_value(key, f.contents(), vtype, sub)
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    back = GGUFReader(dst).fields
    bad = [k for k, f in r.fields.items() if kept(k) and (k not in back or back[k].contents() != f.contents())]
    if bad:
        sys.exit("vocab-only-gguf: keys that read back different: %s" % ", ".join(bad))
    print("vocab-only-gguf: wrote %s, %d keys, every key but split.* equal to %s" % (dst, n, src))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
