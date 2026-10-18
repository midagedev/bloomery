#!/bin/sh
# The ldd of the check scripts' PATH (tools/check-oracle-tree.sh, tools/check-emb2.sh), where oracle_tree
# (tools/ref/emb2/oracle-tree.sh) finds the library a stub binary loads. A stub is a shell script, which the real ldd refuses:
# it is answered with the libllama.so beside it, or with what the shell snippet `<stub>.ldd` does when that file exists. An ELF
# goes to the real ldd.
if [ -f "$1.ldd" ]; then
  . "$1.ldd"
elif [ "$(head -c 2 "$1")" = '#!' ]; then
  printf '\tlibllama.so => %s/libllama.so (0x0)\n' "$(dirname "$1")"
else
  exec /usr/bin/ldd "$@"
fi
