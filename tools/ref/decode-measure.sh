#!/usr/bin/env bash
# 1-4 의 첫 tok/s. 박스 전역 임대 안에서, 행마다 증인을 남기고 잰다.
#
# 같은 자리에서 ik 를 한 번 더 재는 것이 이 스크립트의 절반이다. 우리 숫자만 적으면
# 비교할 것이 없고, 다른 날 다른 조건의 ik 숫자와 나란히 놓으면 그건 비교가 아니라
# 착시다. 둘 다 CPU 전용(-ngl 0, CUDA 숨김), 같은 파일, 같은 임대 안이다.
#
# CUDA 를 숨기는 이유는 오라클 덤프와 같다: 백엔드가 등록돼 있으면 ik 가 -ngl 0 에서도
# 그래프를 쪼갠다(실측 2026-09-19: splits 351 대 1).
set -euo pipefail
export BLOOMERY_DATA=${BLOOMERY_DATA:-/root/bloomery-data}
LOCK=/root/bloomery-cpu.lock
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
# 오라클과 프롬프트 파일이 공유하는 id 0 = "The capital of France is".
TOKENS=${BLOOMERY_DECODE_TOKENS:-100000,549,6077,280,7239,317}
N=${BLOOMERY_DECODE_N:-8}
BIN=${BLOOMERY_DECODE_BIN:-target/release/bloomery-decode}
IKBIN=${IKBIN:-/home/user/ik_llama.cpp/build/bin/llama-bench}
[ -x "$BIN" ] || { echo "no decode binary at $BIN — run: just build-decode" >&2; exit 2; }

witness() {
  echo "--- witness $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) ---"
  echo "loadavg: $(cat /proc/loadavg)"
  echo "pressure-cpu: $(grep '^some' /proc/pressure/cpu | head -n1)"
  echo "pressure-io: $(grep '^some' /proc/pressure/io | head -n1)"
  nvidia-smi --query-gpu=index,name,utilization.gpu,power.draw --format=csv,noheader
  echo "lock-holder-pid: $$"
}

exec 9>"$LOCK"
echo "[lease] waiting for $LOCK ..."
flock -w 1800 9 || { echo "[lease] timed out after 30 min"; exit 75; }
echo "[lease] acquired $(date -u +%H:%M:%SZ)"

witness pre-bloomery
"$BIN" -m "$MODEL" --tokens "$TOKENS" -n "$N"
witness post-bloomery

echo
echo "=== ik_llama.cpp, same file, same lease, CPU only ==="
witness pre-ik
CUDA_VISIBLE_DEVICES="" "$IKBIN" -m "$MODEL" -ngl 0 -t 32 -p 0 -n 32 -r 2
witness post-ik
