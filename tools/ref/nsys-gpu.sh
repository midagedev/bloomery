#!/usr/bin/env bash
# 커널 타임라인: 스텝의 시간이 어느 커널에 가는가를 Nsight Systems에 묻는다 (박스에서 실행).
#
#   BLOOMERY_NSYS_DEPTHS="6 4096" bash tools/ref/nsys-gpu.sh
#
# 러너 형제 넷의 자리: depth-gpu.sh가 "얼마나 걸리는가"(기록), generate --ab의 프로브 팔이
# "그 일이 얼마나 비싼가", 이 러너가 "어느 커널이 그 시간을 먹는가", ncu-gpu.sh가 "그 커널은 왜
# 비싼가"다. ncu는 커널을 직렬화하고 클럭을 고정하므로 시간 분포를 묻는 계기가 아니다 — 그 질문은
# 여기서 답한다. nsys는 실제 실행을 그대로 두고 커널 시작·끝만 찍는다(그래프 노드 단위).
#
# **스텝 경계는 재생 횟수로 자른다.** generate는 프롬프트 P토큰을 `step(&tokens)` 하나로 먹이지만
# 그 안에서 토큰마다 그래프를 한 번씩 재생한다(model.rs `step`의 토큰 루프). 그러니 프로파일 안의
# 재생은 P + (n-1)개이고, 마지막 재생이 깊이 P + n - 2의 디코드 스텝이다. "몇 개를 건너뛴다"를
# 런치 수로 세면 틀린다 — 2026-09-22 실측: ncu가 54런치(= 스텝 하나라고 믿은 값)를 건너뛰고 잡은
# "깊이 4096"은 깊이 2였고, 그리드(캐시 높이에서 나온다)는 깊이 0부터 528이라 증거가 못 됐다.
# 이 러너는 재생마다 정확히 한 번 나오는 커널을 찾아 그것으로 경계를 긋고, 그 경계가 P + n - 1개인지
# 스스로 확인한다. 아니면 표를 내지 않는다.
#
# **여기서 나온 µs는 커널 시간이지 스텝 시간이 아니다.** 창 안 커널 합과 창의 벽시계(첫 시작~끝)를
# 같이 찍는다 — 둘의 차가 런치 사이 빈틈이고, 창 벽시계가 depth-gpu.sh의 스텝 ms와 맞아야 이 표를
# 그 기록 옆에 놓을 수 있다.
set -uo pipefail
MODEL=${BLOOMERY_REF_MODEL:-/models/small/DeepSeek-V2-Lite-Chat.Q3_K_M.gguf}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}
DEPTHS=${BLOOMERY_NSYS_DEPTHS:-6 4096}
MODE=${BLOOMERY_NSYS_MODE:-graph}
# 디코드 스텝 수. 마지막 재생을 표로 내고, 그 앞 재생을 대조로 낸다.
NGEN=${BLOOMERY_NSYS_N:-4}
OUTDIR=${BLOOMERY_NSYS_OUT:-/root/bloomery-data/nsys}
TOP=${BLOOMERY_NSYS_TOP:-24}
LOCK=/root/bloomery-cpu.lock
# 카드 핀·증인 줄·바이너리 신선도는 러너 넷이 같은 파일에서 읽는다.
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"

assert_fresh_binary "$BIN" || exit $?
[ -x "$NSYS" ] || { echo "no nsys at $NSYS" >&2; exit 2; }
mkdir -p "$OUTDIR"

witness() {
  echo "--- witness $1 $(now)"
  witness_card
}

# depth-gpu.sh·ncu-gpu.sh와 같은 LCG 수열.
prompt() {
  awk -v n="$1" 'BEGIN{s=12345; printf "100000"; for(i=1;i<n;i++){s=(s*1103515245+12345)%2147483648; printf ",%d", 1000+(s%90000)}}'
}

exec 9>"$LOCK"
echo "[lease] waiting for $LOCK ..."
flock -w 1800 9 || { echo "[lease] timed out after 30 min"; exit 75; }
echo "[lease] held by pid $$ at $(now)"
echo "[config] nsys=$($NSYS --version) mode=$MODE n=$NGEN depths='$DEPTHS' out=$OUTDIR"
witness pre

rc_all=0
for d in $DEPTHS; do
  ctx=$((d + 128))
  out="$OUTDIR/nsys-d${d}-${MODE}-$(date -u +%H%M%S)"
  echo
  echo "=== 깊이 $d (ctx $ctx, mode $MODE, 재생 $((d + NGEN - 1))개 기대) → $out.nsys-rep"
  witness "pre d=$d"
  "$NSYS" profile -t cuda --cuda-graph-trace=node --cuda-event-trace=false \
          -o "$out" --force-overwrite true \
          "$BIN" --tokens "$(prompt "$d")" -n "$NGEN" --ctx "$ctx" --mode "$MODE" \
          > "$out.txt" 2>&1
  rc=$?
  witness "post d=$d"
  echo "[rc] $rc"
  [ $rc -eq 0 ] || { rc_all=$rc; echo "--- 마지막 20줄"; tail -n 20 "$out.txt"; continue; }
  "$NSYS" export --type sqlite --force-overwrite true -o "$out.sqlite" "$out.nsys-rep" > "$out.export.txt" 2>&1 \
    || { rc_all=$?; echo "[export] 실패"; tail -n 10 "$out.export.txt"; continue; }
  python3 - "$out.sqlite" "$d" "$NGEN" "$TOP" <<'PY'
import sqlite3, sys
db, depth, ngen, top = sqlite3.connect(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
names = dict(db.execute("SELECT id, value FROM StringIds"))
rows = db.execute("SELECT start, end, shortName, gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
ks = [(s, e, names.get(n, str(n)).split("(")[0], (gx, gy, gz)) for s, e, n, gx, gy, gz in rows]
print(f"    커널 런치 전체 {len(ks)}개")
replays = depth + ngen - 1
from collections import Counter
cnt = Counter(k[2] for k in ks)
# 재생마다 정확히 한 번 나오는 커널이 경계다. 여럿이면 가장 먼저 나오는 것.
cands = [n for n, c in cnt.items() if c == replays]
if not cands:
    print(f"    경계 없음: 재생 {replays}회에 맞는 커널이 없다. 이름별 횟수 상위:")
    for n, c in cnt.most_common(12):
        print(f"      {c:7d}  {n}")
    raise SystemExit(3)
first_idx = {n: next(i for i, k in enumerate(ks) if k[2] == n) for n in cands}
marker = min(cands, key=first_idx.get)
bounds = [i for i, k in enumerate(ks) if k[2] == marker]
print(f"    경계 커널 {marker} (재생당 1회, {len(bounds)}회 — 기대 {replays}) 후보 {len(cands)}개")
bounds.append(len(ks))
def window(r):
    # 마지막 재생 뒤에는 헤드(argmax)가 붙는다 — 그래프 밖이지만 스텝의 일부이므로 같이 센다.
    return ks[bounds[r]:bounds[r + 1]]
def period(r):
    # 이 재생 시작에서 다음 재생 시작까지. 디코드 스텝이면 호스트 턴어라운드(sync·복사)까지 든
    # 값이라 depth-gpu.sh의 스텝 ms와 맞아야 하는 쪽은 이것이다. 마지막 재생에는 다음이 없다.
    return (ks[bounds[r + 1]][0] - ks[bounds[r]][0]) / 1e3 if bounds[r + 1] < len(ks) else float("nan")
def report(label, w, per):
    wall = (w[-1][1] - w[0][0]) / 1e3
    tot = sum(e - s for s, e, _, _ in w) / 1e3
    print(f"  [{label}] 커널 {len(w)}개  다음 재생까지 {per:9.1f} µs  첫~끝 {wall:9.1f} µs  커널 합 {tot:9.1f} µs  빈틈 {wall - tot:8.1f} µs")
    by = {}
    for s, e, n, g in w:
        d_ = by.setdefault(n, [0.0, 0, set()])
        d_[0] += (e - s) / 1e3; d_[1] += 1; d_[2].add(g)
    print(f"    {'커널':40s} {'런치':>5s} {'합 µs':>10s} {'몫':>7s} {'평균 µs':>9s}  그리드")
    for n, (s, c, g) in sorted(by.items(), key=lambda x: -x[1][0])[:top]:
        gs = " ".join("x".join(map(str, t)) for t in sorted(g)[:3])
        print(f"    {n[:40]:40s} {c:5d} {s:10.1f} {100 * s / (tot or 1):6.1f}% {s / c:9.2f}  {gs}")
# 프롬프트 첫 재생(깊이 0), 마지막 앞 재생, 마지막 재생. 처음과 끝의 차가 깊이 비용이다.
report(f"재생 0 = 깊이 0 (프롬프트 첫 토큰)", window(0), period(0))
report(f"재생 {replays - 2} = 깊이 {depth + ngen - 3}", window(replays - 2), period(replays - 2))
report(f"재생 {replays - 1} = 깊이 {depth + ngen - 2} (마지막 디코드)", window(replays - 1), period(replays - 1))
PY
  echo "--- 원본: $out.nsys-rep, $out.sqlite"
done

witness post
echo "[lease] released at $(now)"
exit $rc_all
