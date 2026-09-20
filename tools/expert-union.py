#!/usr/bin/env python3
"""k개 연속 디코드 토큰이 전문가를 얼마나 공유하는가, 그리고 n-gram 추측이 얼마나 맞는가.

    tools/expert-union.py <dump-dir> [routed_MB fixed_MB]   # 덤프는 tools/ref/expert-union-dump.sh

routed_MB/fixed_MB는 레벨1 프로파일 weight MB 열의 스텝당 값(matmul_q_batch 대 나머지)이다.
주면 수용 토큰당 바이트와 손익분기 수용률(토큰/검증 스텝)을 같이 낸다.

전문가 합집합 비율 = |union(experts of k tokens)| / (k * n_used), 층별 평균.
1.0이면 공유 없음(k토큰 검증이 k배를 읽는다), 낮을수록 한 번 읽은 전문가를 여러 토큰이 쓴다.
n-gram 추측은 프롬프트 룩업: 직전 n토큰(3→1)이 앞 문맥에 나온 가장 최근 자리의 뒤 k토큰을
초안으로 내고, 탐욕 디코드의 실제 토큰열과 앞에서부터 일치한 길이를 센다. 모델 실행 없음.
"""
import glob, os, sys
from collections import defaultdict

def load(d):
    for g in sorted(glob.glob(os.path.join(d, "*.gen"))):
        base = g[:-4]
        gen = [int(t) for t in open(g).read().split("[")[1].split("]")[0].split(",")]
        prompt = [int(t) for t in open(base + ".prompt").read().strip().split(",")]
        steps = defaultdict(list)  # block -> list of id-lists, decode steps only
        for line in open(base + ".experts"):
            b, n, ids = line.rstrip("\n").split("\t")
            if n == "1":
                steps[int(b)].append([int(i) for i in ids.split(",")])
        yield prompt, gen, steps

def main():
    d = sys.argv[1]
    ks = [2, 3, 4, 6, 8]
    num = {k: 0.0 for k in ks}; den = {k: 0 for k in ks}
    acc = {k: [0, 0] for k in ks}  # tokens emitted, verify steps
    hit = [0, 0]
    seqs = 0
    for prompt, gen, steps in load(d):
        seqs += 1
        for b, rows in steps.items():
            for k in ks:
                for i in range(0, len(rows) - k + 1):
                    u = set()
                    for r in rows[i:i + k]: u.update(r)
                    num[k] += len(u) / (k * len(rows[0])); den[k] += 1
        full = prompt + gen
        for k in ks:
            draft_len = k - 1
            pos = len(prompt)
            while pos < len(full):
                ctx = full[:pos]; draft = []
                for n in (3, 2, 1):
                    if len(ctx) < n: continue
                    tail = ctx[-n:]
                    for j in range(len(ctx) - n - 1, -1, -1):
                        if ctx[j:j + n] == tail:
                            draft = ctx[j + n:j + n + draft_len]; break
                    if draft: break
                ok = 0
                for t in draft:
                    if pos + ok < len(full) and full[pos + ok] == t: ok += 1
                    else: break
                if k == ks[0]:
                    hit[1] += 1; hit[0] += 1 if draft else 0
                emitted = min(ok + 1, len(full) - pos)
                acc[k][0] += emitted; acc[k][1] += 1
                pos += emitted
    routed, fixed = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) >= 4 else (None, None)
    print(f"sequences {seqs}")
    print("k  union/(k*n_used)  tokens/verify-step  bytes/accepted-token vs k=1  break-even tokens/step")
    for k in ks:
        r = num[k] / den[k]; tps = acc[k][0] / acc[k][1]
        if routed is None:
            print(f"{k}  {r:.3f}  {tps:.3f}  -  -")
        else:
            step_bytes = fixed + routed * r * k
            print(f"{k}  {r:.3f}  {tps:.3f}  {step_bytes / tps / (fixed + routed):.3f}  {step_bytes / (fixed + routed):.3f}")
    print(f"draft available at {hit[0]}/{hit[1]} positions")

main()
