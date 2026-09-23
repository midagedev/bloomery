# b5plan: the decode-step node model summed by layer kind (the table in the report, section 1.3).
# Run: cd docs/research/v41-b5-plan && python3 bykind.py [depth]
import sys
sys.argv = ['x', sys.argv[1] if len(sys.argv) > 1 else '4096']
import nodes as N
D = N.D
ns = N.nodes(62.5, D)
kinds = {
 'L0 (창만, expert 전부 호스트)': [0],
 'L1 (창만, expert 전부 호스트, engram)': [1],
 'csa 소스+게이트+인덱스(2,8)': [2, 8],
 'csa 소스+engram(14)': [14],
 'csa 별칭(3-7,9-13,15-19)': [3,4,5,6,7,9,10,11,12,13,15,16,17,18,19],
 'hca 소스+인덱스(20)': [20],
 'hca top-k 소스(24,28,32,36)': [24, 28, 32, 36],
 'hca 별칭(21-23,25-27,...)': [21,22,23,25,26,27,29,30,31,33,34,35,37,38,39],
}
def lay(n):
    t = n[0].split(' ', 1)[0]
    return int(t[1:]) if t.startswith('L') and t[1:].isdigit() else None
print(f'depth {D}')
tot_n = tot_c = tot_o = 0
for k, ls in kinds.items():
    rows = [n for n in ns if lay(n) in ls]
    c = sum(n[2] for n in rows if n[3]); o = sum(n[2] for n in rows if not n[3])
    per = len(rows) / len(ls)
    print(f'{k:40s} layers {len(ls):2d}  nodes/layer {per:5.1f}  crit/layer {c/len(ls):7.1f} us  overl/layer {o/len(ls):6.1f} us  | all: nodes {len(rows):4d} crit {c/1000:6.3f} ms overl {o/1000:6.3f} ms')
    tot_n += len(rows); tot_c += c; tot_o += o
rest = [n for n in ns if lay(n) is None]
c = sum(n[2] for n in rest)
print(f'{"embed + head":40s} nodes {len(rest)}  {c/1000:.3f} ms')
print(f'total nodes {tot_n + len(rest)}  crit {(tot_c + c)/1000:.3f}  overl {tot_o/1000:.3f}  sum {(tot_c + c + tot_o)/1000:.3f} ms')
