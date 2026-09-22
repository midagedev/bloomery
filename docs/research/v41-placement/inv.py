import re, json, sys
from collections import defaultdict, Counter
P='../../v41-inventory.md'
rows=[]
intab=False
for line in open(P):
    if line.startswith('## tensors'): intab=True; continue
    if intab and line.startswith('## '): break
    if intab and line.startswith('| ') and not line.startswith('| block') :
        c=[x.strip() for x in line.strip().strip('|').split('|')]
        if len(c)!=6 or c[0].startswith('---'): continue
        blk,name,dims,ty,b,shard=c
        dims=[int(x) for x in dims.split('×')]
        rows.append(dict(block=None if blk in ('','-','—') else (int(blk) if blk.isdigit() else blk),name=name,dims=dims,ty=ty,bytes=int(b.replace(',','')),shard=int(shard)))
json.dump(rows,open('./inv.json','w'))
print(len(rows),'tensors')
t=Counter(r['ty'] for r in rows); print(t)
by=defaultdict(int)
for r in rows: by[r['ty']]+=r['bytes']
print({k:f'{v:,}' for k,v in by.items()}, f"total {sum(by.values()):,}")
# unique (name-suffix, dims, type)
suf=defaultdict(set)
for r in rows:
    s=re.sub(r'^blk\.\d+\.','',r['name'])
    suf[(s,r['ty'],'x'.join(map(str,r['dims'])))].add(r['block'])
for (s,ty,d),bl in sorted(suf.items()):
    bl=sorted(b for b in bl if b is not None) if any(b is not None for b in bl) else ['-']
    print(f'{s:40s} {ty:6s} {d:22s} n={len(bl):2d} blocks={bl if len(bl)<12 else str(bl[:3])+"..."+str(bl[-2:])}')
