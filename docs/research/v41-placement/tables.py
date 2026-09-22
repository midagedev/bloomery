import json,re
from math import ceil
from collections import defaultdict
R=json.load(open('./inv.json'))
def rows_of(d):
    r=1
    for x in d[1:]: r*=x
    return r
def res(t):  # GPU resident in B1's cheapest-existing-kernel format: Q8_0 planes 36/32, bf16->f32, K-quants/f32 as file
    k=t['dims'][0]; rows=rows_of(t['dims'])
    if t['ty']=='q8_0': return rows*(k+4*k//32)
    if t['ty']=='bf16': return rows*k*4
    return t['bytes']
def role(n):
    if n.startswith('token_embd'): return None
    if n.startswith('output'): return 'head'
    s=re.sub(r'^blk\.\d+\.','',n)
    if s.startswith('engram_embd') or s.startswith('exp_probs_b_vl') or '_exps' in s: return None
    if s.startswith('engram_'): return 'engram dense'
    if s.endswith('_shexp.weight'): return 'shared experts'
    if s.startswith('hc_') or s.startswith('ffn_gate_inp') or s.startswith('exp_probs_b.') or s.startswith('ffn_norm'): return 'router+hc+ffn_norm'
    return 'attention'
def split(lo,hi,head):
    d=defaultdict(int)
    for t in R:
        r=role(t['name'])
        if r is None: continue
        if r=='head':
            if head: d[r]+=res(t)
            continue
        L=int(t['name'].split('.')[1])
        if lo<=L<hi: d[r]+=res(t)
    return d
for lab,lo,hi,head in (('(a) A6000',0,40,True),('(b) A6000 0-19',0,20,False),('(b) 3090 20-39+head',20,40,True)):
    d=split(lo,hi,head)
    print(lab,{k:f'{v:,}' for k,v in d.items()},'sum',f'{sum(d.values()):,}')
