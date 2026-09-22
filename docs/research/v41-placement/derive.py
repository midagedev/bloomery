import json, re
from math import ceil
from collections import defaultdict
R=json.load(open('./inv.json'))
KQ={'q3_K':110,'q4_K':144,'q6_K':210,'q5_K':176}
def rows_of(d):
    r=1
    for x in d[1:]: r*=x
    return r
def rb_kq(ty,k): return KQ[ty]*(k//256)
# ---- section 1: resident per format
def res_current(t):
    ty=t['ty']; k=t['dims'][0]; rows=rows_of(t['dims'])
    if ty=='f32': return rows*k*4
    if ty in ('q3_K','q4_K','q6_K'):
        rb=rb_kq(ty,k); w=ceil(rb*rows/4)
        return None if w%rows else 4*w
    return None   # q8_0/bf16 Unknown(8)/Unknown(30), q5_K refused
def res_size_fn(t):  # weights.rs:239-255 as written
    ty=t['ty']; k=t['dims'][0]; rows=rows_of(t['dims'])
    if ty=='f32': return rows*k*4
    if ty in ('q3_K','q4_K','q6_K'): return rb_kq(ty,k)*ceil(rows/4)*4
    return None
def res_new(t, bf16='f32', q8='planes32', q5k='q5_1'):
    ty=t['ty']; k=t['dims'][0]; rows=rows_of(t['dims'])
    if ty=='q8_0': return rows*(k + (4 if q8=='planes32' else 2)*k//32)
    if ty=='bf16': return rows*k*(4 if bf16=='f32' else 2)
    if ty=='q5_K':
        if q5k=='q5_1': return rows*(1024*ceil(k/1024) + 8*(k//32))
        if q5k=='native': return t['bytes']
        if q5k=='f32': return rows*k*4
    return res_current(t)
tot=defaultdict(lambda:defaultdict(int))
trap=[]
for t in R:
    ty=t['ty']; a=tot[ty]; a['n']+=1; a['file']+=t['bytes']
    if ty in ('q3_K','q4_K','q6_K'):
        k=t['dims'][0]; rows=rows_of(t['dims']); rb=rb_kq(ty,k)
        if rows%4 or rb%4: trap.append((t['name'],t['dims'],ty,rb,rows))
        if res_current(t)!=res_size_fn(t): trap.append(('DISAGREE',t['name']))
    for lab,kw in [('planes32',{}),('planes16',dict(q8='planes16')),('bf16nat',dict(bf16='bf16')),('bf16f32',{}),('q5_1',{}),('q5nat',dict(q5k='native')),('q5f32',dict(q5k='f32'))]:
        a[lab]+=res_new(t,**kw)
print('SECTION1 per-type totals')
for ty,a in tot.items():
    print(ty, {k:(f'{v:,}' if isinstance(v,int) else v) for k,v in a.items()})
print('trap hits:',trap)
# engram tables separately (they never go to VRAM)
eng=[t for t in R if 'engram_embd' in t['name']]
print('engram_embd file', f"{sum(t['bytes'] for t in eng):,}")
# ---- section 3: roles
def role(n):
    if n.startswith('token_embd'): return 'token_embd'
    if n.startswith('output'): return 'head'
    s=re.sub(r'^blk\.\d+\.','',n)
    if s.startswith('engram_embd'): return 'engram_table'
    if s.startswith('engram_'): return 'engram_dense'
    if s.startswith('hc_'): return 'hc'
    if s.startswith('ffn_gate_inp') or s.startswith('exp_probs_b.'): return 'router'
    if s.startswith('exp_probs_b_vl'): return 'unused_vl'
    if s.endswith('_shexp.weight'): return 'shexp'
    if '_exps' in s: return 'routed'
    if s.startswith('ffn_norm'): return 'ffn_norm'
    if s.startswith('attn') or s.startswith('indexer'): return 'attn'
    return 'OTHER:'+s
RB=defaultdict(lambda:defaultdict(int))
for t in R:
    r=role(t['name']); x=RB[r]; x['n']+=1; x['file']+=t['bytes']
    x['res_gpu']+=res_new(t)                      # planes32, bf16->f32, q5k->q5_1
    x['res_gpu16']+=res_new(t,q8='planes16',bf16='bf16',q5k='native')
    if t['ty']=='q8_0': x['q8']+=t['bytes']
    if t['ty']=='bf16': x['bf16']+=t['bytes']
print('SECTION3 roles (whole-model bytes)')
for r,x in sorted(RB.items()):
    print(f"{r:14s} n={x['n']:4d} file={x['file']:>16,} res_gpu={x['res_gpu']:>16,} res_gpu_min={x['res_gpu16']:>16,} q8={x['q8']:>16,} bf16={x['bf16']:>14,}")
# per token
pt={}
for r,x in RB.items():
    if r=='routed': pt[r]=x['file']*6//384
    elif r=='engram_table': pt[r]=48*272
    elif r=='token_embd': pt[r]=5120*2
    elif r=='unused_vl': pt[r]=0
    else: pt[r]=x['file']
print('per-token file bytes:',{k:f'{v:,}' for k,v in pt.items()}, 'sum', f"{sum(pt.values()):,}")
dense=sum(v for k,v in pt.items() if k not in ('routed','engram_table'))
print('dense per token (file, incl token row, engram_wkv):',f'{dense:,}')
# per layer routed
per_layer=defaultdict(int)
for t in R:
    if role(t['name'])=='routed':
        L=int(t['name'].split('.')[1]); per_layer[L]+=t['bytes']
print('routed per layer file: L0',f'{per_layer[0]:,}','L2',f'{per_layer[2]:,}','per-token L0',f'{per_layer[0]*6//384:,}','L2',f'{per_layer[2]*6//384:,}')
print('expert bytes: q4K-layer',f'{per_layer[2]//384:,}','q5K-layer',f'{per_layer[0]//384:,}')
# per layer dense (attn+hc+router+ffn_norm+shexp)
pl=defaultdict(lambda:defaultdict(int))
for t in R:
    r=role(t['name'])
    if r in ('attn','hc','router','ffn_norm','shexp','engram_dense','unused_vl') and t['name'].startswith('blk.'):
        L=int(t['name'].split('.')[1]); pl[L]['file']+=t['bytes'] if r!='unused_vl' else 0; pl[L]['res']+=res_new(t) if r!='unused_vl' else 0
        pl[L]['res_vl']+=res_new(t)
for L in (0,1,2,3,8,14,20,21,24,39):
    print('layer',L,'dense file',f"{pl[L]['file']:,}",'resident(planes32,f32 router)',f"{pl[L]['res']:,}", '(+vl bias resident)',f"{pl[L]['res_vl']:,}")
json.dump({'pl':{L:dict(v) for L,v in pl.items()},'pt':pt,'RB':{k:dict(v) for k,v in RB.items()},'per_layer':per_layer},open('./roles.json','w'))
