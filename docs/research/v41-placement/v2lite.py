# V2-Lite shapes (plan.md type table + gguf RESULTS census), ne order: dims[0]=k.
from math import ceil
TS = {'q3_K':(256,110),'q4_K':(256,144),'q6_K':(256,210),'q5_0':(32,22),'q5_1':(32,24),'f32':(1,4)}
def file_bytes(ty,dims):
    b,s = TS[ty]; n=1
    for d in dims: n*=d
    return n//b*s
def rows_of(dims):
    r=1
    for d in dims[1:]: r*=d
    return r
def resident_size(ty,k,rows):   # weights.rs:239-255 verbatim
    qsw = lambda k: 256*ceil((k//32)/32)
    if ty=='f32': return rows*k*4
    if ty=='q5_0': return rows*(qsw(k)+k//32)*4
    if ty=='q5_1': return rows*(qsw(k)+2*k//32)*4
    b,s=TS[ty]
    if k%b: return None
    rb=s*(k//b)
    return rb*ceil(rows/4)*4
def upload_bytes(ty,k,rows):    # what upload_file_tensor actually allocates
    if ty in ('q3_K','q4_K','q6_K'):
        b,s=TS[ty]; rb=s*(k//b); words=ceil(rb*rows/4)
        if words % rows: return 'REFUSED(line302)'
        return words*4
    return resident_size(ty,k,rows)
T=[]
for L in range(27):
    p=f'blk.{L}.'
    T += [(p+'attn_norm','f32',[2048]),(p+'attn_q','q3_K',[2048,3072]),(p+'attn_kv_a_mqa','q3_K',[2048,576]),
          (p+'attn_kv_a_norm','f32',[512]),(p+'attn_kv_b','q3_K',[512,4096]),(p+'attn_output','q4_K',[2048,2048]),
          (p+'ffn_norm','f32',[2048])]
    if L==0:
        T += [(p+'ffn_gate','q3_K',[2048,10944]),(p+'ffn_up','q3_K',[2048,10944]),(p+'ffn_down','q5_1',[10944,2048])]
    else:
        T += [(p+'ffn_gate_inp','f32',[2048,64]),(p+'ffn_gate_exps','q3_K',[2048,1408,64]),(p+'ffn_up_exps','q3_K',[2048,1408,64]),
              (p+'ffn_down_exps','q5_0',[1408,2048,64]),(p+'ffn_gate_shexp','q3_K',[2048,2816]),(p+'ffn_up_shexp','q3_K',[2048,2816]),
              (p+'ffn_down_shexp','q4_K',[2816,2048])]
G=[('token_embd','q3_K',[2048,102400]),('output','q6_K',[2048,102400]),('output_norm','f32',[2048])]
T+=G
from collections import Counter,defaultdict
cnt=Counter(t[1] for t in T); print('counts',dict(cnt),'total',len(T))
agg=defaultdict(lambda:[0,0,0])
for n,ty,d in T:
    k=d[0]; r=rows_of(d); a=agg[ty]; a[0]+=1; a[1]+=file_bytes(ty,d); a[2]+=resident_size(ty,k,r)
    if resident_size(ty,k,r)!=upload_bytes(ty,k,r): print('DISAGREE',n,ty,d,resident_size(ty,k,r),upload_bytes(ty,k,r))
derived_per_layer = 16*512*(128//32)*36
tf=sum(a[1] for a in agg.values()); tr=sum(a[2] for a in agg.values())
for ty,a in sorted(agg.items()): print(f'{ty:5s} n={a[0]:3d} file={a[1]:>14,} resident={a[2]:>14,} ratio={a[2]/a[1]:.4f}')
print(f'derived 27 x {derived_per_layer:,} = {27*derived_per_layer:,}')
print(f'TOTAL file={tf:,} ({tf/2**30:.4f} GiB) resident={tr+27*derived_per_layer:,} ({(tr+27*derived_per_layer)/1e9:.3f} GB)')
# layer 1 + globals (A2-1 record)
l1=[t for t in T if t[0].startswith('blk.1.')]+G
s=sum(resident_size(ty,d[0],rows_of(d)) for n,ty,d in l1)+derived_per_layer
print(f'layer1+globals resident={s:,} (record 729,072,372)')
