import json
from math import ceil, floor
S=json.load(open('./roles.json'))
pl={int(k):v for k,v in S['pl'].items()}
MiB=1<<20
EXP={L:(18_247_680 if L in (0,1) else 16_773_120) for L in range(40)}   # file bytes per expert (gate+up+down)
HEAD=542_996_480
def kv_alloc(C, layers=range(40)):
    """KV bytes for ctx C (f16), for the layers in `layers` (window per layer; group caches owned by source layer)."""
    L=set(layers); b=0
    b+=sum(1 for l in L)*min(C,128)*512*2
    for src,ratio in ((2,2),(8,2),(14,2),(20,1)):
        if src in L:
            n=ceil(C/ratio); b+=n*512*2 + n*128*2 + ratio*512*4*2
    return b
def kv_read(D):
    """KV bytes read per decode token at depth D (f16)."""
    w=min(D,128); b=2*w*1024
    for l in range(2,40):
        vis = D//2 if l<20 else D
        b+=(w+min(vis,512))*1024
    b+=3*ceil(D/2)*256 + 5*D*256
    return b
for D in (6,1024,4096,32768,1048576):
    print(f'D={D:>8}: KV alloc all layers {kv_alloc(D):>14,} B   KV read/token {kv_read(D):>14,} B')
print('kv_alloc split c=20: layers0-19',f'{kv_alloc(32768,range(0,20)):,}','20-39',f'{kv_alloc(32768,range(20,40)):,}', '@1M', f'{kv_alloc(1048576,range(0,20)):,}', f'{kv_alloc(1048576,range(20,40)):,}')
CTX=512*MiB; SCR=64*MiB
A6000=(49140-548)*MiB; R3090=(24576-400)*MiB
print('usable A6000',f'{A6000:,}','3090',f'{R3090:,}')
def dense(layers, head):
    return sum(pl[l]['res'] for l in layers) + (HEAD if head else 0)
def fill_experts(budget, layers):
    """whole-expert fill, q4_K layers first (layers 0,1 = q5_K last), in layer order from the END of the range
    (any order is equivalent under uniform routing). Returns {layer: n_experts_in_vram}."""
    n={l:0 for l in layers}
    order=[l for l in sorted(layers, reverse=True) if l not in (0,1)] + [l for l in (1,0) if l in layers]
    for l in order:
        k=min(384, int(budget//EXP[l])); n[l]=k; budget-=k*EXP[l]
        if k<384: break
    return n, budget
def evaluate(name, cards, BW, BWh=147.7e9, D=4096, C=32768, cnode=0.80e-6, tb=14e-6, xfer=0.0, kvbw=195e9, eng=0.0, overlap=True, verbose=True):
    # cards: list of (label, usable, layers, head)
    vram={}; T_gpu=0; lines=[]; nexp={}
    for lab,cap,layers,head in cards:
        d=dense(layers,head); kv=kv_alloc(C,layers)
        budget=cap-d-kv-CTX-SCR
        n,left=fill_experts(budget, layers); nexp.update(n)
        ev=sum(n[l]*EXP[l] for l in layers)
        vram[lab]=dict(cap=cap,dense=d,kv=kv,ctx=CTX,scr=SCR,experts=ev,left=left,n=n)
        t_d=d/BW[lab]; t_e=ev*6/384/BW[lab]
        T_gpu+=t_d+t_e
        if verbose: lines.append(f'  {lab}: layers {min(layers)}-{max(layers)}{" +head" if head else ""}: dense {d/1e9:.3f} GB, KV@C={C} {kv/1e6:.1f} MB, ctx {CTX/1e9:.3f}, scratch {SCR/1e9:.3f}, experts {ev/1e9:.3f} GB ({sum(n.values())} experts; full layers {[l for l in layers if n[l]==384]}, partial {[ (l,n[l]) for l in layers if 0<n[l]<384]}), headroom {left/1e6:.0f} MB; t_dense {t_d*1e3:.2f} ms t_gpuexp {t_e*1e3:.2f} ms')
    host_bytes=sum((384-nexp[l])*EXP[l] for l in range(40))*6/384
    host_layers=[l for l in range(40) if nexp[l]<384]
    T_host=host_bytes/BWh
    T_b=len(host_layers)*tb
    Nn=32*40+4; T_n=Nn*cnode
    T_kv=kv_read(D)/kvbw
    # R1: shared expert + in-VRAM experts of a host layer overlap its host leg
    sav=0
    if overlap:
        for l in host_layers:
            lab=[c[0] for c in cards if l in c[2]][0]
            shexp=37_601_280*36/34
            t_g=(shexp + nexp[l]*EXP[l]*6/384)/BW[lab]
            t_h=(384-nexp[l])*EXP[l]*6/384/BWh
            sav+=min(t_g,t_h+tb)
    T=T_gpu+T_host+T_b+T_n+T_kv+xfer+eng-sav
    if verbose:
        print(name); [print(x) for x in lines]
        print(f'  host experts/token {host_bytes/1e9:.3f} GB over {len(host_layers)} host layers; T_gpu {T_gpu*1e3:.2f} T_host {T_host*1e3:.2f} T_bnd {T_b*1e3:.2f} T_node {T_n*1e3:.2f} T_kv(D={D}) {T_kv*1e3:.3f} xfer {xfer*1e6:.0f}us eng {eng*1e3:.2f} R1 saving {sav*1e3:.2f} -> {T*1e3:.2f} ms = {1/T:.2f} tok/s')
    return T, host_bytes, vram
BW={'A6000':575e9,'3090':700e9}
a=[('A6000',A6000,range(0,40),True)]
b20=[('3090',R3090,range(0,20),False),('A6000',A6000,range(20,40),True)]
b20r=[('A6000',A6000,range(0,20),False),('3090',R3090,range(20,40),True)]
b14=[('3090',R3090,range(0,14),False),('A6000',A6000,range(14,40),True)]
b8=[('3090',R3090,range(0,8),False),('A6000',A6000,range(8,40),True)]
b32=[('3090',R3090,range(0,32),False),('A6000',A6000,range(32,40),True)]
X=20e-6
evaluate('(a) A6000+DDR4',a,BW)
evaluate('(b) cut20: 3090 0-19, A6000 20-39+head',b20,BW,xfer=X)
evaluate('(b) cut20 reversed: A6000 0-19, 3090 20-39+head',b20r,BW,xfer=X)
evaluate('(b) cut14',b14,BW,xfer=X)
evaluate('(b) cut8',b8,BW,xfer=X)
evaluate('(b) cut32 (3090-heavy)',b32,BW,xfer=X)
