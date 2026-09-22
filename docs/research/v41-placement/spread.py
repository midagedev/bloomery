import json
from math import ceil, comb
exec(open('./place.py').read().split("BW={'A6000'")[0])
GiB=1<<30; MARGIN=1*GiB
SHEXP_RES=37_601_280*36//34
EXPB={L:EXP[L] for L in range(40)}          # bytes per expert (file = KQuant resident)
def hyper(n, K=384, draw=6):
    # P(k of the 6 selected experts are among the n resident ones)
    return [comb(n,k)*comb(K-n,draw-k)/comb(K,draw) for k in range(draw+1)]
def layer_time(l, n, BWg, BWh, tb, overlap):
    e=EXPB[l]; pre=(pl[l]['res']-SHEXP_RES)/BWg
    t=0
    for k,p in enumerate(hyper(n)):
        tg=(SHEXP_RES + k*e)/BWg
        th=((6-k)*e)/BWh + (tb if n<384 else 0)
        t+=p*(max(tg,th) if overlap else tg+th)
    return pre+t
def plan_spread(cards, C, exclude=(0,1)):
    n={}; P=[]
    for lab,cap,layers,head in cards:
        d=dense(layers,head); kv=kv_alloc(C,layers)
        budget=cap-d-kv-CTX-SCR-MARGIN
        el=[l for l in layers if l not in exclude]
        for l in layers: n[l]=0
        # equal count per eligible layer
        per=int(budget//sum(EXPB[l] for l in el)); per=min(per,384)
        for l in el: n[l]=per
        used=sum(n[l]*EXPB[l] for l in layers)
        # distribute remainder one expert at a time
        rem=budget-used; i=0
        while True:
            l=el[i%len(el)]
            if n[l]<384 and rem>=EXPB[l]: n[l]+=1; rem-=EXPB[l]; i+=1
            else: break
        P.append(dict(lab=lab,dense=d,kv=kv,experts=sum(n[l]*EXPB[l] for l in layers),left=rem+MARGIN,per=[n[l] for l in layers]))
    return n,P
def plan_whole(cards, C):
    n={}; P=[]
    for lab,cap,layers,head in cards:
        d=dense(layers,head); kv=kv_alloc(C,layers)
        budget=cap-d-kv-CTX-SCR-MARGIN
        m,left=fill_experts(budget,layers); n.update(m)
        P.append(dict(lab=lab,dense=d,kv=kv,experts=sum(m[l]*EXPB[l] for l in layers),left=left+MARGIN))
    return n,P
def total(cards, n, BWg, BWh, D, overlap, tb=14e-6, cnode=0.80e-6, xfer=0.0, eng=0.0, kvbw=195e9):
    card={}
    for lab,cap,layers,head in cards:
        for l in layers: card[l]=lab
    t=sum(layer_time(l,n[l],BWg[card[l]],BWh,tb,overlap) for l in range(40))
    headcard=[c[0] for c in cards if c[3]][0]
    t+=HEAD/BWg[headcard] + (32*40+4)*cnode + kv_read(D)/kvbw + xfer + eng
    return t
a=[('A6000',A6000,range(0,40),True)]
b=[('A6000',A6000,range(0,20),False),('3090',R3090,range(20,40),True)]
H={'A6000':575e9,'3090':700e9}
for nm,cards,x in (('(a)',a,0),('(b)',b,20e-6)):
    for pname,planf in (('whole',plan_whole),('spread',plan_spread)):
        n,P=planf(cards,32768)
        hb=sum((384-n[l])*EXPB[l] for l in range(40))*6/384
        s=[]
        for D in (6,1024,4096):
            for ov in (False,True):
                t=total(cards,n,H,147.7e9,D,ov,xfer=x); s.append(f'D{D} {"R1" if ov else "ser"} {1/t:.2f}')
        print(nm,pname,'VRAMexp',f"{sum(p['experts'] for p in P)/1e9:.3f} GB",'host/token',f'{hb/1e9:.4f} GB',' | '.join(s))
        if pname=='spread':
            for p in P: print('    ',p['lab'],'dense',f"{p['dense']:,}",'kv',f"{p['kv']:,}",'experts',f"{p['experts']:,}",'per-layer',sorted(set(p['per'])),'headroom',f"{p['left']:,}")
    # bands for spread R1 & serial at 4096
    n,P=plan_spread(cards,32768)
    for BWh in (122.6e9,147.7e9,230.4e9):
        row=[]
        for g in (450e9,575e9,650e9):
            BWg={'A6000':g,'3090':g*936/768}
            row.append(f'g{g/1e9:.0f}: {1/total(cards,n,BWg,BWh,4096,True,xfer=x):5.2f}/{1/total(cards,n,BWg,BWh,4096,False,xfer=x):5.2f}')
        print(f'   spread BWh {BWh/1e9:6.1f}: '+' | '.join(row))
    n1,P1=plan_spread(cards,1048576)
    print('   spread ctx 1M R1 @4096:', f'{1/total(cards,n1,H,147.7e9,4096,True,xfer=x):.2f}', 'experts', f"{sum(p['experts'] for p in P1)/1e9:.3f} GB")
    print('   spread + engram 0.31ms R1 @4096:', f'{1/total(cards,n,H,147.7e9,4096,True,xfer=x,eng=0.31e-3):.2f}')
    print('   spread D=32768 R1:', f'{1/total(cards,n,H,147.7e9,32768,True,xfer=x):.2f}')
