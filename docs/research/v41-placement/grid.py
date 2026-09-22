import json
from math import ceil
exec(open('./place.py').read().split("BW={'A6000'")[0])
GiB=1<<30; MARGIN=1*GiB
SHEXP=37_601_280*36//34   # q8_0 planes read per layer
def plan(cards, C=32768):
    out=[]; nexp={}
    for lab,cap,layers,head in cards:
        d=dense(layers,head); kv=kv_alloc(C,layers)
        budget=cap-d-kv-CTX-SCR-MARGIN
        n,left=fill_experts(budget, layers); nexp.update(n)
        ev=sum(n[l]*EXP[l] for l in layers)
        out.append(dict(lab=lab,cap=cap,layers=layers,head=head,dense=d,kv=kv,experts=ev,left=left+MARGIN,n=n))
    return out,nexp
def T(cards, BWg, BWh, D, overlap, C=32768, tb=14e-6, cnode=0.80e-6, xfer=0.0, eng=0.0, kvbw=195e9, extra_remote=None):
    P,nexp=plan(cards,C)
    t=0; card_of={}
    for p in P:
        bw=BWg[p['lab']]
        t+=p['dense']/bw + p['experts']*6/384/bw
        for l in p['layers']: card_of[l]=p['lab']
    host_layers=[l for l in range(40) if nexp[l]<384]
    hb=sum((384-nexp[l])*EXP[l] for l in range(40))*6/384
    t+=hb/BWh + len(host_layers)*tb + (32*40+4)*cnode + kv_read(D)/kvbw + xfer + eng
    if overlap:
        for l in host_layers:
            tg=(SHEXP+nexp[l]*EXP[l]*6/384)/BWg[card_of[l]]
            th=(384-nexp[l])*EXP[l]*6/384/BWh
            t-=min(tg,th+tb)
    return t,hb,P
a=[('A6000',A6000,range(0,40),True)]
b=[('A6000',A6000,range(0,20),False),('3090',R3090,range(20,40),True)]
X=20e-6
for name,cards,x in (('(a)',a,0.0),('(b)',b,X)):
    P,nexp=plan(cards)
    for p in P:
        print(name,p['lab'],'dense',f"{p['dense']:,}",'kv32k',f"{p['kv']:,}",'experts',f"{p['experts']:,}",sum(p['n'].values()),'full',[l for l in p['layers'] if p['n'][l]==384],'partial',[(l,p['n'][l]) for l in p['layers'] if 0<p['n'][l]<384],'headroom',f"{p['left']:,}")
    hb=sum((384-nexp[l])*EXP[l] for l in range(40))*6/384
    vr=sum(p['experts'] for p in P)
    print(name,'VRAM experts total',f'{vr:,}','host expert bytes total',f'{258_767_585_280-vr:,}','host/token',f'{hb:,.0f}','frac GPU',f'{vr/258_767_585_280:.4f}')
    print(' headline (BWg A6000 575 / 3090 700, BWh 147.7):')
    for D in (6,1024,4096):
        for ov in (False,True):
            t,_,_=T(cards,{'A6000':575e9,'3090':700e9},147.7e9,D,ov,xfer=x)
            print(f'   D={D:5d} {"R1 " if ov else "ser"} {t*1e3:6.2f} ms {1/t:6.2f} tok/s')
    print(' bands @D=4096, R1 / serial:')
    for BWh in (122.6e9,147.7e9,230.4e9):
        row=[]
        for g in (450e9,575e9,650e9):
            BWg={'A6000':g,'3090':g*936/768}
            tr,_,_=T(cards,BWg,BWh,4096,True,xfer=x); ts,_,_=T(cards,BWg,BWh,4096,False,xfer=x)
            row.append(f'g{g/1e9:.0f}: {1/tr:5.2f}/{1/ts:5.2f}')
        print(f'   BWh {BWh/1e9:6.1f}: '+' | '.join(row))
    # engram on step thread
    t,_,_=T(cards,{'A6000':575e9,'3090':700e9},147.7e9,4096,True,xfer=x,eng=0.31e-3)
    print(' with engram 0.31 ms on step thread @4096 R1:',f'{1/t:.2f}')
    t,_,_=T(cards,{'A6000':575e9,'3090':700e9},147.7e9,4096,True,xfer=x,C=1048576)
    print(' ctx_max 1M @4096 R1:',f'{1/t:.2f}')
    t,_,_=T(cards,{'A6000':247e9,'3090':247e9*936/768},147.7e9,4096,False,xfer=x)
    print(' ik-like GPU 247 GB/s, serial:',f'{1/t:.2f}')
# ms per GB of VRAM moved to experts
print('ms/token per GB VRAM->experts @147.7:', 1e9*6/384/147.7e9*1e3)
print('---debug')
for C in (32768,1048576):
    for cards,nm in ((a,'a'),(b,'b')):
        P,nexp=plan(cards,C)
        t,hb,_=T(cards,{'A6000':575e9,'3090':700e9},147.7e9,4096,True,C=C,xfer=X if nm=='b' else 0)
        print(nm,C,[ (p['lab'],p['kv'],p['experts'],[(l,p['n'][l]) for l in p['layers'] if 0<p['n'][l]<384]) for p in P],'host/token',f'{hb:,.0f}',f'{t*1e3:.3f} ms')
