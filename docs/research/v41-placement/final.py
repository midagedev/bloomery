exec(open('./spread.py').read().split("a=[('A6000'")[0])
a=[('A6000',A6000,range(0,40),True)]
b=[('A6000',A6000,range(0,20),False),('3090',R3090,range(20,40),True)]
BANDS={'low':{'A6000':400e9,'3090':433e9},'mid':{'A6000':575e9,'3090':700e9},'high':{'A6000':650e9,'3090':792e9}}
HOST=[('122.6',122.6e9),('147.7',147.7e9),('230.4',230.4e9)]
X=20e-6
def bprime(BWg,BWh,D,overlap,tb=14e-6,hop=15e-6,C=32768):
    # A6000 all dense + spread experts; 3090 experts only (spread over q4_K layers), fed through the host tier
    nA,PA=plan_spread(a,C)
    cap3=R3090-CTX-SCR-MARGIN
    el=[l for l in range(40) if l not in (0,1)]
    m={l:0 for l in range(40)}; per=int(cap3//sum(EXPB[l] for l in el))
    for l in el: m[l]=min(per,384-nA[l])
    t=0
    for l in range(40):
        e=EXPB[l]; pre=(pl[l]['res']-SHEXP_RES)/BWg['A6000']
        # expected selected counts (linear; host leg dominates)
        kA=6*nA[l]/384; k3=6*m[l]/384; kh=6-kA-k3
        tgA=(SHEXP_RES+kA*e)/BWg['A6000']
        tcpu=kh*e/BWh; t3=k3*e/BWg['3090']+(2*hop if m[l] else 0)
        if overlap: t+=pre+max(tgA, max(tcpu,t3)+tb)
        else: t+=pre+tgA+tcpu+t3+tb
    t+=HEAD/BWg['A6000']+(32*40+4)*0.80e-6+kv_read(D)/195e9
    hb=sum((384-nA[l]-m[l])*EXPB[l] for l in range(40))*6/384
    return t,hb,sum(m[l]*EXPB[l] for l in range(40))
print('== headline: n=1, ctx_max 32768, BWg mid (A6000 575 / 3090 700), BWh 147.7, engram 0 on step thread')
for nm,cards,x in (('(a)',a,0),('(b)',b,X)):
    n,P=plan_spread(cards,32768)
    for D in (6,1024,4096):
        ts=total(cards,n,BANDS['mid'],147.7e9,D,False,xfer=x); tr=total(cards,n,BANDS['mid'],147.7e9,D,True,xfer=x)
        print(f'{nm} D={D:5d}: serial {ts*1e3:6.2f} ms {1/ts:5.2f} tok/s | R1 {tr*1e3:6.2f} ms {1/tr:5.2f} tok/s')
for D in (6,1024,4096):
    ts,hb,e3=bprime(BANDS['mid'],147.7e9,D,False); tr,_,_=bprime(BANDS['mid'],147.7e9,D,True)
    print(f"(b') D={D:5d}: serial {ts*1e3:6.2f} ms {1/ts:5.2f} | R1 {tr*1e3:6.2f} ms {1/tr:5.2f}  (3090 experts {e3/1e9:.2f} GB, host/token {hb/1e9:.4f} GB)")
print('== bands at D=4096 (R1 / serial tok/s)')
for nm,cards,x in (('(a)',a,0),('(b)',b,X)):
    n,P=plan_spread(cards,32768)
    for hl,BWh in HOST:
        row=[f"{k}: {1/total(cards,n,BANDS[k],BWh,4096,True,xfer=x):5.2f}/{1/total(cards,n,BANDS[k],BWh,4096,False,xfer=x):5.2f}" for k in ('low','mid','high')]
        print(f'{nm} BWh {hl}: '+' | '.join(row))
for hl,BWh in HOST:
    row=[f"{k}: {1/bprime(BANDS[k],BWh,4096,True)[0]:5.2f}/{1/bprime(BANDS[k],BWh,4096,False)[0]:5.2f}" for k in ('low','mid','high')]
    print(f"(b') BWh {hl}: "+' | '.join(row))
print('== ik-like GPU 247 GB/s (roofline derived) serial, BWh 147.7:')
for nm,cards,x in (('(a)',a,0),('(b)',b,X)):
    n,P=plan_spread(cards,32768)
    print(nm, f"{1/total(cards,n,{'A6000':247e9,'3090':247e9},147.7e9,4096,False,xfer=x):.2f}")
print('== side rows (mid, 147.7, D=4096, R1)')
for nm,cards,x in (('(a)',a,0),('(b)',b,X)):
    n,P=plan_spread(cards,32768); n1,P1=plan_spread(cards,1048576)
    print(nm,'engram 0.31ms step thread:',f"{1/total(cards,n,BANDS['mid'],147.7e9,4096,True,xfer=x,eng=0.31e-3):.2f}",
          '| ctx_max 1M:',f"{1/total(cards,n1,BANDS['mid'],147.7e9,4096,True,xfer=x):.2f}",
          '| D=32768:',f"{1/total(cards,n,BANDS['mid'],147.7e9,32768,True,xfer=x):.2f}")
# whole-layer vs spread (mid,147.7,4096)
for nm,cards,x in (('(a)',a,0),('(b)',b,X)):
    nw,_=plan_whole(cards,32768); ns,_=plan_spread(cards,32768)
    print(nm,'whole ser/R1',f"{1/total(cards,nw,BANDS['mid'],147.7e9,4096,False,xfer=x):.2f}/{1/total(cards,nw,BANDS['mid'],147.7e9,4096,True,xfer=x):.2f}",
          'spread ser/R1',f"{1/total(cards,ns,BANDS['mid'],147.7e9,4096,False,xfer=x):.2f}/{1/total(cards,ns,BANDS['mid'],147.7e9,4096,True,xfer=x):.2f}")
# formats lever: Q8_0 f16 plane + bf16 router (dense bytes 7.992 GB instead of 8.577)
