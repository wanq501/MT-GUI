import os, json, numpy as np
from collections import Counter
exec(open('figs_rq.py', encoding='utf-8').read().split('# ===================== RQ1 =====================')[0])
INVR=['downscale','jpeg','blur','dim','grayscale','bezel','tilt','swap','statusbar']
IMG=['downscale','jpeg','blur','dim','grayscale','bezel','tilt','statusbar','swap']
def kappa(pairs):
    n=len(pairs); 
    if n==0: return float('nan')
    po=np.mean([a==b for a,b in pairs]); ca=Counter(a for a,_ in pairs); cb=Counter(b for _,b in pairs)
    pe=sum(ca[k]*cb.get(k,0) for k in ca)/n/n
    return (po-pe)/(1-pe) if pe<1 else 0.0
def H(labels):
    c=np.array(list(Counter(labels).values()),float); p=c/c.sum(); return float(-(p*np.log2(p)).sum())
res={}
for t in MOD:
    ss=[s for s in screens_for(t) if m(t,s,'original') and m(t,s,'original')['topic']]
    orig=[m(t,s,'original')['topic'] for s in ss]
    flips=[]; kap=[]; ffl=[]; facc=[]
    for r in INVR:
        prs=[(m(t,s,'original')['topic'], m(t,s,r)['topic']) for s in ss if avail(s,r) and m(t,s,r) and m(t,s,r)['topic']]
        flips.append(np.mean([a!=b for a,b in prs])); kap.append(kappa(prs))
    # content-normalized fusion: screenshot verdict fused with the text-only verdict (view-hierarchy text, unaffected by image relations)
    def fuse(s, cond):
        a=m(t,s,cond); b=m(t,s,'textonly')
        if not a or not a['topic']: return None
        if not b or not b['topic'] or b['topic']==a['topic']: return a['topic']
        ca=a['conf'] if a['conf'] is not None else 0; cb=b['conf'] if b['conf'] is not None else 0
        return a['topic'] if ca>=cb else b['topic']
    f0={s:fuse(s,'original') for s in ss}
    acc_img=np.mean([m(t,s,'original')['ok'] for s in ss]); acc_fus=np.mean([f0[s]==SCR[s]['topic'] for s in ss if f0[s]])
    for r in INVR:
        prs=[(f0[s], fuse(s,r)) for s in ss if avail(s,r) and f0[s] and fuse(s,r)]
        ffl.append(np.mean([a!=b for a,b in prs]))
    rep=[(m(t,s,'original')['topic'], m(t,s,'repeat1')['topic']) for s in ss if m(t,s,'repeat1') and m(t,s,'repeat1')['topic']]
    res[t]=dict(acc=acc_img, flip=float(np.mean(flips)), kappa=float(np.mean(kap)), kappa_min=float(min(kap)), H=H(orig), ntypes=len(set(orig)),
                kappa_rep=kappa(rep) if rep else float('nan'), acc_fused=acc_fus, flip_fused=float(np.mean(ffl)))
for t,v in res.items(): print(f"{t:12s} acc {100*v['acc']:5.1f} flip {100*v['flip']:5.1f} kappa {v['kappa']:.3f} (min {v['kappa_min']:.3f}) H {v['H']:.2f} types {v['ntypes']:2d} | fused: acc {100*v['acc_fused']:5.1f} flip {100*v['flip_fused']:5.1f}")
from scipy.stats import spearmanr
acc=[res[t]['acc'] for t in MOD]; st=[-res[t]['flip'] for t in MOD]; kp=[res[t]['kappa'] for t in MOD]
print('Spearman accuracy vs stability (1-flip):', spearmanr(acc,st)); print('Spearman flip-based vs kappa-based ranking:', spearmanr(st,kp))
json.dump(res, open(os.path.dirname(os.path.abspath(__file__)) + '/out/joint.json','w'), indent=1)
