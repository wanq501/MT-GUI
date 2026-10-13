import os, json, numpy as np
exec(open('figs_rq.py', encoding='utf-8').read().split('# ===================== RQ1 =====================')[0])
INVR=['downscale','jpeg','blur','dim','grayscale','bezel','tilt','swap','statusbar']
def topic(t,s,c):
    a=m(t,s,c); return a['topic'] if a and a['topic'] else None
def fuse(t,s,c):
    a=m(t,s,c); b=m(t,s,'textonly')
    if not a or not a['topic']: return None
    if not b or not b['topic'] or b['topic']==a['topic']: return a['topic']
    return a['topic'] if (a['conf'] or 0)>=(b['conf'] or 0) else b['topic']
out={}
for t in MOD:
    ss=[s for s in screens_for(t) if m(t,s,'original')]
    acc_img=np.mean([topic(t,s,'original')==SCR[s]['topic'] for s in ss])          # unparsed answers count as wrong, as in the paper
    acc_fus=np.mean([fuse(t,s,'original')==SCR[s]['topic'] for s in ss])
    fl_i=[]; fl_f=[]
    for r in INVR:
        pi=[(topic(t,s,'original'),topic(t,s,r)) for s in ss if avail(s,r) and topic(t,s,'original') and topic(t,s,r)]
        pf=[(fuse(t,s,'original'),fuse(t,s,r)) for s in ss if avail(s,r) and fuse(t,s,'original') and fuse(t,s,r)]
        fl_i.append(np.mean([a!=b for a,b in pi])); fl_f.append(np.mean([a!=b for a,b in pf]))
    out[t]=dict(acc_img=acc_img, acc_fused=acc_fus, flip_img=float(np.mean(fl_i)), flip_fused=float(np.mean(fl_f)))
    print(f"{t:12s} acc img {100*acc_img:5.1f} fused {100*acc_fus:5.1f} (d {100*(acc_fus-acc_img):+5.1f}) | flip img {100*np.mean(fl_i):5.1f} fused {100*np.mean(fl_f):5.1f} (d {100*(np.mean(fl_f)-np.mean(fl_i)):+5.1f})")
print('paper baseline acc:', {t: round(100*N['base'][t]['acc'],1) for t in MOD})
j=json.load(open('out/joint.json')); 
for t in MOD: j[t].update(out[t])
json.dump(j, open('out/joint.json','w'), indent=1)
