import json, random, collections, warnings, numpy as np
warnings.filterwarnings('ignore')
exec(open('figs_rq.py', encoding='utf-8').read().split('# ===================== RQ1 =====================')[0])
from scipy.stats import spearmanr, norm, binom
OUTJ={}; rng=np.random.default_rng(1)
INVR=['downscale','jpeg','blur','dim','grayscale','bezel','tilt','swap','statusbar']
def topic(r): return r.get('topic') if r and has_topic(r) else None
def conf(r):
    try: return float(r.get('conf'))
    except Exception: return None
def rawconf(r):
    try: return float((r.get('parsed') or {}).get('confidence')) if isinstance(r.get('parsed'),dict) else None
    except Exception: return None
def run(name,f):
    try: OUTJ[name]=f(); print(name,'ok')
    except Exception as e: print(name,'FAILED',repr(e)[:160])
# A1 accuracy change CIs and TOST (margin 5 pp)
def a1():
    res={}; cnt=0; tot=0
    for t in MODELS:
        for r in INVR:
            ss=[s for s in screens_for(t) if m(t,s,'original') and m(t,s,r)]
            if len(ss)<20: continue
            d=np.array([(topic(m(t,s,r))==SCR[s]['topic'])-(topic(m(t,s,'original'))==SCR[s]['topic']) for s in ss],float)
            bs=[rng.choice(d,len(d)).mean() for _ in range(2000)]
            lo90,hi90=np.percentile(bs,[5,95]); lo95,hi95=np.percentile(bs,[2.5,97.5])
            eq= lo90>-0.05 and hi90<0.05; cnt+=eq; tot+=1
            res[f'{t}|{r}']=[round(100*d.mean(),1),round(100*lo95,1),round(100*hi95,1),bool(eq)]
    return {'cells':res,'equivalent':cnt,'total':tot,'ci_min':min(v[1] for v in res.values()),'ci_max':max(v[2] for v in res.values())}
run('A1_acc_equivalence',a1)
# pair table for GEE
rows=[]
for t in MODELS:
    for s in screens_for(t):
        o=m(t,s,'original')
        if not o or not has_topic(o): continue
        for c in INVR+['repeat1']:
            f=m(t,s,c)
            if f and has_topic(f): rows.append((t,s,c,int(topic(f)!=topic(o)),int((o.get('sent_h') or 0)>1500)))
import pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
df=pd.DataFrame(rows,columns=['model','screen','cond','flip','hires']); df['rel']=(df.cond!='repeat1').astype(int)
def a2():
    out={}
    for t in MODELS:
        d=df[df.model==t]
        if d[d.rel==0].flip.sum()==0: out[t]={'or':None,'note':'zero baseline flips','excess_pp':round(100*(d[d.rel==1].flip.mean()-0),1)}; continue
        g=smf.gee('flip ~ rel',groups='screen',data=d,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
        b=g.params['rel']; se=g.bse['rel']; out[t]={'or':round(float(np.exp(b)),2),'lo':round(float(np.exp(b-1.96*se)),2),'hi':round(float(np.exp(b+1.96*se)),2),'p':float(g.pvalues['rel'])}
    return out
run('A2_per_model_or',a2)
def a3():
    d=df.copy(); g=smf.gee('flip ~ C(cond, Treatment("repeat1")) + C(model) + hires',groups='screen',data=d,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
    ors={k.split('T.')[-1].rstrip(']'):round(float(np.exp(v)),2) for k,v in g.params.items() if 'cond' in k}
    return {'relation_or_min':min(ors.values()),'relation_or_max':max(ors.values()),'hires_or':round(float(np.exp(g.params['hires'])),2),'hires_p':float(g.pvalues['hires'])}
run('A3_batch_adjusted',a3)
def a4():
    ss=N['screenspot']; ens=N['sens']; xs=[];ys=[];grp=[]
    for t in ss:
        for r in ss[t]:
            if r in ens.get(t,{}) and r in INVR: xs.append(ens[t][r]['flip']); ys.append(ss[t][r]['flip'] if isinstance(ss[t][r],dict) else ss[t][r]); grp.append(t)
    rho=spearmanr(xs,ys).correlation; xs=np.array(xs); ys=np.array(ys); grp=np.array(grp); cnt=0
    for _ in range(10000):
        yp=ys.copy()
        for gname in set(grp):
            idx=np.where(grp==gname)[0]; yp[idx]=rng.permutation(yp[idx])
        cnt+= spearmanr(xs,yp).correlation>=rho
    return {'rho':round(float(rho),2),'n':len(xs),'p_within_model':round((cnt+1)/10001,4)}
run('A4_screenspot_perm',a4)
def a5():
    res={}; sig=0; tot=0
    for t in LOCAL:
        for r in ['downscale','jpeg','blur','dim','tilt']:
            lv=N['dose'][t].get(r)
            if not lv: continue
            items=lv['levels'] if isinstance(lv,dict) and 'levels' in lv else lv
            n=np.array([x['n'] for x in items],float); f=np.array([x['flip']*x['n'] for x in items]); sc=np.arange(len(n),dtype=float)
            p=f.sum()/n.sum(); T=(sc*(f-n*p)).sum(); V=p*(1-p)*((n*sc**2).sum()-((n*sc).sum())**2/n.sum())
            z=T/np.sqrt(V) if V>0 else 0; pv=1-norm.cdf(z); res[f'{t}|{r}']=round(float(pv),4); sig+=pv<0.05; tot+=1
    return {'p':res,'significant':int(sig),'total':tot}
run('A5_trend',a5)
# per-screen scores for gate analyses
def views_for(t,s):
    o=m(t,s,'original'); V=[o]+[m(t,s,r) for r in INVR if m(t,s,r) and has_topic(m(t,s,r))]
    return o,[v for v in V if v is not None and has_topic(v)]
SC={}
for t in MODELS:
    L=[]
    for s in screens_for(t):
        o,V=views_for(t,s)
        if not o or not has_topic(o): continue
        t0=topic(o); A=[v for v in V if topic(v)==t0]; c0=conf(o)
        cA=[conf(v) for v in A if conf(v) is not None]; call=[conf(v) for v in V if conf(v) is not None]
        L.append(dict(sid=s,ok=int(t0==SCR[s]['topic']),conf=(c0 or 0)/100,cwc=(len(A)/len(V))*(np.mean(cA)/100 if cA else 0),
                      meanall=np.mean(call)/100 if call else 0,agrconf=(len(A)/len(V))*((c0 or 0)/100),agr=len(A)/len(V),true=SCR[s]['topic'],t0=t0))
    SC[t]=L
from sklearn.metrics import roc_auc_score
def a6():
    out={}
    for t in MODELS:
        L=SC[t]; y=np.array([1-x['ok'] for x in L]); res={}
        for k in ['conf','cwc','meanall','agrconf']:
            res[k]=round(roc_auc_score(1-y,[x[k] for x in L]),3)
        d=[]
        for _ in range(2000):
            i=rng.integers(0,len(L),len(L)); yy=1-y[i]
            if yy.min()==yy.max(): continue
            d.append(roc_auc_score(yy,[L[j]['cwc'] for j in i])-roc_auc_score(yy,[L[j]['conf'] for j in i]))
        res['diff_ci']=[round(float(np.percentile(d,2.5)),3),round(float(np.percentile(d,97.5)),3)]; out[t]=res
    return out
run('A6_cwc_ablation_auroc',a6)
def crc_thr(cal,alpha,key):
    n=len(cal); sc=sorted(set(x[key] for x in cal))
    wrong_above=lambda th: sum(1 for x in cal if x[key]>th and not x['ok'])
    for th in [-1]+sc:
        if (wrong_above(th)/n)*n/(n+1)+1/(n+1)<=alpha: return th
    return float('inf')
def a7():
    out={}
    for t in MODELS:
        L=SC[t]; cov={'cwc':[],'conf':[],'conf_rand':[]}; allab=0; selr=[]
        for sp in range(200):
            r2=random.Random(sp+1); idx=list(range(len(L))); r2.shuffle(idx); h=len(L)//2
            cal=[L[i] for i in idx[:h]]; te=[L[i] for i in idx[h:]]
            for key in ['cwc','conf']:
                th=crc_thr(cal,0.15,key); kept=[x for x in te if x[key]>th]; cov[key].append(len(kept)/len(te))
                if key=='cwc':
                    allab+= len(kept)==0
                    if kept: selr.append(1-np.mean([x['ok'] for x in kept]))
            jit=lambda x: x['conf']+r2.random()*1e-6
            cal2=[dict(x,cr=jit(x)) for x in cal]; te2=[dict(x,cr=jit(x)) for x in te]
            th=crc_thr(cal2,0.15,'cr'); cov['conf_rand'].append(np.mean([x['cr']>th for x in te2]))
        out[t]={'cov_cwc':round(100*np.mean(cov['cwc']),1),'cov_conf':round(100*np.mean(cov['conf']),1),'cov_conf_random_ties':round(100*np.mean(cov['conf_rand']),1),
                'all_abstain_splits':allab,'selective_risk_mean':round(100*np.mean(selr),1) if selr else None}
    return out
run('A7_gate_ties_abstain',a7)
def a8():
    out={}; target=0.25; delta=0.1
    for t in MODELS:
        L=SC[t]; covs=[]; sel=[]
        for sp in range(200):
            r2=random.Random(sp+1); idx=list(range(len(L))); r2.shuffle(idx); h=len(L)//2
            cal=[L[i] for i in idx[:h]]; te=[L[i] for i in idx[h:]]
            ths=sorted(set(x['cwc'] for x in cal),reverse=True); chosen=float('inf')
            for th in ths:  # fixed-sequence testing from strict to lenient thresholds
                k=[x for x in cal if x['cwc']>=th]; nk=len(k)
                if nk==0: continue
                err=sum(1-x['ok'] for x in k); pval=binom.cdf(err,nk,target)
                if pval<=delta: chosen=th
                else: break
            kept=[x for x in te if x['cwc']>=chosen]; covs.append(len(kept)/len(te))
            if kept: sel.append(1-np.mean([x['ok'] for x in kept]))
        out[t]={'coverage':round(100*np.mean(covs),1),'selective_risk':round(100*np.mean(sel),1) if sel else None,'share_splits_le_target':round(100*np.mean([s<=target for s in sel]),1) if sel else None}
    return out
run('A8_ltt_selective',a8)
ADJ=[('list','news'),('form','login'),('other','tutorial'),('other','list'),('bare','other'),('terms','tutorial')]
def a9():
    adj=set(ADJ)|set((b,a) for a,b in ADJ); out={}
    for t in MODELS:
        L=SC[t]; strict=np.mean([x['ok'] for x in L]); tol=np.mean([x['ok'] or (x['t0'],x['true']) in adj for x in L]); out[t]=[round(100*strict,1),round(100*tol,1)]
    return out
run('A9_adjacent_tolerant',a9)
def a10():
    out={}
    for t in MODELS:
        def jac(c):
            v=[]
            for s in screens_for(t):
                o=m(t,s,'original'); f=m(t,s,c)
                if not(o and f and o.get('parsed') and f.get('parsed')): continue
                A=set(map(str,o.get('comps') or [])); B=set(map(str,f.get('comps') or []))
                if A|B: v.append(len(A&B)/len(A|B))
            return np.mean(v) if v else None
        rep=jac('repeat1'); rel={r:jac(r) for r in INVR}
        out[t]={'repeat':round(rep,3) if rep else None,'below_repeat':sum(1 for v in rel.values() if v is not None and rep is not None and v<rep),'n_rel':sum(1 for v in rel.values() if v is not None)}
    return out
run('A10_component_flakiness',a10)
def a11():
    out={}
    for t in MODELS:
        L=SC[t]
        if len(L)<4: out[t]=None; continue
        TYPES=sorted(set(x['true'] for x in L)); fa=[];miss=[];routed=[]
        for sp in range(200):
            r2=random.Random(sp+1); idx=list(range(len(L))); r2.shuffle(idx); h=len(L)//2
            cal=[L[i] for i in idx[:h]]; te=[L[i] for i in idx[h:]]; th=crc_thr(cal,0.15,'cwc')
            kept=[x for x in te if x['cwc']>th]; routed.append(1-len(kept)/len(te))
            fa.append(np.mean([1-x['ok'] for x in kept]) if kept else 0)
            mm=[]
            for x in kept:
                exp=r2.choice([u for u in TYPES if u!=x['true']]); mm.append(x['t0']==exp)
            miss.append(np.mean(mm) if mm else 0)
        base_fa=np.mean([1-x['ok'] for x in L])
        out[t]={'false_alarm_all':round(100*base_fa,1),'false_alarm_kept':round(100*np.mean(fa),1),'missed_mismatch_kept':round(100*np.mean(miss),2),'routed':round(100*np.mean(routed),1)}
    return out
run('A11_expected_type_check',a11)
def a12():
    out={}
    for t in MODELS:
        recs=[r for (tg,s,c),r in REC.items() if tg==t]
        if not recs: out[t]={'n':0}; continue
        ts=sorted(r['ts'] for r in recs if r.get('ts'))
        rt=[((r.get('usage') or {}).get('completion_tokens_details') or {}).get('reasoning_tokens') for r in recs]
        rt=[x for x in rt if isinstance(x,(int,float))]
        gaps=[]
        for (tg,s,c),r in REC.items():
            if tg==t and c=='repeat1':
                o=REC.get((t,s,'original'))
                if o and o.get('ts') and r.get('ts'): gaps.append(abs(r['ts']-o['ts']) if isinstance(r['ts'],(int,float)) else 0)
        out[t]={'model_id':collections.Counter(r.get('model_id') for r in recs).most_common(1)[0][0],'returned':[k for k,_ in collections.Counter(r.get('model_returned') for r in recs).most_common(2)],
                'providers':[k for k,_ in collections.Counter(r.get('provider') for r in recs).most_common(3)],'first':str(ts[0])[:19] if ts else None,'last':str(ts[-1])[:19] if ts else None,
                'reasoning_tokens_median':float(np.median(rt)) if rt else 0,'repeat_gap_median_s':float(np.median(gaps)) if gaps else None}
    return out
run('A12_settings',a12)
def a13():
    out={}
    for t in MODELS:
        c=collections.Counter(); 
        for (tg,s,cd),r in REC.items():
            if tg!=t or SCR.get(s,{}).get('extra'): continue
            c['requests']+=1; c['ok']+=bool(r.get('ok')); c['json']+=bool(r.get('parsed')); c['type']+=bool(has_topic(r)); c['conf']+=rawconf(r) is not None
        out[t]=dict(c)
    return out
run('A13_dataflow',a13)
def sk(o):
    if isinstance(o,dict): return {str(k):sk(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [sk(v) for v in o]
    return o
json.dump(sk(OUTJ),open('out/revision.json','w'),indent=1,default=str); print('saved'); print('SC sizes',{t:len(SC[t]) for t in MODELS}); print('REC tags',collections.Counter(k[0] for k in REC).most_common(10))
