import json, collections
exec(open('figs_rq.py', encoding='utf-8').read().split('# ===================== RQ1 =====================')[0])
out={}
for t in MODELS:
    c=collections.Counter()
    for (tg,s,cd),r in REC.items():
        if tg!=t or s not in SCR or SCR[s].get('extra'): continue
        x=m(tg,s,cd); c['requests']+=1; c['answered']+=bool(r.get('ok')); c['json']+=bool(x and x.get('parsed')); c['type']+=bool(x and x.get('topic')); c['conf']+=bool(x and x.get('conf') is not None)
    out[t]=dict(c)
json.dump(out,open('out/dataflow.json','w'),indent=1); print('dataflow saved')
