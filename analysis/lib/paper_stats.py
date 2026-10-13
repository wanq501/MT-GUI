"""All paper numbers, tables and figures from results.json (exported by the console; older exports are named MTGUI_results.json)."""
import json, math, sys, collections
import numpy as np
from scipy import stats
import mtgui_stats as S

PATH = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else 'out'
d, ids, gt, tags, recs = S.load(PATH)
avail = lambda sid, c: c != 'swap' or gt[sid]['swap_applied']
cache = {}
def M(tag, sid, c):
    k = (tag, sid, c)
    if k not in cache: cache[k] = S.metrics(recs.get(k), gt[sid])
    return cache[k]
INV, PROBES, CONDS = S.INV, S.PROBES, S.CONDS
res = {'n_screens': len(ids), 'n_swap': sum(gt[s]['swap_applied'] for s in ids), 'tags': tags,
       'n_records': len(d['records']), 'n_ok': len(recs)}
# ---------------- capability on all conditions
cap = []
for tag in tags:
    for c in CONDS:
        sids = [s for s in ids if avail(s, c)]; ms = [M(tag, s, c) for s in sids]
        sm = lambda k: sum(m[k] for m in ms)
        ytrue = [gt[s]['topic'] for s in sids]; ypred = [m['topic'] or '__none__' for m in ms]
        f1s = []
        for t in S.TOPICS:
            tp = sum(a == t and b == t for a, b in zip(ytrue, ypred)); fp = sum(a != t and b == t for a, b in zip(ytrue, ypred)); fn = sum(a == t and b != t for a, b in zip(ytrue, ypred))
            if tp + fn == 0: continue
            p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn); f1s.append(2 * p * r / (p + r) if p + r else 0)
        ans = [m for m in ms if m['answered']]
        cp, cr, cf = S.prf(sm('cm'), sm('cnp'), sm('cng')); tp_, tr, tf = S.prf(sm('tm'), sm('tnp'), sm('tng'))
        cap.append(dict(model=tag, cond=c, n=len(sids), json=np.mean([m['parsed'] for m in ms]), valid_topic=np.mean([m['topic'] is not None for m in ms]),
                        acc=np.mean([m['ok'] for m in ms]), macro_f1=np.mean(f1s), cp=cp, cr=cr, cf=cf, tp=tp_, tr=tr, tf=tf,
                        tokens=np.mean([m['tokens'] for m in ans]), latency=np.median([m['latency'] for m in ans])))
res['cap'] = cap
# ---------------- sensitivity
sens = []
for tag in tags:
    pv = {}
    for c in ['repeat1'] + INV:
        sids = [s for s in ids if avail(s, c)]
        flips, jc, jt, b, cc = [], [], [], 0, 0
        for s in sids:
            o, m = M(tag, s, 'original'), M(tag, s, c)
            if o['parsed'] and m['parsed']:
                flips.append(int(o['topic'] != m['topic'])); jc.append(S.jac(o['comps'], m['comps'])); jt.append(S.jac_text(o['texts'], m['texts']))
            b += o['ok'] and not m['ok']; cc += m['ok'] and not o['ok']
        co = np.array([M(tag, s, 'original')['cf'] for s in sids]); cm_ = np.array([M(tag, s, c)['cf'] for s in sids])
        to = np.array([M(tag, s, 'original')['tf'] for s in sids]); tm_ = np.array([M(tag, s, c)['tf'] for s in sids])
        wil = lambda x, y: stats.wilcoxon(x, y, zero_method='zsplit').pvalue if np.any(x != y) else 1.0
        cliff = lambda x, y: float((np.sign(x[:, None] - y[None, :])).mean())
        # flips beyond the repeat floor: paired on screens where both comparisons exist
        fr_rep = []
        if c != 'repeat1':
            for s in sids:
                o, r, m = M(tag, s, 'original'), M(tag, s, 'repeat1'), M(tag, s, c)
                if o['parsed'] and r['parsed'] and m['parsed']: fr_rep.append((int(o['topic'] != r['topic']), int(o['topic'] != m['topic'])))
        bb = sum(1 for a, z in fr_rep if a and not z); cc2 = sum(1 for a, z in fr_rep if z and not a)
        row = dict(model=tag, cond=c, n=len(sids), acc_o=np.mean([M(tag, s, 'original')['ok'] for s in sids]), acc=np.mean([M(tag, s, c)['ok'] for s in sids]),
                   flip=np.mean(flips), deg=b, imp=cc, p=S.mcnemar(b, cc), cf_o=co.mean(), cf=cm_.mean(), p_cf=wil(co, cm_), d_cf=cliff(cm_, co), jc=np.mean(jc),
                   tf_o=to.mean(), tf=tm_.mean(), p_tf=wil(to, tm_), d_tf=cliff(tm_, to), jt=np.mean(jt), p_vs_rep=S.mcnemar(bb, cc2) if c != 'repeat1' else None)
        row['delta'] = row['acc'] - row['acc_o']
        sens.append(row)
        if c != 'repeat1': pv[c] = row['p']
    for c, p in S.holm(pv).items(): next(r for r in sens if r['model'] == tag and r['cond'] == c)['p_holm'] = p
    pv2 = {r['cond']: r['p_vs_rep'] for r in sens if r['model'] == tag and r['cond'] != 'repeat1'}
    for c, p in S.holm(pv2).items(): next(r for r in sens if r['model'] == tag and r['cond'] == c)['p_vs_rep_holm'] = p
res['sens'] = sens
rob = []
for tag in tags:
    rr = [r for r in sens if r['model'] == tag and r['cond'] in INV]
    rob.append(dict(model=tag, acc_o=np.mean([M(tag, s, 'original')['ok'] for s in ids]), ri=np.mean([r['acc'] / r['acc_o'] for r in rr]),
                    flip=np.mean([r['flip'] for r in rr]), jc=np.mean([r['jc'] for r in rr]), jt=np.mean([r['jt'] for r in rr]),
                    worst=min(rr, key=lambda r: r['delta'])['cond'], rep_flip=next(r['flip'] for r in sens if r['model'] == tag and r['cond'] == 'repeat1')))
res['rob'] = rob
# ---------------- probes
probes, bytopic = [], []
for tag in tags:
    ao = np.mean([M(tag, s, 'original')['ok'] for s in ids])
    for pr in PROBES:
        ap = np.mean([M(tag, s, pr)['ok'] for s in ids]); b = sum(M(tag, s, 'original')['ok'] and not M(tag, s, pr)['ok'] for s in ids); c = sum(M(tag, s, pr)['ok'] and not M(tag, s, 'original')['ok'] for s in ids)
        probes.append(dict(model=tag, probe=pr, acc_o=ao, acc=ap, cri=(ao - ap) / ao, deg=b, imp=c, p=S.mcnemar(b, c), json=np.mean([M(tag, s, pr)['parsed'] for s in ids])))
for t in S.TOPICS:
    sids = [s for s in ids if gt[s]['topic'] == t]
    if not sids: continue
    row = dict(topic=t, n=len(sids), acc_o=np.mean([M(tag, s, 'original')['ok'] for tag in tags for s in sids]))
    for pr in PROBES: row[pr] = np.mean([M(tag, s, pr)['ok'] for tag in tags for s in sids])
    bytopic.append(row)
res['probes'] = probes; res['bytopic'] = bytopic
# ---------------- selective prediction and voting
STR = [('conf', None), ('rep', ['original', 'repeat1']), ('mr3', ['original', 'jpeg', 'dim']), ('mr6', ['original', 'jpeg', 'dim', 'grayscale', 'bezel', 'statusbar']), ('mrall', ['original'] + INV)]
def majority(tag, s, views):
    vs = [v for v in views if avail(s, v)]; cnt = collections.Counter(M(tag, s, v)['topic'] for v in vs if M(tag, s, v)['topic'])
    if not cnt: return None, 0.0, vs
    top = cnt.most_common(); best = top[0][1]; tied = [t for t, n in top if n == best]; o = M(tag, s, 'original')['topic']
    return (o if o in tied else tied[0]), best / len(vs), vs
sel, curves, vote = [], {}, []
for tag in tags:
    for key, views in STR:
        sc, cor = [], []
        for s in ids:
            if views is None: o = M(tag, s, 'original'); pred, score = o['topic'], (o['conf'] if o['conf'] is not None else 0.0)
            else: pred, score, _ = majority(tag, s, views)
            sc.append(score); cor.append(float(pred == gt[s]['topic']))
        order = np.argsort(-np.asarray(sc), kind='stable'); cs = np.asarray(cor)[order]; risks = 1 - np.cumsum(cs) / (np.arange(len(cs)) + 1)
        acc_at = lambda cov: float(np.mean(cs[:max(1, int(round(cov * len(cs))))]))
        # AUROC of the score for separating correct from incorrect answers
        pos = [x for x, y in zip(sc, cor) if y == 1]; neg = [x for x, y in zip(sc, cor) if y == 0]
        auroc = float(stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))) if pos and neg else float('nan')
        sel.append(dict(model=tag, strat=key, views=1 if views is None else len(views), acc=float(np.mean(cor)), aurc=float(np.mean(risks)), a50=acc_at(.5), a80=acc_at(.8), a90=acc_at(.9), auroc=auroc,
                        distinct=len(set(sc))))
        curves[(tag, key)] = ((np.arange(len(cs)) + 1) / len(cs), risks)
    base = [M(tag, s, 'original')['ok'] for s in ids]
    for key, views in [('single', ['original'])] + [(k, v) for k, v in STR if v]:
        oks, cf = [], []
        for s in ids:
            pred, _, vs = majority(tag, s, views); oks.append(float(pred == gt[s]['topic']))
            cnt = collections.Counter(k for v in vs for k in M(tag, s, v)['comps']); pset = {k for k, n in cnt.items() if n * 2 >= len(vs)}
            cf.append(S.prf(len(pset & set(gt[s]['components'])), len(pset), len(gt[s]['components']))[2])
        b = sum(1 for x, y in zip(base, oks) if x and not y); c = sum(1 for x, y in zip(base, oks) if y and not x)
        vote.append(dict(model=tag, strat=key, views=len(views), acc=float(np.mean(oks)), delta=float(np.mean(oks) - np.mean(base)), deg=b, imp=c, p=S.mcnemar(b, c), cf=float(np.mean(cf))))
res['sel'] = sel; res['vote'] = vote
# ---------------- component recall pooled
crec = []
for comp in S.COMPONENTS:
    for c in [x for x in CONDS if x not in ('layout', 'textonly')]:
        tp = fn = 0
        for tag in tags:
            for s in ids:
                if not avail(s, c) or comp not in gt[s]['components']: continue
                if comp in M(tag, s, c)['comps']: tp += 1
                else: fn += 1
        if tp + fn: crec.append(dict(comp=comp, cond=c, n=tp + fn, recall=tp / (tp + fn)))
res['crec'] = crec
# ---------------- extras: confusion on original, flip targets
conf_pairs = collections.Counter(); flip_to = collections.Counter()
for tag in tags:
    for s in ids:
        o = M(tag, s, 'original')
        if o['topic'] and not o['ok']: conf_pairs[(gt[s]['topic'], o['topic'])] += 1
        for c in INV:
            if not avail(s, c): continue
            m = M(tag, s, c)
            if o['parsed'] and m['parsed'] and o['topic'] != m['topic']: flip_to[(o['topic'], m['topic'])] += 1
res['confusions'] = conf_pairs.most_common(12); res['flip_pairs'] = flip_to.most_common(12)
res['gt_stats'] = dict(comps_mean=np.mean([len(gt[s]['components']) for s in ids]), texts_median=float(np.median([len(gt[s]['texts']) for s in ids])), texts_mean=np.mean([len(gt[s]['texts']) for s in ids]), textmask_nodes=np.mean([gt[s]['textmask_nodes'] for s in ids]))
res['providers'] = {t: collections.Counter(r.get('provider') for (tg, _, _), r in recs.items() if tg == t).most_common() for t in tags}
res['cost'] = {t: sum(((r.get('usage') or {}).get('cost') or 0) for (tg, _, _), r in recs.items() if tg == t) for t in tags}
json.dump({k: v for k, v in res.items()}, open(f'{OUT}/results.json', 'w'), default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o), indent=1)
np.save(f'{OUT}/curves.npy', {f'{k[0]}|{k[1]}': (v[0], v[1]) for k, v in curves.items()}, allow_pickle=True)
# ---------------- console summary
f = lambda x: f'{x:.3f}'
print('screens', res['n_screens'], 'swap', res['n_swap'], 'records', res['n_records'], 'ok', res['n_ok'], 'gt', {k: round(v, 2) for k, v in res['gt_stats'].items()})
print('\nRQ1 original:'); [print(' ', r['model'], 'acc', f(r['acc']), 'mF1', f(r['macro_f1']), 'comp P/R/F', f(r['cp']), f(r['cr']), f(r['cf']), 'text P/R/F', f(r['tp']), f(r['tr']), f(r['tf']), 'tok', round(r['tokens']), 'lat', round(r['latency'], 2)) for r in cap if r['cond'] == 'original']
print('\nRQ2 delta_acc(pp)/flip  [* holm<.05]  vs-repeat holm:')
for tag in tags:
    print(' ', tag, ' '.join(f"{r['cond'][:5]}:{r['delta']*100:+.1f}{'*' if r.get('p_holm', 1) < .05 else ''}/{r['flip']*100:.0f}%{'^' if (r.get('p_vs_rep_holm') or 1) < .05 else ''}" for r in sens if r['model'] == tag))
print('\n comp F1 delta / text F1 delta:')
for tag in tags:
    print(' ', tag, ' '.join(f"{r['cond'][:5]}:{(r['cf']-r['cf_o']):+.3f}{'*' if r['p_cf']<.05/9 else ''}/{(r['tf']-r['tf_o']):+.3f}{'*' if r['p_tf']<.05/9 else ''}" for r in sens if r['model'] == tag))
print('\nrobustness:'); [print(' ', r['model'], 'RI', f(r['ri']), 'flip', f(r['flip']), 'rep', f(r['rep_flip']), 'jc', f(r['jc']), 'jt', f(r['jt']), 'worst', r['worst']) for r in rob]
print('\nRQ3 probes:'); [print(' ', r['model'], r['probe'], 'acc', f(r['acc']), 'CRI', f(r['cri']), 'p', f(r['p'])) for r in probes]
print(' by topic (acc_o, textmask, layout, textonly):'); [print('   ', r['topic'], r['n'], f(r['acc_o']), f(r['textmask']), f(r['layout']), f(r['textonly'])) for r in bytopic]
print('\nRQ4 selective (acc, AURC, a80, a90, AUROC, distinct):'); [print(' ', r['model'], r['strat'], f(r['acc']), f(r['aurc']), f(r['a80']), f(r['a90']), f(r['auroc']), r['distinct']) for r in sel]
print('\n voting (acc, delta, deg/imp, p, compF1):'); [print(' ', r['model'], r['strat'], f(r['acc']), f'{r["delta"]*100:+.1f}', f"{r['deg']}/{r['imp']}", f(r['p']), f(r['cf'])) for r in vote]
print('\nconfusions', res['confusions'][:8]); print('flip pairs', res['flip_pairs'][:8]); print('cost', {k: round(v, 2) for k, v in res['cost'].items()})
