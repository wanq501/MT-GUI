"""All numbers for the manuscript, computed from the final console export (results.json).
Outputs: out/numbers.json (every figure quoted in the text), out/tables/*.tex, out/cache.pkl (per-record metrics)."""
import json, re, sys, os, math, random, pickle, collections
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import mtgui_stats as S

ROOT = os.path.dirname(os.path.abspath(__file__))
import gzip
_p = f'{ROOT}/data/MTGUI_results.json'
D = json.load(open(_p, encoding='utf-8')) if os.path.exists(_p) else json.load(gzip.open(_p + '.gz', 'rt', encoding='utf-8'))
os.makedirs(f'{ROOT}/out/tables', exist_ok=True)
SCR = D['screens']; IDS = D['manifest']['ids']; EXTRA = [s for s, v in SCR.items() if v.get('extra')]
MODELS = ['gpt5', 'gpt5mini', 'haiku', 'gemini', 'qwen3vl', 'qwen25vl7b', 'qwen3vl8b', 'internvl3']
NAME = {'gpt5': 'GPT-5', 'gpt5mini': 'GPT-5 mini', 'haiku': 'Claude Haiku 4.5', 'gemini': 'Gemini 2.5 Flash', 'qwen3vl': 'Qwen3-VL-235B', 'qwen25vl7b': 'Qwen2.5-VL-7B', 'qwen3vl8b': 'Qwen3-VL-8B', 'internvl3': 'InternVL3-8B'}
LOCAL = ['qwen25vl7b', 'qwen3vl8b', 'internvl3']; CLOSED = ['gpt5', 'gpt5mini', 'haiku', 'gemini']
INV = ['downscale', 'jpeg', 'blur', 'dim', 'grayscale', 'bezel', 'tilt', 'swap', 'statusbar']
INV_ZH = {'downscale': 'Downscale ½', 'jpeg': 'JPEG q15', 'blur': 'Blur σ1.2', 'dim': 'Dim ×0.6', 'grayscale': 'Grayscale', 'bezel': 'Bezel', 'tilt': 'Tilt 3°', 'swap': 'Swap items', 'statusbar': 'Status bar'}
PROBES = ['textmask', 'layout', 'textonly']
DOSE = {'downscale': ['downscale75', 'downscale', 'downscale25'], 'jpeg': ['jpeg50', 'jpeg', 'jpeg5'], 'blur': ['blur06', 'blur', 'blur24'], 'dim': ['dim08', 'dim', 'dim04'], 'tilt': ['tilt1', 'tilt', 'tilt6']}
DOSE_LVL = {'downscale': ['75%', '50%', '25%'], 'jpeg': ['q50', 'q15', 'q5'], 'blur': ['σ0.6', 'σ1.2', 'σ2.4'], 'dim': ['×0.8', '×0.6', '×0.4'], 'tilt': ['1°', '3°', '6°']}
SAMPLES = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']; REPEATS = ['repeat1', 'repeat2', 'repeat3']
SS_CONDS = ['repeat1', 'downscale', 'jpeg', 'blur', 'dim', 'grayscale', 'bezel', 'tilt', 'statusbar']
for sid in IDS: SCR[sid]['text_count'] = len([y for y in (S.norm_text(x) for x in SCR[sid]['texts'] if isinstance(x, str) and x.strip()) if y])
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- per-record metrics (with topic recovery from truncated JSON)
REC = {(r['tag'], r['sid'], r['cond']): r for r in D['records'] if r.get('ok')}
def metric(tag, sid, cond):
    r = REC.get((tag, sid, cond)); g = SCR[sid]
    if g.get('extra'): g = dict(g, components=[], texts=[], text_count=0, topic=None)
    m = S.metrics(r, g) if r else S.metrics(None, g)
    m['recovered'] = False
    if r and not m['parsed']:   # output cut off by the token limit: the topic field comes first and is intact
        mm = re.search(r'"topic"\s*:\s*"([^"]+)"', r.get('raw') or '')
        if mm:
            m['topic'] = S.norm_topic(mm.group(1)); m['ok'] = int(m['topic'] == g['topic']) if not SCR[sid].get('extra') else 0; m['recovered'] = True
            cm = re.search(r'"confidence"\s*:\s*(\d+(?:\.\d+)?)', r.get('raw') or '')
            m['conf'] = min(100.0, float(cm.group(1))) if cm else None
    m['provider'] = r.get('provider') if r else None
    m['sent_h'] = r.get('sent_h') if r else None
    return m
cache = f'{ROOT}/out/cache.pkl'
if os.path.exists(cache): M = pickle.load(open(cache, 'rb'))
else:
    M = {}
    for (tag, sid, cond) in REC: M[(tag, sid, cond)] = metric(tag, sid, cond)
    pickle.dump(M, open(cache, 'wb'))
def m(tag, sid, cond): return M.get((tag, sid, cond))
def has_topic(x): return x is not None and x['topic'] is not None
def screens_for(tag): return [s for s in IDS if m(tag, s, 'original')]   # InternVL3 covers 221 screens at the data freeze
def avail(sid, cond): return cond != 'swap' or SCR[sid].get('swap_applied')
def same_prov(tag, sid, a, b):   # for API-served open models, keep pairs answered by the same inference provider
    if tag != 'qwen3vl': return True
    x, y = m(tag, sid, a), m(tag, sid, b); return x and y and x['provider'] == y['provider']

def boot_ci(vals, f=np.mean, B=2000):
    v = np.asarray(vals, dtype=float)
    if not len(v): return (float('nan'), float('nan'))
    idx = rng.integers(0, len(v), size=(B, len(v))); bs = np.array([f(v[i]) for i in idx])
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
def mcnemar_midp(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = 2 * stats.binom.cdf(k, n, 0.5) - stats.binom.pmf(k, n, 0.5)
    return float(min(1.0, p))
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order): run = max(run, (len(ps) - rank) * ps[i]); out[i] = min(1.0, run)
    return out
def macro_f1(tag, cond):
    ids = [s for s in screens_for(tag) if avail(s, cond)]
    f1 = []
    for t in S.TOPICS:
        tp = sum(1 for s in ids if SCR[s]['topic'] == t and (m(tag, s, cond) or {}).get('topic') == t)
        fp = sum(1 for s in ids if SCR[s]['topic'] != t and (m(tag, s, cond) or {}).get('topic') == t)
        fn = sum(1 for s in ids if SCR[s]['topic'] == t and (m(tag, s, cond) or {}).get('topic') != t)
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0; f1.append(2 * p * r / (p + r) if p + r else 0)
    return float(np.mean(f1))
N = {'models': MODELS, 'name': NAME, 'n_screens': len(IDS), 'n_extra': len(EXTRA), 'n_records': len(D['records']), 'n_ok': len(REC)}

# ---------------------------------------------------------------- RQ1 baseline on the original screenshot
base = {}
for tag in MODELS:
    ids = screens_for(tag); mm = [m(tag, s, 'original') for s in ids]
    oks = [x['ok'] for x in mm]; parsed = [x['parsed'] for x in mm]; rec = [x['recovered'] for x in mm]
    cost = sum(((REC[(tag, s, 'original')].get('usage') or {}).get('cost') or 0) for s in ids)
    lat = [float(REC[(tag, s, 'original')].get('latency') or 0) for s in ids]
    allc = [c for c in ['original'] + INV + ['repeat1'] + PROBES]
    off = sum(1 for s in ids for c in allc if m(tag, s, c) and m(tag, s, c)['parsed'] and m(tag, s, c)['topic'] is None)
    tot = sum(1 for s in ids for c in allc if m(tag, s, c) and m(tag, s, c)['parsed'])
    base[tag] = {'n': len(ids), 'acc': float(np.mean(oks)), 'acc_ci': boot_ci(oks), 'macro_f1': macro_f1(tag, 'original'), 'json_rate': float(np.mean(parsed)), 'recovered': int(sum(rec)),
                 'comp_p': float(np.mean([S.prf(x['cm'], x['cnp'], x['cng'])[0] for x in mm if x['parsed']])), 'comp_r': float(np.mean([S.prf(x['cm'], x['cnp'], x['cng'])[1] for x in mm if x['parsed']])), 'comp_f1': float(np.mean([x['cf'] for x in mm if x['parsed']])),
                 'text_p': float(np.mean([S.prf(x['tm'], x['tnp'], x['tng'])[0] for x in mm if x['parsed']])), 'text_r': float(np.mean([S.prf(x['tm'], x['tnp'], x['tng'])[1] for x in mm if x['parsed']])), 'text_f1': float(np.mean([x['tf'] for x in mm if x['parsed']])),
                 'latency_med': float(np.median(lat)), 'cost_per_1k': 1000 * cost / len(ids), 'off_label_rate': off / tot if tot else 0, 'conf_mean': float(np.mean([x['conf'] for x in mm if x['conf'] is not None]))}
N['base'] = base
# per-topic accuracy on the original (pooled over models)
N['topic_acc'] = {t: float(np.mean([m(tag, s, 'original')['ok'] for tag in MODELS for s in screens_for(tag) if SCR[s]['topic'] == t])) for t in S.TOPICS}
N['topic_n'] = {t: sum(1 for s in IDS if SCR[s]['topic'] == t) for t in S.TOPICS}

# ---------------------------------------------------------------- RQ2 invariance relations
def pairs(tag, cond, ids=None, ref='original'):
    out = []
    for s in (ids or screens_for(tag)):
        if not avail(s, cond) or not same_prov(tag, s, ref, cond): continue
        a, b = m(tag, s, ref), m(tag, s, cond)
        if has_topic(a) and has_topic(b): out.append((s, a, b))
    return out
sens = {}
for tag in MODELS:
    rows = {}; ps_acc = []; ps_flip = []
    rep = {s: (a['topic'] != b['topic']) for s, a, b in pairs(tag, 'repeat1')}
    for cond in INV + ['repeat1']:
        pr = pairs(tag, cond)
        flips = [a['topic'] != b['topic'] for s, a, b in pr]
        acc_o = [a['ok'] for s, a, b in pr]; acc_c = [b['ok'] for s, a, b in pr]
        b_ = sum(1 for s, a, b in pr if a['ok'] and not b['ok']); c_ = sum(1 for s, a, b in pr if b['ok'] and not a['ok'])
        # flips beyond the repeat floor: discordant screens (relation flips, repeat does not) against the reverse
        fb = sum(1 for s, a, b in pr if (a['topic'] != b['topic']) and s in rep and not rep[s]); fc = sum(1 for s, a, b in pr if (a['topic'] == b['topic']) and s in rep and rep[s])
        row = {'n': len(pr), 'acc': float(np.mean(acc_c)) if pr else float('nan'), 'd_acc': float(np.mean(acc_c) - np.mean(acc_o)) if pr else float('nan'), 'p_acc': mcnemar_midp(b_, c_), 'degraded': b_, 'improved': c_,
               'flip': float(np.mean(flips)) if pr else float('nan'), 'flip_ci': boot_ci(flips), 'p_floor': mcnemar_midp(fb, fc) if cond != 'repeat1' else None,
               'comp_jac': float(np.mean([S.jac(a['comps'], b['comps']) for s, a, b in pr if a['parsed'] and b['parsed']] or [float('nan')])),
               'text_jac': float(np.mean([S.jac_text(a['texts'], b['texts']) for s, a, b in pr if a['parsed'] and b['parsed']] or [float('nan')]))}
        rows[cond] = row
        if cond != 'repeat1': ps_acc.append(row['p_acc']); ps_flip.append(row['p_floor'])
    ha, hf = holm(ps_acc), holm(ps_flip)
    for i, cond in enumerate(INV): rows[cond]['p_acc_holm'] = ha[i]; rows[cond]['p_floor_holm'] = hf[i]
    fl = [rows[c]['flip'] for c in INV]
    rows['summary'] = {'mean_flip': float(np.mean(fl)), 'max_flip': float(np.max(fl)), 'max_rel': INV[int(np.argmax(fl))], 'min_flip': float(np.min(fl)), 'floor': rows['repeat1']['flip'],
                       'above_floor': sum(1 for c in INV if rows[c]['p_floor_holm'] < 0.05), 'sig_acc': sum(1 for c in INV if rows[c]['p_acc_holm'] < 0.05), 'max_abs_dacc': float(np.max(np.abs([rows[c]['d_acc'] for c in INV]))),
                       'robustness_index': float(1 - np.mean(fl)), 'min_acc_p_holm': float(np.min([rows[c]['p_acc_holm'] for c in INV]))}
    sens[tag] = rows
N['sens'] = sens
# provider effect for the API-served open-weight model (all pairs, not only same-provider)
pv = {'same': [], 'diff': []}
for s in screens_for('qwen3vl'):
    a = m('qwen3vl', s, 'original')
    for cond in INV:
        b = m('qwen3vl', s, cond)
        if avail(s, cond) and has_topic(a) and has_topic(b): pv['same' if a['provider'] == b['provider'] else 'diff'].append(a['topic'] != b['topic'])
N['provider'] = {'same_n': len(pv['same']), 'same_flip': float(np.mean(pv['same'])), 'diff_n': len(pv['diff']), 'diff_flip': float(np.mean(pv['diff'])),
                 'share': {k: v / sum(1 for r in REC.values() if r['tag'] == 'qwen3vl') for k, v in collections.Counter(r.get('provider') for r in REC.values() if r['tag'] == 'qwen3vl').items()}}
# status-bar crop depth actually applied (rows removed / image height)
sb = collections.Counter()
for (tag, sid, cond), r in REC.items():
    if cond == 'statusbar' and not SCR[sid].get('extra'):
        o = REC.get((tag, sid, 'original'))
        if o and o.get('sent_h') and r.get('sent_h'): sb[(tag, round(100 * (o['sent_h'] - r['sent_h']) / o['sent_h'], 1))] += 1
N['statusbar_depth'] = {f'{t}|{d}': n for (t, d), n in sb.items()}

# ---------------------------------------------------------------- dose, compound, prompt relations, sampling (local models)
dose = {}
for tag in LOCAL:
    dose[tag] = {}
    for fam, conds in DOSE.items():
        levels = []
        for cond in conds:
            pr = pairs(tag, cond); flips = [a['topic'] != b['topic'] for s, a, b in pr]
            levels.append({'cond': cond, 'n': len(pr), 'flip': float(np.mean(flips)), 'flip_ci': boot_ci(flips), 'acc': float(np.mean([b['ok'] for s, a, b in pr])), 'd_acc': float(np.mean([b['ok'] - a['ok'] for s, a, b in pr]))})
        # Cochran's Q over the three levels (screens paired)
        ids = [s for s in screens_for(tag) if all(has_topic(m(tag, s, c)) for c in conds) and has_topic(m(tag, s, 'original'))]
        X = np.array([[int(m(tag, s, c)['topic'] != m(tag, s, 'original')['topic']) for c in conds] for s in ids])
        k = X.shape[1]; Tj = X.sum(0); Li = X.sum(1); Q = (k - 1) * (k * (Tj ** 2).sum() - Tj.sum() ** 2) / max(1e-9, (k * Li.sum() - (Li ** 2).sum()))
        dose[tag][fam] = {'levels': levels, 'q': float(Q), 'p_q': float(1 - stats.chi2.cdf(Q, k - 1)), 'monotone': bool(levels[0]['flip'] <= levels[1]['flip'] <= levels[2]['flip'])}
    extra = {}
    for cond in ['pipeline', 'promptorder', 'promptpara', 'repeat2', 'repeat3'] + SAMPLES:
        pr = pairs(tag, cond); flips = [a['topic'] != b['topic'] for s, a, b in pr]
        extra[cond] = {'n': len(pr), 'flip': float(np.mean(flips)), 'flip_ci': boot_ci(flips), 'acc': float(np.mean([b['ok'] for s, a, b in pr])), 'd_acc': float(np.mean([b['ok'] - a['ok'] for s, a, b in pr]))}
    rep3 = [a['topic'] != b['topic'] for c in REPEATS for s, a, b in pairs(tag, c)]
    smp = [a['topic'] != b['topic'] for c in SAMPLES for s, a, b in pairs(tag, c)]
    extra['repeat_floor3'] = {'n': len(rep3), 'flip': float(np.mean(rep3)), 'flip_ci': boot_ci(rep3)}
    extra['sample_floor'] = {'n': len(smp), 'flip': float(np.mean(smp)), 'flip_ci': boot_ci(smp)}
    dose[tag]['extra'] = extra
N['dose'] = dose

# ---------------------------------------------------------------- contemporary, label-free set (ScreenSpot)
ss = {}
SS_MODELS = [t for t in MODELS if any(m(t, s, 'original') for s in EXTRA)]
for tag in SS_MODELS:
    rows = {}
    for cond in SS_CONDS:
        pr = pairs(tag, cond, ids=EXTRA); flips = [a['topic'] != b['topic'] for s, a, b in pr]
        rows[cond] = {'n': len(pr), 'flip': float(np.mean(flips)), 'flip_ci': boot_ci(flips), 'comp_jac': float(np.mean([S.jac(a['comps'], b['comps']) for s, a, b in pr if a['parsed'] and b['parsed']] or [float('nan')]))}
    en = [sens[tag][c]['flip'] for c in SS_CONDS if c != 'repeat1']; sp = [rows[c]['flip'] for c in SS_CONDS if c != 'repeat1']
    rows['summary'] = {'mean_flip': float(np.mean(sp)), 'mean_flip_enrico': float(np.mean(en)), 'floor': rows['repeat1']['flip'], 'spearman': float(stats.spearmanr(en, sp)[0]), 'n_screens': len({s for s in EXTRA if m(tag, s, 'original')})}
    ss[tag] = rows
N['screenspot'] = ss; N['ss_models'] = SS_MODELS
# across all models: correlation of relation-wise flip rates, contemporary vs Enrico
en_all = [sens[t][c]['flip'] for t in SS_MODELS for c in SS_CONDS if c != 'repeat1']; sp_all = [ss[t][c]['flip'] for t in SS_MODELS for c in SS_CONDS if c != 'repeat1']
N['ss_corr'] = {'spearman': float(stats.spearmanr(en_all, sp_all)[0]), 'p': float(stats.spearmanr(en_all, sp_all)[1]), 'n': len(en_all), 'mean_enrico': float(np.mean(en_all)), 'mean_ss': float(np.mean(sp_all))}

# ---------------------------------------------------------------- RQ3 cue probes
probes = {}
for tag in MODELS:
    rows = {}; ps = []
    for cond in PROBES:
        pr = pairs(tag, cond); b_ = sum(1 for s, a, b in pr if a['ok'] and not b['ok']); c_ = sum(1 for s, a, b in pr if b['ok'] and not a['ok'])
        rows[cond] = {'n': len(pr), 'acc': float(np.mean([b['ok'] for s, a, b in pr])), 'd_acc': float(np.mean([b['ok'] - a['ok'] for s, a, b in pr])), 'p': mcnemar_midp(b_, c_), 'agree': float(np.mean([a['topic'] == b['topic'] for s, a, b in pr]))}
        ps.append(rows[cond]['p'])
    for cond, p in zip(PROBES, holm(ps)): rows[cond]['p_holm'] = p
    probes[tag] = rows
N['probes'] = probes
# per-topic accuracy under each probe, pooled over models
N['probe_topic'] = {cond: {t: float(np.mean([m(tag, s, cond)['ok'] for tag in MODELS for s in screens_for(tag) if SCR[s]['topic'] == t and m(tag, s, cond)])) for t in S.TOPICS} for cond in ['original'] + PROBES}

# ---------------------------------------------------------------- RQ4 reliability: scores, selective prediction, conformal gate, voting
def score(tag, sid, kind, views):
    o = m(tag, sid, 'original')
    if kind == 'conf': return (o['conf'] or 0) / 100
    vs = [v for v in views if avail(sid, v) and has_topic(m(tag, sid, v))]
    if not vs: return 0.0
    agree = sum(1 for v in vs if m(tag, sid, v)['topic'] == o['topic']) / len(vs)
    if kind == 'agree': return agree
    cs = [m(tag, sid, v)['conf'] for v in vs if m(tag, sid, v)['topic'] == o['topic'] and m(tag, sid, v)['conf'] is not None]
    return agree * (np.mean(cs) / 100) if cs else 0.0
STRATS = [('conf', 'Verbalised confidence', 'conf', ['original'], 'baseline'), ('mr10', 'Metamorphic agreement (10 views)', 'agree', ['original'] + INV, 'ablation'),
          ('cwc3', 'CWC-3 (ours, 3 views)', 'cwc', ['original', 'jpeg', 'dim'], 'ours'), ('cwc10', 'CWC-10 (ours, 10 views)', 'cwc', ['original'] + INV, 'ours'),
          ('rep', 'Repeat agreement (3 repeats)', 'agree', ['original'] + REPEATS, 'baseline'), ('sc', 'Sampling self-consistency (5 samples)', 'agree', ['original'] + SAMPLES, 'baseline')]
ALPHA = 0.15
def crc(scores, correct, alpha):
    n = len(scores); order = np.argsort(scores); wrong = sum(1 for c in correct if not c)
    ok = lambda e: n / (n + 1) * e / n + 1 / (n + 1) <= alpha
    if ok(wrong): return -np.inf
    i = 0
    while i < n:
        v = scores[order[i]]
        while i < n and scores[order[i]] == v:
            if not correct[order[i]]: wrong -= 1
            i += 1
        if ok(wrong): return scores[order[i]] if i < n else np.inf
    return np.inf
def guarantee(scores, correct, alpha=ALPHA, reps=200, seed=1):
    r = random.Random(seed); n = len(scores); covs, risks, accs = [], [], []
    for _ in range(reps):
        idx = list(range(n)); r.shuffle(idx); cal, test = idx[:n // 2], idx[n // 2:]
        lam = crc([scores[i] for i in cal], [correct[i] for i in cal], alpha)
        kept = [i for i in test if scores[i] >= lam]; wrong = sum(1 for i in kept if not correct[i])
        covs.append(len(kept) / len(test)); risks.append(wrong / len(test)); accs.append(1 - wrong / len(kept) if kept else np.nan)
    return float(np.mean(covs)), float(np.mean(risks)), float(np.nanmean(accs))
rel = {}
for tag in MODELS:
    ids = [s for s in screens_for(tag) if has_topic(m(tag, s, 'original'))]
    correct = [bool(m(tag, s, 'original')['ok']) for s in ids]
    rel[tag] = {}
    for key, name, kind, views, role in STRATS:
        vs = [v for v in views if any(m(tag, s, v) for s in ids)]
        if kind != 'conf' and len(vs) < 2: continue
        sc = [score(tag, s, kind, views) for s in ids]
        pos = [x for x, c in zip(sc, correct) if c]; neg = [x for x, c in zip(sc, correct) if not c]
        auroc = float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg]))
        order = np.argsort(-np.array(sc)); cs = np.array(correct)[order]; cum = np.cumsum(cs) / np.arange(1, len(cs) + 1); aurc = float(np.mean(1 - cum))
        acc80 = float(np.mean(cs[:max(1, round(0.8 * len(cs)))]))
        cov, risk, acck = guarantee(sc, correct)
        rel[tag][key] = {'name': name, 'role': role, 'views': len(vs), 'auroc': auroc, 'aurc': aurc, 'acc80': acc80, 'cov': cov, 'risk': risk, 'acc_kept': acck, 'full_acc': float(np.mean(correct)), 'curve': (cum.tolist()[::max(1, len(cs) // 60)])}
    # majority vote over the ten views
    votes = []
    for s in ids:
        vs = [v for v in ['original'] + INV if avail(s, v) and has_topic(m(tag, s, v))]
        cnt = collections.Counter(m(tag, s, v)['topic'] for v in vs); top = max(cnt.values()); win = [t for t, c in cnt.items() if c == top]
        pred = m(tag, s, 'original')['topic'] if m(tag, s, 'original')['topic'] in win else win[0]
        votes.append(int(pred == SCR[s]['topic']))
    b_ = sum(1 for i, s in enumerate(ids) if correct[i] and not votes[i]); c_ = sum(1 for i, s in enumerate(ids) if votes[i] and not correct[i])
    rel[tag]['vote'] = {'acc': float(np.mean(votes)), 'd_acc': float(np.mean(votes) - np.mean(correct)), 'p': mcnemar_midp(b_, c_)}
N['rel'] = rel
# ranking of our method against the baselines available for every model
rank = {}
for tag in MODELS:
    cands = {k: v for k, v in rel[tag].items() if k in ('conf', 'cwc3', 'cwc10', 'rep', 'sc')}
    order = sorted(cands, key=lambda k: -cands[k]['auroc'])
    bases = {k: v for k, v in cands.items() if v['role'] == 'baseline'}; bb = max(bases, key=lambda k: bases[k]['auroc'])
    bc = max(bases, key=lambda k: bases[k]['cov'])
    rank[tag] = {'rank': order.index('cwc10') + 1, 'of': len(order), 'best_baseline': bb, 'lead_auroc': cands['cwc10']['auroc'] - bases[bb]['auroc'], 'lead_cov': cands['cwc10']['cov'] - bases[bc]['cov'], 'best_cov_baseline': bc,
                 'lead_acc80': cands['cwc10']['acc80'] - max(v['acc80'] for v in bases.values())}
N['rank'] = rank
N['rank_summary'] = {'wins': sum(1 for t in MODELS if rank[t]['rank'] == 1), 'mean_lead_auroc': float(np.mean([rank[t]['lead_auroc'] for t in MODELS])), 'mean_lead_cov': float(np.mean([rank[t]['lead_cov'] for t in MODELS])),
                     'cov_ours': float(np.mean([rel[t]['cwc10']['cov'] for t in MODELS])), 'cov_conf': float(np.mean([rel[t]['conf']['cov'] for t in MODELS])), 'risk_ours': float(np.mean([rel[t]['cwc10']['risk'] for t in MODELS])),
                     'auroc_ours': [rel[t]['cwc10']['auroc'] for t in MODELS], 'auroc_conf': [rel[t]['conf']['auroc'] for t in MODELS]}

# ---------------------------------------------------------------- component recall by condition (pooled) and costs
comp = {}
for cond in ['original'] + INV + ['textmask']:
    for c in S.COMPONENTS:
        hit = tot = 0
        for tag in MODELS:
            for s in screens_for(tag):
                if c in SCR[s]['components'] and avail(s, cond) and m(tag, s, cond) and m(tag, s, cond)['parsed']: tot += 1; hit += int(c in m(tag, s, cond)['comps'])
        if tot: comp[f'{cond}|{c}'] = (hit / tot, tot)
N['comp_recall'] = comp
N['cost_total'] = float(sum(((r.get('usage') or {}).get('cost') or 0) for r in REC.values()))
N['queries'] = {t: sum(1 for r in REC.values() if r['tag'] == t) for t in MODELS}
json.dump(N, open(f'{ROOT}/out/numbers.json', 'w'), indent=1, default=float)
print('numbers written:', len(json.dumps(N)) // 1000, 'kB')
