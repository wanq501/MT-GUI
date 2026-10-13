"""Paper statistics from results.json (exported by the console; older exports are named MTGUI_results.json) (browser export). Same definitions as the console (app/core.js)."""
import difflib, json, math, re, sys
from collections import Counter
import numpy as np
from scipy import stats

TOPICS = ['bare', 'calculator', 'camera', 'chat', 'editor', 'form', 'gallery', 'list', 'login', 'maps', 'mediaplayer', 'menu', 'modal', 'news', 'other', 'profile', 'search', 'settings', 'terms', 'tutorial']
COMPONENTS = ['Text Button', 'Image', 'Input', 'List Item', 'Toolbar', 'Card', 'Checkbox', 'Radio Button', 'On/Off Switch', 'Slider', 'Pager Indicator', 'Bottom Navigation', 'Multi-Tab', 'Drawer', 'Modal', 'Web View', 'Map View', 'Video', 'Advertisement', 'Date Picker', 'Number Stepper', 'Button Bar']
CONDS = ['original', 'repeat1', 'downscale', 'jpeg', 'blur', 'dim', 'grayscale', 'bezel', 'tilt', 'swap', 'statusbar', 'textmask', 'layout', 'textonly']
INV = ['downscale', 'jpeg', 'blur', 'dim', 'grayscale', 'bezel', 'tilt', 'swap', 'statusbar']
PROBES = ['textmask', 'layout', 'textonly']
TA = {'media player': 'mediaplayer', 'sign in': 'login', 'signin': 'login', 'sign up': 'login', 'map': 'maps', 'setting': 'settings', 'onboarding': 'tutorial', 'term': 'terms', 'forms': 'form', 'lists': 'list', 'menus': 'menu', 'dialog': 'modal', 'popup': 'modal', 'dialer': 'calculator', 'dial pad': 'calculator', 'dialpad': 'calculator', 'keypad': 'calculator'}
CA = {'button': 'Text Button', 'text button': 'Text Button', 'textbutton': 'Text Button', 'buttons': 'Text Button', 'image': 'Image', 'images': 'Image', 'picture': 'Image', 'photo': 'Image', 'input': 'Input', 'text input': 'Input', 'text field': 'Input', 'textfield': 'Input', 'edit text': 'Input', 'search bar': 'Input', 'list item': 'List Item', 'list': 'List Item', 'list items': 'List Item', 'toolbar': 'Toolbar', 'app bar': 'Toolbar', 'top bar': 'Toolbar', 'action bar': 'Toolbar', 'navigation bar': 'Toolbar', 'card': 'Card', 'cards': 'Card', 'checkbox': 'Checkbox', 'check box': 'Checkbox', 'checkboxes': 'Checkbox', 'radio button': 'Radio Button', 'radio': 'Radio Button', 'radio buttons': 'Radio Button', 'on/off switch': 'On/Off Switch', 'switch': 'On/Off Switch', 'toggle': 'On/Off Switch', 'toggle switch': 'On/Off Switch', 'slider': 'Slider', 'seekbar': 'Slider', 'pager indicator': 'Pager Indicator', 'page indicator': 'Pager Indicator', 'dots': 'Pager Indicator', 'bottom navigation': 'Bottom Navigation', 'bottom nav': 'Bottom Navigation', 'tab bar': 'Bottom Navigation', 'multi-tab': 'Multi-Tab', 'multi tab': 'Multi-Tab', 'tabs': 'Multi-Tab', 'tab': 'Multi-Tab', 'drawer': 'Drawer', 'navigation drawer': 'Drawer', 'modal': 'Modal', 'dialog': 'Modal', 'popup': 'Modal', 'web view': 'Web View', 'webview': 'Web View', 'map view': 'Map View', 'map': 'Map View', 'video': 'Video', 'video player': 'Video', 'advertisement': 'Advertisement', 'ad': 'Advertisement', 'ads': 'Advertisement', 'banner ad': 'Advertisement', 'date picker': 'Date Picker', 'datepicker': 'Date Picker', 'calendar': 'Date Picker', 'number stepper': 'Number Stepper', 'stepper': 'Number Stepper', 'button bar': 'Button Bar'}
LUT = {c.lower(): c for c in COMPONENTS}

def norm_topic(x):
    if not isinstance(x, str): return None
    t = x.strip().lower().replace('_', ' ').replace('-', ' ').split('(')[0]
    t = re.sub(r'\s+', ' ', re.sub(r'[^a-z /]', '', t)).strip()
    if t in TOPICS: return t
    if t.replace(' ', '') in TOPICS: return t.replace(' ', '')
    return TA.get(t)
def norm_comps(x):
    if isinstance(x, str): x = [p.strip() for p in re.split(r'[,;\n]', x) if p.strip()]
    if not isinstance(x, list): return []
    out = []
    for it in x:
        if not isinstance(it, str): continue
        k = re.sub(r'\s+', ' ', it.strip().lower()); c = LUT.get(k) or CA.get(k)
        if c and c not in out: out.append(c)
    return out
def norm_text(s):
    s = re.sub(r'[^\w\s%$€£.-]', ' ', s.lower()); return re.sub(r'\s+', ' ', s).strip()
def match_texts(pred, gt, thr=0.85):
    p = [y for y in (norm_text(x) for x in pred if isinstance(x, str) and x.strip()) if y]
    g = [y for y in (norm_text(x) for x in gt if isinstance(x, str) and x.strip()) if y]
    used, m = set(), 0
    for a in p:
        best, bj = 0.0, -1
        for j, b in enumerate(g):
            if j in used: continue
            if a == b: r = 1.0
            else:
                sm = difflib.SequenceMatcher(None, a, b)
                if sm.real_quick_ratio() < thr or sm.quick_ratio() < thr: continue
                r = sm.ratio()
            if r > best: best, bj = r, j
        if bj >= 0 and best >= thr: used.add(bj); m += 1
    return m, len(p), len(g)
def prf(m, np_, ng):
    p = m / np_ if np_ else (1.0 if ng == 0 else 0.0); r = m / ng if ng else (1.0 if np_ == 0 else 0.0)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
def jac(a, b):
    a, b = set(a), set(b); return 1.0 if not a and not b else len(a & b) / len(a | b)
def jac_text(a, b):
    m, na, nb = match_texts(a, b); u = na + nb - m; return 1.0 if u == 0 else m / u
def mcnemar(b, c):
    n = b + c; return 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))
def holm(pv):
    out, run = {}, 0.0
    for i, (k, p) in enumerate(sorted(pv.items(), key=lambda kv: kv[1])): run = max(run, min(1.0, (len(pv) - i) * p)); out[k] = run
    return out

def load(path):
    d = json.load(open(path, encoding='utf-8'))
    ids = d['manifest']['ids']; gt = d['screens']
    for sid in ids:
        g = gt[sid]; g['text_count'] = len([y for y in (norm_text(x) for x in g['texts'] if isinstance(x, str) and x.strip()) if y])
    tags = [m['tag'] for m in d['config']['models'] if m.get('enabled', True)]
    recs = {(r['tag'], r['sid'], r['cond']): r for r in d['records'] if r.get('ok')}
    return d, ids, gt, tags, recs

def metrics(rec, g):
    o = dict(answered=rec is not None, parsed=False, topic=None, ok=0, conf=None, comps=[], texts=[], cm=0, cnp=0, cng=len(g['components']), cf=0.0, tm=0, tnp=0, tng=g['text_count'], tf=0.0, tokens=0, latency=0.0)
    if rec is None: return o
    o['tokens'] = int((rec.get('usage') or {}).get('total_tokens') or 0); o['latency'] = float(rec.get('latency') or 0)
    p = rec.get('parsed')
    if not isinstance(p, dict): return o
    o['parsed'] = True; o['topic'] = norm_topic(p.get('topic')); o['ok'] = int(o['topic'] == g['topic'])
    try:
        c = p.get('confidence'); o['conf'] = None if c in (None, '') else min(100.0, max(0.0, float(c)))
    except Exception: o['conf'] = None
    o['comps'] = norm_comps(p.get('components')); o['cm'] = len(set(o['comps']) & set(g['components'])); o['cnp'] = len(o['comps'])
    o['cf'] = prf(o['cm'], o['cnp'], o['cng'])[2]
    o['texts'] = [t for t in (p.get('texts') if isinstance(p.get('texts'), list) else []) if isinstance(t, str)]
    o['tm'], o['tnp'], o['tng'] = match_texts(o['texts'], g['texts']); o['tf'] = prf(o['tm'], o['tnp'], o['tng'])[2]
    return o

def analyse(path):
    d, ids, gt, tags, recs = load(path)
    avail = lambda sid, c: c != 'swap' or gt[sid]['swap_applied']
    cache = {}
    def M(tag, sid, c):
        k = (tag, sid, c)
        if k not in cache: cache[k] = metrics(recs.get(k), gt[sid])
        return cache[k]
    out = {'cap': [], 'sens': [], 'rob': [], 'probes': [], 'sel': [], 'vote': []}
    for tag in tags:
        for c in CONDS:
            sids = [s for s in ids if avail(s, c)]; ms = [M(tag, s, c) for s in sids]; n = len(ms); ans = [m for m in ms if m['answered']]
            sm = lambda k: sum(m[k] for m in ms)
            out['cap'].append(dict(model=tag, condition=c, n=n, json_rate=sum(m['parsed'] for m in ms) / n, topic_acc=np.mean([m['ok'] for m in ms]),
                                   comp_micro_f1=prf(sm('cm'), sm('cnp'), sm('cng'))[2], text_micro_f1=prf(sm('tm'), sm('tnp'), sm('tng'))[2],
                                   avg_tokens=np.mean([m['tokens'] for m in ans]) if ans else float('nan')))
        pv = {}
        for c in ['repeat1'] + INV:
            sids = [s for s in ids if avail(s, c)]; flips, jc, jt = [], [], []; b = cc = 0
            for s in sids:
                o, m = M(tag, s, 'original'), M(tag, s, c)
                if o['parsed'] and m['parsed']: flips.append(o['topic'] != m['topic']); jc.append(jac(o['comps'], m['comps'])); jt.append(jac_text(o['texts'], m['texts']))
                b += o['ok'] and not m['ok']; cc += m['ok'] and not o['ok']
            co = [M(tag, s, 'original')['cf'] for s in sids]; cm_ = [M(tag, s, c)['cf'] for s in sids]
            try: pw = stats.wilcoxon(co, cm_, zero_method='zsplit').pvalue if np.any(np.array(co) != np.array(cm_)) else 1.0
            except Exception: pw = 1.0
            row = dict(model=tag, condition=c, n_screens=len(sids), topic_acc_orig=np.mean([M(tag, s, 'original')['ok'] for s in sids]), topic_acc=np.mean([M(tag, s, c)['ok'] for s in sids]),
                       flip_rate=np.mean(flips) if flips else float('nan'), degraded=int(b), improved=int(cc), p_mcnemar=mcnemar(b, cc), comp_jaccard=np.mean(jc) if jc else float('nan'),
                       text_jaccard=np.mean(jt) if jt else float('nan'), p_wilcoxon_comp=pw)
            row['delta_acc'] = row['topic_acc'] - row['topic_acc_orig']
            out['sens'].append(row)
            if c != 'repeat1': pv[c] = row['p_mcnemar']
        for c, p in holm(pv).items(): next(r for r in out['sens'] if r['model'] == tag and r['condition'] == c)['p_mcnemar_holm'] = p
        d_ = [r for r in out['sens'] if r['model'] == tag and r['condition'] in INV]
        out['rob'].append(dict(model=tag, robustness_index=np.mean([r['topic_acc'] / r['topic_acc_orig'] if r['topic_acc_orig'] else 0 for r in d_])))
        acc_o = np.mean([M(tag, s, 'original')['ok'] for s in ids])
        for pr in PROBES:
            acc_p = np.mean([M(tag, s, pr)['ok'] for s in ids])
            out['probes'].append(dict(model=tag, probe=pr, topic_acc_probe=acc_p, cue_reliance_index=(acc_o - acc_p) / acc_o if acc_o else float('nan')))
        def majority(s, views):
            vs = [v for v in views if avail(s, v)]; cnt = Counter(M(tag, s, v)['topic'] for v in vs if M(tag, s, v)['topic'])
            if not cnt: return None, 0.0, vs
            top = cnt.most_common(); best = top[0][1]; tied = [t for t, n in top if n == best]; o = M(tag, s, 'original')['topic']
            return (o if o in tied else tied[0]), best / len(vs), vs
        STR = {'conf': None, 'rep': ['original', 'repeat1'], 'mr3': ['original', 'jpeg', 'dim'], 'mr6': ['original', 'jpeg', 'dim', 'grayscale', 'bezel', 'statusbar'], 'mrall': ['original'] + INV}
        for key, views in STR.items():
            sc, cor = [], []
            for s in ids:
                if views is None: o = M(tag, s, 'original'); pred, score = o['topic'], (o['conf'] if o['conf'] is not None else 0.0)
                else: pred, score, _ = majority(s, views)
                sc.append(score); cor.append(float(pred == gt[s]['topic']))
            order = np.argsort(-np.asarray(sc), kind='stable'); cs = np.asarray(cor)[order]; risks = 1 - np.cumsum(cs) / (np.arange(len(cs)) + 1)
            acc_at = lambda cov: float(np.mean(cs[:max(1, int(round(cov * len(cs))))]))
            out['sel'].append(dict(model=tag, strategy=key, full_acc=float(np.mean(cor)), aurc=float(np.mean(risks)), acc80=acc_at(0.8), acc90=acc_at(0.9)))
        base = [M(tag, s, 'original')['ok'] for s in ids]
        for key, views in [('single', ['original'])] + [(k, v) for k, v in STR.items() if v]:
            oks, cf = [], []
            for s in ids:
                pred, _, vs = majority(s, views); oks.append(float(pred == gt[s]['topic']))
                cnt = Counter(k for v in vs for k in M(tag, s, v)['comps']); pset = {k for k, n in cnt.items() if n * 2 >= len(vs)}
                cf.append(prf(len(pset & set(gt[s]['components'])), len(pset), len(gt[s]['components']))[2])
            out['vote'].append(dict(model=tag, strategy=key, topic_acc=float(np.mean(oks)), comp_macro_f1=float(np.mean(cf))))
    return out

if __name__ == '__main__':
    py = analyse(sys.argv[1]); js = json.load(open(sys.argv[2]))
    def cmp(name, keys, fields, jrows, prows):
        worst = 0.0; n = 0
        for pr in prows:
            jr = next((r for r in jrows if all(r.get(k) == pr[k] for k in keys)), None)
            if jr is None: print('missing in JS', name, [pr[k] for k in keys]); continue
            for f in fields:
                a, b = pr.get(f), jr.get(f)
                if a is None or b is None or (isinstance(a, float) and math.isnan(a)): continue
                worst = max(worst, abs(float(a) - float(b))); n += 1
        print(f'{name:8} compared {n:5d} values, max |py - js| = {worst:.2e}')
    cmp('cap', ['model', 'condition'], ['json_rate', 'topic_acc', 'comp_micro_f1', 'text_micro_f1', 'avg_tokens'], js['cap'], py['cap'])
    cmp('sens', ['model', 'condition'], ['topic_acc_orig', 'topic_acc', 'delta_acc', 'flip_rate', 'degraded', 'improved', 'p_mcnemar', 'p_mcnemar_holm', 'comp_jaccard', 'text_jaccard'], js['sens'], py['sens'])
    cmp('rob', ['model'], ['robustness_index'], js['rob'], py['rob'])
    cmp('probes', ['model', 'probe'], ['topic_acc_probe', 'cue_reliance_index'], js['probes']['rows'], py['probes'])
    cmp('sel', ['model', 'strategy'], ['full_acc', 'aurc', 'acc80', 'acc90'], js['sel'], py['sel'])
    cmp('vote', ['model', 'strategy'], ['topic_acc', 'comp_macro_f1'], js['vote'], py['vote'])
    print('wilcoxon (python only), gpt5mini blur p =', next(r['p_wilcoxon_comp'] for r in py['sens'] if r['model'] == 'gpt5mini' and r['condition'] == 'blur'))
