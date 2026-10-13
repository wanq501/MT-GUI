"""E1-E3: three analyses of the existing MT-GUI data (no new model calls).
E3  flips ~ relation + model, logistic GEE with exchangeable correlation within screen (cluster-robust).
E2  empirical validity of the conformal risk-control gate: realised risk on held-out screens over 200 random splits.
E1  CWC view ablation: leave-one-relation-out, and AUROC against the number of views (= model calls per screen).
Writes out/extra.json and out/figs/e{1,2,3}_*.pdf|png. Run after analysis.py (uses its data loading and score functions)."""
import os, json, itertools, random, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'analysis.py'), encoding='utf-8').read()
exec(src.split("# ---------------------------------------------------------------- RQ1 baseline")[0])
exec("def score(tag, sid, kind, views):\n" + src.split("def score(tag, sid, kind, views):\n")[1].split("STRATS = [")[0])
exec("ALPHA = 0.15\n" + src.split("ALPHA = 0.15\n")[1].split("rel = {}")[0])
import pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(HERE) if os.path.basename(HERE) == 'analysis' else HERE, 'out'); os.makedirs(os.path.join(OUT, 'figs'), exist_ok=True)
NAME = {'gpt5': 'GPT-5', 'gpt5mini': 'GPT-5 mini', 'haiku': 'Claude Haiku 4.5', 'gemini': 'Gemini 2.5 Flash', 'qwen3vl': 'Qwen3-VL-235B', 'qwen25vl7b': 'Qwen2.5-VL-7B', 'qwen3vl8b': 'Qwen3-VL-8B', 'internvl3': 'InternVL3-8B'}
RL = {'downscale': 'Downscale', 'jpeg': 'JPEG', 'blur': 'Blur', 'dim': 'Dim', 'grayscale': 'Grayscale', 'bezel': 'Bezel', 'tilt': 'Tilt', 'swap': 'Swap', 'statusbar': 'Status bar', 'repeat1': 'Repeat (floor)'}
R = {}
# ------------------------------------------------------------------ E3
rows = []
for t in MODELS:
    for s in screens_for(t):
        o = m(t, s, 'original')
        if not has_topic(o): continue
        for c in ['repeat1'] + INV:
            x = m(t, s, c)
            if not has_topic(x) or (c == 'swap' and not avail(s, 'swap')): continue
            rows.append({'flip': int(x['topic'] != o['topic']), 'rel': c, 'model': t, 'screen': s})
df = pd.DataFrame(rows)
mod = smf.gee('flip ~ C(rel, Treatment("repeat1")) + C(model, Treatment("haiku"))', 'screen', df, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
ci = mod.conf_int(); e3 = {'n_pairs': int(len(df)), 'n_screens': int(df.screen.nunique()), 'within_screen_corr': float(mod.cov_struct.dep_params), 'relation': {}, 'model': {}}
for k in mod.params.index:
    if k == 'Intercept': continue
    rec = {'or': float(np.exp(mod.params[k])), 'lo': float(np.exp(ci.loc[k, 0])), 'hi': float(np.exp(ci.loc[k, 1])), 'p': float(mod.pvalues[k])}
    if 'C(rel' in k: e3['relation'][k.split('[T.')[1].rstrip(']')] = rec
    else: e3['model'][k.split('[T.')[1].rstrip(']')] = rec
R['E3'] = e3
# ------------------------------------------------------------------ E2
alphas = [0.10, 0.15, 0.20, 0.25]; e2 = {}
for t in MODELS:
    ids = [s for s in screens_for(t) if has_topic(m(t, s, 'original'))]
    sc = np.array([score(t, s, 'cwc', ['original'] + INV) for s in ids]); ok = np.array([bool(m(t, s, 'original')['ok']) for s in ids])
    e2[t] = {}
    for a in alphas:
        risks, covs = [], []
        rr = random.Random(1)          # the same 200 calibration/test splits as the main analysis (guarantee() in analysis.py)
        for _ in range(200):
            idx = list(range(len(ids))); rr.shuffle(idx); h = len(ids) // 2; cal, te = np.array(idx[:h]), np.array(idx[h:])
            lam = crc(list(sc[cal]), list(ok[cal]), a); keep = sc[te] >= lam
            risks.append(float(np.mean(keep & ~ok[te]))); covs.append(float(np.mean(keep)))
        e2[t][str(a)] = {'mean_risk': float(np.mean(risks)), 'share_le_alpha': float(np.mean(np.array(risks) <= a)), 'q90_risk': float(np.quantile(risks, 0.9)), 'coverage': float(np.mean(covs))}
R['E2'] = e2
# ------------------------------------------------------------------ E1
def auroc(scores, ok):
    s, y = np.array(scores), np.array(ok, bool); pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg): return float('nan')
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))
e1 = {'full': {}, 'loo': {}, 'k': {}}
for t in MODELS:
    ids = [s for s in screens_for(t) if has_topic(m(t, s, 'original'))]; ok = [bool(m(t, s, 'original')['ok']) for s in ids]
    e1['full'][t] = auroc([score(t, s, 'cwc', ['original'] + INV) for s in ids], ok)
    e1['loo'][t] = {r: auroc([score(t, s, 'cwc', ['original'] + [v for v in INV if v != r]) for s in ids], ok) for r in INV}
    rnd = random.Random(11); e1['k'][t] = {}
    for k in [1, 2, 3, 5, 7, 9]:
        subs = [tuple(rnd.sample(INV, k)) for _ in range(30)] if k < 9 else [tuple(INV)]
        e1['k'][t][k + 1] = float(np.nanmean([auroc([score(t, s, 'cwc', ['original'] + list(v)) for s in ids], ok) for v in subs]))
    e1['k'][t][1] = auroc([score(t, s, 'conf', ['original']) for s in ids], ok)
R['E1'] = e1
json.dump(R, open(os.path.join(OUT, 'extra.json'), 'w'), indent=1)
# ------------------------------------------------------------------ figures
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'axes.titleweight': 'bold', 'axes.titlesize': 7.5})
ACC = '#1f5fbf'; MC = {'gpt5': '#0072B2', 'gpt5mini': '#56B4E9', 'haiku': '#D55E00', 'gemini': '#CC79A7', 'qwen3vl': '#E69F00', 'qwen25vl7b': '#009E73', 'qwen3vl8b': '#8C8C00', 'internvl3': '#222222'}
def save(fig, n): fig.savefig(os.path.join(OUT, 'figs', n + '.pdf'), bbox_inches='tight'); fig.savefig(os.path.join(OUT, 'figs', n + '.png'), dpi=200, bbox_inches='tight'); plt.close(fig)
fig, ax = plt.subplots(figsize=(3.4, 2.6)); ks = sorted(e3['relation'], key=lambda k: e3['relation'][k]['or'])
for i, k in enumerate(ks):
    r = e3['relation'][k]; ax.plot([r['lo'], r['hi']], [i, i], color=ACC, lw=1.2); ax.scatter(r['or'], i, s=18, color=ACC, zorder=3)
ax.axvline(1, color='#8a94a6', lw=0.7, ls='--'); ax.set_xscale('log'); ax.set_yticks(range(len(ks))); ax.set_yticklabels([RL[k] for k in ks]); ax.set_xlabel('Odds ratio of a flip vs a repeated query (95% CI)')
ax.set_title('Every relation flips more than a repeat', loc='left'); save(fig, 'e3_gee_forest')
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for t in MODELS: ax.plot(alphas, [e2[t][str(a)]['mean_risk'] for a in alphas], '-o', ms=2.5, lw=1, color=MC[t], label=NAME[t])
ax.plot([0.08, 0.27], [0.08, 0.27], color='#8a94a6', ls='--', lw=0.8); ax.set_xlabel('Risk level α'); ax.set_ylabel('Realized risk on held-out screens'); ax.legend(fontsize=5, frameon=False, ncol=2)
ax.set_title('The guarantee holds on new screens', loc='left'); save(fig, 'e2_validity')
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for t in MODELS: kk = sorted(e1['k'][t], key=int); ax.plot([int(k) for k in kk], [e1['k'][t][k] for k in kk], '-o', ms=2.5, lw=1, color=MC[t])
ax.set_xlabel('Model calls per screen (views)'); ax.set_ylabel('AUROC'); ax.set_title('Most of the gain comes from a few views', loc='left'); save(fig, 'e1_views')
print(json.dumps({'E3_relation_OR': {RL[k]: round(v['or'], 2) for k, v in e3['relation'].items()}, 'corr': round(e3['within_screen_corr'], 3), 'n': e3['n_pairs'],
                  'E2_mean_risk@0.15': {NAME[t]: round(e2[t]['0.15']['mean_risk'], 3) for t in MODELS}, 'E2_share<=a@0.15': {NAME[t]: round(e2[t]['0.15']['share_le_alpha'], 2) for t in MODELS},
                  'E1_auc_by_views(mean over models)': {k: round(float(np.nanmean([e1['k'][t][k] for t in MODELS])), 3) for k in [1, 2, 3, 4, 6, 8, 10]},
                  'E1_loo_drop_max': {NAME[t]: max(((r, round(e1['full'][t] - v, 3)) for r, v in e1['loo'][t].items()), key=lambda z: z[1]) for t in MODELS}}, ensure_ascii=False, indent=0))
