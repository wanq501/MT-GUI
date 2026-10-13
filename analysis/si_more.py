"""More Supplementary Information from the full MT-GUI data: tables S4-S10 and figures S1-S2.
Tables: full text width (tabular*), \\footnotesize, one line per header and row. Figures: the style of figs_rq.py."""
import os, json
from collections import Counter, defaultdict
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_rq.py'), encoding='utf-8').read().split('# ===================== RQ1 =====================')[0])
TAB = os.path.join(HERE, 'tex', 'tables'); os.makedirs(TAB, exist_ok=True)
NAME = {t: MODEL_STYLE[t][0] for t in MOD}
RLN = {'downscale': 'Downscale', 'jpeg': 'JPEG', 'blur': 'Blur', 'dim': 'Dim', 'grayscale': 'Grayscale', 'bezel': 'Bezel', 'tilt': 'Rotation', 'swap': 'Swap', 'statusbar': 'Status bar'}
def tex_table(path, spec, header, rows):
    body = '\n'.join(' & '.join(str(c) for c in r) + ' \\\\' for r in rows)
    open(path, 'w', encoding='utf-8').write('\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}' + spec + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular*}\n')
TYPES = sorted({SCR[s]['topic'] for s in IDS}); TN = lambda t: t.replace('mediaplayer', 'media player')

# S4 serving statistics per model (all answered requests)
R = D['records']; rows = []
for t in MOD:
    rr = [r for r in R if r['tag'] == t]; lat = [r['latency'] for r in rr if r.get('latency') is not None]
    u = [r.get('usage') or {} for r in rr]; pin = [x.get('prompt_tokens') for x in u if x.get('prompt_tokens')]; pout = [x.get('completion_tokens') for x in u if x.get('completion_tokens')]
    cost = sum((x.get('cost') or 0) for x in u)
    rows.append([NAME[t], f'{len(rr):,}', f"{100*np.mean([r['attempts'] > 1 for r in rr]):.1f}", f"{100*np.mean([r.get('finish_reason') == 'length' for r in rr]):.1f}",
                 f'{np.median(lat):.1f}' if lat else '--', f'{np.mean(pin):,.0f}' if pin else '--', f'{np.mean(pout):,.0f}' if pout else '--', f'{cost:.2f}' if cost else '0 (local)'])
tex_table(f'{TAB}/s4_serving.tex', 'lrrrrrrr', 'Model & Requests & Retried (\\%) & Truncated (\\%) & Median latency (s) & Input tokens & Output tokens & Cost (US\\$)', rows)

# S5 dataset composition: Enrico screens per type, pooled accuracy and flip rate
rows = []
for tp in TYPES:
    ss = [s for s in IDS if SCR[s]['topic'] == tp]
    acc = [m(t, s, 'original')['ok'] for t in MOD for s in ss if m(t, s, 'original')]
    fl = [m(t, s, 'original')['topic'] != m(t, s, r)['topic'] for t in MOD for s in ss for r in INV if m(t, s, 'original') and m(t, s, r) and m(t, s, r)['topic'] and m(t, s, 'original')['topic'] and avail(s, r) and same_prov(t, s, 'original', r)]
    rows.append([TN(tp), len(ss), f'{100*np.mean(acc):.1f}', f'{100*np.mean(fl):.1f}'])
tex_table(f'{TAB}/s5_types.tex', 'lrrr', 'Screen type & Screens & Accuracy, pooled (\\%) & Flip rate, pooled (\\%)', rows)

# S6 most frequent flip transitions (original answer -> follow-up answer), pooled over models and relations
tr = Counter(); by_rel = defaultdict(Counter); total = 0
for t in MOD:
    for s in screens_for(t):
        o = m(t, s, 'original')
        if not o or not o['topic']: continue
        for r in INV:
            x = m(t, s, r)
            if x and x['topic'] and avail(s, r) and same_prov(t, s, 'original', r):   # same pairs as Table S1
                if x['topic'] != o['topic']: tr[(o['topic'], x['topic'])] += 1; by_rel[(o['topic'], x['topic'])][r] += 1; total += 1
rows = [[TN(a), TN(b), n, f'{100*n/total:.1f}', RLN[by_rel[(a, b)].most_common(1)[0][0]]] for (a, b), n in tr.most_common(15)]
tex_table(f'{TAB}/s6_transitions.tex', 'llrrl', 'Original answer & Follow-up answer & Flips & Share of all flips (\\%) & Most frequent relation', rows)
TOTAL_FLIPS = total

# S7 the full GEE model: relations relative to a repeated query, models relative to Claude Haiku 4.5
e3 = EX['E3']; rows = [[f'Relation: {RLN[k]}', f"{v['or']:.2f}", f"{v['lo']:.2f}--{v['hi']:.2f}", '$<10^{-15}$' if v['p'] < 1e-15 else f"{v['p']:.2g}"] for k, v in sorted(e3['relation'].items(), key=lambda kv: -kv[1]['or'])]
rows += [[f'Model: {NAME[k]}', f"{v['or']:.2f}", f"{v['lo']:.2f}--{v['hi']:.2f}", f"{v['p']:.2g}" if v['p'] >= 1e-15 else '$<10^{-15}$'] for k, v in sorted(e3['model'].items(), key=lambda kv: -kv[1]['or'])]
tex_table(f'{TAB}/s7_gee.tex', 'lrrr', 'Term & Odds ratio & 95\\% CI & $p$', rows)

# S8 calibration of verbalised confidence
rows = []; CAL = {}
for t in MOD:
    xs = [(m(t, s, 'original')['conf'] / 100, m(t, s, 'original')['ok']) for s in screens_for(t) if m(t, s, 'original') and m(t, s, 'original')['conf'] is not None]
    c = np.array([a for a, _ in xs]); y = np.array([b for _, b in xs], float); bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(c, bins) - 1, 0, 9)
    ece = sum((idx == b).mean() * abs(y[idx == b].mean() - c[idx == b].mean()) for b in range(10) if (idx == b).any())
    CAL[t] = (c, y); rows.append([NAME[t], f'{100*y.mean():.1f}', f'{100*c.mean():.1f}', f'{100*c[y == 1].mean():.1f}', f'{100*c[y == 0].mean():.1f}', f'{100*ece:.1f}', f'{np.mean((c - y) ** 2):.3f}'])
tex_table(f'{TAB}/s8_calibration.tex', 'lrrrrrr', 'Model & Accuracy (\\%) & Mean conf. (\\%) & Conf. if correct & Conf. if wrong & ECE (\\%) & Brier', rows)

# S9 the gate across risk levels (200 calibration/test splits)
AL = ['0.1', '0.15', '0.2', '0.25']; rows = []
for t in MOD:
    rows.append([NAME[t]] + [f"{100*EX['E2'][t][a]['coverage']:.0f} / {100*EX['E2'][t][a]['mean_risk']:.1f}" for a in AL] + [f"{100*min(EX['E2'][t][a]['share_le_alpha'] for a in AL):.0f}"])
tex_table(f'{TAB}/s9_alpha.tex', 'lrrrrr', 'Model & $\\alpha=0.10$ & $\\alpha=0.15$ & $\\alpha=0.20$ & $\\alpha=0.25$ & Min. splits $\\le\\alpha$ (\\%)', rows)

# S10 leave-one-relation-out AUROC of CWC
rows = [[NAME[t], f"{EX['E1']['full'][t]:.3f}"] + [f"{EX['E1']['loo'][t][r]:.3f}" for r in INV] for t in MOD]
tex_table(f'{TAB}/s10_loo.tex', 'l' + 'r' * 10, 'Model & All & ' + ' & '.join(['Down.', 'JPEG', 'Blur', 'Dim', 'Gray', 'Bezel', 'Rotation', 'Swap', 'Status']), rows)

# Figure S1: reliability diagrams per model
for i, t in enumerate(MOD):
    c, y = CAL[t]; fig, ax = panel(NAME[t]); bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(c, bins) - 1, 0, 9)
    xs = [c[idx == b].mean() * 100 for b in range(10) if (idx == b).sum() >= 3]; ys = [y[idx == b].mean() * 100 for b in range(10) if (idx == b).sum() >= 3]; ns = [(idx == b).sum() for b in range(10) if (idx == b).sum() >= 3]
    ax.plot([0, 100], [0, 100], color=GREY, ls='--', lw=1); ax.scatter(xs, ys, s=[12 + 0.6 * n for n in ns], color=RED, edgecolor=INK, linewidth=0.5, zorder=3); ax.plot(xs, ys, color=RED, lw=1.2)
    ax.set_xlim(40, 100); ax.set_ylim(0, 100); ax.set_xlabel('Verbalized confidence (%)'); ax.set_ylabel('Accuracy (%)'); save(fig, f'sf1_{t}', 'abcdefgh'[i])

# Figure S2: the full transition matrix (original answer -> follow-up answer), pooled
FS6 = 6.0; W6 = 6.925
with plt.rc_context({'font.size': FS6, 'axes.titlesize': FS6, 'axes.labelsize': FS6, 'xtick.labelsize': FS6, 'ytick.labelsize': FS6}):
    labs = sorted({a for a, _ in tr} | {b for _, b in tr}); Hm = np.array([[tr.get((a, b), 0) for b in labs] for a in labs], float)
    cell = (W6 - 1.2) / len(labs); ch = 0.225; fh = 0.4 + 0.8 + ch * len(labs); fig = plt.figure(figsize=(W6, fh)); aw = cell * len(labs)
    ax = fig.add_axes([1.1 / W6, 0.8 / fh, aw / W6, ch * len(labs) / fh]); im = ax.imshow(np.log10(Hm + 1), cmap='Blues', aspect='auto'); ax.grid(False)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([TN(l) for l in labs], rotation=45, ha='right'); ax.set_yticks(range(len(labs))); ax.set_yticklabels([TN(l) for l in labs])
    ax.set_xlabel('Answer to the follow-up input'); ax.set_ylabel('Answer to the original'); ax.set_title(f'Flips from one screen type to another, pooled over models and relations (n = {TOTAL_FLIPS:,})', fontweight='bold')
    for i_ in range(len(labs)):
        for j_ in range(len(labs)):
            if Hm[i_, j_] > 0: ax.text(j_, i_, f'{int(Hm[i_, j_])}', ha='center', va='center', color='white' if Hm[i_, j_] > 60 else INK)
    fig.savefig(f'{OUT}/sf2_transitions.pdf'); fig.savefig(f'{OUT}/sf2_transitions.png', dpi=150); plt.close(fig)
print('S4-S10 tables and S1-S2 figures written; flips pooled:', TOTAL_FLIPS)
