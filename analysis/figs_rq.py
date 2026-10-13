"""Results figures for RQ1-RQ4 in the style of the authors' previous paper (TOSEM-2026-1243).

Every panel is its own square chart, saved as its own PDF and placed four per row in LaTeX
(width 0.24\\linewidth each). Canvas and type are chosen so that all text prints at 6 pt:
panel width 0.24 x 6.925 in = 1.662 in, canvas 3.88 in, type 14 pt (14 x 1.662 / 3.88 = 6.0).

Style (from the previous paper's figure scripts): Times New Roman, boxed axes in #3a3a3a, dashed grey grid,
legend on a white card with a grey edge, one red (#E0555A) for our method and for flips, lavender (#8E8AD8)
as the second accent, greys for everything neutral, the panel letter below the chart.
Run after analysis.py and extra_analyses.py.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'analysis.py'), encoding='utf-8').read()
exec(src.split("# ---------------------------------------------------------------- RQ1 baseline")[0])
exec("def score(tag, sid, kind, views):\n" + src.split("def score(tag, sid, kind, views):\n")[1].split("STRATS = [")[0])
N = json.load(open(os.path.join(HERE, 'out', 'numbers.json'))); EX = json.load(open(os.path.join(HERE, 'out', 'extra.json')))
OUT = os.path.join(HERE, 'out', 'panels'); os.makedirs(OUT, exist_ok=True)

# ===================== settings =====================
SIZE = 3.88; FS = 14                      # canvas (in) and the one type size (pt); prints at 6 pt
RED, LAV, GREY, LGREY, INK = '#E0555A', '#8E8AD8', '#8A8A8A', '#C8C8C8', '#3a3a3a'
MODEL_STYLE = {'gpt5': ('GPT-5', '#4C72B0', 'o'), 'gpt5mini': ('GPT-5 mini', '#8DA0CB', 's'), 'haiku': ('Claude Haiku 4.5', '#DD8452', '^'),
               'gemini': ('Gemini 2.5 Flash', '#937860', 'D'), 'qwen3vl': ('Qwen3-VL-235B', '#55A868', 'v'), 'qwen25vl7b': ('Qwen2.5-VL-7B', '#64B5CD', 'P'),
               'qwen3vl8b': ('Qwen3-VL-8B', '#8172B3', 'X'), 'internvl3': ('InternVL3-8B', '#4D4D4D', '*')}
RL = {'downscale': 'Down.', 'jpeg': 'JPEG', 'blur': 'Blur', 'dim': 'Dim', 'grayscale': 'Gray', 'bezel': 'Bezel', 'tilt': 'Rotation', 'swap': 'Swap', 'statusbar': 'Status'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': FS, 'axes.titlesize': FS, 'axes.labelsize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS,
                     'axes.edgecolor': INK, 'axes.linewidth': 1.0, 'axes.spines.top': True, 'axes.spines.right': True, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.labelcolor': INK, 'text.color': INK, 'axes.grid': True, 'grid.color': 'gray', 'grid.alpha': 0.5, 'grid.linestyle': '--', 'grid.linewidth': 0.8,
                     'axes.axisbelow': True, 'legend.frameon': True, 'legend.framealpha': 0.85, 'legend.edgecolor': '#c9c9c9', 'legend.facecolor': 'white',
                     'legend.borderpad': 0.35, 'legend.handletextpad': 0.4, 'legend.labelspacing': 0.3, 'pdf.fonttype': 42, 'savefig.dpi': 300})

SHORT = {'gpt5': 'GPT-5', 'gpt5mini': 'GPT-5 mini', 'haiku': 'Haiku 4.5', 'gemini': 'Gemini 2.5', 'qwen3vl': 'Qwen3-235B', 'qwen25vl7b': 'Qwen2.5-7B', 'qwen3vl8b': 'Qwen3-8B', 'internvl3': 'InternVL3'}
def panel(title, left=0.2, bottom=0.25):
    fig = plt.figure(figsize=(SIZE, SIZE)); ax = fig.add_axes([left, bottom, 0.95 - left, 0.91 - bottom]); ax.set_title(title, fontweight='bold', pad=6); return fig, ax
def save(fig, name, letter):
    fig.text(0.5, 0.015, f'({letter})', ha='center', va='bottom', fontweight='bold')
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text(): t.set_fontsize(FS)
    fig.savefig(f'{OUT}/{name}.pdf'); fig.savefig(f'{OUT}/{name}.png', dpi=110); plt.close(fig)
def strip(ax, cats, vals_by_model, mean=True, hollow=None):
    for j, c in enumerate(cats):
        for k, (t, v) in enumerate(vals_by_model[c].items()):
            nm, col, mk = MODEL_STYLE[t]; x = j + (k - (len(vals_by_model[c]) - 1) / 2) * 0.06
            ax.scatter(x, v, s=42, marker=mk, facecolor='none' if hollow and hollow[c][t] else col, edgecolor=col, linewidth=1.0, zorder=3)
        if mean: m = np.mean(list(vals_by_model[c].values())); ax.plot([j - 0.3, j + 0.3], [m, m], color=INK, lw=2.6, zorder=4)
    ax.set_xticks(range(len(cats))); ax.set_xlim(-0.6, len(cats) - 0.4)
def model_legend(ax, tags, loc='upper left', ncol=1):
    h = [Line2D([], [], marker=MODEL_STYLE[t][2], color=MODEL_STYLE[t][1], ls='none', ms=6, label=MODEL_STYLE[t][0]) for t in tags]; ax.legend(handles=h, loc=loc, ncol=ncol)
MOD = MODELS; SS = N['ss_models']; LOCAL = ['qwen25vl7b', 'qwen3vl8b', 'internvl3']; RELS_SS = [r for r in INV if r != 'swap']

# ===================== RQ1 =====================
fig, ax = panel('Change in accuracy (Enrico)')
strip(ax, INV, {r: {t: N['sens'][t][r]['d_acc'] * 100 for t in MOD} for r in INV}); ax.axhline(0, color=INK, lw=0.8)
ax.set_xticklabels([RL[r] for r in INV], rotation=60, ha='right'); ax.set_ylabel('Δ accuracy (pp)'); ax.set_ylim(-8, 8); save(fig, 'rq1_a', 'a')
fig, ax = panel('Flip rate (Enrico)'); fl = np.mean([N['sens'][t]['repeat1']['flip'] * 100 for t in MOD])
strip(ax, INV, {r: {t: N['sens'][t][r]['flip'] * 100 for t in MOD} for r in INV}, hollow={r: {t: N['sens'][t][r]['p_floor_holm'] >= 0.05 for t in MOD} for r in INV})
ax.axhline(fl, color=RED, lw=1.2, ls='--'); ax.text(8.4, fl + 0.6, 'mean flakiness', ha='right', color=RED)
ax.set_xticklabels([RL[r] for r in INV], rotation=60, ha='right'); ax.set_ylabel('Flip rate (%)'); ax.set_ylim(0, 22)
ax.legend(handles=[Line2D([], [], marker='o', color=GREY, ls='none', ms=6, label='above flakiness'), Line2D([], [], marker='o', mfc='none', color=GREY, ls='none', ms=6, label='not above')], loc='upper left'); save(fig, 'rq1_b', 'b')
fig, ax = panel('Flip rate (ScreenSpot)'); fls = np.mean([N['screenspot'][t]['repeat1']['flip'] * 100 for t in SS])
strip(ax, RELS_SS, {r: {t: N['screenspot'][t][r]['flip'] * 100 for t in SS} for r in RELS_SS}); ax.axhline(fls, color=RED, lw=1.2, ls='--')
ax.text(7.4, fls + 0.6, 'mean flakiness', ha='right', color=RED); ax.set_xticklabels([RL[r] for r in RELS_SS], rotation=60, ha='right'); ax.set_ylabel('Flip rate (%)'); ax.set_ylim(0, 22); save(fig, 'rq1_c', 'c')
fig, ax = panel('Enrico vs ScreenSpot')
for t in SS: nm, col, mk = MODEL_STYLE[t]; ax.scatter([N['sens'][t][r]['flip'] * 100 for r in RELS_SS], [N['screenspot'][t][r]['flip'] * 100 for r in RELS_SS], s=40, marker=mk, color=col, zorder=3)
ax.plot([0, 25], [0, 25], color=GREY, ls='--', lw=1); ax.set_xlim(0, 25); ax.set_ylim(0, 25); ax.set_xlabel('Enrico flip rate (%)'); ax.set_ylabel('ScreenSpot flip rate (%)')
ax.text(1.2, 22.5, f"ρ = {N['ss_corr']['spearman']:.2f}"); save(fig, 'rq1_d', 'd')
fig, ax = panel('Odds of a flip vs repeat (GEE)', left=0.24); e3 = EX['E3']['relation']; ks = sorted(e3, key=lambda k: e3[k]['or'])
for i, k in enumerate(ks): r = e3[k]; ax.plot([r['lo'], r['hi']], [i, i], color=INK, lw=1.4); ax.scatter(r['or'], i, s=40, color=RED, zorder=3)
ax.axvline(1, color=GREY, ls='--', lw=1); ax.set_xscale('log'); ax.set_xlim(0.8, 6); ax.set_xticks([1, 2, 3, 5]); ax.set_xticklabels(['1', '2', '3', '5']); ax.minorticks_off()
ax.set_yticks(range(len(ks))); ax.set_yticklabels([RL[k] for k in ks]); ax.set_xlabel('Odds ratio (95% CI)'); save(fig, 'rq1_e', 'e')
fig, ax = panel('Mean flip rate vs flakiness', left=0.33); y = np.arange(len(MOD))[::-1]
for i, t in enumerate(MOD):
    f0, f1 = N['sens'][t]['repeat1']['flip'] * 100, N['sens'][t]['summary']['mean_flip'] * 100; ax.plot([f0, f1], [y[i], y[i]], color=LGREY, lw=3, zorder=1)
    ax.scatter(f0, y[i], s=40, color=GREY, zorder=3); ax.scatter(f1, y[i], s=40, color=RED, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([SHORT[t] for t in MOD]); ax.set_xlabel('Flip rate (%)'); ax.set_xlim(-1, 27)
ax.legend(handles=[Line2D([], [], marker='o', color=GREY, ls='none', ms=6, label='flakiness (repeat)'), Line2D([], [], marker='o', color=RED, ls='none', ms=6, label='relations, mean')], loc='lower right'); save(fig, 'rq1_f', 'f')
fig, ax = panel('Least stable screen types', left=0.33)
tf = {tp: 100 * np.mean([m(t, s, 'original')['topic'] != m(t, s, r)['topic'] for t in MOD for s in screens_for(t) if SCR[s]['topic'] == tp for r in INV if m(t, s, r) and m(t, s, r)['topic'] and m(t, s, 'original')['topic'] and avail(s, r)] or [np.nan]) for tp in sorted({SCR[s]['topic'] for s in IDS})}
top = sorted(tf, key=lambda k: -tf[k])[:10][::-1]; ax.barh(range(10), [tf[k] for k in top], color=RED, height=0.65); ax.set_yticks(range(10)); ax.set_yticklabels([k.replace('mediaplayer', 'media player') for k in top])
ax.set_xlabel('Flip rate (%)'); ax.grid(axis='y', visible=False); save(fig, 'rq1_g', 'g')
fig, ax = panel('Set overlap with the original', left=0.33)
for i, t in enumerate(MOD):
    c = np.mean([N['sens'][t][r]['comp_jac'] for r in INV]); x = np.mean([N['sens'][t][r]['text_jac'] for r in INV]); ax.plot([c, x], [y[i], y[i]], color=LGREY, lw=3, zorder=1)
    ax.scatter(c, y[i], s=40, color=LAV, zorder=3); ax.scatter(x, y[i], s=40, color=GREY, zorder=3)
ax.set_yticks(y); ax.set_yticklabels([SHORT[t] for t in MOD]); ax.set_xlabel('Mean Jaccard index'); ax.set_xlim(0.6, 1.0)
ax.legend(handles=[Line2D([], [], marker='o', color=LAV, ls='none', ms=6, label='components'), Line2D([], [], marker='o', color=GREY, ls='none', ms=6, label='texts')], loc='upper left'); save(fig, 'rq1_h', 'h')

# ===================== RQ2 =====================
fam = [('downscale', ['75%', '50%', '25%'], 'Downscale', 'resolution kept'), ('jpeg', ['q50', 'q15', 'q5'], 'JPEG', 'quality'), ('blur', ['0.6', '1.2', '2.4'], 'Blur', 'σ (px)'),
       ('dim', ['×0.8', '×0.6', '×0.4'], 'Dim', 'brightness'), ('tilt', ['1°', '3°', '6°'], 'Rotation', 'rotation')]
for i, (fm, lv, ti, xl) in enumerate(fam):
    fig, ax = panel(ti)
    for t in LOCAL:
        L = N['dose'][t][fm]['levels']; nm, col, mk = MODEL_STYLE[t]
        ax.plot([0, 1, 2], [l['flip'] * 100 for l in L], '-', marker=mk, color=col, ms=7, lw=1.6)
        ax.fill_between([0, 1, 2], [l['flip_ci'][0] * 100 for l in L], [l['flip_ci'][1] * 100 for l in L], color=col, alpha=0.12, lw=0)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(lv); ax.set_xlabel(xl); ax.set_ylabel('Flip rate (%)'); ax.set_ylim(0, 30); ax.set_xlim(-0.3, 2.3)
    save(fig, f'rq2_{"abcde"[i]}', 'abcde'[i])
fig, ax = panel('Prompt and sampling'); keys = [('repeat_floor3', 'Repeat'), ('promptpara', 'Para.'), ('pipeline', 'Comp.'), ('sample_floor', 'Sample'), ('promptorder', 'Order')]
x = np.arange(len(keys)); w = 0.26
for i, t in enumerate(LOCAL): ax.bar(x + (i - 1) * w, [N['dose'][t]['extra'][k]['flip'] * 100 for k, _ in keys], w, color=MODEL_STYLE[t][1], edgecolor=INK, linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in keys], rotation=45, ha='right'); ax.set_ylabel('Flip rate (%)'); ax.set_ylim(0, 30); ax.grid(axis='x', visible=False); save(fig, 'rq2_f', 'f')
fig, ax = panel('Inference provider (Qwen3-VL-235B)'); pv = N['provider']
ax.bar([0, 1], [pv['same_flip'] * 100, pv['diff_flip'] * 100], 0.55, color=[GREY, RED], edgecolor=INK, linewidth=0.5)
for i, v in enumerate([pv['same_flip'], pv['diff_flip']]): ax.text(i, v * 100 + 0.4, f'{v*100:.1f}', ha='center')
ax.set_xticks([0, 1]); ax.set_xticklabels(['same', 'different']); ax.set_ylabel('Flip rate of relation pairs (%)'); [ax.text(i, 100*v + 0.5, f'n = {n:,}', ha='center') for i, (v, n) in enumerate([(pv['same_flip'], pv['same_n']), (pv['diff_flip'], pv['diff_n'])])]; ax.set_ylim(0, 16); ax.grid(axis='x', visible=False); save(fig, 'rq2_g', 'g')
fig, ax = panel('Status-bar crop', left=0.33); d54 = {'gpt5', 'qwen25vl7b', 'qwen3vl8b', 'internvl3'}; yy = np.arange(len(MOD))[::-1]   # model names on the y axis, clear of the panel letter
ax.barh(yy, [N['sens'][t]['statusbar']['flip'] * 100 for t in MOD], 0.65, color=[LAV if t in d54 else GREY for t in MOD], edgecolor=INK, linewidth=0.5)
ax.set_yticks(yy); ax.set_yticklabels([SHORT[t] for t in MOD]); ax.set_xlabel('Flip rate (%)'); ax.set_xlim(0, 30); ax.grid(axis='y', visible=False)
ax.legend(handles=[Patch(facecolor=GREY, edgecolor=INK, label='27 px'), Patch(facecolor=LAV, edgecolor=INK, label='54 px')], loc='lower right'); save(fig, 'rq2_h', 'h')

# ===================== RQ3 =====================
conds = [('original', 'Original'), ('textmask', 'Masked'), ('layout', 'Wireframe'), ('textonly', 'Text only')]
fig, ax = panel('Accuracy under the ablations')
strip(ax, [c for c, _ in conds], {c: {t: (N['base'][t]['acc'] if c == 'original' else N['probes'][t][c]['acc']) * 100 for t in MOD} for c, _ in conds})
ax.axhline(5, color=GREY, ls=':', lw=1); ax.set_xticklabels([l for _, l in conds], rotation=30, ha='right'); ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 70); save(fig, 'rq3_a', 'a')
TOP = sorted(N['topic_acc'], key=lambda k: -N['topic_acc'][k])
for letter, c, ttl in [('b', 'textonly', 'Text only vs original'), ('c', 'layout', 'Wireframe vs original')]:
    fig, ax = panel(ttl + ' (per type)')
    ax.scatter([N['probe_topic']['original'][k] * 100 for k in TOP], [N['probe_topic'][c][k] * 100 for k in TOP], s=40, color=RED if c == 'textonly' else LAV, edgecolor=INK, linewidth=0.5, zorder=3)
    for k in TOP:
        if k in (('list', 'gallery', 'camera') if c == 'textonly' else ('list', 'news', 'camera')): ax.annotate(k, (N['probe_topic']['original'][k] * 100, N['probe_topic'][c][k] * 100), textcoords='offset points', xytext=(4, 3))
    ax.plot([0, 100], [0, 100], color=GREY, ls='--', lw=1); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_xlabel('Original (%)'); ax.set_ylabel(('Text only' if c == 'textonly' else 'Wireframe') + ' (%)'); save(fig, 'rq3_' + letter, letter)
fig, ax = panel('Component recall'); comps = [k.split('|')[1] for k in N['comp_recall'] if k.startswith('original|')]
xs = [N['comp_recall']['original|' + c][0] * 100 for c in comps]; ys = [N['comp_recall']['textmask|' + c][0] * 100 for c in comps]
ax.scatter(xs, ys, s=40, color=LAV, edgecolor=INK, linewidth=0.5, zorder=3)
for c, a_, b_ in zip(comps, xs, ys):
    if c in ('Text Button', 'List Item', 'Multi-Tab'): ax.annotate(c, (a_, b_), textcoords='offset points', xytext=(6, -12), ha='left')
ax.plot([0, 100], [0, 100], color=GREY, ls='--', lw=1); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_xlabel('Original (%)'); ax.set_ylabel('Text masked (%)'); save(fig, 'rq3_d', 'd')

# ===================== RQ4 =====================
scores_ = [('conf', 'Conf.'), ('rep', 'Repeat'), ('mr10', 'Agree.'), ('cwc3', 'CWC-3'), ('cwc10', 'CWC-10')]
fig, ax = panel('AUROC of the scores')
strip(ax, [k for k, _ in scores_], {k: {t: N['rel'][t][k]['auroc'] for t in MOD if k in N['rel'][t] and N['rel'][t][k].get('auroc') is not None} for k, _ in scores_})
ax.get_lines()[-1].set_color(RED); ax.set_xticklabels([l for _, l in scores_], rotation=30, ha='right'); ax.set_ylabel('AUROC'); ax.set_ylim(0.45, 0.8); save(fig, 'rq4_a', 'a')
fig, ax = panel('Screens kept at α = 15%')
for t in MOD:
    c0, c1 = N['cov_alpha'][t]['0.15']['conf'][0] * 100, N['cov_alpha'][t]['0.15']['ours'][0] * 100; nm, col, mk = MODEL_STYLE[t]
    ax.plot([0, 1], [c0, c1], color=col, lw=1.4); ax.scatter([0, 1], [c0, c1], marker=mk, color=col, s=40, zorder=3)
ax.set_xticks([0, 1]); ax.set_xticklabels(['confidence', 'CWC gate']); ax.set_xlim(-0.3, 1.3); ax.set_ylabel('Screens kept (%)'); ax.set_ylim(-2, 70); save(fig, 'rq4_b', 'b')
fig, ax = panel('Verbalized confidence')
ok = [m(t, s, 'original')['conf'] for t in MOD for s in screens_for(t) if m(t, s, 'original')['ok'] and m(t, s, 'original')['conf'] is not None]
bad = [m(t, s, 'original')['conf'] for t in MOD for s in screens_for(t) if not m(t, s, 'original')['ok'] and m(t, s, 'original')['conf'] is not None]
bins = np.arange(40, 101, 5); ax.hist(ok, bins=bins, density=True, color=GREY, alpha=0.75, label='correct', edgecolor=INK, linewidth=0.4); ax.hist(bad, bins=bins, density=True, color=RED, alpha=0.6, label='wrong', edgecolor=INK, linewidth=0.4)
ax.set_xlabel('Confidence'); ax.set_ylabel('Density'); ax.legend(loc='upper left'); save(fig, 'rq4_c', 'c')
fig, ax = panel('Realized risk on held-out screens'); al = [0.10, 0.15, 0.20, 0.25]
for t in MOD: nm, col, mk = MODEL_STYLE[t]; ax.plot([a * 100 for a in al], [EX['E2'][t][str(a)]['mean_risk'] * 100 for a in al], '-', marker=mk, color=col, ms=6, lw=1.3)
ax.plot([8, 27], [8, 27], color=RED, ls='--', lw=1.2); ax.text(26.5, 2, 'dashed: risk = α', ha='right', color=RED); ax.set_xlabel('α (%)'); ax.set_ylabel('Realized risk (%)'); ax.set_xlim(8, 27); ax.set_ylim(-1, 28); save(fig, 'rq4_d', 'd')
fig, ax = panel('AUROC vs calls per screen'); ks_ = [1, 2, 3, 4, 6, 8, 10]
for t in MOD: nm, col, mk = MODEL_STYLE[t]; ax.plot(ks_, [EX['E1']['k'][t][str(k)] for k in ks_], '-', marker=mk, color=col, ms=5, lw=1.0, alpha=0.8)
ax.plot(ks_, [np.mean([EX['E1']['k'][t][str(k)] for t in MOD]) for k in ks_], '-', color=RED, lw=2.8, label='mean'); ax.set_xticks(ks_); ax.set_xlabel('Calls per screen'); ax.set_ylabel('AUROC'); ax.legend(loc='lower right'); save(fig, 'rq4_e', 'e')
for letter, t in [('f', 'gpt5mini'), ('g', 'qwen25vl7b')]:
    fig, ax = panel('Risk–coverage, ' + MODEL_STYLE[t][0]); r = N['rel'][t]
    for key, c, ls, lab in [('conf', GREY, '--', 'confidence'), ('cwc10', RED, '-', 'CWC-10')]:
        cv = np.array(r[key]['curve']); cov = np.linspace(1 / len(cv), 1, len(cv)); k2 = cov >= 0.05; ax.plot(cov[k2] * 100, (1 - cv[k2]) * 100, color=c, ls=ls, lw=1.8, label=lab)
    ax.set_xlabel('Coverage (%)'); ax.set_ylabel('Error of kept decisions (%)'); ax.set_ylim(0, 60); ax.set_xlim(0, 100); ax.legend(loc='lower right'); save(fig, 'rq4_' + letter, letter)
fig, ax = panel('CWC on both datasets', left=0.33); pos = np.arange(len(SS))
en = [[score(t, s, 'cwc', ['original'] + INV) for s in screens_for(t) if m(t, s, 'original') and m(t, s, 'original')['topic']] for t in SS]
ssv = [[score(t, s, 'cwc', ['original'] + RELS_SS) for s in EXTRA if m(t, s, 'original') and m(t, s, 'original')['topic']] for t in SS]
for data, off, col in [(en, -0.18, GREY), (ssv, 0.18, RED)]:
    b = ax.boxplot(data, positions=pos + off, widths=0.3, patch_artist=True, showfliers=False, vert=False)
    for pa in b['boxes']: pa.set_facecolor(col); pa.set_alpha(0.75); pa.set_edgecolor(INK)
    for k in ('whiskers', 'caps', 'medians'): [l.set_color(INK) for l in b[k]]
ax.set_yticks(pos); ax.set_yticklabels([SHORT[t] for t in SS]); ax.set_ylim(len(SS) + 0.75, -0.6); ax.set_xlabel('CWC'); ax.set_xlim(0, 1.05)   # free band under the last model for the legend
ax.legend(handles=[Patch(facecolor=GREY, edgecolor=INK, label='Enrico'), Patch(facecolor=RED, edgecolor=INK, label='ScreenSpot')], loc='lower center', ncol=2, columnspacing=0.8, handlelength=1.2); ax.grid(axis='y', visible=False); save(fig, 'rq4_h', 'h')
# one legend for all model colours and markers, set above each figure in the paper
lw = 6.925 * FS / 6.0; fig = plt.figure(figsize=(lw, 0.42)); fig.legend(handles=[Line2D([], [], marker=MODEL_STYLE[t][2], color=MODEL_STYLE[t][1], ls='none', ms=8, label=MODEL_STYLE[t][0]) for t in MOD], loc='center', ncol=8, frameon=False, columnspacing=1.4, handletextpad=0.3)
fig.savefig(f'{OUT}/legend_models.pdf'); fig.savefig(f'{OUT}/legend_models.png', dpi=110); plt.close(fig)
fig = plt.figure(figsize=(lw, 0.42)); fig.legend(handles=[Line2D([], [], marker=MODEL_STYLE[t][2], color=MODEL_STYLE[t][1], ls='-', ms=8, label=MODEL_STYLE[t][0]) for t in LOCAL], loc='center', ncol=3, frameon=False, columnspacing=1.6, handletextpad=0.3)
fig.savefig(f'{OUT}/legend_local.pdf'); plt.close(fig)
print('panels written to', OUT)
