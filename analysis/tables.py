"""LaTeX tables from out/numbers.json."""
import json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); N = json.load(open(f'{ROOT}/out/numbers.json'))
M = N['models']; NM = N['name']; LOCAL = ['qwen25vl7b', 'qwen3vl8b', 'internvl3']
INV = ['downscale', 'jpeg', 'blur', 'dim', 'grayscale', 'bezel', 'tilt', 'swap', 'statusbar']
RL = {'downscale': 'Downscale', 'jpeg': 'JPEG', 'blur': 'Blur', 'dim': 'Dim', 'grayscale': 'Grayscale', 'bezel': 'Bezel', 'tilt': 'Tilt', 'swap': 'Swap', 'statusbar': 'Status bar'}
ACCESS = {'gpt5': 'closed, API', 'gpt5mini': 'closed, API', 'haiku': 'closed, API', 'gemini': 'closed, API', 'qwen3vl': 'open weights, API', 'qwen25vl7b': 'open weights, local', 'qwen3vl8b': 'open weights, local', 'internvl3': 'open weights, local'}
COND = {'gpt5': '14', 'gpt5mini': '14 + 10', 'haiku': '14 + 10', 'gemini': '14 + 10', 'qwen3vl': '14', 'qwen25vl7b': '34 + 10', 'qwen3vl8b': '34 + 10', 'internvl3': '34'}
def p(x, d=1): return f'{100 * x:.{d}f}'
def ps(x): return '$<$0.001' if x < 0.001 else f'{x:.3f}'
def w(name, body): open(f'{ROOT}/out/tables/{name}.tex', 'w', encoding='utf-8').write(body)

# ---------------- Table 1: models and baseline comprehension
rows = []
for t in M:
    b = N['base'][t]
    rows.append(f"{NM[t]} & {ACCESS[t]} & {COND[t]} & {b['n']} & {p(b['acc'])} [{p(b['acc_ci'][0])}, {p(b['acc_ci'][1])}] & {b['macro_f1']:.3f} & {b['comp_f1']:.3f} & {b['text_f1']:.3f} & {p(b['json_rate'])} & {b['latency_med']:.1f} & {b['cost_per_1k']:.2f} \\\\")
w('t1_models', r"""\begin{tabular}{llcccccccrr}
\toprule
Model & Access & Conditions & Screens & Screen-type accuracy, \% [95\% CI] & Macro-F1 & Component F1 & Text F1 & Valid JSON, \% & Latency, s & Cost per 1,000, \$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")

# ---------------- Table 2: instability summary
rows = []
for t in M:
    s = N['sens'][t]; sm = s['summary']
    cj = np.mean([s[c]['comp_jac'] for c in INV]); tj = np.mean([s[c]['text_jac'] for c in INV])
    ex = N['dose'][t]['extra'] if t in LOCAL else None
    rows.append(f"{NM[t]} & {p(sm['floor'])} & {p(sm['mean_flip'])} ({p(sm['min_flip'])}--{p(sm['max_flip'])}) & {RL[sm['max_rel']]} & {sm['above_floor']}/9 & {100 * sm['max_abs_dacc']:.1f} & {ps(sm['min_acc_p_holm'])} & {cj:.2f} & {tj:.2f} & "
                + (f"{p(ex['promptorder']['flip'])} & {p(ex['sample_floor']['flip'])}" if ex else '-- & --') + r' \\')
w('t2_instability', r"""\begin{tabular}{lcccccccccc}
\toprule
 & Repeat floor & Flip rate, mean & Most sensitive & Relations & Max $|\Delta$acc$|$ & Smallest & Component & Text & Label order & Sampling \\
Model & (\%) & (range), \% & relation & above floor & (pp) & Holm $p$ & Jaccard & Jaccard & flip, \% & flip, \% \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")

# ---------------- Table 3: reliability scores and the gate
rows = []
for t in M:
    r = N['rel'][t]; rk = N['rank'][t]
    au = {k: r[k]['auroc'] for k in r if k in ('conf', 'mr10', 'cwc3', 'cwc10', 'sc')}; best = max(au, key=au.get)
    f = lambda k: ('--' if k not in r else (r'\textbf{' + f"{r[k]['auroc']:.3f}" + '}' if k == best else f"{r[k]['auroc']:.3f}"))
    c15 = N['cov_alpha'][t]['0.15']; c20 = N['cov_alpha'][t]['0.2']
    rows.append(f"{NM[t]} & {f('conf')} & {f('sc')} & {f('mr10')} & {f('cwc3')} & {f('cwc10')} & {p(r['conf']['acc80'])} / {p(r['cwc10']['acc80'])} & {p(c15['conf'][0], 0)} / {p(c15['ours'][0], 0)} & {p(c20['conf'][0], 0)} / {p(c20['ours'][0], 0)} & {p(c15['ours'][1])} & {rk['rank']} \\\\")
w('t3_reliability', r"""\begin{tabular}{lccccccccc c}
\toprule
 & \multicolumn{5}{c}{AUROC of the reliability score} & Accuracy at & \multicolumn{2}{c}{Screens kept, confidence / ours (\%)} & Realised & Rank of \\
\cmidrule(lr){2-6} \cmidrule(lr){8-9}
Model & Confidence & Sampling & Agreement & CWC-3 & CWC-10 & 80\% cov., \% & risk $\le$ 15\% & risk $\le$ 20\% & risk, \% & CWC-10 \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")

# ---------------- Supplementary: per-relation detail
for t in M:
    pass
rows = []
for t in M:
    s = N['sens'][t]
    for c in INV:
        r = s[c]; rows.append(f"{NM[t]} & {RL[c]} & {r['n']} & {p(r['acc'])} & {100 * r['d_acc']:+.1f} & {ps(r['p_acc_holm'])} & {p(r['flip'])} [{p(r['flip_ci'][0])}, {p(r['flip_ci'][1])}] & {ps(r['p_floor_holm'])} & {r['comp_jac']:.2f} & {r['text_jac']:.2f} \\\\")
w('s1_relations', r"""\begin{longtable}{llrrrrrrrr}
\toprule
Model & Relation & Pairs & Accuracy, \% & $\Delta$acc (pp) & Holm $p$ (acc) & Flip rate, \% [95\% CI] & Holm $p$ (vs floor) & Comp. Jaccard & Text Jaccard \\
\midrule
\endhead
""" + '\n'.join(rows) + r"""
\bottomrule
\end{longtable}""")
rows = []
for t in LOCAL:
    for fam, d in N['dose'][t].items():
        if fam == 'extra': continue
        for lv in d['levels']: rows.append(f"{NM[t]} & {RL[fam]} & {lv['cond']} & {lv['n']} & {p(lv['flip'])} [{p(lv['flip_ci'][0])}, {p(lv['flip_ci'][1])}] & {100 * lv['d_acc']:+.1f} & {ps(d['p_q'])} \\\\")
    for k, nm in [('pipeline', 'Compound'), ('promptorder', 'Label order'), ('promptpara', 'Paraphrase'), ('repeat_floor3', 'Repeats (3)'), ('sample_floor', 'Samples (5)')]:
        e = N['dose'][t]['extra'][k]; rows.append(f"{NM[t]} & {nm} & {k.replace('_', ' ')} & {e['n']} & {p(e['flip'])} [{p(e['flip_ci'][0])}, {p(e['flip_ci'][1])}] & {100 * e.get('d_acc', 0):+.1f} & -- \\\\")
w('s2_dose', r"""\begin{longtable}{llllrrr}
\toprule
Model & Family & Condition & Pairs & Flip rate, \% [95\% CI] & $\Delta$acc (pp) & Cochran's $Q$ $p$ \\
\midrule
\endhead
""" + '\n'.join(rows) + r"""
\bottomrule
\end{longtable}""")
rows = []
for t in N['ss_models']:
    s = N['screenspot'][t]
    for c in ['repeat1'] + [x for x in INV if x != 'swap']:
        r = s[c]; rows.append(f"{NM[t]} & {'Repeat' if c == 'repeat1' else RL[c]} & {r['n']} & {p(r['flip'])} [{p(r['flip_ci'][0])}, {p(r['flip_ci'][1])}] & {p(N['sens'][t][c]['flip'])} & {r['comp_jac']:.2f} \\\\")
w('s3_screenspot', r"""\begin{longtable}{llrrrr}
\toprule
Model & Relation & Pairs & Flip rate, ScreenSpot, \% [95\% CI] & Flip rate, Enrico, \% & Comp. Jaccard \\
\midrule
\endhead
""" + '\n'.join(rows) + r"""
\bottomrule
\end{longtable}""")
rows = []
for t in M:
    pr = N['probes'][t]; rows.append(f"{NM[t]} & {p(N['base'][t]['acc'])} & " + ' & '.join(f"{p(pr[c]['acc'])} ({100 * pr[c]['d_acc']:+.1f}; {ps(pr[c]['p_holm'])})" for c in ['textmask', 'layout', 'textonly']) + r' \\')
w('s4_probes', r"""\begin{tabular}{lcccc}
\toprule
Model & Original, \% & Text masked, \% ($\Delta$; Holm $p$) & Wireframe, \% ($\Delta$; Holm $p$) & Text only, \% ($\Delta$; Holm $p$) \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")
rows = []
comps = sorted({k.split('|')[1] for k in N['comp_recall']}, key=lambda c: -N['comp_recall'].get('original|' + c, (0, 0))[1])
for c in comps:
    o = N['comp_recall'].get('original|' + c)
    if not o or o[1] < 40: continue
    rows.append(f"{c} & {o[1]} & {p(o[0], 0)} & " + ' & '.join(p(N['comp_recall'][f'{k}|{c}'][0], 0) if f'{k}|{c}' in N['comp_recall'] else '--' for k in INV + ['textmask']) + r' \\')
w('s5_components', r"""\begin{tabular}{lrr""" + 'r' * 10 + r"""}
\toprule
Component & Support & Original & """ + ' & '.join(RL[k] for k in INV) + r""" & Text masked \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}""")
print('tables written')
