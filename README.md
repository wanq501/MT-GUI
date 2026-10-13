# MT-GUI: Metamorphic Testing of Multimodal LLMs as GUI Test Oracles

MT-GUI tests multimodal large language models (MLLMs) that act as GUI test oracles. It derives follow-up screenshots and prompts from metamorphic relations, tests every relation against the flakiness baseline of a repeated query, and gates verdicts with a consistency-weighted confidence (CWC) calibrated by conformal risk control.

This repository contains the experiment console, the complete raw answers of all eight models, and the analysis code that reproduces every number, table and figure of the paper.

## Reproduction package for peer review

This package contains everything needed to reproduce every number, table and figure of the manuscript: the complete raw answers of all eight models and the analysis code. The experiment console that generated the queries is not part of this package and will be released with the full repository upon publication.

| Path | Content |
|---|---|
| `analysis/data/MTGUI_results.json.gz` | All 53,719 raw model answers with timestamps, token counts, providers and costs |
| `analysis/analysis.py` | Main analysis: accuracy, flip rates, McNemar and Holm tests, GEE, CWC and the conformal gate |
| `analysis/figs_rq.py` | Figure panels of RQ1 to RQ4 |
| `analysis/extra_analyses.py`, `si_more.py`, `joint_analysis*.py` | Supplementary analyses and tables |
| `analysis/revision_analyses.py` | Equivalence tests, per-model GEE fits, trend tests, CWC ablations, the expected-type check and the serving settings |
| `analysis/dataflow.py` | Per-model data flow |
| `analysis/lib/` | Shared statistics helpers |
| `analysis/out/` | Precomputed results for comparison |

## Reproducing the analysis

```bash
pip install -r requirements.txt
cd analysis
python analysis.py            # about 5 minutes on the first run
python figs_rq.py
python extra_analyses.py
python revision_analyses.py
python dataflow.py
```

The outputs match the files in `analysis/out/`.

## License

All rights reserved. Provided for peer review only. See `LICENSE`.
