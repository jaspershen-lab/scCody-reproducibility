# Current manuscript checkpoints

| Model | Selected epoch/phase, folds 0/1/2 |
|---|---|
| scCody-G | 30 / 24 / 28 |
| scCody-CT | 29 / 26 / 30 |
| scNODE | 5 / 6 / 11 |
| scDiffEq | 500 / 500 / 475 |
| ARTEMIS | 19 / 17 / 16 |
| PsychAD scCody-G/CT | epoch 5 |

The AIDA/comparator selections use internal holdout_L_gene. ARTEMIS values are phases.
The eight scCody weights are distributed as `scCody_weights.zip` on [Figshare](https://figshare.com/articles/software/scCody_Model_Weight/34951348?file=71249662). Extract the ZIP in the parent directory of the sibling
checkouts:

```bash
unzip scCody_weights.zip
```

All checkpoints use one directory:

```text
scCody-reproducibility/
└── checkpoints/
    ├── AIDA_scCody-G_fit0.pt
    ├── AIDA_scCody-G_fit1.pt
    ├── AIDA_scCody-G_fit2.pt
    ├── AIDA_scCody-CT_fit0.pt
    ├── AIDA_scCody-CT_fit1.pt
    ├── AIDA_scCody-CT_fit2.pt
    ├── PsychAD_scCody-G_fit0.pt
    └── PsychAD_scCody-CT_fit0.pt
```

The AIDA ordered development-only panel is `configs/aida_genes.tsv`. PsychAD
uses its own ordered gene panel and preprocessing. The weight files are excluded
from Git.

All checkpoints include representation and transport weights, runtime settings,
the ordered panel, age coordinates and donor splits. Healthy-reference
embeddings are not included: call `model.set_reference(healthy_adata)` before
localization, or use `run_pipeline --phase attach` for the AIDA models.
Encoding, transport, decoding and generation do not require those references.
Model runs are not repeated by the figure renderer.
