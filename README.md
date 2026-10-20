# KneeScope backbone gate: a radiograph foundation model did not transfer better than a generic one

Pre-registered frozen linear-probe comparison for knee osteoarthritis Kellgren–Lawrence grading, from the KneeScope
research prototype (not a medical device). Paper: [`paper/main.pdf`](paper/main.pdf).

8,260 OAI-derived knee radiographs, 4,130 patients, leakage-audited patient-level split, validation only (test sealed).

| Frozen backbone | QWK [95% CI, patient bootstrap] | Δ vs ImageNet |
|---|---|---|
| ImageNet ResNet-50 | 0.602 [0.546, 0.654] | — |
| RAD-DINO (chest radiographs) | 0.629 [0.577, 0.677] | +0.027 [−0.023, +0.073] |
| DINOv2 ViT-B (natural images) | 0.686 [0.636, 0.732] | +0.084 [+0.040, +0.127] |
| DINOv2 + RAD-DINO | 0.682 [0.631, 0.724] | +0.080 [+0.037, +0.122] |

* DINOv2 beat RAD-DINO directly: +0.057 [+0.012, +0.100].
* DINOv2 led at every training-set size; with 25% of the training images it matched RAD-DINO trained on all of them.
* Every probe failed on the doubtful grade (KL1 recall 0.11–0.19).
* Errors on a patient's two knees are correlated, so confidence intervals must resample patients.

## Reproduce

The cached features come from KneeScope's extraction step and are derived from OAI data, so they are not redistributed.
With them in `data/features/`:

```bash
pip install -r requirements.txt
python src/probe_study.py
python src/figures.py
cd paper && latexmk -pdf main.tex
```

## AI assistance

An AI assistant (Claude, Anthropic) helped write code and draft text. The author designed the project and the
pre-registered rule, ran the feature extraction, reviewed all results and is responsible for the content.

## Licence

MIT (code). Data remain under their original terms.
