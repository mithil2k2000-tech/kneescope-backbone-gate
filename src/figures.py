"""Figure and tables for the knee paper."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, TAB = ROOT / "results", ROOT / "paper" / "figures", ROOT / "paper" / "tables"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
NAME = {"imagenet_resnet50": "ImageNet ResNet-50", "rad_dino": "RAD-DINO", "dinov2_base": "DINOv2 ViT-B",
        "fusion": "DINOv2 + RAD-DINO"}
COL = {"imagenet_resnet50": "#7f7f7f", "rad_dino": "#c0504d", "dinov2_base": "#1f4e79", "fusion": "#2e8b57"}

b = pd.read_csv(RES / "b_label_efficiency.csv")
fig, ax = plt.subplots(figsize=(4.2, 2.6))
for a in NAME:
    g = b[b.arm == a].groupby("n_images").qwk
    m, s = g.mean(), g.std().fillna(0)
    ax.errorbar(m.index, m.values, yerr=s.values, marker="o", ms=3, lw=1.1, capsize=2, color=COL[a], label=NAME[a],
                ls="--" if a == "fusion" else "-")
ax.set_xscale("log")
ax.set_xlabel("Training images (patient-level subsample)")
ax.set_ylabel("Validation QWK")
ax.legend(fontsize=6, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "label_efficiency.pdf")
plt.close(fig)

a = pd.read_csv(RES / "a_probes.csv")
f = json.loads((RES / "facts.json").read_text())
rows = []
for _, r in a.iterrows():
    p = f["paired"].get(r.arm)
    rows.append({"Backbone": NAME[r.arm], "QWK [95\\% CI]": f"{r.qwk_cvC:.3f} [{r.qwk_lo:.3f}, {r.qwk_hi:.3f}]",
                 "$\\Delta$ vs ImageNet": "--" if p is None else f"{p['delta']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]",
                 "KL1 recall": f"{r.kl1_recall:.3f}",
                 "KL0/1 AUROC": f"{r.auroc_kl0_kl1:.3f}", "ECE": f"{r.ece:.3f}", "AURC": f"{r.aurc:.3f}"})
(TAB / "main.tex").write_text(
    "\\begin{table*}[t]\n\\centering\n\\caption{Linear probes on frozen features, validation split (826 knees, 413 patients). "
    "Confidence intervals resample patients; $\\Delta$ is a paired bootstrap against the ImageNet probe on the same patients. "
    "AURC is the area under the risk--coverage curve using the top-class probability (lower is better).}\n\\label{tab:main}\n"
    "\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n" + pd.DataFrame(rows).to_latex(index=False, escape=False, column_format="lrrrrrr")
    + "\\end{table*}\n")
t = b.pivot_table(index="n_images", columns="arm", values="qwk", aggfunc="mean")[list(NAME)]
t.columns = ["ImageNet", "RAD-DINO", "DINOv2", "Fusion"]
t.index.name = "Training images"
(TAB / "label_eff.tex").write_text(
    "\\begin{table}[t]\n\\centering\n\\caption{Mean validation QWK by training-set size (5 patient-level draws below 100\\%).}\n"
    "\\label{tab:label}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n" + t.reset_index().to_latex(index=False, column_format="rrrrr", float_format="%.3f")
    + "\\end{table}\n")
print(t.round(3))
