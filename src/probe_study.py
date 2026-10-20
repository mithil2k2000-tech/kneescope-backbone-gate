"""Follow-up analyses on KneeScope's frozen C0 features (validation split only; the test split stays sealed).

Inputs: concat(CLS, mean patch token) features extracted by KneeScope for three frozen backbones on the
OAI-derived knee KL dataset (Chen 2018), with KneeScope's patient-level split.

A  Reproduction: multinomial logistic-regression probe, C chosen on validation (as C0 did) and, more
   conservatively, by 5-fold patient-grouped CV on the training split.
B  Label efficiency: probes trained on patient-level subsamples of the training split (1-100%, 5 draws).
C  Fusion: concatenated DINOv2 + RAD-DINO features.
D  Selective prediction and calibration: expected calibration error and risk-coverage of each probe.
E  Bilateral dependence: how correlated are errors on a patient's two knees (why bootstraps must resample patients).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "data" / "features"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
ARMS = {"imagenet_resnet50": "ImageNet ResNet-50", "rad_dino": "RAD-DINO", "dinov2_base": "DINOv2 ViT-B"}
GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
H = "224px__96ba6ba5a149"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(arm, split):
    z = np.load(F / f"{arm}__{split}__{H}.npz", allow_pickle=True)
    order = np.argsort(z["image_ids"].astype(str), kind="stable")  # arms were cached in different orders
    return (z["features"][order].astype(np.float32), z["labels"][order].astype(int),
            z["patient_ids"][order].astype(str), z["image_ids"][order].astype(str))


def qwk(y, p):
    return cohen_kappa_score(y, p, weights="quadratic")


def probe(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def patient_boot(y, pred, pids, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    up = np.unique(pids)
    idx_by = {p: np.flatnonzero(pids == p) for p in up}
    vals = []
    for _ in range(B):
        idx = np.concatenate([idx_by[p] for p in rng.choice(up, len(up))])
        vals.append(fn(y[idx], pred[idx]))
    return np.quantile(vals, [0.025, 0.975])


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def aurc(conf, correct):
    order = np.argsort(-conf)
    risk = np.cumsum(1 - correct[order]) / np.arange(1, len(order) + 1)
    return float(risk.mean())


def main():
    data = {a: (load(a, "train"), load(a, "val")) for a in ARMS}
    # Same images, same order, for every arm?
    for a in ARMS:
        assert (data[a][0][3] == data["dinov2_base"][0][3]).all() and (data[a][1][3] == data["dinov2_base"][1][3]).all()
    (Xtr_d, ytr, ptr, _), (Xva_d, yva, pva, iva) = data["dinov2_base"]
    data["fusion"] = ((np.hstack([data["dinov2_base"][0][0], data["rad_dino"][0][0]]), ytr, ptr, None),
                      (np.hstack([data["dinov2_base"][1][0], data["rad_dino"][1][0]]), yva, pva, iva))
    arms = list(ARMS) + ["fusion"]
    facts = {"n_train": int(len(ytr)), "n_val": int(len(yva)), "patients_train": int(len(set(ptr))),
             "patients_val": int(len(set(pva))), "val_class_counts": np.bincount(yva).tolist(),
             "train_class_counts": np.bincount(ytr).tolist()}

    # ---------------- A: reproduction and CV-selected C
    rows, preds = [], {}
    gkf = GroupKFold(5)
    for a in arms:
        (Xtr, _, _, _), (Xva, _, _, _) = data[a]
        val_curve = {}
        for C in GRID:
            m = probe(C).fit(Xtr, ytr)
            val_curve[C] = qwk(yva, m.predict(Xva))
        C_val = max(val_curve, key=val_curve.get)
        cv_curve = {}
        for C in GRID:
            s = []
            for tr, te in gkf.split(Xtr, ytr, ptr):
                m = probe(C).fit(Xtr[tr], ytr[tr])
                s.append(qwk(ytr[te], m.predict(Xtr[te])))
            cv_curve[C] = float(np.mean(s))
        C_cv = max(cv_curve, key=cv_curve.get)
        m = probe(C_cv).fit(Xtr, ytr)
        P = m.predict_proba(Xva)
        pred = P.argmax(1)
        preds[a] = (P, pred)
        lo, hi = patient_boot(yva, pred, pva, qwk)
        early = np.isin(yva, [0, 1])
        p01 = P[early][:, 1] / (P[early][:, 0] + P[early][:, 1])
        conf = P.max(1)
        corr = (pred == yva).astype(float)
        rows.append({"arm": a, "C_val": C_val, "qwk_valC": val_curve[C_val], "C_cv": C_cv, "qwk_cvC": qwk(yva, pred),
                     "qwk_lo": lo, "qwk_hi": hi, "bal_acc": balanced_accuracy_score(yva, pred),
                     "mae": float(np.abs(pred - yva).mean()), "gross_err": float((np.abs(pred - yva) >= 2).mean()),
                     "kl1_recall": float((pred[yva == 1] == 1).mean()), "auroc_kl0_kl1": roc_auc_score(yva[early] == 1, p01),
                     "ece": ece(conf, corr), "aurc": aurc(conf, corr),
                     "aurc_expected_score": aurc(-(np.abs(np.arange(5)[None] - (P @ np.arange(5))[:, None]).min(1)), corr)})
        log("A", a, rows[-1]["qwk_valC"], rows[-1]["qwk_cvC"])
    A = pd.DataFrame(rows)
    A.to_csv(RES / "a_probes.csv", index=False)

    # Paired patient bootstrap vs ImageNet with CV-selected C.
    rng = np.random.default_rng(1)
    up = np.unique(pva)
    idx_by = {p: np.flatnonzero(pva == p) for p in up}
    boots = {a: [] for a in arms if a != "imagenet_resnet50"}
    for _ in range(2000):
        idx = np.concatenate([idx_by[p] for p in rng.choice(up, len(up))])
        base = qwk(yva[idx], preds["imagenet_resnet50"][1][idx])
        for a in boots:
            boots[a].append(qwk(yva[idx], preds[a][1][idx]) - base)
    paired = {a: {"delta": float(qwk(yva, preds[a][1]) - qwk(yva, preds["imagenet_resnet50"][1])),
                  "lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975))} for a, v in boots.items()}
    # DINOv2 vs RAD-DINO directly.
    dv = []
    for _ in range(2000):
        idx = np.concatenate([idx_by[p] for p in rng.choice(up, len(up))])
        dv.append(qwk(yva[idx], preds["dinov2_base"][1][idx]) - qwk(yva[idx], preds["rad_dino"][1][idx]))
    paired["dinov2_minus_rad_dino"] = {"delta": float(qwk(yva, preds["dinov2_base"][1]) - qwk(yva, preds["rad_dino"][1])),
                                       "lo": float(np.quantile(dv, 0.025)), "hi": float(np.quantile(dv, 0.975))}
    facts["paired"] = paired

    # ---------------- B: label efficiency
    rows = []
    up_tr = np.unique(ptr)
    idx_tr = {p: np.flatnonzero(ptr == p) for p in up_tr}
    C_of = dict(zip(A.arm, A.C_cv))
    for frac in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        for draw in range(5 if frac < 1 else 1):
            r = np.random.default_rng(100 + draw)
            pick = r.choice(up_tr, max(5, int(round(frac * len(up_tr)))), replace=False)
            idx = np.concatenate([idx_tr[p] for p in pick])
            if len(np.unique(ytr[idx])) < 5:
                continue
            for a in arms:
                (Xtr, _, _, _), (Xva, _, _, _) = data[a]
                m = probe(C_of[a]).fit(Xtr[idx], ytr[idx])
                rows.append({"frac": frac, "draw": draw, "n_images": len(idx), "arm": a, "qwk": qwk(yva, m.predict(Xva))})
        log("B", frac)
    pd.DataFrame(rows).to_csv(RES / "b_label_efficiency.csv", index=False)

    # ---------------- E: bilateral dependence
    df = pd.DataFrame({"pid": pva, "img": iva, "y": yva, "pred": preds["dinov2_base"][1]})
    df["side"] = df.img.str.extract(r"(\d+)([LR])\.png$")[1]
    w = df.pivot_table(index="pid", columns="side", values=["y", "pred"])
    w = w.dropna()
    errL = (w["pred"]["L"] != w["y"]["L"]).astype(int)
    errR = (w["pred"]["R"] != w["y"]["R"]).astype(int)
    facts["bilateral"] = {"patients_with_both": int(len(w)),
                          "kl_same_both_knees": float((w["y"]["L"] == w["y"]["R"]).mean()),
                          "kl_qwk_left_vs_right": float(qwk(w["y"]["L"].astype(int), w["y"]["R"].astype(int))),
                          "error_rate": float(np.r_[errL, errR].mean()),
                          "p_err_R_given_err_L": float(errR[errL == 1].mean()),
                          "p_err_R_given_ok_L": float(errR[errL == 0].mean()),
                          "error_phi": float(np.corrcoef(errL, errR)[0, 1])}
    # Naive (image-level) vs patient-level bootstrap CI width for DINOv2.
    rng = np.random.default_rng(2)
    naive = [qwk(yva[i], preds["dinov2_base"][1][i]) for i in (rng.integers(0, len(yva), len(yva)) for _ in range(1000))]
    lo, hi = patient_boot(yva, preds["dinov2_base"][1], pva, qwk)
    facts["ci_width"] = {"image_level": float(np.quantile(naive, 0.975) - np.quantile(naive, 0.025)),
                         "patient_level": float(hi - lo)}
    (RES / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
    print(A.round(4).to_string())
    print(json.dumps({k: v for k, v in facts.items() if k in ("paired", "bilateral", "ci_width")}, indent=1))


if __name__ == "__main__":
    main()
