"""v4 build step 6a: the neural challenger, exactly as designed in the design round.

30 -> 64 (LayerNorm, SiLU, Dropout 0.15) -> 32 (LayerNorm, SiLU, Dropout 0.10) -> 16 (SiLU) -> 1.
Loss: non-negative PU risk (Kiryo et al. 2017; beta = 0, gamma = 1, logistic loss) under three
assumed positive fractions (0.003, 0.005, 0.008) plus 0.10 * E_N[BCE(0)] on the confirmed
negatives. AdamW 3e-4 / weight decay 1e-3, batches of 32 positives + 512 unlabelled + 128
confirmed negatives, ten-epoch linear warm-up then cosine to 3e-5, at most 200 epochs,
patience 25 on the inner validation region's case-control PR-AUC, gradient clip 1.0, five
seeds per prior (each seed rotates a different inner validation region), 15 networks per
outer fold averaged. Same five macroregion outer folds as the trees; thresholds from the
training regions only. Inputs standardised by training-region median and IQR (label-free).

Outputs: recovery/mlp_oof_scores.parquet, recovery/mlp_results.json
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
T0 = time.time()
torch.set_num_threads(4)

man = json.load(open(os.path.join(OUT, "feature_manifest.json")))
FEATURES = man["features"]
REGIONS = ["CEERS", "GOODS-N", "GOODS-S", "COSMOS", "UDS"]
PRIORS = [0.003, 0.005, 0.008]
SEEDS = [0, 1, 2, 3, 4]
BATCH = (32, 512, 128)
STEPS_PER_EPOCH = 40
MAX_EPOCHS, PATIENCE, WARMUP = 200, 25, 10
LR, LR_MIN, WD = 3e-4, 3e-5, 1e-3
ANCHOR = 0.10

d = pd.read_parquet(os.path.join(OUT, "features.parquet"))
d = d[d.in_support.values].reset_index(drop=True)
n = len(d)
X = d[FEATURES].values.astype(np.float32)
y = d.y.values
region = d.region.values
is_pos = y == 1
is_neg = y == 0
is_unl = np.isnan(y) & ~d.ambiguous.values.astype(bool)
# twin rows of any labelled or ambiguous object (overlapping tiles, same sky group) never enter the unlabelled pool
_lab_groups = np.unique(d.sky_group.values[is_pos | is_neg | d.ambiguous.values.astype(bool)])
is_unl = is_unl & ~np.isin(d.sky_group.values, _lab_groups)
picked = d.picked.values.astype(bool)
strict = d.y_strict.values.astype(bool)


class Net(nn.Module):
    def __init__(self, p=30):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(p, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.f(x).squeeze(-1)


bce = nn.BCEWithLogitsLoss(reduction="mean")


def nnpu_loss(zP, zU, zN, prior):
    ones, zeros = torch.ones_like(zP), torch.zeros_like(zU)
    r_p_pos = prior * bce(zP, ones)
    r_p_neg = prior * bce(zP, torch.zeros_like(zP))
    r_u_neg = bce(zU, zeros)
    neg_risk = r_u_neg - r_p_neg
    anchor = ANCHOR * bce(zN, torch.zeros_like(zN))
    if neg_risk.item() < 0.0:  # beta = 0, gamma = 1
        return -neg_risk + anchor
    return r_p_pos + neg_risk + anchor


def lr_at(epoch):
    if epoch < WARMUP:
        return LR * (epoch + 1) / WARMUP
    t = (epoch - WARMUP) / max(MAX_EPOCHS - WARMUP, 1)
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + np.cos(np.pi * t))


def train_net(tr_mask, val_mask, prior, seed, scale):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    P = np.where(tr_mask & is_pos)[0]
    U = np.where(tr_mask & is_unl)[0]
    N = np.where(tr_mask & is_neg)[0]
    vP = np.where(val_mask & is_pos)[0]
    vN = np.where(val_mask & is_neg)[0]
    Xt = torch.from_numpy((X - scale[0]) / scale[1])
    v_rows = np.concatenate([vP, vN])
    v_lab = np.r_[np.ones(len(vP)), np.zeros(len(vN))]
    net = Net(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    best, best_state, bad = -1.0, None, 0
    for ep in range(MAX_EPOCHS):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep)
        net.train()
        for _ in range(STEPS_PER_EPOCH):
            bP = rng.choice(P, BATCH[0], replace=True)
            bU = rng.choice(U, BATCH[1], replace=False)
            bN = rng.choice(N, BATCH[2], replace=len(N) < BATCH[2])
            opt.zero_grad()
            loss = nnpu_loss(net(Xt[bP]), net(Xt[bU]), net(Xt[bN]), prior)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            s = net(Xt[v_rows]).numpy()
        ap = average_precision_score(v_lab, s)
        if ap > best + 1e-4:
            best, bad = ap, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best, ep + 1


def score(nets, rows, scale):
    Xt = torch.from_numpy((X[rows] - scale[0]) / scale[1])
    with torch.no_grad():
        S = np.stack([torch.sigmoid(net(Xt)).numpy() for net in nets])
    return S.mean(axis=0), S.std(axis=0)


oof_mean = np.full(n, np.nan)
oof_std = np.full(n, np.nan)
thr = {k: np.full(n, np.nan) for k in ("t_burden", "t_neg1", "t_05")}
results = {"folds": {}}
for R in REGIONS:
    T = [x for x in REGIONS if x != R]
    trm = np.isin(region, T)
    tem = region == R
    f_T = float(picked[trm].mean())
    med = np.median(X[trm], axis=0)
    iqr = np.subtract(*np.percentile(X[trm], [75, 25], axis=0))
    iqr = np.where(iqr > 1e-6, iqr, 1.0).astype(np.float32)
    scale = (med.astype(np.float32), iqr)
    nets, log, frac_neg1 = [], [], []
    for prior in PRIORS:
        for seed in SEEDS:
            val = T[seed % len(T)]
            trm_i = np.isin(region, [x for x in T if x != val])
            valm = region == val
            net, ap, eps = train_net(
                trm_i,
                valm,
                prior,
                1000 * REGIONS.index(R) + int(prior * 1e4) * 10 + seed,
                scale,
            )
            nets.append(net)
            # the inner validation region also gives the catalog fraction at 1% negative rate
            vP, vN = np.where(valm & is_pos)[0], np.where(valm & is_neg)[0]
            vS = np.random.default_rng(seed).choice(
                np.where(valm)[0], 20000, replace=False
            )
            sN, _ = score([net], vN, scale)
            sS, _ = score([net], vS, scale)
            frac_neg1.append(float((sS >= np.quantile(sN, 0.99)).mean()))
            log.append(
                dict(
                    prior=prior,
                    seed=seed,
                    val_region=val,
                    val_prauc=round(float(ap), 4),
                    epochs=eps,
                )
            )
            print(
                f"  {R} prior {prior} seed {seed} val {val}: PR-AUC {ap:.3f} after {eps} epochs ({time.time() - T0:.0f}s)",
                flush=True,
            )
    te_rows, tr_rows = np.where(tem)[0], np.where(trm)[0]
    m_te, s_te = score(nets, te_rows, scale)
    m_tr, _ = score(nets, tr_rows, scale)
    f_neg1 = float(np.mean(frac_neg1))
    t_b, t_n, t_5 = (
        float(np.quantile(m_tr, 1 - f_T)),
        float(np.quantile(m_tr, 1 - f_neg1)),
        float(np.quantile(m_tr, 0.995)),
    )
    oof_mean[te_rows], oof_std[te_rows] = m_te, s_te
    thr["t_burden"][te_rows], thr["t_neg1"][te_rows], thr["t_05"][te_rows] = (
        t_b,
        t_n,
        t_5,
    )
    P, N = tem & is_pos, tem & is_neg
    results["folds"][R] = dict(
        networks=log,
        union_burden_train=f_T,
        frac_for_neg1pct=f_neg1,
        t_burden=t_b,
        t_neg1=t_n,
        t_05=t_5,
        selected_at_burden=int((m_te >= t_b).sum()),
        union_selected_in_region=int((picked & tem).sum()),
        positives=int(P.sum()),
        recall_at_burden=int((oof_mean[P] >= t_b).sum()),
        union_recall=int((picked & P).sum()),
        rule_missed=int((P & ~picked).sum()),
        rule_missed_recovered=int((oof_mean[P & ~picked] >= t_b).sum()),
        negatives=int(N.sum()),
        neg_selected_at_burden=int((oof_mean[N] >= t_b).sum()),
        case_control_prauc=round(
            float(
                average_precision_score(
                    np.r_[np.ones(P.sum()), np.zeros(N.sum())],
                    np.r_[oof_mean[P], oof_mean[N]],
                )
            ),
            4,
        ),
        seconds=round(time.time() - T0),
    )
    fr = results["folds"][R]
    print(
        f"MLP {R}: recall {fr['recall_at_burden']}/{fr['positives']} (union {fr['union_recall']}) | rule-missed {fr['rule_missed_recovered']}/{fr['rule_missed']} | neg {fr['neg_selected_at_burden']}/{fr['negatives']} | PR-AUC {fr['case_control_prauc']}",
        flush=True,
    )
    json.dump(results, open(os.path.join(OUT, "mlp_results.json"), "w"), indent=1)

oof = d[["source_id", "region", "y", "y_strict", "picked"]].copy()
oof["score_mean"], oof["score_std"] = oof_mean, oof_std
for k, v in thr.items():
    oof[k] = v
oof["sel_burden"] = oof.score_mean >= oof.t_burden
oof["sel_neg1"] = oof.score_mean >= oof.t_neg1
oof["sel_05"] = oof.score_mean >= oof.t_05
oof.to_parquet(os.path.join(OUT, "mlp_oof_scores.parquet"), index=False)
sel = oof.sel_burden.values
results["pooled"] = {
    "recall_at_burden": int((sel & is_pos).sum()),
    "union_recall": int((picked & is_pos).sum()),
    "rule_missed_recovered": int((sel & is_pos & ~picked).sum()),
    "rule_missed": int((is_pos & ~picked).sum()),
    "strict_recovered": int((sel & strict).sum()),
    "strict_missed_recovered": int((sel & strict & ~picked).sum()),
    "neg_selected_at_burden": int((sel & is_neg).sum()),
    "negatives": int(is_neg.sum()),
    "selected_at_burden": int(sel.sum()),
    "recall_at_05": int((oof.sel_05.values & is_pos).sum()),
    "case_control_prauc_pooled": round(
        float(
            average_precision_score(
                np.r_[np.ones(is_pos.sum()), np.zeros(is_neg.sum())],
                np.r_[oof_mean[is_pos], oof_mean[is_neg]],
            )
        ),
        4,
    ),
    "seconds_total": round(time.time() - T0),
}
json.dump(results, open(os.path.join(OUT, "mlp_results.json"), "w"), indent=1)
print("MLP pooled:", json.dumps(results["pooled"]), flush=True)
