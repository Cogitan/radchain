"""Erasure policy versus the burst-detection (flag) threshold.

The policy study uses one flag threshold, 180 keV
(Willow's inferred value). A higher threshold flags fewer healthy chips, which
helps FLAG-ALL, but lets losses below it go unflagged. This sweeps it.

Model (generalises rad_erasure_policy.per_event, which assumed losses happen
only on flagged chips): per event, chip i is lost with P_i = P_loss(E_i) for
ALL deposits, and flagged if E_i >= T. The [[4,1,2]] outer code (d = 2)
detects a single error but cannot locate it, so a loss on an UNFLAGGED chip
fails every policy except the oracle (conservative: a soft policy guessing
among all five chips would sometimes be right).

  ORACLE       P(>= 2 lost)
  FLAG-ALL     1 if n_F >= 2, else P(any unflagged chip lost)
  SOFT-RANKED  P(>= 2 lost) + sum_j P(only j lost) [j unflagged or j != top flagged deposit]
  SOFT-RANDOM  P(>= 2 lost) + sum_j P(only j lost) [1 if unflagged, else 1 - 1/n_F]

Independent and misattribution terms as in rad_erasure_policy, with
lam_loss counting ALL losses (so it does not move with T) and lam_flag(T).
Protection is lam_loss over the failure rate, as there.

`--legacy` reproduces rad_erasure_policy exactly (losses zeroed below T) at
T = 180 keV, as a check of this code.

    python analysis/rad_policy_threshold.py [--legacy]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import rad_apply_perase as ap
import rad_code_block as cb

HERE = os.path.dirname(os.path.abspath(__file__))
TAU_H = 1e-3 / 3600.0
N = 4
TAGS = ("d7_conv0.25_echo", "d7_conv0.5_echo", "d7_conv1.0_echo", "d7_conv0.5_noecho")
THRESHOLDS = (100.0, 180.0, 300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0)


def per_event(E, Pfn, T, legacy=False, floor=None):
    F = E >= T
    P = Pfn(E)
    if legacy:
        P = np.where(F, P, 0.0)
    if floor is not None:
        # stated assumption: deposits below `floor` keV never destroy the
        # inner logical (bursts too small to detect at Willow's 180 keV are
        # treated as harmless); the loss curve there is unresolved from zero
        P = np.where(E >= floor, P, 0.0)
    nF = F.sum(axis=1)
    q = 1.0 - P
    prod_q = np.prod(q, axis=1)
    only = np.empty_like(P)
    for j in range(P.shape[1]):
        only[:, j] = P[:, j] * np.prod(np.delete(q, j, axis=1), axis=1)
    p_one = only.sum(axis=1)
    p_ge2 = np.clip(1.0 - prod_q - p_one, 0.0, 1.0)
    # top flagged deposit (index), -1 if none flagged
    Ef = np.where(F, E, -1.0)
    top = np.where(nF > 0, np.argmax(Ef, axis=1), -1)
    unflag = ~F
    blamed = np.zeros_like(F)
    has = top >= 0
    blamed[np.where(has)[0], top[has]] = True
    wrong_ranked = (only * (unflag | ~blamed)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        miss_rand = np.where(unflag, 1.0, 1.0 - 1.0 / np.maximum(nF, 1)[:, None])
    wrong_random = (only * miss_rand).sum(axis=1)
    p_any_unflag = 1.0 - np.prod(np.where(unflag, q, 1.0), axis=1)
    flagall = np.where(nF >= 2, 1.0, p_any_unflag)
    return {"FLAG-ALL": flagall, "ORACLE": p_ge2, "SOFT-RANKED": p_ge2 + wrong_ranked,
            "SOFT-RANDOM": p_ge2 + wrong_random,
            "lost": P.sum(axis=1), "flag": F.sum(axis=1).astype(float)}


def layout(L, Pfn, T, legacy=False, floor=None):
    tot = {}
    for src, flux in (("muon", cb.MUON_FLUX), ("gamma-lines", cb.GFLUX)):
        ev, n, area = cb.load(L, src)
        if n == 0 or len(ev) == 0:
            continue
        w = flux * area / n * 3600.0
        for k, v in per_event(ev, Pfn, T, legacy, floor).items():
            tot[k] = tot.get(k, 0.0) + v.sum() * w
    lam_loss, lam_flag = tot["lost"] / 5, tot["flag"] / 5
    ind_loss = N * lam_loss * (N + 1) * lam_loss * TAU_H
    ind_flag = N * lam_flag * (N + 1) * lam_flag * TAU_H
    mis = N * lam_loss * N * lam_flag * TAU_H
    fail = {"FLAG-ALL": tot["FLAG-ALL"] + ind_flag, "ORACLE": tot["ORACLE"] + ind_loss,
            "SOFT-RANKED": tot["SOFT-RANKED"] + ind_loss + mis,
            "SOFT-RANDOM": tot["SOFT-RANDOM"] + ind_loss + mis}
    return lam_loss, lam_flag, {k: lam_loss / v for k, v in fail.items()}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--legacy", action="store_true")
    a.add_argument("--floor", type=float, default=None,
                   help="zero P_loss below this deposit (keV): sub-detection bursts harmless")
    a.add_argument("--floor-at-threshold", action="store_true",
                   help="floor = max(180 keV, T): every unflagged burst is harmless "
                        "(a lower bound on harm)")
    a = a.parse_args()
    curves = {}
    for t in TAGS:
        P, _ = ap.perase_curve(t)
        curves[t] = (lambda E, P=P: np.minimum(1.0, 2.0 * P(E)))
    out = {}
    thr = (180.0,) if a.legacy else THRESHOLDS
    for t in TAGS:
        print(f"\n{t}")
        print(f"  {'T keV':>6s} {'layout':6s} {'lam_flag/h':>10s} {'FLAG-ALL':>9s} "
              f"{'RANDOM':>8s} {'RANKED':>8s} {'ORACLE':>9s}")
        for T in thr:
            for L in ("X1", "XQ50"):
                fl = max(180.0, T) if a.floor_at_threshold else a.floor
                ll, lf, prot = layout(L, curves[t], T, a.legacy, fl)
                print(f"  {T:6.0f} {L:6s} {lf:10.2f} {prot['FLAG-ALL']:9.3g} "
                      f"{prot['SOFT-RANDOM']:8.3g} {prot['SOFT-RANKED']:8.3g} "
                      f"{prot['ORACLE']:9.3g}   (lam_loss {ll:.3f}/h)")
                out.setdefault(t, {}).setdefault(str(T), {})[L] = {
                    "lam_loss_h": ll, "lam_flag_h": lf, "protection": prot}
    if not a.legacy:
        name = "rad_policy_threshold" + ("_floorT" if a.floor_at_threshold else
                                         f"_floor{a.floor:.0f}" if a.floor else "") + ".json"
        os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
        path = os.path.join(HERE, "results", name)
        json.dump(out, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
