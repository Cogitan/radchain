"""Erasure policy for a cross-chip [[4,1,2]] block: flag-all vs oracle vs soft.

Data: code-block Geant4 events (bench/rad_data/block, rad_code_block.load) and
physics loss curves (rad_apply_perase.perase_curve; P_loss = min(1, 2 P_flip)).

Per event (5 chips, deposits E_i): flagged F_i = [E_i >= 180 keV]; lost with
probability P_i = P_loss(E_i) (only flagged chips can be lost in this model:
P_loss below 180 keV is ~0 and is zeroed for consistency).
  FLAG-ALL     fail if sum F_i >= 2
  ORACLE       fail with P(>= 2 lost)
  SOFT-RANKED  fail with P(>= 2 lost) + sum_j P(only j lost) [j != argmax P]
  SOFT-RANDOM  fail with P(>= 2 lost) + sum_j P(only j lost) (1 - 1/n_flag)

    python bench/rad_erasure_policy.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import rad_apply_perase as ap
import rad_code_block as cb

HERE = os.path.dirname(os.path.abspath(__file__))
E_FLAG = 180.0
TAU = 1e-3
N = 4


def per_event(E, Pfn):
    F = (E >= E_FLAG)
    P = np.where(F, Pfn(E), 0.0)
    nflag = F.sum(axis=1)
    q = 1.0 - P
    prod_q = np.prod(q, axis=1)
    # P(only j lost) = P_j prod_{k != j} q_k
    with np.errstate(divide="ignore", invalid="ignore"):
        only = np.where(q > 0, P * (prod_q[:, None] / q), 0.0)
    # exact handling when some q_k == 0 (P_k == 1)
    zero = (q == 0)
    if zero.any():
        for r in np.where(zero.any(axis=1))[0]:
            for j in range(E.shape[1]):
                only[r, j] = P[r, j] * np.prod(np.delete(q[r], j))
    p_any = 1.0 - prod_q
    p_one = only.sum(axis=1)
    p_ge2 = np.clip(p_any - p_one, 0.0, 1.0)
    top = np.argmax(P, axis=1)
    wrong_ranked = p_one - only[np.arange(len(P)), top]
    wrong_random = p_one * np.where(nflag > 0, 1.0 - 1.0 / np.maximum(nflag, 1), 0.0)
    return {"flag2": (nflag >= 2).astype(float), "oracle": p_ge2,
            "ranked": p_ge2 + wrong_ranked, "random": p_ge2 + wrong_random,
            "lost_per_chip": P.sum(axis=1), "flag_per_chip": F.sum(axis=1)}


def layout_rates(L, Pfn):
    tot = {k: 0.0 for k in ("flag2", "oracle", "ranked", "random",
                            "lost_per_chip", "flag_per_chip")}
    for src, flux in (("muon", cb.MUON_FLUX), ("gamma-lines", cb.GFLUX)):
        ev, n, area = cb.load(L, src)
        if n == 0 or len(ev) == 0:
            continue
        w = flux * area / n * 3600.0          # per hour
        r = per_event(ev, Pfn)
        for k in tot:
            tot[k] += r[k].sum() * w
    lam_loss = tot["lost_per_chip"] / 5
    lam_flag = tot["flag_per_chip"] / 5
    h = TAU / 3600.0                           # tau in hours
    ind_loss = N * lam_loss * (N + 1) * lam_loss * h
    ind_flag = N * lam_flag * (N + 1) * lam_flag * h
    misattr = N * lam_loss * N * lam_flag * h
    fail = {"FLAG-ALL": tot["flag2"] + ind_flag,
            "ORACLE": tot["oracle"] + ind_loss,
            "SOFT-RANKED": tot["ranked"] + ind_loss + misattr,
            "SOFT-RANDOM": tot["random"] + ind_loss + misattr}
    return lam_loss, lam_flag, fail


def main():
    out = {}
    for tag in ("d7_conv0.25_echo", "d7_conv0.5_echo", "d7_conv1.0_echo",
                "d7_conv0.5_noecho"):
        P, _ = ap.perase_curve(tag)
        Pfn = lambda E, P=P: np.minimum(1.0, 2.0 * P(E))
        print(f"\n{tag}")
        for L in ("X1", "XQ50", "XQ100"):
            lam_loss, lam_flag, fail = layout_rates(L, Pfn)
            line = f"  {L:6s} lam_loss {lam_loss:.3f}/h lam_flag {lam_flag:.1f}/h |"
            for k, v in fail.items():
                line += f" {k} {lam_loss / v if v else float('inf'):9.3g}x"
            print(line + ("   [XQ100: muons only]" if L == "XQ100" else ""))
            out.setdefault(tag, {})[L] = {"lam_loss_h": lam_loss,
                                          "lam_flag_h": lam_flag,
                                          "fail_per_h": fail}
    json.dump(out, open(os.path.join(HERE, "results", "rad_erasure_policy.json"),
                        "w"), indent=1)


if __name__ == "__main__":
    main()
