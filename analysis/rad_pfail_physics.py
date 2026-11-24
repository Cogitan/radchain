"""Physics replacement for the "erased if E >= E*" proxy: P_erase(E).


Chain, per event (deposit E at strike (x0, y0) on a 12 mm Willow-class die
carrying a distance-d rotated surface code):
  1. Phonon transport (a radchain.phonon backend; default: recorded G4CMP
     kernels, stock geometry at --thickness) gives the fraction of phonon
     energy absorbed in each qubit's cell; peak shift df_i = K * E * frac_i.
     K (kHz per keV absorbed in a 1.15 mm cell) is the DEVICE's response,
     fixed across designs: qp_shift.K_WILLOW_KHZ_PER_KEV by default (fitted
     so Willow's median burst peaks at 2 MHz on the stock 0.381 mm kernel),
     or --k, or --fit-k-reference to refit it on the no-trap reference kernel.
     It is never refitted on the design's own (trapped) transport, which
     would cancel the traps' effect. K depends on the backend: a K fitted
     with one backend must not be reused with another.
  2. Recovery: df_i(t) = df_i / (1 + t / t_i), t_i = 1 / (r x0_i),
     x0_i = df_i / (a fq), r = 1/(88 ns), a = 0.77, fq = 6 GHz
     (recovery_google2025.py, arXiv:2506.18228).
  3. Error per round: excess detection eps(df) = 0.02 (df / 1 MHz)^1.32 WITH
     echo (2% at 1 MHz, 5% at 2 MHz, arXiv:2506.18228); per-qubit error
     p = eps * conv, conv = 0.5 nominal, bracketed 0.25 and 1.0. NO-ECHO
     variant: eps = 0.17 (df / 1 MHz)^1.32 (17% at 1 MHz), same exponent
     ASSUMED.
  4. Stim rotated_memory_z, base circuit noise 1e-3, 1 us rounds. Burst noise
     per qubit per segment: DEPOLARIZE1 on data qubits at round start and
     X_ERROR on measure qubits before measurement, evaluated at the segment's
     midpoint. Segments grow geometrically (20 rounds x 1.25^k) until every
     qubit's burst error is under 2e-4. Decoder: PyMatching on the BASE error
     model (oblivious, what ships). P_fail = (f_burst - f_base) / (1 - f_base).

    python analysis/rad_pfail_physics.py [--d 7] [--events 150] [--shots 1000]
                                      [--conv 0.5] [--noecho] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

K_KHZ_PER_KEV = None            # set in _init: fitted, or --k
PITCH_MM = 1.15
R_RECOMB_S = 88e-9
A_COEFF = 0.77
FQ_HZ = 6.0e9
P_BASE = 1e-3
ROUND_S = 1e-6
DIE_MM = 12.0
P_STOP = 2e-4                   # burst error below which the window ends
MAX_ROUNDS = 9000               # hard cap on the window (see segments)
MARK_DATA = 0.000999            # marker values to find insertion points
MARK_MEAS = 0.000998


def eps_excess(df_hz: np.ndarray, echo: bool = True) -> np.ndarray:
    a = 0.02 if echo else 0.17
    return a * np.power(np.clip(df_hz, 0, None) / 1e6, 1.32)


def base_circuit(d: int, rounds: int):
    import stim
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=P_BASE,
        before_round_data_depolarization=MARK_DATA,
        before_measure_flip_probability=MARK_MEAS,
        after_reset_flip_probability=P_BASE)


def layout(d: int):
    """Qubit index -> physical (x, y) mm, from stim coordinates.

    Stim's rotated layout puts nearest neighbours sqrt(2) apart; scale so they
    are PITCH_MM apart, centred on the die."""
    c = base_circuit(d, 1)
    coords = c.get_final_qubit_coordinates()
    q = sorted(coords)
    xy = np.array([coords[i][:2] for i in q], float) * (PITCH_MM / math.sqrt(2))
    xy -= xy.mean(axis=0)
    return q, xy


def cell_fractions(hx, hy, n_ph, x0, y0, xy):
    """Fraction of phonons absorbed in each qubit's cell.

    The qubits sit on a square lattice rotated 45 deg (stim's rotated layout),
    nearest neighbours PITCH_MM apart. Each qubit's Voronoi cell is the unit
    cell of that lattice, area PITCH_MM^2, the cell K is defined on.
    A hit is assigned by rounding its rotated lattice coordinates."""
    x = hx + x0
    y = hy + y0
    h = DIE_MM / 2
    on = (np.abs(x) <= h) & (np.abs(y) <= h)
    x, y = x[on], y[on]
    s = PITCH_MM / math.sqrt(2)
    su = (xy[:, 0] + xy[:, 1]) / (2 * s)
    sv = (xy[:, 0] - xy[:, 1]) / (2 * s)
    ou, ov = su[0] - round(su[0]), sv[0] - round(sv[0])
    iu = np.rint((x + y) / (2 * s) - ou).astype(np.int64)
    iv = np.rint((x - y) / (2 * s) - ov).astype(np.int64)
    pairs, cnt = np.unique(np.stack([iu, iv], axis=1), axis=0,
                           return_counts=True)
    cells = {(int(a), int(b)): int(c) for (a, b), c in zip(pairs, cnt)}
    return np.array([cells.get((int(round(a - ou)), int(round(b - ov))), 0)
                     for a, b in zip(su, sv)], float) / n_ph


def segments(df0: np.ndarray, conv: float, echo: bool):
    """[(n_rounds, p_data_per_qubit)] until burst error < P_STOP everywhere."""
    x0 = np.clip(df0, 1.0, None) / (A_COEFF * FQ_HZ)
    tstar = 1.0 / (x0 / R_RECOMB_S)
    out = []
    t = 0.0
    n = 20
    while True:
        tm = (t + n / 2 * ROUND_S)
        df = df0 / (1.0 + tm / tstar)
        p = np.clip(eps_excess(df, echo) * conv, 0, 0.5)
        # Stop when the burst is negligible, or at MAX_ROUNDS. The cap
        # truncates the recombination tail for strong conversions (biases
        # P_fail slightly LOW) and bounds memory: 2x the rounds OOM'd 6
        # workers on 2026-10-15.
        if p.max() < P_STOP or t / ROUND_S >= MAX_ROUNDS:
            break
        out.append((n, p))
        t += n * ROUND_S
        n = int(math.ceil(n * 1.25))
    return out


def burst_circuit(d: int, segs, q_index: dict):
    """Base circuit over the burst's rounds with per-qubit burst noise added."""
    import stim
    rounds = sum(n for n, _ in segs)
    base = base_circuit(d, rounds).flattened()
    per_round = []
    for n, p in segs:
        per_round += [p] * n
    out = stim.Circuit()
    rd = rm = 0
    for inst in base:
        out.append(inst)
        if inst.name == "DEPOLARIZE1" and abs(inst.gate_args_copy()[0] - MARK_DATA) < 1e-9:
            p = per_round[min(rd, rounds - 1)]
            _append_grouped(out, "DEPOLARIZE1",
                            [t.value for t in inst.targets_copy()], p, q_index)
            rd += 1
        elif inst.name == "X_ERROR" and abs(inst.gate_args_copy()[0] - MARK_MEAS) < 1e-9:
            p = per_round[min(rm, rounds - 1)]
            _append_grouped(out, "X_ERROR",
                            [t.value for t in inst.targets_copy()], p, q_index)
            rm += 1
    return out, rounds


def _append_grouped(c, gate, qubits, p, q_index):
    """Add `gate` with each qubit's own probability, grouped into log bins."""
    groups = {}
    for qb in qubits:
        pq = float(p[q_index[qb]])
        if pq < 1e-5:
            continue
        key = round(math.log10(pq), 2)
        groups.setdefault(key, []).append(qb)
    for key, qs in groups.items():
        c.append(gate, qs, min(10 ** key, 0.5))


_CACHE = {}


def logical_fail(circ, base_rounds_circ, shots, seed):
    import pymatching
    key = base_rounds_circ.num_detectors, base_rounds_circ.num_qubits
    if key not in _CACHE:
        dem = base_rounds_circ.detector_error_model(decompose_errors=True)
        _CACHE[key] = pymatching.Matching.from_detector_error_model(dem)
    m = _CACHE[key]
    # Batches of 100: one batch is 100 x ~3.4e5 detectors (~34 MB). Sampling
    # all shots at once with 20 workers exhausted memory (2026-10-15).
    sampler = circ.compile_detector_sampler(seed=seed)
    wrong = 0
    done = 0
    while done < shots:
        n = min(100, shots - done)
        det, obs = sampler.sample(n, separate_observables=True)
        pred = m.decode_batch(det)
        wrong += int(np.sum(pred[:, 0] != obs[:, 0]))
        done += n
    return wrong / shots


def one_event(args):
    (i, e_kev, x0, y0, d, shots, conv, echo, seed) = args
    hits = _HITS
    q, xy = _LAYOUT
    q_index = {qb: k for k, qb in enumerate(q)}
    frac = cell_fractions(hits[0], hits[1], _NPH, x0, y0, xy)
    df0 = K_KHZ_PER_KEV * 1e3 * e_kev * frac          # Hz
    segs = segments(df0, conv, echo)
    if not segs:
        return {"i": i, "e_keV": e_kev, "x0": x0, "y0": y0, "rounds": 0,
                "p_fail": 0.0, "f_burst": None, "f_base": None,
                "peak_khz": float(df0.max() / 1e3)}
    circ, rounds = burst_circuit(d, segs, q_index)
    base = base_circuit(d, rounds)
    fb = logical_fail(circ, base, shots, seed)
    f0 = logical_fail(base, base, shots, seed + 1)
    pf = max(0.0, (fb - f0) / (1.0 - f0)) if f0 < 1 else 0.0
    return {"i": i, "e_keV": e_kev, "x0": x0, "y0": y0, "rounds": rounds,
            "p_fail": pf, "f_burst": fb, "f_base": f0,
            "peak_khz": float(df0.max() / 1e3),
            "n_over_1MHz": int((df0 > 1e6).sum())}


def transport_hits(nph, seed, backend, phonon):
    from radchain import phonon as ph
    return ph.get_backend(backend).film_hits(n_phonons=nph, seed=seed, **phonon)


def fit_k_willow(hits, nph, seed=7):
    """K such that Willow's median burst peaks at 2 MHz with THIS transport.

    Call it with the hits of Willow's REFERENCE transport (no back-side traps),
    never with a trapped design's hits: K is the device's response per keV
    absorbed, and refitting it on a trapped kernel would scale the traps'
    suppression back out.
    """
    from radchain import qubit_footprint as qf
    pool = qf.detected_mixed_pool(40_000, np.random.default_rng(seed))
    return qf.grade(qf.Geometry(), None, n_events=2000, n_phonons=nph,
                    hits=hits, deposits_keV=pool).k_khz_per_keV


def _init(d, nph, seed, backend, phonon, k):
    global _HITS, _LAYOUT, _NPH, K_KHZ_PER_KEV
    _HITS = transport_hits(nph, seed, backend, phonon)
    _LAYOUT = layout(d)
    _NPH = nph
    K_KHZ_PER_KEV = k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=7)
    ap.add_argument("--events", type=int, default=150)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--conv", type=float, default=0.5)
    ap.add_argument("--noecho", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--nph", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--backend", default=None,
                    help="phonon backend name (default: g4cmp-kernel)")
    ap.add_argument("--thickness", type=float, default=0.381)
    ap.add_argument("--trap-pitch", type=float, default=0.0)
    ap.add_argument("--trap-coverage", type=float, default=0.0)
    ap.add_argument("--trap-absorption", type=float, default=0.0)
    ap.add_argument("--nearest", action="store_true",
                    help="use the nearest simulated G4CMP configuration")
    ap.add_argument("--k", type=float, default=None,
                    help="kHz per keV per 1.15 mm cell; default: "
                         "qp_shift.K_WILLOW_KHZ_PER_KEV")
    ap.add_argument("--fit-k-reference", type=float, default=None,
                    metavar="THICKNESS_MM",
                    help="refit K on the no-trap reference kernel at this "
                         "thickness (the device's response is fixed; traps "
                         "in the design are NOT refitted away)")
    a = ap.parse_args()
    phonon = {"thickness_mm": a.thickness, "trap_pitch_mm": a.trap_pitch,
              "trap_coverage": a.trap_coverage,
              "trap_absorption": a.trap_absorption, "nearest": a.nearest}
    k = a.k
    if k is None and a.fit_k_reference is not None:
        ref = {"thickness_mm": a.fit_k_reference, "nearest": a.nearest}
        k = fit_k_willow(transport_hits(a.nph, a.seed, a.backend, ref), a.nph)
        print(f"  K fitted to Willow's 2 MHz median peak on the no-trap "
              f"reference at {a.fit_k_reference} mm: {k:.2f} kHz/keV")
    elif k is None:
        if a.backend not in (None, "g4cmp-kernel"):
            raise SystemExit("--k or --fit-k-reference is required with a "
                             "non-default backend: qp_shift's K was fitted "
                             "with g4cmp-kernel")
        from radchain import qp_shift
        k = qp_shift.K_WILLOW_KHZ_PER_KEV
        print(f"  K = qp_shift.K_WILLOW_KHZ_PER_KEV = {k:.2f} kHz/keV "
              "(fitted on the stock 0.381 mm kernel)")
    rng = np.random.default_rng(a.seed)
    es = np.exp(rng.uniform(np.log(30.0), np.log(5000.0), a.events))
    h = DIE_MM / 2
    xs = rng.uniform(-h, h, a.events)
    ys = rng.uniform(-h, h, a.events)
    jobs = [(i, float(es[i]), float(xs[i]), float(ys[i]), a.d, a.shots,
             a.conv, not a.noecho, 1000 + 7 * i) for i in range(a.events)]
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.d, a.nph, a.seed, a.backend, phonon, k)) as p:
        rows = []
        for r in p.imap_unordered(one_event, jobs):
            rows.append(r)
            if len(rows) % 10 == 0:
                print(f"  {len(rows)}/{a.events} events, {time.time() - t0:.0f}s",
                      flush=True)
    rows.sort(key=lambda r: r["i"])
    tag = a.tag or f"d{a.d}_conv{a.conv}_{'noecho' if a.noecho else 'echo'}"
    out = {"d": a.d, "shots": a.shots, "conv": a.conv, "echo": not a.noecho,
           "p_base": P_BASE, "k_khz_per_keV": k,
           "backend": a.backend or "g4cmp-kernel", "phonon": phonon, "events": rows, "seconds": time.time() - t0}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"rad_pfail_{tag}.json")
    # atomic: a run killed mid-write must not leave a file that looks finished
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, path)
    # summary: P_erase in log-E bins
    E = np.array([r["e_keV"] for r in rows])
    P = np.array([r["p_fail"] for r in rows])
    edges = np.geomspace(30, 5000, 9)
    print(f"\n{tag}: {path}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (E >= lo) & (E < hi)
        if m.any():
            print(f"  E {lo:7.0f}-{hi:7.0f} keV: n {m.sum():3d}  mean P_fail "
                  f"{P[m].mean():.3f}  (max {P[m].max():.3f})")


if __name__ == "__main__":
    main()
