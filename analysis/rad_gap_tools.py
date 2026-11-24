"""Complementary-gap soft output for a Stim rotated surface-code memory.

Used by the gap-attribution study.
The technique is the one in Gidney, Newman, Brooks, Jones, "Yoked surface
codes" (arXiv:2312.04522): turn the logical observable into one extra
detector and decode twice, with that detector forced to 0 and to 1. The gap is
the weight difference |w0 - w1|, the log-likelihood ratio between the best
corrections in the two logical classes.

That needs every observable-flipping error to touch exactly ONE detector, or
the converted errors become 3-node hyperedges matching cannot represent.
Stim's generated memory puts the observable on an interior row, so
`retarget_observable` moves it to a boundary row of data qubits (an
equivalent logical: rows differ by Z stabilizers, all measured), choosing
whichever boundary row makes every observable edge single-detector.
`validate` checks this and the two decoding identities before any use.
"""
from __future__ import annotations

import numpy as np
import stim
import pymatching


def _final_data_measure(circ: stim.Circuit):
    """(index of the final M instruction, its qubit targets)."""
    last = None
    for i, inst in enumerate(circ):
        if isinstance(inst, stim.CircuitInstruction) and inst.name == "M":
            last = i
    if last is None:
        raise ValueError("no final M")
    return last, [t.value for t in circ[last].targets_copy()]


def retarget_observable(circ: stim.Circuit, row: str) -> stim.Circuit:
    """Copy of a FLATTENED circuit with OBSERVABLE_INCLUDE(0) moved to the data
    qubits of one boundary row: row in {'xmin','xmax','ymin','ymax'}."""
    coords = circ.get_final_qubit_coordinates()
    mi, qubits = _final_data_measure(circ)
    xy = np.array([coords[q][:2] for q in qubits])
    ax = 0 if row[0] == "x" else 1
    target = xy[:, ax].min() if row.endswith("min") else xy[:, ax].max()
    sel = [j for j in range(len(qubits)) if abs(xy[j, ax] - target) < 1e-6]
    n_after = sum(len(inst.targets_copy()) for inst in circ[mi + 1:]
                  if isinstance(inst, stim.CircuitInstruction)
                  and inst.name in ("M", "MR", "MX", "MY", "MZ"))
    if n_after:
        raise ValueError("measurements after the final data M")
    n = len(qubits)
    out = stim.Circuit()
    for inst in circ:
        if isinstance(inst, stim.CircuitInstruction) and inst.name == "OBSERVABLE_INCLUDE":
            continue
        out.append(inst)
    out.append("OBSERVABLE_INCLUDE",
               [stim.target_rec(-(n - j)) for j in sel], 0)
    return out


def _components(dem: stim.DetectorErrorModel):
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comp = []
        for t in list(inst.targets_copy()) + [None]:
            if t is None or t.is_separator():
                dets = [x.val for x in comp if x.is_relative_detector_id()]
                ob = any(x.is_logical_observable_id() for x in comp)
                yield inst.args_copy()[0], dets, ob
                comp = []
            else:
                comp.append(t)


def observable_edges_single(circ: stim.Circuit) -> bool:
    dem = circ.detector_error_model(decompose_errors=True)
    return all(len(d) <= 1 for _, d, ob in _components(dem) if ob)


def choose_row(base: stim.Circuit) -> str:
    for row in ("ymin", "ymax", "xmin", "xmax"):
        if observable_edges_single(retarget_observable(base, row)):
            return row
    raise ValueError("no boundary row makes the observable single-detector")


def gap_dem(circ: stim.Circuit) -> stim.DetectorErrorModel:
    """DEM with L0 replaced by one extra detector D_N (N = num_detectors)."""
    dem = circ.detector_error_model(decompose_errors=True).flattened()
    N = dem.num_detectors
    lines = []
    for inst in dem:
        if inst.type == "error":
            parts = []
            for t in inst.targets_copy():
                if t.is_separator():
                    parts.append("^")
                elif t.is_logical_observable_id():
                    parts.append(f"D{N}")
                else:
                    parts.append(f"D{t.val}")
            lines.append(f"error({inst.args_copy()[0]}) " + " ".join(parts))
    lines.append(f"detector D{N}")
    return stim.DetectorErrorModel("\n".join(lines))


class GapDecoder:
    """Ordinary prediction + complementary gap, both from the BASE model."""

    def __init__(self, base_circ: stim.Circuit):
        dem = base_circ.detector_error_model(decompose_errors=True)
        self.m = pymatching.Matching.from_detector_error_model(dem)
        self.g = pymatching.Matching.from_detector_error_model(gap_dem(base_circ))
        self.n = dem.num_detectors

    def decode(self, det: np.ndarray):
        pred, w = self.m.decode_batch(det, return_weights=True)
        z = np.zeros((det.shape[0], 1), dtype=det.dtype)
        _, w0 = self.g.decode_batch(np.hstack([det, z]), return_weights=True)
        _, w1 = self.g.decode_batch(np.hstack([det, 1 - z]), return_weights=True)
        return pred[:, 0], w, w0, w1


def validate(d: int = 5, rounds: int = 10, shots: int = 2000, p: float = 3e-3):
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p
    ).flattened()
    row = choose_row(base)
    circ = retarget_observable(base, row)
    gd = GapDecoder(circ)
    det, obs = circ.compile_detector_sampler(seed=1).sample(
        shots, separate_observables=True)
    pred, w, w0, w1 = gd.decode(det)
    wmin = np.minimum(w0, w1)
    ok_w = np.allclose(wmin, w, rtol=1e-6, atol=1e-6)
    pred_gap = (w1 < w0).astype(int)
    ok_p = float(np.mean(pred_gap == pred))
    fail = pred != obs[:, 0]
    gap = np.abs(w0 - w1)
    print(f"row {row}: single-detector observable edges: "
          f"{observable_edges_single(circ)}; min(w0,w1)==w: {ok_w}; "
          f"gap-class == prediction: {ok_p:.4f}; logical failure rate "
          f"{fail.mean():.4f}; median gap failed {np.median(gap[fail]) if fail.any() else float('nan'):.2f} "
          f"vs ok {np.median(gap[~fail]):.2f}")
    return ok_w and ok_p > 0.999


if __name__ == "__main__":
    print("validation", "PASS" if validate() else "FAIL")
