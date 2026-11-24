"""ingest_kernel_runs.py — append G4CMP job JSONs into the consolidated record.

The L3 kernel sweeps used to land as one JSON per job in `bench/rad_l3_kernel/`
and were consolidated into a single `runs.jsonl` owned by `_data.RUNS`. The
readers (`kernel_runs.iter_runs` / `load_runs`) came with that consolidation;
a WRITER did not, so a new sweep had nowhere to land. This is the writer.

Jobs are produced in WSL by `l3_kernel_job.sh` into `~/rad/kernel/out/`, which
is outside the repository, so the source directory is given explicitly.

    python -m radchain.ingest_kernel_runs --src /mnt/.../out --dry-run
    python -m radchain.ingest_kernel_runs --src /mnt/.../out

REPLACE, DO NOT DUPLICATE. Tags are unique per job and a re-run of the same
tag with the same seed is the same measurement, so an existing tag is replaced
rather than appended. Appending would double-count it in every mean the fit
takes, silently and in the direction of tighter error bars.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

from radchain import _data

#: Fields every record must carry for the phonon backend and other readers to work.
REQUIRED = ("tag", "n_primaries", "pair_breaks", "counts", "bin_edges_mm",
            "wall_s")


def read_existing() -> dict:
    """tag -> record, in file order."""
    out = {}
    if os.path.exists(_data.RUNS):
        with open(_data.RUNS) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out[str(d.get("tag"))] = d
    return out


def _iter_incoming(src: str):
    """(name, record) pairs from a directory of JSONs, or from one JSONL.

    A JSONL source exists so that many small job files produced on another
    machine or filesystem can be moved as one file and ingested directly. The filename check
    below still applies, using each record's own `tag`.
    """
    if src.endswith(".jsonl"):
        with open(src) as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError as exc:
                    yield ("line {}".format(i), exc)
                    continue
                yield (str(d.get("tag")), d)
        return
    for f in sorted(glob.glob(os.path.join(src, "*.json"))):
        try:
            with open(f) as fh:
                yield (os.path.splitext(os.path.basename(f))[0], json.load(fh))
        except (OSError, ValueError) as exc:
            yield (os.path.basename(f), exc)


def read_incoming(src: str) -> tuple[dict, list]:
    """tag -> record from a directory of per-job JSONs or one JSONL, plus rejects."""
    out, bad = {}, []
    for name, d in _iter_incoming(src):
        if isinstance(d, Exception):
            bad.append((name, str(d)[:60]))
            continue
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            bad.append((name, "missing " + ",".join(missing)))
            continue
        # The consolidation checked tag == basename for all 716 records. Keep
        # that invariant: a record whose tag disagrees with its source name
        # would be parsed into the wrong sweep by the phonon backend's tag parser.
        if str(d["tag"]) != name:
            bad.append((name, "tag {!r} != source name".format(d["tag"])))
            continue
        out[str(d["tag"])] = d
    return out, bad


def ingest(src: str, dry_run: bool = False) -> dict:
    existing = read_existing()
    incoming, bad = read_incoming(src)

    added = [t for t in incoming if t not in existing]
    replaced = [t for t in incoming if t in existing]
    # A replacement that CHANGES the measurement is worth reporting: same tag
    # and same seed should reproduce, so a difference means something moved.
    changed = [t for t in replaced
               if (existing[t].get("pair_breaks"), existing[t].get("n_primaries"))
               != (incoming[t].get("pair_breaks"), incoming[t].get("n_primaries"))]

    report = {"src": src, "existing": len(existing), "incoming": len(incoming),
              "added": len(added), "replaced": len(replaced),
              "changed_on_replace": len(changed),
              "changed_tags": sorted(changed)[:12],
              "rejected": bad[:12], "n_rejected": len(bad),
              "total_after": len(existing) + len(added), "dry_run": dry_run}
    if dry_run:
        return report

    merged = dict(existing)
    merged.update(incoming)
    if os.path.exists(_data.RUNS):
        shutil.copy2(_data.RUNS, str(_data.RUNS) + ".bak")
        report["backup"] = str(_data.RUNS) + ".bak"
    with open(_data.RUNS, "w") as fh:
        for tag in sorted(merged):
            fh.write(json.dumps(merged[tag], separators=(",", ":")) + "\n")
    report["written"] = len(merged)
    return report


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="directory of per-job JSONs, or one .jsonl of them")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    r = ingest(a.src, dry_run=a.dry_run)
    print("ingest {}".format("(dry run)" if r["dry_run"] else "WROTE"))
    for k in ("existing", "incoming", "added", "replaced",
              "changed_on_replace", "n_rejected", "total_after", "written"):
        if k in r:
            print("  {:20s} {}".format(k, r[k]))
    if r["changed_tags"]:
        print("  tags whose measurement CHANGED on replace "
              "(same tag and seed should reproduce):")
        for t in r["changed_tags"]:
            print("    " + t)
    if r["rejected"]:
        print("  rejected:")
        for name, why in r["rejected"]:
            print("    {}: {}".format(name, why))


if __name__ == "__main__":
    _main()
