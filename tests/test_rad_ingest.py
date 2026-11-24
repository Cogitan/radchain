"""The writer for the consolidated kernel record.

The consolidation that collapsed 719 per-job JSONs into `runs.jsonl` shipped
readers and no writer, so a new sweep had nowhere to land. These
tests pin the two things a writer of an append-only measurement record has to
get right: it must not double-count, and it must not silently accept a record
that would be parsed into the wrong sweep.
"""
import json

import pytest

from radchain import ingest_kernel_runs as ing


def _rec(tag, primaries=5, breaks=700, wall=4):
    return {"tag": tag, "x_mm": 0.0, "y_mm": 0.0, "z_mm": 4.9,
            "energy_eV": 0.0369, "n_primaries": primaries, "seed": 1009,
            "wall_s": wall, "pair_breaks": breaks,
            "pair_breaks_per_primary": breaks / primaries,
            "outside_binning": 0, "bin_edges_mm": [0.03, 0.04, 0.05],
            "counts": [1, 2, 3]}


def _write_jsonl(tmp_path, recs):
    p = tmp_path / "incoming.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return str(p)


def test_a_repeated_tag_is_replaced_not_appended(tmp_path, monkeypatch):
    """A re-run of the same tag with the same seed is the SAME measurement.
    Appending it would double-count it in every mean a reader takes, silently,
    and in the direction of tighter error bars."""
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps(_rec("g_t0.2_p0.5_c0.10_a0.5_1")) + "\n")
    monkeypatch.setattr(ing._data, "RUNS", str(runs))

    src = _write_jsonl(tmp_path, [_rec("g_t0.2_p0.5_c0.10_a0.5_1"),
                                  _rec("g_t0.2_p0.5_c0.10_a0.5_2")])
    r = ing.ingest(src)
    assert r["added"] == 1 and r["replaced"] == 1
    assert r["written"] == 2
    lines = [json.loads(x) for x in runs.read_text().splitlines() if x.strip()]
    assert len({d["tag"] for d in lines}) == 2, "a tag was duplicated"


def test_a_replacement_that_changes_the_measurement_is_reported(tmp_path,
                                                                monkeypatch):
    """Same tag and same seed should reproduce. If it does not, something
    moved in the solver or the geometry and it must not pass silently."""
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps(_rec("g_t0.2_p0.5_c0.10_a0.5_1", breaks=700))
                    + "\n")
    monkeypatch.setattr(ing._data, "RUNS", str(runs))
    src = _write_jsonl(tmp_path,
                       [_rec("g_t0.2_p0.5_c0.10_a0.5_1", breaks=999)])
    r = ing.ingest(src, dry_run=True)
    assert r["changed_on_replace"] == 1
    assert "g_t0.2_p0.5_c0.10_a0.5_1" in r["changed_tags"]


def test_a_tag_that_disagrees_with_its_source_name_is_rejected(tmp_path,
                                                               monkeypatch):
    """Readers parse the sweep parameters out of the TAG. A record whose tag
    disagrees with the file it came from would be filed under the wrong
    configuration, which is worse than being dropped."""
    runs = tmp_path / "runs.jsonl"
    runs.write_text("")
    monkeypatch.setattr(ing._data, "RUNS", str(runs))
    bad = _rec("g_t0.2_p0.5_c0.10_a0.5_1")
    p = tmp_path / "incoming.jsonl"
    # a JSONL keys on the record's own tag, so force the mismatch via a dir
    d = tmp_path / "jobs"
    d.mkdir()
    (d / "g_t0.2_p9.9_c0.10_a0.5_1.json").write_text(json.dumps(bad))
    r = ing.ingest(str(d), dry_run=True)
    assert r["incoming"] == 0
    assert r["n_rejected"] == 1
    assert "!= source name" in r["rejected"][0][1]


def test_a_record_missing_a_required_field_is_rejected(tmp_path, monkeypatch):
    runs = tmp_path / "runs.jsonl"
    runs.write_text("")
    monkeypatch.setattr(ing._data, "RUNS", str(runs))
    short = _rec("g_t0.2_p0.5_c0.10_a0.5_1")
    del short["counts"]
    r = ing.ingest(_write_jsonl(tmp_path, [short]), dry_run=True)
    assert r["n_rejected"] == 1 and "missing counts" in r["rejected"][0][1]


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    runs = tmp_path / "runs.jsonl"
    runs.write_text("")
    monkeypatch.setattr(ing._data, "RUNS", str(runs))
    before = runs.read_text()
    ing.ingest(_write_jsonl(tmp_path, [_rec("g_t0.2_p0.5_c0.10_a0.5_9")]),
               dry_run=True)
    assert runs.read_text() == before


def test_a_real_write_leaves_a_backup(tmp_path, monkeypatch):
    """The record is a measurement archive. An ingest that goes wrong must be
    recoverable without re-running G4CMP."""
    runs = tmp_path / "runs.jsonl"
    runs.write_text(json.dumps(_rec("g_t0.2_p0.5_c0.10_a0.5_1")) + "\n")
    monkeypatch.setattr(ing._data, "RUNS", str(runs))
    r = ing.ingest(_write_jsonl(tmp_path, [_rec("g_t0.2_p0.5_c0.10_a0.5_2")]))
    assert r["backup"].endswith(".bak")
    import os
    assert os.path.exists(r["backup"])
