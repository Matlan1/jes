"""Roundtrip tests for jes_history archive primitives. Run: python tests/test_history_primitives.py"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jes_history import GenBodyCache, HistoryStore, PopArchiveList, TieredRankings

CC, TRAIT, EVERY = 10, 20, 5
CALM = (2 + 1) * (1 + 1) * 4  # matches FakeSim CH=2, CW=1


class FakeSim:
    CH, CW = 2, 1


class FakeCreature:
    def __init__(self, i):
        self.IDNumber = i
        self.pinned = False
        self.dna = None
        self.calmState = None
        self.sim = FakeSim()


def test_snapshot_roundtrip():
    tmp = tempfile.mkdtemp(prefix="jes_hist_")
    try:
        st = HistoryStore(tmp, CC, TRAIT, CALM, EVERY)
        dna0 = np.random.rand(CC, TRAIT).astype(np.float32)
        calm0 = np.random.rand(CC, CALM).astype(np.float64) * 3.0
        st.write_snapshot(0, dna0, calm0)
        st.write_snapshot(5, np.random.rand(CC, TRAIT).astype(np.float32), np.random.rand(CC, CALM))
        assert st.has_snapshot(0) and st.has_snapshot(5)
        assert not st.has_snapshot(3) and not st.has_snapshot(10)
        d, c = st.read_snapshot(0)
        assert np.array_equal(d, dna0) and np.array_equal(c, calm0)
        assert d.dtype == np.float32 and c.dtype == np.float64
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rankings_roundtrip():
    tmp = tempfile.mkdtemp(prefix="jes_hist_")
    try:
        st = HistoryStore(tmp, CC, TRAIT, CALM, EVERY)
        r0 = np.arange(CC, dtype=np.int32)[::-1].copy()
        st.write_rankings(0, r0)
        st.write_rankings(5, np.arange(CC, dtype=np.int32))
        assert np.array_equal(st.read_rankings(0), r0)
        assert np.array_equal(st.read_rankings(5), np.arange(CC, dtype=np.int32))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pops_insertion_order_and_types():
    tmp = tempfile.mkdtemp(prefix="jes_hist_")
    try:
        st = HistoryStore(tmp, CC, TRAIT, CALM, EVERY)
        pd = {7: [3, 0, 3], 2: [5, 3, 8], 11: [2, 8, 10]}
        st.append_pops(0, pd)
        back = st.read_pops(0)
        assert list(back.keys()) == [7, 2, 11], "insertion order must survive"
        assert all(type(v) is list for v in back.values()), "values must be lists (max() lexicographic compare)"
        assert back[2] == [5, 3, 8]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_torn_tail_recovery_and_reopen():
    tmp = tempfile.mkdtemp(prefix="jes_hist_")
    try:
        st = HistoryStore(tmp, CC, TRAIT, CALM, EVERY, _run_id="torn-tail-run")
        dna0 = np.random.rand(CC, TRAIT).astype(np.float32)
        calm0 = np.random.rand(CC, CALM).astype(np.float64)
        st.write_snapshot(0, dna0, calm0)
        r0 = np.arange(CC, dtype=np.int32)[::-1].copy()
        st.write_rankings(0, r0)
        st.append_pops(0, {7: [3, 0, 3]})
        with open(st.pops_path, "ab") as f:  # simulate power loss mid-record
            f.write(np.array([5, 4], dtype=np.int32).tobytes() + b"\x00" * 30)
        good_size = os.path.getsize(st.pops_path) - 38
        st2 = HistoryStore(tmp, CC, TRAIT, CALM, EVERY, _run_id="torn-tail-run")
        assert st2.has_pops(0) and not st2.has_pops(5)
        assert os.path.getsize(st2.pops_path) == good_size, "torn tail truncated"
        assert np.array_equal(st2.read_snapshot(0)[0], dna0)
        assert np.array_equal(st2.read_rankings(0), r0)
        assert st2.read_pops(0)[7] == [3, 0, 3]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pop_archive_list_semantics():
    tmp = tempfile.mkdtemp(prefix="jes_hist_")
    try:
        st = HistoryStore(tmp, CC, TRAIT, CALM, EVERY)
        pl = PopArchiveList(st, tail_size=2, cache_size=4)
        dicts = []
        for g in range(6):
            dd = {sp: [sp + g, g, g + sp + 1] for sp in range(g + 1)}
            dicts.append(dd)
            pl.append(dd)
        assert len(pl) == 6
        assert pl.base_gen == 4, "oldest two spilled to disk"
        assert pl[0] == dicts[0], "materialized dict equals original"
        assert pl[-1] is dicts[5], "tail served by identity"
        m = pl[0]
        assert list(m.keys()) == list(dicts[0].keys())
        assert m[0] == dicts[0][0] and type(m[0]) is list
        try:
            pl[99]
            raise AssertionError("expected IndexError")
        except IndexError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tiered_rankings():
    tmp = tempfile.mkdtemp(prefix="jes_hist_")
    try:
        st = HistoryStore(tmp, CC, TRAIT, CALM, EVERY)
        for g in (0, 5):
            st.write_snapshot(g, np.random.rand(CC, TRAIT).astype(np.float32),
                              np.random.rand(CC, CALM).astype(np.float64))
        tr = TieredRankings(st, window_gens=2)
        rows = [np.arange(CC, dtype=np.int32) * (g + 1) for g in range(12)]
        for row in rows:
            tr.append(row)
        assert len(tr) == 12
        assert np.array_equal(tr[-1], rows[11]), "newest served from ring"
        assert np.array_equal(tr[10], rows[10]), "second-newest served from ring"
        assert np.array_equal(tr[5], rows[5]), "checkpoint row served from disk"
        assert np.array_equal(tr[0], rows[0]), "checkpoint row served from disk"
        try:
            tr[6]  # cold non-checkpoint: must never be requested (UI snaps first)
            raise AssertionError("expected KeyError for cold non-checkpoint gen")
        except KeyError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gen_body_cache_pins_and_eviction():
    tmp = tempfile.mkdtemp(prefix="jes_hist_")
    try:
        st = HistoryStore(tmp, CC, TRAIT, CALM, EVERY)
        snap_dna = {g: np.random.rand(CC, TRAIT).astype(np.float32) for g in (0, 5)}
        snap_calm = {g: np.random.rand(CC, CALM).astype(np.float64) for g in (0, 5)}
        for g in (0, 5):
            st.write_snapshot(g, snap_dna[g], snap_calm[g])
        shells = [FakeCreature(i) for i in range(CC)]
        shells[3].pinned = True
        evicted = []
        protected = lambda c: c.pinned or getattr(c, "temp_pin", False)
        cache = GenBodyCache(st, capacity=1, on_evict=lambda g: evicted.append((g, shells)),
                             is_protected=protected)
        cache.load(0, shells)
        # wiring covers EVERY shell - protection is about stripping, not loading
        assert shells[0].dna is not None
        assert np.array_equal(shells[0].dna, snap_dna[0][0])
        assert shells[0].calmState.shape == (FakeSim.CH + 1, FakeSim.CW + 1, 4)
        assert np.array_equal(shells[0].calmState.reshape(-1), snap_calm[0][0])
        assert shells[3].dna is not None and np.array_equal(shells[3].dna, snap_dna[0][3]), \
            "pinned creature must be reloaded too (it may have been pinned post-retirement)"
        shells[4].temp_pin = True  # simulates movie-playback protection
        cache.load(5, shells)
        assert evicted and evicted[0][0] == 0, "LRU evicted gen 0"
        cache.strip_shells(evicted[0][1])
        assert shells[0].dna is None and shells[0].calmState is None, "eviction strips unpinned"
        assert shells[3].dna is not None, "permanently pinned survives stripping"
        assert shells[4].dna is not None and shells[4].calmState is not None, "temp-pinned survives stripping"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_stale_archive_from_other_run_is_wiped():
    tmp = tempfile.mkdtemp(prefix="jes_hist_")
    try:
        st = HistoryStore(tmp, CC, TRAIT, CALM, EVERY, _run_id="run-one")
        st.write_snapshot(0, np.random.rand(CC, TRAIT).astype(np.float32),
                          np.random.rand(CC, CALM).astype(np.float64))
        st.write_rankings(0, np.arange(CC, dtype=np.int32))
        st.append_pops(0, {7: [3, 0, 3]})
        assert st.has_snapshot(0)

        # same run id (crash-recovery style reopen) keeps the data
        st_again = HistoryStore(tmp, CC, TRAIT, CALM, EVERY, _run_id="run-one")
        assert st_again.has_snapshot(0), "same-run reopen must keep data"

        # a NEW run finding this archive must not inherit its bytes
        st_fresh = HistoryStore(tmp, CC, TRAIT, CALM, EVERY)  # random run id
        assert not st_fresh.has_snapshot(0), "stale snapshot leaked into new run"
        assert not st_fresh.has_rankings(0) and not st_fresh.has_pops(0)
        assert st_fresh.next_slot == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"ALL {len(tests)} PRIMITIVE TESTS PASSED")
