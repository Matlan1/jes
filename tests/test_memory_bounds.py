"""End-to-end memory-bounds oracle for the tiered history system.

Runs a real (small) simulation headless and proves:
  1. creature bodies stay bounded to the hot window + loaded checkpoints + pins
  2. disk archive grows at checkpoint cadence, not per-generation
  3. archived dna/calmState/rankings/species_pops reload bit-exactly
  4. mosaic pixels for an archived checkpoint are identical after a full reload
  5. movie replays of an archived generation match the pre-archive replay
  6. snap logic resolves display generations correctly
  7. dropped non-checkpoint generations degrade honestly (no crash)
  8. the disk-space guard actually fires

Run: python tests/test_memory_bounds.py
"""
import gc
import hashlib
import os
import shutil
import sys
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

import jes_sim
jes_sim.HAS_NUMBA = False          # force the deterministic NumPy physics path
from jes_sim import Sim            # noqa: E402
from jes_ui import UI              # noqa: E402
from utils import get_mosaic_dim   # noqa: E402

POP = 60
WINDOW = 3
SNAPSHOT_EVERY = 5
LRU = 2
POPS_TAIL = 4
GENS_TO_RUN = 23                   # retires checkpoints 0, 5, 10, 15, 20


def build_sim_ui(archive_dir):
    sim = Sim(_c_count=POP, _stabilization_time=30, _trial_time=40,
              _beat_time=20, _beat_fade_time=5, _c_dim=[4, 4],
              _beats_per_cycle=3, _node_coor_count=4,
              _y_clips=[-10000000, 0], _ground_friction_coef=25,
              _gravity_acceleration_coef=0.002, _calming_friction_coef=0.7,
              _typical_friction_coef=0.8, _muscle_coef=0.08,
              _traits_per_box=3, _traits_extra=1,
              _mutation_rate=0.07, _big_mutation_rate=0.025,
              _UNITS_PER_METER=0.05,
              _ram_window_gens=WINDOW, _snapshot_every=SNAPSHOT_EVERY,
              _lru_capacity=LRU, _archive_dir=archive_dir, _pops_tail_gens=POPS_TAIL)
    ui = UI(_W_W=1600, _W_H=900, _MOVIE_SINGLE_DIM=(200, 200),
            _GRAPH_COOR=(900, 40, 400, 250), _SAC_COOR=(900, 300, 400, 200),
            _GENEALOGY_COOR=(20, 80, 500, 500, 42),
            _COLUMN_MARGIN=330, _MOSAIC_DIM=get_mosaic_dim(POP),
            _MENU_TEXT_UP=180, _CM_MARGIN1=20, _CM_MARGIN2=1)
    sim.ui = ui
    ui.sim = sim
    ui.addButtonsAndSliders()
    return sim, ui


def mosaic_hash(ui):
    arr = pygame.surfarray.array3d(ui.mosaicScreen)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def replay_checksum(sim, gen, c_idx):
    mem = sim.simulateImport(gen, c_idx, c_idx + 1, True)
    result = sim.simulateRun(mem, sim.trial_time, False)
    node = result[0]
    return float(np.sum(node[:, :, :, 0])) + float(np.sum(node[:, :, :, 1]))


def replay_checksum_obj(sim, creature):
    """Replay via OBJECT import - the path pinned reps use when their
    generation row has been retired."""
    mem = sim.simulateImportCreatures([creature])
    node = sim.simulateRun(mem, sim.trial_time, False)[0]
    return float(np.sum(node))


def test_end_to_end():
    tmp = tempfile.mkdtemp(prefix="jes_memtest_")
    pygame.init()
    try:
        sim, ui = build_sim_ui(tmp)
        sim.initializeUniverse()

        # ---- run one generation so gen 0 is fully evaluated (flipDNA etc.) ----
        sim.doGeneration(ui.doGenButton)
        assert len(sim.percentiles) == 1

        # capture pre-archive truth of generation 0
        pre_dna = [c.dna.copy() for c in sim.creatures[0]]
        pre_calm = [None if c.calmState is None else c.calmState.copy() for c in sim.creatures[0]]
        pre_rankings = np.array(sim.rankings[0])
        pops0 = sim.species_pops[0]
        pre_pops_keys = list(pops0.keys())
        pre_pops_vals = {k: list(v) for k, v in pops0.items()}
        ui.refreshMosaic()
        # force every icon to re-render from the EVALUATED dna (init drew them
        # pre-evaluation; flipDNA may have replaced dna since)
        for c in sim.creatures[0]:
            c.icons = [None, None]
        ui.refreshMosaic()
        pre_mosaic = mosaic_hash(ui)
        pre_replay = replay_checksum(sim, 0, POP // 3)

        # pin one creature of gen 0 (as prominence would) before it retires
        pinned_creature = sim.creatures[0][7]
        pin_dna = pinned_creature.dna.copy()
        sim.pinCreature(pinned_creature)

        # ---- run the rest; pin a non-checkpoint creature mid-run (late-prominence case) ----
        commit_samples = []
        late_creature = None
        late_checksum_pre = None
        for i in range(GENS_TO_RUN - 1):
            sim.doGeneration(ui.doGenButton)
            if late_creature is None and len(sim.creatures) > SNAPSHOT_EVERY + 1:
                g = SNAPSHOT_EVERY + 1  # non-checkpoint generation: its row will be dropped
                late_creature = sim.creatures[g][5]
                sim.pinCreature(late_creature)
                late_checksum_pre = replay_checksum_obj(sim, late_creature)
            gc.collect()
            from jes_crash_logger import get_system_memory_info
            commit_samples.append(get_system_memory_info()[5])

        newest = len(sim.percentiles) - 1
        assert newest == GENS_TO_RUN - 1
        expected_watermark = len(sim.creatures) - WINDOW
        assert sim.archived_upto == expected_watermark, \
            f"watermark at {sim.archived_upto}, expected {expected_watermark}"

        # ---- 1. structural body bound ----
        live_bodies = sum(1 for row in sim.creatures if row is not None
                          for c in row if c.dna is not None)
        bound = (WINDOW + LRU + 8) * POP  # window + cache + pins + slack
        assert live_bodies <= bound, f"{live_bodies} live bodies exceed bound {bound}"

        # ---- 2. archive cadence ----
        total_bytes = sum(os.path.getsize(os.path.join(tmp, f))
                          for f in os.listdir(tmp))
        expected_checkpoints = len([g for g in range(0, GENS_TO_RUN - WINDOW + 1)
                                    if g % SNAPSHOT_EVERY == 0])
        per_snapshot = POP * (sim.trait_count * 4 + (sim.CH + 1) * (sim.CW + 1) * 4 * 8)
        cap = int(expected_checkpoints * per_snapshot * 1.5 + GENS_TO_RUN * 64 * 1024)
        assert total_bytes < cap, f"archive grew to {total_bytes} bytes (cap {cap})"

        # ---- 3. bit-exact reloads through the proxies and the store ----
        sim.ensureGenBodies(0)
        assert sim.body_cache.is_loaded(0), "checkpoint 0 should be loaded"
        for idx in range(POP):
            if sim.creatures[0][idx] is pinned_creature:
                continue
            got = sim.creatures[0][idx].dna
            want = pre_dna[idx]
            if got is None:
                continue  # placeholder-covered creatures are checked via store below
            assert np.array_equal(got, want), f"dna mismatch on reload, creature {idx}"
        dna_block, calm_block = sim.store.read_snapshot(0)
        for idx in range(POP):
            assert np.array_equal(dna_block[idx], pre_dna[idx]), f"store dna mismatch #{idx}"
            if pre_calm[idx] is not None:
                assert calm_block[idx].dtype == np.float64
                assert np.array_equal(calm_block[idx], pre_calm[idx].reshape(-1)), \
                    f"store calm mismatch #{idx}"
        assert np.array_equal(np.array(sim.rankings[0]), pre_rankings), "rankings roundtrip"
        reloaded_pops = sim.species_pops[0]
        assert list(reloaded_pops.keys()) == pre_pops_keys, "pops insertion order lost"
        for k in pre_pops_keys:
            assert list(reloaded_pops[k]) == pre_pops_vals[k], f"pops value mismatch sp={k}"

        # negative-test the comparators: corruption must be detectable
        tampered = dna_block.copy()
        tampered[0][0] = np.float32(tampered[0][0] + 1.0)
        assert not np.array_equal(tampered[0], pre_dna[0]), "comparator is insensitive!"

        # ---- 4. pinned creature survived retirement with its body intact ----
        assert pinned_creature.pinned is True
        assert pinned_creature.dna is not None and np.array_equal(pinned_creature.dna, pin_dna)
        assert sim.getCreatureWithID(pinned_creature.IDNumber) is pinned_creature

        # ---- 5. mosaic pixel-identity after forced icon regeneration ----
        for c in sim.creatures[0]:
            if c is not pinned_creature:
                c.icons = [None, None]   # force drawIcon -> ensure_dna -> cache reload
        ui.genSlider.manualUpdate(0)
        ui.refreshMosaic()
        post_mosaic = mosaic_hash(ui)
        assert post_mosaic == pre_mosaic, "mosaic pixels differ after archive roundtrip"

        # ---- 6. replay determinism across the archive boundary ----
        post_replay = replay_checksum(sim, 0, POP // 3)
        assert abs(post_replay - pre_replay) < 1e-6, \
            f"replay checksum changed across archive: {pre_replay} vs {post_replay}"

        # ---- 7. dropped non-checkpoint generations degrade honestly ----
        dropped_gen = next(g for g in range(newest - WINDOW)
                           if g % SNAPSHOT_EVERY != 0)
        assert sim.creatures[dropped_gen] is None, "non-checkpoint gen should be dropped"
        some_id = dropped_gen * POP + 3
        assert sim.getCreatureWithID(some_id) is None or \
               sim.getCreatureWithID(some_id).pinned
        assert sim.displayGenFor(dropped_gen) == (dropped_gen // SNAPSHOT_EVERY) * SNAPSHOT_EVERY
        assert sim.displayGenFor(newest) == newest, "hot gens must not snap"

        # ---- 8. crash-logger-style negative indexing works ----
        assert len(sim.species_pops[-1]) > 0
        assert len(sim.rankings[-1]) == POP

        # ---- 9. a creature pinned before its non-checkpoint gen was dropped still replays ----
        g_late = SNAPSHOT_EVERY + 1
        assert sim.creatures[g_late] is None, "expected the late creature's row to be dropped"
        assert late_creature.dna is not None and late_creature.calmState is not None
        assert sim.getCreatureWithID(late_creature.IDNumber) is late_creature
        post_checksum = replay_checksum_obj(sim, late_creature)
        assert abs(post_checksum - late_checksum_pre) < 1e-6, \
            "pinned rep in a dropped generation must replay identically"

        # ---- 10. lightboard/info bar stays safe with the slider parked at val_max ----
        ui.genSlider.manualUpdate(int(ui.genSlider.val_max))
        sp_any = next(iter(sim.species_pops[-1].keys()))
        ui.drawInfoBarSpecies(sp_any)  # clamp regression: must not raise IndexError

        # ---- commit-memory slope sanity (loose; structural bound above is exact) ----
        half = len(commit_samples) // 2
        if half >= 2:
            slope = (commit_samples[-1] - commit_samples[half]) / max(1, len(commit_samples) - 1 - half)
            assert slope < 2.0, f"commit growing at {slope:.2f} MB/gen"

        print(f"PASS end_to_end: {newest} gens, {live_bodies} live bodies "
              f"(bound {bound}), archive {total_bytes / 1024:.0f} KB, "
              f"commit slope {slope if half >= 2 else 0:.3f} MB/gen")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        pygame.quit()


def test_oracle_catches_the_old_bug():
    """Negative test: with retirement disabled (the old always-retain behavior),
    the structural body bound MUST fail - otherwise this oracle proves nothing."""
    tmp = tempfile.mkdtemp(prefix="jes_neg_")
    pygame.init()
    try:
        sim, ui = build_sim_ui(tmp)
        sim.initializeUniverse()
        sim._retireGeneration = lambda g: None  # simulate pre-fix retention
        for _ in range(GENS_TO_RUN):
            sim.doGeneration(ui.doGenButton)
        live_bodies = sum(1 for row in sim.creatures if row is not None
                          for c in row if c.dna is not None)
        bound = (WINDOW + LRU + 8) * POP
        assert live_bodies > bound, \
            f"oracle is insensitive: unbounded retention stayed under bound ({live_bodies} <= {bound})"
        print(f"PASS negative: disabled retirement keeps {live_bodies} bodies alive > bound {bound}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        pygame.quit()


def test_disk_guard_fires():
    tmp = tempfile.mkdtemp(prefix="jes_guard_")
    try:
        sim, ui = build_sim_ui(tmp)
        sim.store.disk_free_bytes = lambda: 1024 ** 3  # 1 GB free -> below minimum
        try:
            sim._checkArchiveDiskSpace()
            raise AssertionError("disk guard did not fire")
        except RuntimeError as e:
            assert "nearly full" in str(e)
        print("PASS disk_guard")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_disk_guard_fires()
    test_oracle_catches_the_old_bug()
    test_end_to_end()
    print("ALL MEMORY-BOUNDS TESTS PASSED")
