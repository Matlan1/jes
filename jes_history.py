"""Disk-backed history tiers for very long evolution runs.

Keeps RAM flat by moving everything outside a sliding hot window out of memory:
- checkpoint generations (every N gens): dna + calmState blob + rankings row
- older generations: species population ranges only (feeds the SAC chart and
  species labels at full resolution)

Layout inside the archive directory:
  snapshot.bin   fixed stride per checkpoint slot:
                   dna  [c_count * trait_count] float32
                   calm [c_count * calm_len]    float64 (stored verbatim)
  rankings.bin   fixed stride: [c_count] int32 per checkpoint slot
  pops.bin       variable records: [gen:int32][K:int32][K*4 int32 rows]
                 rows kept in the dict's original insertion order

Fixed-stride offsets are pure arithmetic, so a partially written file can be
rescanned/truncated on open without any sidecar index.
"""
import os
import shutil
import uuid
from collections import OrderedDict

import numpy as np


class HistoryStore:
    def __init__(self, archive_dir, c_count, trait_count, calm_len, snapshot_every, _run_id=None):
        self.dir = archive_dir
        os.makedirs(archive_dir, exist_ok=True)
        self.c_count = c_count
        self.trait_count = trait_count
        self.calm_len = calm_len
        self.snapshot_every = snapshot_every

        self.snap_path = os.path.join(archive_dir, "snapshot.bin")
        self.rank_path = os.path.join(archive_dir, "rankings.bin")
        self.pops_path = os.path.join(archive_dir, "pops.bin")

        # Archive identity: the binaries belong to exactly one simulation run.
        # A different run finding them here starts fresh - stale bodies under new
        # statistics would be silent corruption. (Tests pass a fixed _run_id to share.)
        self.run_id = _run_id if _run_id is not None else uuid.uuid4().hex
        id_path = os.path.join(archive_dir, "run_id.txt")
        try:
            existing_id = open(id_path, "r").read().strip() if os.path.exists(id_path) else None
        except OSError:
            existing_id = None
        if existing_id is not None and existing_id != self.run_id:
            for p in (self.snap_path, self.rank_path, self.pops_path):
                if os.path.exists(p):
                    os.remove(p)
            print("[jes] Starting a fresh history archive (the existing one belonged to an older run).")
        if existing_id != self.run_id:
            with open(id_path, "w") as f:
                f.write(self.run_id)

        self.dna_stride = c_count * trait_count * 4   # bytes (float32)
        self.calm_stride = c_count * calm_len * 8     # bytes (float64)
        self.snap_stride = self.dna_stride + self.calm_stride
        self.rank_stride = c_count * 4                # bytes (int32)

        # Checkpoints must be written sequentially (slot 0, 1, 2, ...).
        self.next_slot = os.path.getsize(self.snap_path) // self.snap_stride if os.path.exists(self.snap_path) else 0
        self.pops_index = {}  # gen -> byte offset of record start
        self._recover_pops()

    # --- snapshots ---------------------------------------------------------

    def slot_of(self, gen):
        if gen % self.snapshot_every != 0:
            raise ValueError(f"gen {gen} is not a checkpoint (every {self.snapshot_every})")
        return gen // self.snapshot_every

    def has_snapshot(self, gen):
        return gen % self.snapshot_every == 0 and self.slot_of(gen) < self.next_slot

    def write_snapshot(self, gen, dna_block, calm_block):
        """dna_block: [c_count, trait_count] float32; calm_block: [c_count, calm_len] float64."""
        slot = self.slot_of(gen)
        if slot != self.next_slot:
            raise RuntimeError(f"snapshot slots must be written in order (expected {self.next_slot}, got {slot})")
        with open(self.snap_path, "ab") as f:
            f.write(np.ascontiguousarray(dna_block, dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(calm_block, dtype=np.float64).tobytes())
        self.next_slot += 1

    def read_snapshot(self, gen):
        if not self.has_snapshot(gen):
            raise KeyError(f"no archived snapshot for gen {gen}")
        slot = self.slot_of(gen)
        with open(self.snap_path, "rb") as f:
            f.seek(slot * self.snap_stride)
            dna = np.frombuffer(f.read(self.dna_stride), dtype=np.float32).reshape(self.c_count, self.trait_count)
            calm = np.frombuffer(f.read(self.calm_stride), dtype=np.float64).reshape(self.c_count, self.calm_len)
        return dna, calm

    # --- rankings ----------------------------------------------------------

    def has_rankings(self, gen):
        if gen % self.snapshot_every != 0 or not os.path.exists(self.rank_path):
            return False
        return self.slot_of(gen) < os.path.getsize(self.rank_path) // self.rank_stride

    def write_rankings(self, gen, rankings_row):
        slot = self.slot_of(gen)
        expected = os.path.getsize(self.rank_path) // self.rank_stride if os.path.exists(self.rank_path) else 0
        if slot != expected:
            raise RuntimeError(f"ranking slots must be written in order (expected {expected}, got {slot})")
        with open(self.rank_path, "ab") as f:
            f.write(np.ascontiguousarray(rankings_row, dtype=np.int32).tobytes())

    def read_rankings(self, gen):
        if not self.has_rankings(gen):
            raise KeyError(f"no archived rankings for gen {gen}")
        with open(self.rank_path, "rb") as f:
            f.seek(self.slot_of(gen) * self.rank_stride)
            return np.frombuffer(f.read(self.rank_stride), dtype=np.int32)

    # --- species pops ------------------------------------------------------

    def _recover_pops(self):
        """Rebuild {gen: offset} by scanning record headers; truncate torn tails."""
        self.pops_index.clear()
        if not os.path.exists(self.pops_path):
            return
        good_end = 0
        head_dtype = np.dtype([("gen", np.int32), ("K", np.int32)])
        with open(self.pops_path, "rb") as f:
            while True:
                head = f.read(head_dtype.itemsize)
                if len(head) < head_dtype.itemsize:
                    break
                gen, K = np.frombuffer(head, dtype=head_dtype)[0]
                payload = int(K) * 16  # K rows of 4 int32
                data = f.read(payload)
                if len(data) < payload:
                    break
                self.pops_index[int(gen)] = good_end
                good_end += head_dtype.itemsize + payload
        size = os.path.getsize(self.pops_path)
        if size != good_end:  # power-loss torn tail: drop the partial record
            with open(self.pops_path, "r+b") as f:
                f.truncate(good_end)

    def has_pops(self, gen):
        return gen in self.pops_index

    def append_pops(self, gen, pops_dict):
        """pops_dict: {species_id: [pop, start, end]} - insertion order is preserved."""
        rows = np.empty((len(pops_dict), 4), dtype=np.int32)
        for i, (sp, v) in enumerate(pops_dict.items()):
            rows[i, 0] = sp
            rows[i, 1] = v[0]
            rows[i, 2] = v[1]
            rows[i, 3] = v[2]
        with open(self.pops_path, "ab") as f:
            f.write(np.array([gen, len(pops_dict)], dtype=np.int32).tobytes())
            if len(rows):
                f.write(rows.tobytes())
        self.pops_index[gen] = os.path.getsize(self.pops_path) - 8 - len(rows) * 16

    def read_pops(self, gen):
        """Rebuilds {species_id: [pop, start, end]} preserving stored insertion order."""
        if gen not in self.pops_index:
            raise KeyError(f"no archived pops for gen {gen}")
        offset = self.pops_index[gen]
        with open(self.pops_path, "rb") as f:
            f.seek(offset)
            gen_, K = np.frombuffer(f.read(8), dtype=np.int32)
            rows = np.frombuffer(f.read(int(K) * 16), dtype=np.int32).reshape(int(K), 4)
        out = {}
        for i in range(rows.shape[0]):
            out[int(rows[i, 0])] = [int(rows[i, 1]), int(rows[i, 2]), int(rows[i, 3])]
        return out

    # --- misc --------------------------------------------------------------

    def disk_free_bytes(self):
        return shutil.disk_usage(self.dir).free


class PopArchiveList:
    """List-like stand-in for sim.species_pops: a resident recent tail, older
    generations materialized on demand from HistoryStore with a small FIFO cache."""

    def __init__(self, store, tail_size, cache_size=256):
        self.store = store
        self.tail_size = max(1, int(tail_size))
        self.cache_size = max(1, int(cache_size))
        self._tail = []          # dicts for gens base_gen .. len-1
        self.base_gen = 0        # gen index of _tail[0]
        self._cache = OrderedDict()

    def append(self, item):
        self._tail.append(item)
        while len(self._tail) > self.tail_size:
            self.store.append_pops(self.base_gen, self._tail[0])
            self._tail.pop(0)
            self.base_gen += 1

    def __len__(self):
        return self.base_gen + len(self._tail)

    def __getitem__(self, i):
        n = len(self)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(f"gen index {i} out of range ({n} generations)")
        if i >= self.base_gen:
            return self._tail[i - self.base_gen]
        hit = self._cache.get(i)
        if hit is not None:
            self._cache.move_to_end(i)
            return hit
        d = self.store.read_pops(i)
        self._cache[i] = d
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return d

    def __iter__(self):
        for g in range(len(self)):
            yield self[g]


class TieredRankings:
    """List-like stand-in for sim.rankings: the hot window stays resident,
    checkpoint rows spill to HistoryStore, other old rows are simply gone
    (the UI only ever reads hot or checkpoint generations)."""

    def __init__(self, store, window_gens):
        self.store = store
        self.window = max(1, int(window_gens))
        self._recent = []        # rows for gens total-len-window .. total-len-1
        self.total = 0

    def append(self, row):
        self._recent.append(np.asarray(row, dtype=np.int32))
        self.total += 1
        while len(self._recent) > self.window:
            g = self.total - len(self._recent)
            if g % self.store.snapshot_every == 0 and not self.store.has_rankings(g):
                self.store.write_rankings(g, self._recent[0])
            self._recent.pop(0)

    def __len__(self):
        return self.total

    def __getitem__(self, i):
        n = self.total
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(f"gen index {i} out of range ({n} generations)")
        age = (n - 1) - i
        if age < len(self._recent):
            return self._recent[len(self._recent) - 1 - age]
        return self.store.read_rankings(i)


class GenBodyCache:
    """LRU of loaded checkpoint-generation bodies (dna + calm blocks).

    Loading a generation wires its body blocks back onto the generation's
    Creature shells; on eviction the owner (via on_evict) strips them again.
    Loads are always whole-generation - partial loads would break code that
    inspects shells[0].calmState to pick a starting pose."""

    def __init__(self, store, capacity, on_evict=None, is_protected=None):
        self.store = store
        self.capacity = max(1, int(capacity))
        self.on_evict = on_evict  # callable(gen)
        # Protection only exempts creatures from STRIPPING - reloading always
        # wires every shell, because archived values always equal resident ones
        # (retired generations are never evaluated or mutated again).
        self.is_protected = is_protected or (lambda c: getattr(c, "pinned", False))
        self._entries = OrderedDict()  # gen -> (dna_block, calm_block)

    def peek(self, gen):
        return self._entries.get(gen)

    def is_loaded(self, gen):
        return gen in self._entries

    def wire_shells(self, shells, dna_block, calm_block):
        ncc = getattr(shells[0].sim, "node_coor_count", 4) if shells else 4
        for j, c in enumerate(shells):
            if c is not None:
                c.dna = dna_block[j]
                c.calmState = calm_block[j].reshape(c.sim.CH + 1, c.sim.CW + 1, ncc)

    def strip_shells(self, shells):
        for c in shells:
            if c is not None and not self.is_protected(c):
                c.dna = None
                c.calmState = None

    def load(self, gen, shells):
        """shells: this generation's Creature objects (protected ones untouched)."""
        hit = self._entries.get(gen)
        if hit is not None:
            self._entries.move_to_end(gen)
            self.wire_shells(shells, hit[0], hit[1])
            return hit
        dna_block, calm_block = self.store.read_snapshot(gen)
        self._entries[gen] = (dna_block, calm_block)
        while len(self._entries) > self.capacity:
            old_gen, _ = self._entries.popitem(last=False)
            if self.on_evict is not None:
                self.on_evict(old_gen)
        self.wire_shells(shells, dna_block, calm_block)
        return (dna_block, calm_block)
