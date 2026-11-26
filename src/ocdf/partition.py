# ---- OCDF F2 duplicate grouping and partitioning helpers (Notebook 02) ----
import random, collections
import numpy as np

DEV_SOURCES = ('D1', 'D4', 'D6')
EXT_SOURCES = ('D3',)
PARTITIONS = [('train', 0.70), ('select', 0.10), ('calib_T', 0.05), ('thresh', 0.05), ('test', 0.10)]
SOURCE_PRIORITY = {'D1': 0, 'D4': 1, 'D6': 2, 'D3': 3}

class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[max(ra, rb)] = min(ra, rb)

def near_duplicate_pairs(hashes, max_dist=8, chunk=1500):
    """All index pairs (a < b) with Hamming distance <= max_dist between 64-bit hex hashes.

    Exhaustive comparison in chunks with NumPy's bitwise_count (NumPy 2.0 or later).
    """
    H = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    pairs = []
    for i0 in range(0, len(H), chunk):
        blk = H[i0:i0 + chunk]
        d = np.bitwise_count(blk[:, None] ^ H[None, :])
        ii, jj = np.nonzero(d <= max_dist)
        for a, b in zip(ii + i0, jj):
            if a < b: pairs.append((int(a), int(b), int(d[a - i0, b])))
    return pairs

def grouped_split(groups_by_stratum, seed=2026, partitions=PARTITIONS):
    """Assign whole groups to partitions, per stratum, approximating the target shares by image count.

    groups_by_stratum: {stratum: [(group_id, n_images), ...]}. Returns {group_id: partition}.
    Groups are shuffled with the seed, then placed largest first into the partition whose share is
    furthest below its target. Group integrity always wins over exact shares.
    """
    rng = random.Random(seed)
    out = {}
    names = [p for p, _ in partitions]; target = dict(partitions)
    for stratum in sorted(groups_by_stratum):
        gs = list(groups_by_stratum[stratum]); rng.shuffle(gs)
        gs.sort(key=lambda g: -g[1])                     # stable: equal sizes keep shuffled order
        total = sum(n for _, n in gs); filled = {p: 0 for p in names}
        for gid, n in gs:
            # deficit after placing: choose partition with the largest remaining shortfall
            best = max(names, key=lambda p: (target[p] * total - filled[p]) / max(target[p], 1e-9))
            out[gid] = best; filled[best] += n
    return out

def class_weights(train_counts, cap=5.0):
    """Training recipe: inverse-frequency weights normalised as n / (k * n_c) (so the image-weighted mean is 1),
    then capped at `cap`."""
    labels = sorted(train_counts); n = sum(train_counts.values()); k = len(labels)
    return {c: round(min(n / (k * train_counts[c]), cap), 4) for c in labels}
