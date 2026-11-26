import math
import numpy as np
from hash_utils import mix_hash_vec

class CountMinSketch:

    def __init__(self, memory_bytes: int, d: int = 5, seed: int = 0):
        self.d = d
        self.w = max(1, memory_bytes // (4 * d))
        self.table = np.zeros((self.d, self.w), dtype=np.uint32)
        rng = np.random.default_rng(seed)
        self.seeds = rng.integers(1, 2**62, size=self.d, dtype=np.uint64)

    @classmethod
    def from_epsilon_delta(cls, eps: float, delta: float, seed: int = 0):
        w = int(math.ceil(math.e / eps))
        d = int(math.ceil(math.log(1.0 / delta)))
        return cls(memory_bytes=w * d * 4, d=d, seed=seed)

    def update_batch(self, stream: np.ndarray):
        for r in range(self.d):
            cols = (mix_hash_vec(stream, self.seeds[r]) % np.uint64(self.w)).astype(np.int64)
            self.table[r] += np.bincount(cols, minlength=self.w).astype(np.uint32)

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        est = np.full(len(keys), np.iinfo(np.uint32).max, dtype=np.uint32)
        for r in range(self.d):
            cols = (mix_hash_vec(keys, self.seeds[r]) % np.uint64(self.w)).astype(np.int64)
            est = np.minimum(est, self.table[r, cols])
        return est.astype(np.float64)

    def memory_bytes(self) -> int:
        return int(self.table.nbytes)

class CountSketch:

    def __init__(self, memory_bytes: int, d: int = 5, seed: int = 0):
        self.d = d
        self.w = max(1, memory_bytes // (4 * d))
        self.table = np.zeros((self.d, self.w), dtype=np.int32)
        rng = np.random.default_rng(seed)
        self.col_seeds = rng.integers(1, 2**62, size=self.d, dtype=np.uint64)
        self.sign_seeds = rng.integers(1, 2**62, size=self.d, dtype=np.uint64)

    def update_batch(self, stream: np.ndarray):
        for r in range(self.d):
            cols = (mix_hash_vec(stream, self.col_seeds[r]) % np.uint64(self.w)).astype(np.int64)
            signs = np.where((mix_hash_vec(stream, self.sign_seeds[r]) & np.uint64(1)) == 0, 1.0, -1.0)
            self.table[r] += np.bincount(cols, weights=signs, minlength=self.w).astype(np.int32)

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        ests = np.zeros((self.d, len(keys)), dtype=np.int32)
        for r in range(self.d):
            cols = (mix_hash_vec(keys, self.col_seeds[r]) % np.uint64(self.w)).astype(np.int64)
            signs = np.where((mix_hash_vec(keys, self.sign_seeds[r]) & np.uint64(1)) == 0, 1, -1)
            ests[r] = self.table[r, cols] * signs
        return np.median(ests, axis=0)

    def memory_bytes(self) -> int:
        return int(self.table.nbytes)

class CountingBloomEstimator:

    def __init__(self, memory_bytes: int, k: int = 5, seed: int = 0):
        self.k = k
        self.m = max(1, memory_bytes // 4)
        self.counters = np.zeros(self.m, dtype=np.uint32)
        rng = np.random.default_rng(seed)
        self.seeds = rng.integers(1, 2**62, size=self.k, dtype=np.uint64)

    def update_batch(self, stream: np.ndarray):
        for s in self.seeds:
            pos = (mix_hash_vec(stream, s) % np.uint64(self.m)).astype(np.int64)
            self.counters += np.bincount(pos, minlength=self.m).astype(np.uint32)

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        est = np.full(len(keys), np.iinfo(np.uint32).max, dtype=np.uint32)
        for s in self.seeds:
            pos = (mix_hash_vec(keys, s) % np.uint64(self.m)).astype(np.int64)
            est = np.minimum(est, self.counters[pos])
        return est.astype(np.float64)

    def memory_bytes(self) -> int:
        return int(self.counters.nbytes)

class LossyCounting:

    def __init__(self, eps: float):
        self.eps = eps
        self.w = int(math.ceil(1.0 / eps))
        self.table = {}
        self.max_entries = 0

    def update_stream(self, stream: np.ndarray):
        w = self.w
        table = self.table
        b_current = 1
        for i, key in enumerate(stream, 1):
            k = int(key)
            if k in table:
                table[k][0] += 1
            else:
                table[k] = [1, b_current - 1]
            if len(table) > self.max_entries:
                self.max_entries = len(table)
            if i % w == 0:
                to_del = [item for item, val in table.items() if val[0] + val[1] <= b_current]
                for item in to_del:
                    del table[item]
                b_current += 1

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        return np.array(
            [self.table[int(k)][0] if int(k) in self.table else 0.0 for k in keys],
            dtype=np.float64,
        )

    def peak_memory_bytes(self) -> int:
        return self.max_entries * 16
