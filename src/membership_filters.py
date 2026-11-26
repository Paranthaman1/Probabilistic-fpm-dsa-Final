import math
import random
import numpy as np
from hash_utils import mix_hash_vec

class BloomFilter:

    def __init__(self, n: int, p: float, seed: int = 0):
        self.m = int(math.ceil(-(n * math.log(p)) / (math.log(2) ** 2)))
        self.k = max(1, int(round((self.m / n) * math.log(2))))
        self.bits = np.zeros((self.m + 7) // 8, dtype=np.uint8)
        rng = np.random.default_rng(seed)
        self.seeds = rng.integers(1, 2**62, size=self.k, dtype=np.uint64)

    def insert_batch(self, keys: np.ndarray):
        for s in self.seeds:
            pos = mix_hash_vec(keys, s) % np.uint64(self.m)
            byte_idx = pos >> np.uint64(3)
            bit_mask = (np.uint8(1) << (pos & np.uint64(7)).astype(np.uint8))
            np.bitwise_or.at(self.bits, byte_idx, bit_mask)

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        present = np.ones(len(keys), dtype=bool)
        for s in self.seeds:
            pos = mix_hash_vec(keys, s) % np.uint64(self.m)
            byte_idx = pos >> np.uint64(3)
            bit_mask = (np.uint8(1) << (pos & np.uint64(7)).astype(np.uint8))
            present &= (self.bits[byte_idx] & bit_mask) != 0
        return present

    def memory_bytes(self) -> int:
        return int(self.bits.nbytes)

class CountingBloomFilter:

    def __init__(self, n: int, p: float, counter_bits: int = 8, seed: int = 0):
        self.m = int(math.ceil(-(n * math.log(p)) / (math.log(2) ** 2)))
        self.k = max(1, int(round((self.m / n) * math.log(2))))
        self.dtype = {8: np.uint8, 16: np.uint16, 32: np.uint32}[counter_bits]
        self.max_val = int(np.iinfo(self.dtype).max)
        self.counters = np.zeros(self.m, dtype=self.dtype)
        self.overflows = 0
        rng = np.random.default_rng(seed)
        self.seeds = rng.integers(1, 2**62, size=self.k, dtype=np.uint64)

    def insert_batch(self, keys: np.ndarray):
        temp = self.counters.astype(np.uint64)
        for s in self.seeds:
            pos = mix_hash_vec(keys, s) % np.uint64(self.m)
            np.add.at(temp, pos, 1)
        self.overflows += int(np.sum(temp > self.max_val))
        self.counters = np.minimum(temp, self.max_val).astype(self.dtype)

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        present = np.ones(len(keys), dtype=bool)
        for s in self.seeds:
            pos = mix_hash_vec(keys, s) % np.uint64(self.m)
            present &= self.counters[pos] > 0
        return present

    def delete_batch(self, keys: np.ndarray):
        for s in self.seeds:
            pos = mix_hash_vec(keys, s) % np.uint64(self.m)
            vals = self.counters[pos]
            self.counters[pos] = np.where(vals > 0, vals - 1, 0)

    def memory_bytes(self) -> int:
        return int(self.counters.nbytes)

class CuckooFilter:

    def __init__(
        self,
        capacity: int,
        bucket_size: int = 4,
        fp_bits: int = 12,
        max_kicks: int = 150,
        seed: int = 0,
    ):
        raw_buckets = max(1, int(math.ceil(capacity / bucket_size)))
        self.num_buckets = 1 << (raw_buckets - 1).bit_length()
        self.bucket_size = bucket_size
        self.fp_bits = fp_bits
        self.max_kicks = max_kicks
        self.fp_mask = (1 << fp_bits) - 1
        self.dtype = np.uint8 if fp_bits <= 8 else np.uint16
        self.buckets = np.zeros((self.num_buckets, bucket_size), dtype=self.dtype)
        self.seed = np.uint64(seed + 1234567)
        self.mask = self.num_buckets - 1
        self.failures = 0

    def _hash_and_fp(self, keys: np.ndarray):
        h = mix_hash_vec(keys, self.seed)
        b1 = (h & np.uint64(self.mask)).astype(np.int64)
        fp = ((h >> np.uint64(32)) & np.uint64(self.fp_mask)).astype(np.int64)
        fp = np.where(fp == 0, 1, fp)
        fp_h = ((fp.astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)) >> np.uint64(16)) & np.uint64(self.mask)
        b2 = (b1 ^ fp_h.astype(np.int64)) & self.mask
        return b1, b2, fp

    def insert_batch(self, keys: np.ndarray):
        b1_arr, b2_arr, fp_arr = self._hash_and_fp(keys)
        buckets = self.buckets
        b_size = self.bucket_size
        mask = self.mask
        max_kicks = self.max_kicks
        rng_kick = random.Random(int(self.seed))

        for i in range(len(keys)):
            b1, b2, fp = int(b1_arr[i]), int(b2_arr[i]), int(fp_arr[i])
            placed = False
            for s in range(b_size):
                if buckets[b1, s] == 0:
                    buckets[b1, s] = fp
                    placed = True
                    break
            if placed:
                continue
            for s in range(b_size):
                if buckets[b2, s] == 0:
                    buckets[b2, s] = fp
                    placed = True
                    break
            if placed:
                continue

            curr_b = b1 if (i & 1) == 0 else b2
            curr_fp = fp
            for _ in range(max_kicks):
                slot = rng_kick.randrange(b_size)
                curr_fp, buckets[curr_b, slot] = int(buckets[curr_b, slot]), curr_fp
                fp_h = ((curr_fp * 0xBF58476D1CE4E5B9) >> 16) & mask
                curr_b = (curr_b ^ fp_h) & mask
                for s in range(b_size):
                    if buckets[curr_b, s] == 0:
                        buckets[curr_b, s] = curr_fp
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                self.failures += 1

    def query_batch(self, keys: np.ndarray) -> np.ndarray:
        b1, b2, fp = self._hash_and_fp(keys)
        fp_col = fp[:, None]
        return np.any(self.buckets[b1] == fp_col, axis=1) | np.any(self.buckets[b2] == fp_col, axis=1)

    def delete_batch(self, keys: np.ndarray) -> int:
        b1_arr, b2_arr, fp_arr = self._hash_and_fp(keys)
        buckets = self.buckets
        b_size = self.bucket_size
        deleted = 0
        for i in range(len(keys)):
            b1, b2, fp = int(b1_arr[i]), int(b2_arr[i]), int(fp_arr[i])
            done = False
            for s in range(b_size):
                if buckets[b1, s] == fp:
                    buckets[b1, s] = 0
                    deleted += 1
                    done = True
                    break
            if not done:
                for s in range(b_size):
                    if buckets[b2, s] == fp:
                        buckets[b2, s] = 0
                        deleted += 1
                        break
        return deleted

    def load_factor(self) -> float:
        return float(np.sum(self.buckets > 0) / (self.num_buckets * self.bucket_size))

    def memory_bytes(self) -> int:
        return int(self.buckets.nbytes)
