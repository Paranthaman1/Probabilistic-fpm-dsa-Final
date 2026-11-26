import numpy as np

def mix_hash_vec(keys: np.ndarray, seed: int) -> np.ndarray:
    h = (keys ^ np.uint64(seed)) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))
