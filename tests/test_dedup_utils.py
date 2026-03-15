from __future__ import annotations

import unittest

from src.utils.dedup_utils import (
    generate_random_hash_functions,
    get_lsh_candidates,
    get_min_hash_signature,
    get_shingles,
    get_similarity,
)


class DedupUtilsTests(unittest.TestCase):
    def test_minhash_similarity_prefers_identical_text(self) -> None:
        hash_functions = generate_random_hash_functions(64, seed=7)
        sig_a = get_min_hash_signature(hash_functions, get_shingles("the cat sat on the mat"))
        sig_b = get_min_hash_signature(hash_functions, get_shingles("the cat sat on the mat"))
        sig_c = get_min_hash_signature(hash_functions, get_shingles("quantum fields and gauge theory"))
        self.assertEqual(get_similarity(sig_a, sig_b), 1.0)
        self.assertLess(get_similarity(sig_a, sig_c), 0.5)

    def test_lsh_candidates_for_identical_signatures(self) -> None:
        signature = list(range(128))
        candidates = get_lsh_candidates([signature, signature, list(range(128, 256))], num_bands=32)
        self.assertIn(1, candidates[0])
        self.assertIn(0, candidates[1])


if __name__ == "__main__":
    unittest.main()
