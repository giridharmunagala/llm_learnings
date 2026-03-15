"""De-duplication utilities for MinHash + LSH candidate generation."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Set
import hashlib
import re
import random
import xxhash


def _intdigest64(value: str) -> int:
    return xxhash.xxh3_64_intdigest(value)


def _intdigest128(value: str) -> int:
    return xxhash.xxh3_128_intdigest(value)


def get_min_hash_signature(hash_functions: Sequence[Callable[[str], int]], data_set: Iterable[str]) -> List[int]:
    """
    Using Min hash function computes the signature of a set of data.
    @args:
        hash_functions: A list of hash functions to be applied on the data set.
        data_set: A set of data for which the signature is to be computed.
    """
    values = list(data_set)
    if not values:
        return []

    signature = []
    for hash_function in hash_functions:
        min_hash_value = min(hash_function(item) for item in values)
        signature.append(min_hash_value)
    return signature


def get_lsh_candidates(all_signatures: List[List[int]], num_bands: int = 32) -> List[List[int]]:
    """
    Using LSH function computes the candidates for a given signature.
    Steps: Divides all signatures into bands -> converts each band into a hash value using xxh3_64 algorithm.
    -> Compares hashes with all other candidates, checking if any 2 hashes are matched. 
    If none matched then no point in checking. If 2 matched, full comparison is required.

    @args:
        all_signatures: A list of signatures for which the candidates are to be computed.
        num_bands: The number of bands to be used for LSH.
    """
    if not all_signatures:
        return []

    signature_len = len(all_signatures[0])
    if signature_len == 0 or num_bands <= 0 or signature_len < num_bands:
        raise ValueError("signature length must be >= num_bands and both must be positive")
    if any(len(sig) != signature_len for sig in all_signatures):
        raise ValueError("all signatures must have the same length")
    if signature_len % num_bands != 0:
        raise ValueError("signature length must be divisible by num_bands")

    # Divide the signatures into bands
    band_width = signature_len // num_bands
    all_sig_band_hashes = list()
    for signature in all_signatures:
        sig_band_hash = []
        for band_idx in range(num_bands):
            start = band_idx * band_width
            band = " ".join(str(band_val) for band_val in signature[start : start + band_width])
            sig_band_hash.append(_intdigest64(band))
        all_sig_band_hashes.append(sig_band_hash)

    def is_candidate(band_hashes_1: List[int], band_hashes_2: List[int]) -> bool:
        return bool(set(band_hashes_1) & set(band_hashes_2))

    # Identifies candidates for each signature. Provides the corresponding indices.
    all_sig_candidates = list()
    for s_1 in range(len(all_sig_band_hashes)):
        sig_candidates = list()
        for s_2 in range(len(all_sig_band_hashes)):
            if s_1 != s_2:
                if is_candidate(all_sig_band_hashes[s_1],all_sig_band_hashes[s_2]):
                    sig_candidates.append(s_2)
        all_sig_candidates.append(sig_candidates)
    
    return all_sig_candidates

def get_similarity(signature_1: Sequence[int], signature_2: Sequence[int]) -> float:
    """
    Computes the approx Jaccard similarity between two sets of MinHash signatures.
    @args:
        signature_1: The first set of signatures for which the similarity is to be computed.
        signature_2:  The second set of signatures for which the similarity is to be computed.
    """
    assert len(signature_1) == len(signature_2) and len(signature_1) > 0, "Both signatures must be of the same non-zero length"
    approx_jc_similarity = 1 / len(signature_1) * sum(1 if s1 == s2 else 0 for s1, s2 in zip(signature_1, signature_2))
    return approx_jc_similarity


def get_shingles(text: str, min_k: int = 2, max_k: int = 5) -> Set[str]:
    """
    Get the set of shingles for a given text.
    @args:
        text: The input text for which the shingles are to be generated.
        min_k: The minimum length of the shingles.
        max_k: The maximum length of the shingles.
    """
    all_shingles: list[str] = []
    def normalize_text(text: str) -> str:
        # Remove punctuation and special characters, and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation
        text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single space
        return text.lower()  # Convert to lowercase for uniformity
    
    text_normalized = normalize_text(text)  # Convert to lowercase for uniformity
    tokens = text_normalized.split()  # Split the text into tokens
    for k in range(min_k, max_k + 1):
        all_k_shingles = list()
        for i in range(len(tokens) - k + 1):
            shingle = ' '.join(tokens[i:i+k])
            all_k_shingles.append(shingle)
        all_shingles.extend(all_k_shingles)
    return set(all_shingles)

def generate_random_hash_functions(
    num_functions: int,
    max_value: int = 2147483647,
    seed: int | None = 8,
) -> List[Callable[[str], int]]:
    """
    Generate a list of random hash functions. 
    Each hash function takes a string value and hashes it using xxh3_128 algorithm & then applies one of the random hash functions to it.
    @args:
        num_functions: The number of hash functions to generate.
        max_value: The maximum value for the hash functions.
    """
    rng = random.Random(seed)
    hash_functions = []
    for _ in range(num_functions):
        a = rng.randint(1, max_value - 1)
        b = rng.randint(0, max_value - 1)
        hash_function = lambda x, a=a, b=b: (a * _intdigest128(x) + b) % max_value
        hash_functions.append(hash_function)
    return hash_functions
