import unicodedata
import re

import nltk

from typing import Callable


def normalize_text(text: str) -> str:
    # lowercasing
    text = text.lower()
    # unicode NFD: split compound str into units str, avoiding decoding issues
    text = unicodedata.normalize("NFD", text)
    # remove accent
    text = "".join(c for c in text if not unicodedata.combining(c))
    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # normalize whitespaces: replace multiple whitespaces with singe one
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_ngrams(words: list[str], n: int) -> set[tuple[str]]:
    if n > len(words):
        return set()
    return set(nltk.ngrams(words, n))


def minhash_signature(ngrams: set[tuple[str]], hash_funcs: list[Callable]) -> tuple[int]:
    sig = []
    for func in hash_funcs:
        if ngrams:
            sig.append(min(func(g) for g in ngrams))
        else:
            sig.append(0)
    return tuple(sig)


def jaccard_sim(set1: set[tuple[str]], set2: set[tuple[str]]) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def stable_hash_func(seed: int) -> Callable:
    def _hash(x: list[str] | tuple[str] | str) -> int:
        s = " ".join(x) if isinstance(x, (tuple, list)) else str(x)
        return hash((seed, s))
    return _hash


def load_ngrams_from_cache(index: int, input_file: str, cache: dict[int, dict], cnts: list[int]) -> set[tuple[str]]:
    if index not in cache:
        text = open(input_file, encoding="utf-8").read()
        norm_text = normalize_text(text)
        words = norm_text.split()
        ngrams = get_ngrams(words, ngrams)
        if cnts[index] > 1:
            cache[index] = ngrams
        else:  # for doc only show once, no need to cache it
            return ngrams
    else:
        ngrams = cache[index]
    cnts[index] -= 1
    if cnts[index] == 0:  # never used in later, thus remove it from cache
        cache.pop(index)
    return ngrams