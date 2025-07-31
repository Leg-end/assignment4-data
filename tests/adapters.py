from __future__ import annotations

import os
from typing import Any
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
import fasttext
import re
import pathlib
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from collections import defaultdict
import itertools
import hashlib
import random

from cs336_data.deduplicate import *


DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data"


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        html_str = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        if encoding:
            try:
                html_str = html_bytes.decode(encoding, errors="ignore")
            except (UnicodeDecodeError, LookupError):
                print("unidentified encoding:", encoding)
                return None
        else:
            print("Unkown encoding")
            return None
    plain_text = extract_plain_text(html_str)
    return plain_text


def run_identify_language(text: str) -> tuple[Any, float]:
    model = fasttext.load_model(f"{DATA_PATH}/classifiers/lid.176.bin")
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    label = labels[0].replace("__label__", "")
    conf = float(probs[0])
    return label, conf


def run_mask_emails(text: str) -> tuple[str, int]:
    masked_text, count = re.subn(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', "|||EMAIL_ADDRESS|||", text)
    return masked_text, count


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    # 这里正则覆盖多种常见美国电话号码格式
    pattern = re.compile(
        r'''
        (?:(?:\+?1[\s\-.]*)?)         # 可选国家码
        (?:\(?\d{3}\)?)[\s\-\.]*      # 区号，可以有括号，后可跟空格、横线、点
        \d{3}[\s\-\.]*\d{4}           # 主体7位数字（3位+4位），中间分隔符可有
        ''', re.VERBOSE
    )
    masked_text, count = pattern.subn("|||PHONE_NUMBER|||", text)
    return masked_text, count


def run_mask_ips(text: str) -> tuple[str, int]:
    # IPv4 pattern, numbers from 0 to 255
    octet = r'(?:25[0-5]|2[0-4][0-9]|1?\d\d?)'
    ip_pattern = rf'\b{octet}\.{octet}\.{octet}\.{octet}\b'
    masked_text, count = re.subn(ip_pattern, "|||IP_ADDRESS|||", text)
    return masked_text, count


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    model = fasttext.load_model(f"{DATA_PATH}/classifiers/jigsaw_fasttext_bigrams_nsfw_final.bin")
    labels, probs = model.predict(text.strip().replace("\n", " "), k=1)
    label = labels[0].replace("__label__", "")
    conf = float(probs[0])
    return label, conf


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    model = fasttext.load_model(f"{DATA_PATH}/classifiers/jigsaw_fasttext_bigrams_hatespeech_final.bin")
    labels, probs = model.predict(text.strip().replace("\n", " "), k=1)
    label = labels[0].replace("__label__", "")
    conf = float(probs[0])
    return label, conf


def run_classify_quality(text: str) -> tuple[Any, float]:
    model = fasttext.load_model(f"{DATA_PATH}/classifiers/quality_classifier.bin")
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    label = labels[0].replace("__label__", "")
    conf = float(probs[0])
    return label, conf


def run_gopher_quality_filter(text: str) -> bool:
    words = word_tokenize(text)
    n = len(words)
    # Contain less than 50 or more than 100,000 words.
    if n < 50 or n > 100000:
        return False
    
    # Have a mean word length outside the range of 3 to 10 characters.
    num_word = sum(len(word) for word in words)
    mean_word = num_word / n
    if mean_word < 3 or mean_word > 10:
        return False
    
    # Have more than 30% of lines ending with an ellipsis (“...”).
    lines = text.splitlines()
    end_ellipsis_threshold = len(lines) * 0.3
    end_ellipsis = 0
    for line in lines:
        if line.endswith("..."):
            end_ellipsis += 1
            if end_ellipsis > end_ellipsis_threshold:
                return False
    
    # Contain less than 80% of words with at least one alphabetic character.
    without_alpha_threshold = n * 0.2
    without_alpha = 0
    for word in words:
        if not re.search('[a-zA-Z]', word):
            without_alpha += 1
            if without_alpha > without_alpha_threshold:
                return False
    return True        


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    line_count = defaultdict(int)
    for input_file in input_files:
        with open(input_file, 'r', encoding="utf-8") as f:
            for line in f:
                hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
                line_count[hash] += 1
    
    for input_file in input_files:
        with open(input_file, encoding="utf-8") as fr, \
            open(os.path.join(output_directory, os.path.basename(input_file)), "w", encoding="utf-8") as fw:
                for line in fr:
                    hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
                    if line_count[hash] == 1:
                        fw.write(line)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    fuzzy deduplication
    0. normalize the text by
      lowercasing, removing punctuation,
      normalizing whitespaces, removing accents
      applying NFD unicode normalization
    1. compute minhash signatures for each document
    2. use LSH to identify candidate duplicates
    3. compute the true ngram Jaccard similarity between candidate duplicates
    4. remove those that exceed a given threshold
    """
    hash_funcs = [stable_hash_func(seed) for seed in range(num_hashes)]
    band_size = num_hashes // num_bands
    buckets = [defaultdict(list) for _ in range(num_bands)]
    for i, input_file in enumerate(input_files):
        with open(input_file, encoding="utf-8") as f:
            text = f.read()
        norm_text = normalize_text(text)
        words = norm_text.split()
        ngrams = get_ngrams(words, ngrams)
        sig = minhash_signature(ngrams, hash_funcs)
        for band in range(num_bands):
            lo = band * band_size
            hi = lo + band_size
            buckets[band][sig[lo: hi]].append(i)
    
    cnts = defaultdict(int)
    candidate_pairs = set()
    for band in range(num_bands):
        for pair in itertools.combinations(buckets[band].values(), 2):
            candidate_pairs.add(pair)
            cnts[pair[0]] += 1
            cnts[pair[1]] += 1
    
    parent = list(range(len(input_files)))  # union set        
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
    
    cache = {}  # in case of storing all ngrams in memory, we use cache to buffer most frequently used ngrams
    for i, j in candidate_pairs:
        ngrams_i = load_ngrams_from_cache(i, input_files[i], cache, cnts)
        ngrams_j = load_ngrams_from_cache(j, input_files[j], cache, cnts)
        jac = jaccard_sim(ngrams_i, ngrams_j)
        if jac >= jaccard_threshold:  # cluster pair with jaccard similarity above certain threshold
            union(i, j)
    del cache
            
    clusters = defaultdict(list)
    for i in range(len(input_files)):
        root = find(i)
        clusters[root].append(i)
    
    for value in clusters.values():
        if len(value) > 1:  # keep only one doc in duplicated cluster
            keep_idx = random.choice(value)
        else:
            keep_idx = value[0]
        keep_file = input_files[keep_idx]
        output_path = os.path.join(output_directory, os.path.basename(keep_file))
        with open(keep_file, encoding="utf-8") as fr, \
            open(output_path, "w", encoding="utf-8") as fw:
                fw.write(fr.read())
    
    
