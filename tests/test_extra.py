from cs336_data.extract_text import read_warc, read_wet
from .adapters import run_extract_text_from_html_bytes, run_mask_emails,\
    run_mask_ips, run_mask_phone_numbers, run_gopher_quality_filter
from cs336_data.filters import FastTextFilter
    
import pathlib
import os

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"


def print_nearby(ori: str, masked: str, replacement: str, max_cnt:int=10):
    scope = 10
    i = masked.find(replacement)
    l_ori = len(ori)
    l_masked = len(masked)
    l_replacement = len(replacement)
    while i != -1 and max_cnt > 0:
        max_cnt -= 1
        print(ori[max(i - scope, 0): min(i + l_replacement + scope, l_ori)])
        print(masked[max(i - scope, 0): min(i + l_replacement + scope, l_masked)])
        print("#" * 30)
        i = masked.find(replacement, i + l_replacement)
        

def test_extract_text_from_warc():
    warc_path = os.path.join(DATA_PATH, "CC", "example.warc.gz")
    wet_path = os.path.join(DATA_PATH, "CC", "example.warc.wet.gz")
    for warc, wet in zip(read_warc(warc_path), read_wet(wet_path)):
        warc = run_extract_text_from_html_bytes(warc)
        wet = wet.decode("utf-8")
        break
    assert warc == wet


def test_identify_language_warc():
    warc_path = os.path.join(DATA_PATH, "CC", "example.warc.gz")
    model_path = os.path.join(DATA_PATH, "classifiers", "lid.176.bin")
    model = FastTextFilter(model_path)
    k = 20
    print()
    for warc in read_warc(warc_path):
        warc = run_extract_text_from_html_bytes(warc)
        k -= 1
        label, conf = model.predict(warc.replace("\n", " "), k=1)
        print(label, conf)
        print(warc[:20])
        print("="*50)
        if k == 0:
            break
    assert isinstance(conf, float)
    assert conf > 0
    
    
def test_classify_nsfw_warc():
    warc_path = os.path.join(DATA_PATH, "CC", "example.warc.gz")
    model_path = os.path.join(DATA_PATH, "classifiers", "jigsaw_fasttext_bigrams_nsfw_final.bin")
    model = FastTextFilter(model_path)
    k = 20
    print()
    for warc in read_warc(warc_path):
        warc = run_extract_text_from_html_bytes(warc)
        # k -= 1
        label, conf = model.predict(warc.strip().replace("\n", " "), k=1)
        if label == "nsfw":
            k -= 1
            print(label, conf)
            print(warc[:100])
            print("="*50)
        if k == 0:
            break
    assert isinstance(conf, float)
    assert conf > 0
    

def test_classify_toxic_speech_warc():
    warc_path = os.path.join(DATA_PATH, "CC", "example.warc.gz")
    model_path = os.path.join(DATA_PATH, "classifiers", "jigsaw_fasttext_bigrams_hatespeech_final.bin")
    model = FastTextFilter(model_path)
    k = 20
    print()
    for warc in read_warc(warc_path):
        warc = run_extract_text_from_html_bytes(warc)
        # k -= 1
        label, conf = model.predict(warc.strip().replace("\n", " "), k=1)
        if label == "toxic":
            k -= 1
            print(label, conf)
            print(warc[:20])
            print("="*50)
        if k == 0:
            break
    assert isinstance(conf, float)
    assert conf > 0

    
def test_mask_emails_warc():
    warc_path = os.path.join(DATA_PATH, "CC", "example.warc.gz")
    k = 20
    print()
    for warc in read_warc(warc_path):
        warc = run_extract_text_from_html_bytes(warc)
        masked_warc, n = run_mask_emails(warc)
        if n > 0:
            print(f"Total {n} emails")
            k -= 1
            print_nearby(warc, masked_warc, "|||EMAIL_ADDRESS|||")
            print("="*50)
        if k == 0:
            break
    assert n >= 0
    

def test_mask_phone_warc():
    warc_path = os.path.join(DATA_PATH, "CC", "example.warc.gz")
    k = 20
    print()
    for warc in read_warc(warc_path):
        warc = run_extract_text_from_html_bytes(warc)
        masked_warc, n = run_mask_phone_numbers(warc)
        if n > 0:
            k -= 1
            print(f"Total {n} phones")
            print_nearby(warc, masked_warc, "|||PHONE_NUMBER|||")
            print("="*50)
        if k == 0:
            break
    assert n >= 0
    

def test_mask_ip_warc():
    warc_path = os.path.join(DATA_PATH, "CC", "example.warc.gz")
    k = 20
    print()
    for warc in read_warc(warc_path):
        warc = run_extract_text_from_html_bytes(warc)
        masked_warc, n = run_mask_ips(warc)
        if n > 0:
            k -= 1
            print(f"Total {n} ips")
            print_nearby(warc, masked_warc, "|||IP_ADDRESS|||")
            print("="*50)
        if k == 0:
            break
    assert n >= 0
    

def test_gopher_quality_filter_warc():
    warc_path = os.path.join(DATA_PATH, "CC", "example.warc.gz")
    k = 20
    print()
    for warc in read_warc(warc_path):
        warc = run_extract_text_from_html_bytes(warc)
        discard = run_gopher_quality_filter(warc)
        k -= 1
        if discard:
            print("Discard bad text")
        else:
            print("Keep good text")
        print(warc[:100])
        print("="*50)
        if k == 0:
            break
    assert k <= 0
