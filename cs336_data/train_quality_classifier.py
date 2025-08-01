from tests.adapters import run_extract_text_from_html_bytes

from .extract_text import read_warc
from .filters import EmailMasker, PhoneNumberMasker, IPMasker, GopherQualityFilter, Compose

import os
import pathlib
import fasttext

from tqdm import tqdm

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"


def test_make_quality_classifier_dataset():
    
    filters = Compose([GopherQualityFilter(), EmailMasker(), PhoneNumberMasker(), IPMasker()])
    
    num_examples = 5000
    i = j = 0
    html_path = os.path.join(DATA_PATH, "positive_urls.warc.warc.gz")
    with open(os.path.join(DATA_PATH, "train.txt"), "w", encoding="utf-8") as f:
        for text in tqdm(read_warc(html_path), desc="Reading wiki"):
            text = run_extract_text_from_html_bytes(text)
            if text:
                text = filters(text.replace('\n', ' '))
                if text:
                    f.write(f"__label__wiki {text}\n")
                    i += 1
                    if i == num_examples:
                        break
    
    warc_path = os.path.join(DATA_PATH, "CC", "example.warc.gz")
    with open(os.path.join(DATA_PATH, "train.txt"), "a", encoding="utf-8") as f:
        for warc in tqdm(read_warc(warc_path), desc="Reading warc"):
            warc = run_extract_text_from_html_bytes(warc)
            if warc:
                warc = filters(warc.replace('\n', ' '))
                if warc:
                    f.write(f"__label__cc {warc}\n")
                    j += 1
                    if j == num_examples:
                        break
                    

def test_train_quality_classifier():
    model = fasttext.train_supervised(
        input=os.path.join(DATA_PATH, "train.txt"),
        lr=0.5,
        epoch=5,
        wordNgrams=2,
        dim=100,
        verbose=2
    )
    
    model.save_model(os.path.join(DATA_PATH, "classifiers", "quality_classifier.ftz"))
    
    print("Complete training, test result on training set:")
    result = model.test(os.path.join(DATA_PATH, "train.txt"))
    print(f"Precision: {result[1]}")
    print(f"Recall:    {result[2]}")
    print(f"Tested samples: {result[0]}")
    

if __name__ == "__main__":
    # test_make_quality_classifier_dataset()
    test_train_quality_classifier()
