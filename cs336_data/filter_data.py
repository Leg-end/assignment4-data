from .extract_text import read_warc, read_wet
from .filters import *
from tldextract import TLDExtract
import concurrent.futures
import os
import pathlib
from tqdm import tqdm

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"

extractor = TLDExtract()
    


def filter_urls_by_domain(urls: list[str], allowed_domains: list[str]) -> list[str]:
    """
    过滤URL，只保留指定域名的链接
    :param urls: URL列表
    :param allowed_domains: 允许的域名集合(如 {'example.com', 'test.org'})
    :return: 过滤后的URL列表
    """
    filtered = []
    for url in urls:
        extracted = extractor(url)
        domain = extracted.registered_domain  # 获取完整注册域名
        if domain in allowed_domains:
            filtered.append(url)
    return filtered


def process_single_wet_file(input_path: str, output_path: str, filters: Compose) -> str:
    for wet in read_wet(input_path):
        wet = filters(wet.decode("utf-8"))
    if wet is not None:
        with open(output_path, "w") as f:
            f.write(wet)
    return output_path


def filter_parallely():
    filters = Compose([
        FastTextFilter(DATA_PATH / "classifiers" / "lid.176.bin", "en", 0.3),  # keep only English text
        GopherQualityFilter(),
        EmailMasker(),
        PhoneNumberMasker(),
        IPMasker(),
        FastTextFilter(DATA_PATH / "classifiers" / "jigsaw_fasttext_bigrams_nsfw_final.bin", "non_nsfw", 0.5),
        FastTextFilter(DATA_PATH / "classifiers" / "jigsaw_fasttext_bigrams_hatespeech_final.bin", "non_toxic", 0.5),
        FastTextFilter(DATA_PATH / "classifiers" / "quality_classifier.ftz", 0.5)
    ])
     # Set up the executor
    num_cpus = len(os.sched_getaffinity(0))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]
    output_directory_path = "/path/to/output_directory/"
    
    futures = []
    for wet_filepath in wet_filepaths:
        # For each warc.wet.gz filepath, submit a job to the executor and get a future back
        wet_filename = str(pathlib.Path(wet_filepath).name)
        future = executor.submit(
            process_single_wet_file,
            wet_filepath,
            os.path.join(output_directory_path, wet_filename),
            filters
        )
        # Store the futures
        futures.append(future)
        
    # Iterate over the completed futures as they finish, using a progress bar
    # to keep track of progress.
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(wet_filepaths),
        ):
        output_file = future.result()
        print(f"Output file written: {output_file}")