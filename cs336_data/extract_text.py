from fastwarc.warc import ArchiveIterator, WarcRecordType
from typing import Generator


def read_warc(file_path: str) -> Generator:
    with open(file_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.response:
                content = record.reader.read()
                yield content
            
def read_wet(file_path: str) -> Generator:
    with open(file_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.record_type == WarcRecordType.conversion:
                text = record.reader.read()
                yield text
