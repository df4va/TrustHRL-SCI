from __future__ import annotations

import gzip
import hashlib
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests


@dataclass(frozen=True)
class GEOAccession:
    accession: str
    organism: str
    modality: str
    role: str
    public_since: str
    series_matrix_url: str
    raw_archive_url: str

    @property
    def prefix(self) -> str:
        digits = self.accession[3:]
        return f"GSE{digits[:-3]}nnn"


ACCESSIONS = {
    "GSE5296": GEOAccession(
        accession="GSE5296",
        organism="Mus musculus",
        modality="microarray",
        role="cross-modal temporal validation",
        public_since="2006-07-12",
        series_matrix_url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE5nnn/GSE5296/matrix/"
            "GSE5296_series_matrix.txt.gz"
        ),
        raw_archive_url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE5nnn/GSE5296/suppl/" "GSE5296_RAW.tar"
        ),
    ),
    "GSE151371": GEOAccession(
        accession="GSE151371",
        organism="Homo sapiens",
        modality="bulk RNA-seq",
        role="cross-species validation",
        public_since="2021-01-22",
        series_matrix_url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE151nnn/GSE151371/matrix/"
            "GSE151371_series_matrix.txt.gz"
        ),
        raw_archive_url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE151nnn/GSE151371/suppl/"
            "GSE151371_raw_gene_counts_de-ID.csv.gz"
        ),
    ),
    "GSE189070": GEOAccession(
        accession="GSE189070",
        organism="Mus musculus",
        modality="single-cell RNA-seq",
        role="out-of-distribution validation",
        public_since="2021-11-21",
        series_matrix_url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE189nnn/GSE189070/matrix/"
            "GSE189070_series_matrix.txt.gz"
        ),
        raw_archive_url=(
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE189nnn/GSE189070/suppl/" "GSE189070_RAW.tar"
        ),
    ),
}


@dataclass(frozen=True)
class DownloadReceipt:
    path: Path
    bytes_received: int
    sha256: str
    source: str


class GEODownloader:
    def __init__(self, destination: str | Path, timeout_seconds: float = 120.0) -> None:
        self.destination = Path(destination)
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "trusthrl-sci/0.1"

    def accession(self, name: str) -> GEOAccession:
        try:
            return ACCESSIONS[name.upper()]
        except KeyError as error:
            raise ValueError(f"unsupported accession {name}") from error

    def stream(self, url: str) -> Iterator[bytes]:
        with self.session.get(url, stream=True, timeout=self.timeout_seconds) as response:
            response.raise_for_status()
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    yield block

    def download(self, url: str, name: str) -> DownloadReceipt:
        self.destination.mkdir(parents=True, exist_ok=True)
        target = self.destination / name
        digest = hashlib.sha256()
        bytes_received = 0
        with tempfile.NamedTemporaryFile(dir=self.destination, delete=False) as stream:
            temporary = Path(stream.name)
            for block in self.stream(url):
                stream.write(block)
                digest.update(block)
                bytes_received += len(block)
        temporary.replace(target)
        return DownloadReceipt(target, bytes_received, digest.hexdigest(), url)

    def series_matrix(self, accession: str) -> DownloadReceipt:
        record = self.accession(accession)
        return self.download(record.series_matrix_url, f"{record.accession}_series_matrix.txt.gz")

    def raw_archive(self, accession: str) -> DownloadReceipt:
        record = self.accession(accession)
        suffix = Path(record.raw_archive_url).suffix
        return self.download(record.raw_archive_url, f"{record.accession}_raw{suffix}")


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def decompress_gzip(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source_path, "rb") as compressed, target.open("wb") as output:
        shutil.copyfileobj(compressed, output)
    return target


def safe_extract_tar(source: str | Path, destination: str | Path) -> tuple[Path, ...]:
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(source, "r") as archive:
        for member in archive.getmembers():
            candidate = (target / member.name).resolve()
            if target not in candidate.parents and candidate != target:
                raise ValueError(f"unsafe archive path {member.name}")
        archive.extractall(target)
        extracted.extend(target / member.name for member in archive.getmembers() if member.isfile())
    return tuple(extracted)


def read_series_matrix(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    rows = []
    columns: list[str] = []
    with gzip.open(source, "rt", encoding="utf-8", errors="replace") as stream:
        inside = False
        for line in stream:
            if line.startswith("!series_matrix_table_begin"):
                inside = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not inside:
                continue
            values = line.rstrip("\n").split("\t")
            if not columns:
                columns = [value.strip('"') for value in values]
            else:
                rows.append([value.strip('"') for value in values])
    if not columns:
        raise ValueError("series matrix table was not found")
    frame = pd.DataFrame(rows, columns=columns)
    identifier = columns[0]
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.set_index(identifier)
