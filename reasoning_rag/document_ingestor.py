"""Utilities for importing external documents into the RAG index."""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".json", ".jsonl", ".csv", ".tsv"}
COMMON_TEXT_FIELDS = [
    "text", "content", "body", "abstract", "abstracttext",
    "summary", "question", "answer", "passage"
]


class DocumentIngestor:
    def __init__(self, chunk_size: int = 180, overlap: int = 30, text_field: Optional[str] = None):
        self.chunk_size = max(chunk_size, 40)
        self.overlap = max(0, min(overlap, self.chunk_size // 2))
        self.text_field = text_field.strip() if text_field else None

    def ingest_files(self, files: List[FileStorage], save_dir: Optional[Path] = None) -> Tuple[List[Dict], Dict]:
        """Convert uploaded files into chunked passages for indexing."""
        all_passages: List[Dict] = []
        imported_files: List[Dict] = []
        next_id = 0

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        for file_storage in files:
            filename = secure_filename(file_storage.filename or "")
            if not filename:
                continue

            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file format: {filename}")

            raw_bytes = file_storage.read()
            if save_dir:
                (save_dir / filename).write_bytes(raw_bytes)

            records = self._extract_records(raw_bytes, suffix)
            passages = []
            for record_index, record_text in enumerate(records):
                for chunk_index, chunk_text in enumerate(self._chunk_text(record_text)):
                    passages.append({
                        "id": next_id,
                        "text": chunk_text,
                        "source": "uploaded_document",
                        "source_file": filename,
                        "record_index": record_index,
                        "chunk_index": chunk_index,
                    })
                    next_id += 1

            imported_files.append({
                "filename": filename,
                "records": len(records),
                "chunks": len(passages),
                "extension": suffix,
            })
            all_passages.extend(passages)

        summary = {
            "files": imported_files,
            "file_count": len(imported_files),
            "chunk_count": len(all_passages),
        }
        return all_passages, summary

    def _extract_records(self, raw_bytes: bytes, suffix: str) -> List[str]:
        if suffix in {".txt", ".md"}:
            text = raw_bytes.decode("utf-8", errors="ignore")
            return [text] if text.strip() else []

        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            joined_text = "\n".join(page.strip() for page in pages if page.strip())
            return [joined_text] if joined_text else []

        if suffix in {".csv", ".tsv"}:
            delimiter = "," if suffix == ".csv" else "\t"
            text = raw_bytes.decode("utf-8", errors="ignore")
            rows = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            return [
                extracted for row in rows
                if (extracted := self._extract_text_from_record(row))
            ]

        if suffix == ".json":
            payload = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
            if isinstance(payload, list):
                return [extracted for item in payload if (extracted := self._extract_text_from_record(item))]
            if isinstance(payload, dict):
                if extracted := self._extract_text_from_record(payload):
                    return [extracted]
                nested_records = []
                for value in payload.values():
                    if isinstance(value, list):
                        nested_records.extend(
                            extracted for item in value
                            if (extracted := self._extract_text_from_record(item))
                        )
                return nested_records
            return []

        if suffix == ".jsonl":
            records = []
            for line in raw_bytes.decode("utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if extracted := self._extract_text_from_record(item):
                    records.append(extracted)
            return records

        return []

    def _extract_text_from_record(self, record) -> Optional[str]:
        if isinstance(record, str):
            return record.strip() or None

        if not isinstance(record, dict):
            return None

        if self.text_field and record.get(self.text_field):
            return str(record[self.text_field]).strip() or None

        lowered_map = {str(key).lower(): value for key, value in record.items()}
        for field in COMMON_TEXT_FIELDS:
            if lowered_map.get(field):
                return str(lowered_map[field]).strip() or None

        # Fall back to concatenating all string-ish values.
        string_values = [
            str(value).strip()
            for value in record.values()
            if isinstance(value, (str, int, float)) and str(value).strip()
        ]
        if string_values:
            return "\n".join(string_values)
        return None

    def _chunk_text(self, text: str) -> List[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        sentences = self._split_sentences(cleaned)
        words = cleaned.split()
        if len(words) <= self.chunk_size:
            return [cleaned]

        if len(sentences) <= 1:
            return self._fallback_word_chunks(words)

        chunks = []
        current_chunk: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_len = len(sentence_words)

            if sentence_len > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())
                    current_chunk = []
                    current_len = 0
                chunks.extend(self._fallback_word_chunks(sentence_words))
                continue

            if current_len + sentence_len > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk).strip())
                overlap_sentences = self._build_sentence_overlap(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_len = sum(len(part.split()) for part in current_chunk)
            else:
                current_chunk.append(sentence)
                current_len += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        return [chunk for chunk in chunks if chunk]

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?。！？])\s+', text)
        sentences = [part.strip() for part in parts if part.strip()]
        return sentences or [text]

    def _fallback_word_chunks(self, words: List[str]) -> List[str]:
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for start in range(0, len(words), step):
            chunk_words = words[start:start + self.chunk_size]
            if not chunk_words:
                continue
            chunks.append(" ".join(chunk_words).strip())
            if start + self.chunk_size >= len(words):
                break
        return chunks

    def _build_sentence_overlap(self, chunk_sentences: List[str]) -> List[str]:
        if not self.overlap or not chunk_sentences:
            return []

        overlap_sentences = []
        total_words = 0
        for sentence in reversed(chunk_sentences):
            sentence_len = len(sentence.split())
            if total_words + sentence_len > self.overlap and overlap_sentences:
                break
            overlap_sentences.insert(0, sentence)
            total_words += sentence_len
            if total_words >= self.overlap:
                break
        return overlap_sentences
