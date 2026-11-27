"""Deterministic file ingestion, encoding detection, format parsing, FHIR R4 validation, and quarantine."""
from __future__ import annotations

import codecs
import csv
import hashlib
import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .security import sha256_bytes

SUPPORTED_EXTS = {".csv", ".tsv", ".json", ".txt"}
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_BAD_CP1252_RE = re.compile(rb"[\x00\x81\x8d\x8f\x90\x9d\xfe\xff]")
_JSON_TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*"|([{}\[\]])')


class QuarantineError(ValueError):
    """Raised with a safe, PII-free reason code when a file cannot be safely processed."""


@dataclass
class IngestedFile:
    rel_path: str
    sha256: str
    ext: str
    fmt: str  # csv | tsv | json | fhir | text
    delimiter: str | None = None
    raw_size: int = 0
    text: str | None = None
    sample_text: str | None = None
    data: Any = None
    quarantine_reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FileMeta:
    """Compact streaming metadata with zero parsed rows/DOM held across files."""

    rel_path: str
    path: Path
    sha256: str
    ext: str
    fmt: str  # csv | tsv | json | fhir | text | quarantined
    delimiter: str | None = None
    encoding: str = "utf-8"
    raw_size: int = 0
    sample_text: str = ""
    quarantine_reason: str | None = None
    json_array_key: str | None = None
    json_hospital: str = ""


def decode_text_strict(raw: bytes) -> str:
    """Decode UTF-8 BOM, UTF-16 BOM, UTF-8, or clean cp1252; reject corrupt binary bytes."""
    if not raw.strip():
        raise QuarantineError("empty_file")

    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            text = raw[3:].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise QuarantineError("corrupt_utf8_bom") from exc
        if "\x00" in text:
            raise QuarantineError("binary_nul_bytes")
        return text

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            text = raw.decode("utf-16", errors="strict")
        except UnicodeDecodeError as exc:
            raise QuarantineError("corrupt_utf16") from exc
        if "\x00" in text:
            raise QuarantineError("binary_nul_bytes")
        return text

    if b"\x00" in raw:
        raise QuarantineError("binary_nul_bytes")

    try:
        text = raw.decode("utf-8", errors="strict")
        # Reject suspicious C0 controls other than \t, \n, \r
        if _CTRL_CHAR_RE.search(text):
            raise QuarantineError("control_bytes")
        return text
    except UnicodeDecodeError:
        pass

    # Fallback cp1252 ONLY for clean legacy text exports (reject corrupt UTF-8 fragments like \xff\xfe\xc3)
    if _BAD_CP1252_RE.search(raw):
        raise QuarantineError("corrupt_encoding")
    if b"\xc3(" in raw or b"\xc2(" in raw:
        raise QuarantineError("corrupt_encoding")

    try:
        text = raw.decode("cp1252", errors="strict")
    except UnicodeDecodeError as exc:
        raise QuarantineError("undecodable_text") from exc

    if _CTRL_CHAR_RE.search(text):
        raise QuarantineError("control_bytes")
    return text


def check_json_depth(text: str, max_depth: int) -> None:
    """Iterative JSON bracket-depth check without Python stack overflow."""
    depth = 0
    for m in _JSON_TOKEN_RE.finditer(text):
        ch = m.group(1)
        if ch is not None:
            if ch in ("{", "["):
                depth += 1
                if depth > max_depth:
                    raise QuarantineError("excessive_nesting")
            else:
                depth -= 1


def looks_non_csv_payload(text: str) -> bool:
    s = text.lstrip()
    return s.startswith(("{", "[", "<?xml", "<ClinicalDocument", "<!DOCTYPE", "<html", "%PDF"))


def validate_fhir_bundle(bundle: dict[str, Any]) -> bool:
    """Validate structural R4 Document/Collection Bundle invariants."""
    if not isinstance(bundle, dict) or bundle.get("resourceType") != "Bundle":
        return False
    if "type" not in bundle or "entry" not in bundle or not isinstance(bundle["entry"], list):
        return False
    full_urls = set()
    resource_refs = set()
    for ent in bundle["entry"]:
        if not isinstance(ent, dict):
            return False
        res = ent.get("resource")
        if not isinstance(res, dict) or not res.get("resourceType"):
            return False
        if ent.get("fullUrl"):
            full_urls.add(ent["fullUrl"])
        if res.get("id"):
            resource_refs.add(f"{res['resourceType']}/{res['id']}")

    # Ensure internal references resolve inside the bundle
    valid_targets = full_urls | resource_refs

    def _check_refs(obj: Any) -> bool:
        if isinstance(obj, dict):
            ref = obj.get("reference")
            if isinstance(ref, str) and (ref.startswith("urn:uuid:") or "/" in ref):
                if ref.startswith("urn:uuid:") and ref not in valid_targets:
                    return False
                if "/" in ref and not ref.startswith(("http://", "https://")) and ref not in valid_targets:
                    return False
            return all(_check_refs(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_check_refs(x) for x in obj)
        return True

    return _check_refs(bundle["entry"])


def _sha256_file_stream(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4194304)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_file(
    path: Path,
    input_root: Path,
    max_json_depth: int = 200,
    max_file_bytes: int = 4294967296,
) -> IngestedFile:
    """Read, validate, and parse a single input file, or mark quarantined on any anomaly."""
    rel = path.relative_to(input_root).as_posix()
    ext = path.suffix.lower()
    raw_size = path.stat().st_size

    if raw_size > max_file_bytes:
        return IngestedFile(
            rel_path=rel,
            sha256=_sha256_file_stream(path),
            ext=ext,
            fmt="quarantined",
            raw_size=raw_size,
            quarantine_reason="file_too_large",
        )
    if ext not in SUPPORTED_EXTS:
        return IngestedFile(
            rel_path=rel,
            sha256=_sha256_file_stream(path),
            ext=ext,
            fmt="quarantined",
            raw_size=raw_size,
            quarantine_reason="unsupported_format",
        )

    raw = path.read_bytes()
    digest = sha256_bytes(raw)

    try:
        text = decode_text_strict(raw)
        del raw

        if ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            if looks_non_csv_payload(text):
                raise QuarantineError("mislabeled_format")
            rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim))
            if not rows:
                raise QuarantineError("empty_csv")
            if len(rows) == 1 and any(c.strip().startswith(("{", "<")) for c in rows[0]):
                raise QuarantineError("mislabeled_format")
            return IngestedFile(
                rel_path=rel,
                sha256=digest,
                ext=ext,
                fmt="tsv" if ext == ".tsv" else "csv",
                delimiter=delim,
                raw_size=raw_size,
                text=None,
                sample_text=text[:65536],
                data=rows,
            )

        if ext == ".json":
            check_json_depth(text, max_depth=max_json_depth)
            try:
                data = json.loads(text)
            except (ValueError, RecursionError) as exc:
                raise QuarantineError("invalid_json") from exc
            is_fhir = isinstance(data, dict) and data.get("resourceType") == "Bundle"
            if is_fhir and not validate_fhir_bundle(data):
                raise QuarantineError("invalid_fhir_bundle")
            return IngestedFile(
                rel_path=rel,
                sha256=digest,
                ext=ext,
                fmt="fhir" if is_fhir else "json",
                raw_size=raw_size,
                text=None,
                sample_text=text[:65536],
                data=data,
            )

        # .txt
        return IngestedFile(
            rel_path=rel,
            sha256=digest,
            ext=ext,
            fmt="text",
            raw_size=raw_size,
            text=text,
            sample_text=text[:65536],
            data=None,
        )

    except QuarantineError as exc:
        return IngestedFile(
            rel_path=rel,
            sha256=digest,
            ext=ext,
            fmt="quarantined",
            raw_size=raw_size,
            quarantine_reason=str(exc),
        )
    except (csv.Error, UnicodeError, RecursionError, ValueError) as exc:
        return IngestedFile(
            rel_path=rel,
            sha256=digest,
            ext=ext,
            fmt="quarantined",
            raw_size=raw_size,
            quarantine_reason=type(exc).__name__,
        )


def ingest_directory(
    input_dir: Path,
    max_json_depth: int = 200,
    max_file_bytes: int = 4294967296,
) -> list[IngestedFile]:
    """Deterministically walk and parse every file under input_dir."""
    paths = sorted((p for p in input_dir.rglob("*") if p.is_file()), key=lambda p: p.relative_to(input_dir).as_posix())
    return [parse_file(p, input_dir, max_json_depth=max_json_depth, max_file_bytes=max_file_bytes) for p in paths]


def _stream_validate_encoding(path: Path) -> tuple[str, str, str]:
    """Compute SHA-256 and strictly validate text encoding in 4 MB chunks (< 16 MB peak RAM)."""
    h = hashlib.sha256()
    has_non_ws = False
    head_bytes = b""
    enc = "utf-8"
    utf8_failed = False
    utf8_dec = None
    sample_parts: list[str] = []
    sample_len = 0

    with path.open("rb") as fh:
        first_chunk = fh.read(4_194_304)
        if not first_chunk:
            raise QuarantineError("empty_file")
        h.update(first_chunk)
        if first_chunk.strip():
            has_non_ws = True

        if first_chunk.startswith(b"\xef\xbb\xbf"):
            enc = "utf-8-sig"
            utf8_dec = codecs.getincrementaldecoder("utf-8")(errors="strict")
            dec_input = first_chunk[3:]
        elif first_chunk.startswith((b"\xff\xfe", b"\xfe\xff")):
            enc = "utf-16"
            utf8_dec = codecs.getincrementaldecoder("utf-16")(errors="strict")
            dec_input = first_chunk
        else:
            enc = "utf-8"
            utf8_dec = codecs.getincrementaldecoder("utf-8")(errors="strict")
            dec_input = first_chunk

        raw_check = first_chunk
        while True:
            if enc != "utf-16" and b"\x00" in raw_check:
                raise QuarantineError("binary_nul_bytes")

            if not utf8_failed:
                try:
                    txt = utf8_dec.decode(dec_input, final=False)
                except UnicodeDecodeError as exc:
                    if enc == "utf-8-sig":
                        raise QuarantineError("corrupt_utf8_bom") from exc
                    if enc == "utf-16":
                        raise QuarantineError("corrupt_utf16") from exc
                    utf8_failed = True
                else:
                    if "\x00" in txt:
                        raise QuarantineError("binary_nul_bytes")
                    if _CTRL_CHAR_RE.search(txt):
                        raise QuarantineError("control_bytes")
                    if sample_len < 65536 and txt:
                        need = 65536 - sample_len
                        sample_parts.append(txt[:need])
                        sample_len += min(len(txt), need)

            raw_check = fh.read(4_194_304)
            if not raw_check:
                break
            h.update(raw_check)
            if not has_non_ws and raw_check.strip():
                has_non_ws = True
            dec_input = raw_check

    if not has_non_ws:
        raise QuarantineError("empty_file")

    if not utf8_failed:
        try:
            tail = utf8_dec.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            if enc == "utf-8-sig":
                raise QuarantineError("corrupt_utf8_bom") from exc
            if enc == "utf-16":
                raise QuarantineError("corrupt_utf16") from exc
            utf8_failed = True
        else:
            if "\x00" in tail:
                raise QuarantineError("binary_nul_bytes")
            if _CTRL_CHAR_RE.search(tail):
                raise QuarantineError("control_bytes")
            if sample_len < 65536 and tail:
                sample_parts.append(tail[: 65536 - sample_len])
            return h.hexdigest(), enc, "".join(sample_parts)

    # cp1252 fallback validation pass in 4 MB chunks
    sample_parts = []
    sample_len = 0
    dec1252 = codecs.getincrementaldecoder("cp1252")(errors="strict")
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(4_194_304)
            if not chunk:
                break
            if _BAD_CP1252_RE.search(chunk) or b"\xc3(" in chunk or b"\xc2(" in chunk:
                raise QuarantineError("corrupt_encoding")
            try:
                txt = dec1252.decode(chunk, final=False)
            except UnicodeDecodeError as exc:
                raise QuarantineError("undecodable_text") from exc
            if _CTRL_CHAR_RE.search(txt):
                raise QuarantineError("control_bytes")
            if sample_len < 65536 and txt:
                need = 65536 - sample_len
                sample_parts.append(txt[:need])
                sample_len += min(len(txt), need)
    return h.hexdigest(), "cp1252", "".join(sample_parts)


def scan_file_meta(
    path: Path,
    input_root: Path,
    max_json_depth: int = 200,
    max_file_bytes: int = 4294967296,
) -> FileMeta:
    """Validate and summarize any file up to 4 GB without keeping its contents in memory."""
    rel = path.relative_to(input_root).as_posix()
    ext = path.suffix.lower()
    raw_size = path.stat().st_size

    if raw_size > max_file_bytes:
        return FileMeta(rel_path=rel, path=path, sha256=_sha256_file_stream(path), ext=ext, fmt="quarantined", raw_size=raw_size, quarantine_reason="file_too_large")
    if ext not in SUPPORTED_EXTS:
        return FileMeta(rel_path=rel, path=path, sha256=_sha256_file_stream(path), ext=ext, fmt="quarantined", raw_size=raw_size, quarantine_reason="unsupported_format")

    # Files <= 128 MB for JSON/FHIR can validate full structure directly in worker then drop memory immediately
    if ext == ".json" and raw_size <= 128_000_000:
        inf = parse_file(path, input_root, max_json_depth=max_json_depth, max_file_bytes=max_file_bytes)
        arr_key = None
        hosp = ""
        if inf.fmt == "json" and isinstance(inf.data, dict):
            if isinstance(inf.data.get("_export_meta"), dict):
                hosp = str(inf.data["_export_meta"].get("hospital", ""))
            elif isinstance(inf.data.get("facility"), dict):
                hosp = str(inf.data["facility"].get("name", ""))
            for k, v in inf.data.items():
                if k.lower() in ("visits", "patients", "results", "records", "claims") and isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    arr_key = k
                    break
        return FileMeta(
            rel_path=rel,
            path=path,
            sha256=inf.sha256,
            ext=ext,
            fmt=inf.fmt,
            encoding="utf-8",
            raw_size=raw_size,
            sample_text=inf.sample_text or "",
            quarantine_reason=inf.quarantine_reason,
            json_array_key=arr_key,
            json_hospital=hosp,
        )

    digest = ""
    try:
        digest, enc, sample = _stream_validate_encoding(path)
        if ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            if looks_non_csv_payload(sample):
                raise QuarantineError("mislabeled_format")
            with path.open("r", encoding=enc, newline="") as fh:
                reader = csv.reader(fh, delimiter=delim)
                first_row = next(reader, None)
                if first_row is None:
                    raise QuarantineError("empty_csv")
                second_row = next(reader, None)
                if second_row is None and any(c.strip().startswith(("{", "<")) for c in first_row):
                    raise QuarantineError("mislabeled_format")
            return FileMeta(
                rel_path=rel, path=path, sha256=digest, ext=ext,
                fmt="tsv" if ext == ".tsv" else "csv",
                delimiter=delim, encoding=enc, raw_size=raw_size, sample_text=sample,
            )

        if ext == ".json":
            # Stream JSON depth & structural validation for files > 128 MB
            _check_large_json_depth(path, enc, max_json_depth)
            is_fhir = '"resourcetype"' in sample.lower() and '"bundle"' in sample.lower()
            arr_key = None
            for cand in ("visits", "patients", "results", "records", "claims", "entry"):
                if f'"{cand}"' in sample:
                    arr_key = cand
                    break
            return FileMeta(
                rel_path=rel, path=path, sha256=digest, ext=ext,
                fmt="fhir" if is_fhir else "json",
                encoding=enc, raw_size=raw_size, sample_text=sample,
                json_array_key=arr_key,
            )

        return FileMeta(
            rel_path=rel, path=path, sha256=digest, ext=ext, fmt="text",
            encoding=enc, raw_size=raw_size, sample_text=sample,
        )
    except QuarantineError as exc:
        if not digest:
            digest = _sha256_file_stream(path)
        return FileMeta(rel_path=rel, path=path, sha256=digest, ext=ext, fmt="quarantined", raw_size=raw_size, quarantine_reason=str(exc))
    except (csv.Error, UnicodeError, RecursionError, ValueError) as exc:
        if not digest:
            digest = _sha256_file_stream(path)
        return FileMeta(rel_path=rel, path=path, sha256=digest, ext=ext, fmt="quarantined", raw_size=raw_size, quarantine_reason=type(exc).__name__)


def _check_large_json_depth(path: Path, encoding: str, max_depth: int) -> None:
    """Iterative chunked JSON bracket-depth check for multi-GB JSON payloads."""
    depth = 0
    in_str = False
    esc = False
    with path.open("r", encoding=encoding) as fh:
        while True:
            chunk = fh.read(4_194_304)
            if not chunk:
                break
            for ch in chunk:
                if esc:
                    esc = False
                elif in_str:
                    if ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch in "{[":
                    depth += 1
                    if depth > max_depth:
                        raise QuarantineError("excessive_nesting")
                elif ch in "}]":
                    depth -= 1


def iter_decoded_text_chunks(path: Path, encoding: str, target_chars: int = 262_144) -> Iterator[str]:
    """Stream large plain-text files along paragraph/newline boundaries with minimal memory."""
    buf = ""
    with path.open("r", encoding=encoding, newline="") as fh:
        while True:
            piece = fh.read(target_chars)
            if not piece and not buf:
                break
            buf += piece
            if not piece:
                yield buf
                break
            while len(buf) >= target_chars:
                cut = buf.rfind("\n\n", 0, target_chars)
                if cut > 0:
                    cut += 2
                else:
                    cut = buf.rfind("\n", 0, target_chars)
                    cut = (cut + 1) if cut > 0 else target_chars
                yield buf[:cut]
                buf = buf[cut:]


def read_large_json_header(
    path: Path,
    encoding: str,
    array_key: str,
) -> tuple[dict[str, Any], int]:
    """Read sibling metadata keys before array_key and return (header_meta, char_offset_after_bracket)."""
    pat = re.compile(rf'"{re.escape(array_key)}"\s*:\s*\[')
    buf = ""
    with path.open("r", encoding=encoding) as fh:
        while True:
            piece = fh.read(262_144)
            buf += piece
            m = pat.search(buf)
            if m:
                prefix_raw = buf[: m.start()].rstrip()
                header_meta: dict[str, Any] = {}
                try:
                    if prefix_raw.startswith("{"):
                        header_meta = json.loads(prefix_raw.rstrip(",") + "}")
                except Exception:
                    header_meta = {}
                return header_meta, m.end()
            if not piece or len(buf) > 4_194_304:
                raise QuarantineError("invalid_json")


def iter_large_json_array_batches(
    path: Path,
    encoding: str,
    start_char_offset: int,
    batch_size: int = 4000,
) -> Iterator[list[dict[str, Any]]]:
    """Stream batches of dicts from a multi-hundred-MB JSON array inside an open file handle."""
    dec = json.JSONDecoder()
    with path.open("r", encoding=encoding) as fh:
        # Advance cleanly by char count so UTF-8 char offsets match exactly
        remaining = start_char_offset
        while remaining > 0:
            skipped = fh.read(min(remaining, 1_048_576))
            if not skipped:
                return
            remaining -= len(skipped)

        buf = ""
        pos = 0
        eof = False
        batch: list[dict[str, Any]] = []

        while True:
            while not eof and len(buf) - pos < 1_048_576:
                more = fh.read(4_194_304)
                if not more:
                    eof = True
                    break
                buf = buf[pos:] + more
                pos = 0
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(buf):
                if eof:
                    break
                continue
            if buf[pos] == "]":
                break
            try:
                obj, end_idx = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise QuarantineError("invalid_json")
                more = fh.read(4_194_304)
                if not more:
                    raise QuarantineError("invalid_json")
                buf = buf[pos:] + more
                pos = 0
                continue
            pos = end_idx
            if isinstance(obj, dict):
                batch.append(obj)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
