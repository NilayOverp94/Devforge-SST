"""Deterministic file ingestion, encoding detection, format parsing, FHIR R4 validation, and quarantine."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
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
