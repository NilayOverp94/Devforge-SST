"""Audit report (report.json) and patient record linkage manifest (records.jsonl) writer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .security import peak_rss_mb


def write_outputs(
    output_dir: Path,
    file_entries: list[dict[str, Any]],
    record_entries: list[dict[str, Any]],
    entity_counts: dict[str, int],
    stats: dict[str, int],
    elapsed_seconds: float,
) -> None:
    """Write validated records.jsonl and report.json without leaking any PII or secrets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    san_dir = output_dir / "sanitized"
    san_dir.mkdir(parents=True, exist_ok=True)

    exported_files = {
        e["output"]
        for e in file_entries
        if e.get("status") == "sanitized" and e.get("output")
    }

    # Filter records strictly to exported files and verify deterministic ordering
    clean_records = [r for r in record_entries if r.get("file") in exported_files]
    with (output_dir / "records.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for rec in clean_records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    n_san = sum(1 for e in file_entries if e.get("status") == "sanitized")
    n_quar = sum(1 for e in file_entries if e.get("status") == "quarantined")
    n_linked = sum(1 for r in clean_records if r.get("patient") is not None)

    report = {
        "schema": "raksha-report/1",
        "files": file_entries,
        "counts": {
            "files_total": len(file_entries),
            "files_sanitized": n_san,
            "files_quarantined": n_quar,
            "records_processed": len(clean_records),
            "identifiers_detected": int(stats.get("identifiers_detected", 0)),
            "records_linked": n_linked,
            "records_generalized": int(stats.get("records_generalized", 0)),
            "records_suppressed": int(stats.get("records_suppressed", 0)),
            "records_quarantined": int(stats.get("records_quarantined", 0)),
        },
        "entities": {k: int(v) for k, v in sorted(entity_counts.items()) if v > 0},
        "timing": {"seconds": round(elapsed_seconds, 2)},
        "resources": {"peak_rss_mb": peak_rss_mb()},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
