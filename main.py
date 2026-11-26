#!/usr/bin/env python3
"""Raksha Kavach: Offline, CPU-only, LLM-free privacy firewall for Indian clinical records."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src.detection import PIIDetector
from src.ingestion import IngestedFile, ingest_directory
from src.linkage import RecordClues, extract_clues_from_file, infer_file_date_hints, link_all_records
from src.privacy import _sanitize_key_or_header, build_privacy_profiles, sanitize_csv_rows, sanitize_file
from src.reporting import write_outputs
from src.security import EncryptedVault, KeyRing, safe_reason, sanitize_filename

_WORKER_DETECTOR: PIIDetector | None = None


def _init_worker() -> None:
    global _WORKER_DETECTOR
    _WORKER_DETECTOR = PIIDetector()


def _extract_clues_worker(args: tuple[IngestedFile, str, int]) -> list[RecordClues]:
    inf, date_hint, modal_year = args
    assert _WORKER_DETECTOR is not None
    return extract_clues_from_file(inf, detector=_WORKER_DETECTOR, date_hint=date_hint, modal_year=modal_year)


def _sanitize_csv_chunk_worker(
    args: tuple[
        str,
        str,
        list[str],
        int | None,
        list[list[str]],
        int,
        str,
        int,
        dict[str, RecordClues],
        dict[str, str | None],
        dict[str, Any],
    ],
) -> tuple[str, list[dict[str, Any]], dict[str, int], int]:
    (
        rel_path,
        delimiter,
        col_roles,
        psn_col_idx,
        rows_chunk,
        start_row_idx,
        date_hint,
        modal_year,
        clues_by_loc,
        unit_to_psn,
        unit_profiles,
    ) = args
    assert _WORKER_DETECTOR is not None
    return sanitize_csv_rows(
        rel_path=rel_path,
        delimiter=delimiter,
        col_roles=col_roles,
        psn_col_idx=psn_col_idx,
        rows_chunk=rows_chunk,
        start_row_idx=start_row_idx,
        date_hint=date_hint,
        modal_year=modal_year,
        clues_by_loc=clues_by_loc,
        unit_to_psn=unit_to_psn,
        unit_profiles=unit_profiles,
        detector=_WORKER_DETECTOR,
    )


def load_policy(policy_path: Path) -> dict[str, Any]:
    if not policy_path.is_file():
        return {}
    loaded = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Raksha Kavach Privacy Firewall")
    ap.add_argument("--input", required=True, help="Read-only input directory")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--policy", default=None, help="Optional path to policy.yaml")
    args = ap.parse_args()

    t0 = time.monotonic()
    root_dir = Path(__file__).resolve().parent
    policy_path = Path(args.policy) if args.policy else (root_dir / "config" / "policy.yaml")
    policy = load_policy(policy_path)

    inp_dir = Path(args.input)
    out_dir = Path(args.output)
    san_dir = out_dir / "sanitized"
    san_dir.mkdir(parents=True, exist_ok=True)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

    keyring = KeyRing(
        pseudo_prefix=str(policy.get("pseudonym_prefix", "PSN")),
        pseudo_len=int(policy.get("pseudonym_length", 6)),
    )
    use_vault = bool(policy.get("encrypt_intermediate_vault", True))
    vault = EncryptedVault(keyring.vault_key()) if use_vault else None
    detector = PIIDetector()

    # Pass 1: Ingestion & Fail-Safe Quarantine
    ingested_files: list[IngestedFile] = ingest_directory(
        inp_dir,
        max_json_depth=int(policy.get("max_json_depth", 200)),
        max_file_bytes=int(policy.get("max_file_bytes", 104857600)),
    )

    # Pass 2: Dataset-wide date-hint & hospital consensus inference
    date_hints, modal_year = infer_file_date_hints(ingested_files)

    # Pass 3: Clue Extraction (parallelize across files when dataset is large)
    all_clues: list[RecordClues] = []
    clues_by_file: dict[str, list[RecordClues]] = {}
    valid_files = [inf for inf in ingested_files if not inf.quarantine_reason]
    total_valid_bytes = sum(inf.raw_size for inf in valid_files)
    n_workers = min(os.cpu_count() or 1, 8)
    use_pool = n_workers > 1 and total_valid_bytes >= 500_000

    if use_pool and len(valid_files) >= 2:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_workers, len(valid_files)), initializer=_init_worker) as ex:
            tasks = [(inf, date_hints.get(inf.rel_path, "dmy"), modal_year) for inf in valid_files]
            for inf, f_clues in zip(valid_files, ex.map(_extract_clues_worker, tasks)):
                clues_by_file[inf.rel_path] = f_clues
                all_clues.extend(f_clues)
    else:
        for inf in valid_files:
            f_clues = extract_clues_from_file(
                inf,
                detector=detector,
                date_hint=date_hints.get(inf.rel_path, "dmy"),
                modal_year=modal_year,
            )
            clues_by_file[inf.rel_path] = f_clues
            all_clues.extend(f_clues)

    # Pass 4: Conservative Cross-Script Record Linkage & HMAC Pseudonyms
    unit_to_psn, psn_clusters, ref_lookup = link_all_records(all_clues, keyring=keyring, policy=policy)

    # Pass 5: Cohort Risk Analysis & Privacy Profiles
    _, unit_profiles = build_privacy_profiles(
        all_clues=all_clues,
        unit_to_psn=unit_to_psn,
        psn_clusters=psn_clusters,
        keyring=keyring,
        policy=policy,
    )

    # Pass 6: Sanitization, Generalization, Filename Scrubbing, & Deterministic Export
    file_entries: list[dict[str, Any]] = []
    record_entries: list[dict[str, Any]] = []
    entity_counts: dict[str, int] = {}
    stats: dict[str, int] = {"records_generalized": 0, "records_suppressed": 0, "records_quarantined": 0}
    used_output_paths: set[str] = set()

    for inf in ingested_files:
        entry: dict[str, Any] = {"input_sha256": inf.sha256}
        if inf.quarantine_reason:
            entry.update(output=None, status="quarantined", reason=inf.quarantine_reason)
            file_entries.append(entry)
            continue

        try:
            f_clues = clues_by_file.get(inf.rel_path, [])
            doc_psn: str | None = None
            if len(f_clues) == 1:
                doc_psn = unit_to_psn.get(f_clues[0].unit_id)

            if inf.fmt in ("csv", "tsv") and inf.data and len(inf.data) > 4001 and n_workers > 1:
                rows = inf.data
                header = list(rows[0])
                body = rows[1:]
                col_roles = [
                    detector.classify_column_header(h, [r[i] for r in body[:25] if i < len(r)])
                    for i, h in enumerate(header)
                ]
                used_headers: set[str] = set()
                clean_header: list[str] = []
                for h in header:
                    ch, _ = _sanitize_key_or_header(h, detector, ref_lookup, keyring, used_headers, entity_counts)
                    used_headers.add(ch)
                    clean_header.append(ch)
                psn_col_idx: int | None = None
                for preferred in ("MRN", "PATIENT_NAME"):
                    if preferred in col_roles:
                        psn_col_idx = col_roles.index(preferred)
                        break

                h_buf = io.StringIO()
                csv.writer(h_buf, delimiter=inf.delimiter or ",", lineterminator="\n").writerow(clean_header)
                clues_by_loc = {rc.loc: rc for rc in f_clues}

                chunk_size = max(2000, (len(body) + n_workers - 1) // n_workers)
                chunk_tasks = []
                for start in range(0, len(body), chunk_size):
                    end = min(len(body), start + chunk_size)
                    sub_clues: dict[str, RecordClues] = {}
                    sub_u2p: dict[str, str | None] = {}
                    sub_prof: dict[str, Any] = {}
                    for r_i in range(start, end):
                        loc = f"row:{r_i}"
                        rc = clues_by_loc.get(loc)
                        if rc is not None:
                            sub_clues[loc] = rc
                            uid = rc.unit_id
                        else:
                            uid = f"{inf.rel_path}|{loc}"
                        if uid in unit_to_psn:
                            sub_u2p[uid] = unit_to_psn[uid]
                        if uid in unit_profiles:
                            sub_prof[uid] = unit_profiles[uid]
                    chunk_tasks.append((
                        inf.rel_path,
                        inf.delimiter or ",",
                        col_roles,
                        psn_col_idx,
                        body[start:end],
                        start,
                        date_hints.get(inf.rel_path, "dmy"),
                        modal_year,
                        sub_clues,
                        sub_u2p,
                        sub_prof,
                    ))

                text_parts = [h_buf.getvalue()]
                file_records: list[dict[str, Any]] = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
                    for c_text, c_recs, c_counts, c_gen in ex.map(_sanitize_csv_chunk_worker, chunk_tasks):
                        text_parts.append(c_text)
                        file_records.extend(c_recs)
                        for k, v in c_counts.items():
                            entity_counts[k] = entity_counts.get(k, 0) + v
                        if c_gen:
                            stats["records_generalized"] = stats.get("records_generalized", 0) + c_gen
                sanitized_text = "".join(text_parts)
            else:
                sanitized_text, file_records = sanitize_file(
                    inf=inf,
                    date_hint=date_hints.get(inf.rel_path, "dmy"),
                    modal_year=modal_year,
                    clues_list=f_clues,
                    unit_to_psn=unit_to_psn,
                    unit_profiles=unit_profiles,
                    ref_lookup=ref_lookup,
                    detector=detector,
                    keyring=keyring,
                    entity_counts=entity_counts,
                    stats=stats,
                )

            # Release raw parsed payload immediately after sanitization to bound memory
            inf.data = None
            inf.text = None

            clean_rel, was_renamed = sanitize_filename(
                inf.rel_path,
                file_sha256=inf.sha256,
                keyring=keyring,
                has_pii_fn=detector.has_pii_in_identifier,
                pseudonym=doc_psn,
            )
            # Ensure unique output relative path if two PII-named files collapsed
            while clean_rel in used_output_paths:
                p = Path(clean_rel)
                clean_rel = (p.parent / f"{p.stem}_{keyring.safe_tag(inf.sha256, 4)}{p.suffix}").as_posix()
            used_output_paths.add(clean_rel)

            if was_renamed:
                for rec in file_records:
                    rec["file"] = f"sanitized/{clean_rel}"

            dest = san_dir / clean_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w", encoding="utf-8", newline="") as f_out:
                f_out.write(sanitized_text)

            entry.update(output=f"sanitized/{clean_rel}", status="sanitized")
            record_entries.extend(file_records)

        except Exception as exc:
            entry.update(output=None, status="quarantined", reason=safe_reason(exc))

        file_entries.append(entry)

    if vault is not None:
        vault.purge()

    stats["identifiers_detected"] = sum(entity_counts.values())
    elapsed = time.monotonic() - t0
    write_outputs(
        output_dir=out_dir,
        file_entries=file_entries,
        record_entries=record_entries,
        entity_counts=entity_counts,
        stats=stats,
        elapsed_seconds=elapsed,
    )

    n_san = sum(1 for e in file_entries if e["status"] == "sanitized")
    # Strictly integer counts only on stdout (PII-free)
    print(f"files {len(file_entries)} sanitized {n_san} quarantined {len(file_entries) - n_san} records {len(record_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
