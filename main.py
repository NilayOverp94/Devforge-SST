#!/usr/bin/env python3
"""Raksha Kavach: Offline, CPU-only, LLM-free privacy firewall for Indian clinical records."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src.detection import PIIDetector
from src.ingestion import (
    FileMeta,
    IngestedFile,
    QuarantineError,
    ingest_directory,
    iter_decoded_text_chunks,
    iter_large_json_array_batches,
    parse_file,
    read_large_json_header,
    scan_file_meta,
)
from src.linkage import (
    RecordClues,
    _RX_ICD_ONLY,
    detect_hospital_name,
    extract_clues_from_file,
    extract_csv_rows_clues,
    extract_json_items_clues,
    extract_text_chunk_into,
    infer_file_date_hints,
    infer_file_date_hints_from_samples,
    link_all_records,
    merge_record_clues_inplace,
    split_text_chunks,
)
from src.privacy import (
    PrivacyProfile,
    _clean_json_node,
    _sanitize_fhir,
    _sanitize_key_or_header,
    build_privacy_profiles,
    finalize_text_doc,
    preprocess_text_doc,
    redact_text_chunk,
    sanitize_csv_rows,
    sanitize_file,
    sanitize_json_list_chunk,
)
from src.reporting import write_outputs, write_report_json
from src.security import EncryptedVault, KeyRing, safe_reason, sanitize_filename

_WORKER_DETECTOR: PIIDetector | None = None


def _available_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _init_worker() -> None:
    global _WORKER_DETECTOR
    _WORKER_DETECTOR = PIIDetector()


def _scan_meta_worker(args: tuple[Path, Path, int, int]) -> FileMeta:
    path, inp_dir, max_json_depth, max_file_bytes = args
    return scan_file_meta(path, inp_dir, max_json_depth=max_json_depth, max_file_bytes=max_file_bytes)


def _extract_clues_worker(args: tuple[IngestedFile, str, int]) -> list[RecordClues]:
    inf, date_hint, modal_year = args
    assert _WORKER_DETECTOR is not None
    return extract_clues_from_file(inf, detector=_WORKER_DETECTOR, date_hint=date_hint, modal_year=modal_year)


def _extract_clues_from_path_worker(
    args: tuple[Path, Path, int, int, str, int],
) -> tuple[str, list[RecordClues], str | None]:
    path, inp_dir, max_json_depth, max_file_bytes, date_hint, modal_year = args
    assert _WORKER_DETECTOR is not None
    inf = parse_file(path, inp_dir, max_json_depth=max_json_depth, max_file_bytes=max_file_bytes)
    if inf.quarantine_reason:
        return inf.rel_path, [], inf.quarantine_reason
    clues = extract_clues_from_file(inf, detector=_WORKER_DETECTOR, date_hint=date_hint, modal_year=modal_year)
    for c in clues:
        c.compact()
    return inf.rel_path, clues, None


def _extract_csv_rows_clues_worker(
    args: tuple[str, str, list[str], list[str], list[list[str]], int, str, int, bool, str],
) -> list[RecordClues]:
    rel, fmt, header, col_roles, rows_chunk, start_row, date_hint, modal_year, is_pat, hosp = args
    return extract_csv_rows_clues(
        rel=rel,
        fmt=fmt,
        header=header,
        col_roles=col_roles,
        rows_chunk=rows_chunk,
        start_row_idx=start_row,
        date_hint=date_hint,
        modal_year=modal_year,
        is_patient_file=is_pat,
        hosp=hosp,
    )


def _extract_text_chunk_clues_worker(args: tuple[str, str, str, int, bool]) -> RecordClues:
    rel, chunk, date_hint, modal_year, is_pat = args
    assert _WORKER_DETECTOR is not None
    rc = RecordClues(
        unit_id=f"{rel}|doc",
        file_rel=rel,
        loc="doc",
        fmt="text",
        is_patient_record=is_pat,
        hospital=detect_hospital_name(chunk),
    )
    if is_pat:
        extract_text_chunk_into(chunk, rc, _WORKER_DETECTOR, date_hint, modal_year)
    return rc


def _extract_json_items_clues_worker(
    args: tuple[str, str, list[dict[str, Any]], int, str, int, bool, str],
) -> list[RecordClues]:
    rel, pkey, items, start_idx, date_hint, modal_year, is_pat, file_hosp = args
    assert _WORKER_DETECTOR is not None
    return extract_json_items_clues(
        rel=rel,
        pkey=pkey,
        items=items,
        start_idx=start_idx,
        detector=_WORKER_DETECTOR,
        date_hint=date_hint,
        modal_year=modal_year,
        is_patient_record=is_pat,
        file_hosp=file_hosp,
    )


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


def _sanitize_json_chunk_worker(
    args: tuple[
        str,
        str,
        str,
        list[dict[str, Any]],
        int,
        str,
        int,
        dict[str, RecordClues],
        dict[str, str | None],
        dict[str, Any],
        dict[str, str],
        KeyRing,
    ],
) -> tuple[list[Any], list[dict[str, Any]], dict[str, int], int]:
    (
        rel_path,
        root_key,
        pkey,
        items_chunk,
        start_idx,
        date_hint,
        modal_year,
        clues_by_loc,
        unit_to_psn,
        unit_profiles,
        ref_lookup,
        keyring,
    ) = args
    assert _WORKER_DETECTOR is not None
    return sanitize_json_list_chunk(
        rel_path=rel_path,
        root_key=root_key,
        pkey=pkey,
        items_chunk=items_chunk,
        start_idx=start_idx,
        date_hint=date_hint,
        modal_year=modal_year,
        clues_by_loc=clues_by_loc,
        unit_to_psn=unit_to_psn,
        unit_profiles=unit_profiles,
        ref_lookup=ref_lookup,
        detector=_WORKER_DETECTOR,
        keyring=keyring,
    )


def _redact_text_chunk_worker(args: tuple[str, list[str], list[str], str | None]) -> tuple[str, dict[str, int]]:
    chunk, extra_pat, extra_rel, psn = args
    assert _WORKER_DETECTOR is not None
    return redact_text_chunk(chunk, extra_pat, extra_rel, psn, _WORKER_DETECTOR)


def _extract_huge_file_worker(
    args: tuple[FileMeta, str, int, bool, int, int],
) -> tuple[str, list[RecordClues], str | None]:
    """Extract compacted RecordClues for a multi-MB CSV/TSV/TXT/JSON/FHIR file directly inside a worker."""
    m, d_hint, modal_year, can_be_pat, max_json_depth, max_file_bytes = args
    assert _WORKER_DETECTOR is not None
    try:
        if m.fmt in ("csv", "tsv"):
            f_clues = _stream_extract_csv_clues(m, d_hint, modal_year, _WORKER_DETECTOR, None, can_be_pat)
        elif m.fmt == "text":
            f_clues = _stream_extract_text_clues(m, d_hint, modal_year, _WORKER_DETECTOR, None, can_be_pat)
        elif m.fmt == "json" and m.json_array_key:
            f_clues = _stream_extract_large_json_clues(m, d_hint, modal_year, _WORKER_DETECTOR, None, can_be_pat)
        else:
            inf = parse_file(m.path, m.path.parents[len(Path(m.rel_path).parts) - 1], max_json_depth=max_json_depth, max_file_bytes=max_file_bytes)
            if inf.quarantine_reason:
                return m.rel_path, [], inf.quarantine_reason
            f_clues = extract_clues_from_file(inf, _WORKER_DETECTOR, d_hint, modal_year)
            for c in f_clues:
                c.compact()
        return m.rel_path, f_clues, None
    except QuarantineError as exc:
        return m.rel_path, [], str(exc)


def _sanitize_streaming_file_worker(
    args: tuple[
        int,
        FileMeta,
        Path,
        str,
        bool,
        Path,
        Path,
        list[RecordClues],
        str,
        int,
        dict[str, str | None],
        dict[str, PrivacyProfile],
        dict[str, str],
        KeyRing,
        int,
        int,
    ],
) -> tuple[int, int, int, dict[str, int], int, str | None]:
    """Sanitize a whole large/medium file inside a worker directly to atomic .part files on disk."""
    (
        entry_idx, m, inp_dir, san_file_label, was_renamed, tmp_dest, tmp_rec,
        f_clues, d_hint, modal_year, sub_u2p, sub_prof, sub_ref, keyring,
        max_json_depth, max_file_bytes,
    ) = args
    assert _WORKER_DETECTOR is not None
    detector = _WORKER_DETECTOR
    c_counts: dict[str, int] = {}
    c_stats: dict[str, int] = {"records_generalized": 0}
    try:
        n_recs = 0
        n_linked = 0
        if m.fmt in ("csv", "tsv") and m.raw_size > 8_000_000:
            n_recs, n_linked = _stream_sanitize_csv_file(
                m, san_file_label, tmp_dest, tmp_rec, f_clues,
                d_hint, modal_year, sub_u2p, sub_prof,
                sub_ref, detector, keyring, c_counts, c_stats, None, 1,
            )
        elif m.fmt == "text" and m.raw_size > 1_000_000:
            n_recs, n_linked = _stream_sanitize_text_file(
                m, san_file_label, tmp_dest, tmp_rec, f_clues,
                d_hint, modal_year, sub_u2p, sub_prof,
                detector, c_counts, c_stats, None,
            )
        elif m.fmt == "json" and m.raw_size > 32_000_000 and m.json_array_key:
            n_recs, n_linked = _stream_sanitize_json_array_file(
                m, san_file_label, tmp_dest, tmp_rec, f_clues,
                d_hint, modal_year, sub_u2p, sub_prof,
                sub_ref, detector, keyring, c_counts, c_stats, None,
            )
        else:
            inf = parse_file(m.path, inp_dir, max_json_depth=max_json_depth, max_file_bytes=max_file_bytes)
            if inf.quarantine_reason:
                raise QuarantineError(inf.quarantine_reason)
            if inf.fmt == "fhir":
                file_records: list[dict[str, Any]] = []
                clues_by_loc = {c.loc: c for c in f_clues}
                sanitized_text = _sanitize_fhir(
                    inf, clues_by_loc, sub_u2p, sub_prof,
                    detector, c_counts, c_stats, file_records, mutate_in_place=True,
                )
            else:
                sanitized_text, file_records = sanitize_file(
                    inf=inf, date_hint=d_hint, modal_year=modal_year, clues_list=f_clues,
                    unit_to_psn=sub_u2p, unit_profiles=sub_prof,
                    ref_lookup=sub_ref, detector=detector, keyring=keyring,
                    entity_counts=c_counts, stats=c_stats,
                )
            del inf
            with tmp_dest.open("w", encoding="utf-8", newline="") as f_out:
                f_out.write(sanitized_text)
            del sanitized_text
            with tmp_rec.open("w", encoding="utf-8", newline="\n") as rf:
                for rec in file_records:
                    if was_renamed:
                        rec["file"] = san_file_label
                    rf.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                    n_recs += 1
                    if rec.get("patient") is not None:
                        n_linked += 1
        return entry_idx, n_recs, n_linked, c_counts, c_stats.get("records_generalized", 0), None
    except Exception as exc:
        tmp_dest.unlink(missing_ok=True)
        tmp_rec.unlink(missing_ok=True)
        return entry_idx, 0, 0, {}, 0, safe_reason(exc)


def _intern_rc_strings(f_clues: list[RecordClues], cache: dict[str, str]) -> None:
    """Share common immutable string values across files/batches to eliminate duplicate heap copies."""
    def _si(s: str) -> str:
        if not s:
            return ""
        r = cache.get(s)
        if r is not None:
            return r
        if len(cache) < 32768:
            cache[s] = s
        return s

    for rc in f_clues:
        rc.hospital = _si(rc.hospital)
        rc.dob = _si(rc.dob)
        rc.gender = _si(rc.gender)
        rc.village = _si(rc.village)
        rc.district = _si(rc.district)
        rc.pin = _si(rc.pin)
        if type(rc._diagnoses) is str and rc._diagnoses:
            rc._diagnoses = _si(rc._diagnoses)
        if type(rc._mrns) is str and rc._mrns:
            rc._mrns = _si(rc._mrns)
        if type(rc._aadhaars) is str and rc._aadhaars:
            rc._aadhaars = _si(rc._aadhaars)
        if type(rc._abhas) is str and rc._abhas:
            rc._abhas = _si(rc._abhas)
        if type(rc._phones) is str and rc._phones:
            rc._phones = _si(rc._phones)
        if type(rc._patient_names) is str and rc._patient_names:
            rc._patient_names = _si(rc._patient_names)
        if type(rc._relative_names) is str and rc._relative_names:
            rc._relative_names = _si(rc._relative_names)


def _stream_text_chunk_worker(
    args: tuple[
        str,
        tuple[str, ...] | list[str],
        tuple[str, ...] | list[str],
        str | None,
        str,
        int,
        PrivacyProfile | None,
        str,
        bool,
    ],
) -> tuple[str, dict[str, int], bool]:
    """Preprocess, redact, and apply per-chunk district/PIN rules in a worker for multi-GB TXT streaming."""
    assert _WORKER_DETECTOR is not None
    return _stream_text_chunk_inproc(args, _WORKER_DETECTOR)


def load_policy(policy_path: Path) -> dict[str, Any]:
    if not policy_path.is_file():
        return {}
    loaded = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _json_chunk_needs_ref(items: list[dict[str, Any]], detector: PIIDetector) -> bool:
    """Only send full ref_lookup to JSON workers if any key in the batch actually looks like PII."""
    seen_keys: set[str] = set()
    for item in items[:50]:
        if isinstance(item, dict):
            for k in item.keys():
                ks = str(k)
                if ks not in seen_keys:
                    seen_keys.add(ks)
                    if detector.has_pii_in_identifier(ks):
                        return True
    return False


def _add_counts(dst: dict[str, int], delta: dict[str, int]) -> None:
    for k, v in delta.items():
        dst[k] = dst.get(k, 0) + v


def _merge_filename_clues(rel: str, rc: RecordClues, detector: PIIDetector) -> None:
    stem = Path(rel).stem
    for s in detector.detect_text(stem.replace("_", " "), allow_pin_scan=False):
        if s.pii_type == "PHONE" and s.normalized:
            rc.phones.add(s.normalized)
        elif s.pii_type == "MRN" and s.normalized:
            rc.mrns.add(re.sub(r"[^A-Z0-9]", "", s.normalized.upper()))
        elif s.pii_type == "PATIENT_NAME" and not rc.patient_names:
            rc.patient_names.append(s.surface)
    for m in re.finditer(r"(?i)\b(?:SNJ|JJH|UHID|DHH|IPD|IP|ADM)[-_]?(?:\d{2}[-_]|20\d{2}[-_])?\d{4,8}\b", stem):
        rc.mrns.add(re.sub(r"[^A-Z0-9]", "", m.group(0).upper()))


def _resolve_clean_output_path(
    rel_path: str,
    sha256: str,
    f_clues: list[RecordClues],
    unit_to_psn: dict[str, str | None],
    detector: PIIDetector,
    keyring: KeyRing,
    used_output_paths: set[str],
) -> tuple[str, bool, str | None]:
    doc_psn: str | None = unit_to_psn.get(f_clues[0].unit_id) if len(f_clues) == 1 else None
    clean_rel, was_renamed = sanitize_filename(
        rel_path,
        file_sha256=sha256,
        keyring=keyring,
        has_pii_fn=detector.has_pii_in_identifier,
        pseudonym=doc_psn,
    )
    while clean_rel in used_output_paths:
        p = Path(clean_rel)
        clean_rel = (p.parent / f"{p.stem}_{keyring.safe_tag(sha256, 4)}{p.suffix}").as_posix()
    used_output_paths.add(clean_rel)
    return clean_rel, was_renamed, doc_psn


def _write_json_item_indented(fh: io.TextIOBase, item: Any, indent_prefix: str = "    ") -> None:
    dumped = json.dumps(item, ensure_ascii=False, indent=2)
    fh.write("\n".join(indent_prefix + line for line in dumped.splitlines()))


def _run_in_memory(
    inp_dir: Path,
    out_dir: Path,
    san_dir: Path,
    policy: dict[str, Any],
    keyring: KeyRing,
    vault: EncryptedVault | None,
    detector: PIIDetector,
    t0: float,
) -> int:
    """Fast path for small and medium batches (<= 25 MB total, max file <= 10 MB)."""
    ingested_files: list[IngestedFile] = ingest_directory(
        inp_dir,
        max_json_depth=int(policy.get("max_json_depth", 200)),
        max_file_bytes=int(policy.get("max_file_bytes", 4294967296)),
    )
    date_hints, modal_year = infer_file_date_hints(ingested_files)

    valid_files = [inf for inf in ingested_files if not inf.quarantine_reason]
    total_valid_bytes = sum(inf.raw_size for inf in valid_files)
    n_workers = min(_available_cpus(), 6)
    has_big_single = any(
        (inf.fmt in ("csv", "tsv") and inf.data and len(inf.data) > 4001)
        or (inf.fmt == "text" and inf.text and len(inf.text) > 100_000)
        or (inf.fmt == "json" and inf.raw_size >= 400_000)
        for inf in valid_files
    )
    use_pool = n_workers > 1 and (total_valid_bytes >= 500_000 or has_big_single)
    pool: concurrent.futures.ProcessPoolExecutor | None = (
        concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker)
        if use_pool
        else None
    )

    try:
        all_clues: list[RecordClues] = []
        clues_by_file: dict[str, list[RecordClues]] = {}
        if pool is not None and len(valid_files) >= 2:
            tasks = [(inf, date_hints.get(inf.rel_path, "dmy"), modal_year) for inf in valid_files]
            for inf, f_clues in zip(valid_files, pool.map(_extract_clues_worker, tasks)):
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

        unit_to_psn, psn_clusters, ref_lookup = link_all_records(all_clues, keyring=keyring, policy=policy)
        _, unit_profiles = build_privacy_profiles(
            all_clues=all_clues,
            unit_to_psn=unit_to_psn,
            psn_clusters=psn_clusters,
            keyring=keyring,
            policy=policy,
        )
        del psn_clusters

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
                if pool is not None and inf.fmt in ("csv", "tsv") and inf.data and len(inf.data) > 4001:
                    sanitized_text, file_records = _parallel_sanitize_csv_mem(
                        inf, f_clues, n_workers, pool, date_hints, modal_year,
                        unit_to_psn, unit_profiles, ref_lookup, detector, keyring, entity_counts, stats,
                    )
                elif (
                    pool is not None
                    and inf.fmt == "json"
                    and isinstance(inf.data, dict)
                    and not inf.rel_path.startswith("clean/")
                    and inf.rel_path not in {"misc/helpdesk_tickets.json", "misc/scanned_consent_form.json", "misc/patient_index.json"}
                    and not ("patient" in inf.data and isinstance(inf.data["patient"], dict))
                    and any(
                        isinstance(v, list) and len(v) > 2000 and all(isinstance(x, dict) for x in v)
                        for k, v in inf.data.items()
                        if k.lower() in ("visits", "patients", "results", "records", "claims")
                    )
                ):
                    sanitized_text, file_records = _parallel_sanitize_json_mem(
                        inf, f_clues, n_workers, pool, date_hints, modal_year,
                        unit_to_psn, unit_profiles, ref_lookup, detector, keyring, entity_counts, stats,
                    )
                elif pool is not None and inf.fmt == "text" and inf.text and len(inf.text) > 100_000:
                    rc = f_clues[0] if f_clues else None
                    uid = rc.unit_id if rc else f"{inf.rel_path}|doc"
                    psn = unit_to_psn.get(uid)
                    prof = unit_profiles.get(uid)
                    if (not rc or not rc.is_patient_record) and not detector.detect_text(inf.text, allow_pin_scan=False):
                        sanitized_text = inf.text
                        file_records = [{"file": f"sanitized/{inf.rel_path}", "loc": "doc", "patient": None}]
                    else:
                        work, gen_any = preprocess_text_doc(
                            inf.text, date_hints.get(inf.rel_path, "dmy"), modal_year, prof, entity_counts,
                        )
                        extra_pat = list(rc.patient_names) if rc else []
                        extra_rel = list(rc.relative_names) if rc else []
                        chunks = split_text_chunks(work)
                        t_tasks = [
                            (ch, extra_pat, extra_rel, psn if idx == 0 else None)
                            for idx, ch in enumerate(chunks)
                        ]
                        cleaned_parts: list[str] = []
                        for c_out, c_counts in pool.map(_redact_text_chunk_worker, t_tasks):
                            cleaned_parts.append(c_out)
                            _add_counts(entity_counts, c_counts)
                        file_records = []
                        sanitized_text = finalize_text_doc(
                            "".join(cleaned_parts), inf.rel_path, rc, psn, prof, gen_any, stats, file_records,
                        )
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

                inf.data = None
                inf.text = None
                clean_rel, was_renamed, _ = _resolve_clean_output_path(
                    inf.rel_path, inf.sha256, f_clues, unit_to_psn, detector, keyring, used_output_paths,
                )
                if was_renamed:
                    san_prefix = f"sanitized/{clean_rel}"
                    for rec in file_records:
                        rec["file"] = san_prefix

                dest = san_dir / clean_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("w", encoding="utf-8", newline="") as f_out:
                    f_out.write(sanitized_text)

                entry.update(output=f"sanitized/{clean_rel}", status="sanitized")
                record_entries.extend(file_records)
            except Exception as exc:
                entry.update(output=None, status="quarantined", reason=safe_reason(exc))

            file_entries.append(entry)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

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
    print(f"files {len(file_entries)} sanitized {n_san} quarantined {len(file_entries) - n_san} records {len(record_entries)}")
    return 0


def _parallel_sanitize_csv_mem(
    inf: IngestedFile,
    f_clues: list[RecordClues],
    n_workers: int,
    pool: concurrent.futures.ProcessPoolExecutor,
    date_hints: dict[str, str],
    modal_year: int,
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    ref_lookup: dict[str, str],
    detector: PIIDetector,
    keyring: KeyRing,
    entity_counts: dict[str, int],
    stats: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
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
    d_hint = date_hints.get(inf.rel_path, "dmy")
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
            inf.rel_path, inf.delimiter or ",", col_roles, psn_col_idx,
            body[start:end], start, d_hint, modal_year, sub_clues, sub_u2p, sub_prof,
        ))

    text_parts = [h_buf.getvalue()]
    file_records: list[dict[str, Any]] = []
    for c_text, c_recs, c_counts, c_gen in pool.map(_sanitize_csv_chunk_worker, chunk_tasks):
        text_parts.append(c_text)
        file_records.extend(c_recs)
        _add_counts(entity_counts, c_counts)
        if c_gen:
            stats["records_generalized"] = stats.get("records_generalized", 0) + c_gen
    return "".join(text_parts), file_records


def _parallel_sanitize_json_mem(
    inf: IngestedFile,
    f_clues: list[RecordClues],
    n_workers: int,
    pool: concurrent.futures.ProcessPoolExecutor,
    date_hints: dict[str, str],
    modal_year: int,
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    ref_lookup: dict[str, str],
    detector: PIIDetector,
    keyring: KeyRing,
    entity_counts: dict[str, int],
    stats: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    clues_by_loc = {rc.loc: rc for rc in f_clues}
    used_top: set[str] = set()
    cleaned_root: dict[str, Any] = {}
    file_records: list[dict[str, Any]] = []
    str_cache: dict[tuple[str, bool], list[Any]] = {}
    d_hint = date_hints.get(inf.rel_path, "dmy")
    for k, v in inf.data.items():
        sk, _ = _sanitize_key_or_header(str(k), detector, ref_lookup, keyring, used_top, entity_counts)
        used_top.add(sk)
        if isinstance(v, list) and len(v) > 2000 and all(isinstance(x, dict) for x in v):
            pkey = k.replace("~", "~0").replace("/", "~1")
            chunk_size = max(500, (len(v) + n_workers - 1) // n_workers)
            j_tasks = []
            for start in range(0, len(v), chunk_size):
                end = min(len(v), start + chunk_size)
                items_slice = v[start:end]
                sub_clues = {}
                sub_u2p = {}
                sub_prof = {}
                for idx in range(start, end):
                    loc = f"/{pkey}/{idx}"
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
                chunk_ref = ref_lookup if _json_chunk_needs_ref(items_slice, detector) else {}
                j_tasks.append((
                    inf.rel_path, str(k), pkey, items_slice, start, d_hint, modal_year,
                    sub_clues, sub_u2p, sub_prof, chunk_ref, keyring,
                ))
            out_list: list[Any] = []
            for c_items, c_recs, c_counts, c_gen in pool.map(_sanitize_json_chunk_worker, j_tasks):
                out_list.extend(c_items)
                file_records.extend(c_recs)
                _add_counts(entity_counts, c_counts)
                if c_gen:
                    stats["records_generalized"] = stats.get("records_generalized", 0) + c_gen
            cleaned_root[sk] = out_list
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            pkey = k.replace("~", "~0").replace("/", "~1")
            out_list, c_recs, c_counts, c_gen = sanitize_json_list_chunk(
                inf.rel_path, str(k), pkey, v, 0, d_hint, modal_year,
                clues_by_loc, unit_to_psn, unit_profiles, ref_lookup, detector, keyring,
            )
            cleaned_root[sk] = out_list
            file_records.extend(c_recs)
            _add_counts(entity_counts, c_counts)
            if c_gen:
                stats["records_generalized"] = stats.get("records_generalized", 0) + c_gen
        else:
            cleaned_root[sk] = _clean_json_node(
                v, (str(k).lower(), ""), None, None, None, [False], [False],
                d_hint, modal_year, detector, ref_lookup, keyring, entity_counts, str_cache,
            )
    return json.dumps(cleaned_root, ensure_ascii=False, indent=2) + "\n", file_records


def _run_streaming(
    inp_dir: Path,
    out_dir: Path,
    san_dir: Path,
    policy: dict[str, Any],
    keyring: KeyRing,
    vault: EncryptedVault | None,
    detector: PIIDetector,
    paths: list[Path],
    total_bytes: int,
    t0: float,
) -> int:
    """Bounded-RAM, multi-core streaming pipeline for large and multi-GB workloads (up to 4 GB+ inside < 4 GB RSS)."""
    max_json_depth = int(policy.get("max_json_depth", 200))
    max_file_bytes = int(policy.get("max_file_bytes", 4294967296))
    n_workers = min(_available_cpus(), 8)

    pool: concurrent.futures.ProcessPoolExecutor | None = (
        concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker)
        if n_workers > 1
        else None
    )

    try:
        # Pass 1: Parallel metadata + encoding/structure validation (never keeps raw payloads across files)
        scan_tasks = [(p, inp_dir, max_json_depth, max_file_bytes) for p in paths]
        if pool is not None and len(paths) >= 2:
            file_metas: list[FileMeta] = list(pool.map(_scan_meta_worker, scan_tasks))
        else:
            file_metas = [scan_file_meta(*t) for t in scan_tasks]

        # Pass 2: Infer file date hints from tiny 64 KB samples
        date_samples = [(m.rel_path, m.sample_text) for m in file_metas if not m.quarantine_reason and m.sample_text]
        date_hints, modal_year = infer_file_date_hints_from_samples(date_samples)
        for m in file_metas:
            m.sample_text = ""

        # Pass 3: Parallel bounded-RAM clue extraction (whole file for small/medium, file-parallel or chunk-streamed for huge)
        clues_by_file: dict[str, list[RecordClues]] = {}
        all_clues: list[RecordClues] = []
        small_metas: list[FileMeta] = []
        huge_metas: list[FileMeta] = []

        for m in file_metas:
            if m.quarantine_reason:
                continue
            is_huge_csv = m.fmt in ("csv", "tsv") and m.raw_size > 8_000_000
            is_huge_txt = m.fmt == "text" and m.raw_size > 1_000_000
            is_huge_json = m.fmt == "json" and m.raw_size > 32_000_000 and bool(m.json_array_key)
            if not (is_huge_csv or is_huge_txt or is_huge_json):
                small_metas.append(m)
            else:
                huge_metas.append(m)

        parent_str_intern: dict[str, str] = {}

        if small_metas:
            if pool is not None and len(small_metas) >= 2:
                ext_tasks = [
                    (m.path, inp_dir, max_json_depth, max_file_bytes, date_hints.get(m.rel_path, "dmy"), modal_year)
                    for m in small_metas
                ]
                meta_by_rel = {m.rel_path: m for m in small_metas}
                for rel, f_clues, q_reason in pool.map(_extract_clues_from_path_worker, ext_tasks):
                    if q_reason:
                        meta_by_rel[rel].quarantine_reason = q_reason
                    else:
                        _intern_rc_strings(f_clues, parent_str_intern)
                        clues_by_file[rel] = f_clues
                        all_clues.extend(f_clues)
            else:
                for m in small_metas:
                    inf = parse_file(m.path, inp_dir, max_json_depth=max_json_depth, max_file_bytes=max_file_bytes)
                    if inf.quarantine_reason:
                        m.quarantine_reason = inf.quarantine_reason
                    else:
                        f_clues = extract_clues_from_file(inf, detector, date_hints.get(m.rel_path, "dmy"), modal_year)
                        for c in f_clues:
                            c.compact()
                        _intern_rc_strings(f_clues, parent_str_intern)
                        clues_by_file[m.rel_path] = f_clues
                        all_clues.extend(f_clues)
                    del inf

        _non_pat_misc = {
            "misc/ambulance_log.csv", "misc/helpdesk_tickets.json",
            "misc/scanned_consent_form.json", "misc/patient_index.json",
            "misc/ward_vitals_sheet.csv", "notes/visitor_log_ward5.txt",
        }

        if huge_metas:
            if pool is not None and len(huge_metas) >= 2:
                huge_tasks = [
                    (
                        m,
                        date_hints.get(m.rel_path, "dmy"),
                        modal_year,
                        not (m.rel_path.startswith("clean/") or m.rel_path == "misc/README_export.txt" or (m.fmt != "json" and m.rel_path in _non_pat_misc)),
                        max_json_depth,
                        max_file_bytes,
                    )
                    for m in huge_metas
                ]
                meta_by_rel_h = {m.rel_path: m for m in huge_metas}
                for rel, f_clues, q_reason in pool.map(_extract_huge_file_worker, huge_tasks, chunksize=1):
                    if q_reason:
                        meta_by_rel_h[rel].quarantine_reason = q_reason
                    else:
                        _intern_rc_strings(f_clues, parent_str_intern)
                        clues_by_file[rel] = f_clues
                        all_clues.extend(f_clues)
            else:
                for m in huge_metas:
                    rel = m.rel_path
                    d_hint = date_hints.get(rel, "dmy")
                    is_clean_ref = rel.startswith("clean/") or rel == "misc/README_export.txt"
                    is_non_patient_misc = rel in _non_pat_misc
                    try:
                        if m.fmt in ("csv", "tsv"):
                            f_clues = _stream_extract_csv_clues(
                                m, d_hint, modal_year, detector, pool, not is_clean_ref and not is_non_patient_misc,
                            )
                        elif m.fmt == "text":
                            f_clues = _stream_extract_text_clues(
                                m, d_hint, modal_year, detector, pool, not is_clean_ref and not is_non_patient_misc,
                            )
                        elif m.fmt == "json" and m.json_array_key:
                            f_clues = _stream_extract_large_json_clues(
                                m, d_hint, modal_year, detector, pool, not is_clean_ref,
                            )
                        else:
                            inf = parse_file(m.path, inp_dir, max_json_depth=max_json_depth, max_file_bytes=max_file_bytes)
                            if inf.quarantine_reason:
                                m.quarantine_reason = inf.quarantine_reason
                                continue
                            f_clues = extract_clues_from_file(inf, detector, d_hint, modal_year)
                            for c in f_clues:
                                c.compact()
                            del inf
                        _intern_rc_strings(f_clues, parent_str_intern)
                        clues_by_file[rel] = f_clues
                        all_clues.extend(f_clues)
                    except QuarantineError as exc:
                        m.quarantine_reason = str(exc)
        del parent_str_intern

        # Pass 4 & 5: Link records + interned privacy profiles
        unit_to_psn, psn_clusters, ref_lookup = link_all_records(all_clues, keyring=keyring, policy=policy)
        _, unit_profiles = build_privacy_profiles(
            all_clues=all_clues,
            unit_to_psn=unit_to_psn,
            psn_clusters=psn_clusters,
            keyring=keyring,
            policy=policy,
        )
        del psn_clusters
        del all_clues

        # Pass 6: Atomic streaming sanitization -> direct sanitized/ + streamed records.jsonl
        file_entries: list[dict[str, Any]] = [{"input_sha256": m.sha256} for m in file_metas]
        entity_counts: dict[str, int] = {}
        stats: dict[str, int] = {"records_generalized": 0, "records_suppressed": 0, "records_quarantined": 0}
        used_output_paths: set[str] = set()
        records_processed = 0
        records_linked = 0

        # Pre-resolve deterministic output filenames cleanly before dispatching parallel file workers
        resolved_slots: list[tuple[int, FileMeta, str, bool, Path, Path, Path] | None] = [None] * len(file_metas)
        big_indices: list[int] = []
        for idx, m in enumerate(file_metas):
            if m.quarantine_reason:
                file_entries[idx].update(output=None, status="quarantined", reason=m.quarantine_reason)
                clues_by_file.pop(m.rel_path, None)
                continue
            f_clues_meta = clues_by_file.get(m.rel_path, [])
            clean_rel, was_renamed, _ = _resolve_clean_output_path(
                m.rel_path, m.sha256, f_clues_meta, unit_to_psn, detector, keyring, used_output_paths,
            )
            dest = san_dir / clean_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp_dest = dest.with_name(dest.name + ".part")
            tmp_rec = out_dir / f".rec_{idx}.part"
            san_file_label = f"sanitized/{clean_rel}"
            resolved_slots[idx] = (idx, m, san_file_label, was_renamed, dest, tmp_dest, tmp_rec)
            if m.raw_size >= 250_000 or m.fmt == "fhir":
                big_indices.append(idx)

        parallel_san_indices = set(big_indices) if (pool is not None and len(big_indices) >= 2) else set()
        worker_results: dict[int, tuple[int, int, dict[str, int], int, str | None]] = {}

        if parallel_san_indices:
            assert pool is not None
            ordered_p = sorted(parallel_san_indices)
            in_flight: dict[concurrent.futures.Future[Any], int] = {}
            next_pos = 0
            max_queue = n_workers * 2

            def _submit_next() -> bool:
                nonlocal next_pos
                if next_pos >= len(ordered_p):
                    return False
                e_idx = ordered_p[next_pos]
                next_pos += 1
                slot = resolved_slots[e_idx]
                assert slot is not None
                _, m_p, san_lbl, was_ren, _dst, tmp_d, tmp_r = slot
                f_c = clues_by_file.pop(m_p.rel_path, [])
                sub_u2p: dict[str, str | None] = {}
                sub_prof: dict[str, PrivacyProfile] = {}
                for rc in f_c:
                    uid = rc.unit_id
                    if uid in unit_to_psn:
                        sub_u2p[uid] = unit_to_psn.pop(uid)
                    if uid in unit_profiles:
                        sub_prof[uid] = unit_profiles.pop(uid)
                sub_ref = ref_lookup if (m_p.fmt == "json") else {}
                task = (
                    e_idx, m_p, inp_dir, san_lbl, was_ren, tmp_d, tmp_r,
                    f_c, date_hints.get(m_p.rel_path, "dmy"), modal_year,
                    sub_u2p, sub_prof, sub_ref, keyring, max_json_depth, max_file_bytes,
                )
                fut = pool.submit(_sanitize_streaming_file_worker, task)
                in_flight[fut] = e_idx
                return True

            while len(in_flight) < max_queue and _submit_next():
                pass
            while in_flight:
                done_futs, _ = concurrent.futures.wait(in_flight.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                for df in done_futs:
                    in_flight.pop(df)
                    r_idx, n_r, n_l, c_cnt, c_g, err = df.result()
                    worker_results[r_idx] = (n_r, n_l, c_cnt, c_g, err)
                    _submit_next()

        records_path = out_dir / "records.jsonl"
        with records_path.open("w", encoding="utf-8", newline="\n") as rec_fh:
            for idx, m in enumerate(file_metas):
                slot = resolved_slots[idx]
                if slot is None:
                    continue
                _, _, san_file_label, was_renamed, dest, tmp_dest, tmp_rec = slot
                entry = file_entries[idx]

                if idx in worker_results:
                    n_recs_file, n_linked_file, c_cnt, c_g, err = worker_results.pop(idx)
                    if err is not None:
                        tmp_dest.unlink(missing_ok=True)
                        tmp_rec.unlink(missing_ok=True)
                        entry.update(output=None, status="quarantined", reason=err)
                    else:
                        _add_counts(entity_counts, c_cnt)
                        if c_g:
                            stats["records_generalized"] = stats.get("records_generalized", 0) + c_g
                        os.replace(tmp_dest, dest)
                        if tmp_rec.is_file():
                            with tmp_rec.open("r", encoding="utf-8") as rf_in:
                                while True:
                                    chunk = rf_in.read(4_194_304)
                                    if not chunk:
                                        break
                                    rec_fh.write(chunk)
                            tmp_rec.unlink(missing_ok=True)
                        records_processed += n_recs_file
                        records_linked += n_linked_file
                        entry.update(output=san_file_label, status="sanitized")
                    continue

                f_clues = clues_by_file.pop(m.rel_path, [])
                d_hint = date_hints.get(m.rel_path, "dmy")
                try:
                    n_recs_file = 0
                    n_linked_file = 0
                    if m.fmt in ("csv", "tsv") and m.raw_size > 8_000_000:
                        n_recs_file, n_linked_file = _stream_sanitize_csv_file(
                            m, san_file_label, tmp_dest, tmp_rec, f_clues,
                            d_hint, modal_year, unit_to_psn, unit_profiles,
                            ref_lookup, detector, keyring, entity_counts, stats, pool, n_workers,
                        )
                    elif m.fmt == "text" and m.raw_size > 1_000_000:
                        n_recs_file, n_linked_file = _stream_sanitize_text_file(
                            m, san_file_label, tmp_dest, tmp_rec, f_clues,
                            d_hint, modal_year, unit_to_psn, unit_profiles,
                            detector, entity_counts, stats, pool,
                        )
                    elif m.fmt == "json" and m.raw_size > 32_000_000 and m.json_array_key:
                        n_recs_file, n_linked_file = _stream_sanitize_json_array_file(
                            m, san_file_label, tmp_dest, tmp_rec, f_clues,
                            d_hint, modal_year, unit_to_psn, unit_profiles,
                            ref_lookup, detector, keyring, entity_counts, stats, pool,
                        )
                    else:
                        inf = parse_file(m.path, inp_dir, max_json_depth=max_json_depth, max_file_bytes=max_file_bytes)
                        if inf.quarantine_reason:
                            raise QuarantineError(inf.quarantine_reason)
                        if inf.fmt == "fhir":
                            file_records: list[dict[str, Any]] = []
                            clues_by_loc = {c.loc: c for c in f_clues}
                            sanitized_text = _sanitize_fhir(
                                inf, clues_by_loc, unit_to_psn, unit_profiles,
                                detector, entity_counts, stats, file_records, mutate_in_place=True,
                            )
                        elif pool is not None and inf.fmt == "json" and isinstance(inf.data, dict) and inf.raw_size >= 400_000 and any(
                            isinstance(v, list) and len(v) > 2000 and all(isinstance(x, dict) for x in v)
                            for k, v in inf.data.items()
                            if k.lower() in ("visits", "patients", "results", "records", "claims")
                        ):
                            sanitized_text, file_records = _parallel_sanitize_json_mem(
                                inf, f_clues, n_workers, pool, date_hints, modal_year,
                                unit_to_psn, unit_profiles, ref_lookup, detector, keyring, entity_counts, stats,
                            )
                        else:
                            sanitized_text, file_records = sanitize_file(
                                inf=inf, date_hint=d_hint, modal_year=modal_year, clues_list=f_clues,
                                unit_to_psn=unit_to_psn, unit_profiles=unit_profiles,
                                ref_lookup=ref_lookup, detector=detector, keyring=keyring,
                                entity_counts=entity_counts, stats=stats,
                            )
                        del inf
                        with tmp_dest.open("w", encoding="utf-8", newline="") as f_out:
                            f_out.write(sanitized_text)
                        del sanitized_text
                        with tmp_rec.open("w", encoding="utf-8", newline="\n") as rf:
                            for rec in file_records:
                                if was_renamed:
                                    rec["file"] = san_file_label
                                rf.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                                n_recs_file += 1
                                if rec.get("patient") is not None:
                                    n_linked_file += 1
                        del file_records

                    os.replace(tmp_dest, dest)
                    if tmp_rec.is_file():
                        with tmp_rec.open("r", encoding="utf-8") as rf_in:
                            while True:
                                chunk = rf_in.read(4_194_304)
                                if not chunk:
                                    break
                                rec_fh.write(chunk)
                        tmp_rec.unlink(missing_ok=True)
                    records_processed += n_recs_file
                    records_linked += n_linked_file
                    entry.update(output=san_file_label, status="sanitized")
                except Exception as exc:
                    tmp_dest.unlink(missing_ok=True)
                    tmp_rec.unlink(missing_ok=True)
                    entry.update(output=None, status="quarantined", reason=safe_reason(exc))

                for rc in f_clues:
                    unit_to_psn.pop(rc.unit_id, None)
                    unit_profiles.pop(rc.unit_id, None)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if vault is not None:
        vault.purge()

    stats["identifiers_detected"] = sum(entity_counts.values())
    elapsed = time.monotonic() - t0
    write_report_json(
        output_dir=out_dir,
        file_entries=file_entries,
        records_processed=records_processed,
        records_linked=records_linked,
        entity_counts=entity_counts,
        stats=stats,
        elapsed_seconds=elapsed,
    )
    n_san = sum(1 for e in file_entries if e["status"] == "sanitized")
    print(f"files {len(file_entries)} sanitized {n_san} quarantined {len(file_entries) - n_san} records {records_processed}")
    return 0


def _stream_extract_csv_clues(
    m: FileMeta,
    d_hint: str,
    modal_year: int,
    detector: PIIDetector,
    pool: concurrent.futures.ProcessPoolExecutor | None,
    can_be_patient: bool,
) -> list[RecordClues]:
    delim = m.delimiter or ","
    hosp = detect_hospital_name(m.rel_path)
    out_clues: list[RecordClues] = []
    with m.path.open("r", encoding=m.encoding, newline="") as fh:
        reader = csv.reader(fh, delimiter=delim)
        header = next(reader)
        sample_rows: list[list[str]] = []
        for _ in range(25):
            r = next(reader, None)
            if r is None:
                break
            sample_rows.append(r)
        col_roles = [
            detector.classify_column_header(h, [r[i] for r in sample_rows if i < len(r)])
            for i, h in enumerate(header)
        ]
        is_pat = can_be_patient and any(r in ("PATIENT_NAME", "MRN", "AADHAAR", "ABHA", "PHONE") for r in col_roles)

        def _row_batches() -> Any:
            batch = list(sample_rows)
            start = 0
            batch_limit = 20_000
            for r in reader:
                batch.append(r)
                if len(batch) >= batch_limit:
                    yield (m.rel_path, m.fmt, header, col_roles, batch, start, d_hint, modal_year, is_pat, hosp)
                    start += len(batch)
                    batch = []
            if batch:
                yield (m.rel_path, m.fmt, header, col_roles, batch, start, d_hint, modal_year, is_pat, hosp)

        if pool is not None:
            for c_clues in pool.map(_extract_csv_rows_clues_worker, _row_batches(), chunksize=1):
                out_clues.extend(c_clues)
        else:
            for task in _row_batches():
                out_clues.extend(_extract_csv_rows_clues_worker(task))
    return out_clues


def _stream_extract_text_clues(
    m: FileMeta,
    d_hint: str,
    modal_year: int,
    detector: PIIDetector,
    pool: concurrent.futures.ProcessPoolExecutor | None,
    is_pat: bool,
) -> list[RecordClues]:
    rc = RecordClues(
        unit_id=f"{m.rel_path}|doc",
        file_rel=m.rel_path,
        loc="doc",
        fmt="text",
        is_patient_record=is_pat,
        hospital=detect_hospital_name(m.rel_path),
    )
    if is_pat:
        chunk_iter = iter_decoded_text_chunks(m.path, m.encoding, target_chars=262_144)
        first_chunk = next(chunk_iter, None)
        if first_chunk is not None:
            if not rc.hospital:
                rc.hospital = detect_hospital_name(first_chunk[:8192])
            extract_text_chunk_into(first_chunk, rc, detector, d_hint, modal_year)
            if len(rc.mrns) > 10 or len(rc.patient_names) > 20:
                # Multi-patient dump: preserve all diagnosis/risk clues via fast ICD scan without linking thousands into 1 PSN
                rc.is_patient_record = False
                rc._patient_names = list(dict.fromkeys(rc.patient_names))[:20]
                rc._relative_names = list(dict.fromkeys(rc.relative_names))[:20]
                for ch in chunk_iter:
                    for m_icd in _RX_ICD_ONLY.finditer(ch):
                        rc._diagnoses.add(m_icd.group(1).upper())
            else:
                tasks = ((m.rel_path, ch, d_hint, modal_year, True) for ch in chunk_iter)
                if pool is not None:
                    for part_rc in pool.map(_extract_text_chunk_clues_worker, tasks, chunksize=1):
                        merge_record_clues_inplace(rc, part_rc)
                        if len(rc.mrns) > 10 or len(rc.patient_names) > 20:
                            rc.is_patient_record = False
                else:
                    for task in tasks:
                        if rc.is_patient_record:
                            extract_text_chunk_into(task[1], rc, detector, d_hint, modal_year)
                            if len(rc.mrns) > 10 or len(rc.patient_names) > 20:
                                rc.is_patient_record = False
                        else:
                            for m_icd in _RX_ICD_ONLY.finditer(task[1]):
                                rc._diagnoses.add(m_icd.group(1).upper())
        _merge_filename_clues(m.rel_path, rc, detector)
    rc.compact()
    return [rc]


def _stream_extract_large_json_clues(
    m: FileMeta,
    d_hint: str,
    modal_year: int,
    detector: PIIDetector,
    pool: concurrent.futures.ProcessPoolExecutor | None,
    is_pat: bool,
) -> list[RecordClues]:
    arr_key = m.json_array_key
    assert arr_key is not None
    header_meta, char_off = read_large_json_header(m.path, m.encoding, arr_key)
    hosp = ""
    if isinstance(header_meta.get("_export_meta"), dict):
        hosp = detect_hospital_name(str(header_meta["_export_meta"].get("hospital", "")))
    elif isinstance(header_meta.get("facility"), dict):
        hosp = detect_hospital_name(str(header_meta["facility"].get("name", "")))
    file_hosp = hosp or detect_hospital_name(m.rel_path)
    pkey = arr_key.replace("~", "~0").replace("/", "~1")

    out_clues: list[RecordClues] = []
    start = 0

    def _item_batches() -> Any:
        nonlocal start
        for batch in iter_large_json_array_batches(m.path, m.encoding, char_off, batch_size=4000):
            cur = start
            start += len(batch)
            yield (m.rel_path, pkey, batch, cur, d_hint, modal_year, is_pat, file_hosp)

    if pool is not None:
        for c_clues in pool.map(_extract_json_items_clues_worker, _item_batches(), chunksize=1):
            out_clues.extend(c_clues)
    else:
        for task in _item_batches():
            out_clues.extend(_extract_json_items_clues_worker(task))
    return out_clues


def _stream_sanitize_csv_file(
    m: FileMeta,
    san_file_label: str,
    tmp_dest: Path,
    tmp_rec: Path,
    f_clues: list[RecordClues],
    d_hint: str,
    modal_year: int,
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    ref_lookup: dict[str, str],
    detector: PIIDetector,
    keyring: KeyRing,
    entity_counts: dict[str, int],
    stats: dict[str, int],
    pool: concurrent.futures.ProcessPoolExecutor | None,
    n_workers: int,
) -> tuple[int, int]:
    delim = m.delimiter or ","
    n_recs = 0
    n_linked = 0
    clues_by_loc = {rc.loc: rc for rc in f_clues}

    with m.path.open("r", encoding=m.encoding, newline="") as fh_in:
        reader = csv.reader(fh_in, delimiter=delim)
        header = next(reader)
        sample_rows: list[list[str]] = []
        for _ in range(25):
            r = next(reader, None)
            if r is None:
                break
            sample_rows.append(r)

        col_roles = [
            detector.classify_column_header(h, [r[i] for r in sample_rows if i < len(r)])
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

        def _san_batches() -> Any:
            batch = list(sample_rows)
            start = 0
            batch_limit = 20_000
            for r in reader:
                batch.append(r)
                if len(batch) >= batch_limit:
                    yield _build_csv_san_task(
                        m.rel_path, delim, col_roles, psn_col_idx, batch, start,
                        d_hint, modal_year, clues_by_loc, unit_to_psn, unit_profiles,
                    )
                    start += len(batch)
                    batch = []
            if batch:
                yield _build_csv_san_task(
                    m.rel_path, delim, col_roles, psn_col_idx, batch, start,
                    d_hint, modal_year, clues_by_loc, unit_to_psn, unit_profiles,
                )

        with tmp_dest.open("w", encoding="utf-8", newline="") as f_out, tmp_rec.open("w", encoding="utf-8", newline="\n") as rf:
            csv.writer(f_out, delimiter=delim, lineterminator="\n").writerow(clean_header)
            if pool is not None:
                res_iter = pool.map(_sanitize_csv_chunk_worker, _san_batches(), chunksize=1)
            else:
                res_iter = (
                    sanitize_csv_rows(*t, detector=detector)
                    for t in _san_batches()
                )
            for c_text, c_recs, c_counts, c_gen in res_iter:
                f_out.write(c_text)
                _add_counts(entity_counts, c_counts)
                if c_gen:
                    stats["records_generalized"] = stats.get("records_generalized", 0) + c_gen
                for rec in c_recs:
                    rec["file"] = san_file_label
                    rf.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                    n_recs += 1
                    if rec.get("patient") is not None:
                        n_linked += 1
    return n_recs, n_linked


def _build_csv_san_task(
    rel_path: str,
    delim: str,
    col_roles: list[str],
    psn_col_idx: int | None,
    rows_chunk: list[list[str]],
    start: int,
    d_hint: str,
    modal_year: int,
    clues_by_loc: dict[str, RecordClues],
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
) -> tuple[Any, ...]:
    sub_clues: dict[str, RecordClues] = {}
    sub_u2p: dict[str, str | None] = {}
    sub_prof: dict[str, Any] = {}
    for r_i in range(start, start + len(rows_chunk)):
        loc = f"row:{r_i}"
        rc = clues_by_loc.get(loc)
        if rc is not None:
            sub_clues[loc] = rc
            uid = rc.unit_id
        else:
            uid = f"{rel_path}|{loc}"
        if uid in unit_to_psn:
            sub_u2p[uid] = unit_to_psn[uid]
        if uid in unit_profiles:
            sub_prof[uid] = unit_profiles[uid]
    return (
        rel_path, delim, col_roles, psn_col_idx, rows_chunk, start,
        d_hint, modal_year, sub_clues, sub_u2p, sub_prof,
    )


def _stream_sanitize_text_file(
    m: FileMeta,
    san_file_label: str,
    tmp_dest: Path,
    tmp_rec: Path,
    f_clues: list[RecordClues],
    d_hint: str,
    modal_year: int,
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    detector: PIIDetector,
    entity_counts: dict[str, int],
    stats: dict[str, int],
    pool: concurrent.futures.ProcessPoolExecutor | None,
) -> tuple[int, int]:
    rc = f_clues[0] if f_clues else None
    uid = rc.unit_id if rc else f"{m.rel_path}|doc"
    psn = unit_to_psn.get(uid)
    prof = unit_profiles.get(uid)
    extra_pat = rc.patient_names if rc else ()
    extra_rel = rc.relative_names if rc else ()
    rc_district = rc.district if rc else ""
    local_names = len(extra_pat) > 200 or len(extra_rel) > 200
    task_pat: tuple[str, ...] = () if local_names else tuple(extra_pat)
    task_rel: tuple[str, ...] = () if local_names else tuple(extra_rel)

    is_clean_only = (not rc or not rc.is_patient_record)
    gen_any_doc = bool(prof and prof.generalize_district and rc)

    def _tasks() -> Any:
        for idx, ch in enumerate(iter_decoded_text_chunks(m.path, m.encoding, target_chars=262_144)):
            yield (ch, task_pat, task_rel, psn if idx == 0 else None, d_hint, modal_year, prof, rc_district, local_names)

    with tmp_dest.open("w", encoding="utf-8", newline="") as f_out:
        psn_emitted = False
        if pool is not None:
            res_iter = pool.map(_stream_text_chunk_worker, _tasks(), chunksize=1)
        else:
            res_iter = (
                _stream_text_chunk_inproc(t, detector)
                for t in _tasks()
            )
        for idx, (c_out, c_counts, c_gen) in enumerate(res_iter):
            _add_counts(entity_counts, c_counts)
            if c_gen:
                gen_any_doc = True
            if idx == 0 and psn and (not is_clean_only or c_counts):
                if psn in c_out:
                    psn_emitted = True
                elif not psn_emitted:
                    c_out = f"Patient: {psn}\n" + c_out
                    psn_emitted = True
            f_out.write(c_out)

    if gen_any_doc:
        stats["records_generalized"] = stats.get("records_generalized", 0) + 1
    rec = {
        "file": san_file_label,
        "loc": "doc",
        "patient": psn,
        **({"status": "generalized"} if gen_any_doc else {}),
    }
    tmp_rec.write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return 1, (1 if psn is not None else 0)


def _stream_text_chunk_inproc(
    args: tuple[str, Any, Any, str | None, str, int, PrivacyProfile | None, str, bool],
    detector: PIIDetector,
) -> tuple[str, dict[str, int], bool]:
    chunk, extra_pat, extra_rel, psn, date_hint, modal_year, prof, rc_district, local_names = args
    c_counts: dict[str, int] = {}
    work, gen_any = preprocess_text_doc(chunk, date_hint, modal_year, prof, c_counts)
    cleaned, r_counts = redact_text_chunk(
        work, extra_pat, extra_rel, psn, detector, local_propagate=local_names,
    )
    for k, v in r_counts.items():
        c_counts[k] = c_counts.get(k, 0) + v
    if prof and prof.generalize_district:
        if rc_district and rc_district.lower() in cleaned.lower():
            cleaned = re.sub(re.escape(rc_district), prof.state_label, cleaned, flags=re.I)
        cleaned = re.sub(
            r"(?im)^([^\n]*(?:Hospital|Sadan)[^\n]*,\s*)([A-Za-z ]{3,25})$",
            rf"\g<1>{prof.state_label}",
            cleaned,
        )
        cleaned = re.sub(r"([1-8]\d{2})\*{3}", "******", cleaned)
        gen_any = True
    return cleaned, c_counts, gen_any


def _stream_sanitize_json_array_file(
    m: FileMeta,
    san_file_label: str,
    tmp_dest: Path,
    tmp_rec: Path,
    f_clues: list[RecordClues],
    d_hint: str,
    modal_year: int,
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    ref_lookup: dict[str, str],
    detector: PIIDetector,
    keyring: KeyRing,
    entity_counts: dict[str, int],
    stats: dict[str, int],
    pool: concurrent.futures.ProcessPoolExecutor | None,
) -> tuple[int, int]:
    """Stream sanitize multi-hundred-MB JSON arrays directly to disk without building a 2 GB string in RAM."""
    arr_key = m.json_array_key
    assert arr_key is not None
    header_meta, char_off = read_large_json_header(m.path, m.encoding, arr_key)
    pkey = arr_key.replace("~", "~0").replace("/", "~1")
    clues_by_loc = {rc.loc: rc for rc in f_clues}

    used_top: set[str] = set()
    str_cache: dict[tuple[str, bool], list[Any]] = {}
    cleaned_prefix_entries: list[tuple[str, Any]] = []
    for k, v in header_meta.items():
        sk, _ = _sanitize_key_or_header(str(k), detector, ref_lookup, keyring, used_top, entity_counts)
        used_top.add(sk)
        cv = _clean_json_node(
            v, (str(k).lower(), ""), None, None, None, [False], [False],
            d_hint, modal_year, detector, ref_lookup, keyring, entity_counts, str_cache,
        )
        cleaned_prefix_entries.append((sk, cv))

    sk_arr, _ = _sanitize_key_or_header(arr_key, detector, ref_lookup, keyring, used_top, entity_counts)
    used_top.add(sk_arr)

    def _san_batches() -> Any:
        start = 0
        for batch in iter_large_json_array_batches(m.path, m.encoding, char_off, batch_size=2000):
            end = start + len(batch)
            sub_clues: dict[str, RecordClues] = {}
            sub_u2p: dict[str, str | None] = {}
            sub_prof: dict[str, Any] = {}
            for idx in range(start, end):
                loc = f"/{pkey}/{idx}"
                rc = clues_by_loc.get(loc)
                if rc is not None:
                    sub_clues[loc] = rc
                    uid = rc.unit_id
                else:
                    uid = f"{m.rel_path}|{loc}"
                if uid in unit_to_psn:
                    sub_u2p[uid] = unit_to_psn[uid]
                if uid in unit_profiles:
                    sub_prof[uid] = unit_profiles[uid]
            chunk_ref = ref_lookup if _json_chunk_needs_ref(batch, detector) else {}
            yield (
                m.rel_path, arr_key, pkey, batch, start, d_hint, modal_year,
                sub_clues, sub_u2p, sub_prof, chunk_ref, keyring,
            )
            start = end

    n_recs = 0
    n_linked = 0
    with tmp_dest.open("w", encoding="utf-8", newline="") as f_out, tmp_rec.open("w", encoding="utf-8", newline="\n") as rf:
        f_out.write("{\n")
        for sk, cv in cleaned_prefix_entries:
            val_dump = json.dumps(cv, ensure_ascii=False, indent=2)
            lines = val_dump.splitlines()
            f_out.write(f"  {json.dumps(sk, ensure_ascii=False)}: {lines[0]}")
            for line in lines[1:]:
                f_out.write("\n  " + line)
            f_out.write(",\n")
        f_out.write(f"  {json.dumps(sk_arr, ensure_ascii=False)}: [\n")

        if pool is not None:
            res_iter = pool.map(_sanitize_json_chunk_worker, _san_batches(), chunksize=1)
        else:
            res_iter = (
                sanitize_json_list_chunk(*t[:-1], detector=detector, keyring=t[-1])
                for t in _san_batches()
            )
        first_item = True
        for c_items, c_recs, c_counts, c_gen in res_iter:
            _add_counts(entity_counts, c_counts)
            if c_gen:
                stats["records_generalized"] = stats.get("records_generalized", 0) + c_gen
            for item in c_items:
                if not first_item:
                    f_out.write(",\n")
                first_item = False
                _write_json_item_indented(f_out, item, "    ")
            for rec in c_recs:
                rec["file"] = san_file_label
                rf.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                n_recs += 1
                if rec.get("patient") is not None:
                    n_linked += 1
        f_out.write("\n  ]\n}\n")
    return n_recs, n_linked


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

    paths = sorted((p for p in inp_dir.rglob("*") if p.is_file()), key=lambda p: p.relative_to(inp_dir).as_posix())
    raw_sizes = [p.stat().st_size for p in paths]
    total_bytes = sum(raw_sizes)
    max_size = max(raw_sizes, default=0)

    if total_bytes <= 25_000_000 and max_size <= 10_000_000:
        return _run_in_memory(inp_dir, out_dir, san_dir, policy, keyring, vault, detector, t0)
    return _run_streaming(inp_dir, out_dir, san_dir, policy, keyring, vault, detector, paths, total_bytes, t0)


if __name__ == "__main__":
    raise SystemExit(main())
