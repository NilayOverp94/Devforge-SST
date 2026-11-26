#!/usr/bin/env python3
"""Raksha Kavach: Offline, CPU-only, LLM-free privacy firewall for Indian clinical records."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from src.detection import PIIDetector
from src.ingestion import IngestedFile, ingest_directory
from src.linkage import RecordClues, extract_clues_from_file, infer_file_date_hints, link_all_records
from src.privacy import build_privacy_profiles, sanitize_file
from src.reporting import write_outputs
from src.security import EncryptedVault, KeyRing, safe_reason, sanitize_filename


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

    # Seal raw text snapshots inside AES-256-GCM memory vault and clear raw text until needed if desired
    vault_tokens: dict[str, str] = {}
    if vault is not None:
        for inf in ingested_files:
            if not inf.quarantine_reason and inf.text is not None:
                vault_tokens[inf.rel_path] = vault.seal(inf.text.encode("utf-8"))

    # Pass 2: Dataset-wide date-hint & hospital consensus inference
    date_hints, modal_year = infer_file_date_hints(ingested_files)

    # Pass 3: Clue Extraction
    all_clues: list[RecordClues] = []
    clues_by_file: dict[str, list[RecordClues]] = {}
    for inf in ingested_files:
        if inf.quarantine_reason:
            continue
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
            dest.write_text(sanitized_text, encoding="utf-8", newline="")

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
