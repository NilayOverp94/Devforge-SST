"""Privacy layer: per-patient date shifting, adaptive k-anonymous age banding, location generalization, and clean serialization."""
from __future__ import annotations

import copy
import csv
import datetime as dt
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .detection import (
    PIIDetector,
    canonicalize_person_name,
    canonicalize_place,
    encode_base64_text,
    shadow_normalize,
    to_ascii_digits,
    try_decode_base64_text,
)
from .ingestion import IngestedFile, validate_fhir_bundle
from .linkage import (
    PatientCluster,
    RecordClues,
    _RX_DATE_MONTH,
    _RX_DATE_NUM,
    norm_mrn_key,
    parse_age_val,
    parse_date_flex,
)
from .security import KeyRing


@dataclass
class PrivacyProfile:
    date_shift: int
    age_band_width: int
    elderly_cap: int
    generalize_district: bool
    pin_prefix_digits: int
    suppress_dob: bool
    state_label: str
    elevated_risk: bool = False


def build_privacy_profiles(
    all_clues: list[RecordClues],
    unit_to_psn: dict[str, str | None],
    psn_clusters: dict[str, PatientCluster],
    keyring: KeyRing,
    policy: dict[str, Any],
) -> tuple[dict[str, PrivacyProfile], dict[str, PrivacyProfile]]:
    """Compute dataset-intrinsic k-anonymity & condition-rarity risk profiles per patient and per unlinked unit."""
    std_band = int(policy.get("age_band_years", 10))
    rare_band = int(policy.get("high_risk_age_band_years", 20))
    elderly_cap = int(policy.get("elderly_cap_min_age", 80))
    ped_max = int(policy.get("pediatric_widen_max_age", 5))
    min_k = int(policy.get("min_group_size", 5))
    rare_cond_th = int(policy.get("rare_condition_threshold", 3))
    min_shift = int(policy.get("date_shift_min_days", 180))
    max_shift = int(policy.get("date_shift_max_days", 540))
    pin_digits = int(policy.get("pin_prefix_digits", 3))
    suppress_dob = bool(policy.get("suppress_dob", True))
    gen_rare_dist = bool(policy.get("generalize_rare_district", True))
    state_label = str(policy.get("default_state_label", "Uttar Pradesh"))

    # 1. Condition/procedure patient frequency across clusters
    cond_patients: dict[str, set[str]] = defaultdict(set)
    for psn, cl in psn_clusters.items():
        for d in cl.diagnoses:
            cond_patients[d].add(psn)

    rare_conditions = {d for d, pts in cond_patients.items() if len(pts) <= rare_cond_th}

    # 2. Cohort cell counts on (district, gender, base_age_band)
    def _cell_key(cl: PatientCluster) -> tuple[str, str, int]:
        dist = canonicalize_place(cl.districts[0]) if cl.districts else ""
        gen = sorted(cl.genders)[0] if cl.genders else ""
        if cl.ages:
            a = max(0, cl.ages[0])
            b = elderly_cap if a >= elderly_cap else (a // std_band) * std_band
        else:
            b = -1
        return (dist, gen, b)

    cell_counts = Counter(_cell_key(cl) for cl in psn_clusters.values())

    psn_profiles: dict[str, PrivacyProfile] = {}
    for psn, cl in psn_clusters.items():
        ckey = cl.canonical_key()
        shift = keyring.date_shift_days(ckey, min_days=min_shift, max_days=max_shift)
        has_rare = bool(cl.diagnoses & rare_conditions)
        has_elderly = any(a >= elderly_cap for a in cl.ages)
        has_ped = any(a <= ped_max for a in cl.ages)
        small_cell = cell_counts[_cell_key(cl)] < min_k
        elevated = has_rare or has_elderly or (has_ped and small_cell) or small_cell
        gen_dist = gen_rare_dist and elevated

        psn_profiles[psn] = PrivacyProfile(
            date_shift=shift,
            age_band_width=rare_band if elevated else std_band,
            elderly_cap=elderly_cap,
            generalize_district=gen_dist,
            pin_prefix_digits=0 if gen_dist else pin_digits,
            suppress_dob=suppress_dob,
            state_label=state_label,
            elevated_risk=elevated,
        )

    unit_profiles: dict[str, PrivacyProfile] = {}
    for rc in all_clues:
        psn = unit_to_psn.get(rc.unit_id)
        if psn and psn in psn_profiles:
            unit_profiles[rc.unit_id] = psn_profiles[psn]
        else:
            shift = keyring.date_shift_days(f"unit|{rc.unit_id}", min_days=min_shift, max_days=max_shift)
            has_rare = bool(rc.diagnoses & rare_conditions)
            has_elderly = rc.age is not None and rc.age >= elderly_cap
            elevated = has_rare or has_elderly
            gen_dist = gen_rare_dist and elevated
            unit_profiles[rc.unit_id] = PrivacyProfile(
                date_shift=shift,
                age_band_width=rare_band if elevated else std_band,
                elderly_cap=elderly_cap,
                generalize_district=gen_dist,
                pin_prefix_digits=0 if gen_dist else pin_digits,
                suppress_dob=suppress_dob,
                state_label=state_label,
                elevated_risk=elevated,
            )

    return psn_profiles, unit_profiles


# --------------------------------------------------------------------------- Formatting Helpers

def format_age_band(age: int, prof: PrivacyProfile) -> str:
    if age >= prof.elderly_cap:
        hi = max(99, ((age // 20) + 1) * 20 - 1)
        return f"{prof.elderly_cap}-{hi}"
    a = max(0, age)
    w = max(1, prof.age_band_width)
    lo = (a // w) * w
    return f"{lo}-{lo + w - 1}"


def format_shifted_date(d: dt.date, style: str, shift_days: int) -> str:
    nd = d + dt.timedelta(days=shift_days)
    if style == "iso":
        return nd.isoformat()
    if style.startswith("iso_dt:"):
        return nd.isoformat() + style[len("iso_dt:"):]
    if style == "month_text":
        return nd.strftime("%d %B %Y")
    if ":" in style:
        order, sep, yr_len = style.split(":", 2)
        yr = f"{nd.year % 100:02d}" if yr_len == "2" else f"{nd.year:04d}"
        if order == "mdy":
            return f"{nd.month:02d}{sep}{nd.day:02d}{sep}{yr}"
        return f"{nd.day:02d}{sep}{nd.month:02d}{sep}{yr}"
    return nd.isoformat()


def shift_date_value(raw: str, hint: str, shift_days: int, default_year: int) -> tuple[str, bool]:
    d, style = parse_date_flex(raw, hint=hint, default_year=default_year)
    if d is None:
        return raw, False
    return format_shifted_date(d, style, shift_days), True


def format_pin_generalized(raw: str, prof: PrivacyProfile) -> str:
    digits = re.sub(r"\D", "", to_ascii_digits(raw))
    if len(digits) != 6:
        return "[PIN]"
    if prof.pin_prefix_digits <= 0:
        return "******"
    k = min(6, prof.pin_prefix_digits)
    return digits[:k] + "*" * (6 - k)


def _lookup_psn_for_token(token: str, ref_lookup: dict[str, str]) -> str | None:
    if not token:
        return None
    mkey = norm_mrn_key(token)
    if mkey in ref_lookup:
        return ref_lookup[mkey]
    cname = " ".join(canonicalize_person_name(token))
    if cname and cname in ref_lookup:
        return ref_lookup[cname]
    return None


def _sanitize_key_or_header(
    text: str,
    detector: PIIDetector,
    ref_lookup: dict[str, str],
    keyring: KeyRing,
    used_keys: set[str],
    entity_counts: dict[str, int],
) -> tuple[str, bool]:
    """Sanitize a CSV column header or JSON dict key without collisions."""
    if not detector.has_pii_in_identifier(text):
        return text, False

    # Try exact lookup via MRN or full name
    for part in re.split(r"[\s,|]+", text.strip()):
        psn = _lookup_psn_for_token(part, ref_lookup)
        if psn:
            entity_counts["MRN" if any(c.isdigit() for c in part) else "PATIENT_NAME"] = (
                entity_counts.get("MRN" if any(c.isdigit() for c in part) else "PATIENT_NAME", 0) + 1
            )
            cand = psn
            idx = 2
            while cand in used_keys:
                cand = f"{psn}_{idx}"
                idx += 1
            used_keys.add(cand)
            return cand, True

    psn = _lookup_psn_for_token(text, ref_lookup)
    if psn:
        entity_counts["PATIENT_NAME"] = entity_counts.get("PATIENT_NAME", 0) + 1
        cand = psn
        idx = 2
        while cand in used_keys:
            cand = f"{psn}_{idx}"
            idx += 1
        used_keys.add(cand)
        return cand, True

    # Generic fallback HMAC tag
    entity_counts["PATIENT_NAME"] = entity_counts.get("PATIENT_NAME", 0) + 1
    tag = keyring.safe_tag(text, 6)
    cand = f"ID_{tag}"
    idx = 2
    while cand in used_keys:
        cand = f"ID_{tag}_{idx}"
        idx += 1
    used_keys.add(cand)
    return cand, True


# --------------------------------------------------------------------------- Main Per-File Sanitization

def sanitize_file(
    inf: IngestedFile,
    date_hint: str,
    modal_year: int,
    clues_list: list[RecordClues],
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    ref_lookup: dict[str, str],
    detector: PIIDetector,
    keyring: KeyRing,
    entity_counts: dict[str, int],
    stats: dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    """Sanitize, generalize, date-shift, and inject pseudonyms into an ingested file."""
    rel = inf.rel_path
    clues_by_loc = {c.loc: c for c in clues_list}
    records_out: list[dict[str, Any]] = []

    if inf.fmt in ("csv", "tsv"):
        out_str = _sanitize_csv(
            inf, date_hint, modal_year, clues_by_loc, unit_to_psn, unit_profiles,
            ref_lookup, detector, keyring, entity_counts, stats, records_out,
        )
        return out_str, records_out

    if inf.fmt == "fhir":
        out_str = _sanitize_fhir(
            inf, clues_by_loc, unit_to_psn, unit_profiles,
            detector, entity_counts, stats, records_out,
        )
        return out_str, records_out

    if inf.fmt == "json":
        out_str = _sanitize_json(
            inf, date_hint, modal_year, clues_by_loc, unit_to_psn, unit_profiles,
            ref_lookup, detector, keyring, entity_counts, stats, records_out,
        )
        return out_str, records_out

    # Plain text
    out_str = _sanitize_text_doc(
        inf, date_hint, modal_year, clues_by_loc, unit_to_psn, unit_profiles,
        detector, entity_counts, stats, records_out,
    )
    return out_str, records_out


# --------------------------------------------------------------------------- CSV / TSV Sanitizer

def _sanitize_csv(
    inf: IngestedFile,
    date_hint: str,
    modal_year: int,
    clues_by_loc: dict[str, RecordClues],
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    ref_lookup: dict[str, str],
    detector: PIIDetector,
    keyring: KeyRing,
    entity_counts: dict[str, int],
    stats: dict[str, int],
    records_out: list[dict[str, Any]],
) -> str:
    rows: list[list[str]] = inf.data
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

    # Choose which column receives the patient pseudonym so it appears INSIDE the record
    psn_col_idx: int | None = None
    for preferred in ("MRN", "PATIENT_NAME"):
        if preferred in col_roles:
            psn_col_idx = col_roles.index(preferred)
            break

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=inf.delimiter or ",", lineterminator="\n")
    writer.writerow(clean_header)
    chunk_text, chunk_records, chunk_counts, chunk_gen = sanitize_csv_rows(
        rel_path=inf.rel_path,
        delimiter=inf.delimiter or ",",
        col_roles=col_roles,
        psn_col_idx=psn_col_idx,
        rows_chunk=body,
        start_row_idx=0,
        date_hint=date_hint,
        modal_year=modal_year,
        clues_by_loc=clues_by_loc,
        unit_to_psn=unit_to_psn,
        unit_profiles=unit_profiles,
        detector=detector,
    )
    for k, v in chunk_counts.items():
        entity_counts[k] = entity_counts.get(k, 0) + v
    if chunk_gen:
        stats["records_generalized"] = stats.get("records_generalized", 0) + chunk_gen
    records_out.extend(chunk_records)
    return buf.getvalue() + chunk_text


def sanitize_csv_rows(
    rel_path: str,
    delimiter: str,
    col_roles: list[str],
    psn_col_idx: int | None,
    rows_chunk: list[list[str]],
    start_row_idx: int,
    date_hint: str,
    modal_year: int,
    clues_by_loc: dict[str, RecordClues],
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    detector: PIIDetector,
) -> tuple[str, list[dict[str, Any]], dict[str, int], int]:
    """Sanitize a slice of CSV/TSV body rows and return (csv_text, records, entity_counts, generalized_count)."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=delimiter, lineterminator="\n")
    cell_spans_cache: dict[tuple[str, bool], list[Any]] = {}
    entity_counts: dict[str, int] = {}
    records_out: list[dict[str, Any]] = []
    generalized_count = 0

    for offset, row in enumerate(rows_chunk):
        r_idx = start_row_idx + offset
        loc = f"row:{r_idx}"
        rc = clues_by_loc.get(loc)
        uid = rc.unit_id if rc else f"{rel_path}|{loc}"
        psn = unit_to_psn.get(uid)
        prof = unit_profiles.get(uid)
        extra_pat = rc.patient_names if rc else []
        extra_rel = rc.relative_names if rc else []
        is_clean_file = not (rc and rc.is_patient_record) and not any(
            detector.has_pii_in_identifier(c) or detector.detect_text(c, allow_pin_scan=False) for c in row
        )

        clean_row: list[str] = []
        generalized_any = False

        if is_clean_file:
            clean_row = list(row)
        else:
            for c_idx, cell in enumerate(row):
                role = col_roles[c_idx] if c_idx < len(col_roles) else "OTHER"
                val = cell

                if psn and psn_col_idx == c_idx:
                    if val.strip():
                        entity_counts[role if role in ("MRN", "PATIENT_NAME") else "MRN"] = (
                            entity_counts.get(role if role in ("MRN", "PATIENT_NAME") else "MRN", 0) + 1
                        )
                    clean_row.append(psn)
                    continue

                if role in ("PATIENT_NAME", "RELATIVE_NAME", "MRN", "AADHAAR", "ABHA", "PAN", "INSURANCE_ID", "PHONE", "EMAIL", "HOUSE", "LOCALITY", "LANDMARK", "VILLAGE", "TEHSIL"):
                    if val.strip():
                        # Count masked Aadhaar properly
                        if role == "AADHAAR" and ("x" in val.lower() or "*" in val):
                            entity_counts["AADHAAR_MASKED"] = entity_counts.get("AADHAAR_MASKED", 0) + 1
                            clean_row.append("[AADHAAR_MASKED]")
                        else:
                            etype = "ABHA_NUMBER" if role == "ABHA" else role
                            entity_counts[etype] = entity_counts.get(etype, 0) + 1
                            clean_row.append(f"[{etype}]")
                    else:
                        clean_row.append(val)
                    if role in ("VILLAGE", "TEHSIL") and val.strip():
                        generalized_any = True
                    continue

                if role == "DOB":
                    if val.strip():
                        entity_counts["DOB"] = entity_counts.get("DOB", 0) + 1
                        if prof and not prof.suppress_dob:
                            shifted, ok = shift_date_value(val, date_hint, prof.date_shift, modal_year)
                            clean_row.append(shifted if ok else "[DOB]")
                        else:
                            clean_row.append("[DOB]")
                        generalized_any = True
                    else:
                        clean_row.append(val)
                    continue

                if role == "AGE" and prof:
                    a, g = parse_age_val(val)
                    if a is not None:
                        band = format_age_band(a, prof)
                        suffix = f"/{g}" if g else ""
                        clean_row.append(f"{band}{suffix}")
                        generalized_any = True
                    else:
                        clean_row.append(val)
                    continue

                if role == "AGE_SEX" and prof:
                    a, g = parse_age_val(val)
                    if a is not None:
                        band = format_age_band(a, prof)
                        clean_row.append(f"{band}/{g}" if g else band)
                        generalized_any = True
                    else:
                        clean_row.append(val)
                    continue

                if role == "PIN" and prof:
                    if val.strip():
                        entity_counts["PIN"] = entity_counts.get("PIN", 0) + 1
                        clean_row.append(format_pin_generalized(val, prof))
                        generalized_any = True
                    else:
                        clean_row.append(val)
                    continue

                if role == "DISTRICT" and prof:
                    if prof.generalize_district and val.strip():
                        clean_row.append(prof.state_label)
                        generalized_any = True
                    else:
                        clean_row.append(val)
                    continue

                if role == "DATE" and prof:
                    if val.strip():
                        shifted, _ = shift_date_value(val, date_hint, prof.date_shift, modal_year)
                        clean_row.append(shifted)
                    else:
                        clean_row.append(val)
                    continue

                if role == "DOCTOR":
                    clean_row.append(val)
                    continue

                if role in ("ADDRESS", "FREE_TEXT", "OTHER"):
                    if not val or not val.strip():
                        clean_row.append(val)
                        continue
                    allow_pin = (role == "ADDRESS")
                    # Fast path: if no extra_pat/extra_rel token appears in val, reuse cached base spans
                    has_extra_hit = False
                    if extra_pat or extra_rel:
                        val_low = val.lower()
                        for nm in extra_pat:
                            for tok in nm.split():
                                if len(tok) >= 3 and tok.lower() in val_low:
                                    has_extra_hit = True
                                    break
                            if has_extra_hit:
                                break
                        if not has_extra_hit:
                            for nm in extra_rel:
                                for tok in nm.split():
                                    if len(tok) >= 3 and tok.lower() in val_low:
                                        has_extra_hit = True
                                        break
                                if has_extra_hit:
                                    break

                    if not has_extra_hit:
                        ckey = (val, allow_pin)
                        cached_spans = cell_spans_cache.get(ckey)
                        if cached_spans is None:
                            cached_spans = detector.detect_text(val, allow_pin_scan=allow_pin)
                            if len(cell_spans_cache) < 65536:
                                cell_spans_cache[ckey] = cached_spans
                        if not cached_spans:
                            txt_clean = val
                        else:
                            out_parts = []
                            cur = 0
                            for sp in cached_spans:
                                out_parts.append(val[cur:sp.start])
                                entity_counts[sp.pii_type] = entity_counts.get(sp.pii_type, 0) + 1
                                out_parts.append(f"{sp.surface[:3]}***" if sp.pii_type == "PIN" else f"[{sp.pii_type}]")
                                cur = sp.end
                            out_parts.append(val[cur:])
                            txt_clean = "".join(out_parts)
                    else:
                        # In ADDRESS or FREE_TEXT or general cells, run full contextual detector + district generalization
                        txt_clean = detector.redact_text(
                            val,
                            extra_patient_names=extra_pat,
                            extra_relative_names=extra_rel,
                            allow_pin_scan=allow_pin,
                            entity_counts=entity_counts,
                        )
                    if role == "ADDRESS" and prof and prof.generalize_district and rc and rc.district:
                        txt_clean = re.sub(re.escape(rc.district), prof.state_label, txt_clean, flags=re.I)
                    clean_row.append(txt_clean)
                    continue

                clean_row.append(val)

        writer.writerow(clean_row)
        if generalized_any:
            generalized_count += 1
        records_out.append({
            "file": f"sanitized/{rel_path}",
            "loc": loc,
            "patient": psn,
            **({"status": "generalized"} if generalized_any else {}),
        })

    return buf.getvalue(), records_out, entity_counts, generalized_count


# --------------------------------------------------------------------------- JSON (Non-FHIR) Sanitizer

def _sanitize_json(
    inf: IngestedFile,
    date_hint: str,
    modal_year: int,
    clues_by_loc: dict[str, RecordClues],
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    ref_lookup: dict[str, str],
    detector: PIIDetector,
    keyring: KeyRing,
    entity_counts: dict[str, int],
    stats: dict[str, int],
    records_out: list[dict[str, Any]],
) -> str:
    data = copy.deepcopy(inf.data)
    rel = inf.rel_path

    # Clean file shortcut (e.g. ward_census.json)
    if rel.startswith("clean/"):
        records_out.append({"file": f"sanitized/{rel}", "loc": "doc", "patient": None})
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"

    is_single_patient_doc = isinstance(data, dict) and "patient" in data and isinstance(data["patient"], dict)
    is_doc = (
        not isinstance(data, dict)
        or is_single_patient_doc
        or not any(
            isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
            for k, v in data.items()
            if k.lower() in ("visits", "patients", "results", "records", "claims")
        )
    )
    is_non_patient_misc = rel in {"misc/helpdesk_tickets.json", "misc/scanned_consent_form.json", "misc/patient_index.json"}

    def _clean_node(
        node: Any,
        path_parts: list[str],
        rc: RecordClues | None,
        psn: str | None,
        prof: PrivacyProfile | None,
        psn_injected: list[bool],
        gen_flag: list[bool],
    ) -> Any:
        if isinstance(node, dict):
            used_k: set[str] = set()
            out_d: dict[str, Any] = {}
            for k, v in node.items():
                sk, key_changed = _sanitize_key_or_header(str(k), detector, ref_lookup, keyring, used_k, entity_counts)
                used_k.add(sk)
                # If key was an MRN/Name and v is string MRN in patient_index.by_name, map v too
                if key_changed and isinstance(v, str) and norm_mrn_key(v) in ref_lookup:
                    entity_counts["MRN"] = entity_counts.get("MRN", 0) + 1
                    out_d[sk] = ref_lookup[norm_mrn_key(v)]
                else:
                    out_d[sk] = _clean_node(v, path_parts + [str(k).lower()], rc, psn, prof, psn_injected, gen_flag)
            return out_d

        if isinstance(node, list):
            return [_clean_node(x, path_parts, rc, psn, prof, psn_injected, gen_flag) for x in node]

        if node is None or isinstance(node, bool):
            return node

        leaf_key = path_parts[-1] if path_parts else ""
        parent_key = path_parts[-2] if len(path_parts) >= 2 else ""
        extra_pat = rc.patient_names if rc else []
        extra_rel = rc.relative_names if rc else []

        # Metadata path keys (e.g. _export_meta.source_path)
        if leaf_key in ("source_path", "filepath", "file_path", "export_path"):
            if isinstance(node, str) and node.strip():
                cleaned_path = detector.redact_text(node.replace("_", " ").replace("\\", " / "), allow_pin_scan=False, entity_counts=entity_counts)
                # Ensure no raw patient segment survives
                return "[REDACTED_PATH]" if cleaned_path != node else "[REDACTED_PATH]"
            return node

        # Base64 attachment handling
        if isinstance(node, str) and leaf_key == "data":
            dec = try_decode_base64_text(node)
            if dec is not None:
                cleaned_dec = detector.redact_text(
                    dec,
                    extra_patient_names=extra_pat,
                    extra_relative_names=extra_rel,
                    allow_pin_scan=True,
                    pseudonym=psn if (psn and not psn_injected[0]) else None,
                    entity_counts=entity_counts,
                )
                if psn and psn in cleaned_dec:
                    psn_injected[0] = True
                return encode_base64_text(cleaned_dec)

        # Direct identifiers by semantic key
        if leaf_key in ("uhid", "patient_ref", "hospital_uhid", "reg_no", "mrn"):
            if str(node).strip():
                entity_counts["MRN"] = entity_counts.get("MRN", 0) + 1
                if psn:
                    psn_injected[0] = True
                    return psn
                return "[MRN]"
            return node

        if leaf_key in ("aadhar_no", "aadhaar", "aadhaar_no"):
            if node is not None and str(node).strip():
                entity_counts["AADHAAR"] = entity_counts.get("AADHAAR", 0) + 1
                return "[AADHAAR]"
            return node

        if leaf_key in ("abha", "abha_no", "abha_number", "abha_address"):
            if node is not None and str(node).strip():
                etype = "ABHA_ADDRESS" if "address" in leaf_key else "ABHA_NUMBER"
                entity_counts[etype] = entity_counts.get(etype, 0) + 1
                return f"[{etype}]"
            return node

        if leaf_key in ("mobile", "phone", "callback"):
            if node is not None and str(node).strip():
                entity_counts["PHONE"] = entity_counts.get("PHONE", 0) + 1
                return "[PHONE]"
            return node

        if leaf_key == "email":
            if node is not None and str(node).strip():
                entity_counts["EMAIL"] = entity_counts.get("EMAIL", 0) + 1
                return "[EMAIL]"
            return node

        if leaf_key in ("patient_name",) or (leaf_key == "name" and parent_key in ("patient", "raised_by")):
            if isinstance(node, str) and node.strip():
                entity_counts["PATIENT_NAME"] = entity_counts.get("PATIENT_NAME", 0) + 1
                if psn and not psn_injected[0]:
                    psn_injected[0] = True
                    return psn
                return "[PATIENT_NAME]"
            return node

        if parent_key == "name" and leaf_key in ("first", "last", "local"):
            if isinstance(node, str) and node.strip():
                entity_counts["PATIENT_NAME"] = entity_counts.get("PATIENT_NAME", 0) + 1
                if psn and leaf_key == "first" and not psn_injected[0]:
                    psn_injected[0] = True
                    return psn
                return "[PATIENT_NAME]"
            return node

        if parent_key == "guardian" and leaf_key == "name":
            if isinstance(node, str) and node.strip():
                entity_counts["RELATIVE_NAME"] = entity_counts.get("RELATIVE_NAME", 0) + 1
                return "[RELATIVE_NAME]"
            return node

        if leaf_key == "dob":
            if node is not None and str(node).strip():
                entity_counts["DOB"] = entity_counts.get("DOB", 0) + 1
                gen_flag[0] = True
                return "[DOB]"
            return node

        if leaf_key == "age" and prof is not None:
            a, _ = parse_age_val(node)
            if a is not None:
                gen_flag[0] = True
                return format_age_band(a, prof)
            return node

        if leaf_key == "age_sex" and prof is not None:
            a, g = parse_age_val(node)
            if a is not None:
                gen_flag[0] = True
                band = format_age_band(a, prof)
                return f"{band}/{g}" if g else band
            return node

        if leaf_key in ("village", "city"):
            if isinstance(node, str) and node.strip():
                entity_counts["VILLAGE"] = entity_counts.get("VILLAGE", 0) + 1
                gen_flag[0] = True
                return "[VILLAGE]"
            return node

        if leaf_key == "tehsil":
            if isinstance(node, str) and node.strip():
                gen_flag[0] = True
                return "[TEHSIL]"
            return node

        if leaf_key == "district" and prof is not None:
            if prof.generalize_district and isinstance(node, str) and node.strip():
                gen_flag[0] = True
                return prof.state_label
            return node

        if leaf_key in ("pin", "postalcode") and prof is not None:
            if isinstance(node, str) and node.strip():
                entity_counts["PIN"] = entity_counts.get("PIN", 0) + 1
                gen_flag[0] = True
                return format_pin_generalized(node, prof)
            return node

        if leaf_key in ("visit_date", "collected", "date", "registered_on") and prof is not None:
            if isinstance(node, str) and node.strip():
                shifted, _ = shift_date_value(node, date_hint, prof.date_shift, modal_year)
                return shifted
            return node

        if leaf_key in ("doctor", "referred_by"):
            return node

        if isinstance(node, str):
            # Free text fields (line, notes, remarks, address_note, reviewed_by, etc.)
            cleaned = detector.redact_text(
                node,
                extra_patient_names=extra_pat,
                extra_relative_names=extra_rel,
                allow_pin_scan=(leaf_key in ("line", "address", "address_note")),
                entity_counts=entity_counts,
            )
            if leaf_key in ("line", "address", "address_note") and prof and prof.generalize_district and rc and rc.district:
                cleaned = re.sub(re.escape(rc.district), prof.state_label, cleaned, flags=re.I)
            return cleaned

        return node

    if is_doc or is_non_patient_misc:
        loc = "doc"
        rc = clues_by_loc.get(loc)
        uid = rc.unit_id if rc else f"{rel}|doc"
        psn = unit_to_psn.get(uid)
        prof = unit_profiles.get(uid)
        psn_inj = [False]
        gen_flag = [False]
        cleaned = _clean_node(data, [], rc, psn, prof, psn_inj, gen_flag)
        if gen_flag[0]:
            stats["records_generalized"] = stats.get("records_generalized", 0) + 1
        records_out.append({
            "file": f"sanitized/{rel}",
            "loc": "doc",
            "patient": psn,
            **({"status": "generalized"} if gen_flag[0] else {}),
        })
        return json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n"

    # List-of-dicts JSON (visits / patients / results)
    # First sanitize metadata keys cleanly
    used_top: set[str] = set()
    cleaned_root: dict[str, Any] = {}
    for k, v in data.items():
        sk, _ = _sanitize_key_or_header(str(k), detector, ref_lookup, keyring, used_top, entity_counts)
        used_top.add(sk)
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            pkey = k.replace("~", "~0").replace("/", "~1")
            out_list = []
            for i, item in enumerate(v):
                loc = f"/{pkey}/{i}"
                rc = clues_by_loc.get(loc)
                uid = rc.unit_id if rc else f"{rel}|{loc}"
                psn = unit_to_psn.get(uid)
                prof = unit_profiles.get(uid)
                psn_inj = [False]
                gen_flag = [False]
                c_item = _clean_node(item, [str(k).lower()], rc, psn, prof, psn_inj, gen_flag)
                if psn and not psn_inj[0] and isinstance(c_item, dict):
                    c_item["patient_ref"] = psn
                out_list.append(c_item)
                if gen_flag[0]:
                    stats["records_generalized"] = stats.get("records_generalized", 0) + 1
                records_out.append({
                    "file": f"sanitized/{rel}",
                    "loc": loc,
                    "patient": psn,
                    **({"status": "generalized"} if gen_flag[0] else {}),
                })
            cleaned_root[sk] = out_list
        else:
            cleaned_root[sk] = _clean_node(v, [str(k).lower()], None, None, None, [False], [False])

    return json.dumps(cleaned_root, ensure_ascii=False, indent=2) + "\n"


# --------------------------------------------------------------------------- FHIR R4 Bundle Sanitizer

def _shift_fhir_datetime(val: str, shift_days: int) -> str:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})(T.*)?$", str(val).strip())
    if not m:
        return val
    try:
        d = dt.date.fromisoformat(m.group(1))
    except ValueError:
        return val
    nd = d + dt.timedelta(days=shift_days)
    return nd.isoformat() + (m.group(2) or "")


def _sanitize_fhir(
    inf: IngestedFile,
    clues_by_loc: dict[str, RecordClues],
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    detector: PIIDetector,
    entity_counts: dict[str, int],
    stats: dict[str, int],
    records_out: list[dict[str, Any]],
) -> str:
    bundle = copy.deepcopy(inf.data)
    rel = inf.rel_path
    rc = clues_by_loc.get("doc")
    uid = rc.unit_id if rc else f"{rel}|doc"
    psn = unit_to_psn.get(uid)
    prof = unit_profiles.get(uid)
    shift_days = prof.date_shift if prof else 0
    extra_pat = rc.patient_names if rc else []
    extra_rel = rc.relative_names if rc else []
    pid_label = psn if psn else "[PATIENT]"
    generalized_any = False

    if isinstance(bundle.get("meta"), dict):
        if bundle["meta"].get("lastUpdated"):
            bundle["meta"]["lastUpdated"] = _shift_fhir_datetime(bundle["meta"]["lastUpdated"], shift_days)
        if bundle["meta"].get("source"):
            entity_counts["MRN"] = entity_counts.get("MRN", 0) + 1
            bundle["meta"]["source"] = f"#{pid_label}"
    if bundle.get("timestamp"):
        bundle["timestamp"] = _shift_fhir_datetime(bundle["timestamp"], shift_days)

    for ent in bundle.get("entry", []):
        res = ent.get("resource")
        if not isinstance(res, dict):
            continue
        rt = res.get("resourceType")

        # Update subject reference display across clinical resources
        if isinstance(res.get("subject"), dict) and "display" in res["subject"]:
            entity_counts["PATIENT_NAME"] = entity_counts.get("PATIENT_NAME", 0) + 1
            res["subject"]["display"] = pid_label

        if rt == "Composition":
            if res.get("date"):
                res["date"] = _shift_fhir_datetime(res["date"], shift_days)

        elif rt == "Patient":
            # Clean narrative text
            if isinstance(res.get("text"), dict) and "div" in res["text"]:
                entity_counts["PATIENT_NAME"] = entity_counts.get("PATIENT_NAME", 0) + 1
                res["text"]["div"] = f'<div xmlns="http://www.w3.org/1999/xhtml">Patient {pid_label}</div>'

            # Replace identifiers cleanly with pseudonym
            idents = res.get("identifier") or []
            for id_obj in idents:
                sys_u = str(id_obj.get("system", "")).lower()
                if "abdm" in sys_u or "healthid" in sys_u:
                    entity_counts["ABHA_NUMBER"] = entity_counts.get("ABHA_NUMBER", 0) + 1
                elif "aadhaar" in sys_u:
                    entity_counts["AADHAAR"] = entity_counts.get("AADHAAR", 0) + 1
                else:
                    entity_counts["MRN"] = entity_counts.get("MRN", 0) + 1
            res["identifier"] = [{"system": "urn:raksha:pseudonym", "value": pid_label}]

            if res.get("name"):
                entity_counts["PATIENT_NAME"] = entity_counts.get("PATIENT_NAME", 0) + 1
            res["name"] = [{"text": pid_label}]

            if res.get("telecom"):
                for tc in res["telecom"]:
                    etype = "EMAIL" if tc.get("system") == "email" else "PHONE"
                    entity_counts[etype] = entity_counts.get(etype, 0) + 1
                res.pop("telecom", None)

            if "birthDate" in res:
                entity_counts["DOB"] = entity_counts.get("DOB", 0) + 1
                res.pop("birthDate", None)
                generalized_any = True

            if isinstance(res.get("address"), list):
                for addr in res["address"]:
                    if addr.get("line"):
                        entity_counts["HOUSE"] = entity_counts.get("HOUSE", 0) + 1
                        addr["line"] = ["[ADDRESS]"]
                    if addr.get("city"):
                        entity_counts["VILLAGE"] = entity_counts.get("VILLAGE", 0) + 1
                        addr["city"] = "[VILLAGE]"
                        generalized_any = True
                    if prof and prof.generalize_district and addr.get("district"):
                        addr["district"] = prof.state_label
                        generalized_any = True
                    if addr.get("postalCode") and prof:
                        entity_counts["PIN"] = entity_counts.get("PIN", 0) + 1
                        addr["postalCode"] = format_pin_generalized(str(addr["postalCode"]), prof)
                        generalized_any = True

            if isinstance(res.get("contact"), list):
                for cont in res["contact"]:
                    if isinstance(cont.get("name"), dict):
                        entity_counts["RELATIVE_NAME"] = entity_counts.get("RELATIVE_NAME", 0) + 1
                        cont["name"] = {"text": "[RELATIVE_NAME]"}

            if isinstance(res.get("extension"), list):
                for ext in res["extension"]:
                    if "valueString" in ext:
                        entity_counts["RELATIVE_NAME"] = entity_counts.get("RELATIVE_NAME", 0) + 1
                        ext["valueString"] = "[RELATIVE_NAME]"

        elif rt == "Encounter":
            if res.get("identifier"):
                entity_counts["MRN"] = entity_counts.get("MRN", 0) + 1
                res["identifier"] = [{"system": "urn:raksha:encounter", "value": f"ENC-{pid_label}"}]
            if isinstance(res.get("period"), dict):
                for k in ("start", "end"):
                    if res["period"].get(k):
                        res["period"][k] = _shift_fhir_datetime(res["period"][k], shift_days)

        elif rt == "Condition":
            if res.get("recordedDate"):
                res["recordedDate"] = _shift_fhir_datetime(res["recordedDate"], shift_days)

        elif rt == "MedicationRequest":
            if res.get("authoredOn"):
                res["authoredOn"] = _shift_fhir_datetime(res["authoredOn"], shift_days)

        elif rt == "Observation":
            if res.get("effectiveDateTime"):
                res["effectiveDateTime"] = _shift_fhir_datetime(res["effectiveDateTime"], shift_days)

        elif rt == "Procedure":
            if res.get("performedDateTime"):
                res["performedDateTime"] = _shift_fhir_datetime(res["performedDateTime"], shift_days)

        elif rt == "DocumentReference":
            for cont in res.get("content", []):
                att = cont.get("attachment")
                if isinstance(att, dict) and isinstance(att.get("data"), str):
                    dec = try_decode_base64_text(att["data"])
                    if dec is not None:
                        cleaned_dec = detector.redact_text(
                            dec,
                            extra_patient_names=extra_pat,
                            extra_relative_names=extra_rel,
                            allow_pin_scan=False,
                            pseudonym=psn,
                            entity_counts=entity_counts,
                        )
                        att["data"] = encode_base64_text(cleaned_dec)

        elif rt == "Organization":
            # Generalize Organization city/district/address for elevated-risk patients while keeping hospital name
            if prof and prof.generalize_district:
                if isinstance(res.get("address"), list):
                    for addr in res["address"]:
                        if addr.get("city"):
                            addr["city"] = prof.state_label
                        if addr.get("district"):
                            addr["district"] = prof.state_label

    assert validate_fhir_bundle(bundle), "Sanitized FHIR bundle failed R4 structural validation"
    if generalized_any:
        stats["records_generalized"] = stats.get("records_generalized", 0) + 1
    records_out.append({
        "file": f"sanitized/{rel}",
        "loc": "doc",
        "patient": psn,
        **({"status": "generalized"} if generalized_any else {}),
    })
    return json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"


# --------------------------------------------------------------------------- Plain Text Sanitizer

_RX_TEXT_AGE = re.compile(
    r"(?i)((?:Age(?:/Sex)?|आयु)\s*:\s*)(-?\d{1,3})(\s*(?:Yrs?|वर्ष)?(?:\s*/\s*(?:M|F|Male|Female|पुरुष|महिला))?)"
)
_RX_TEXT_AGE_INLINE = re.compile(r"(?i)\b(\d{1,3})(\s*Y\s*/\s*[MF])\b")
_RX_TEXT_DATE_LINE = re.compile(
    r"(?i)((?:Date(?:\s+of\s+(?:Admission|Discharge))?|Visit|तारीख|भर्ती\s+की\s+तारीख|छुट्टी\s+की\s+तारीख)\s*:\s*)([^\n]+)"
)
_RX_TEXT_DOB_LINE = re.compile(r"(?im)^(\s*DOB\s*:\s*)([^\n]+)$")
_RX_TEXT_WA_TS = re.compile(r"^(\[)(\d{1,2}/\d{1,2}/\d{2,4})(,\s*[^\]]+\])")
_RX_TEXT_NURSE_TS = re.compile(r"(?m)^(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})(\s+\d{1,2}:\d{2}\b)")


def _sanitize_text_doc(
    inf: IngestedFile,
    date_hint: str,
    modal_year: int,
    clues_by_loc: dict[str, RecordClues],
    unit_to_psn: dict[str, str | None],
    unit_profiles: dict[str, PrivacyProfile],
    detector: PIIDetector,
    entity_counts: dict[str, int],
    stats: dict[str, int],
    records_out: list[dict[str, Any]],
) -> str:
    text = inf.text or ""
    rel = inf.rel_path
    rc = clues_by_loc.get("doc")
    uid = rc.unit_id if rc else f"{rel}|doc"
    psn = unit_to_psn.get(uid)
    prof = unit_profiles.get(uid)
    extra_pat = rc.patient_names if rc else []
    extra_rel = rc.relative_names if rc else []

    # Check if purely clean clinical reference document (zero PII entities anywhere)
    initial_spans = detector.detect_text(text, allow_pin_scan=False)
    if (not rc or not rc.is_patient_record) and not initial_spans:
        records_out.append({"file": f"sanitized/{rel}", "loc": "doc", "patient": None})
        return text

    generalized_any = False

    # 1. Suppress DOB lines before general date shifting
    def _dob_sub(m: re.Match) -> str:
        nonlocal generalized_any
        entity_counts["DOB"] = entity_counts.get("DOB", 0) + 1
        generalized_any = True
        return f"{m.group(1)}[DOB]"

    work = _RX_TEXT_DOB_LINE.sub(_dob_sub, text)

    # 2. Shift dates if profile exists
    if prof:
        def _date_line_sub(m: re.Match) -> str:
            val_orig = m.group(2).strip()
            shifted, ok = shift_date_value(val_orig, date_hint, prof.date_shift, modal_year)
            return m.group(1) + (shifted if ok else val_orig)

        work = _RX_TEXT_DATE_LINE.sub(_date_line_sub, work)
        work = _RX_TEXT_WA_TS.sub(
            lambda m: m.group(1) + shift_date_value(m.group(2), "dmy", prof.date_shift, modal_year)[0] + m.group(3),
            work,
        )
        work = _RX_TEXT_NURSE_TS.sub(
            lambda m: shift_date_value(m.group(1), date_hint, prof.date_shift, modal_year)[0] + m.group(2),
            work,
        )

        # 3. Age banding (preserving gender & unit suffix, never touching durations)
        def _age_sub(m: re.Match) -> str:
            nonlocal generalized_any
            age_val = int(to_ascii_digits(m.group(2)))
            generalized_any = True
            return f"{m.group(1)}{format_age_band(age_val, prof)}{m.group(3)}"

        work = _RX_TEXT_AGE.sub(_age_sub, work)
        work = _RX_TEXT_AGE_INLINE.sub(
            lambda m: (lambda b: f"{b}{m.group(2)}")(format_age_band(int(to_ascii_digits(m.group(1))), prof)),
            work,
        )
        if _RX_TEXT_AGE_INLINE.search(text):
            generalized_any = True

    # 4. Run full multilingual PII redaction (injecting PSN-XXXXXX if linked)
    cleaned = detector.redact_text(
        work,
        extra_patient_names=extra_pat,
        extra_relative_names=extra_rel,
        allow_pin_scan=True,
        pseudonym=psn,
        entity_counts=entity_counts,
    )

    # 5. District & PIN generalization inside text if elevated risk (including hospital header line ", Barabanki")
    if prof and prof.generalize_district and rc:
        for d_name in ([rc.district] if rc.district else []):
            cleaned = re.sub(re.escape(d_name), prof.state_label, cleaned, flags=re.I)
        cleaned = re.sub(
            r"(?im)^([^\n]*(?:Hospital|Sadan)[^\n]*,\s*)([A-Za-z ]{3,25})$",
            rf"\g<1>{prof.state_label}",
            cleaned,
        )
        cleaned = re.sub(r"([1-8]\d{2})\*{3}", "******", cleaned)
        generalized_any = True

    # Guarantee pseudonym is physically present inside linked patient document
    if psn and psn not in cleaned:
        cleaned = f"Patient: {psn}\n" + cleaned

    if generalized_any:
        stats["records_generalized"] = stats.get("records_generalized", 0) + 1
    records_out.append({
        "file": f"sanitized/{rel}",
        "loc": "doc",
        "patient": psn,
        **({"status": "generalized"} if generalized_any else {}),
    })
    return cleaned
