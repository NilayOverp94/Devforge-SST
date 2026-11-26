"""Cross-script, cross-hospital record linkage with conflict guards and keyed HMAC pseudonyms."""
from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .detection import (
    PIIDetector,
    _consonant_skeleton,
    canonicalize_person_name,
    canonicalize_place,
    names_similarity,
    places_similarity,
    shadow_normalize,
    to_ascii_digits,
    try_decode_base64_text,
)
from .ingestion import IngestedFile
from .security import KeyRing


@dataclass
class RecordClues:
    unit_id: str  # f"{rel_path}|{loc}"
    file_rel: str
    loc: str
    fmt: str
    is_patient_record: bool = True
    hospital: str = ""
    mrns: set[str] = field(default_factory=set)
    aadhaars: set[str] = field(default_factory=set)
    aadhaar_last4: set[str] = field(default_factory=set)
    abhas: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    pans: set[str] = field(default_factory=set)
    insurance_ids: set[str] = field(default_factory=set)
    patient_names: list[str] = field(default_factory=list)
    relative_names: list[str] = field(default_factory=list)
    dob: str = ""          # YYYY-MM-DD
    age: int | None = None
    gender: str = ""       # M | F | O
    village: str = ""
    district: str = ""
    pin: str = ""
    diagnoses: set[str] = field(default_factory=set)
    dates: list[str] = field(default_factory=list)  # YYYY-MM-DD clinical dates


# --------------------------------------------------------------------------- Date Convention & Parsing

_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10, "oct": 10,
    "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_RX_DATE_NUM = re.compile(r"\b(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{2,4})\b")
_RX_DATE_MONTH = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)(?:\s+(\d{4}))?\b",
    re.I,
)


def detect_hospital_name(text: str) -> str:
    low = text.lower()
    if "sanjeevani" in low:
        return "sanjeevani"
    if "jeevan jyoti" in low or "jeevanjyoti" in low:
        return "jeevanjyoti"
    if "arogya" in low:
        return "arogya"
    if "district hospital" in low or "dhh" in low:
        return "dhhardoi"
    m = re.search(r"(?im)^([A-Z][A-Za-z\s]{3,40}(?:Hospital|Sadan|Clinic))\b", text)
    if m:
        return re.sub(r"\s+", "_", m.group(1).strip().lower())
    return ""


def infer_file_date_hints(files: list[IngestedFile]) -> tuple[dict[str, str], int]:
    """Infer per-file 'dmy' vs 'mdy' cleanly using column headers, unambiguous dates, and hospital consensus."""
    file_votes: dict[str, tuple[int, int]] = {}
    hosp_votes: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [dmy, mdy]
    file_hosp: dict[str, str] = {}
    year_counts: dict[int, int] = defaultdict(int)

    for inf in files:
        raw_sample = inf.text or inf.sample_text
        if inf.quarantine_reason or not raw_sample:
            continue
        txt = to_ascii_digits(raw_sample)
        hosp = detect_hospital_name(txt) or detect_hospital_name(inf.rel_path)
        file_hosp[inf.rel_path] = hosp

        dmy_v = mdy_v = 0
        if "mm/dd/yyyy" in txt.lower():
            mdy_v += 100
        elif "dd/mm/yyyy" in txt.lower() or "dd-mm-yyyy" in txt.lower():
            dmy_v += 100

        for m in _RX_DATE_NUM.finditer(txt):
            a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if len(m.group(1)) == 4 and 1900 <= a <= 2100:
                year_counts[a] += 1
                continue
            yr = c + 2000 if c < 100 else c
            if 1900 <= yr <= 2100:
                year_counts[yr] += 1
            if a > 12 and 1 <= b <= 12:
                dmy_v += 1
            elif b > 12 and 1 <= a <= 12:
                mdy_v += 1

        file_votes[inf.rel_path] = (dmy_v, mdy_v)
        if hosp and (dmy_v or mdy_v):
            hosp_votes[hosp][0] += dmy_v
            hosp_votes[hosp][1] += mdy_v

    modal_year = max(year_counts, key=lambda y: year_counts[y]) if year_counts else 2026
    hints: dict[str, str] = {}
    for inf in files:
        if inf.quarantine_reason:
            continue
        dmy_v, mdy_v = file_votes.get(inf.rel_path, (0, 0))
        if mdy_v > dmy_v:
            hints[inf.rel_path] = "mdy"
        elif dmy_v > mdy_v:
            hints[inf.rel_path] = "dmy"
        else:
            hosp = file_hosp.get(inf.rel_path, "")
            hd, hm = hosp_votes.get(hosp, [0, 0])
            hints[inf.rel_path] = "mdy" if hm > hd else "dmy"
    return hints, modal_year


def parse_date_flex(raw: str, hint: str = "dmy", default_year: int = 2026) -> tuple[dt.date | None, str]:
    """Parse ISO, DMY, MDY, or '5 September' and return (date, style_tag)."""
    s = to_ascii_digits(str(raw).strip())
    if not s:
        return None, ""
    # ISO timestamp or date
    m_iso_dt = re.match(r"^(\d{4})-(\d{2})-(\d{2})(T.*)$", s)
    if m_iso_dt:
        try:
            return dt.date(int(m_iso_dt.group(1)), int(m_iso_dt.group(2)), int(m_iso_dt.group(3))), "iso_dt:" + m_iso_dt.group(4)
        except ValueError:
            return None, ""
    m_iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m_iso:
        try:
            return dt.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))), "iso"
        except ValueError:
            return None, ""
    m_num = re.match(r"^(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})$", s)
    if m_num:
        a, sep, b, c = int(m_num.group(1)), m_num.group(2), int(m_num.group(3)), int(m_num.group(4))
        short_yr = len(m_num.group(4)) == 2
        yr = c + 2000 if short_yr else c
        if a > 12:
            day, mon = a, b
        elif b > 12:
            mon, day = a, b
        elif hint == "mdy":
            mon, day = a, b
        else:
            day, mon = a, b
        try:
            style = f"{'mdy' if hint == 'mdy' else 'dmy'}:{sep}:{'2' if short_yr else '4'}"
            return dt.date(yr, mon, day), style
        except ValueError:
            return None, ""
    m_mon = _RX_DATE_MONTH.fullmatch(s)
    if m_mon:
        day = int(m_mon.group(1))
        mon = _MONTH_NAMES[m_mon.group(2).lower()]
        yr = int(m_mon.group(3)) if m_mon.group(3) else default_year
        try:
            return dt.date(yr, mon, day), "month_text"
        except ValueError:
            return None, ""
    return None, ""


def norm_gender(val: str) -> str:
    v = to_ascii_digits(str(val).strip()).lower()
    if v in ("m", "male", "पुरुष", "purush"):
        return "M"
    if v in ("f", "female", "महिला", "mahila", "स्त्री"):
        return "F"
    if v in ("o", "other", "अन्य"):
        return "O"
    return ""


def parse_age_val(val: Any) -> tuple[int | None, str]:
    s = to_ascii_digits(str(val).strip())
    if not s:
        return None, ""
    m = re.match(r"^(-?\d{1,3})\s*(?:Yrs?|Y|वर्ष)?\s*(?:/\s*([MF]|पुरुष|महिला))?$", s, re.I)
    if m:
        age = int(m.group(1))
        gen = norm_gender(m.group(2)) if m.group(2) else ""
        if -5 <= age <= 125:
            return age, gen
    return None, ""


def norm_mrn_key(mrn: str) -> str:
    """Normalize MRN while keeping hospital/prefix distinction (SNJ26064127, JJH0076900, UHID010607, ADM000161)."""
    s = to_ascii_digits(mrn.strip()).upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


# --------------------------------------------------------------------------- Clue Extraction

_RX_ICD_ONLY = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b")


def extract_clues_from_file(
    inf: IngestedFile,
    detector: PIIDetector,
    date_hint: str,
    modal_year: int,
) -> list[RecordClues]:
    """Extract structured & free-text identity clues for every record unit in an ingested file."""
    rel = inf.rel_path
    if inf.quarantine_reason:
        return []

    is_clean_ref = rel.startswith("clean/") or rel == "misc/README_export.txt"
    is_non_patient_misc = rel in {
        "misc/ambulance_log.csv",
        "misc/helpdesk_tickets.json",
        "misc/scanned_consent_form.json",
        "misc/patient_index.json",
        "misc/ward_vitals_sheet.csv",
        "notes/visitor_log_ward5.txt",
    }

    clues_list: list[RecordClues] = []

    if inf.fmt in ("csv", "tsv"):
        rows: list[list[str]] = inf.data
        header = rows[0]
        body = rows[1:]
        col_roles = []
        for idx, h in enumerate(header):
            samples = [r[idx] for r in body[:25] if idx < len(r)]
            col_roles.append(detector.classify_column_header(h, samples))

        # Check if table actually has patient-bearing records
        has_patient_cols = any(r in ("PATIENT_NAME", "MRN", "AADHAAR", "ABHA", "PHONE") for r in col_roles)
        for r_idx, row in enumerate(body):
            uid = f"{rel}|row:{r_idx}"
            rc = RecordClues(
                unit_id=uid,
                file_rel=rel,
                loc=f"row:{r_idx}",
                fmt=inf.fmt,
                is_patient_record=(not is_clean_ref and not is_non_patient_misc and has_patient_cols),
                hospital=detect_hospital_name(rel),
            )
            if not rc.is_patient_record:
                clues_list.append(rc)
                continue

            first_nm, last_nm = "", ""
            for c_idx, cell in enumerate(row):
                if c_idx >= len(col_roles):
                    break
                role = col_roles[c_idx]
                h_low = header[c_idx].lower()
                val = cell.strip()
                if not val:
                    continue
                if role == "PATIENT_NAME":
                    if "first" in h_low:
                        first_nm = val
                    elif "last" in h_low:
                        last_nm = val
                    else:
                        rc.patient_names.append(val)
                elif role == "RELATIVE_NAME":
                    rc.relative_names.append(val)
                elif role == "MRN":
                    rc.mrns.add(norm_mrn_key(val))
                elif role == "AADHAAR":
                    if "x" in val.lower() or "*" in val:
                        m4 = re.search(r"(\d{4})$", to_ascii_digits(val))
                        if m4:
                            rc.aadhaar_last4.add(m4.group(1))
                    else:
                        d = re.sub(r"\D", "", to_ascii_digits(val))
                        if len(d) == 12:
                            rc.aadhaars.add(d)
                elif role == "ABHA":
                    d = re.sub(r"\D", "", to_ascii_digits(val))
                    if len(d) >= 12:
                        rc.abhas.add(d)
                elif role == "PAN":
                    rc.pans.add(val.upper())
                elif role == "INSURANCE_ID":
                    rc.insurance_ids.add(val.upper())
                elif role == "PHONE":
                    d = re.sub(r"\D", "", to_ascii_digits(val))
                    if len(d) >= 10:
                        rc.phones.add(d[-10:])
                elif role == "EMAIL":
                    rc.emails.add(val.lower())
                elif role == "DOB":
                    d, _ = parse_date_flex(val, hint=date_hint, default_year=modal_year)
                    if d:
                        rc.dob = d.isoformat()
                elif role == "AGE":
                    a, g = parse_age_val(val)
                    if a is not None:
                        rc.age = a
                    if g:
                        rc.gender = g
                elif role == "AGE_SEX":
                    a, g = parse_age_val(val)
                    if a is not None:
                        rc.age = a
                    if g:
                        rc.gender = g
                elif role == "GENDER":
                    g = norm_gender(val)
                    if g:
                        rc.gender = g
                elif role == "VILLAGE":
                    rc.village = val
                elif role == "DISTRICT":
                    rc.district = val
                elif role == "PIN":
                    d = re.sub(r"\D", "", to_ascii_digits(val))
                    if len(d) == 6:
                        rc.pin = d
                elif role == "DATE":
                    d, _ = parse_date_flex(val, hint=date_hint, default_year=modal_year)
                    if d:
                        rc.dates.append(d.isoformat())
                elif role == "OTHER" and any(k in h_low for k in ("procedure", "diagnosis", "icd")):
                    rc.diagnoses.add(val.lower())

            if first_nm or last_nm:
                full = f"{first_nm} {last_nm}".strip()
                if full:
                    rc.patient_names.insert(0, full)
            clues_list.append(rc)
        return clues_list

    if inf.fmt == "json":
        data = inf.data
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
        if is_doc or is_non_patient_misc:
            uid = f"{rel}|doc"
            rc = RecordClues(
                unit_id=uid,
                file_rel=rel,
                loc="doc",
                fmt="json",
                is_patient_record=(not is_clean_ref and not is_non_patient_misc and isinstance(data, dict) and "patient" in data),
                hospital=detect_hospital_name(str(data.get("lab", "")) if isinstance(data, dict) else ""),
            )
            if rc.is_patient_record and isinstance(data, dict):
                _extract_from_json_obj(data, rc, detector, date_hint, modal_year)
                # Filename phone/MRN fallback
                _add_filename_clues(rel, rc, detector)
            clues_list.append(rc)
            return clues_list

        # List-of-dicts JSON (opd, dhh_patients, bulk_lab_export)
        hosp = ""
        if isinstance(data.get("_export_meta"), dict):
            hosp = detect_hospital_name(str(data["_export_meta"].get("hospital", "")))
        elif isinstance(data.get("facility"), dict):
            hosp = detect_hospital_name(str(data["facility"].get("name", "")))

        for k, v in data.items():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                pkey = k.replace("~", "~0").replace("/", "~1")
                for i, item in enumerate(v):
                    loc = f"/{pkey}/{i}"
                    rc = RecordClues(
                        unit_id=f"{rel}|{loc}",
                        file_rel=rel,
                        loc=loc,
                        fmt="json",
                        is_patient_record=not is_clean_ref,
                        hospital=hosp or detect_hospital_name(rel),
                    )
                    _extract_from_json_obj(item, rc, detector, date_hint, modal_year)
                    clues_list.append(rc)
        return clues_list

    if inf.fmt == "fhir":
        bundle = inf.data
        rc = RecordClues(
            unit_id=f"{rel}|doc",
            file_rel=rel,
            loc="doc",
            fmt="fhir",
            is_patient_record=True,
        )
        for ent in bundle.get("entry", []):
            res = ent.get("resource", {})
            rt = res.get("resourceType")
            if rt == "Organization":
                rc.hospital = detect_hospital_name(str(res.get("name", "")))
            elif rt == "Patient":
                for ident in res.get("identifier", []):
                    val = str(ident.get("value", "")).strip()
                    sys_u = str(ident.get("system", "")).lower()
                    if not val:
                        continue
                    if "abdm" in sys_u or "healthid" in sys_u:
                        rc.abhas.add(re.sub(r"\D", "", val))
                    elif "aadhaar" in sys_u:
                        rc.aadhaars.add(re.sub(r"\D", "", val))
                    else:
                        rc.mrns.add(norm_mrn_key(val))
                for nm in res.get("name", []):
                    if nm.get("text"):
                        rc.patient_names.append(nm["text"])
                for tc in res.get("telecom", []):
                    if tc.get("system") == "phone" and tc.get("value"):
                        d = re.sub(r"\D", "", str(tc["value"]))
                        if len(d) >= 10:
                            rc.phones.add(d[-10:])
                    elif tc.get("system") == "email" and tc.get("value"):
                        rc.emails.add(str(tc["value"]).lower())
                if res.get("gender"):
                    rc.gender = norm_gender(res["gender"])
                if res.get("birthDate"):
                    d, _ = parse_date_flex(str(res["birthDate"]), hint="iso")
                    if d:
                        rc.dob = d.isoformat()
                for addr in res.get("address", []):
                    if addr.get("city"):
                        rc.village = str(addr["city"])
                    if addr.get("district"):
                        rc.district = str(addr["district"])
                    if addr.get("postalCode"):
                        rc.pin = str(addr["postalCode"])
                for cont in res.get("contact", []):
                    if isinstance(cont.get("name"), dict) and cont["name"].get("text"):
                        rc.relative_names.append(cont["name"]["text"])
                for ext in res.get("extension", []):
                    if ext.get("valueString"):
                        rc.relative_names.append(ext["valueString"])
                if res.get("text", {}).get("div"):
                    div_txt = re.sub(r"<[^>]+>", " ", res["text"]["div"])
                    m_age = re.search(r"\b(\d{1,3})\s*yrs?\b", div_txt, re.I)
                    if m_age:
                        rc.age = int(m_age.group(1))
            elif rt == "Encounter":
                for ident in res.get("identifier", []):
                    if ident.get("value"):
                        rc.mrns.add(norm_mrn_key(str(ident["value"])))
                per = res.get("period", {})
                for pkey in ("start", "end"):
                    if per.get(pkey):
                        d, _ = parse_date_flex(str(per[pkey]), hint="iso")
                        if d:
                            rc.dates.append(d.isoformat())
            elif rt == "Condition":
                for coding in res.get("code", {}).get("coding", []):
                    if coding.get("code"):
                        rc.diagnoses.add(str(coding["code"]).upper())
                    if coding.get("display"):
                        rc.diagnoses.add(str(coding["display"]).lower())
            elif rt == "Procedure":
                txt = res.get("code", {}).get("text")
                if txt:
                    rc.diagnoses.add(str(txt).lower())
        clues_list.append(rc)
        return clues_list

    # Text document
    txt = inf.text or ""
    rc = RecordClues(
        unit_id=f"{rel}|doc",
        file_rel=rel,
        loc="doc",
        fmt="text",
        is_patient_record=(not is_clean_ref and not is_non_patient_misc),
        hospital=detect_hospital_name(txt) or detect_hospital_name(rel),
    )
    if rc.is_patient_record:
        _extract_from_text(txt, rc, detector, date_hint, modal_year)
        _add_filename_clues(rel, rc, detector)
    clues_list.append(rc)
    return clues_list


def _add_filename_clues(rel: str, rc: RecordClues, detector: PIIDetector) -> None:
    stem = Path(rel).stem
    for s in detector.detect_text(stem.replace("_", " "), allow_pin_scan=False):
        if s.pii_type == "PHONE" and s.normalized:
            rc.phones.add(s.normalized)
        elif s.pii_type == "MRN" and s.normalized:
            rc.mrns.add(norm_mrn_key(s.normalized))
        elif s.pii_type == "PATIENT_NAME" and not rc.patient_names:
            rc.patient_names.append(s.surface)
    # Catch MRNs like JJH-0076928 or SNJ-26-064279 or adm000161 in stem
    for m in re.finditer(r"(?i)\b(?:SNJ|JJH|UHID|DHH|IPD|IP|ADM)[-_]?(?:\d{2}[-_]|20\d{2}[-_])?\d{4,8}\b", stem):
        rc.mrns.add(norm_mrn_key(m.group(0)))


def _extract_from_json_obj(
    obj: dict[str, Any],
    rc: RecordClues,
    detector: PIIDetector,
    date_hint: str,
    modal_year: int,
) -> None:
    def _walk(node: Any, path_parts: list[str]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, path_parts + [str(k).lower()])
            return
        if isinstance(node, list):
            for x in node:
                _walk(x, path_parts)
            return
        if node is None:
            return

        leaf_key = path_parts[-1] if path_parts else ""
        parent_key = path_parts[-2] if len(path_parts) >= 2 else ""
        sval = str(node).strip()
        if not sval:
            return

        if leaf_key in ("uhid", "patient_ref", "hospital_uhid", "reg_no", "mrn"):
            rc.mrns.add(norm_mrn_key(sval))
        elif leaf_key in ("aadhar_no", "aadhaar", "aadhaar_no"):
            d = re.sub(r"\D", "", to_ascii_digits(sval))
            if len(d) == 12:
                rc.aadhaars.add(d)
        elif leaf_key in ("abha", "abha_no", "abha_number"):
            d = re.sub(r"\D", "", to_ascii_digits(sval))
            if len(d) >= 12:
                rc.abhas.add(d)
        elif leaf_key in ("mobile", "phone", "callback"):
            d = re.sub(r"\D", "", to_ascii_digits(sval))
            if len(d) >= 10:
                rc.phones.add(d[-10:])
        elif leaf_key == "email":
            rc.emails.add(sval.lower())
        elif leaf_key == "dob":
            d, _ = parse_date_flex(sval, hint=date_hint, default_year=modal_year)
            if d:
                rc.dob = d.isoformat()
        elif leaf_key == "age":
            a, _ = parse_age_val(sval)
            if a is not None:
                rc.age = a
        elif leaf_key in ("gender", "sex"):
            g = norm_gender(sval)
            if g:
                rc.gender = g
        elif leaf_key == "age_sex":
            a, g = parse_age_val(sval)
            if a is not None:
                rc.age = a
            if g:
                rc.gender = g
        elif leaf_key in ("village", "city"):
            rc.village = sval
        elif leaf_key == "district":
            rc.district = sval
        elif leaf_key in ("pin", "postalcode"):
            d = re.sub(r"\D", "", to_ascii_digits(sval))
            if len(d) == 6:
                rc.pin = d
        elif leaf_key in ("visit_date", "collected", "date", "registered_on"):
            d, _ = parse_date_flex(sval, hint=date_hint, default_year=modal_year)
            if d:
                rc.dates.append(d.isoformat())
        elif leaf_key in ("patient_name",) or (leaf_key == "name" and parent_key == "patient"):
            rc.patient_names.append(sval)
        elif parent_key == "name" and leaf_key in ("first", "last", "local"):
            # dhh_patients has first, last, local
            if leaf_key == "local":
                rc.patient_names.append(sval)
            elif leaf_key == "first":
                last = str(obj.get("name", {}).get("last", "")).strip()
                rc.patient_names.append(f"{sval} {last}".strip())
        elif parent_key == "guardian" and leaf_key == "name":
            rc.relative_names.append(sval)
        elif parent_key == "diagnosis" and leaf_key in ("code", "display"):
            rc.diagnoses.add(sval.upper() if leaf_key == "code" else sval.lower())
        elif leaf_key in ("notes", "remarks", "address_note"):
            for sp in detector.detect_text(sval, allow_pin_scan=False):
                if sp.pii_type == "PATIENT_NAME":
                    rc.patient_names.append(sp.surface)
                elif sp.pii_type == "RELATIVE_NAME":
                    rc.relative_names.append(sp.surface)
                elif sp.pii_type == "PHONE" and sp.normalized:
                    rc.phones.add(sp.normalized)
                elif sp.pii_type == "MRN" and sp.normalized:
                    rc.mrns.add(norm_mrn_key(sp.normalized))
        elif leaf_key == "data":
            dec = try_decode_base64_text(sval)
            if dec:
                for sp in detector.detect_text(dec, allow_pin_scan=False):
                    if sp.pii_type == "PATIENT_NAME":
                        rc.patient_names.append(sp.surface)
                    elif sp.pii_type == "PHONE" and sp.normalized:
                        rc.phones.add(sp.normalized)
                    elif sp.pii_type == "MRN" and sp.normalized:
                        rc.mrns.add(norm_mrn_key(sp.normalized))

    _walk(obj, [])


def _extract_from_text(
    text: str,
    rc: RecordClues,
    detector: PIIDetector,
    date_hint: str,
    modal_year: int,
) -> None:
    spans = detector.detect_text(text, allow_pin_scan=True)
    for sp in spans:
        if sp.pii_type == "MRN" and sp.normalized:
            rc.mrns.add(norm_mrn_key(sp.normalized))
        elif sp.pii_type == "PHONE" and sp.normalized:
            rc.phones.add(sp.normalized)
        elif sp.pii_type == "ABHA_NUMBER" and sp.normalized:
            rc.abhas.add(sp.normalized)
        elif sp.pii_type == "AADHAAR" and sp.normalized:
            rc.aadhaars.add(sp.normalized)
        elif sp.pii_type == "EMAIL" and sp.normalized:
            rc.emails.add(sp.normalized)
        elif sp.pii_type == "PAN" and sp.normalized:
            rc.pans.add(sp.normalized)
        elif sp.pii_type == "PATIENT_NAME":
            rc.patient_names.append(sp.surface)
        elif sp.pii_type == "RELATIVE_NAME":
            rel_clean = re.sub(r"(?i)^\s*(?:[sdwfmc]/o|c/o)\s+", "", sp.surface).strip()
            rc.relative_names.append(rel_clean or sp.surface)
        elif sp.pii_type == "PIN" and sp.normalized:
            rc.pin = sp.normalized
        elif sp.pii_type == "VILLAGE":
            # Strip prefix label
            v = re.sub(r"(?i)^(?:vill\.?|village|gram|gaon|ग्राम|गाँव|गांव)\s*[:\-]?\s*", "", sp.surface).strip()
            if v:
                rc.village = v

    # Extract Age, Gender, DOB, District, Diagnoses, Dates from text lines
    norm = to_ascii_digits(text)
    m_age = re.search(
        r"(?i)(?:Age(?:/Sex)?|आयु)\s*:\s*(-?\d{1,3})\s*(?:Yrs?|वर्ष)?(?:\s*/\s*(M|F|Male|Female|पुरुष|महिला))?",
        norm,
    )
    if not m_age:
        m_age = re.search(r"(?i)\b(\d{1,3})\s*Y\s*/\s*([MF])\b", norm)
    if m_age:
        rc.age = int(m_age.group(1))
        if m_age.group(2):
            rc.gender = norm_gender(m_age.group(2))

    if not rc.gender:
        if re.search(r"\bW/o\b|पति\s*का\s*नाम", text, re.I):
            rc.gender = "F"

    m_dob = re.search(r"(?i)\bDOB\s*:\s*([^\n,;]+)", norm)
    if m_dob:
        d, _ = parse_date_flex(m_dob.group(1), hint=date_hint, default_year=modal_year)
        if d:
            rc.dob = d.isoformat()

    m_dist = re.search(r"(?i)(?:\bDistt?\.?|\bDistrict|जिला)\s*[:\-]?\s*([A-Za-z\u0900-\u097f ]{3,25}?)(?=\s*[-–,\n]|$)", norm)
    if m_dist:
        rc.district = m_dist.group(1).strip()

    for m_icd in _RX_ICD_ONLY.finditer(norm):
        rc.diagnoses.add(m_icd.group(1).upper())

    for m_d in re.finditer(
        r"(?i)(?:Date(?:\s+of\s+(?:Admission|Discharge))?|Visit|तारीख|भर्ती\s+की\s+तारीख|छुट्टी\s+की\s+तारीख)\s*:\s*([^\n]+)",
        norm,
    ):
        d, _ = parse_date_flex(m_d.group(1).strip(), hint=date_hint, default_year=modal_year)
        if d:
            rc.dates.append(d.isoformat())

    m_wa = re.match(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),", norm)
    if m_wa:
        d, _ = parse_date_flex(m_wa.group(1), hint="dmy", default_year=modal_year)
        if d:
            rc.dates.append(d.isoformat())


# --------------------------------------------------------------------------- Conservative Clustering

@dataclass
class PatientCluster:
    cluster_id: str
    members: list[RecordClues] = field(default_factory=list)
    mrns: set[str] = field(default_factory=set)
    aadhaars: set[str] = field(default_factory=set)
    aadhaar_last4: set[str] = field(default_factory=set)
    abhas: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    pans: set[str] = field(default_factory=set)
    insurance_ids: set[str] = field(default_factory=set)
    dobs: set[str] = field(default_factory=set)
    ages: list[int] = field(default_factory=list)
    genders: set[str] = field(default_factory=set)
    villages: list[str] = field(default_factory=list)
    districts: list[str] = field(default_factory=list)
    pins: set[str] = field(default_factory=set)
    patient_names: list[str] = field(default_factory=list)
    relative_names: list[str] = field(default_factory=list)
    diagnoses: set[str] = field(default_factory=set)

    def add(self, rc: RecordClues) -> None:
        self.members.append(rc)
        self.mrns |= rc.mrns
        self.aadhaars |= rc.aadhaars
        self.aadhaar_last4 |= rc.aadhaar_last4
        for a in rc.aadhaars:
            self.aadhaar_last4.add(a[-4:])
        self.abhas |= rc.abhas
        self.phones |= rc.phones
        self.emails |= rc.emails
        self.pans |= rc.pans
        self.insurance_ids |= rc.insurance_ids
        if rc.dob:
            self.dobs.add(rc.dob)
        if rc.age is not None:
            self.ages.append(rc.age)
        if rc.gender:
            self.genders.add(rc.gender)
        if rc.village:
            self.villages.append(rc.village)
        if rc.district:
            self.districts.append(rc.district)
        if rc.pin:
            self.pins.add(rc.pin)
        self.patient_names.extend(rc.patient_names)
        self.relative_names.extend(rc.relative_names)
        self.diagnoses |= rc.diagnoses

    def merge_from(self, other: PatientCluster) -> None:
        for m in other.members:
            self.add(m)

    def canonical_key(self) -> str:
        """Deterministic canonical identifier sorted from strongest immutable cluster tokens."""
        for s in (self.aadhaars, self.abhas, self.mrns, self.phones, self.pans, self.emails):
            if s:
                return f"id:{sorted(s)[0]}"
        nm = sorted(canonicalize_person_name(x) for x in self.patient_names if x)
        dob = sorted(self.dobs)[0] if self.dobs else ""
        return f"bio:{nm[0] if nm else self.cluster_id}|{dob}|{sorted(u.unit_id for u in self.members)[0]}"


def _has_hard_conflict(
    c1: PatientCluster,
    c2: PatientCluster,
    max_age_diff: int = 2,
    strict_place: bool = True,
) -> bool:
    """Strictly forbid merging clusters with any incompatible demographic or ID signal."""
    if c1.genders and c2.genders and c1.genders.isdisjoint(c2.genders):
        return True
    if c1.dobs and c2.dobs and c1.dobs.isdisjoint(c2.dobs):
        return True
    if c1.aadhaars and c2.aadhaars and c1.aadhaars.isdisjoint(c2.aadhaars):
        return True
    if c1.aadhaar_last4 and c2.aadhaar_last4 and c1.aadhaar_last4.isdisjoint(c2.aadhaar_last4):
        return True
    if c1.abhas and c2.abhas and c1.abhas.isdisjoint(c2.abhas):
        return True
    if c1.pans and c2.pans and c1.pans.isdisjoint(c2.pans):
        return True
    if c1.ages and c2.ages:
        if min(abs(a1 - a2) for a1 in c1.ages for a2 in c2.ages) > max_age_diff:
            return True

    # Full-name compatibility check (only if both have multi-token names)
    full1 = [n for n in c1.patient_names if len(canonicalize_person_name(n)) >= 2]
    full2 = [n for n in c2.patient_names if len(canonicalize_person_name(n)) >= 2]
    if full1 and full2:
        best_sim = max(names_similarity(a, b) for a in full1 for b in full2)
        if best_sim < 0.72:
            return True

    # Relative compatibility check
    rel1 = [n for n in c1.relative_names if canonicalize_person_name(n)]
    rel2 = [n for n in c2.relative_names if canonicalize_person_name(n)]
    if rel1 and rel2:
        best_rel = max(names_similarity(a, b) for a in rel1 for b in rel2)
        if best_rel < 0.70:
            first_match = max(
                names_similarity(canonicalize_person_name(a)[0], canonicalize_person_name(b)[0])
                for a in rel1 for b in rel2
                if canonicalize_person_name(a) and canonicalize_person_name(b)
            )
            if first_match < 0.80:
                return True

    # Village & District cross-script check (fuzzy Indic transliteration-aware)
    if strict_place:
        if c1.villages and c2.villages:
            best_v = max(places_similarity(v1, v2) for v1 in c1.villages for v2 in c2.villages)
            if best_v < 0.75:
                return True
        if c1.districts and c2.districts:
            best_d = max(places_similarity(d1, d2) for d1 in c1.districts for d2 in c2.districts)
            if best_d < 0.75:
                return True
    return False


def _best_full_name_sim(c1: PatientCluster, c2: PatientCluster) -> float:
    n1 = [n for n in c1.patient_names if len(canonicalize_person_name(n)) >= 2]
    n2 = [n for n in c2.patient_names if len(canonicalize_person_name(n)) >= 2]
    if not n1 or not n2:
        return 0.0
    return max(names_similarity(a, b) for a in n1 for b in n2)


def _best_given_name_sim(c1: PatientCluster, c2: PatientCluster) -> float:
    g1 = [canonicalize_person_name(n)[0] for n in c1.patient_names if canonicalize_person_name(n)]
    g2 = [canonicalize_person_name(n)[0] for n in c2.patient_names if canonicalize_person_name(n)]
    if not g1 or not g2:
        return 0.0
    from rapidfuzz import fuzz
    return max(fuzz.ratio(a, b) / 100.0 for a in g1 for b in g2)


def _best_rel_sim(c1: PatientCluster, c2: PatientCluster) -> float:
    r1 = [n for n in c1.relative_names if canonicalize_person_name(n)]
    r2 = [n for n in c2.relative_names if canonicalize_person_name(n)]
    if not r1 or not r2:
        return 0.0
    from rapidfuzz import fuzz
    best = 0.0
    for a in r1:
        for b in r2:
            best = max(best, names_similarity(a, b))
            ta, tb = canonicalize_person_name(a), canonicalize_person_name(b)
            if ta and tb and (len(ta) == 1 or len(tb) == 1):
                best = max(best, fuzz.ratio(ta[0], tb[0]) / 100.0)
    return best


def _best_village_sim(c1: PatientCluster, c2: PatientCluster) -> float:
    if not c1.villages or not c2.villages:
        return 0.0
    return max(places_similarity(v1, v2) for v1 in c1.villages for v2 in c2.villages)


def link_all_records(
    all_clues: list[RecordClues],
    keyring: KeyRing,
    policy: dict[str, Any],
) -> tuple[dict[str, str | None], dict[str, PatientCluster], dict[str, str]]:
    """Link records across files/languages/hospitals conservatively and issue keyed HMAC pseudonyms."""
    name_th = float(policy.get("linkage_name_threshold", 0.85))
    rel_th = float(policy.get("linkage_relative_threshold", 0.82))
    vil_th = float(policy.get("linkage_village_threshold", 0.82))
    max_age_diff = int(policy.get("linkage_max_age_diff", 2))

    # Initialize singleton clusters for patient-bearing units in deterministic order
    clusters: dict[str, PatientCluster] = {}
    unit_to_cid: dict[str, str] = {}
    for idx, rc in enumerate(all_clues):
        if not rc.is_patient_record:
            continue
        cid = f"C{idx:05d}"
        cl = PatientCluster(cluster_id=cid)
        cl.add(rc)
        clusters[cid] = cl
        unit_to_cid[rc.unit_id] = cid

    def _merge(cid_keep: str, cid_drop: str) -> None:
        if cid_keep == cid_drop or cid_drop not in clusters or cid_keep not in clusters:
            return
        keep, drop = clusters[cid_keep], clusters[cid_drop]
        keep.merge_from(drop)
        for m in drop.members:
            unit_to_cid[m.unit_id] = cid_keep
        del clusters[cid_drop]

    # Stage 1: Exact Strong-ID merges (Aadhaar, ABHA, PAN, Email, Insurance, MRN, Phone with conflict guard)
    id_getters = [
        ("aadhaar", lambda c: c.aadhaars),
        ("abha", lambda c: c.abhas),
        ("pan", lambda c: c.pans),
        ("email", lambda c: c.emails),
        ("insurance", lambda c: c.insurance_ids),
        ("mrn", lambda c: c.mrns),
        ("phone", lambda c: c.phones),
    ]
    for kind, getter in id_getters:
        changed = True
        while changed:
            changed = False
            buckets: dict[str, list[str]] = defaultdict(list)
            for cid in sorted(clusters):
                for key in sorted(getter(clusters[cid])):
                    if key:
                        buckets[key].append(cid)
            for key in sorted(buckets):
                cids = buckets[key]
                if len(cids) < 2:
                    continue
                anchor = cids[0]
                for other in cids[1:]:
                    if anchor not in clusters or other not in clusters:
                        continue
                    # Allow up to 4-year age drift between registration (e.g. 2024) and clinical visits (2026) when MRN/ID + name align
                    eff_age_diff = 4 if kind in ("mrn", "aadhaar", "abha", "pan", "email") else max_age_diff
                    if _has_hard_conflict(clusters[anchor], clusters[other], eff_age_diff, strict_place=(kind == "phone")):
                        continue
                    # For phone-only merge, require compatible names if both have names
                    if kind == "phone":
                        g_sim = _best_given_name_sim(clusters[anchor], clusters[other])
                        if clusters[anchor].patient_names and clusters[other].patient_names and g_sim < 0.65:
                            continue
                    _merge(anchor, other)
                    changed = True

    # Stage 2: Cross-hospital / Cross-script / Bio+Relative+Village merges
    # Use inverted blocking indexes (DOB, (name_skel, rel_skel), (name_skel, vil_skel)) for O(N) scaling
    _generic_skels = {_consonant_skeleton(g) for g in ("devi", "kumari", "kumar", "lal", "singh", "bano", "begum", "khatun", "kaur")}

    def _cluster_block_keys(cl: PatientCluster) -> tuple[set[str], set[str], set[str]]:
        n_keys: set[str] = set()
        for nm in cl.patient_names:
            toks = canonicalize_person_name(nm)
            sks = [_consonant_skeleton(tok) for tok in toks if tok]
            non_gen = [s for s in sks if s not in _generic_skels]
            for s in (non_gen or sks):
                if s:
                    n_keys.add(s)
        r_keys: set[str] = set()
        for rl in cl.relative_names:
            toks = canonicalize_person_name(rl)
            sks = [_consonant_skeleton(tok) for tok in toks if tok]
            non_gen = [s for s in sks if s not in _generic_skels]
            for s in (non_gen or sks):
                if s:
                    r_keys.add(s)
        v_keys: set[str] = set()
        for vl in cl.villages:
            for tok in canonicalize_place(vl).split():
                sk = _consonant_skeleton(tok)
                if sk:
                    v_keys.add(sk)
        return n_keys, r_keys, v_keys

    for _pass in range(3):
        cids_sorted = sorted(clusters.keys())
        dob_index: dict[str, list[str]] = defaultdict(list)
        name_rel_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        name_rel_no_aadhaar: dict[tuple[str, str], list[str]] = defaultdict(list)
        name_vil_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        name_vil_no_aadhaar: dict[tuple[str, str], list[str]] = defaultdict(list)
        cluster_keys: dict[str, tuple[set[str], set[str], set[str]]] = {}
        has_aadhaar_cids: set[str] = set()
        has_abha_cids: set[str] = set()
        has_pan_cids: set[str] = set()
        gender_cids: dict[str, set[str]] = defaultdict(set)

        for cid in cids_sorted:
            cl = clusters[cid]
            has_aad = bool(cl.aadhaars)
            if has_aad:
                has_aadhaar_cids.add(cid)
            if cl.abhas:
                has_abha_cids.add(cid)
            if cl.pans:
                has_pan_cids.add(cid)
            for g in cl.genders:
                gender_cids[g].add(cid)
            for d in cl.dobs:
                dob_index[d].append(cid)
            nk, rk, vk = _cluster_block_keys(cl)
            cluster_keys[cid] = (nk, rk, vk)
            for n1 in nk:
                for r1 in rk:
                    name_rel_index[(n1, r1)].append(cid)
                    if not has_aad:
                        name_rel_no_aadhaar[(n1, r1)].append(cid)
                for v1 in vk:
                    name_vil_index[(n1, v1)].append(cid)
                    if not has_aad:
                        name_vil_no_aadhaar[(n1, v1)].append(cid)

        merged_this_pass = False
        for cid_a in cids_sorted:
            if cid_a not in clusters:
                continue
            ca = clusters[cid_a]
            nk, rk, vk = cluster_keys[cid_a]
            cand_cids: set[str] = set()
            for d in ca.dobs:
                cand_cids.update(dob_index[d])
            nr_src = name_rel_no_aadhaar if ca.aadhaars else name_rel_index
            nv_src = name_vil_no_aadhaar if ca.aadhaars else name_vil_index
            for n1 in nk:
                for r1 in rk:
                    if (n1, r1) in nr_src:
                        cand_cids.update(nr_src[(n1, r1)])
                for v1 in vk:
                    if (n1, v1) in nv_src:
                        cand_cids.update(nv_src[(n1, v1)])
            cand_cids.discard(cid_a)
            if not cand_cids:
                continue
            # Fast conflict pruning: Stage 1 already merged identical aadhaar/abha/pan,
            # so two clusters that both have aadhaar (or abha, or pan) cannot merge in Stage 2.
            if ca.aadhaars:
                cand_cids.difference_update(has_aadhaar_cids)
            if ca.abhas:
                cand_cids.difference_update(has_abha_cids)
            if ca.pans:
                cand_cids.difference_update(has_pan_cids)
            if len(ca.genders) == 1:
                g_a = next(iter(ca.genders))
                for g_other, g_set in gender_cids.items():
                    if g_other != g_a:
                        cand_cids.difference_update(g_set)

            candidates: list[tuple[float, str]] = []
            tie_20 = 0
            tie_25 = 0
            for cid_b in sorted(cand_cids):
                if cid_b not in clusters:
                    continue
                cb = clusters[cid_b]
                if _has_hard_conflict(ca, cb, max_age_diff):
                    continue

                f_sim = _best_full_name_sim(ca, cb)
                g_sim = _best_given_name_sim(ca, cb)
                r_sim = _best_rel_sim(ca, cb)
                v_sim = _best_village_sim(ca, cb)
                dob_match = bool(ca.dobs and cb.dobs and not ca.dobs.isdisjoint(cb.dobs))

                score = 0.0
                if dob_match and (f_sim >= 0.80 or (g_sim >= 0.85 and r_sim >= 0.78)):
                    score = 3.0 + f_sim
                elif f_sim >= name_th and r_sim >= rel_th:
                    score = 2.5 + f_sim + r_sim
                elif f_sim >= name_th and v_sim >= vil_th:
                    score = 2.0 + f_sim + v_sim
                elif g_sim >= 0.86 and r_sim >= rel_th:
                    # Fragmented identity: e.g. "Asha, W/o Kamal" matching full cluster
                    score = 1.5 + g_sim + r_sim

                if score > 0.0:
                    candidates.append((score, cid_b))
                    if score >= 2.5:
                        tie_25 += 1
                    elif score >= 2.0:
                        tie_20 += 1
                    # Early exit: if no DOB and no relative match is possible and >=2 ambiguous name+village ties found,
                    # line 997 will unconditionally reject merging cid_a.
                    if not ca.dobs and not ca.relative_names and tie_25 == 0 and tie_20 >= 2:
                        break

            if not candidates:
                continue
            candidates.sort(reverse=True)
            top_score, top_cid = candidates[0]
            # Anti-over-merge guard: if fragmented (score < 2.0) and multiple distinct clusters tie/compete, DO NOT merge!
            if top_score < 2.0 and len(candidates) > 1:
                if candidates[1][0] >= 1.5:
                    continue
            # Also if full-name + village only (no DOB/relative) and two clusters both match, skip
            if 2.0 <= top_score < 2.5 and len(candidates) > 1 and candidates[1][0] >= 2.0:
                continue

            # Merge smaller cluster into canonical anchor
            keep, drop = (cid_a, top_cid) if cid_a < top_cid else (top_cid, cid_a)
            _merge(keep, drop)
            merged_this_pass = True
        if not merged_this_pass:
            break

    # Issue deterministic keyed HMAC pseudonyms sorted by canonical_key
    ordered_cids = sorted(clusters.keys(), key=lambda c: clusters[c].canonical_key())
    cid_to_psn: dict[str, str] = {}
    psn_clusters: dict[str, PatientCluster] = {}
    for cid in ordered_cids:
        cl = clusters[cid]
        # Require at least one concrete identity clue (MRN/Aadhaar/ABHA/Phone/Name) to link
        if not (cl.mrns or cl.aadhaars or cl.abhas or cl.phones or cl.patient_names):
            continue
        psn = keyring.pseudonym(cl.canonical_key())
        cid_to_psn[cid] = psn
        psn_clusters[psn] = cl

    unit_to_psn: dict[str, str | None] = {}
    for rc in all_clues:
        if not rc.is_patient_record:
            unit_to_psn[rc.unit_id] = None
        else:
            cid = unit_to_cid.get(rc.unit_id)
            unit_to_psn[rc.unit_id] = cid_to_psn.get(cid) if cid else None

    # Build MRN/Name lookup for cross-reference tables (patient_index.json, ward_vitals_sheet.csv)
    ref_lookup: dict[str, str] = {}
    for psn, cl in psn_clusters.items():
        for m in cl.mrns:
            ref_lookup[m] = psn
        for nm in cl.patient_names:
            c_toks = canonicalize_person_name(nm)
            if len(c_toks) >= 2:
                ref_lookup[" ".join(c_toks)] = psn

    return unit_to_psn, psn_clusters, ref_lookup
