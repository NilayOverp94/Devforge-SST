"""Cross-script, cross-hospital record linkage with conflict guards and keyed HMAC pseudonyms."""
from __future__ import annotations

import datetime as dt
import gc
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .detection import (
    PIIDetector,
    _canonicalize_person_name_cached,
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

_EMPTY_FS: frozenset[str] = frozenset()
_EMPTY_TUP: tuple[str, ...] = ()
_EMPTY_MUT_MARKER = ""


def _pack_set_field(val: set[str] | frozenset[str] | tuple[str, ...] | str) -> str | tuple[str, ...]:
    if type(val) is str:
        return val
    if type(val) is tuple:
        if not val:
            return ""
        return val[0] if len(val) == 1 else val
    n = len(val)
    if n == 0:
        return ""
    if n == 1:
        return next(iter(val))
    return tuple(sorted(val))


def _pack_list_field(val: list[str] | tuple[str, ...] | str) -> str | tuple[str, ...]:
    if type(val) is str:
        return val
    if type(val) is tuple:
        if not val:
            return ""
        return val[0] if len(val) == 1 else val
    n = len(val)
    if n == 0:
        return ""
    if n == 1:
        return val[0]
    dedup = tuple(dict.fromkeys(val))
    return dedup[0] if len(dedup) == 1 else dedup


def _as_tup(val: Any) -> tuple[str, ...]:
    if not val:
        return _EMPTY_TUP
    tv = type(val)
    if tv is str:
        return (val,)
    if tv is tuple:
        return val
    return tuple(val)


class RecordClues:
    """Compact memory-optimized identity & clinical clue record."""

    __slots__ = (
        "file_rel", "loc", "fmt", "is_patient_record", "hospital",
        "_mrns", "_aadhaars", "_aadhaar_last4", "_abhas", "_phones",
        "_emails", "_pans", "_insurance_ids", "_patient_names", "_relative_names",
        "dob", "age", "gender", "village", "district", "pin", "_diagnoses",
    )

    def __init__(
        self,
        unit_id: str = "",
        file_rel: str = "",
        loc: str = "",
        fmt: str = "",
        is_patient_record: bool = True,
        hospital: str = "",
        mrns: Any = None,
        aadhaars: Any = None,
        aadhaar_last4: Any = None,
        abhas: Any = None,
        phones: Any = None,
        emails: Any = None,
        pans: Any = None,
        insurance_ids: Any = None,
        patient_names: Any = None,
        relative_names: Any = None,
        dob: str = "",
        age: int | None = None,
        gender: str = "",
        village: str = "",
        district: str = "",
        pin: str = "",
        diagnoses: Any = None,
    ) -> None:
        self.file_rel = file_rel
        self.loc = loc
        self.fmt = fmt
        self.is_patient_record = is_patient_record
        self.hospital = hospital
        self._mrns: Any = set() if mrns is None else mrns
        self._aadhaars: Any = set() if aadhaars is None else aadhaars
        self._aadhaar_last4: Any = set() if aadhaar_last4 is None else aadhaar_last4
        self._abhas: Any = set() if abhas is None else abhas
        self._phones: Any = set() if phones is None else phones
        self._emails: Any = set() if emails is None else emails
        self._pans: Any = set() if pans is None else pans
        self._insurance_ids: Any = set() if insurance_ids is None else insurance_ids
        self._patient_names: Any = [] if patient_names is None else patient_names
        self._relative_names: Any = [] if relative_names is None else relative_names
        self.dob = dob
        self.age = age
        self.gender = gender
        self.village = village
        self.district = district
        self.pin = pin
        self._diagnoses: Any = set() if diagnoses is None else diagnoses

    @classmethod
    def _from_compact(
        cls,
        file_rel: str,
        loc: str,
        fmt: str,
        is_patient_record: bool,
        hospital: str,
        mrns: str | tuple[str, ...],
        aadhaars: str | tuple[str, ...],
        aadhaar_last4: str | tuple[str, ...],
        abhas: str | tuple[str, ...],
        phones: str | tuple[str, ...],
        emails: str | tuple[str, ...],
        pans: str | tuple[str, ...],
        insurance_ids: str | tuple[str, ...],
        patient_names: str | tuple[str, ...],
        relative_names: str | tuple[str, ...],
        dob: str,
        age: int | None,
        gender: str,
        village: str,
        district: str,
        pin: str,
        diagnoses: str | tuple[str, ...],
    ) -> RecordClues:
        obj = cls.__new__(cls)
        obj.file_rel = file_rel
        obj.loc = loc
        obj.fmt = fmt
        obj.is_patient_record = is_patient_record
        obj.hospital = hospital
        obj._mrns = mrns
        obj._aadhaars = aadhaars
        obj._aadhaar_last4 = aadhaar_last4
        obj._abhas = abhas
        obj._phones = phones
        obj._emails = emails
        obj._pans = pans
        obj._insurance_ids = insurance_ids
        obj._patient_names = patient_names
        obj._relative_names = relative_names
        obj.dob = dob
        obj.age = age
        obj.gender = gender
        obj.village = village
        obj.district = district
        obj.pin = pin
        obj._diagnoses = diagnoses
        return obj

    @property
    def unit_id(self) -> str:
        return f"{self.file_rel}|{self.loc}"

    @property
    def mrns(self) -> Any:
        v = self._mrns
        return v if type(v) is set else _as_tup(v)

    @mrns.setter
    def mrns(self, v: Any) -> None:
        self._mrns = v

    @property
    def aadhaars(self) -> Any:
        v = self._aadhaars
        return v if type(v) is set else _as_tup(v)

    @aadhaars.setter
    def aadhaars(self, v: Any) -> None:
        self._aadhaars = v

    @property
    def aadhaar_last4(self) -> Any:
        v = self._aadhaar_last4
        return v if type(v) is set else _as_tup(v)

    @aadhaar_last4.setter
    def aadhaar_last4(self, v: Any) -> None:
        self._aadhaar_last4 = v

    @property
    def abhas(self) -> Any:
        v = self._abhas
        return v if type(v) is set else _as_tup(v)

    @abhas.setter
    def abhas(self, v: Any) -> None:
        self._abhas = v

    @property
    def phones(self) -> Any:
        v = self._phones
        return v if type(v) is set else _as_tup(v)

    @phones.setter
    def phones(self, v: Any) -> None:
        self._phones = v

    @property
    def emails(self) -> Any:
        v = self._emails
        return v if type(v) is set else _as_tup(v)

    @emails.setter
    def emails(self, v: Any) -> None:
        self._emails = v

    @property
    def pans(self) -> Any:
        v = self._pans
        return v if type(v) is set else _as_tup(v)

    @pans.setter
    def pans(self, v: Any) -> None:
        self._pans = v

    @property
    def insurance_ids(self) -> Any:
        v = self._insurance_ids
        return v if type(v) is set else _as_tup(v)

    @insurance_ids.setter
    def insurance_ids(self, v: Any) -> None:
        self._insurance_ids = v

    @property
    def patient_names(self) -> Any:
        v = self._patient_names
        return v if type(v) is list else _as_tup(v)

    @patient_names.setter
    def patient_names(self, v: Any) -> None:
        self._patient_names = v

    @property
    def relative_names(self) -> Any:
        v = self._relative_names
        return v if type(v) is list else _as_tup(v)

    @relative_names.setter
    def relative_names(self, v: Any) -> None:
        self._relative_names = v

    @property
    def diagnoses(self) -> Any:
        v = self._diagnoses
        return v if type(v) is set else _as_tup(v)

    @diagnoses.setter
    def diagnoses(self, v: Any) -> None:
        self._diagnoses = v

    def compact(self) -> RecordClues:
        """Pack singletons to bare str and empties to '' so 5M-10M records fit well inside < 2 GB."""
        self._mrns = _pack_set_field(self._mrns)
        self._aadhaars = _pack_set_field(self._aadhaars)
        self._aadhaar_last4 = _pack_set_field(self._aadhaar_last4)
        self._abhas = _pack_set_field(self._abhas)
        self._phones = _pack_set_field(self._phones)
        self._emails = _pack_set_field(self._emails)
        self._pans = _pack_set_field(self._pans)
        self._insurance_ids = _pack_set_field(self._insurance_ids)
        self._patient_names = _pack_list_field(self._patient_names)
        self._relative_names = _pack_list_field(self._relative_names)
        self._diagnoses = _pack_set_field(self._diagnoses)
        return self


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
    samples = [
        (inf.rel_path, inf.text or inf.sample_text or "")
        for inf in files
        if not inf.quarantine_reason and (inf.text or inf.sample_text)
    ]
    return infer_file_date_hints_from_samples(samples)


def infer_file_date_hints_from_samples(samples: list[tuple[str, str]]) -> tuple[dict[str, str], int]:
    """Shared streaming-friendly date-convention inference from (rel_path, sample_text) pairs."""
    file_votes: dict[str, tuple[int, int]] = {}
    hosp_votes: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [dmy, mdy]
    file_hosp: dict[str, str] = {}
    year_counts: dict[int, int] = defaultdict(int)

    for rel_path, raw_sample in samples:
        if not raw_sample:
            continue
        txt = to_ascii_digits(raw_sample)
        hosp = detect_hospital_name(txt) or detect_hospital_name(rel_path)
        file_hosp[rel_path] = hosp

        dmy_v = mdy_v = 0
        txt_low = txt.lower()
        if "mm/dd/yyyy" in txt_low:
            mdy_v += 100
        elif "dd/mm/yyyy" in txt_low or "dd-mm-yyyy" in txt_low:
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

        file_votes[rel_path] = (dmy_v, mdy_v)
        if hosp and (dmy_v or mdy_v):
            hosp_votes[hosp][0] += dmy_v
            hosp_votes[hosp][1] += mdy_v

    modal_year = max(year_counts, key=lambda y: year_counts[y]) if year_counts else 2026
    hints: dict[str, str] = {}
    for rel_path, _raw_sample in samples:
        dmy_v, mdy_v = file_votes.get(rel_path, (0, 0))
        if mdy_v > dmy_v:
            hints[rel_path] = "mdy"
        elif dmy_v > mdy_v:
            hints[rel_path] = "dmy"
        else:
            hosp = file_hosp.get(rel_path, "")
            hd, hm = hosp_votes.get(hosp, [0, 0])
            hints[rel_path] = "mdy" if hm > hd else "dmy"
    return hints, modal_year


_RX_ISO_DT = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(T.*)$")
_RX_ISO_D = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_RX_NUM_D = re.compile(r"^(\d{1,2})([/.\-])(\d{1,2})\2(\d{2,4})$")
_RX_AGE_V = re.compile(r"^(-?\d{1,3})\s*(?:Yrs?|Y|वर्ष)?\s*(?:/\s*([MF]|पुरुष|महिला))?$", re.I)
_RX_LAST4 = re.compile(r"(\d{4})$")
_GENDER_FAST = {"M": "M", "F": "F", "O": "O", "m": "M", "f": "F", "o": "O", "male": "M", "female": "F", "other": "O"}


def _ascii_strip_digits(val: str) -> str:
    s = val if val.isascii() else to_ascii_digits(val)
    if s.isdigit():
        return s
    return _RX_NON_DIGITS.sub("", s)


def parse_date_flex(raw: str, hint: str = "dmy", default_year: int = 2026) -> tuple[dt.date | None, str]:
    """Parse ISO, DMY, MDY, or '5 September' and return (date, style_tag)."""
    s0 = raw.strip() if isinstance(raw, str) else str(raw).strip()
    if not s0:
        return None, ""
    s = s0 if s0.isascii() else to_ascii_digits(s0)
    if len(s) == 10 and s[4] == "-" and s[7] == "-" and s[:4].isdigit() and s[5:7].isdigit() and s[8:].isdigit():
        try:
            return dt.date.fromisoformat(s), "iso"
        except ValueError:
            return None, ""
    if "T" in s:
        m_iso_dt = _RX_ISO_DT.match(s)
        if m_iso_dt:
            try:
                return dt.date(int(m_iso_dt.group(1)), int(m_iso_dt.group(2)), int(m_iso_dt.group(3))), "iso_dt:" + m_iso_dt.group(4)
            except ValueError:
                return None, ""
    m_iso = _RX_ISO_D.match(s)
    if m_iso:
        try:
            return dt.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))), "iso"
        except ValueError:
            return None, ""
    m_num = _RX_NUM_D.match(s)
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
    g = _GENDER_FAST.get(val)
    if g is not None:
        return g
    s0 = val.strip() if isinstance(val, str) else str(val).strip()
    v = (s0 if s0.isascii() else to_ascii_digits(s0)).lower()
    if v in ("m", "male", "पुरुष", "purush"):
        return "M"
    if v in ("f", "female", "महिला", "mahila", "स्त्री"):
        return "F"
    if v in ("o", "other", "अन्य"):
        return "O"
    return ""


def parse_age_val(val: Any) -> tuple[int | None, str]:
    s0 = val.strip() if isinstance(val, str) else str(val).strip()
    if not s0:
        return None, ""
    s = s0 if s0.isascii() else to_ascii_digits(s0)
    if len(s) <= 3 and s.isdigit():
        a = int(s)
        return (a, "") if -5 <= a <= 125 else (None, "")
    m = _RX_AGE_V.match(s)
    if m:
        age = int(m.group(1))
        gen = norm_gender(m.group(2)) if m.group(2) else ""
        if -5 <= age <= 125:
            return age, gen
    return None, ""


_RX_NON_ALNUM_UPPER = re.compile(r"[^A-Z0-9]")
_RX_NON_DIGITS = re.compile(r"\D")


def norm_mrn_key(mrn: str) -> str:
    """Normalize MRN while keeping hospital/prefix distinction (SNJ26064127, JJH0076900, UHID010607, ADM000161)."""
    s0 = mrn.strip()
    if s0.isascii():
        su = s0.upper()
        if su.isalnum():
            return su
        if "-" in su and " " not in su and "/" not in su:
            c = su.replace("-", "")
            if c.isalnum():
                return c
        return _RX_NON_ALNUM_UPPER.sub("", su)
    return _RX_NON_ALNUM_UPPER.sub("", to_ascii_digits(s0).upper())


# --------------------------------------------------------------------------- Clue Extraction

_RX_ICD_ONLY = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b")


def _add_packed_str(cur: str | tuple[str, ...], val: str) -> str | tuple[str, ...]:
    if not cur:
        return val
    if type(cur) is str:
        return cur if cur == val else (cur, val)
    return cur if val in cur else (*cur, val)


def extract_csv_rows_clues(
    rel: str,
    fmt: str,
    header: list[str],
    col_roles: list[str],
    rows_chunk: list[list[str]],
    start_row_idx: int,
    date_hint: str,
    modal_year: int,
    is_patient_file: bool,
    hosp: str,
) -> list[RecordClues]:
    """Extract compact RecordClues directly without temporary set/list allocations per row."""
    clues_list: list[RecordClues] = []
    h_lows = [h.lower() for h in header]
    role_count = len(col_roles)
    if not is_patient_file:
        for off in range(len(rows_chunk)):
            clues_list.append(RecordClues._from_compact(
                rel, f"row:{start_row_idx + off}", fmt, False, hosp,
                "", "", "", "", "", "", "", "", "", "", "", None, "", "", "", "", "",
            ))
        return clues_list

    s_intern: dict[str, str] = {}

    def _sin(x: str) -> str:
        r = s_intern.get(x)
        if r is not None:
            return r
        if len(s_intern) < 16384:
            s_intern[x] = x
        return x

    hosp_i = _sin(hosp) if hosp else ""

    for off, row in enumerate(rows_chunk):
        loc = f"row:{start_row_idx + off}"
        mrns: str | tuple[str, ...] = ""
        aadhaars: str | tuple[str, ...] = ""
        last4s: str | tuple[str, ...] = ""
        abhas: str | tuple[str, ...] = ""
        phones: str | tuple[str, ...] = ""
        emails: str | tuple[str, ...] = ""
        pans: str | tuple[str, ...] = ""
        ins_ids: str | tuple[str, ...] = ""
        pats: str | tuple[str, ...] = ""
        rels: str | tuple[str, ...] = ""
        diags: str | tuple[str, ...] = ""
        dob = ""
        age: int | None = None
        gender = ""
        village = ""
        district = ""
        pin = ""
        first_nm, last_nm = "", ""
        n_cols = min(len(row), role_count)
        for c_idx in range(n_cols):
            val = row[c_idx].strip()
            if not val:
                continue
            role = col_roles[c_idx]
            h_low = h_lows[c_idx]
            if role == "PATIENT_NAME":
                vi = _sin(val)
                if "first" in h_low:
                    first_nm = vi
                elif "last" in h_low:
                    last_nm = vi
                else:
                    pats = _add_packed_str(pats, vi)
            elif role == "RELATIVE_NAME":
                rels = _add_packed_str(rels, _sin(val))
            elif role == "MRN":
                mrns = _add_packed_str(mrns, _sin(norm_mrn_key(val)))
            elif role == "AADHAAR":
                if "x" in val.lower() or "*" in val:
                    s_a = val if val.isascii() else to_ascii_digits(val)
                    m4 = _RX_LAST4.search(s_a)
                    if m4:
                        last4s = _add_packed_str(last4s, _sin(m4.group(1)))
                else:
                    d = _ascii_strip_digits(val)
                    if len(d) == 12:
                        aadhaars = _add_packed_str(aadhaars, _sin(d))
            elif role == "ABHA":
                d = _ascii_strip_digits(val)
                if len(d) >= 12:
                    abhas = _add_packed_str(abhas, _sin(d))
            elif role == "PAN":
                pans = _add_packed_str(pans, _sin(val.upper()))
            elif role == "INSURANCE_ID":
                ins_ids = _add_packed_str(ins_ids, _sin(val.upper()))
            elif role == "PHONE":
                d = _ascii_strip_digits(val)
                if len(d) >= 10:
                    phones = _add_packed_str(phones, _sin(d[-10:]))
            elif role == "EMAIL":
                emails = _add_packed_str(emails, _sin(val.lower()))
            elif role == "DOB":
                dv, _ = parse_date_flex(val, hint=date_hint, default_year=modal_year)
                if dv:
                    dob = _sin(dv.isoformat())
            elif role == "AGE" or role == "AGE_SEX":
                a, g = parse_age_val(val)
                if a is not None:
                    age = a
                if g:
                    gender = g
            elif role == "GENDER":
                g = norm_gender(val)
                if g:
                    gender = g
            elif role == "VILLAGE":
                village = _sin(val)
            elif role == "DISTRICT":
                district = _sin(val)
            elif role == "PIN":
                d = _ascii_strip_digits(val)
                if len(d) == 6:
                    pin = _sin(d)
            elif role == "OTHER" and ("procedure" in h_low or "diagnosis" in h_low or "icd" in h_low):
                diags = _add_packed_str(diags, _sin(val.lower()))

        if first_nm or last_nm:
            full = _sin(f"{first_nm} {last_nm}".strip())
            if full:
                if not pats:
                    pats = full
                elif type(pats) is str:
                    pats = pats if pats == full else (full, pats)
                elif full not in pats:
                    pats = (full, *pats)

        clues_list.append(RecordClues._from_compact(
            rel, loc, fmt, True, hosp_i,
            mrns if type(mrns) is str else tuple(sorted(mrns)),
            aadhaars if type(aadhaars) is str else tuple(sorted(aadhaars)),
            last4s if type(last4s) is str else tuple(sorted(last4s)),
            abhas if type(abhas) is str else tuple(sorted(abhas)),
            phones if type(phones) is str else tuple(sorted(phones)),
            emails if type(emails) is str else tuple(sorted(emails)),
            pans if type(pans) is str else tuple(sorted(pans)),
            ins_ids if type(ins_ids) is str else tuple(sorted(ins_ids)),
            pats, rels, dob, age, gender, village, district, pin,
            diags if type(diags) is str else tuple(sorted(diags)),
        ))
    return clues_list


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
        col_roles = [
            detector.classify_column_header(h, [r[idx] for r in body[:25] if idx < len(r)])
            for idx, h in enumerate(header)
        ]
        has_patient_cols = any(r in ("PATIENT_NAME", "MRN", "AADHAAR", "ABHA", "PHONE") for r in col_roles)
        return extract_csv_rows_clues(
            rel=rel,
            fmt=inf.fmt,
            header=header,
            col_roles=col_roles,
            rows_chunk=body,
            start_row_idx=0,
            date_hint=date_hint,
            modal_year=modal_year,
            is_patient_file=(not is_clean_ref and not is_non_patient_misc and has_patient_cols),
            hosp=detect_hospital_name(rel),
        )

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
            rc.compact()
            clues_list.append(rc)
            return clues_list

        # List-of-dicts JSON (opd, dhh_patients, bulk_lab_export)
        hosp = ""
        if isinstance(data.get("_export_meta"), dict):
            hosp = detect_hospital_name(str(data["_export_meta"].get("hospital", "")))
        elif isinstance(data.get("facility"), dict):
            hosp = detect_hospital_name(str(data["facility"].get("name", "")))
        file_hosp = hosp or detect_hospital_name(rel)

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
                        hospital=file_hosp,
                    )
                    _extract_from_json_obj(item, rc, detector, date_hint, modal_year)
                    rc.compact()
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
        n_patients = 0
        for ent in bundle.get("entry", []):
            res = ent.get("resource", {})
            rt = res.get("resourceType")
            if rt == "Organization" and not rc.hospital:
                rc.hospital = detect_hospital_name(str(res.get("name", "")))
            elif rt == "Patient":
                n_patients += 1
                if n_patients > 1:
                    continue
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
            elif rt == "Encounter" and n_patients <= 1:
                for ident in res.get("identifier", []):
                    if ident.get("value"):
                        rc.mrns.add(norm_mrn_key(str(ident["value"])))
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
        if n_patients > 1:
            rc.is_patient_record = False
        rc.compact()
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
        hospital=detect_hospital_name(txt[:8192]) or detect_hospital_name(rel),
    )
    if rc.is_patient_record:
        _extract_from_text(txt, rc, detector, date_hint, modal_year)
        _add_filename_clues(rel, rc, detector)
        if len(rc.mrns) > 10 or len(rc.patient_names) > 20:
            rc.is_patient_record = False
    rc.compact()
    clues_list.append(rc)
    return clues_list


def extract_json_items_clues(
    rel: str,
    pkey: str,
    items: list[dict[str, Any]],
    start_idx: int,
    detector: PIIDetector,
    date_hint: str,
    modal_year: int,
    is_patient_record: bool,
    file_hosp: str,
) -> list[RecordClues]:
    """Extract compacted RecordClues from a slice of JSON dict items (for huge arrays)."""
    out: list[RecordClues] = []
    for off, item in enumerate(items):
        loc = f"/{pkey}/{start_idx + off}"
        rc = RecordClues(
            unit_id=f"{rel}|{loc}",
            file_rel=rel,
            loc=loc,
            fmt="json",
            is_patient_record=is_patient_record,
            hospital=file_hosp,
        )
        if is_patient_record:
            _extract_from_json_obj(item, rc, detector, date_hint, modal_year)
        rc.compact()
        out.append(rc)
    return out


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
    def _walk(node: Any, leaf_key: str, parent_key: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, str(k).lower(), leaf_key)
            return
        if isinstance(node, list):
            for x in node:
                _walk(x, leaf_key, parent_key)
            return
        if node is None:
            return

        sval = str(node).strip()
        if not sval:
            return

        if leaf_key in ("uhid", "patient_ref", "hospital_uhid", "reg_no", "mrn"):
            rc.mrns.add(norm_mrn_key(sval))
        elif leaf_key in ("aadhar_no", "aadhaar", "aadhaar_no"):
            d = _RX_NON_DIGITS.sub("", to_ascii_digits(sval))
            if len(d) == 12:
                rc.aadhaars.add(d)
        elif leaf_key in ("abha", "abha_no", "abha_number"):
            d = _RX_NON_DIGITS.sub("", to_ascii_digits(sval))
            if len(d) >= 12:
                rc.abhas.add(d)
        elif leaf_key in ("mobile", "phone", "callback"):
            d = _RX_NON_DIGITS.sub("", to_ascii_digits(sval))
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
            d = _RX_NON_DIGITS.sub("", to_ascii_digits(sval))
            if len(d) == 6:
                rc.pin = d
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

    _walk(obj, "", "")


def split_text_chunks(text: str, target_bytes: int = 65536) -> list[str]:
    """Split large plain-text documents along paragraph/line boundaries without breaking lines."""
    n = len(text)
    if n <= target_bytes:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(n, start + target_bytes)
        if end < n:
            split_at = text.rfind("\n\n", start, end)
            if split_at > start:
                end = split_at + 2
            else:
                split_at = text.rfind("\n", start, end)
                if split_at > start:
                    end = split_at + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def extract_text_chunk_into(
    chunk: str,
    rc: RecordClues,
    detector: PIIDetector,
    date_hint: str,
    modal_year: int,
) -> None:
    """Extract PII and demographic/clinical clues from a text chunk into rc."""
    for sub in split_text_chunks(chunk):
        spans = detector.detect_text(sub, allow_pin_scan=True)
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
                v = re.sub(r"(?i)^(?:vill\.?|village|gram|gaon|ग्राम|गाँव|गांव)\s*[:\-]?\s*", "", sp.surface).strip()
                if v:
                    rc.village = v

    norm = to_ascii_digits(chunk)
    if rc.age is None:
        m_age = re.search(
            r"(?i)(?:Age(?:/Sex)?|आयु)\s*:\s*(-?\d{1,3})\s*(?:Yrs?|वर्ष)?(?:\s*/\s*(M|F|Male|Female|पुरुष|महिला))?",
            norm,
        )
        if not m_age:
            m_age = re.search(r"(?i)\b(\d{1,3})\s*Y\s*/\s*([MF])\b", norm)
        if m_age:
            rc.age = int(m_age.group(1))
            if m_age.group(2) and not rc.gender:
                rc.gender = norm_gender(m_age.group(2))

    if not rc.gender:
        if re.search(r"\bW/o\b|पति\s*का\s*नाम", chunk, re.I):
            rc.gender = "F"

    if not rc.dob:
        m_dob = re.search(r"(?i)\bDOB\s*:\s*([^\n,;]+)", norm)
        if m_dob:
            d, _ = parse_date_flex(m_dob.group(1), hint=date_hint, default_year=modal_year)
            if d:
                rc.dob = d.isoformat()

    if not rc.district:
        m_dist = re.search(r"(?i)(?:\bDistt?\.?|\bDistrict|जिला)\s*[:\-]?\s*([A-Za-z\u0900-\u097f ]{3,25}?)(?=\s*[-–,\n]|$)", norm)
        if m_dist:
            rc.district = m_dist.group(1).strip()

    for m_icd in _RX_ICD_ONLY.finditer(norm):
        rc.diagnoses.add(m_icd.group(1).upper())


def _extract_from_text(
    text: str,
    rc: RecordClues,
    detector: PIIDetector,
    date_hint: str,
    modal_year: int,
) -> None:
    extract_text_chunk_into(text, rc, detector, date_hint, modal_year)
    if rc.patient_names:
        rc.patient_names = list(dict.fromkeys(rc.patient_names))
    if rc.relative_names:
        rc.relative_names = list(dict.fromkeys(rc.relative_names))


def merge_record_clues_inplace(dst: RecordClues, src: RecordClues) -> None:
    """Merge partial chunk RecordClues (for parallel chunked huge TXT files) into dst."""
    for slot_attr, prop_attr in (
        ("_mrns", "mrns"), ("_aadhaars", "aadhaars"), ("_aadhaar_last4", "aadhaar_last4"),
        ("_abhas", "abhas"), ("_phones", "phones"), ("_emails", "emails"),
        ("_pans", "pans"), ("_insurance_ids", "insurance_ids"), ("_diagnoses", "diagnoses"),
    ):
        cur = getattr(dst, slot_attr)
        if type(cur) is not set:
            cur = set(_as_tup(cur))
            setattr(dst, slot_attr, cur)
        cur.update(getattr(src, prop_attr))
    if not dst.hospital and src.hospital:
        dst.hospital = src.hospital
    if not dst.dob and src.dob:
        dst.dob = src.dob
    if dst.age is None and src.age is not None:
        dst.age = src.age
    if not dst.gender and src.gender:
        dst.gender = src.gender
    if not dst.village and src.village:
        dst.village = src.village
    if not dst.district and src.district:
        dst.district = src.district
    if not dst.pin and src.pin:
        dst.pin = src.pin
    if type(dst._patient_names) is not list:
        dst._patient_names = list(_as_tup(dst._patient_names))
    dst._patient_names.extend(src.patient_names)
    if type(dst._relative_names) is not list:
        dst._relative_names = list(_as_tup(dst._relative_names))
    dst._relative_names.extend(src.relative_names)


# --------------------------------------------------------------------------- Conservative Clustering

@dataclass(slots=True)
class PatientCluster:
    cluster_id: int | str
    min_unit_id: str = ""
    mrns: set[str] | frozenset[str] = _EMPTY_FS
    aadhaars: set[str] | frozenset[str] = _EMPTY_FS
    aadhaar_last4: set[str] | frozenset[str] = _EMPTY_FS
    abhas: set[str] | frozenset[str] = _EMPTY_FS
    phones: set[str] | frozenset[str] = _EMPTY_FS
    emails: set[str] | frozenset[str] = _EMPTY_FS
    pans: set[str] | frozenset[str] = _EMPTY_FS
    insurance_ids: set[str] | frozenset[str] = _EMPTY_FS
    dobs: set[str] | frozenset[str] = _EMPTY_FS
    ages: set[int] | frozenset[int] = field(default_factory=lambda: _EMPTY_FS)  # type: ignore[arg-type]
    min_age: int = 999
    max_age: int = -999
    first_age: int | None = None
    genders: set[str] | frozenset[str] = _EMPTY_FS
    villages: list[str] | tuple[str, ...] = _EMPTY_TUP
    districts: list[str] | tuple[str, ...] = _EMPTY_TUP
    pins: set[str] | frozenset[str] = _EMPTY_FS
    patient_names: list[str] | tuple[str, ...] = _EMPTY_TUP
    relative_names: list[str] | tuple[str, ...] = _EMPTY_TUP
    diagnoses: set[str] | frozenset[str] = _EMPTY_FS

    def add(self, rc: RecordClues) -> None:
        if not self.min_unit_id or rc.unit_id < self.min_unit_id:
            self.min_unit_id = rc.unit_id
        if rc.mrns:
            if not isinstance(self.mrns, set):
                self.mrns = set(rc.mrns)
            else:
                self.mrns.update(rc.mrns)
        if rc.aadhaars:
            if not isinstance(self.aadhaars, set):
                self.aadhaars = set(rc.aadhaars)
            else:
                self.aadhaars.update(rc.aadhaars)
            if not isinstance(self.aadhaar_last4, set):
                self.aadhaar_last4 = {a[-4:] for a in rc.aadhaars}
            else:
                for a in rc.aadhaars:
                    self.aadhaar_last4.add(a[-4:])
        if rc.aadhaar_last4:
            if not isinstance(self.aadhaar_last4, set):
                self.aadhaar_last4 = set(rc.aadhaar_last4)
            else:
                self.aadhaar_last4.update(rc.aadhaar_last4)
        if rc.abhas:
            if not isinstance(self.abhas, set):
                self.abhas = set(rc.abhas)
            else:
                self.abhas.update(rc.abhas)
        if rc.phones:
            if not isinstance(self.phones, set):
                self.phones = set(rc.phones)
            else:
                self.phones.update(rc.phones)
        if rc.emails:
            if not isinstance(self.emails, set):
                self.emails = set(rc.emails)
            else:
                self.emails.update(rc.emails)
        if rc.pans:
            if not isinstance(self.pans, set):
                self.pans = set(rc.pans)
            else:
                self.pans.update(rc.pans)
        if rc.insurance_ids:
            if not isinstance(self.insurance_ids, set):
                self.insurance_ids = set(rc.insurance_ids)
            else:
                self.insurance_ids.update(rc.insurance_ids)
        if rc.dob:
            if not isinstance(self.dobs, set):
                self.dobs = {rc.dob}
            else:
                self.dobs.add(rc.dob)
        if rc.age is not None:
            a = rc.age
            if self.first_age is None:
                self.first_age = a
            if not isinstance(self.ages, set):
                self.ages = {a}
            else:
                self.ages.add(a)
            if a < self.min_age:
                self.min_age = a
            if a > self.max_age:
                self.max_age = a
        if rc.gender:
            if not isinstance(self.genders, set):
                self.genders = {rc.gender}
            else:
                self.genders.add(rc.gender)
        if rc.village:
            if not isinstance(self.villages, list):
                self.villages = [rc.village]
            elif rc.village not in self.villages:
                self.villages.append(rc.village)
        if rc.district:
            if not isinstance(self.districts, list):
                self.districts = [rc.district]
            elif rc.district not in self.districts:
                self.districts.append(rc.district)
        if rc.pin:
            if not isinstance(self.pins, set):
                self.pins = {rc.pin}
            else:
                self.pins.add(rc.pin)
        if rc.patient_names:
            if not isinstance(self.patient_names, list):
                self.patient_names = list(rc.patient_names)
            else:
                for nm in rc.patient_names:
                    if nm not in self.patient_names:
                        self.patient_names.append(nm)
        if rc.relative_names:
            if not isinstance(self.relative_names, list):
                self.relative_names = list(rc.relative_names)
            else:
                for rl in rc.relative_names:
                    if rl not in self.relative_names:
                        self.relative_names.append(rl)
        if rc.diagnoses:
            if not isinstance(self.diagnoses, set):
                self.diagnoses = set(rc.diagnoses)
            else:
                self.diagnoses.update(rc.diagnoses)

    def merge_from(self, other: PatientCluster) -> None:
        if other.min_unit_id and (not self.min_unit_id or other.min_unit_id < self.min_unit_id):
            self.min_unit_id = other.min_unit_id
        if other.mrns:
            if not isinstance(self.mrns, set):
                self.mrns = set(other.mrns)
            else:
                self.mrns.update(other.mrns)
        if other.aadhaars:
            if not isinstance(self.aadhaars, set):
                self.aadhaars = set(other.aadhaars)
            else:
                self.aadhaars.update(other.aadhaars)
        if other.aadhaar_last4:
            if not isinstance(self.aadhaar_last4, set):
                self.aadhaar_last4 = set(other.aadhaar_last4)
            else:
                self.aadhaar_last4.update(other.aadhaar_last4)
        if other.abhas:
            if not isinstance(self.abhas, set):
                self.abhas = set(other.abhas)
            else:
                self.abhas.update(other.abhas)
        if other.phones:
            if not isinstance(self.phones, set):
                self.phones = set(other.phones)
            else:
                self.phones.update(other.phones)
        if other.emails:
            if not isinstance(self.emails, set):
                self.emails = set(other.emails)
            else:
                self.emails.update(other.emails)
        if other.pans:
            if not isinstance(self.pans, set):
                self.pans = set(other.pans)
            else:
                self.pans.update(other.pans)
        if other.insurance_ids:
            if not isinstance(self.insurance_ids, set):
                self.insurance_ids = set(other.insurance_ids)
            else:
                self.insurance_ids.update(other.insurance_ids)
        if other.dobs:
            if not isinstance(self.dobs, set):
                self.dobs = set(other.dobs)
            else:
                self.dobs.update(other.dobs)
        if other.ages:
            if self.first_age is None:
                self.first_age = other.first_age
            if not isinstance(self.ages, set):
                self.ages = set(other.ages)
            else:
                self.ages.update(other.ages)
            if other.min_age < self.min_age:
                self.min_age = other.min_age
            if other.max_age > self.max_age:
                self.max_age = other.max_age
        if other.genders:
            if not isinstance(self.genders, set):
                self.genders = set(other.genders)
            else:
                self.genders.update(other.genders)
        if other.villages:
            if not isinstance(self.villages, list):
                self.villages = list(other.villages)
            else:
                for v in other.villages:
                    if v not in self.villages:
                        self.villages.append(v)
        if other.districts:
            if not isinstance(self.districts, list):
                self.districts = list(other.districts)
            else:
                for d in other.districts:
                    if d not in self.districts:
                        self.districts.append(d)
        if other.pins:
            if not isinstance(self.pins, set):
                self.pins = set(other.pins)
            else:
                self.pins.update(other.pins)
        if other.patient_names:
            if not isinstance(self.patient_names, list):
                self.patient_names = list(other.patient_names)
            else:
                for nm in other.patient_names:
                    if nm not in self.patient_names:
                        self.patient_names.append(nm)
        if other.relative_names:
            if not isinstance(self.relative_names, list):
                self.relative_names = list(other.relative_names)
            else:
                for rl in other.relative_names:
                    if rl not in self.relative_names:
                        self.relative_names.append(rl)
        if other.diagnoses:
            if not isinstance(self.diagnoses, set):
                self.diagnoses = set(other.diagnoses)
            else:
                self.diagnoses.update(other.diagnoses)

    def canonical_key(self) -> str:
        """Deterministic canonical identifier sorted from strongest immutable cluster tokens."""
        for s in (self.aadhaars, self.abhas, self.mrns, self.phones, self.pans, self.emails):
            if s:
                return f"id:{min(s)}"
        nm = sorted(canonicalize_person_name(x) for x in self.patient_names if x)
        dob = min(self.dobs) if self.dobs else ""
        cid_tag = f"C{self.cluster_id:05d}" if isinstance(self.cluster_id, int) else self.cluster_id
        return f"bio:{nm[0] if nm else cid_tag}|{dob}|{self.min_unit_id}"


def _obj_canonical_key(obj: PatientCluster | RecordClues, cid: int) -> str:
    if isinstance(obj, PatientCluster):
        return obj.canonical_key()
    for s in (obj.aadhaars, obj.abhas, obj.mrns, obj.phones, obj.pans, obj.emails):
        if s:
            return f"id:{min(s)}"
    nm = sorted(canonicalize_person_name(x) for x in obj.patient_names if x)
    return f"bio:{nm[0] if nm else f'C{cid:05d}'}|{obj.dob}|{obj.unit_id}"


def _disjoint_iter(a: Any, b: Any) -> bool:
    if isinstance(a, (set, frozenset)):
        return a.isdisjoint(b)
    if isinstance(b, (set, frozenset)):
        return b.isdisjoint(a)
    if len(a) == 1 and len(b) == 1:
        return a[0] != b[0]
    sb = set(b)
    return not any(x in sb for x in a)


def _has_hard_conflict(
    c1: PatientCluster | RecordClues,
    c2: PatientCluster | RecordClues,
    max_age_diff: int = 2,
    strict_place: bool = True,
) -> bool:
    """Strictly forbid merging clusters/records with any incompatible demographic or ID signal."""
    if isinstance(c1, PatientCluster):
        g1, d1, l4_1, ages1, min1, max1, v1_seq, dist1_seq = (
            c1.genders, c1.dobs, c1.aadhaar_last4, c1.ages, c1.min_age, c1.max_age, c1.villages, c1.districts
        )
    else:
        g1 = (c1.gender,) if c1.gender else ()
        d1 = (c1.dob,) if c1.dob else ()
        l4_1 = c1.aadhaar_last4 if (c1.aadhaar_last4 or not c1.aadhaars) else tuple(a[-4:] for a in c1.aadhaars)
        if not c1.aadhaar_last4 and c1.aadhaars:
            l4_1 = tuple(a[-4:] for a in c1.aadhaars)
        elif c1.aadhaar_last4 and c1.aadhaars:
            l4_1 = tuple(c1.aadhaar_last4) + tuple(a[-4:] for a in c1.aadhaars)
        ages1 = (c1.age,) if c1.age is not None else ()
        min1 = c1.age if c1.age is not None else 999
        max1 = c1.age if c1.age is not None else -999
        v1_seq = (c1.village,) if c1.village else ()
        dist1_seq = (c1.district,) if c1.district else ()

    if isinstance(c2, PatientCluster):
        g2, d2, l4_2, ages2, min2, max2, v2_seq, dist2_seq = (
            c2.genders, c2.dobs, c2.aadhaar_last4, c2.ages, c2.min_age, c2.max_age, c2.villages, c2.districts
        )
    else:
        g2 = (c2.gender,) if c2.gender else ()
        d2 = (c2.dob,) if c2.dob else ()
        if not c2.aadhaar_last4 and c2.aadhaars:
            l4_2 = tuple(a[-4:] for a in c2.aadhaars)
        elif c2.aadhaar_last4 and c2.aadhaars:
            l4_2 = tuple(c2.aadhaar_last4) + tuple(a[-4:] for a in c2.aadhaars)
        else:
            l4_2 = c2.aadhaar_last4
        ages2 = (c2.age,) if c2.age is not None else ()
        min2 = c2.age if c2.age is not None else 999
        max2 = c2.age if c2.age is not None else -999
        v2_seq = (c2.village,) if c2.village else ()
        dist2_seq = (c2.district,) if c2.district else ()

    if g1 and g2 and _disjoint_iter(g1, g2):
        return True
    if d1 and d2 and _disjoint_iter(d1, d2):
        return True
    if c1.aadhaars and c2.aadhaars and _disjoint_iter(c1.aadhaars, c2.aadhaars):
        return True
    if l4_1 and l4_2 and _disjoint_iter(l4_1, l4_2):
        return True
    if c1.abhas and c2.abhas and _disjoint_iter(c1.abhas, c2.abhas):
        return True
    if c1.pans and c2.pans and _disjoint_iter(c1.pans, c2.pans):
        return True
    if ages1 and ages2:
        if min1 > max2 + max_age_diff or min2 > max1 + max_age_diff:
            return True
        if len(ages1) > 1 or len(ages2) > 1:
            if min(abs(a1 - a2) for a1 in ages1 for a2 in ages2) > max_age_diff:
                return True

    # Full-name compatibility check (only if both have multi-token names)
    p1 = c1.patient_names
    p2 = c2.patient_names
    if p1 and p2:
        if len(p1) == 1 and len(p2) == 1 and p1[0] == p2[0]:
            pass
        else:
            full1 = [n for n in p1[:12] if len(_canonicalize_person_name_cached(n)) >= 2]
            full2 = [n for n in p2[:12] if len(_canonicalize_person_name_cached(n)) >= 2]
            if full1 and full2:
                if len(full1) == 1 and len(full2) == 1:
                    if full1[0] != full2[0] and names_similarity(full1[0], full2[0]) < 0.72:
                        return True
                elif set(full1).isdisjoint(full2):
                    best_sim = max(names_similarity(a, b) for a in full1 for b in full2)
                    if best_sim < 0.72:
                        return True

    # Relative compatibility check
    r1_all = c1.relative_names
    r2_all = c2.relative_names
    if r1_all and r2_all:
        if len(r1_all) == 1 and len(r2_all) == 1 and r1_all[0] == r2_all[0]:
            pass
        else:
            rel1 = [n for n in r1_all[:12] if _canonicalize_person_name_cached(n)]
            rel2 = [n for n in r2_all[:12] if _canonicalize_person_name_cached(n)]
            if rel1 and rel2 and set(rel1).isdisjoint(rel2):
                best_rel = names_similarity(rel1[0], rel2[0]) if (len(rel1) == 1 and len(rel2) == 1) else max(names_similarity(a, b) for a in rel1 for b in rel2)
                if best_rel < 0.70:
                    first_match = max(
                        names_similarity(_canonicalize_person_name_cached(a)[0], _canonicalize_person_name_cached(b)[0])
                        for a in rel1 for b in rel2
                    )
                    if first_match < 0.80:
                        return True

    # Village & District cross-script check (fuzzy Indic transliteration-aware)
    if strict_place:
        if v1_seq and v2_seq:
            if not (len(v1_seq) == 1 and len(v2_seq) == 1 and v1_seq[0] == v2_seq[0]):
                if set(v1_seq).isdisjoint(v2_seq):
                    best_v = places_similarity(v1_seq[0], v2_seq[0]) if (len(v1_seq) == 1 and len(v2_seq) == 1) else max(places_similarity(v1, v2) for v1 in v1_seq[:8] for v2 in v2_seq[:8])
                    if best_v < 0.75:
                        return True
        if dist1_seq and dist2_seq:
            if not (len(dist1_seq) == 1 and len(dist2_seq) == 1 and dist1_seq[0] == dist2_seq[0]):
                if set(dist1_seq).isdisjoint(dist2_seq):
                    best_d = places_similarity(dist1_seq[0], dist2_seq[0]) if (len(dist1_seq) == 1 and len(dist2_seq) == 1) else max(places_similarity(d1, d2) for d1 in dist1_seq[:8] for d2 in dist2_seq[:8])
                    if best_d < 0.75:
                        return True
    return False


def _best_full_name_sim(c1: PatientCluster | RecordClues, c2: PatientCluster | RecordClues) -> float:
    n1 = [n for n in c1.patient_names[:12] if len(_canonicalize_person_name_cached(n)) >= 2]
    n2 = [n for n in c2.patient_names[:12] if len(_canonicalize_person_name_cached(n)) >= 2]
    if not n1 or not n2:
        return 0.0
    if len(n1) == 1 and len(n2) == 1:
        return 1.0 if n1[0] == n2[0] else names_similarity(n1[0], n2[0])
    return max(names_similarity(a, b) for a in n1 for b in n2)


def _best_given_name_sim(c1: PatientCluster | RecordClues, c2: PatientCluster | RecordClues) -> float:
    g1 = [ct[0] for n in c1.patient_names[:12] if (ct := _canonicalize_person_name_cached(n))]
    g2 = [ct[0] for n in c2.patient_names[:12] if (ct := _canonicalize_person_name_cached(n))]
    if not g1 or not g2:
        return 0.0
    from rapidfuzz import fuzz
    if len(g1) == 1 and len(g2) == 1:
        return 1.0 if g1[0] == g2[0] else (fuzz.ratio(g1[0], g2[0]) / 100.0)
    return max(fuzz.ratio(a, b) / 100.0 for a in g1 for b in g2)


def _best_rel_sim(c1: PatientCluster | RecordClues, c2: PatientCluster | RecordClues) -> float:
    r1 = [n for n in c1.relative_names[:12] if _canonicalize_person_name_cached(n)]
    r2 = [n for n in c2.relative_names[:12] if _canonicalize_person_name_cached(n)]
    if not r1 or not r2:
        return 0.0
    if len(r1) == 1 and len(r2) == 1 and r1[0] == r2[0]:
        return 1.0
    from rapidfuzz import fuzz
    best = 0.0
    for a in r1:
        ta = _canonicalize_person_name_cached(a)
        for b in r2:
            best = max(best, names_similarity(a, b))
            tb = _canonicalize_person_name_cached(b)
            if ta and tb and (len(ta) == 1 or len(tb) == 1):
                best = max(best, fuzz.ratio(ta[0], tb[0]) / 100.0)
    return best


def _best_village_sim(c1: PatientCluster | RecordClues, c2: PatientCluster | RecordClues) -> float:
    v1_seq = c1.villages if isinstance(c1, PatientCluster) else ((c1.village,) if c1.village else ())
    v2_seq = c2.villages if isinstance(c2, PatientCluster) else ((c2.village,) if c2.village else ())
    if not v1_seq or not v2_seq:
        return 0.0
    if len(v1_seq) == 1 and len(v2_seq) == 1:
        return 1.0 if v1_seq[0] == v2_seq[0] else places_similarity(v1_seq[0], v2_seq[0])
    return max(places_similarity(v1, v2) for v1 in v1_seq[:8] for v2 in v2_seq[:8])


def link_all_records(
    all_clues: list[RecordClues],
    keyring: KeyRing,
    policy: dict[str, Any],
) -> tuple[dict[str, str | None], dict[str, PatientCluster | RecordClues], dict[str, str]]:
    """Link records conservatively using lazy Union-Find clustering (zero PatientCluster allocation for singletons)."""
    name_th = float(policy.get("linkage_name_threshold", 0.85))
    rel_th = float(policy.get("linkage_relative_threshold", 0.82))
    vil_th = float(policy.get("linkage_village_threshold", 0.82))
    max_age_diff = int(policy.get("linkage_max_age_diff", 2))

    n = len(all_clues)
    parent = list(range(n))
    multi_clusters: dict[int, PatientCluster] = {}
    alive_roots = [i for i, rc in enumerate(all_clues) if rc.is_patient_record]

    def _find_root(cid: int) -> int:
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid

    def _get_obj(cid: int) -> PatientCluster | RecordClues:
        return multi_clusters.get(cid) or all_clues[cid]

    def _merge(cid_keep: int, cid_drop: int) -> int:
        rk = _find_root(cid_keep)
        rd = _find_root(cid_drop)
        if rk == rd:
            return rk
        if rk > rd:
            rk, rd = rd, rk
        ck = multi_clusters.get(rk)
        if ck is None:
            ck = PatientCluster(cluster_id=rk)
            ck.add(all_clues[rk])
            multi_clusters[rk] = ck
        cd = multi_clusters.pop(rd, None)
        if cd is not None:
            ck.merge_from(cd)
        else:
            ck.add(all_clues[rd])
        parent[rd] = rk
        return rk

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        # Stage 1: Exact Strong-ID merges (single-hit keys never allocate list buckets)
        id_specs = [
            ("aadhaar", "_aadhaars", "aadhaars"),
            ("abha", "_abhas", "abhas"),
            ("pan", "_pans", "pans"),
            ("email", "_emails", "emails"),
            ("insurance", "_insurance_ids", "insurance_ids"),
            ("mrn", "_mrns", "mrns"),
            ("phone", "_phones", "phones"),
        ]
        for kind, rc_slot, cl_attr in id_specs:
            alive_roots = [c for c in alive_roots if parent[c] == c]
            first_cid: dict[str, int] = {}
            dup_buckets: dict[str, list[int]] = {}
            for cid in alive_roots:
                cl = multi_clusters.get(cid)
                if cl is None:
                    raw = getattr(all_clues[cid], rc_slot)
                    if not raw:
                        continue
                    if type(raw) is str:
                        prev = first_cid.get(raw)
                        if prev is None:
                            first_cid[raw] = cid
                        else:
                            lst = dup_buckets.get(raw)
                            if lst is None:
                                dup_buckets[raw] = [prev, cid]
                            else:
                                lst.append(cid)
                        continue
                    vals = raw
                else:
                    vals = getattr(cl, cl_attr)
                for key in vals:
                    if not key:
                        continue
                    prev = first_cid.get(key)
                    if prev is None:
                        first_cid[key] = cid
                    else:
                        lst = dup_buckets.get(key)
                        if lst is None:
                            dup_buckets[key] = [prev, cid]
                        else:
                            lst.append(cid)
            del first_cid
            if not dup_buckets:
                continue

            active_keys = sorted(dup_buckets.keys())
            eff_age_diff = 4 if kind in ("mrn", "aadhaar", "abha", "pan", "email") else max_age_diff
            strict_pl = (kind == "phone")

            while active_keys:
                next_dirty: set[str] = set()
                for key in active_keys:
                    raw_cids = dup_buckets[key]
                    seen_r: set[int] = set()
                    cids: list[int] = []
                    for rc_id in raw_cids:
                        r = _find_root(rc_id)
                        if r not in seen_r:
                            seen_r.add(r)
                            cids.append(r)
                    if len(cids) < 2:
                        continue
                    cids.sort()
                    dup_buckets[key] = cids
                    anchor = cids[0]
                    merged_here = False
                    for other in cids[1:]:
                        r_a = _find_root(anchor)
                        r_o = _find_root(other)
                        if r_a == r_o:
                            continue
                        ca, co = _get_obj(r_a), _get_obj(r_o)
                        if _has_hard_conflict(ca, co, eff_age_diff, strict_place=strict_pl):
                            continue
                        if strict_pl and ca.patient_names and co.patient_names:
                            if _best_given_name_sim(ca, co) < 0.65:
                                continue
                        other_keys = getattr(co, cl_attr)
                        anchor = _merge(r_a, r_o)
                        merged_here = True
                        for ok in other_keys:
                            if ok and ok != key and ok in dup_buckets:
                                dup_buckets[ok].append(anchor)
                                next_dirty.add(ok)
                    if merged_here:
                        for ak in getattr(_get_obj(anchor), cl_attr):
                            if ak and ak != key and ak in dup_buckets:
                                next_dirty.add(ak)
                active_keys = sorted(next_dirty) if next_dirty else []

        # Stage 2: Cross-hospital / Cross-script / Bio+Relative+Village merges
        _generic_skels = {_consonant_skeleton(g) for g in ("devi", "kumari", "kumar", "lal", "singh", "bano", "begum", "khatun", "kaur")}

        def _cluster_block_keys(obj: PatientCluster | RecordClues) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
            n_list: list[str] = []
            seen_n: set[str] = set()
            for nm in obj.patient_names:
                toks = _canonicalize_person_name_cached(nm)
                sks = [_consonant_skeleton(tok) for tok in toks if tok]
                non_gen = [s for s in sks if s not in _generic_skels]
                for s in (non_gen or sks):
                    if s and s not in seen_n:
                        seen_n.add(s)
                        n_list.append(s)
            r_list: list[str] = []
            seen_r: set[str] = set()
            for rl in obj.relative_names:
                toks = _canonicalize_person_name_cached(rl)
                sks = [_consonant_skeleton(tok) for tok in toks if tok]
                non_gen = [s for s in sks if s not in _generic_skels]
                for s in (non_gen or sks):
                    if s and s not in seen_r:
                        seen_r.add(s)
                        r_list.append(s)
            v_list: list[str] = []
            seen_v: set[str] = set()
            vils = obj.villages if isinstance(obj, PatientCluster) else ((obj.village,) if obj.village else ())
            for vl in vils:
                for tok in canonicalize_place(vl).split():
                    sk = _consonant_skeleton(tok)
                    if sk and sk not in seen_v:
                        seen_v.add(sk)
                        v_list.append(sk)
            return tuple(n_list), tuple(r_list), tuple(v_list)

        for _pass in range(3):
            alive_roots = [c for c in alive_roots if parent[c] == c]
            # Fast pass: only allocate multi-bucket lists for DOB / (n1,r1) / (n1,v1) pairs appearing >= 2 times
            dob_first: dict[str, int] = {}
            dup_dobs: dict[str, list[int]] = {}
            nr_state: dict[tuple[str, str], int] = {}
            nv_state: dict[tuple[str, str], int] = {}
            block_cache: dict[int, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str, tuple[int, ...]]] = {}

            for cid in alive_roots:
                obj = _get_obj(cid)
                if isinstance(obj, PatientCluster):
                    dobs = obj.dobs
                    g_tag = next(iter(obj.genders)) if len(obj.genders) == 1 else "U"
                    ages = obj.ages
                else:
                    dobs = (obj.dob,) if obj.dob else ()
                    g_tag = obj.gender or "U"
                    ages = (obj.age,) if obj.age is not None else ()
                if not obj.patient_names and not dobs:
                    continue
                for d in dobs:
                    prev = dob_first.get(d)
                    if prev is None:
                        dob_first[d] = cid
                    else:
                        lst = dup_dobs.get(d)
                        if lst is None:
                            dup_dobs[d] = [prev, cid]
                        else:
                            lst.append(cid)
                if not obj.patient_names or (not obj.relative_names and not (isinstance(obj, PatientCluster) and obj.villages) and not (isinstance(obj, RecordClues) and obj.village)):
                    if dobs:
                        nk, _, _ = _cluster_block_keys(obj)
                        a_bins = tuple(sorted({a // 5 for a in ages})) if ages else (-1,)
                        block_cache[cid] = (nk, (), (), g_tag, a_bins)
                    continue
                nk, rk, vk = _cluster_block_keys(obj)
                a_bins = tuple(sorted({a // 5 for a in ages})) if ages else (-1,)
                block_cache[cid] = (nk, rk, vk, g_tag, a_bins)
                for n1 in nk:
                    for r1 in rk:
                        pr = (n1, r1)
                        st = nr_state.get(pr)
                        if st is None:
                            nr_state[pr] = cid
                        elif st >= 0:
                            nr_state[pr] = -1
                    for v1 in vk:
                        pv = (n1, v1)
                        st = nv_state.get(pv)
                        if st is None:
                            nv_state[pv] = cid
                        elif st >= 0:
                            nv_state[pv] = -1
            del dob_first

            dup_nr_pairs = {p for p, st in nr_state.items() if st == -1}
            dup_nv_pairs = {p for p, st in nv_state.items() if st == -1}
            del nr_state, nv_state

            if not dup_dobs and not dup_nr_pairs and not dup_nv_pairs:
                break

            # Sub-index large DOB buckets (> 48) by (dob, first_consonant) to avoid O(N^2) birthday cross-checks
            dob_sub: dict[tuple[str, str], list[int]] = {}
            big_dobs: set[str] = set()
            for d, d_cids in dup_dobs.items():
                if len(d_cids) > 48:
                    big_dobs.add(d)
                    for cid in d_cids:
                        meta = block_cache.get(cid)
                        if meta:
                            for s in meta[0]:
                                dob_sub.setdefault((d, s[:1]), []).append(cid)

            name_rel_idx: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
            name_rel_no_aad: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
            name_vil_idx: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
            name_vil_no_aad: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
            nr_bins_all: dict[tuple[str, str, str], set[int]] = defaultdict(set)
            nr_bins_no_aad: dict[tuple[str, str, str], set[int]] = defaultdict(set)
            nv_bins_all: dict[tuple[str, str, str], set[int]] = defaultdict(set)
            nv_bins_no_aad: dict[tuple[str, str, str], set[int]] = defaultdict(set)

            for cid, (nk, rk, vk, g_tag, a_bins) in block_cache.items():
                if not nk or (not dup_nr_pairs and not dup_nv_pairs):
                    continue
                obj = _get_obj(cid)
                has_aad = bool(obj.aadhaars)
                for n1 in nk:
                    for r1 in rk:
                        if (n1, r1) not in dup_nr_pairs:
                            continue
                        nrg = (n1, r1, g_tag)
                        nr_bins_all[nrg].update(a_bins)
                        if not has_aad:
                            nr_bins_no_aad[nrg].update(a_bins)
                        for ab in a_bins:
                            key = (n1, r1, g_tag, ab)
                            name_rel_idx[key].append(cid)
                            if not has_aad:
                                name_rel_no_aad[key].append(cid)
                    for v1 in vk:
                        if (n1, v1) not in dup_nv_pairs:
                            continue
                        nvg = (n1, v1, g_tag)
                        nv_bins_all[nvg].update(a_bins)
                        if not has_aad:
                            nv_bins_no_aad[nvg].update(a_bins)
                        for ab in a_bins:
                            key = (n1, v1, g_tag, ab)
                            name_vil_idx[key].append(cid)
                            if not has_aad:
                                name_vil_no_aad[key].append(cid)

            merged_this_pass = False
            for cid_a in alive_roots:
                if parent[cid_a] != cid_a:
                    continue
                meta = block_cache.get(cid_a)
                if not meta:
                    continue
                nk, rk, vk, g_tag, a_bins = meta
                ca = _get_obj(cid_a)
                ca_dobs = ca.dobs if isinstance(ca, PatientCluster) else ((ca.dob,) if ca.dob else ())
                if not nk and not ca_dobs:
                    continue
                g_targets = (g_tag, "U") if g_tag != "U" else ("M", "F", "U")
                missing_age = -1 in a_bins
                ab_targets: tuple[int, ...] = ()
                if not missing_age:
                    ab_set: set[int] = {-1}
                    ca_ages = ca.ages if isinstance(ca, PatientCluster) else ((ca.age,) if ca.age is not None else ())
                    for a in ca_ages:
                        ab_set.add((a - max_age_diff) // 5)
                        ab_set.add((a + max_age_diff) // 5)
                    ab_targets = tuple(ab_set)

                cand_cids: set[int] = set()
                for d in ca_dobs:
                    if d in big_dobs:
                        for s in nk:
                            b = dob_sub.get((d, s[:1]))
                            if b and len(b) <= 256:
                                cand_cids.update(b)
                    else:
                        b = dup_dobs.get(d)
                        if b and len(b) <= 256:
                            cand_cids.update(b)
                nr_src = name_rel_no_aad if ca.aadhaars else name_rel_idx
                nv_src = name_vil_no_aad if ca.aadhaars else name_vil_idx
                nr_bmap = nr_bins_no_aad if ca.aadhaars else nr_bins_all
                nv_bmap = nv_bins_no_aad if ca.aadhaars else nv_bins_all
                for n1 in nk:
                    for r1 in rk:
                        if (n1, r1) not in dup_nr_pairs:
                            continue
                        for gt in g_targets:
                            bins_iter = nr_bmap.get((n1, r1, gt)) if missing_age else ab_targets
                            if not bins_iter:
                                continue
                            for ab in bins_iter:
                                b = nr_src.get((n1, r1, gt, ab))
                                if b and len(b) <= 128:
                                    cand_cids.update(b)
                    for v1 in vk:
                        if (n1, v1) not in dup_nv_pairs:
                            continue
                        for gt in g_targets:
                            bins_iter = nv_bmap.get((n1, v1, gt)) if missing_age else ab_targets
                            if not bins_iter:
                                continue
                            for ab in bins_iter:
                                b = nv_src.get((n1, v1, gt, ab))
                                if b and len(b) <= 32:
                                    cand_cids.update(b)
                cand_cids.discard(cid_a)
                if not cand_cids:
                    continue

                candidates: list[tuple[float, int]] = []
                tie_20 = 0
                tie_25 = 0
                for cid_b in sorted(cand_cids):
                    if parent[cid_b] != cid_b:
                        continue
                    cb = _get_obj(cid_b)
                    if _has_hard_conflict(ca, cb, max_age_diff):
                        continue

                    f_sim = _best_full_name_sim(ca, cb)
                    g_sim = _best_given_name_sim(ca, cb)
                    r_sim = _best_rel_sim(ca, cb)
                    v_sim = _best_village_sim(ca, cb)
                    cb_dobs = cb.dobs if isinstance(cb, PatientCluster) else ((cb.dob,) if cb.dob else ())
                    dob_match = bool(ca_dobs and cb_dobs and not _disjoint_iter(ca_dobs, cb_dobs))

                    score = 0.0
                    if dob_match and (f_sim >= 0.80 or (g_sim >= 0.85 and r_sim >= 0.78)):
                        score = 3.0 + f_sim
                    elif f_sim >= name_th and r_sim >= rel_th:
                        score = 2.5 + f_sim + r_sim
                    elif f_sim >= name_th and v_sim >= vil_th:
                        score = 2.0 + f_sim + v_sim
                    elif g_sim >= 0.86 and r_sim >= rel_th:
                        score = 1.5 + g_sim + r_sim

                    if score > 0.0:
                        candidates.append((score, cid_b))
                        if score >= 2.5:
                            tie_25 += 1
                        elif score >= 2.0:
                            tie_20 += 1
                        if not ca_dobs and not ca.relative_names and tie_25 == 0 and tie_20 >= 2:
                            break

                if not candidates:
                    continue
                candidates.sort(reverse=True)
                top_score, top_cid = candidates[0]
                if top_score < 2.0 and len(candidates) > 1 and candidates[1][0] >= 1.5:
                    continue
                if 2.0 <= top_score < 2.5 and len(candidates) > 1 and candidates[1][0] >= 2.0:
                    continue

                _merge(cid_a, top_cid)
                merged_this_pass = True
            if not merged_this_pass:
                break

        # Finalize deterministic keyed HMAC pseudonyms across alive roots
        final_roots = [c for c in range(n) if all_clues[c].is_patient_record and parent[c] == c]
        ordered_cids = sorted(final_roots, key=lambda c: _obj_canonical_key(_get_obj(c), c))
        cid_to_psn: dict[int, str] = {}
        psn_clusters: dict[str, PatientCluster | RecordClues] = {}
        for cid in ordered_cids:
            obj = _get_obj(cid)
            if not (obj.mrns or obj.aadhaars or obj.abhas or obj.phones or obj.patient_names):
                continue
            psn = keyring.pseudonym(_obj_canonical_key(obj, cid))
            cid_to_psn[cid] = psn
            psn_clusters[psn] = obj

        unit_to_psn: dict[str, str | None] = {}
        for idx, rc in enumerate(all_clues):
            if not rc.is_patient_record:
                unit_to_psn[rc.unit_id] = None
            else:
                unit_to_psn[rc.unit_id] = cid_to_psn.get(_find_root(idx))

        # Build MRN/Name lookup for cross-reference tables
        ref_lookup: dict[str, str] = {}
        for psn, obj in psn_clusters.items():
            for m in obj.mrns:
                ref_lookup[m] = psn
            for nm in obj.patient_names:
                c_toks = _canonicalize_person_name_cached(nm)
                if len(c_toks) >= 2:
                    ref_lookup[" ".join(c_toks)] = psn

        return unit_to_psn, psn_clusters, ref_lookup
    finally:
        if gc_was_enabled:
            gc.enable()
