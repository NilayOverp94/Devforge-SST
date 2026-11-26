"""Multilingual PII detection engine (English, Devanagari, Hinglish) with clinical guards and ML NER."""
from __future__ import annotations

import base64
import binascii
import json
import re as std_re
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import phonenumbers
import regex
from indic_transliteration import sanscript
from rapidfuzz import fuzz


# --------------------------------------------------------------------------- Script & Phonetic Normalization

# 1-to-1 character translation preserving string length & indices
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_HOMOGLYPHS = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "і": "i", "І": "I",
})

_NUKTA_MAP = [
    ("क़", "k"), ("ख़", "kh"), ("ग़", "g"), ("ज़", "z"), ("ड़", "r"), ("ढ़", "rh"), ("फ़", "f"), ("य़", "y"),
    ("क\u093c", "k"), ("ख\u093c", "kh"), ("ग\u093c", "g"), ("ज\u093c", "z"),
    ("ड\u093c", "r"), ("ढ\u093c", "rh"), ("फ\u093c", "f"), ("य\u093c", "y"),
]

_ITRANS_NUKTA_RE = std_re.compile(r"([a-zA-Z])\.([a-zA-Z])")


def shadow_normalize(text: str) -> str:
    """Length-preserving normalization (Indic digits -> ASCII, homoglyphs -> Latin) for exact-span regex."""
    if text.isascii():
        return text
    return text.translate(_DEVA_DIGITS).translate(_HOMOGLYPHS)


def to_ascii_digits(text: str) -> str:
    if text.isascii():
        return text
    return text.translate(_DEVA_DIGITS)


@lru_cache(maxsize=65536)
def transliterate_deva_to_latin(text: str) -> str:
    """Transliterate Devanagari text to readable Latin using indic-transliteration + nukta handling."""
    if not text or not any("\u0900" <= ch <= "\u097f" for ch in text):
        return text
    s = unicodedata.normalize("NFC", text).translate(_DEVA_DIGITS)
    for src, dst in _NUKTA_MAP:
        s = s.replace(src, dst)
    # Replace anusvara/chandrabindu cleanly before Sanskrit ITRANS to avoid '.n'/'.N' artifacts
    s = s.replace("ँ", "n").replace("ं", "n")
    try:
        out = sanscript.transliterate(s, sanscript.DEVANAGARI, sanscript.ITRANS)
    except Exception:
        out = s
    out = out.replace("~N", "n").replace("~n", "n").replace(".N", "n").replace(".n", "n")
    out = out.replace("Ch", "ch").replace("Sh", "sh").replace("GY", "gy")
    out = _ITRANS_NUKTA_RE.sub(r"\1\2", out)
    # Trim schwa '-a' at the end of multi-char ITRANS words (राम -> rAma -> ram)
    words = []
    for w in out.split():
        wl = w.lower()
        if len(wl) >= 4 and wl.endswith("a") and not wl.endswith(("aa", "ia", "ua", "na", "ta", "da", "ma", "ka", "ra", "sa", "la")):
            wl = wl[:-1]
        words.append(wl)
    return " ".join(words)


_HONORIFIC_TOKENS = {
    "ji", "sahab", "saheb", "saab", "didi", "behenji", "bahenji", "behen", "bhaiya",
    "bhaiyya", "bhai", "babu", "amma", "chachu", "chacha", "mama", "mausi", "bua",
    "kaka", "dada", "dadi", "nana", "nani", "papa", "mummy", "uncle", "aunty", "auntie",
    "mr", "mrs", "ms", "miss", "shri", "shree", "smt", "kumari", "km", "late",
    "sri", "madam", "sir", "s/o", "d/o", "w/o", "c/o", "f/o", "m/o",
}

_REL_PREFIX_RE = std_re.compile(r"(?i)^\s*(?:[sdwfmc]\s*/\s*o|care\s+of)\s+")

_GENERIC_NAME_PARTS = {"devi", "kumari", "kumar", "lal", "singh", "bano", "begum", "khatun", "kaur"}

_HINDI_VARIANT_SUBS = [
    (std_re.compile(r"\b(?:mohd|md|muhd|mohamad|muhammad)\b"), "mohammad"),
    (std_re.compile(r"ee"), "i"),
    (std_re.compile(r"ii"), "i"),
    (std_re.compile(r"oo"), "u"),
    (std_re.compile(r"uu"), "u"),
    (std_re.compile(r"aa"), "a"),
    (std_re.compile(r"ou"), "au"),
    (std_re.compile(r"w"), "v"),
    (std_re.compile(r"ph"), "f"),
    (std_re.compile(r"sh"), "s"),
    (std_re.compile(r"th"), "t"),
    (std_re.compile(r"dh"), "d"),
    (std_re.compile(r"bh"), "b"),
    (std_re.compile(r"kh"), "k"),
    (std_re.compile(r"gh"), "g"),
    (std_re.compile(r"jh"), "j"),
    (std_re.compile(r"chh"), "c"),
    (std_re.compile(r"ch"), "c"),
    (std_re.compile(r"z"), "j"),
    (std_re.compile(r"q"), "k"),
    (std_re.compile(r"([a-z])\1+"), r"\1"),
]


_RX_NON_WORD_SPACE = std_re.compile(r"[^\w\s]")
_RX_NON_WORD = std_re.compile(r"[^\w]")
_RX_DIGITS_ONLY = std_re.compile(r"\D")
_RX_HAS_10_DIGITS = std_re.compile(r"(?:\D*\d){10}")
_RX_ADDR_FOLLOW = std_re.compile(r"^\s*,\s*(?:h\.?no|house|makan|ward|mohalla|vill|gram|tehsil|distt)")
_RX_MRN_EXCLUDE = std_re.compile(r"(?i)(?:ICD|LOINC|MG|SG|SE|OPD|BL)[-/0-9A-Z]*")
_CAND_RX_LIST = (
    regex.compile(r"\b[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,2}\b"),
    regex.compile(r"\b[A-Z]{2,20}\s+[A-Z]{2,20}\b"),
    regex.compile(r"[\p{Devanagari}]{2,20}\s+[\p{Devanagari}]{2,20}"),
)
_RX_CAP_WORD_4 = regex.compile(r"\b[A-Z][a-z]{3,20}\b")


@lru_cache(maxsize=4096)
def _compile_token_rx(tok: str) -> regex.Pattern[str]:
    return regex.compile(r"(?i)(?<!\p{L})" + regex.escape(tok) + r"(?!\p{L})")


@lru_cache(maxsize=65536)
def _canonicalize_person_name_cached(name: str, strip_honorifics: bool = True) -> tuple[str, ...]:
    """Transliterate and canonically normalize a person name into phonetic tokens."""
    if not name:
        return ()
    cleaned = _REL_PREFIX_RE.sub("", name.strip())
    lat = transliterate_deva_to_latin(unicodedata.normalize("NFKC", cleaned))
    lat = _RX_NON_WORD_SPACE.sub(" ", lat.lower())
    tokens = []
    for raw_t in lat.split():
        t = raw_t.strip("_")
        if not t or t.isdigit():
            continue
        if strip_honorifics and t in _HONORIFIC_TOKENS:
            continue
        # Handle ITRANS trailing 'a' schwa and common English spelling endings ('eya' vs 'ey', 'au' vs 'ou')
        for rx, repl in _HINDI_VARIANT_SUBS:
            t = rx.sub(repl, t)
        if len(t) >= 5 and t.endswith("a"):
            t = t[:-1]
        if t.endswith("eya"):
            t = t[:-1]
        if t:
            tokens.append(t)
    return tuple(tokens)


def canonicalize_person_name(name: str, strip_honorifics: bool = True) -> list[str]:
    """Transliterate and canonically normalize a person name into phonetic tokens."""
    return list(_canonicalize_person_name_cached(name, strip_honorifics))


@lru_cache(maxsize=65536)
def canonicalize_place(name: str) -> str:
    if not name:
        return ""
    lat = transliterate_deva_to_latin(unicodedata.normalize("NFKC", name))
    lat = _RX_NON_WORD_SPACE.sub(" ", lat.lower())
    toks = []
    for t in lat.split():
        for rx, repl in _HINDI_VARIANT_SUBS:
            t = rx.sub(repl, t)
        if len(t) >= 5 and t.endswith("a"):
            t = t[:-1]
        if t:
            toks.append(t)
    return " ".join(toks)


_RX_VOWELS = std_re.compile(r"[aeiou]")


@lru_cache(maxsize=65536)
def _consonant_skeleton(tok: str) -> str:
    """Indic consonant skeleton keeping initial vowel (handles Devanagari schwa-deletion variants)."""
    if not tok:
        return ""
    return tok[0] + _RX_VOWELS.sub("", tok[1:])


@lru_cache(maxsize=65536)
def names_similarity(name_a: str, name_b: str) -> float:
    """Cross-script phonetic + Indic consonant-skeleton similarity in [0, 1]."""
    ta = _canonicalize_person_name_cached(name_a)
    tb = _canonicalize_person_name_cached(name_b)
    if not ta or not tb:
        return 0.0
    sa = " ".join(ta)
    sb = " ".join(tb)
    if sa == sb:
        return 1.0
    ska = " ".join(_consonant_skeleton(t) for t in ta)
    skb = " ".join(_consonant_skeleton(t) for t in tb)
    if ska == skb:
        return 0.98
    sk_sim = fuzz.token_sort_ratio(ska, skb) / 100.0
    if len(ta) == 1 and len(tb) == 1:
        return max(fuzz.ratio(sa, sb) / 100.0, sk_sim)
    # Compare sorted tokens & direct token alignment
    s1 = fuzz.token_sort_ratio(sa, sb) / 100.0
    s2 = fuzz.ratio(sa, sb) / 100.0
    # Also check first+last alignment ONLY if given names actually resemble each other
    if len(ta) >= 2 and len(tb) >= 2 and len(ta[0]) >= 2 and len(tb[0]) >= 2:
        first_sim = max(
            fuzz.ratio(ta[0], tb[0]) / 100.0,
            fuzz.ratio(_consonant_skeleton(ta[0]), _consonant_skeleton(tb[0])) / 100.0,
        )
        last_sim = max(
            fuzz.ratio(ta[-1], tb[-1]) / 100.0,
            fuzz.ratio(_consonant_skeleton(ta[-1]), _consonant_skeleton(tb[-1])) / 100.0,
        )
        if first_sim >= 0.72:
            pair_sim = 0.55 * first_sim + 0.45 * last_sim
            return max(s1, s2, sk_sim, pair_sim)
        return min(s1, s2)
    return max(s1, s2, sk_sim)


@lru_cache(maxsize=65536)
def places_similarity(place_a: str, place_b: str) -> float:
    """Cross-script place/district/village similarity handling Devanagari schwa deletion."""
    pa = canonicalize_place(place_a)
    pb = canonicalize_place(place_b)
    if not pa or not pb:
        return 0.0
    if pa == pb:
        return 1.0
    ska = " ".join(_consonant_skeleton(t) for t in pa.split())
    skb = " ".join(_consonant_skeleton(t) for t in pb.split())
    if ska == skb:
        return 0.98
    return max(fuzz.ratio(pa, pb) / 100.0, fuzz.ratio(ska, skb) / 100.0)


# --------------------------------------------------------------------------- Classical ML Name Classifier

class NameSpanClassifier:
    """Lightweight Averaged Perceptron char-ngram + context classifier (<50K sparse params)."""

    def __init__(self, model_path: Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / "name_ner.json"
        self.weights: dict[str, float] = {}
        if model_path.is_file():
            try:
                payload = json.loads(model_path.read_text(encoding="utf-8"))
                self.weights = {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}
            except Exception:
                self.weights = {}

    @staticmethod
    def _norm_tok(tok: str) -> str:
        if any("\u0900" <= c <= "\u097f" for c in tok):
            tok = transliterate_deva_to_latin(tok)
        return std_re.sub(r"[^a-z0-9]", "", tok.lower())

    def extract_features(self, span_text: str, left_ctx: str = "", right_ctx: str = "") -> dict[str, float]:
        feats: dict[str, float] = {"bias": 1.0}
        raw_tokens = span_text.strip().split()
        if not raw_tokens:
            return feats
        feats[f"ntok={min(len(raw_tokens), 4)}"] = 1.0
        has_deva = any("\u0900" <= c <= "\u097f" for c in span_text)
        feats["script=deva" if has_deva else "script=latin"] = 1.0
        feats["all_title"] = 1.0 if all(t[:1].isupper() and t[1:].islower() for t in raw_tokens if t.isalpha()) else 0.0
        feats["all_upper"] = 1.0 if all(t.isupper() for t in raw_tokens if t.isalpha()) else 0.0
        feats["has_digit"] = 1.0 if any(c.isdigit() for c in span_text) else 0.0
        feats["has_apostrophe"] = 1.0 if "'" in span_text or "’" in span_text else 0.0

        norm_tokens = [self._norm_tok(t) for t in raw_tokens]
        norm_tokens = [t for t in norm_tokens if t]
        for idx, nt in enumerate(norm_tokens):
            pos_tag = "first" if idx == 0 else ("last" if idx == len(norm_tokens) - 1 else "mid")
            if len(nt) >= 2:
                feats[f"w:{nt}"] = 1.0
                feats[f"p2:{pos_tag}:{nt[:2]}"] = 1.0
                feats[f"s2:{pos_tag}:{nt[-2:]}"] = 1.0
            if len(nt) >= 3:
                feats[f"p3:{pos_tag}:{nt[:3]}"] = 1.0
                feats[f"s3:{pos_tag}:{nt[-3:]}"] = 1.0
            if len(nt) >= 4:
                feats[f"s4:{nt[-4:]}"] = 1.0
            for g in (2, 3, 4):
                if len(nt) >= g:
                    for i in range(len(nt) - g + 1):
                        feats[f"ng{g}:{nt[i:i+g]}"] = 1.0

        lc = left_ctx.strip().lower()
        rc = right_ctx.strip().lower()
        if lc:
            for part in lc.split()[-2:]:
                feats[f"L:{part}"] = 1.0
            feats[f"Lfull:{lc}"] = 1.0
        if rc:
            for part in rc.split()[:2]:
                feats[f"R:{part}"] = 1.0
            feats[f"Rfull:{rc}"] = 1.0
        return feats

    def score(self, span_text: str, left_ctx: str = "", right_ctx: str = "") -> float:
        if not self.weights:
            return 0.0
        feats = self.extract_features(span_text, left_ctx, right_ctx)
        return sum(self.weights.get(k, 0.0) * v for k, v in feats.items())


# --------------------------------------------------------------------------- Gazetteers & Medical Lexicons

# General Indian given/family name gazetteer (Latin lowercase) - excludes ambiguous clinical words
_LATIN_NAME_GAZETTEER = {
    "aarav", "abhishek", "aditya", "ajay", "akash", "akram", "alok", "amit", "amrit", "anil",
    "anita", "aneeta", "anju", "ananya", "anthony", "archana", "arjun", "arun", "asha", "ashok",
    "aslam", "ayesha", "aisha", "babita", "balwinder", "bano", "bansal", "bhardwaj", "bhavna",
    "bhattacharya", "bhatnagar", "bisht", "bose", "brajesh", "brijesh", "chandra", "charanjit",
    "chatterjee", "chauhan", "chouhan", "chaudhary", "choudhary", "chetan", "das", "david", "deepak",
    "desai", "devendra", "devi", "dhillon", "dinesh", "diya", "dubey", "dutta", "dwivedi",
    "faisal", "fareeda", "farida", "fatima", "fatma", "ganesh", "gaurav", "geeta", "gita", "gill",
    "girish", "goyal", "grewal", "gupta", "hamid", "hareesh", "harish", "harjeet", "harpreet",
    "heena", "hemant", "ilavenil", "imran", "indra", "indu", "iqbal", "irfan", "ishaan", "iyer",
    "jagdish", "jain", "jaiswal", "jaspreet", "javed", "jitendra", "joseph", "joshi", "jyoti",
    "kabir", "kalpana", "kamal", "kamala", "kamla", "kamlesh", "karan", "kaur", "kaveri", "kavita",
    "keshav", "khalid", "khan", "khatun", "kiran", "kishan", "krishna", "krishnan", "kuldeep",
    "kulshrestha", "kumar", "kumari", "kunal", "kushwaha", "kusum", "lakshmi", "lal", "lalit",
    "laxmi", "lodhi", "madhav", "madhu", "mahendra", "mahesh", "majid", "malik", "mamta", "manju",
    "manoj", "manpreet", "masih", "mathur", "maurya", "mayank", "meena", "meera", "mehta", "menon",
    "mina", "mirza", "mishra", "misra", "mittal", "mohammad", "mohan", "mohd", "mukherjee", "mukesh",
    "murthy", "naidu", "nair", "nambiar", "nargis", "naresh", "narendra", "naseem", "nasreen",
    "navdeep", "navya", "neha", "nigam", "nikhil", "nisha", "nishad", "nitin", "nitya", "pande",
    "pandey", "pandeya", "pankaj", "parminder", "parvati", "parvathi", "paswan", "patel", "pathak",
    "pathan", "patil", "pooja", "prabha", "pradeep", "prakash", "pramod", "pranav", "prem", "priya",
    "puja", "qadri", "qureshi", "radha", "rahul", "rajesh", "rajput", "raju", "rakesh", "ramesh",
    "ranjana", "rashid", "rastogi", "rathore", "ravi", "ravindra", "rawat", "reddy", "reena",
    "reeta", "rekha", "renu", "rina", "rishabh", "rita", "riya", "rizvi", "rizwan", "rohan",
    "rohit", "roy", "rukhsana", "sahu", "sajid", "saleem", "salim", "salma", "salman", "sandhu",
    "sanjay", "sarita", "sarma", "satish", "savita", "saxena", "seema", "sen", "shabana", "shah",
    "shahid", "shahnaz", "shaikh", "shanti", "shanthi", "sharma", "sheikh", "shiv", "shobha",
    "shubham", "shukla", "siddharth", "siddiqui", "siddiqi", "sidhu", "sima", "simran", "singh",
    "sita", "sneha", "sodhi", "sohan", "srivastava", "subhash", "sudha", "sukhvinder", "sukhwinder",
    "sultana", "suman", "suneeta", "sunil", "sunita", "surendra", "suresh", "surya", "swati",
    "syed", "tara", "tariq", "tarun", "tewari", "thakur", "thomas", "tiwari", "tripathi", "tsering",
    "tushar", "umesh", "usha", "vaibhav", "vandana", "varma", "varun", "verma", "vijay", "vikas",
    "vikram", "vimal", "vinod", "vivek", "wangmo", "wasim", "yadav", "yasmin", "yogesh", "yusuf",
    "zaidi", "zainab", "zenab", "zorawar", "zubair",
}

_DEVA_NAME_GAZETTEER = {
    "कमला", "रीना", "हरीश", "नेहा", "कमल", "ज़ैनब", "अनीता", "महेश", "मुकेश", "फ़ातिमा",
    "सरिता", "शांति", "गीता", "मोहम्मद", "अशोक", "दिनेश", "मनोज", "पार्वती", "सुनीता",
    "सुखविंदर", "सलमा", "शबाना", "उषा", "कविता", "विजय", "बृजेश", "ब्रजेश", "राहुल",
    "रमेश", "सुरेश", "अजय", "प्रकाश", "किशन", "दीपक", "जगदीश", "नरेश", "रवि", "ज़ुबैर",
    "अकरम", "यूसुफ", "कुलदीप", "जावेद", "अनिल", "रोहित", "सोहन", "मीरा", "आशा", "राधा",
    "निषाद", "पांडे", "पासवान", "ठाकुर", "सिद्दीकी", "शर्मा", "चौहान", "क़ुरैशी", "तिवारी",
    "चौधरी", "यादव", "लोधी", "पटेल", "मिश्रा", "राजपूत", "कौर", "अंसारी", "बानो",
    "ख़ातून", "शेख़", "कुमार", "वर्मा", "गुप्ता", "सिंह", "देवी", "खान", "मलिक", "कादरी",
    "इमरान", "आरिफ", "इरफ़ान", "सलीम", "नसीम", "रोहित", "अमित", "सुनील", "अनिल",
    "अरुण", "वरुण", "संजय", "सुमन", "किरण", "रेखा", "पूजा", "प्रिया", "स्वाति", "ज्योति",
}

# Clinical / Operational non-PII words that MUST NOT be redacted as person names
_NON_PII_WORDS = {
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "morning", "evening", "night", "next", "review",
    "discharge", "summary", "nursing", "notes", "note", "opd", "ipd", "patient", "name",
    "age", "sex", "gender", "dob", "date", "visit", "ward", "bed", "medicine", "surgery",
    "pulmonology", "neurology", "cardiology", "orthopedics", "pediatrics", "gynaecology",
    "general", "emergency", "icu", "hospital", "multispeciality", "charitable", "district",
    "seva", "sadan", "clinic", "pathology", "diagnostics", "diagnostic", "labs", "lab",
    "lifeline", "pathkind", "city", "sanjeevani", "jeevan", "jyoti", "arogya", "hardoi",
    "lucknow", "barabanki", "sitapur", "sultanpur", "unnao", "bahraich", "rae", "bareli",
    "raebareli", "uttar", "pradesh", "india", "worker", "workers", "dr", "doctor",
    "consultant", "referred", "author", "practitioner", "organization", "composition",
    "encounter", "condition", "observation", "procedure", "bundle", "document", "referral",
    "letter", "consent", "surgical", "helpdesk", "system", "export", "lims", "bulk",
    "extract", "routine", "panel", "panels", "results", "result", "remarks", "advice",
    "adv", "rx", "tab", "cap", "inj", "syr", "sachet", "ors", "dolo", "paracetamol",
    "reg", "no", "uhid", "mrn", "pin", "zip", "unit", "range", "loinc", "test", "value",
    "aadhaar", "aadhar", "abha", "pan", "id", "code", "status", "amount", "inr",
    "generalised", "generalized", "of", "for", "from", "to", "with", "and", "the", "in",
    "on", "at", "by", "is", "are", "was", "were", "has", "have", "had",
    "schedule", "form", "reference", "weight", "kg", "height", "bmi", "spo2", "temp", "bp",
    "pulse", "visits", "visit", "patients", "patient_ref", "exported", "generated", "meta",
    "source", "resource", "resourcetype", "fullurl", "entry", "subject", "display", "coding",
    "identifier", "telecom", "address", "contact", "extension", "birth", "birthdate", "postalcode",
    "authoredon", "effectivedatetime", "performeddatetime", "recordeddate", "clinicalstatus",
    "dosageinstruction", "medicationcodeableconcept", "valuequantity", "lastupdated", "serviceprovider",
    "contenttype", "attachment", "data", "title", "section", "div", "text", "status", "intent",
    "category", "tickets", "ticket", "raised", "callback", "week", "wards", "beds", "occupied",
    "dengue", "cases", "sample", "collected", "reported", "bill", "package", "admission", "drug",
    "strength", "frequency", "duration", "department", "facility", "schema", "time", "trip",
    "from", "to", "by", "at", "on", "id_no", "hos_id",
    "दिन", "से", "है", "हैं", "को", "का", "की", "के", "में", "पर", "और", "था", "थी", "थे",
    "ने", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस",
    "सं", "नं", "पिन", "यूनिट", "तथा", "एवं",
    "amlodipine", "telmisartan", "ramipril", "metformin", "glimepiride", "azithromycin",
    "ceftriaxone", "doxycycline", "pantoprazole", "levothyroxine", "atorvastatin",
    "artesunate", "ferrous", "sulphate", "folic", "acid", "levodopa", "carbidopa",
    "insulin", "glargine", "rifampicin", "isoniazid", "haemoglobin", "hemoglobin",
    "random", "blood", "sugar", "serum", "creatinine", "platelet", "count", "total",
    "wbc", "sodium", "hba1c", "tsh", "bp", "pulse", "temp", "spo2", "weight", "fever",
    "pain", "cough", "headache", "giddiness", "vomiting", "typhoid", "malaria", "dengue",
    "asthma", "pneumonia", "tuberculosis", "anaemia", "anemia", "hypertension",
    "diabetes", "mellitus", "hypothyroidism", "kidney", "disease", "syndrome", "palsy",
    "addison", "crohn", "parkinson", "bell", "wilson", "alzheimer", "unspecified",
    "essential", "primary", "chronic", "infectious", "gastroenteritis", "colitis",
    "visceral", "leishmaniasis", "kala", "azar", "rabies", "atherosclerotic", "heart",
    "urinary", "tract", "infection", "cataract", "phacoemulsification", "appendicectomy",
    "laparoscopic", "cholecystectomy", "inguinal", "hernia", "repair", "caesarean",
    "section", "haemodialysis", "session", "deep", "brain", "stimulation", "whipple",
    "stable", "condition", "treated", "continue", "low", "salt", "diet", "weeks", "days",
    "month", "months", "yrs", "years", "forwarded", "attachment", "report", "protocol",
    "version", "ns1", "antigen", "nsaids", "differential", "nodal", "officer", "crisis",
    "tehsil", "distt", "vill", "gram", "mohalla", "house", "makan", "mandir", "masjid",
    "gurudwara", "school", "bhawan", "tanki", "chowk", "colony", "basti", "nagar",
    "approved", "pending", "settled", "closed", "active", "final", "current", "male",
    "female", "unknown", "other", "none", "null", "hindi", "english", "hinglish",
    "मरीज़", "रोगी", "नाम", "आयु", "वर्ष", "महिला", "पुरुष", "पिता", "पति", "माता",
    "पता", "ग्राम", "मकान", "तहसील", "जिला", "मोबाइल", "भर्ती", "छुट्टी", "तारीख",
    "निदान", "डिस्चार्ज", "सारांश", "पर्चा", "ओपीडी", "पंजीकरण", "आईपीडी", "बुखार",
    "दिन", "दर्द", "चक्कर", "उल्टी", "दस्त", "पेशाब", "संक्रमण", "फेफड़ों", "टीबी",
    "किडनी", "बीमारी", "सीओपीडी", "रक्तचाप", "डायबिटीज़", "दमा", "रेबीज़", "निमोनिया",
    "मलेरिया", "काला-अज़ार", "मंदिर", "मस्जिद", "गुरुद्वारे", "पंचायत", "भवन", "टंकी",
}

_SYMPTOM_WORDS_RE = regex.compile(
    r"(?i)^(?:"
    r"(?:generalised|generalized|chronic|severe|mild|high|low|acute|persistent|recurrent|intermittent)\s+)?"
    r"(?:"
    r"fever|pain|cough|cold|headache|giddiness|dizziness|vomiting|nausea|loose\s+stools?|"
    r"diarrh(?:oea|ea)|dysuria|burning|breathlessness|dyspnoea|chest|abdomen|abdominal|"
    r"weakness|fatigue|swelling|rash|itching|chills|rigors|bleeding|edema|joint|back|"
    r"sore\s+throat|earache|pet\s+mein|chakkar|ulti|dast|bukhar|dard|khansi|sujan|"
    r"कमजोरी|बुखार|दर्द|खांसी|उल्टी|दस्त|चक्कर|सूजन|सांस"
    r")\b"
)


@dataclass
class PIISpan:
    start: int
    end: int
    pii_type: str
    surface: str
    normalized: str = ""
    role: str = "patient"  # patient | relative


# --------------------------------------------------------------------------- Regex Definitions

# Direct identifiers (run on length-preserving shadow_normalize text)
_RX_ABHA_NUM = regex.compile(r"(?<!\d)(?:91|\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)")
_RX_ABHA_ADDR = regex.compile(r"(?i)\b[a-z0-9._-]{3,}@(?:abdm|sbx|ndhm)\b")
_RX_AADHAAR = regex.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_RX_AADHAAR_MASKED = regex.compile(r"(?i)(?<![a-z0-9])[x*]{4}[\s-]?[x*]{4}[\s-]?\d{4}(?!\d)")
_RX_PHONE_1 = regex.compile(r"(?<!\d)(?:\+?91[\s.-]?|0)?[6-9]\d{4}[\s.-]?\d{5}(?!\d)")
_RX_PHONE_2 = regex.compile(r"(?<!\d)(?:\+?91[\s.-]?|0)?[6-9]\d{2}[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_RX_PHONE_0 = regex.compile(r"(?<!\d)0[6-9]\d{4}[-\s]?\d{5}(?!\d)")
_RX_EMAIL = regex.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_RX_PAN = regex.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_RX_INSURANCE = regex.compile(
    r"(?i)\b(?:PMJAY[-\s/]?[A-Z]{2}[-\s/]?\d{6,12}|POL[/\-]\d{2,4}[/\-]\d{5,12})\b"
)

# MRN / UHID / IPD / ADM (known + unseen hospital prefixes; excludes OPD/, BL/, AMB-, HD-, ICD, LOINC, package codes)
_RX_MRN_KNOWN = regex.compile(
    r"(?i)\b(?:SNJ|JJH|UHID|DHH|IPD|IP|ADM|MRN|REG|PAT)(?:[-/]?[A-Z]{2,5})?[-/]?(?:20\d{2}[-/]|\d{2}[-/])?\d{4,8}\b"
)
_RX_MRN_GENERIC = regex.compile(
    r"\b(?!OPD/|BL/|AMB-|HD-|PMJAY|POL/|LOINC|ICD|COVID)[A-Z]{2,6}(?:[-/][A-Z]{2,5})?[-/](?:20\d{2}[-/]|\d{2}[-/])?\d{4,8}\b"
)
_RX_MRN_LABELED = regex.compile(
    r"(?i)(?<=\b(?:UHID|MRN|IPD\s*No\.?|Reg\.?\s*No\.?|Patient\s*ID|Patient\s*Ref|पंजीकरण\s*सं\.?|आईपीडी\s*सं\.?)\s*[:\-]\s*)[A-Z0-9][A-Z0-9/\-]{3,20}"
)

# Preserve / meaning-trap guards
_RX_DOCTOR_LAT = regex.compile(
    r"(?:"
    r"(?:Consultant|Referred\s+by|Ref\.?\s*By|Reviewed\s+by|Author)\s*:\s*)?"
    r"\b(?:Dr\.?|Prof\.?)\s+[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3}"
    r"(?:\s*\((?:MD|MS|MBBS|DNB|DM|MCh|BDS|MDS)[^)]*\))?",
)
_RX_DOCTOR_DEVA = regex.compile(
    r"(?:डॉ\.?|डॉक्टर)\s*[\p{Devanagari}]+(?:\s+[\p{Devanagari}]+){0,3}"
)
_RX_EPONYM = regex.compile(
    r"(?i)\b[a-z\u00c0-\u024f]+(?:'s|’s|s')?\s+"
    r"(?:disease|syndrome|palsy|lymphoma|dystrophy|triad|sign|phenomenon|fracture|aneurysm|reflex|coma|chorea|tetrad|crisis)\b"
)
_RX_HISTORY = regex.compile(
    r"(?i)\b(?:K/C/O|H/O|P/H/O|F/H/O|K\.C\.O\.|H\.O\.)\s*[:\-]?\s*[^\n.;]+"
)
_RX_HOSPITAL_WARD = regex.compile(
    r"(?i)(?:(?:Ward/Bed|Ward|NURSING\s+NOTES\s*-|admission\s+to)\s*:?\s*)?"
    r"\bWard\s+\d+\s*\([^)]+\)(?:\s*/\s*Bed\s+\d+)?"
    r"|\bWard/Bed\s*:\s*Ward\s+\d+(?:\s*/\s*Bed\s+\d+)?"
    r"|\badmission\s+to\s+Ward\s+\d+\b"
)
_RX_ASHA = regex.compile(r"\bASHA\s+(?:worker|workers|didi|bai)\b|\bASHA\b(?=\s+(?:worker|workers))")
_RX_CLINICAL_NUMS = regex.compile(
    r"(?<![\dA-Za-z])(?:"
    r"\d{1,3}-\d{1,3}-\d{1,3}(?:-\d{1,3})?"  # 1-0-1
    r"|(?i:\bBP\s*:?\s*)\d{2,3}/\d{2,3}"
    r"|\d+(?:\.\d+)?\s*(?i:mg/dL|g/dL|mmol/L|mIU/L|/uL|/min|mg|mcg|ug|gm|ml|iu|units?|°[FC])\b"
    r"|[A-Z]\d{2}(?:\.\d{1,2})?"  # Strictly uppercase ICD-10
    r")(?![\dA-Za-z])"
)
_RX_LAB_RANGE = regex.compile(
    r"(?<![\d\-/])\d{1,3}(?:\.\d{1,2})?\s*[-–]\s*\d{1,3}(?:\.\d{1,2})?(?![\d\-/])"
    r"|(?<![\d\-/])\d{4,6}\s*[-–]\s*\d{4,6}(?![\d\-/])"
)
_RX_HOSPITALS = regex.compile(
    r"(?i)\b(?:Sanjeevani\s+Multispeciality\s+Hospital|Jeevan\s+Jyoti\s+Charitable\s+Hospital|"
    r"Arogya\s+Seva\s+Sadan|District\s+Hospital\s+[A-Z][a-z]+|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+(?:Hospital|Diagnostics|Pathology|Labs|Clinic|Medical\s+Centre))\b"
)

# Name & Relative Label Anchors (Latin + Devanagari + Hinglish)
_RX_PATIENT_LABEL = regex.compile(
    r"(?im)(?<=^(?:Patient'?s?\s*Name|Full\s*Name|Name|Pt\.?\s*Name|Subject|मरीज़\s*का\s*नाम|रोगी\s*का\s*नाम|नाम)\s*:\s*)"
    r"[^\n\t|;,]+?(?=\s{2,}|\s+(?:Age|Sex|Gender|UHID|MRN|Reg|Mob|Phone|DOB|आयु|लिंग|पता)\b|$|\n)"
)
_RX_INLINE_PATIENT_ANCHOR = regex.compile(
    r"(?i)(?<=\b(?:Report\s+of|OPD\s+note\s+for|Rx\s+of|Referral|Patient)\s*:\s*|\b(?:Report\s+of|OPD\s+note\s+for|Rx\s+of)\s+)"
    r"[\p{L}\p{M}]+(?:\s+[\p{L}\p{M}]+){0,2}"
    r"(?=\s+(?:ji|sahab|saheb|saab|didi|behenji|bahenji|bhaiya|babu|amma|\d+|Y\s*/|yrs|age|[\(\.,\n])|\s*$)"
)
_RX_REL_LABEL = regex.compile(
    r"(?i)(?<=\b(?:S/o|D/o|W/o|F/o|M/o|S/O|D/O|W/O|Father(?:'s)?\s+Name|Husband(?:'s)?\s+Name|Spouse|Guardian|Attendant|"
    r"पिता\s*का\s*नाम|पति\s*का\s*नाम|माता\s*का\s*नाम|अभिभावक)\s*[:\(]?\s*)"
    r"(?:(?:S/o|D/o|W/o|C/O)\s+)?"
    r"[\p{L}\p{M}.]+(?:\s+[\p{L}\p{M}.]+){0,2}"
    r"(?=\s*(?:[,;\)\n\.]|\b(?:Mob|Mobile|Phone|Ward|H\.?No|House|Makan|informed)\b|$))"
)
_RX_CO_CARE_OF = regex.compile(
    r"(?i)\bC/O\s*[:\-]?\s*([\p{L}\p{M}.]+(?:\s+[\p{L}\p{M}.]+){0,2})"
)
_RX_HONORIFIC_NAME = regex.compile(
    r"(?i)\b((?!(?:of|for|from|to|with|and|the|report|patient|mr|mrs|ms|dr)\b)[\p{L}\p{M}]{2,20}"
    r"(?:\s+(?!(?:ji|sahab|saheb|saab|didi|behenji|bahenji|behen|bhaiya|bhaiyya|bhai|babu|amma|chachu|chacha|mama)\b)[\p{L}\p{M}]{2,20})?)\s+"
    r"(ji|sahab|saheb|saab|didi|behenji|bahenji|behen|bhaiya|bhaiyya|bhai|babu|amma|chachu|chacha|mama|जी|साहब|दीदी|बहनजी|भैया|बाबू)\b"
)
_RX_PREFIX_HONORIFIC_NAME = regex.compile(
    r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Shri|Shree|Smt\.?|Kumari|श्री|श्रीमती|सुश्री)\s+"
    r"([\p{L}\p{M}]{2,20}(?:\s+[\p{L}\p{M}]{2,20}){0,2})"
)
_RX_WHATSAPP_SENDER = regex.compile(
    r"(?m)^\[\d{1,2}/\d{1,2}/\d{2,4},\s*[^\]]+\]\s*([^\n:]{2,40}?)(?=:)"
)
_RX_DEVA_POSTPOS = regex.compile(
    r"([\p{Devanagari}]{2,20}(?:\s+[\p{Devanagari}]{2,20}){0,2})\s+(?:को\s+\d+\s+दिन|का\s+BP|की|ने)\b"
)
_RX_HINGLISH_POSTPOS = regex.compile(
    r"\b([A-Z][a-z]{2,20}(?:\s+[A-Z][a-z]{2,20}){0,2})(?:\s+(?:ka\s+BP|ki\s+|ko\s+\d+|ne\s+(?:raat|subah|kaha))|'s\s+attendant)\b"
)
_RX_FRAG_REL = regex.compile(
    r"(?m)^([A-Z\p{Devanagari}][\p{L}\p{M}]{1,20}(?:\s+[\p{L}\p{M}]{1,20})?)\s*,\s*(?:S/o|D/o|W/o)\s+([A-Z\p{Devanagari}][\p{L}\p{M}]{1,20}(?:\s+[\p{L}\p{M}]{1,20})?)"
)
_RX_ATTENDANT_CONTEXT = regex.compile(
    r"(?i)(?:\bcalled\s+by|\bwitnessed\s+by|\braised\s+by|\bsubmitted\s+by|\bcontact\s+person)\s+([A-Z\p{Devanagari}][\p{L}\p{M}]{1,20}(?:\s+[A-Z\p{Devanagari}][\p{L}\p{M}]{1,20}){0,2})"
)
_RX_PAREN_ATTENDANT = regex.compile(
    r"([A-Z\p{Devanagari}][\p{L}\p{M}]{1,20}(?:\s+[A-Z\p{Devanagari}][\p{L}\p{M}]{1,20}){0,2})\s*\(\s*attendant\b"
)

# Address components (Latin & Devanagari)
_RX_HOUSE = regex.compile(
    r"(?i)(?:\b(?:H\.?\s*No\.?|House(?:\s+No\.?)?|Makan(?:\s+No\.?)?|Flat(?:\s+No\.?)?|Plot(?:\s+No\.?)?)\s*\.?\s*[A-Za-z0-9/\-]*\d+[A-Za-z0-9/\-]*"
    r"|मकान\s*नं\.?\s*[0-9०-९A-Za-z/\-]*[0-9०-९][0-9०-९A-Za-z/\-]*)"
)
_RX_LOCALITY = regex.compile(
    r"(?i)(?:\bMohalla\s+[^,\n;]+?(?=,|\s+(?:near|opp|behind|Tehsil|Distt|Vill|Gram)\b|$|\n)"
    r"|मोहल्ला\s+[^,\n;]+?(?=,|\s+(?:के\s+पास|के\s+पीछे|के\s+सामने|तहसील|जिला|ग्राम)\b|$|\n))"
)
_RX_ADDR_WARD = regex.compile(r"(?i)\bWard\s+\d{1,3}\b(?!\s*\()")
_RX_LANDMARK = regex.compile(
    r"(?i)(?:\b(?:near|opp\.?|opposite|behind|beside)\s+[^,\n;]+?(?=,|\s+(?:Tehsil|Distt|District|Vill|Gram)\b|$|\n)"
    r"|[^\n,;]+?\s+के\s+(?:पास|पीछे|सामने)(?=,|\s+(?:ग्राम|तहसील|जिला)\b|$|\n))"
)
_RX_VILLAGE_TEXT = regex.compile(
    r"(?i)(?:\b(?:Vill\.?|Village|Gram|Gaon)\s*[:\-]?\s*[^,\n;]+?(?=,|\s+(?:Tehsil|Distt|District|PIN)\b|$|\n)"
    r"|(?:ग्राम|गाँव|गांव)\s*[:\-]?\s*[^,\n;]+?(?=,|\s+(?:तहसील|जिला|पिन)\b|$|\n))"
)
_RX_VILLAGE_BEFORE_TEHSIL = regex.compile(
    r"(?i)(?<=,\s*)(?!(?:House|H\.?\s*No|Makan|Ward|Mohalla|near|opp\.?|behind|beside|मकान|वार्ड|मोहल्ला)\b)"
    r"[A-Za-z\p{Devanagari}][A-Za-z\p{Devanagari}\s.-]{2,35}?(?=\s*,\s*(?:Tehsil|तहसील)\b)"
)
_RX_TEHSIL_TEXT = regex.compile(
    r"(?i)(?:\bTehsil\s+[^,\n;]+?(?=,|\s+(?:Distt|District|PIN)\b|$|\n)|तहसील\s+[^,\n;]+?(?=,|\s+(?:जिला|पिन)\b|$|\n))"
)
_RX_PIN_ADDR = regex.compile(r"(?<!\d)([1-8]\d{5})(?!\d)")


class _IntervalSet:
    """Fast O(log M) interval overlap checker with lazy batch merging."""

    __slots__ = ("_starts", "_ends", "_pending")

    def __init__(self, intervals: list[tuple[int, int]] | None = None) -> None:
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._pending: list[tuple[int, int]] = []
        if intervals:
            self._rebuild(intervals)

    def _rebuild(self, intervals: list[tuple[int, int]]) -> None:
        if not intervals:
            return
        s_iv = sorted(intervals)
        starts = [s_iv[0][0]]
        ends = [s_iv[0][1]]
        for s, e in s_iv[1:]:
            if s <= ends[-1]:
                if e > ends[-1]:
                    ends[-1] = e
            else:
                starts.append(s)
                ends.append(e)
        self._starts = starts
        self._ends = ends

    def _flush(self) -> None:
        if not self._pending:
            return
        all_iv = list(zip(self._starts, self._ends)) + self._pending
        self._pending.clear()
        self._rebuild(all_iv)

    def add(self, start: int, end: int) -> None:
        if start >= end:
            return
        self._pending.append((start, end))
        if len(self._pending) >= 64:
            self._flush()

    def extend(self, intervals: list[tuple[int, int]]) -> None:
        if not intervals:
            return
        self._pending.extend(intervals)
        if len(self._pending) >= 64:
            self._flush()

    def overlaps(self, start: int, end: int) -> bool:
        if self._starts:
            idx = bisect_left(self._starts, end) - 1
            if idx >= 0 and self._ends[idx] > start:
                return True
        for s, e in self._pending:
            if start < e and end > s:
                return True
        return False


def _overlaps_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if start < e and end > s:
            return True
    return False


def _add_span(out: list[PIISpan], start: int, end: int, ptype: str, orig: str, norm: str = "", role: str = "patient") -> None:
    while start < end and orig[start].isspace():
        start += 1
    while end > start and orig[end - 1].isspace():
        end -= 1
    if start < end and orig[start:end].strip():
        out.append(PIISpan(start=start, end=end, pii_type=ptype, surface=orig[start:end], normalized=norm, role=role))


def _is_non_pii_phrase(phrase: str) -> bool:
    toks = [t.strip(".,:;()[]").lower() for t in phrase.split() if t.strip(".,:;()[]")]
    if not toks:
        return True
    if all(t in _NON_PII_WORDS or t.isdigit() for t in toks):
        return True
    # Any token ending with medical drug/disease suffixes?
    if any(t.endswith(("mycin", "prazole", "statin", "sartan", "dipine", "formin", "cillin", "itis", "osis", "emia", "pathy", "ectomy", "otomy")) for t in toks):
        return True
    return False


class PIIDetector:
    """Unified multilingual PII detector with medical preservation and classical ML disambiguation."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.ml = NameSpanClassifier(model_path=model_path)
        self._id_pii_cache: dict[str, bool] = {}

    def protected_spans(self, text: str) -> list[tuple[int, int]]:
        """Find clinical spans (doctors, eponyms, K/C/O, C/O complaints, wards, doses, labs, ASHA) to preserve."""
        spans: list[tuple[int, int]] = []
        for rx in (_RX_DOCTOR_LAT, _RX_DOCTOR_DEVA, _RX_EPONYM, _RX_HISTORY, _RX_HOSPITAL_WARD, _RX_ASHA, _RX_CLINICAL_NUMS, _RX_HOSPITALS):
            for m in rx.finditer(text):
                spans.append((m.start(), m.end()))
        for m in _RX_LAB_RANGE.finditer(text):
            # Only protect numeric range if it's NOT a 10-digit Indian mobile (5-5 split)
            digits = _RX_DIGITS_ONLY.sub("", m.group(0))
            if not (len(digits) == 10 and digits[0] in "6789"):
                spans.append((m.start(), m.end()))

        # C/O <symptom> (only protect if NOT a relative/care-of!)
        for m in _RX_CO_CARE_OF.finditer(text):
            after = m.group(1).strip()
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_prefix = text[line_start:m.start()].lower()
            is_addr_ctx = any(k in line_prefix for k in ("address", "pata", "attendant", "guardian", "पता"))
            after_comma = text[m.end():m.end() + 30].lower()
            has_addr_follow = bool(_RX_ADDR_FOLLOW.search(after_comma))
            if not is_addr_ctx and not has_addr_follow and _SYMPTOM_WORDS_RE.match(after):
                # Protect the whole C/O complaint phrase up to newline/semicolon
                end_line = text.find("\n", m.start())
                if end_line == -1:
                    end_line = len(text)
                spans.append((m.start(), end_line))
        return spans

    def looks_like_person_name(self, text: str, left_ctx: str = "", right_ctx: str = "") -> bool:
        s = text.strip(" \t\r\n.,:;()[]{}\"'")
        if not s or len(s) < 2 or any(c.isdigit() for c in s):
            return False
        if _is_non_pii_phrase(s):
            return False
        toks = s.split()
        if len(toks) > 4:
            return False
        lower_toks = [_RX_NON_WORD.sub("", t.lower()) for t in toks]
        # Gazetteer hit (Latin or Devanagari)
        if any(t in _LATIN_NAME_GAZETTEER for t in lower_toks if len(t) >= 3):
            return True
        if any(t in _DEVA_NAME_GAZETTEER for t in toks):
            return True
        # Transliterated Devanagari check against gazetteer
        if any("\u0900" <= c <= "\u097f" for c in s):
            canon = canonicalize_person_name(s)
            if any(c in _LATIN_NAME_GAZETTEER for c in canon if len(c) >= 3):
                return True
        # Classical ML classifier score (catches unseen names & canaries!)
        # Require multi-word span or high confidence margin to avoid false positives on short labels
        thresh = 0.45 if len(toks) >= 2 else 1.25
        if self.ml.score(s, left_ctx, right_ctx) >= thresh:
            return True
        return False

    def detect_text(
        self,
        text: str,
        extra_patient_names: list[str] | None = None,
        extra_relative_names: list[str] | None = None,
        allow_pin_scan: bool = True,
    ) -> list[PIISpan]:
        """Detect all PII spans in arbitrary English/Hindi/Hinglish text while respecting clinical guards."""
        if not text:
            return []
        shadow = shadow_normalize(text)
        prot = self.protected_spans(text)
        prot_iv = _IntervalSet(prot)
        spans: list[PIISpan] = []

        # 1. Structured numeric & alphanumeric direct identifiers (highest priority - never blocked by lab range guards)
        for m in _RX_ABHA_NUM.finditer(shadow):
            _add_span(spans, m.start(), m.end(), "ABHA_NUMBER", text, _RX_DIGITS_ONLY.sub("", m.group(0)))

        for m in _RX_ABHA_ADDR.finditer(shadow):
            _add_span(spans, m.start(), m.end(), "ABHA_ADDRESS", text, m.group(0).lower())

        for m in _RX_AADHAAR_MASKED.finditer(shadow):
            _add_span(spans, m.start(), m.end(), "AADHAAR_MASKED", text, m.group(0).lower())

        span_iv = _IntervalSet([(s.start, s.end) for s in spans])
        for m in _RX_AADHAAR.finditer(shadow):
            if not span_iv.overlaps(m.start(), m.end()):
                prev_len = len(spans)
                _add_span(spans, m.start(), m.end(), "AADHAAR", text, _RX_DIGITS_ONLY.sub("", m.group(0)))
                if len(spans) > prev_len:
                    span_iv.add(spans[-1].start, spans[-1].end)

        for m in _RX_EMAIL.finditer(shadow):
            _add_span(spans, m.start(), m.end(), "EMAIL", text, m.group(0).lower())

        for m in _RX_PAN.finditer(shadow):
            _add_span(spans, m.start(), m.end(), "PAN", text, m.group(0).upper())

        for m in _RX_INSURANCE.finditer(shadow):
            _add_span(spans, m.start(), m.end(), "INSURANCE_ID", text, m.group(0).upper())

        for rx in (_RX_PHONE_0, _RX_PHONE_1, _RX_PHONE_2):
            for m in rx.finditer(shadow):
                digits = _RX_DIGITS_ONLY.sub("", m.group(0))
                if len(digits) >= 10:
                    _add_span(spans, m.start(), m.end(), "PHONE", text, digits[-10:])

        # Fallback offline libphonenumber matcher for unusual formats (only if >=10 digits exist outside already-matched spans)
        if _RX_HAS_10_DIGITS.search(shadow):
            # Quick check: count digits not already covered by PHONE/AADHAAR/ABHA spans
            covered_iv = _IntervalSet([
                (sp.start, sp.end)
                for sp in spans
                if sp.pii_type in ("PHONE", "AADHAAR", "AADHAAR_MASKED", "ABHA_NUMBER")
            ])
            uncovered_digits = 0
            for idx, ch in enumerate(shadow):
                if ch.isdigit() and not covered_iv.overlaps(idx, idx + 1):
                    uncovered_digits += 1
                    if uncovered_digits >= 10:
                        break
            if uncovered_digits >= 10:
                try:
                    span_iv = _IntervalSet([(s.start, s.end) for s in spans])
                    for match in phonenumbers.PhoneNumberMatcher(shadow, "IN"):
                        digits = _RX_DIGITS_ONLY.sub("", match.raw_string)
                        if len(digits) >= 10 and not span_iv.overlaps(match.start, match.end):
                            prev_len = len(spans)
                            _add_span(spans, match.start, match.end, "PHONE", text, digits[-10:])
                            if len(spans) > prev_len:
                                span_iv.add(spans[-1].start, spans[-1].end)
                except Exception:
                    pass

        for rx in (_RX_MRN_KNOWN, _RX_MRN_GENERIC, _RX_MRN_LABELED):
            for m in rx.finditer(shadow):
                tok = m.group(0).strip()
                if not _RX_MRN_EXCLUDE.fullmatch(tok):
                    _add_span(spans, m.start(), m.end(), "MRN", text, tok.upper())

        # 2. Address components (House, Locality, Landmark, Village, Tehsil, PIN)
        for m in _RX_HOUSE.finditer(shadow):
            _add_span(spans, m.start(), m.end(), "HOUSE", text)

        for m in _RX_LOCALITY.finditer(shadow):
            _add_span(spans, m.start(), m.end(), "LOCALITY", text)

        for m in _RX_ADDR_WARD.finditer(shadow):
            # Only treat Ward N as LOCALITY when NOT inside hospital-ward protected spans
            if not prot_iv.overlaps(m.start(), m.end()):
                # Check if line looks like clinical ward header vs address/locality
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_end = text.find("\n", m.end())
                line = text[line_start:(len(text) if line_end == -1 else line_end)].lower()
                if not any(w in line for w in ("ward/bed", "nursing notes", "bed ", "admitted to", "admission to", "census", "medicine)", "surgery)")):
                    _add_span(spans, m.start(), m.end(), "LOCALITY", text)

        for m in _RX_LANDMARK.finditer(shadow):
            _add_span(spans, m.start(), m.end(), "LANDMARK", text)

        for m in _RX_VILLAGE_TEXT.finditer(text):
            _add_span(spans, m.start(), m.end(), "VILLAGE", text)

        span_iv_vill = None
        for m in _RX_VILLAGE_BEFORE_TEHSIL.finditer(text):
            if not prot_iv.overlaps(m.start(), m.end()):
                if span_iv_vill is None:
                    span_iv_vill = _IntervalSet([(s.start, s.end) for s in spans])
                if not span_iv_vill.overlaps(m.start(), m.end()):
                    prev_len = len(spans)
                    _add_span(spans, m.start(), m.end(), "VILLAGE", text)
                    if len(spans) > prev_len:
                        span_iv_vill.add(spans[-1].start, spans[-1].end)

        for m in _RX_TEHSIL_TEXT.finditer(text):
            _add_span(spans, m.start(), m.end(), "TEHSIL", text)

        if allow_pin_scan:
            for m in _RX_PIN_ADDR.finditer(shadow):
                if not prot_iv.overlaps(m.start(), m.end()):
                    # Verify PIN context (near Distt/Tehsil/PIN/Pin Code/-) or not a lab count/amount
                    ctx = shadow[max(0, m.start() - 45):min(len(shadow), m.end() + 25)].lower()
                    if any(k in ctx for k in ("pin", "zip", "distt", "district", "जिला", "तहसील", "tehsil", "uttar pradesh", "- " + m.group(1), ", " + m.group(1))):
                        if not any(u in ctx for u in ("/ul", "inr", "amount", "wbc", "platelet", "rs.")):
                            _add_span(spans, m.start(1), m.end(1), "PIN", text, m.group(1))

        # 3. Context-anchored person & relative names
        for m in _RX_FRAG_REL.finditer(text):
            if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")
            if not prot_iv.overlaps(m.start(2), m.end(2)) and not _is_non_pii_phrase(m.group(2)):
                _add_span(spans, m.start(2), m.end(2), "RELATIVE_NAME", text, role="relative")

        for m in _RX_PATIENT_LABEL.finditer(text):
            if not prot_iv.overlaps(m.start(), m.end()) and not _is_non_pii_phrase(m.group(0)):
                _add_span(spans, m.start(), m.end(), "PATIENT_NAME", text, role="patient")

        for m in _RX_INLINE_PATIENT_ANCHOR.finditer(text):
            if not prot_iv.overlaps(m.start(), m.end()) and not _is_non_pii_phrase(m.group(0)):
                _add_span(spans, m.start(), m.end(), "PATIENT_NAME", text, role="patient")

        for m in _RX_REL_LABEL.finditer(text):
            if not prot_iv.overlaps(m.start(), m.end()) and not _is_non_pii_phrase(m.group(0)):
                _add_span(spans, m.start(), m.end(), "RELATIVE_NAME", text, role="relative")

        for m in _RX_CO_CARE_OF.finditer(text):
            if not prot_iv.overlaps(m.start(), m.end()):
                cand = m.group(1).strip()
                if not _SYMPTOM_WORDS_RE.match(cand) and not _is_non_pii_phrase(cand):
                    _add_span(spans, m.start(1), m.end(1), "RELATIVE_NAME", text, role="relative")

        for m in _RX_WHATSAPP_SENDER.finditer(text):
            cand = m.group(1).strip()
            if cand and not _is_non_pii_phrase(cand):
                _add_span(spans, m.start(1), m.end(1), "RELATIVE_NAME", text, role="relative")

        for m in _RX_HONORIFIC_NAME.finditer(text):
            if not prot_iv.overlaps(m.start(), m.end()) and not _is_non_pii_phrase(m.group(1)):
                # Include the honorific or keep only name? Redact full "<Name> ji" to prevent first-name leakage,
                # or redact group(1) and keep/redact honorific — redacting full span removes "Asha ji" cleanly!
                role = "relative" if m.group(2).lower() in ("bhaiya", "bhaiyya", "chachu", "chacha", "mama", "papa") else "patient"
                _add_span(spans, m.start(), m.end(), "RELATIVE_NAME" if role == "relative" else "PATIENT_NAME", text, role=role)

        for m in _RX_PREFIX_HONORIFIC_NAME.finditer(text):
            if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        for m in _RX_DEVA_POSTPOS.finditer(text):
            if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        for m in _RX_HINGLISH_POSTPOS.finditer(text):
            if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        for m in _RX_ATTENDANT_CONTEXT.finditer(text):
            if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        for m in _RX_PAREN_ATTENDANT.finditer(text):
            if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        # 4. General Classical ML + Gazetteer sweep for un-anchored 2-3 word names & canaries
        occ_iv = _IntervalSet(prot + [(s.start, s.end) for s in spans])
        for cand_rx in _CAND_RX_LIST:
            for m in cand_rx.finditer(text):
                if occ_iv.overlaps(m.start(), m.end()):
                    continue
                cand = m.group(0)
                lctx = text[max(0, m.start() - 25):m.start()]
                rctx = text[m.end():min(len(text), m.end() + 25)]
                if self.looks_like_person_name(cand, lctx, rctx):
                    prev_len = len(spans)
                    _add_span(spans, m.start(), m.end(), "PATIENT_NAME", text, role="patient")
                    if len(spans) > prev_len:
                        occ_iv.add(spans[-1].start, spans[-1].end)

        # 5. Propagate known patient/relative name variants (including phonetic equivalents like Hareesh/Harish)
        if extra_patient_names or extra_relative_names:
            text_lower = text.lower()
            span_iv = _IntervalSet([(s.start, s.end) for s in spans])
            for name_list, ptype, role in (
                (extra_patient_names or (), "PATIENT_NAME", "patient"),
                (extra_relative_names or (), "RELATIVE_NAME", "relative"),
            ):
                if not name_list:
                    continue
                extra_firsts: list[str] = []
                seen_toks: set[str] = set()
                for nm in dict.fromkeys(name_list):
                    for tok in self._expand_name_variants(nm):
                        tl = tok.lower()
                        if len(tok) < 3 or tl in _NON_PII_WORDS or tl in _GENERIC_NAME_PARTS or tl in seen_toks:
                            continue
                        seen_toks.add(tl)
                        extra_firsts.append(tok)
                        if tl not in text_lower:
                            continue
                        pat = _compile_token_rx(tok)
                        for m in pat.finditer(text):
                            if not prot_iv.overlaps(m.start(), m.end()) and not span_iv.overlaps(m.start(), m.end()):
                                prev_len = len(spans)
                                _add_span(spans, m.start(), m.end(), ptype, text, role=role)
                                if len(spans) > prev_len:
                                    span_iv.add(spans[-1].start, spans[-1].end)
                # Phonetic match on capitalized single prose tokens >= 4 chars (catches Harish vs Hareesh, Aneeta vs Anita)
                if extra_firsts:
                    for m in _RX_CAP_WORD_4.finditer(text):
                        if prot_iv.overlaps(m.start(), m.end()) or span_iv.overlaps(m.start(), m.end()):
                            continue
                        cand = m.group(0)
                        if cand.lower() in _NON_PII_WORDS:
                            continue
                        if any(names_similarity(cand, ref) >= 0.88 for ref in extra_firsts):
                            prev_len = len(spans)
                            _add_span(spans, m.start(), m.end(), ptype, text, role=role)
                            if len(spans) > prev_len:
                                span_iv.add(spans[-1].start, spans[-1].end)

        # Resolve overlapping spans: keep longer span, earlier start
        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        non_overlapping: list[PIISpan] = []
        last_end = -1
        for s in spans:
            if s.start >= last_end:
                non_overlapping.append(s)
                last_end = s.end
        return non_overlapping

    @staticmethod
    def _expand_name_variants(name: str) -> list[str]:
        parts = [p.strip(".,") for p in name.split() if len(p.strip(".,")) >= 3 and not p.strip(".,").isdigit()]
        variants = []
        if name.strip() and not name.strip().isdigit():
            variants.append(name.strip())
        for p in parts:
            if p.lower() not in _HONORIFIC_TOKENS and p.lower() not in _GENERIC_NAME_PARTS:
                variants.append(p)
        return variants

    def redact_text(
        self,
        text: str,
        extra_patient_names: list[str] | None = None,
        extra_relative_names: list[str] | None = None,
        allow_pin_scan: bool = True,
        pseudonym: str | None = None,
        entity_counts: dict[str, int] | None = None,
    ) -> str:
        """Replace detected PII spans with typed placeholders (injecting pseudonym on first patient name/MRN if requested)."""
        spans = self.detect_text(
            text,
            extra_patient_names=extra_patient_names,
            extra_relative_names=extra_relative_names,
            allow_pin_scan=allow_pin_scan,
        )
        if not spans:
            return text
        out_parts = []
        cur = 0
        injected_psn = False
        for s in spans:
            out_parts.append(text[cur:s.start])
            if entity_counts is not None:
                entity_counts[s.pii_type] = entity_counts.get(s.pii_type, 0) + 1
            if pseudonym and not injected_psn and s.pii_type in ("PATIENT_NAME", "MRN"):
                out_parts.append(pseudonym)
                injected_psn = True
            elif s.pii_type == "PIN":
                out_parts.append(f"{s.surface[:3]}***")
            else:
                out_parts.append(f"[{s.pii_type}]")
            cur = s.end
        out_parts.append(text[cur:])
        return "".join(out_parts)

    def has_pii_in_identifier(self, text: str) -> bool:
        """Check whether a filename, header, or JSON key contains PII."""
        if not text:
            return False
        cached = self._id_pii_cache.get(text)
        if cached is not None:
            return cached
        res = self._has_pii_in_identifier_uncached(text)
        if len(self._id_pii_cache) < 65536:
            self._id_pii_cache[text] = res
        return res

    def _has_pii_in_identifier_uncached(self, text: str) -> bool:
        # Technical snake_case / lowercase keys or filenames without digits/gazetteer names are schema/file labels
        is_all_lower_ascii = bool(std_re.fullmatch(r"[a-z0-9_\-./\s]+", text))
        if self.classify_column_header(text) != "OTHER" and not std_re.search(r"\d{4,}", text):
            return False

        shadow = shadow_normalize(text)
        for rx in (_RX_ABHA_NUM, _RX_ABHA_ADDR, _RX_AADHAAR, _RX_EMAIL, _RX_PAN, _RX_PHONE_0, _RX_PHONE_1, _RX_PHONE_2, _RX_MRN_KNOWN, _RX_MRN_GENERIC):
            if rx.search(shadow):
                return True

        cleaned = std_re.sub(r"[_\-/]+", " ", text).strip()
        if _is_non_pii_phrase(cleaned):
            return False
        for tok in cleaned.split():
            tl = tok.lower().strip(".,:")
            if tl in _NON_PII_WORDS or tl.isdigit():
                continue
            if tl in _LATIN_NAME_GAZETTEER or tok in _DEVA_NAME_GAZETTEER:
                return True
        if is_all_lower_ascii:
            return False
        toks = cleaned.split()
        if len(toks) >= 2 and self.ml.score(cleaned) >= 0.45:
            return True
        return False

    # ----------------------------------------------------------------------- Structured Column / Key Classification

    @staticmethod
    def classify_column_header(header: str, sample_values: list[str] | None = None) -> str:
        """Semantic role of a CSV/TSV column (generalizes across English/Hindi/Hinglish/unseen headers)."""
        h = transliterate_deva_to_latin(header).lower().strip()
        h_clean = std_re.sub(r"[^a-z0-9\s]", " ", h).strip()
        tokens = set(h_clean.split())

        if any(k in h_clean for k in ("patient name", "mariz ka naam", "marija ka nama", "rogi ka naam", "full name", "first name", "firstname", "last name", "lastname")) or "naam" in tokens:
            if not ({"relative", "father", "husband", "spouse", "mother", "guardian", "pita", "pati", "mata"} & tokens):
                return "PATIENT_NAME"
        if h_clean in ("name", "patient", "subject"):
            return "PATIENT_NAME"
        if ({"relative", "guardian", "father", "husband", "spouse", "mother", "pita", "pati", "mata"} & tokens) or "care of" in h_clean:
            return "RELATIVE_NAME"
        if h_clean in ("relation", "rel"):
            return "RELATION"
        if any(k in h_clean for k in ("aadhaar", "aadhar", "uidai")):
            return "AADHAAR"
        if "abha" in h_clean:
            return "ABHA"
        if any(k in h_clean for k in ("pan", "payer pan")):
            return "PAN"
        if any(k in h_clean for k in ("pmjay", "policy", "insurance", "claim id")):
            return "INSURANCE_ID"
        if any(k in h_clean for k in ("uhid", "mrn", "patientid", "patient id", "patient ref", "reg no", "reg id", "ipd", "reg_no")):
            return "MRN"
        if any(k in h_clean for k in ("mobile", "phone", "contact", "callback", "tel")) or "mob" in tokens:
            return "PHONE"
        if "email" in h_clean or "mail" in h_clean:
            return "EMAIL"
        if any(k in h_clean for k in ("dob", "birth", "janm", "janma")):
            return "DOB"
        if any(k in h_clean for k in ("age sex", "age/sex")):
            return "AGE_SEX"
        if "age" in tokens or "umar" in tokens or "umra" in tokens or "aayu" in tokens:
            return "AGE"
        if any(k in tokens for k in ("gender", "sex", "ling", "linga")):
            return "GENDER"
        if any(k in h_clean for k in ("village", "gaon", "gaav", "gram", "city")):
            return "VILLAGE"
        if any(k in h_clean for k in ("mohalla", "locality", "colony", "basti")):
            return "LOCALITY"
        if "tehsil" in h_clean or "taluka" in h_clean or "block" in tokens:
            return "TEHSIL"
        if any(k in h_clean for k in ("district", "jila", "zila", "distt")) or "dist" in tokens:
            return "DISTRICT"
        if any(k in tokens for k in ("pin", "pincode", "zip", "zipcode", "postal")) or "pin code" in h_clean:
            return "PIN"
        if any(k in h_clean for k in ("address", "pata", "street", "house", "makan", "landmark")):
            return "ADDRESS"
        if any(k in h_clean for k in ("reg date", "registration date", "registered", "admission", "admit", "discharge", "disch", "collected", "reported", "visit date")) or "dt" in tokens:
            return "DATE"
        if any(k in h_clean for k in ("doctor", "ref doctor", "consultant")):
            return "DOCTOR"
        if any(k in h_clean for k in ("notes", "remark", "comment")):
            return "FREE_TEXT"

        # Content-based column profiling for unseen/obfuscated column names on chaos round!
        if sample_values:
            non_empty = [shadow_normalize(str(v).strip()) for v in sample_values if str(v).strip()]
            if non_empty:
                n = len(non_empty)
                if sum(bool(_RX_PHONE_0.fullmatch(v) or _RX_PHONE_1.fullmatch(v) or _RX_PHONE_2.fullmatch(v)) for v in non_empty) >= 0.6 * n:
                    return "PHONE"
                if sum(bool(_RX_AADHAAR.fullmatch(v) or _RX_AADHAAR_MASKED.fullmatch(v)) for v in non_empty) >= 0.6 * n:
                    return "AADHAAR"
                if sum(bool(_RX_ABHA_NUM.fullmatch(v)) for v in non_empty) >= 0.6 * n:
                    return "ABHA"
                if sum(bool(_RX_PAN.fullmatch(v)) for v in non_empty) >= 0.6 * n:
                    return "PAN"
                if sum(bool(_RX_EMAIL.fullmatch(v)) for v in non_empty) >= 0.6 * n:
                    return "EMAIL"
                if sum(bool(_RX_MRN_KNOWN.fullmatch(v) or _RX_MRN_GENERIC.fullmatch(v)) for v in non_empty) >= 0.6 * n:
                    return "MRN"
                if sum(bool(std_re.fullmatch(r"[1-8]\d{5}", v)) for v in non_empty) >= 0.6 * n:
                    # Exclude lab/billing numeric columns
                    if not any(k in h_clean for k in ("amount", "inr", "price", "cost", "count", "wbc", "platelet", "result", "value")):
                        return "PIN"
                if sum(bool(std_re.fullmatch(r"\d{1,3}/[MFmf]", v)) for v in non_empty) >= 0.6 * n:
                    return "AGE_SEX"
                if sum(bool(std_re.fullmatch(r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})", v)) for v in non_empty) >= 0.6 * n:
                    return "DATE"
        return "OTHER"


# --------------------------------------------------------------------------- Base64 Attachment Helper

_B64_CANDIDATE = std_re.compile(r"^[A-Za-z0-9+/=\s]{24,}$")


def try_decode_base64_text(value: str) -> str | None:
    stripped = std_re.sub(r"\s+", "", value)
    if len(stripped) < 24 or len(stripped) % 4 not in (0, 2, 3):
        return None
    if not _B64_CANDIDATE.fullmatch(value):
        return None
    try:
        raw = base64.b64decode(stripped + "=" * (-len(stripped) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    # Verify decoded text is mostly printable human text
    printable = sum(1 for ch in decoded if ch.isprintable() or ch in "\t\n\r")
    if printable / max(len(decoded), 1) < 0.85:
        return None
    return decoded


def encode_base64_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")
