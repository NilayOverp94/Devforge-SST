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

try:
    from rapidfuzz import fuzz
except ImportError:  # Pure-Python fallback for mobile/Chaquopy runtimes without C++ wheels
    from difflib import SequenceMatcher as _SequenceMatcher

    class _PureFuzz:
        @staticmethod
        def ratio(a: str, b: str) -> float:
            if not a and not b:
                return 100.0
            return _SequenceMatcher(None, a, b).ratio() * 100.0

        @staticmethod
        def token_sort_ratio(a: str, b: str) -> float:
            ta = " ".join(sorted(a.split()))
            tb = " ".join(sorted(b.split()))
            if not ta and not tb:
                return 100.0
            return _SequenceMatcher(None, ta, tb).ratio() * 100.0

    fuzz = _PureFuzz()


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
_RX_DIGITS_ONLY = std_re.compile(r"\D", std_re.ASCII)


def _digits_only(s: str) -> str:
    return s if s.isdigit() else _RX_DIGITS_ONLY.sub("", s)
_RX_ANY_DIGIT = std_re.compile(r"\d", std_re.ASCII)
_RX_HAS_4_DIGITS = std_re.compile(r"\d{4}", std_re.ASCII)
_RX_HAS_10_DIGITS = std_re.compile(r"(?:\D*\d){10}", std_re.ASCII)
_RX_PHONE_CAND_WINDOW = std_re.compile(r"\+?\(?\d[\d\s.\-()/]{8,20}\d", std_re.ASCII)
_RX_ADDR_FOLLOW = std_re.compile(r"^\s*,\s*(?:h\.?no|house|makan|ward|mohalla|vill|gram|tehsil|distt)")
_RX_MRN_EXCLUDE = std_re.compile(r"(?i)(?:ICD|LOINC|MG|SG|SE|OPD|BL)[-/0-9A-Z]*")
_CAND_RX_LIST = (
    std_re.compile(r"\b[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,2}\b", std_re.ASCII),
    std_re.compile(r"\b[A-Z]{2,20}\s+[A-Z]{2,20}\b", std_re.ASCII),
    regex.compile(r"[\p{Devanagari}]{2,20}\s+[\p{Devanagari}]{2,20}"),
)
_RX_CAP_WORD_4 = std_re.compile(r"\b[A-Z][a-z]{3,20}\b", std_re.ASCII)
_RX_WORD_ALL = regex.compile(r"[\p{L}\p{M}]{3,30}")
_RX_WORD_ASCII = std_re.compile(r"[A-Za-z]{3,30}", std_re.ASCII)
_RX_ANY_UPPER = std_re.compile(r"[A-Z]", std_re.ASCII)
_RX_ANY_ALPHA_LOWER = std_re.compile(r"[a-z]", std_re.ASCII)
_RX_HONORIFIC_HINT = regex.compile(
    r"(?i)\b(?:ji|sahab|saheb|saab|didi|behenji|bahenji|behen|bhaiya|bhaiyya|bhai|babu|amma|chachu|chacha|mama)\b|जी|साहब|दीदी|बहनजी|भैया|बाबू"
)
_RX_BAD_PRINTABLE = std_re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


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
        self._span_score_cache: dict[str, float] = {}
        self._ctx_L: dict[str, float] = {}
        self._ctx_Lfull: dict[str, float] = {}
        self._ctx_R: dict[str, float] = {}
        self._ctx_Rfull: dict[str, float] = {}
        if model_path.is_file():
            try:
                payload = json.loads(model_path.read_text(encoding="utf-8"))
                self.weights = {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}
            except Exception:
                self.weights = {}
        for k, v in self.weights.items():
            if k.startswith("L:"):
                self._ctx_L[k[2:]] = v
            elif k.startswith("Lfull:"):
                self._ctx_Lfull[k[6:]] = v
            elif k.startswith("R:"):
                self._ctx_R[k[2:]] = v
            elif k.startswith("Rfull:"):
                self._ctx_Rfull[k[6:]] = v

    @staticmethod
    @lru_cache(maxsize=65536)
    def _norm_tok(tok: str) -> str:
        if not tok.isascii() and any("\u0900" <= c <= "\u097f" for c in tok):
            tok = transliterate_deva_to_latin(tok)
        return std_re.sub(r"[^a-z0-9]", "", tok.lower())

    def extract_features(self, span_text: str, left_ctx: str = "", right_ctx: str = "") -> dict[str, float]:
        feats: dict[str, float] = {"bias": 1.0}
        raw_tokens = span_text.strip().split()
        if not raw_tokens:
            return feats
        feats[f"ntok={min(len(raw_tokens), 4)}"] = 1.0
        has_deva = (not span_text.isascii()) and any("\u0900" <= c <= "\u097f" for c in span_text)
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

    def _span_only_score(self, span_text: str) -> float:
        cached = self._span_score_cache.get(span_text)
        if cached is not None:
            return cached
        w_get = self.weights.get
        total = w_get("bias", 0.0)
        raw_tokens = span_text.strip().split()
        if raw_tokens:
            seen: set[str] = set()
            keys: list[str] = [
                f"ntok={min(len(raw_tokens), 4)}",
                "script=deva" if ((not span_text.isascii()) and any("\u0900" <= c <= "\u097f" for c in span_text)) else "script=latin",
            ]
            if all(t[:1].isupper() and t[1:].islower() for t in raw_tokens if t.isalpha()):
                keys.append("all_title")
            if all(t.isupper() for t in raw_tokens if t.isalpha()):
                keys.append("all_upper")
            if any(c.isdigit() for c in span_text):
                keys.append("has_digit")
            if "'" in span_text or "’" in span_text:
                keys.append("has_apostrophe")

            norm_tokens = [nt for t in raw_tokens if (nt := self._norm_tok(t))]
            n_norm = len(norm_tokens)
            for idx, nt in enumerate(norm_tokens):
                pos_tag = "first" if idx == 0 else ("last" if idx == n_norm - 1 else "mid")
                l_nt = len(nt)
                if l_nt >= 2:
                    keys.append(f"w:{nt}")
                    keys.append(f"p2:{pos_tag}:{nt[:2]}")
                    keys.append(f"s2:{pos_tag}:{nt[-2:]}")
                if l_nt >= 3:
                    keys.append(f"p3:{pos_tag}:{nt[:3]}")
                    keys.append(f"s3:{pos_tag}:{nt[-3:]}")
                if l_nt >= 4:
                    keys.append(f"s4:{nt[-4:]}")
                for g in (2, 3, 4):
                    if l_nt >= g:
                        for i in range(l_nt - g + 1):
                            keys.append(f"ng{g}:{nt[i:i+g]}")
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    val = w_get(k)
                    if val is not None:
                        total += val
        if len(self._span_score_cache) < 65536:
            self._span_score_cache[span_text] = total
        return total

    def _ctx_score(self, left_ctx: str = "", right_ctx: str = "") -> float:
        total = 0.0
        if left_ctx:
            lc = left_ctx.strip().lower()
            if lc:
                parts = lc.split()[-2:]
                w_L = self._ctx_L
                if len(parts) == 2:
                    p0, p1 = parts[0], parts[1]
                    total += w_L.get(p0, 0.0)
                    if p1 != p0:
                        total += w_L.get(p1, 0.0)
                elif parts:
                    total += w_L.get(parts[0], 0.0)
                total += self._ctx_Lfull.get(lc, 0.0)
        if right_ctx:
            rc = right_ctx.strip().lower()
            if rc:
                parts = rc.split()[:2]
                w_R = self._ctx_R
                if len(parts) == 2:
                    p0, p1 = parts[0], parts[1]
                    total += w_R.get(p0, 0.0)
                    if p1 != p0:
                        total += w_R.get(p1, 0.0)
                elif parts:
                    total += w_R.get(parts[0], 0.0)
                total += self._ctx_Rfull.get(rc, 0.0)
        return total

    def score(self, span_text: str, left_ctx: str = "", right_ctx: str = "") -> float:
        if not self.weights:
            return 0.0
        return self._span_only_score(span_text) + self._ctx_score(left_ctx, right_ctx)


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
_RX_ABHA_NUM = std_re.compile(r"(?<!\d)(?:91|\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}(?!\d)", std_re.ASCII)
_RX_ABHA_ADDR = std_re.compile(r"(?i)\b[a-z0-9._-]{3,}@(?:abdm|sbx|ndhm)\b", std_re.ASCII)
_RX_AADHAAR = std_re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)", std_re.ASCII)
_RX_AADHAAR_MASKED = std_re.compile(r"(?i)(?<![a-z0-9])[x*]{4}[\s-]?[x*]{4}[\s-]?\d{4}(?!\d)", std_re.ASCII)
_RX_PHONE_ALL = std_re.compile(
    r"(?<!\d)(?:\+?91[\s.-]?|0)?[6-9](?:\d{4}[\s.-]?\d{5}|\d{2}[\s.-]?\d{3}[\s.-]?\d{4})(?!\d)",
    std_re.ASCII,
)
_RX_EMAIL = std_re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", std_re.ASCII)
_RX_PAN = std_re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", std_re.ASCII)
_RX_INSURANCE = std_re.compile(
    r"(?i)\b(?:PMJAY[-\s/]?[A-Z]{2}[-\s/]?\d{6,12}|POL[/\-]\d{2,4}[/\-]\d{5,12})\b", std_re.ASCII
)

# MRN / UHID / IPD / ADM (known + unseen hospital prefixes; excludes OPD/, BL/, AMB-, HD-, ICD, LOINC, package codes)
_RX_MRN_KNOWN = std_re.compile(
    r"(?i)\b(?:SNJ|JJH|UHID|DHH|IPD|IP|ADM|MRN|REG|PAT)(?:[-/]?[A-Z]{2,5})?[-/]?(?:20\d{2}[-/]|\d{2}[-/])?\d{4,8}\b", std_re.ASCII
)
_RX_MRN_GENERIC = std_re.compile(
    r"\b(?!OPD/|BL/|AMB-|HD-|PMJAY|POL/|LOINC|ICD|COVID)[A-Z]{2,6}(?:[-/][A-Z]{2,5})?[-/](?:20\d{2}[-/]|\d{2}[-/])?\d{4,8}\b", std_re.ASCII
)
_RX_MRN_LABELED = regex.compile(
    r"(?i)\b(?:UHID|MRN|IPD\s*No\.?|Reg\.?\s*No\.?|Patient\s*ID|Patient\s*Ref|पंजीकरण\s*सं\.?|आईपीडी\s*सं\.?)\s*[:\-]\s*([A-Z0-9][A-Z0-9/\-]{3,20})"
)
_RX_MRN_LABELED_ASCII = std_re.compile(
    r"(?i)\b(?:UHID|MRN|IPD\s*No\.?|Reg\.?\s*No\.?|Patient\s*ID|Patient\s*Ref)\s*[:\-]\s*([A-Z0-9][A-Z0-9/\-]{3,20})",
    std_re.ASCII,
)

# Preserve / meaning-trap guards
_RX_DOCTOR_PREFIX = std_re.compile(
    r"(?i)(?:Consultant|Referred\s+by|Ref\.?\s*By|Reviewed\s+by|Author)\s*:\s*$"
)
_RX_DOCTOR_LAT = std_re.compile(
    r"\b(?:Dr\.?|Prof\.?)\s+[A-Z][\w'.-]*(?:\s+[A-Z][\w'.-]*){0,3}"
    r"(?:\s*\((?:MD|MS|MBBS|DNB|DM|MCh|BDS|MDS)[^)]*\))?",
)
_RX_DOCTOR_DEVA = regex.compile(
    r"(?:डॉ\.?|डॉक्टर)\s*[\p{Devanagari}]+(?:\s+[\p{Devanagari}]+){0,3}"
)
_RX_EPONYM = std_re.compile(
    r"(?i)\b[a-z\u00c0-\u024f]+(?:'s|’s|s')?\s+"
    r"(?:disease|syndrome|palsy|lymphoma|dystrophy|triad|sign|phenomenon|fracture|aneurysm|reflex|coma|chorea|tetrad|crisis)\b"
)
_RX_HISTORY = std_re.compile(
    r"(?i)\b(?:K/C/O|H/O|P/H/O|F/H/O|K\.C\.O\.|H\.O\.)\s*[:\-]?\s*[^\n.;]+"
)
_RX_HOSPITAL_WARD = std_re.compile(
    r"(?i)(?:(?:Ward/Bed|Ward|NURSING\s+NOTES\s*-|admission\s+to)\s*:?\s*)?"
    r"\bWard\s+\d+\s*\([^)]+\)(?:\s*/\s*Bed\s+\d+)?"
    r"|\bWard/Bed\s*:\s*Ward\s+\d+(?:\s*/\s*Bed\s+\d+)?"
    r"|\badmission\s+to\s+Ward\s+\d+\b"
)
_RX_ASHA = std_re.compile(r"\bASHA\s+(?:worker|workers|didi|bai)\b|\bASHA\b(?=\s+(?:worker|workers))", std_re.ASCII)
_RX_CLINICAL_NUMS = std_re.compile(
    r"(?<![\dA-Za-z])(?:"
    r"\d{1,3}-\d{1,3}-\d{1,3}(?:-\d{1,3})?"  # 1-0-1
    r"|(?i:\bBP\s*:?\s*)\d{2,3}/\d{2,3}"
    r"|\d+(?:\.\d+)?\s*(?i:mg/dL|g/dL|mmol/L|mIU/L|/uL|/min|mg|mcg|ug|gm|ml|iu|units?|°[FC])\b"
    r"|[A-Z]\d{2}(?:\.\d{1,2})?"  # Strictly uppercase ICD-10
    r")(?![\dA-Za-z])"
)
_RX_LAB_RANGE = std_re.compile(
    r"(?<![\d\-/])\d{1,3}(?:\.\d{1,2})?\s*[-–]\s*\d{1,3}(?:\.\d{1,2})?(?![\d\-/])"
    r"|(?<![\d\-/])\d{4,6}\s*[-–]\s*\d{4,6}(?![\d\-/])"
)
_RX_HOSPITALS = std_re.compile(
    r"\b(?:(?i:Sanjeevani\s+Multispeciality\s+Hospital|Jeevan\s+Jyoti\s+Charitable\s+Hospital|"
    r"Arogya\s+Seva\s+Sadan|District\s+Hospital\s+[A-Z][a-z]+)|"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+(?i:Hospital|Diagnostics|Pathology|Labs|Clinic|Medical\s+Centre))\b",
    std_re.ASCII,
)

# Name & Relative Label Anchors (Latin + Devanagari + Hinglish) - capture groups avoid slow variable lookbehind
_RX_PATIENT_LABEL = regex.compile(
    r"(?im)^(?:Patient'?s?\s*Name|Full\s*Name|Name|Pt\.?\s*Name|Subject|मरीज़\s*का\s*नाम|रोगी\s*का\s*नाम|नाम)\s*:\s*"
    r"([^\n\t|;,]+?)(?=\s{2,}|\s+(?:Age|Sex|Gender|UHID|MRN|Reg|Mob|Phone|DOB|आयु|लिंग|पता)\b|$|\n)"
)
_RX_PATIENT_LABEL_ASCII = std_re.compile(
    r"(?im)^(?:Patient'?s?\s*Name|Full\s*Name|Name|Pt\.?\s*Name|Subject)\s*:\s*"
    r"([^\n\t|;,]+?)(?=\s{2,}|\s+(?:Age|Sex|Gender|UHID|MRN|Reg|Mob|Phone|DOB)\b|$|\n)",
    std_re.ASCII,
)
_RX_INLINE_PATIENT_ANCHOR = regex.compile(
    r"(?i)\b(?:(?:Report\s+of|OPD\s+note\s+for|Rx\s+of|Referral|Patient)\s*:\s*|(?:Report\s+of|OPD\s+note\s+for|Rx\s+of)\s+)"
    r"([\p{L}\p{M}]+(?:\s+[\p{L}\p{M}]+){0,2})"
    r"(?=\s+(?:ji|sahab|saheb|saab|didi|behenji|bahenji|bhaiya|babu|amma|\d+|Y\s*/|yrs|age|[\(\.,\n])|\s*$)"
)
_RX_INLINE_PATIENT_ANCHOR_ASCII = std_re.compile(
    r"(?i)\b(?:(?:Report\s+of|OPD\s+note\s+for|Rx\s+of|Referral|Patient)\s*:\s*|(?:Report\s+of|OPD\s+note\s+for|Rx\s+of)\s+)"
    r"([A-Za-z]+(?:\s+[A-Za-z]+){0,2})"
    r"(?=\s+(?:ji|sahab|saheb|saab|didi|behenji|bahenji|bhaiya|babu|amma|\d+|Y\s*/|yrs|age|[\(\.,\n])|\s*$)",
    std_re.ASCII,
)
_RX_REL_LABEL = regex.compile(
    r"(?i)\b(?:S/o|D/o|W/o|F/o|M/o|S/O|D/O|W/O|Father(?:'s)?\s+Name|Husband(?:'s)?\s+Name|Spouse|Guardian|Attendant|"
    r"पिता\s*का\s*नाम|पति\s*का\s*नाम|माता\s*का\s*नाम|अभिभावक)\s*[:\(]?\s*"
    r"((?:(?:S/o|D/o|W/o|C/O)\s+)?[\p{L}\p{M}.]+(?:\s+[\p{L}\p{M}.]+){0,2})"
    r"(?=\s*(?:[,;\)\n\.]|\b(?:Mob|Mobile|Phone|Ward|H\.?No|House|Makan|informed)\b|$))"
)
_RX_REL_LABEL_ASCII = std_re.compile(
    r"(?i)\b(?:S/o|D/o|W/o|F/o|M/o|S/O|D/O|W/O|Father(?:'s)?\s+Name|Husband(?:'s)?\s+Name|Spouse|Guardian|Attendant)\s*[:\(]?\s*"
    r"((?:(?:S/o|D/o|W/o|C/O)\s+)?[A-Za-z.]+(?:\s+[A-Za-z.]+){0,2})"
    r"(?=\s*(?:[,;\)\n\.]|\b(?:Mob|Mobile|Phone|Ward|H\.?No|House|Makan|informed)\b|$))",
    std_re.ASCII,
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
_RX_WHATSAPP_SENDER = std_re.compile(
    r"(?m)^\[\d{1,2}/\d{1,2}/\d{2,4},\s*[^\]]+\]\s*([^\n:]{2,40}?)(?=:)"
)
_RX_DEVA_POSTPOS = regex.compile(
    r"([\p{Devanagari}]{2,20}(?:\s+[\p{Devanagari}]{2,20}){0,2})\s+(?:को\s+\d+\s+दिन|का\s+BP|की|ने)\b"
)
_RX_HINGLISH_POSTPOS = std_re.compile(
    r"\b([A-Z][a-z]{2,20}(?:\s+[A-Z][a-z]{2,20}){0,2})(?:\s+(?:ka\s+BP|ki\s+|ko\s+\d+|ne\s+(?:raat|subah|kaha))|'s\s+attendant)\b",
    std_re.ASCII,
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
_RX_HOUSE_ASCII = std_re.compile(
    r"(?i)\b(?:H\.?\s*No\.?|House(?:\s+No\.?)?|Makan(?:\s+No\.?)?|Flat(?:\s+No\.?)?|Plot(?:\s+No\.?)?)\s*\.?\s*[A-Za-z0-9/\-]*\d+[A-Za-z0-9/\-]*",
    std_re.ASCII,
)
_RX_LOCALITY = std_re.compile(
    r"(?i)(?:\bMohalla\s+[^,\n;]+?(?=,|\s+(?:near|opp|behind|Tehsil|Distt|Vill|Gram)\b|$|\n)"
    r"|मोहल्ला\s+[^,\n;]+?(?=,|\s+(?:के\s+पास|के\s+पीछे|के\s+सामने|तहसील|जिला|ग्राम)\b|$|\n))"
)
_RX_ADDR_WARD = std_re.compile(r"(?i)\bWard\s+\d{1,3}\b(?!\s*\()", std_re.ASCII)
_RX_LANDMARK = std_re.compile(
    r"(?i)(?:\b(?:near|opp\.?|opposite|behind|beside)\s+[^,\n;]+?(?=,|\s+(?:Tehsil|Distt|District|Vill|Gram)\b|$|\n)"
    r"|[^\n,;]+?\s+के\s+(?:पास|पीछे|सामने)(?=,|\s+(?:ग्राम|तहसील|जिला)\b|$|\n))"
)
_RX_VILLAGE_TEXT = std_re.compile(
    r"(?i)(?:\b(?:Vill\.?|Village|Gram|Gaon)\s*[:\-]?\s*[^,\n;]+?(?=,|\s+(?:Tehsil|Distt|District|PIN)\b|$|\n)"
    r"|(?:ग्राम|गाँव|गांव)\s*[:\-]?\s*[^,\n;]+?(?=,|\s+(?:तहसील|जिला|पिन)\b|$|\n))"
)
_RX_VILLAGE_BEFORE_TEHSIL = regex.compile(
    r"(?i)(?<=,\s*)(?!(?:House|H\.?\s*No|Makan|Ward|Mohalla|near|opp\.?|behind|beside|मकान|वार्ड|मोहल्ला)\b)"
    r"[A-Za-z\p{Devanagari}][A-Za-z\p{Devanagari}\s.-]{2,35}?(?=\s*,\s*(?:Tehsil|तहसील)\b)"
)
_RX_TEHSIL_TEXT = std_re.compile(
    r"(?i)(?:\bTehsil\s+[^,\n;]+?(?=,|\s+(?:Distt|District|PIN)\b|$|\n)|तहसील\s+[^,\n;]+?(?=,|\s+(?:जिला|पिन)\b|$|\n))"
)
_RX_PIN_ADDR = std_re.compile(r"(?<!\d)([1-8]\d{5})(?!\d)", std_re.ASCII)


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
        self._base_cache: dict[tuple[str, bool], tuple[tuple[tuple[int, int], ...], tuple[PIISpan, ...]]] = {}
        self._cand_fast_cache: dict[str, tuple[int, float]] = {}
        self._word_cache: dict[str, tuple[tuple[int, int, str, str, bool], ...]] = {}

    def protected_spans(self, text: str, _low: str | None = None, _has_digit: bool | None = None, _is_ascii: bool | None = None) -> list[tuple[int, int]]:
        """Find clinical spans (doctors, eponyms, K/C/O, C/O complaints, wards, doses, labs, ASHA) to preserve."""
        spans: list[tuple[int, int]] = []
        low = text.lower() if _low is None else _low
        is_ascii = text.isascii() if _is_ascii is None else _is_ascii
        has_digit = bool(_RX_ANY_DIGIT.search(text)) if _has_digit is None else _has_digit

        if "dr" in low or "prof" in low:
            for m in _RX_DOCTOR_LAT.finditer(text):
                ms, me = m.start(), m.end()
                if ms > 0:
                    pstart = max(0, ms - 24)
                    pm = _RX_DOCTOR_PREFIX.search(text[pstart:ms])
                    if pm:
                        ms = pstart + pm.start()
                spans.append((ms, me))
        if not is_ascii and "डॉ" in text:
            for m in _RX_DOCTOR_DEVA.finditer(text):
                spans.append((m.start(), m.end()))
        if any(w in low for w in ("disease", "syndrome", "palsy", "lymphoma", "dystrophy", "triad", "sign", "phenomenon", "fracture", "aneurysm", "reflex", "coma", "chorea", "tetrad", "crisis")):
            for m in _RX_EPONYM.finditer(text):
                spans.append((m.start(), m.end()))
        if "/o" in low or ".o" in low:
            for m in _RX_HISTORY.finditer(text):
                spans.append((m.start(), m.end()))
        if "ward" in low:
            for m in _RX_HOSPITAL_WARD.finditer(text):
                spans.append((m.start(), m.end()))
        if "ASHA" in text:
            for m in _RX_ASHA.finditer(text):
                spans.append((m.start(), m.end()))
        if has_digit:
            for m in _RX_CLINICAL_NUMS.finditer(text):
                spans.append((m.start(), m.end()))
            if "-" in text or "–" in text:
                for m in _RX_LAB_RANGE.finditer(text):
                    # Only protect numeric range if it's NOT a 10-digit Indian mobile (5-5 split)
                    digits = _digits_only(m.group(0))
                    if not (len(digits) == 10 and digits[0] in "6789"):
                        spans.append((m.start(), m.end()))
        if any(h in low for h in ("hospital", "sadan", "clinic", "diagnostic", "pathology", "lab", "medical")):
            for m in _RX_HOSPITALS.finditer(text):
                spans.append((m.start(), m.end()))

        # C/O <symptom> (only protect if NOT a relative/care-of!)
        if "c/o" in low:
            for m in _RX_CO_CARE_OF.finditer(text):
                after = m.group(1).strip()
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_prefix = low[line_start:m.start()]
                is_addr_ctx = any(k in line_prefix for k in ("address", "pata", "attendant", "guardian", "पता"))
                after_comma = low[m.end():m.end() + 30]
                has_addr_follow = bool(_RX_ADDR_FOLLOW.search(after_comma))
                if not is_addr_ctx and not has_addr_follow and _SYMPTOM_WORDS_RE.match(after):
                    # Protect the whole C/O complaint phrase up to newline/semicolon
                    end_line = text.find("\n", m.start())
                    if end_line == -1:
                        end_line = len(text)
                    spans.append((m.start(), end_line))
        return spans

    def _classify_name_candidate(self, cand: str) -> tuple[int, float]:
        cached = self._cand_fast_cache.get(cand)
        if cached is not None:
            return cached
        s = cand.strip(" \t\r\n.,:;()[]{}\"'")
        res: tuple[int, float]
        if not s or len(s) < 2 or any(c.isdigit() for c in s):
            res = (-1, 0.0)
        elif _is_non_pii_phrase(s):
            res = (-1, 0.0)
        else:
            toks = s.split()
            if len(toks) > 4:
                res = (-1, 0.0)
            else:
                has_lat = False
                for t in toks:
                    lt = t.lower() if (t.isascii() and t.isalpha()) else _RX_NON_WORD.sub("", t.lower())
                    if len(lt) >= 3 and lt in _LATIN_NAME_GAZETTEER:
                        has_lat = True
                        break
                if has_lat or any(t in _DEVA_NAME_GAZETTEER for t in toks):
                    res = (1, 0.0)
                elif any("\u0900" <= c <= "\u097f" for c in s) and any(
                    c in _LATIN_NAME_GAZETTEER for c in canonicalize_person_name(s) if len(c) >= 3
                ):
                    res = (1, 0.0)
                else:
                    thresh = 0.45 if len(toks) >= 2 else 1.25
                    res = (0, self.ml._span_only_score(s) - thresh)
        if len(self._cand_fast_cache) < 65536:
            self._cand_fast_cache[cand] = res
        return res

    def looks_like_person_name(self, text: str, left_ctx: str = "", right_ctx: str = "") -> bool:
        state, margin = self._classify_name_candidate(text)
        if state == 1:
            return True
        if state == -1:
            return False
        return (margin + self.ml._ctx_score(left_ctx, right_ctx)) >= 0.0

    def _detect_base_spans(
        self,
        text: str,
        allow_pin_scan: bool = True,
    ) -> tuple[tuple[tuple[int, int], ...], tuple[PIISpan, ...]]:
        use_cache = len(text) <= 512
        if use_cache:
            cached = self._base_cache.get((text, allow_pin_scan))
            if cached is not None:
                return cached

        shadow = shadow_normalize(text)
        low = text.lower()
        is_ascii = text.isascii()
        has_digit = bool(_RX_ANY_DIGIT.search(shadow))
        prot = self.protected_spans(text, _low=low, _has_digit=has_digit, _is_ascii=is_ascii)
        prot_iv = _IntervalSet(prot)
        spans: list[PIISpan] = []

        # 1. Structured numeric & alphanumeric direct identifiers (highest priority - never blocked by lab range guards)
        has_4_digits = bool(_RX_HAS_4_DIGITS.search(shadow)) if has_digit else False
        has_10_digits = bool(_RX_HAS_10_DIGITS.search(shadow)) if has_4_digits else False
        if has_4_digits:
            if has_10_digits:
                for m in _RX_ABHA_NUM.finditer(shadow):
                    _add_span(spans, m.start(), m.end(), "ABHA_NUMBER", text, _digits_only(m.group(0)))

            if "xx" in low or "**" in text or (low.count("x") + text.count("*") >= 4):
                for m in _RX_AADHAAR_MASKED.finditer(shadow):
                    _add_span(spans, m.start(), m.end(), "AADHAAR_MASKED", text, m.group(0).lower())

            if has_10_digits:
                span_iv = _IntervalSet([(s.start, s.end) for s in spans]) if spans else None
                for m in _RX_AADHAAR.finditer(shadow):
                    if span_iv is None or not span_iv.overlaps(m.start(), m.end()):
                        prev_len = len(spans)
                        _add_span(spans, m.start(), m.end(), "AADHAAR", text, _digits_only(m.group(0)))
                        if len(spans) > prev_len and span_iv is not None:
                            span_iv.add(spans[-1].start, spans[-1].end)

            if _RX_ANY_UPPER.search(shadow):
                for m in _RX_PAN.finditer(shadow):
                    _add_span(spans, m.start(), m.end(), "PAN", text, m.group(0).upper())

            if "pmjay" in low or "pol" in low:
                for m in _RX_INSURANCE.finditer(shadow):
                    _add_span(spans, m.start(), m.end(), "INSURANCE_ID", text, m.group(0).upper())

            if has_10_digits:
                for m in _RX_PHONE_ALL.finditer(shadow):
                    digits = _digits_only(m.group(0))
                    if len(digits) >= 10:
                        _add_span(spans, m.start(), m.end(), "PHONE", text, digits[-10:])

                # Fallback offline libphonenumber matcher ONLY on compact uncovered digit windows (never whole 64KB chunk)
                covered_iv: _IntervalSet | None = None
                span_iv_ph: _IntervalSet | None = None
                for wm in _RX_PHONE_CAND_WINDOW.finditer(shadow):
                    ws, we = wm.start(), wm.end()
                    if covered_iv is None:
                        covered_iv = _IntervalSet([
                            (sp.start, sp.end)
                            for sp in spans
                            if sp.pii_type in ("PHONE", "AADHAAR", "AADHAAR_MASKED", "ABHA_NUMBER")
                        ])
                    if covered_iv.overlaps(ws, we):
                        continue
                    win = wm.group(0)
                    if sum(1 for ch in win if "0" <= ch <= "9") < 10:
                        continue
                    try:
                        if span_iv_ph is None:
                            span_iv_ph = _IntervalSet([(s.start, s.end) for s in spans])
                        for match in phonenumbers.PhoneNumberMatcher(win, "IN"):
                            digits = _digits_only(match.raw_string)
                            ms, me = ws + match.start, ws + match.end
                            if len(digits) >= 10 and not span_iv_ph.overlaps(ms, me):
                                prev_len = len(spans)
                                _add_span(spans, ms, me, "PHONE", text, digits[-10:])
                                if len(spans) > prev_len:
                                    span_iv_ph.add(spans[-1].start, spans[-1].end)
                    except Exception:
                        pass

            if _RX_ANY_ALPHA_LOWER.search(low):
                for m in _RX_MRN_KNOWN.finditer(shadow):
                    tok = m.group(0).strip()
                    if not _RX_MRN_EXCLUDE.fullmatch(tok):
                        _add_span(spans, m.start(), m.end(), "MRN", text, tok.upper())
                if ("-" in shadow or "/" in shadow) and _RX_ANY_UPPER.search(shadow):
                    for m in _RX_MRN_GENERIC.finditer(shadow):
                        tok = m.group(0).strip()
                        if not _RX_MRN_EXCLUDE.fullmatch(tok):
                            _add_span(spans, m.start(), m.end(), "MRN", text, tok.upper())
        if has_digit and (":" in text or "-" in text):
            rx_mlab = _RX_MRN_LABELED_ASCII if is_ascii else _RX_MRN_LABELED
            for m in rx_mlab.finditer(shadow):
                tok = m.group(1).strip()
                if not _RX_MRN_EXCLUDE.fullmatch(tok):
                    _add_span(spans, m.start(1), m.end(1), "MRN", text, tok.upper())

        if "@" in shadow:
            for m in _RX_ABHA_ADDR.finditer(shadow):
                _add_span(spans, m.start(), m.end(), "ABHA_ADDRESS", text, m.group(0).lower())
            for m in _RX_EMAIL.finditer(shadow):
                _add_span(spans, m.start(), m.end(), "EMAIL", text, m.group(0).lower())

        # 2. Address components (House, Locality, Landmark, Village, Tehsil, PIN)
        if has_digit:
            if any(k in low for k in ("no", "house", "makan", "flat", "plot")) or (not is_ascii and "मकान" in text):
                rx_house = _RX_HOUSE_ASCII if is_ascii else _RX_HOUSE
                for m in rx_house.finditer(shadow):
                    _add_span(spans, m.start(), m.end(), "HOUSE", text)

            if "ward" in low:
                for m in _RX_ADDR_WARD.finditer(shadow):
                    # Only treat Ward N as LOCALITY when NOT inside hospital-ward protected spans
                    if not prot_iv.overlaps(m.start(), m.end()):
                        line_start = text.rfind("\n", 0, m.start()) + 1
                        line_end = text.find("\n", m.end())
                        line = low[line_start:(len(text) if line_end == -1 else line_end)]
                        if not any(w in line for w in ("ward/bed", "nursing notes", "bed ", "admitted to", "admission to", "census", "medicine)", "surgery)")):
                            _add_span(spans, m.start(), m.end(), "LOCALITY", text)

        if "mohalla" in low or (not is_ascii and "मोहल्ला" in text):
            for m in _RX_LOCALITY.finditer(shadow):
                _add_span(spans, m.start(), m.end(), "LOCALITY", text)

        if any(k in low for k in ("near", "opp", "behind", "beside")) or (not is_ascii and "के " in text):
            for m in _RX_LANDMARK.finditer(shadow):
                _add_span(spans, m.start(), m.end(), "LANDMARK", text)

        if any(k in low for k in ("vill", "gram", "gaon")) or (not is_ascii and ("ग्राम" in text or "गाँव" in text or "गांव" in text)):
            for m in _RX_VILLAGE_TEXT.finditer(text):
                _add_span(spans, m.start(), m.end(), "VILLAGE", text)

        if "tehsil" in low or (not is_ascii and "तहसील" in text):
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

        if allow_pin_scan and has_4_digits and (
            "- " in shadow or ", " in shadow or not is_ascii
            or any(k in low for k in ("pin", "zip", "dist", "tehsil", "uttar pradesh"))
        ):
            for m in _RX_PIN_ADDR.finditer(shadow):
                if not prot_iv.overlaps(m.start(), m.end()):
                    ctx = shadow[max(0, m.start() - 45):min(len(shadow), m.end() + 25)].lower()
                    if any(k in ctx for k in ("pin", "zip", "distt", "district", "जिला", "तहसील", "tehsil", "uttar pradesh", "- " + m.group(1), ", " + m.group(1))):
                        if not any(u in ctx for u in ("/ul", "inr", "amount", "wbc", "platelet", "rs.")):
                            _add_span(spans, m.start(1), m.end(1), "PIN", text, m.group(1))

        # 3. Context-anchored person & relative names
        if "/" in text:
            for m in _RX_FRAG_REL.finditer(text):
                if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                    _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")
                if not prot_iv.overlaps(m.start(2), m.end(2)) and not _is_non_pii_phrase(m.group(2)):
                    _add_span(spans, m.start(2), m.end(2), "RELATIVE_NAME", text, role="relative")

        if ":" in text:
            rx_pl = _RX_PATIENT_LABEL_ASCII if is_ascii else _RX_PATIENT_LABEL
            for m in rx_pl.finditer(text):
                if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                    _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        if any(k in low for k in ("report of", "opd note", "rx of", "referral", "patient")):
            rx_ip = _RX_INLINE_PATIENT_ANCHOR_ASCII if is_ascii else _RX_INLINE_PATIENT_ANCHOR
            for m in rx_ip.finditer(text):
                if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                    _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        if "/" in text or ":" in text or not is_ascii or any(k in low for k in ("name", "spouse", "guardian", "attendant")):
            rx_rl = _RX_REL_LABEL_ASCII if is_ascii else _RX_REL_LABEL
            for m in rx_rl.finditer(text):
                if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                    _add_span(spans, m.start(1), m.end(1), "RELATIVE_NAME", text, role="relative")

        if "c/o" in low:
            for m in _RX_CO_CARE_OF.finditer(text):
                if not prot_iv.overlaps(m.start(), m.end()):
                    cand = m.group(1).strip()
                    if not _SYMPTOM_WORDS_RE.match(cand) and not _is_non_pii_phrase(cand):
                        _add_span(spans, m.start(1), m.end(1), "RELATIVE_NAME", text, role="relative")

        if has_digit and (text.startswith("[") or "\n[" in text):
            for m in _RX_WHATSAPP_SENDER.finditer(text):
                cand = m.group(1).strip()
                if cand and not _is_non_pii_phrase(cand):
                    _add_span(spans, m.start(1), m.end(1), "RELATIVE_NAME", text, role="relative")

        if " " in text and _RX_HONORIFIC_HINT.search(text):
            for m in _RX_HONORIFIC_NAME.finditer(text):
                if not prot_iv.overlaps(m.start(), m.end()) and not _is_non_pii_phrase(m.group(1)):
                    role = "relative" if m.group(2).lower() in ("bhaiya", "bhaiyya", "chachu", "chacha", "mama", "papa") else "patient"
                    _add_span(spans, m.start(), m.end(), "RELATIVE_NAME" if role == "relative" else "PATIENT_NAME", text, role=role)

        if any(k in low for k in ("mr", "ms", "miss", "shri", "shree", "smt", "kumari")) or (not is_ascii and ("श्री" in text or "सुश्री" in text)):
            for m in _RX_PREFIX_HONORIFIC_NAME.finditer(text):
                if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                    _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        if not is_ascii:
            for m in _RX_DEVA_POSTPOS.finditer(text):
                if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                    _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        if "bp" in low or " ki " in low or " ko " in low or " ne " in low or "'s attendant" in low:
            for m in _RX_HINGLISH_POSTPOS.finditer(text):
                if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                    _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        if "by " in low or "contact person" in low:
            for m in _RX_ATTENDANT_CONTEXT.finditer(text):
                if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                    _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        if "attendant" in low:
            for m in _RX_PAREN_ATTENDANT.finditer(text):
                if not prot_iv.overlaps(m.start(1), m.end(1)) and not _is_non_pii_phrase(m.group(1)):
                    _add_span(spans, m.start(1), m.end(1), "PATIENT_NAME", text, role="patient")

        # 4. General Classical ML + Gazetteer sweep for un-anchored 2-3 word names & canaries
        if " " in text and (not is_ascii or _RX_ANY_UPPER.search(text)):
            occ_iv = _IntervalSet(prot + [(s.start, s.end) for s in spans])
            has_upper = bool(_RX_ANY_UPPER.search(text))
            for c_idx, cand_rx in enumerate(_CAND_RX_LIST):
                if c_idx == 2:
                    if is_ascii:
                        continue
                elif not has_upper:
                    continue
                for m in cand_rx.finditer(text):
                    ms, me = m.start(), m.end()
                    if occ_iv.overlaps(ms, me):
                        continue
                    cand = m.group(0)
                    state, margin = self._classify_name_candidate(cand)
                    if state == -1:
                        continue
                    if state == 1 or (
                        margin + self.ml._ctx_score(
                            text[max(0, ms - 25):ms],
                            text[me:min(len(text), me + 25)],
                        )
                    ) >= 0.0:
                        prev_len = len(spans)
                        _add_span(spans, ms, me, "PATIENT_NAME", text, role="patient")
                        if len(spans) > prev_len:
                            occ_iv.add(spans[-1].start, spans[-1].end)

        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        non_overlapping: list[PIISpan] = []
        last_end = -1
        for s in spans:
            if s.start >= last_end:
                non_overlapping.append(s)
                last_end = s.end

        res = (tuple(prot), tuple(non_overlapping))
        if use_cache and len(self._base_cache) < 65536:
            self._base_cache[(text, allow_pin_scan)] = res
        return res

    @staticmethod
    @lru_cache(maxsize=4096)
    def _prep_extra_names(
        pat_tup: tuple[str, ...],
        rel_tup: tuple[str, ...],
    ) -> tuple[
        dict[str, tuple[str, str]],
        dict[str, list[tuple[regex.Pattern[str], str, str]]],
        tuple[tuple[regex.Pattern[str], str, str, str], ...],
        dict[str, tuple[tuple[str, str, str], ...]],
    ]:
        exact_map: dict[str, tuple[str, str]] = {}
        phrase_by_first: dict[str, list[tuple[regex.Pattern[str], str, str]]] = {}
        rare_rxs: list[tuple[regex.Pattern[str], str, str, str]] = []
        phonetic_buckets_mut: dict[str, list[tuple[str, str, str]]] = {}
        for name_list, ptype, role in (
            (pat_tup, "PATIENT_NAME", "patient"),
            (rel_tup, "RELATIVE_NAME", "relative"),
        ):
            if not name_list:
                continue
            seen_toks: set[str] = set()
            for nm in dict.fromkeys(name_list):
                for tok in PIIDetector._expand_name_variants(nm):
                    tl = tok.lower()
                    if len(tok) < 3 or tl in _NON_PII_WORDS or tl in _GENERIC_NAME_PARTS or tl in seen_toks:
                        continue
                    seen_toks.add(tl)
                    if " " in tok or any(not (ch.isalpha() or "\u0900" <= ch <= "\u097f") for ch in tok):
                        first_w = tok.split()[0].lower()
                        if first_w.isalpha():
                            phrase_by_first.setdefault(first_w, []).append((_compile_token_rx(tok), ptype, role))
                        else:
                            rare_rxs.append((_compile_token_rx(tok), tl, ptype, role))
                    else:
                        exact_map.setdefault(tl, (ptype, role))
                        if len(tok) >= 4 and tok.isascii():
                            c_toks = _canonicalize_person_name_cached(tok)
                            sk0 = _consonant_skeleton(c_toks[0])[:1] if c_toks else ""
                            b_entry = (tok, ptype, role)
                            phonetic_buckets_mut.setdefault(tl[0], []).append(b_entry)
                            if sk0 and sk0 != tl[0]:
                                phonetic_buckets_mut.setdefault(sk0, []).append(b_entry)
        phonetic_buckets = {k: tuple(v) for k, v in phonetic_buckets_mut.items()}
        return exact_map, phrase_by_first, tuple(rare_rxs), phonetic_buckets

    def _get_word_tokens(self, text: str, is_ascii: bool) -> Any:
        if len(text) <= 512:
            cached = self._word_cache.get(text)
            if cached is not None:
                return cached
            rx = _RX_WORD_ASCII if is_ascii else _RX_WORD_ALL
            toks = tuple(
                (m.start(), m.end(), w, w.lower(), w[0].isupper() and w.isascii())
                for m in rx.finditer(text)
                for w in (m.group(0),)
            )
            if len(self._word_cache) < 65536:
                self._word_cache[text] = toks
            return toks
        rx = _RX_WORD_ASCII if is_ascii else _RX_WORD_ALL
        return (
            (m.start(), m.end(), w, w.lower(), w[0].isupper() and w.isascii())
            for m in rx.finditer(text)
            for w in (m.group(0),)
        )

    def detect_text(
        self,
        text: str,
        extra_patient_names: list[str] | tuple[str, ...] | None = None,
        extra_relative_names: list[str] | tuple[str, ...] | None = None,
        allow_pin_scan: bool = True,
        local_propagate: bool = False,
    ) -> list[PIISpan]:
        """Detect all PII spans in arbitrary English/Hindi/Hinglish text while respecting clinical guards."""
        if not text:
            return []

        prot_tup, base_tup = self._detect_base_spans(text, allow_pin_scan)
        if not extra_patient_names and not extra_relative_names:
            if not local_propagate or not base_tup:
                return list(base_tup)
            extra_patient_names = tuple(dict.fromkeys(s.surface for s in base_tup if s.pii_type == "PATIENT_NAME"))
            extra_relative_names = tuple(dict.fromkeys(s.surface for s in base_tup if s.pii_type == "RELATIVE_NAME"))
            if not extra_patient_names and not extra_relative_names:
                return list(base_tup)

        pat_tup = tuple(extra_patient_names) if extra_patient_names else ()
        rel_tup = tuple(extra_relative_names) if extra_relative_names else ()
        exact_map, phrase_by_first, rare_rxs, phonetic_buckets = self._prep_extra_names(pat_tup, rel_tup)
        if not exact_map and not phrase_by_first and not rare_rxs:
            return list(base_tup)

        spans = list(base_tup)
        added_extra = False
        prot_iv: _IntervalSet | None = None
        span_iv: _IntervalSet | None = None
        low: str | None = None

        for rx, tl, ptype, role in rare_rxs:
            if low is None:
                low = text.lower()
            if tl in low:
                for m in rx.finditer(text):
                    if prot_iv is None:
                        prot_iv = _IntervalSet(list(prot_tup))
                        span_iv = _IntervalSet([(s.start, s.end) for s in spans])
                    assert span_iv is not None
                    if not prot_iv.overlaps(m.start(), m.end()) and not span_iv.overlaps(m.start(), m.end()):
                        prev_len = len(spans)
                        _add_span(spans, m.start(), m.end(), ptype, text, role=role)
                        if len(spans) > prev_len:
                            span_iv.add(spans[-1].start, spans[-1].end)
                            added_extra = True

        triggered_phrases: list[tuple[regex.Pattern[str], str, str]] = []
        seen_firsts: set[str] | None = None
        is_ascii = text.isascii()

        for ws, we, w, wl, is_cap_ascii in self._get_word_tokens(text, is_ascii):
            if phrase_by_first and wl in phrase_by_first:
                if seen_firsts is None:
                    seen_firsts = set()
                if wl not in seen_firsts:
                    seen_firsts.add(wl)
                    triggered_phrases.extend(phrase_by_first[wl])
            hit = exact_map.get(wl)
            if hit is not None:
                if prot_iv is None:
                    prot_iv = _IntervalSet(list(prot_tup))
                    span_iv = _IntervalSet([(s.start, s.end) for s in spans])
                assert span_iv is not None
                if not prot_iv.overlaps(ws, we) and not span_iv.overlaps(ws, we):
                    prev_len = len(spans)
                    _add_span(spans, ws, we, hit[0], text, role=hit[1])
                    if len(spans) > prev_len:
                        span_iv.add(spans[-1].start, spans[-1].end)
                        added_extra = True
                continue
            if is_cap_ascii and len(w) >= 4 and phonetic_buckets and wl not in _NON_PII_WORDS:
                c_w = _canonicalize_person_name_cached(w)
                sk0 = _consonant_skeleton(c_w[0])[:1] if c_w else ""
                cands = phonetic_buckets.get(wl[0], ())
                cands_sk = phonetic_buckets.get(sk0, ()) if (sk0 and sk0 != wl[0]) else ()
                if not cands and not cands_sk:
                    continue
                if prot_iv is None:
                    prot_iv = _IntervalSet(list(prot_tup))
                    span_iv = _IntervalSet([(s.start, s.end) for s in spans])
                assert span_iv is not None
                if prot_iv.overlaps(ws, we) or span_iv.overlaps(ws, we):
                    continue
                lw = len(w)
                for bucket in (cands, cands_sk):
                    matched = False
                    for ref, ptype, role in bucket:
                        lr = len(ref)
                        if abs(lw - lr) > max(2, int(0.25 * max(lw, lr))):
                            continue
                        if names_similarity(w, ref) >= 0.88:
                            prev_len = len(spans)
                            _add_span(spans, ws, we, ptype, text, role=role)
                            if len(spans) > prev_len:
                                span_iv.add(spans[-1].start, spans[-1].end)
                                added_extra = True
                            matched = True
                            break
                    if matched:
                        break

        for rx, ptype, role in triggered_phrases:
            for m in rx.finditer(text):
                if prot_iv is None:
                    prot_iv = _IntervalSet(list(prot_tup))
                    span_iv = _IntervalSet([(s.start, s.end) for s in spans])
                assert span_iv is not None
                if not prot_iv.overlaps(m.start(), m.end()) and not span_iv.overlaps(m.start(), m.end()):
                    prev_len = len(spans)
                    _add_span(spans, m.start(), m.end(), ptype, text, role=role)
                    if len(spans) > prev_len:
                        span_iv.add(spans[-1].start, spans[-1].end)
                        added_extra = True

        if not added_extra:
            return spans

        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        non_overlapping: list[PIISpan] = []
        last_end = -1
        for s in spans:
            if s.start >= last_end:
                non_overlapping.append(s)
                last_end = s.end
        return non_overlapping

    @staticmethod
    @lru_cache(maxsize=65536)
    def _expand_name_variants(name: str) -> tuple[str, ...]:
        parts = [p.strip(".,") for p in name.split() if len(p.strip(".,")) >= 3 and not p.strip(".,").isdigit()]
        variants = []
        if name.strip() and not name.strip().isdigit():
            variants.append(name.strip())
        for p in parts:
            pl = p.lower()
            if pl not in _HONORIFIC_TOKENS and pl not in _GENERIC_NAME_PARTS:
                variants.append(p)
        return tuple(variants)

    def redact_text(
        self,
        text: str,
        extra_patient_names: list[str] | tuple[str, ...] | None = None,
        extra_relative_names: list[str] | tuple[str, ...] | None = None,
        allow_pin_scan: bool = True,
        pseudonym: str | None = None,
        entity_counts: dict[str, int] | None = None,
        local_propagate: bool = False,
    ) -> str:
        """Replace detected PII spans with typed placeholders (injecting pseudonym on first patient name/MRN if requested)."""
        spans = self.detect_text(
            text,
            extra_patient_names=extra_patient_names,
            extra_relative_names=extra_relative_names,
            allow_pin_scan=allow_pin_scan,
            local_propagate=local_propagate,
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
        for rx in (_RX_ABHA_NUM, _RX_ABHA_ADDR, _RX_AADHAAR, _RX_EMAIL, _RX_PAN, _RX_PHONE_ALL, _RX_MRN_KNOWN, _RX_MRN_GENERIC):
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
                if sum(bool(_RX_PHONE_ALL.fullmatch(v)) for v in non_empty) >= 0.6 * n:
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
    if decoded.isascii():
        if _RX_BAD_PRINTABLE.search(decoded):
            return None
        return decoded
    printable = sum(1 for ch in decoded if ch.isprintable() or ch in "\t\n\r")
    if printable / max(len(decoded), 1) < 0.85:
        return None
    return decoded


def encode_base64_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")
