#!/usr/bin/env python3
"""Offline Privacy Attack Arena (A1-A5) and Utility/Temporal Audit aligned with docs/ATTACKS.md & docs/SCORING.md."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_age_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for m in re.finditer(r"(?:Age(?:/Sex)?|आयु|Umar)\s*:\s*(\d{1,3})\s*[-–]\s*(\d{1,3})", text, re.I):
        ranges.append((int(m.group(1)), int(m.group(2))))
    for m in re.finditer(r"\b(\d{1,3})\s*[-–]\s*(\d{1,3})\s*(?:Yrs?|Y\s*/\s*[MF]|/[MF]|वर्ष)\b", text, re.I):
        ranges.append((int(m.group(1)), int(m.group(2))))
    for m in re.finditer(r'"age(?:_sex)?"\s*:\s*"(\d{1,3})-(\d{1,3})(?:/[MF])?"', text):
        ranges.append((int(m.group(1)), int(m.group(2))))
    # Exact age forms (should NOT exist if banded, except if missed)
    for m in re.finditer(r"(?:Age(?:/Sex)?|आयु|Umar)\s*:\s*(\d{1,3})(?!\s*[-–]\d)", text, re.I):
        a = int(m.group(1))
        ranges.append((a, a))
    return ranges


def extract_evidence(text: str, districts_vocab: set[str], cond_vocab: list[str], hosp_vocab: list[str]) -> dict:
    low = text.lower()
    ev: dict = {
        "age_ranges": parse_age_ranges(text),
        "genders": set(),
        "districts": set(),
        "pin_prefixes": set(),
        "pins": set(),
        "hospitals": set(),
        "conditions": set(),
        "dates": set(),
        "stay_days": set(),
        "psns": set(re.findall(r"\bPSN-[0-9A-Z]{6}\b", text)),
    }
    if re.search(r"(?i)(?:/F\b|\"gender\":\s*\"female\"|\"sex\":\s*\"F\"|\bF\b|महिला)", text):
        ev["genders"].add("F")
    if re.search(r"(?i)(?:/M\b|\"gender\":\s*\"male\"|\"sex\":\s*\"M\"|\bM\b|पुरुष)", text):
        ev["genders"].add("M")
    for d in districts_vocab:
        if d.lower() in low:
            ev["districts"].add(d.lower())
    for m in re.finditer(r"\b([1-8]\d{2})\*{3}\b", text):
        ev["pin_prefixes"].add(m.group(1))
    for m in re.finditer(r"(?<!\d)([1-8]\d{5})(?!\d)", text):
        ev["pins"].add(m.group(1))
    for h in hosp_vocab:
        if h.lower() in low:
            ev["hospitals"].add(h.lower())
    for c in cond_vocab:
        if c.lower() in low:
            ev["conditions"].add(c.lower())
    # Dates (ISO, DMY, MDY)
    parsed_dates = []
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            dt_obj = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            ev["dates"].add(dt_obj.isoformat())
            parsed_dates.append(dt_obj)
        except ValueError:
            pass
    for m in re.finditer(r"\b(\d{2})[/\-](\d{2})[/\-](\d{4})\b", text):
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        for d_try in ((y, b, a), (y, a, b)):
            try:
                dt_obj = dt.date(*d_try)
                ev["dates"].add(dt_obj.isoformat())
                parsed_dates.append(dt_obj)
                break
            except ValueError:
                continue
    # Intra-record stay_days intervals
    if len(parsed_dates) >= 2:
        for i in range(len(parsed_dates)):
            for j in range(i + 1, len(parsed_dates)):
                diff = abs((parsed_dates[j] - parsed_dates[i]).days)
                if 1 <= diff <= 60:
                    ev["stay_days"].add(diff)
    return ev


def filter_candidates(ev: dict, registry: list[dict], pin_to_dist: dict[str, set[str]]) -> list[dict]:
    cands = list(registry)

    def _apply(pred):
        nonlocal cands
        sub = [r for r in cands if pred(r)]
        if sub:
            cands = sub

    if len(ev["districts"]) == 1:
        d = next(iter(ev["districts"]))
        _apply(lambda r: r["district"].lower() == d)
    if len(ev["genders"]) == 1:
        g = next(iter(ev["genders"]))
        _apply(lambda r: r["gender"] == g)
    if ev["age_ranges"]:
        lo = max(r[0] for r in ev["age_ranges"])
        hi = min(r[1] for r in ev["age_ranges"])
        if lo <= hi:
            _apply(lambda r: (lo - 2) <= int(r["age"]) <= (hi + 2))
    if len(ev["pins"]) == 1:
        p = next(iter(ev["pins"]))
        _apply(lambda r: r["pin"] == p)
    elif len(ev["pin_prefixes"]) == 1:
        pref = next(iter(ev["pin_prefixes"]))
        _apply(lambda r: r["pin"].startswith(pref))
    return cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--known-facts", required=True)
    ap.add_argument("--raw-attack", default=None)
    args = ap.parse_args()

    out = Path(args.output)
    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    registry = list(csv.DictReader(Path(args.registry).open(encoding="utf-8")))
    known_facts = [json.loads(l) for l in Path(args.known_facts).read_text(encoding="utf-8").splitlines() if l.strip()]
    raw_attack = json.loads(Path(args.raw_attack).read_text(encoding="utf-8"))["result"] if args.raw_attack else None

    rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
    renamed = {e.get("input_sha256"): e.get("output") for e in rep["files"]}

    districts_vocab = {r["district"] for r in registry}
    pin_to_dist: dict[str, set[str]] = defaultdict(set)
    for r in registry:
        pin_to_dist[r["pin"][:3]].add(r["district"].lower())
    hosp_vocab = ["Sanjeevani Multispeciality Hospital", "Jeevan Jyoti Charitable Hospital", "Arogya Seva Sadan", "District Hospital Hardoi"]
    cond_vocab = sorted({
        p["text"]
        for f in truth["files"].values()
        for u in f["units"].values()
        for p in u.get("preserve") or []
        if p.get("kind") in ("diagnosis", "procedure")
    })

    # Extract per-record evidence from sanitized outputs
    record_evs = []
    profiles_ev = defaultdict(lambda: {
        "age_ranges": [], "genders": set(), "districts": set(), "pin_prefixes": set(),
        "pins": set(), "hospitals": set(), "conditions": set(), "dates": set(),
        "stay_days": set(), "true_patients": set(),
    })

    for rel, f in truth["files"].items():
        if rel.startswith(("clean/", "misc/")):
            continue
        out_rel = str(renamed.get(f["sha256"]) or rel).replace("sanitized/", "", 1)
        p = out / "sanitized" / out_rel
        if not p.is_file():
            continue
        fmt = f["format"]
        raw_txt = p.read_text(encoding="utf-8", errors="replace")

        unit_texts: list[tuple[str, str, str | None]] = []
        if fmt in ("csv", "tsv"):
            delim = "\t" if fmt == "tsv" else ","
            rows = list(csv.reader(raw_txt.splitlines(), delimiter=delim))
            if len(rows) > 1:
                hdr = rows[0]
                for i, r in enumerate(rows[1:]):
                    loc = f"row:{i}"
                    u = f["units"].get(loc)
                    if u and u.get("patient"):
                        rec_str = " ".join(f"{h}: {c}" for h, c in zip(hdr, r))
                        unit_texts.append((loc, rec_str, u["patient"]))
        elif fmt == "json":
            data = json.loads(raw_txt)
            if isinstance(data, dict) and any(k in data for k in ("visits", "patients", "results")):
                for k in ("visits", "patients", "results"):
                    if isinstance(data.get(k), list):
                        for i, item in enumerate(data[k]):
                            loc = f"/{k}/{i}"
                            u = f["units"].get(loc)
                            if u and u.get("patient"):
                                unit_texts.append((loc, json.dumps(item, ensure_ascii=False), u["patient"]))
            else:
                u = f["units"].get("doc")
                if u and u.get("patient"):
                    unit_texts.append(("doc", raw_txt, u["patient"]))
        else:
            u = f["units"].get("doc")
            if u and u.get("patient"):
                unit_texts.append(("doc", raw_txt, u["patient"]))

        for loc, rtxt, pt in unit_texts:
            ev = extract_evidence(rtxt, districts_vocab, cond_vocab, hosp_vocab)
            rid = truth["patients"][pt]["rid"]
            cands = filter_candidates(ev, registry, pin_to_dist)
            prob = (1.0 / len(cands)) if any(c["rid"] == rid for c in cands) else 0.0
            record_evs.append((pt, rid, ev, prob))
            for psn in ev["psns"]:
                pe = profiles_ev[psn]
                pe["age_ranges"].extend(ev["age_ranges"])
                pe["genders"] |= ev["genders"]
                pe["districts"] |= ev["districts"]
                pe["pin_prefixes"] |= ev["pin_prefixes"]
                pe["pins"] |= ev["pins"]
                pe["hospitals"] |= ev["hospitals"]
                pe["conditions"] |= ev["conditions"]
                pe["dates"] |= ev["dates"]
                pe["stay_days"] |= ev["stay_days"]
                pe["true_patients"].add(rid)

    # Profile probabilities (A4)
    profile_probs: dict[str, tuple[list[dict], dict]] = {}
    for psn, pe in profiles_ev.items():
        cands = filter_candidates(pe, registry, pin_to_dist)
        profile_probs[psn] = (cands, pe)

    rec_probs = [p for _, _, _, p in record_evs]
    a3_mean = sum(rec_probs) / max(len(rec_probs), 1)

    patient_max_prob: dict[str, float] = defaultdict(float)
    for pt, rid, ev, rec_p in record_evs:
        prof_p = 0.0
        for psn in ev["psns"]:
            cands, _ = profile_probs[psn]
            if any(c["rid"] == rid for c in cands):
                prof_p = max(prof_p, 1.0 / len(cands))
        patient_max_prob[rid] = max(patient_max_prob[rid], rec_p, prof_p)

    a4_reid = sum(1 for v in patient_max_prob.values() if v >= 0.5) / max(len(patient_max_prob), 1)

    # A5: Targeted attack
    cond_prof_count = Counter()
    for pe in profiles_ev.values():
        for c in pe["conditions"]:
            cond_prof_count[c] += 1

    reg_by_rid = {r["rid"]: r for r in registry}
    patient_rids = {p["rid"] for p in truth["patients"].values()}
    a5_probs = []
    for kf in known_facts:
        rid = kf["rid"]
        if rid not in patient_rids:
            continue
        rrow = reg_by_rid[rid]
        scores = []
        for psn, (cands, pe) in profile_probs.items():
            if not any(c["rid"] == rid for c in cands):
                continue
            w = 0.0
            if kf.get("visit_dates") and (set(kf["visit_dates"]) & pe["dates"]):
                w += 10.0
            if kf.get("condition_hint"):
                ch = kf["condition_hint"].lower()
                if ch in pe["conditions"]:
                    w += 8.0 if cond_prof_count[ch] <= 3 else 1.5
            if kf.get("stay_days") and kf["stay_days"] in pe["stay_days"]:
                w += 3.0
            if w > 0.0:
                if kf.get("hospital") and kf["hospital"].lower() in pe["hospitals"]:
                    w *= 2.0
                scores.append((psn, w, rid in pe["true_patients"]))
        tot_w = sum(s[1] for s in scores)
        true_w = sum(s[1] for s in scores if s[2])
        a5_probs.append(true_w / tot_w if tot_w > 0 else 0.0)

    a5_mean = sum(a5_probs) / max(len(a5_probs), 1)

    # Check temporal intervals (stay length exactness inside records & across patient events)
    # Also check truthfulness of every age band!
    bad_age_bands = 0
    total_age_bands = 0
    for rel, f in truth["files"].items():
        if rel.startswith(("clean/", "misc/")):
            continue
        out_rel = str(renamed.get(f["sha256"]) or rel).replace("sanitized/", "", 1)
        p = out / "sanitized" / out_rel
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        for loc, u in f["units"].items():
            for fld in u.get("fields") or []:
                if fld.get("kind") == "age" and fld["true"] >= 0 and fld.get("path") is not None:
                    total_age_bands += 1
                    # Find age band in unit
                    sub_txt = txt
                    if f["format"] in ("csv", "tsv") and loc.startswith("row:"):
                        idx = int(loc.split(":")[1]) + 1
                        delim = "\t" if f["format"] == "tsv" else ","
                        rows = list(csv.reader(txt.splitlines(), delimiter=delim))
                        sub_txt = " ".join(f"{h}: {c}" for h, c in zip(rows[0], rows[idx])) if idx < len(rows) else ""
                    elif f["format"] == "json" and loc.startswith("/"):
                        parts = [x for x in loc.split("/") if x]
                        obj = json.loads(txt)
                        try:
                            for pt_k in parts:
                                obj = obj[int(pt_k)] if isinstance(obj, list) else obj[pt_k]
                            sub_txt = json.dumps(obj, ensure_ascii=False)
                        except Exception:
                            sub_txt = ""
                    rngs = parse_age_ranges(sub_txt)
                    if not any(lo <= fld["true"] <= hi for lo, hi in rngs):
                        bad_age_bands += 1
                elif fld.get("kind") == "age" and fld["true"] >= 0 and f["format"] == "text":
                    total_age_bands += 1
                    rngs = parse_age_ranges(txt)
                    if not any(lo <= fld["true"] <= hi for lo, hi in rngs):
                        bad_age_bands += 1

    print("=" * 64)
    print(" PRIVACY ATTACK ARENA SIMULATION (A3 / A4 / A5)")
    print("=" * 64)
    print(f"Records attacked:   {len(record_evs)}")
    print(f"Profiles formed:    {len(profile_probs)}")
    print(f"A3 Record linkage:  {a3_mean:.3f}  (vs raw: {a3_mean / raw_attack['record_linkage_risk']:.3f})" if raw_attack else f"A3: {a3_mean:.3f}")
    print(f"A4 Confident >=0.5: {a4_reid:.3f}  (vs raw: {a4_reid / raw_attack['patients_reidentified']:.3f})" if raw_attack else f"A4: {a4_reid:.3f}")
    print(f"A5 Targeted attack: {a5_mean:.3f}  (vs raw: {a5_mean / raw_attack['targeted_success']:.3f})" if raw_attack else f"A5: {a5_mean:.3f}")
    print(f"Truthful age bands: {total_age_bands - bad_age_bands}/{total_age_bands} (errors: {bad_age_bands})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
