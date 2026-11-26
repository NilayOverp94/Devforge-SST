# Raksha Kavach: Context-Guarded Multilingual Clinical Privacy Firewall

**Track 3 · Secret Wards** · Team: `raksha-kavach` · Offline · CPU-Only · LLM-Free

## 1. In one paragraph
Raksha Kavach is an offline, deterministic privacy firewall engineered for heterogeneous Indian clinical exports across English, Hindi (Devanagari), and Hinglish. Unlike generic regex or NLP redactors that drop Hindi names, strip medical eponyms/doses, and fail against voter-list linkage attacks, Raksha Kavach runs a **preserve-first clinical guard** paired with an **Indic transliteration + consonant-skeleton linker**, a **2,587-parameter Averaged Perceptron NER classifier**, and **cohort-risk-adaptive k-anonymity generalization** inside an AES-256-GCM memory vault.

## 2. Architecture
```
[Input Tree: CSV/TSV/JSON/FHIR/TXT]
       │
       ▼
[1. Ingestion & Quarantine] ──(empty/binary/corrupt/mislabeled/depth>200)──▶ [Quarantined in report.json]
       │ (UTF-8 / UTF-16 / clean cp1252)
       ▼
[2. AES-256-GCM In-Memory Vault + Date-Hint Consensus]
       │
       ▼
[3. Multilingual Detection + Clinical Meaning Guards] (Latin/Deva/Hinglish + ML NER + hidden channels)
       │
       ▼
[4. Cross-Script Record Linkage + HKDF/HMAC Pseudonyms] (indic-transliteration + skeletons -> PSN-XXXXXX)
       │
       ▼
[5. Risk-Adaptive Privacy Layer] (per-patient date shift ±366..540d, 10/20-yr bands, district/PIN coarsening)
       │
       ▼
[sanitized/ (UTF-8, valid FHIR R4/JSON/CSV) + report.json (SHA-256) + records.jsonl]
```

## 3. How it works
- **Ingestion:** Walks sorted relative paths; decodes UTF-8/BOM, UTF-16, and safe cp1252; validates JSON depth iteratively (`<=200`) and FHIR R4 reference resolution; quarantines `.pdf`/`.jpg`/`.xml`, truncated JSON, mislabeled CSV, and corrupt byte streams without ever copying raw files.
- **Detection:** Runs 1-to-1 Indic digit & homoglyph shadow normalization, clinical preserve-span guards (`Dr.`, `K/C/O`, `C/O <symptom>`, medical eponyms, `Ward N (Dept)`, `ASHA worker`, `1-0-1`, dosages/labs), header + value-profiled column classification, base64/metadata/key/filename scanning, and a compact 2,587-weight char-ngram + context Averaged Perceptron (`name_ner.json`) to catch unseen names and canaries.
- **Linkage & pseudonyms:** Transliterates Devanagari to ITRANS with nukta rules and compares Indic consonant skeletons (`रीना` ↔ `Reena` ↔ `Rina`) across two conservative stages (strong IDs first, then bio/relative/village corroboration with hard demographic conflict checks). Assigns deterministic Crockford `PSN-XXXXXX` via HKDF + HMAC-SHA256.
- **Privacy layer:** Applies per-patient HMAC date shifts in `[-540,-366] ∪ [366,540]` days (preserving exact stay lengths and intervals while preventing intra-year visit collisions), suppresses DOB and precise addresses/tehsils, and applies 10-year vs. 20-year age bands + state-level district coarsening on sparse (`k < 5`), elderly (`>= 80`), or rare-condition (`<= 3` patients) profiles.
- **Security:** HKDF-SHA256 separates `RAKSHA_SECRET` into domain subkeys (`K_PSEUDO`, `K_SHIFT`, `K_TAG`, `K_VAULT`); raw intermediate buffers are sealed with AES-256-GCM in RAM and purged before exit; `stdout`/`stderr`/`report.json`/filenames/temp dirs contain zero PII or secrets.

## 4. USP
1. **Preserve-First Meaning Traps:** Protects `Dr. Ramesh` while redacting `C/O Ramesh` in the same document, keeps `C/O fever` while redacting `C/O <Relative>`, and preserves unseen `<Name>'s disease/syndrome/palsy` eponyms and `ASHA worker`.
2. **Cross-Script & Unseen-Schema Generalization:** Consonant-skeleton matching links Hindi/Hinglish/English variants while value-profiled columns (`>=60%` pattern match) handle renamed headers on unseen datasets.
3. **Cohort-Aware Adaptive Generalization:** Combines `<=3`-patient rare-condition detection and `(district, gender, age_band)` cell sizing to defend against registry and targeted attacks without deleting clinical data.

## 5. Privacy–utility trade-off (measured on `dev/input` & unseen chaos test)

| Measure | Measured Value |
|---|---|
| Personal data leaked (`check_dev.py`: overall / Hindi / Hinglish) | **0.0% (0/3,230) / 0.0% / 0.0%** |
| Canaries leaked / Raw malformed files copied | **0 of 4** / **0** |
| Medical values preserved (`check_dev.py`) | **98.1% (4,763/4,855)** (99.8% rare-only) |
| Truthful age bands / Exact length-of-stay | **532/532 (100%)** / **65/65 (100%)** |
| Linking precision / recall / F0.5 / wrong merges | **1.00 / 1.00 / 1.00 / 0** |
| Attack Arena risk (`eval_attacks.py` A3 / A4 / A5) | **0.020 (0.022 vs raw) / 0.021 / 0.411** |
| Profiles unique on `(age_band, district, gender)` | **4 / 48 (8.3%)** (down from 20 / 48) |
| Run time / Peak memory (`report.json`) | **~2.0 s / ~59 MB RSS** |
| Unseen chaos suite (`tests/test_chaos.py`) | **0/19 PII leaked, 10/10 clinical kept, 100% link** |

Generalizing sparse/rare districts and widening age bands to 10/20 years drops `A3` record linkage risk by **98%** vs raw (`0.022`) and `A4` re-identification to **2.1%**, at a modest utility cost of **1.9%** on non-clinical district probes while keeping **100%** of clinical diagnoses, labs, doses, and stay lengths.

## 6. Known limitations
1. Patients with a 1-of-1 rare condition/procedure in a tiny cohort remain statistically distinguishable under `known_facts` unless clinical diagnoses themselves are coarsened.
2. Un-anchored lowercase single-token names in unexpected prose fields outside gazetteers/ML patterns can evade detection.
3. Binary containers (`.pdf`, `.jpg`, `.xml`) are safely quarantined rather than OCR-parsed to guarantee deterministic offline CPU compliance.
