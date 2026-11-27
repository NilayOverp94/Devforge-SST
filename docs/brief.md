# Raksha Kavach: Multilingual Clinical Privacy Firewall

**Track 3 · Secret Wards · Team:** `raksha-kavach` · **Repo:** `https://github.com/NilayOverp94/Devforge-SST` · **100% Offline · CPU-Only · LLM-Free**

## 1. Project Pitch
**Raksha Kavach** is an offline, deterministic privacy firewall for Indian clinical datasets across **English, Hindi (Devanagari), and Hinglish**. Unlike standard PII tools that miss Hindi/Hinglish names, strip medical terms, or collapse on multi-GB dumps, it pairs **preserve-first clinical guards** with **cross-script identity linkage**, **cohort-adaptive k-anonymity**, and **multi-process streaming** — processing **4.95M records (2.15 GB) in 235.9 s**.

## 2. Architecture
```
Input (CSV/TSV/JSON/FHIR/TXT)
  ├──▶ 1. Ingestion & Quarantine ──(corrupt/binary/deep JSON)──▶ report.json (never copied)
  ▼
2. Clinical Guards + Multilingual NER (Latin/Deva/Hinglish + 2,587-param Perceptron)
  ▼
3. Cross-Script Linkage (Indic Consonant Skeletons + Lazy Union-Find) ──▶ PSN-XXXXXX (HMAC)
  ▼
4. Risk-Adaptive Privacy Layer (±366–540d Shift, 10/20-yr Bands, Cohort Generalization)
  ▼
Outputs: sanitized/ (UTF-8) + report.json (SHA-256) + records.jsonl
```

## 3. Core System Design
- **Fail-Safe Ingestion (`src/ingestion.py`):** Decodes UTF-8/BOM, UTF-16, and safe `cp1252`; validates JSON depth (`<= 200`) and FHIR R4 links; quarantines `.pdf`/`.jpg`/`.xml`, truncated JSON, and corrupt streams without copying raw bytes.
- **Detection & Clinical Guards (`src/detection.py`):** Protects clinical spans (`Dr.`, `K/C/O`, `C/O <symptom>`, eponyms, wards, `ASHA worker`, doses/vitals/ICD-10) *before* scrubbing PII across text, CSV/JSON headers, filenames, metadata, and base64; uses a **2,587-weight Averaged Perceptron** (`name_ner.json`) to catch unseen names.
- **Cross-Script Linkage (`src/linkage.py`):** Transliterates Devanagari into Indic consonant skeletons to link Hindi/Hinglish/Latin records across exact IDs and corroborated demographics with conflict guards, assigning HKDF/HMAC-SHA256 `PSN-XXXXXX` pseudonyms.
- **Privacy Layer (`src/privacy.py`):** Shifts visit dates per patient by `±366–540` days (preserving exact stay lengths), strips granular addresses/DOB, and widens sparse (`k < 5`), elderly (`>= 80`), or rare-condition (`<= 3`) cohorts to 20-year age bands and state-level regions.
- **Security & Scale (`main.py`, `src/security.py`, `gui.py`):** Buffers raw data inside an in-memory **AES-256-GCM** vault (`HKDF-SHA256` keys) with zero PII in logs; scales to multi-GB corpora using packed slots and parallel streaming; includes an offline 5-tab GUI and standalone `.dmg`, `.exe`, and `.apk` builds.

## 4. Key Outcomes & Privacy–Utility Trade-Off

| Evaluation Metric | Measured Result | Scope |
|---|---|---|
| PII leaked (`dev/input`, 200 files) | **0.0% (0 / 3,230)** | `0.0%` across English, Hindi, Hinglish & all 6 channels |
| Canaries leaked / Malformed raw copies | **0 / 4** canaries · **0** raw | 8 broken/unsupported files quarantined cleanly |
| Medical data preserved (`check_dev.py`) | **98.1% (4,763 / 4,855)** | **100%** truthful age bands (`532/532`) & exact stay lengths |
| Cross-script linking (`check_dev.py`) | **P 1.00 / R 1.00 / F0.5 1.00** | **0** homonym collisions across 48 patients |
| Attack Arena risk (`eval_attacks.py`) | **A3 0.020 · A4 0.021 · A5 0.411** | Practice run: **0.9 s · 58.58 MB** RSS (`CONTRACT OK`) |
| **Multi-GB Stress Test (`2.15 GB`)** | **4,954,024 recs in 235.91 s** | **51.12M** PII scrubbed · **3.04 GiB** max RSS (`CONTRACT OK`) |
| Unseen chaos & GUI (`test_chaos.py`, `test_gui.py`) | **0/19 PII · 10/10 clin. · 6/6 GUI** | Unseen names, renamed columns, `DD.MM.YYYY`, base64 |

**Trade-Off:** Widening sparse/rare cohorts to 20-year bands and state-level districts costs **1.9%** on non-clinical geographic probes while keeping **A3** linkage risk at **0.020** and preserving **100%** of diagnoses, medications, doses, vitals, and stay intervals.

## 5. Unique Differentiators (USP)
1. **Context-Aware Meaning Traps:** Preserves `Dr. Ramesh` and `C/O chest pain` while redacting patient/relative names in the same record.
2. **Zero-LLM Multi-GB Speed:** Scales from 816 records (`0.9 s`) to **4.95M records (`235.9 s`)** purely on CPU without external APIs or GPUs.
3. **Cross-Platform Offline Delivery:** Runs identically via CLI, 5-tab localhost web console, and standalone **macOS `.dmg`**, **Windows `.exe`**, and **Android `.apk`** packages.

## 6. Known Limitations
1. **1-of-1 Rare Conditions (A5 = 0.411):** Unique procedures/diagnoses remain distinguishable under external `known_facts` unless generalized to broader ICD-10 chapters.
2. **Un-anchored Lowercase Names:** Isolated lowercase names in prose without honorifics or gazetteer overlap are left intact to protect clinical words.
3. **Binary Attachments:** `.pdf`, `.jpg`, and `.xml` files are quarantined instead of OCR-processed to guarantee bounded offline CPU/memory execution.
