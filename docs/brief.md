# Raksha Kavach: Context-Guarded Multilingual Clinical Privacy Firewall

**Track 3 · Secret Wards · Team:** `raksha-kavach` · **Repo:** `https://github.com/NilayOverp94/Devforge-SST` · **100% Offline · CPU-Only · LLM-Free**

## 1. Overview & Problem Pitch
**Raksha Kavach** is a deterministic, CPU-only privacy firewall built for messy Indian hospital dumps spanning **English, Hindi (Devanagari), and Hinglish**. Generic regex/NER tools miss Devanagari & transliterated names, accidentally redact medical eponyms or dosing frequencies, fail on multi-GB corpora, and collapse under external linkage attacks. Raksha Kavach combines a **preserve-first clinical guard**, a **2,587-parameter Averaged Perceptron NER**, an **Indic consonant-skeleton linker**, **cohort-risk-adaptive k-anonymity**, and a **multi-process streaming engine** capable of sanitizing **4.95M records (2.15 GB) in under 4 minutes inside ~3.1 GB RAM**.

## 2. Architecture
```
[Input Tree: CSV/TSV/JSON/FHIR/TXT]
       │
       ├──▶ [1. Ingestion & Quarantine] ──(empty/binary/corrupt/mislabeled/depth>200)──▶ [report.json (Never copied)]
       │         (UTF-8/BOM, UTF-16, clean cp1252)
       ▼
[2. AES-256-GCM Vault + Date-Hint Consensus] ──▶ [3. Multilingual Detection + Clinical Guards + ML NER]
                                                         │
                                                         ▼
[4. Cross-Script Linkage & HMAC Pseudonyms] ──(Lazy Union-Find + Consonant Skeletons)──▶ PSN-XXXXXX
       │
       ▼
[5. Risk-Adaptive Privacy Layer + Streaming Export] ──▶ sanitized/ (UTF-8) + report.json + records.jsonl
```

## 3. Approach & Implementation
- **Ingestion & Fail-Safe Quarantine (`src/ingestion.py`):** Walks sorted POSIX paths; decodes UTF-8/BOM, UTF-16, and safe `cp1252`; enforces an iterative JSON nesting cap (`<= 200`) and FHIR R4 reference checks; quarantines `.pdf`/`.jpg`/`.xml`, truncated JSON, mislabeled CSVs, and corrupt byte streams without copying raw bytes.
- **Detection, Clinical Guards & ML NER (`src/detection.py`, `src/models/name_ner.json`):** Performs 1-to-1 Indic digit (`०–९`) and homoglyph shadow normalization; protects clinical spans (`Dr.`, `K/C/O`, `C/O <symptom>`, eponyms, `Ward N (Dept)`, `ASHA worker`, `1-0-1`, vitals/labs/ICD-10) *before* PII redaction; infers CSV/JSON schemas via headers + `>= 60%` value profiling; inspects filenames, keys, headers, metadata paths, and base64 payloads; and runs a **2,587-weight sparse Averaged Perceptron** (`name_ner.json`) to catch unseen names and canaries.
- **Cross-Script Record Linkage (`src/linkage.py`):** Transliterates Devanagari (`indic-transliteration` + nukta rules) into consonant skeletons to match Hindi/Hinglish/Latin variants across two conservative stages (Stage 1 exact IDs; Stage 2 bio/relative/village corroboration with hard demographic conflict checks). Assigns deterministic Crockford `PSN-XXXXXX` pseudonyms via HKDF + HMAC-SHA256.
- **Risk-Adaptive Privacy Layer (`src/privacy.py`, `config/policy.yaml`):** Shifts dates per patient by a keyed offset in `[-540,-366] ∪ [366,540]` days (preserving stay lengths & intervals), suppresses DOB/houses/tehsils/villages, and applies 10/20-year age bands plus state-level district coarsening for sparse (`k < 5`), elderly (`>= 80`), or rare-condition (`<= 3` patients) cohorts.
- **Multi-GB Streaming & Security (`main.py`, `src/security.py`, `gui.py`):** Uses bounded-RAM file/chunk workers (`ProcessPoolExecutor`), packed `str | tuple` `RecordClues` slots, lazy Union-Find clustering, and atomic `.part` commits; seals intermediate buffers in an in-memory **AES-256-GCM** vault (`HKDF-SHA256` domain subkeys from `RAKSHA_SECRET`) with zero PII or secrets written to logs or temp storage. Delivered via CLI, a **5-tab offline localhost console (`gui.py`)**, and packaged **macOS `.dmg`**, **Windows `.exe`**, and **Android `.apk`** builds.

## 4. Key Outcomes & Measured Privacy–Utility Trade-Off

| Benchmark / Metric | Measured Result | Scope / Notes |
|---|---|---|
| PII leaked (`check_dev.py`, `dev/input`: 200 files) | **0.0% (0 / 3,230)** | `0.0%` across Latin, Devanagari, Hinglish & all 6 channels |
| Planted canaries / Raw malformed files copied | **0 / 4** canaries · **0** raw copies | 8 broken/unsupported inputs quarantined cleanly |
| Medical data preserved (`check_dev.py`) | **98.1% (4,763 / 4,855)** | **532/532 (100%)** truthful age bands; **100%** LOS kept |
| Cross-script patient linking (`check_dev.py`) | **P 1.00 / R 1.00 / F0.5 1.00** | **0** wrong homonym merges across 48 patients |
| Attack Arena (`tests/eval_attacks.py`) | **A3 0.020 · A4 0.021 · A5 0.411** | Registry linkage cut vs raw (`0.022`); 0 date leaks |
| Practice run speed & RAM (`report.json`) | **0.9 s · 58.58 MB peak RSS** | Offline CPU-only (`validate_output.py`: `CONTRACT OK`) |
| **Multi-GB Scale Corpus (`2.15 GB`, 39 files)** | **4,954,024 records in 235.91 s** | **51.12M** PII detected; **3.04 GiB** max RSS (`CONTRACT OK`) |
| Unseen chaos suite (`tests/test_chaos.py` + `test_gui.py`) | **0 / 19 PII · 10 / 10 clinical · 6/6 GUI** | Unseen names, renamed columns, `DD.MM.YYYY`, base64 |

**Trade-off Analysis:** Suppressing granular addresses/DOB, shifting visit dates outside ±1 year, and widening sparse/rare-condition cohorts to 20-year age bands and state-level districts trades **1.9%** on geographic probes to achieve **0.0% PII leakage** and **0.020** A3 linkage risk while keeping **100%** of clinical diagnoses, medications, doses, labs, vitals, and stay intervals intact.

## 5. Known Limitations & Future Work
1. **Unique Clinical Signatures (A5):** 1-of-1 rare procedures/diagnoses in tiny cohorts remain statistically distinguishable under external `known_facts` unless diagnoses are coarsened to parent ICD-10 chapters.
2. **Un-anchored Lowercase Single-Token Names:** Isolated lowercase single names in prose without honorifics or gazetteer context are conservatively preserved to protect clinical vocabulary.
3. **Binary Containers:** `.pdf`, `.jpg`, and `.xml` files are quarantined by design instead of OCR-processed to guarantee bounded offline CPU/memory execution.
