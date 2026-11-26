"""Offline trainer for the lightweight classical ML Averaged Perceptron Name-vs-Medical classifier.

Generates purely synthetic Indian + international names, medical terms, drug doses, and clinical
contexts (completely disjoint from dev/input) and trains a compact linear averaged perceptron
(< 50K parameters, saved to src/models/name_ner.json) for CPU-only, LLM-free inference.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from indic_transliteration import sanscript


FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna",
    "Ishaan", "Shaurya", "Atharv", "Advik", "Pranav", "Kabir", "Ananya", "Diya", "Saanvi",
    "Aadhya", "Pari", "Myra", "Navya", "Ira", "Meera", "Riya", "Kavya", "Nitya", "Tara",
    "Rohan", "Rahul", "Amit", "Suresh", "Rajesh", "Mahesh", "Ramesh", "Dinesh", "Mukesh",
    "Naresh", "Kamlesh", "Umesh", "Yogesh", "Ganesh", "Manoj", "Vikram", "Deepak", "Ashok",
    "Sanjay", "Ajay", "Vijay", "Sunil", "Anil", "Pankaj", "Nikhil", "Nitin", "Gaurav",
    "Harish", "Hareesh", "Girish", "Satish", "Prakash", "Subhash", "Kishan", "Mohan", "Sohan",
    "Raju", "Brijesh", "Brajesh", "Kamal", "Vimal", "Vikas", "Kunal", "Tarun", "Varun",
    "Arun", "Karan", "Dev", "Shyam", "Ram", "Lakshman", "Bharat", "Prem", "Om", "Shiv",
    "Sunita", "Suneeta", "Anita", "Aneeta", "Geeta", "Gita", "Sita", "Rita", "Reeta",
    "Reena", "Rina", "Meena", "Mina", "Seema", "Sima", "Pooja", "Puja", "Neha", "Sneha",
    "Asha", "Usha", "Nisha", "Kavita", "Savita", "Babita", "Sarita", "Mamta", "Suman",
    "Kusum", "Kiran", "Rekha", "Shobha", "Prabha", "Lakshmi", "Laxmi", "Parvati", "Parvathi",
    "Kamla", "Kamala", "Shanti", "Shanthi", "Radha", "Sudha", "Madhu", "Priya", "Swati",
    "Jyoti", "Archana", "Vandana", "Kalpana", "Ranjana", "Manju", "Anju", "Renu", "Indu",
    "Mohammad", "Mohd", "Md", "Arif", "Shahid", "Irfan", "Imran", "Salman", "Aamir",
    "Faisal", "Tariq", "Zubair", "Javed", "Akram", "Yusuf", "Naseem", "Salim", "Saleem",
    "Wasim", "Aslam", "Iqbal", "Rizwan", "Sajid", "Majid", "Hamid", "Rashid", "Khalid",
    "Fatima", "Fatma", "Zainab", "Zenab", "Ayesha", "Aisha", "Salma", "Shabana", "Nasreen",
    "Rukhsana", "Farida", "Fareeda", "Sultana", "Shahnaz", "Nargis", "Yasmin", "Heena",
    "Jaspreet", "Sukhwinder", "Sukhvinder", "Harpreet", "Manpreet", "Gurpreet", "Kuldeep",
    "Balwinder", "Amrit", "Simran", "Navdeep", "Parminder", "Harjeet", "Charanjit",
    "Mary", "Anthony", "Joseph", "Thomas", "David", "Peter", "Paul", "John", "Grace",
    "Zorawar", "Ilavenil", "Tsering", "Kaveri", "Bhavna", "Devendra", "Hemant", "Lalit",
    "Chetan", "Pradeep", "Pramod", "Vinod", "Alok", "Vivek", "Abhishek", "Siddharth",
    "Shubham", "Rishabh", "Tushar", "Akash", "Mayank", "Vaibhav", "Keshav", "Madhav",
    "Chandra", "Surya", "Indra", "Narendra", "Mahendra", "Surendra", "Ravindra", "Jitendra",
]

LAST_NAMES = [
    "Sharma", "Sarma", "Verma", "Varma", "Gupta", "Agarwal", "Agrawal", "Jain", "Bansal",
    "Mittal", "Goyal", "Singhal", "Joshi", "Pandey", "Pande", "Pandeya", "Tiwari", "Tewari",
    "Mishra", "Misra", "Shukla", "Dubey", "Dwivedi", "Tripathi", "Chaturvedi", "Pathak",
    "Srivastava", "Saxena", "Mathur", "Rastogi", "Bhatnagar", "Nigam", "Kulshrestha",
    "Singh", "Kaur", "Gill", "Sandhu", "Sidhu", "Grewal", "Dhillon", "Bedi", "Sodhi",
    "Yadav", "Chauhan", "Chouhan", "Chaudhary", "Choudhary", "Patel", "Patil", "Desai",
    "Mehta", "Shah", "Thakur", "Rajput", "Rathore", "Bisht", "Negi", "Rawat", "Kushwaha",
    "Maurya", "Nishad", "Paswan", "Lodhi", "Sahu", "Jaiswal", "Kesarwani", "Bhardwaj",
    "Ansari", "Siddiqui", "Siddiqi", "Qureshi", "Khan", "Sheikh", "Shaikh", "Pathan",
    "Syed", "Rizvi", "Zaidi", "Naqvi", "Qadri", "Malik", "Mirza", "Baig", "Bano", "Khatun",
    "Begum", "Devi", "Kumari", "Das", "Bose", "Sen", "Roy", "Bhattacharya", "Chatterjee",
    "Mukherjee", "Banerjee", "Ganguly", "Ghosh", "Dutta", "Nair", "Menon", "Nambiar",
    "Iyer", "Iyengar", "Subramanian", "Krishnan", "Reddy", "Rao", "Naidu", "Murthy",
    "Wangmo", "Dorje", "Norbu", "Masih", "Fernandes", "D'Souza", "Rodrigues", "Gomes",
]

DEVA_FIRST = [
    "कमला", "रीना", "हरीश", "नेहा", "कमल", "ज़ैनब", "अनीता", "महेश", "मुकेश", "फ़ातिमा",
    "सरिता", "शांति", "गीता", "मोहम्मद", "अशोक", "दिनेश", "मनोज", "पार्वती", "सुनीता",
    "सुखविंदर", "सलमा", "शबाना", "उषा", "कविता", "विजय", "बृजेश", "ब्रजेश", "राहुल",
    "रमेश", "सुरेश", "अजय", "प्रकाश", "किशन", "दीपक", "जगदीश", "नरेश", "रवि", "ज़ुबैर",
    "अकरम", "यूसुफ", "कुलदीप", "जावेद", "अनिल", "रोहित", "सोहन", "मीरा", "आशा", "राधा",
]

DEVA_LAST = [
    "निषाद", "पांडे", "पासवान", "ठाकुर", "सिद्दीकी", "शर्मा", "चौहान", "क़ुरैशी", "तिवारी",
    "चौधरी", "यादव", "लोधी", "पटेल", "मिश्रा", "राजपूत", "कौर", "अंसारी", "बानो",
    "ख़ातून", "शेख़", "कुमार", "वर्मा", "गुप्ता", "सिंह", "देवी", "खान", "मलिक", "कादरी",
]

MEDICAL_TERMS = [
    "Paracetamol", "Amlodipine", "Telmisartan", "Ramipril", "Metformin", "Glimepiride",
    "Azithromycin", "Ceftriaxone", "Doxycycline", "Pantoprazole", "Levothyroxine",
    "Atorvastatin", "Artesunate", "Dolo", "Ferrous Sulphate", "Folic Acid", "Levodopa",
    "Carbidopa", "Insulin Glargine", "Rifampicin", "Isoniazid", "Salbutamol", "Budesonide",
    "Essential Hypertension", "Type Diabetes", "Typhoid Fever", "Chronic Kidney Disease",
    "Urinary Tract", "Pulmonary Tuberculosis", "Bell's Palsy", "Parkinson's Disease",
    "Addison's Disease", "Crohn's Disease", "Wilson's Disease", "Alzheimer's Disease",
    "Infectious Gastroenteritis", "Visceral Leishmaniasis", "Laparoscopic Cholecystectomy",
    "Inguinal Hernia", "Caesarean Section", "Haemodialysis Session", "Deep Brain Stimulation",
    "Whipple Procedure", "Cataract Surgery", "Phacoemulsification", "Appendicectomy",
    "Haemoglobin", "Random Blood", "Serum Creatinine", "Platelet Count", "Total WBC",
    "Serum Sodium", "Dengue Fever", "Unspecified Malaria", "Hypothyroidism Unspecified",
    "Asthma Unspecified", "Anaemia Unspecified", "Pneumonia Unspecified", "Rabies Unspecified",
    "बुखार", "पेशाब में संक्रमण", "फेफड़ों की टीबी", "किडनी की पुरानी बीमारी", "सीओपीडी",
    "उच्च रक्तचाप", "टाइप 2 डायबिटीज़", "दमा", "रेबीज़", "निमोनिया", "मलेरिया", "काला-अज़ार",
    "दस्त और उल्टी", "पेट दर्द", "चक्कर आना",
]

NON_PERSON_CAPITALIZED = [
    "Discharge Summary", "Nursing Notes", "OPD Prescription", "Ward Medicine",
    "Surgical Consent", "Lifeline Pathology", "City Diagnostics", "Pathkind Labs",
    "Sanjeevani Hospital", "Charitable Hospital", "District Hospital", "Seva Sadan",
    "Multispeciality Hospital", "Low Salt", "Review After", "Stable Condition",
    "Diagnostic Report", "Routine Panel", "Reference Range", "Morning Round",
    "Evening Round", "Primary School", "Hanuman Mandir", "Kali Mata", "Jama Masjid",
    "Panchayat Bhawan", "Pani Tanki", "Shiv Mandir", "Indira Colony", "Subhash Nagar",
    "Kashiram Colony", "Nai Basti", "Purani Basti", "Chowk Kotwali", "Harijan Basti",
    "Rae Bareli", "Sultanpur", "Barabanki", "Sitapur", "Bahraich", "Hardoi", "Unnao",
    "Lucknow", "Uttar Pradesh", "ASHA Worker", "Medicine Ward", "Surgery Ward",
    "Pulmonology Dept", "Neurology Dept", "Referred By", "Emergency Room", "Intensive Care",
]

POS_LEFT = [
    "name:", "patient name:", "naam:", "नाम:", "मरीज़ का नाम:", "s/o", "d/o", "w/o",
    "guardian:", "attendant:", "called by", "witnessed by", "raised by", "report of",
    "opd note for", "rx of", "referral:", "mr.", "mrs.", "shri", "smt.", "पिता का नाम:",
    "पति का नाम:", "पता:", "c/o", "",
]
POS_RIGHT = [
    "ji", "sahab", "didi", "behenji", "bhaiya", "babu", "amma", "ko", "ka", "ki", "ne",
    "को", "का", "की", "age:", "age/sex:", "uhid:", "mob:", "mobile", "(attendant,",
    "w/o", "s/o", "d/o", "discharged", "48", "65", "",
]
NEG_LEFT = [
    "dr.", "dr", "doctor", "डॉ.", "consultant:", "referred by", "ref_doctor:", "k/c/o",
    "h/o", "diagnosis:", "निदान:", "procedure:", "rx", "tab", "cap", "inj", "continue",
    "treated for", "tehsil", "तहसील", "distt.", "district", "जिला", "ward", "hospital:",
    "reviewed by", "near", "opp.", "behind", "",
]
NEG_RIGHT = [
    "disease", "syndrome", "palsy", "fever", "infection", "mg", "mcg", "ml", "g/dl",
    "mg/dl", "hospital", "clinic", "diagnostics", "pathology", "labs", "worker",
    "reviewed", "(md", "(ms", "medicine)", "surgery)", "unspecified", "0-0-1", "1-0-1",
    "bd", "x 5 days", "",
]


def _normalize_token(tok: str) -> str:
    if any("\u0900" <= c <= "\u097f" for c in tok):
        try:
            tok = sanscript.transliterate(tok, sanscript.DEVANAGARI, sanscript.ITRANS)
        except Exception:
            pass
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def extract_features(span_text: str, left_ctx: str = "", right_ctx: str = "") -> dict[str, float]:
    """Extract lightweight lexical, char-ngram, shape, and context features for a 1-3 word span."""
    feats: dict[str, float] = {"bias": 1.0}
    raw_tokens = span_text.strip().split()
    if not raw_tokens:
        return feats

    n_tok = len(raw_tokens)
    feats[f"ntok={min(n_tok, 4)}"] = 1.0
    has_deva = any("\u0900" <= c <= "\u097f" for c in span_text)
    feats["script=deva" if has_deva else "script=latin"] = 1.0
    feats["all_title"] = 1.0 if all(t[:1].isupper() and t[1:].islower() for t in raw_tokens if t.isalpha()) else 0.0
    feats["all_upper"] = 1.0 if all(t.isupper() for t in raw_tokens if t.isalpha()) else 0.0
    feats["has_digit"] = 1.0 if any(c.isdigit() for c in span_text) else 0.0
    feats["has_apostrophe"] = 1.0 if "'" in span_text or "’" in span_text else 0.0

    norm_tokens = [_normalize_token(t) for t in raw_tokens]
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


def build_synthetic_dataset(seed: int = 42, n_samples: int = 6000) -> list[tuple[dict[str, float], int]]:
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        if rng.random() < 0.50:
            # Positive person name example (Latin, Devanagari, Hinglish, single/multi-token)
            r = rng.random()
            if r < 0.55:
                span = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                if rng.random() < 0.15:
                    span = span.upper()
            elif r < 0.80:
                span = f"{rng.choice(DEVA_FIRST)} {rng.choice(DEVA_LAST)}"
            elif r < 0.90:
                span = rng.choice(FIRST_NAMES)
            else:
                span = rng.choice(DEVA_FIRST)
            lc = rng.choice(POS_LEFT)
            rc = rng.choice(POS_RIGHT)
            data.append((extract_features(span, lc, rc), 1))
        else:
            # Negative non-person example (medical terms, doctor names, locations, UI labels)
            r = rng.random()
            if r < 0.45:
                span = rng.choice(MEDICAL_TERMS)
                lc = rng.choice(NEG_LEFT)
                rc = rng.choice(NEG_RIGHT)
            elif r < 0.70:
                span = rng.choice(NON_PERSON_CAPITALIZED)
                lc = rng.choice(NEG_LEFT)
                rc = rng.choice(NEG_RIGHT)
            else:
                # Doctor / location name that MUST be negative due to context!
                span = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                lc = rng.choice(["dr.", "dr", "doctor", "डॉ.", "consultant:", "referred by", "tehsil", "distt.", "near"])
                rc = rng.choice(["(md", "(ms", "reviewed", "disease", "syndrome", "palsy", "mandir", ""])
            data.append((extract_features(span, lc, rc), -1))
    rng.shuffle(data)
    return data


def train_averaged_perceptron(dataset: list[tuple[dict[str, float], int]], epochs: int = 8) -> dict[str, float]:
    weights: dict[str, float] = defaultdict(float)
    totals: dict[str, float] = defaultdict(float)
    step = 1
    for _ in range(epochs):
        for feats, label in dataset:
            score = sum(weights[k] * v for k, v in feats.items() if k in weights)
            pred = 1 if score >= 0.0 else -1
            if pred != label:
                for k, v in feats.items():
                    weights[k] += label * v
                    totals[k] += step * label * v
            step += 1

    # Average and prune near-zero weights to keep artifact tiny
    avg = {}
    for k, w in weights.items():
        val = w - totals[k] / max(step, 1)
        if abs(val) >= 0.05:
            avg[k] = round(val, 4)
    return avg


def main() -> None:
    dataset = build_synthetic_dataset(seed=2026, n_samples=10000)
    split = int(0.85 * len(dataset))
    train_set, val_set = dataset[:split], dataset[split:]
    weights = train_averaged_perceptron(train_set, epochs=10)

    correct = 0
    for feats, label in val_set:
        score = sum(weights.get(k, 0.0) * v for k, v in feats.items())
        if (1 if score >= 0.0 else -1) == label:
            correct += 1
    acc = correct / len(val_set)

    out_path = Path(__file__).resolve().parents[1] / "src" / "models" / "name_ner.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "averaged_perceptron_char_ngram_context_v1",
        "params": len(weights),
        "val_accuracy": round(acc, 4),
        "weights": weights,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Trained {len(weights)} weights, val_accuracy={acc:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
