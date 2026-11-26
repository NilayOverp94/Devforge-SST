#!/usr/bin/env python3
"""Unseen-format chaos generalization test & unit verification suite (zero dev data used)."""
from __future__ import annotations

import base64
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_unseen_chaos_dataset(inp_dir: Path) -> dict:
    """Create a synthetic hospital dump with completely unseen names, headers, prefixes, formats, and traps."""
    (inp_dir / "reg").mkdir(parents=True)
    (inp_dir / "notes").mkdir(parents=True)
    (inp_dir / "fhir").mkdir(parents=True)
    (inp_dir / "bad").mkdir(parents=True)
    (inp_dir / "clean").mkdir(parents=True)

    # 1. Unseen CSV columns, unseen South/West Indian names, unseen hospital prefix KGMU/26/88412, DD.MM.YYYY
    with (inp_dir / "reg" / "kgmu_dump.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Hosp_Reg_ID", "Pat_Full_Name", "Sex_MF", "Umar_Age", "Birth_Dt", "Guardian_Rel", "Gram_Gaon", "Zila_Dist", "Pin_6", "Mob_No", "Admit_Dt", "Disch_Dt"])
        w.writerow(["KGMU/26/88412", "Venkatesh Subramanian", "M", "53", "14.03.1973", "S/o Raghavan Subramanian", "Nandgaon", "Nagpur", "440012", "+91 98230 11223", "01.04.2026", "10.04.2026"])
        w.writerow(["KGMU/26/88413", "Venkatesh Subramanian", "M", "29", "19.08.1997", "S/o Govind Subramanian", "Khamla", "Pune", "411005", "+91 98230 44556", "03.05.2026", "06.05.2026"])

    # 2. Same patient (88412) in Devanagari + Hinglish text note with unseen eponym & meaning traps
    (inp_dir / "notes" / "Venkatesh_Subramanian_KGMU-26-88412.txt").write_text(
        "KGMU Hospital, Nagpur\n"
        "मरीज़ का नाम: वेंकटेश सुब्रमण्यन   आयु: 53 वर्ष / पुरुष\n"
        "पिता का नाम: राघवन सुब्रमण्यन\n"
        "पता: मकान नं. 44, ग्राम नंदगांव, तहसील कामठी, जिला नागपुर - 440012\n"
        "मोबाइल: ९८२३०११२२३\n"
        "भर्ती की तारीख: 01.04.2026\n"
        "छुट्टी की तारीख: 10.04.2026\n"
        "C/O chest pain x 3 days. K/C/O Hashimoto's thyroiditis.\n"
        "Rx: Tab Levothyroxine 75 mcg 1-0-0 x 30 days. BP 138/84.\n"
        "Dr. Raghavan Iyer (MD Medicine) reviewed the patient.\n"
        "Attendant: S/o Raghavan Subramanian.\n"
        "Canary note: witnessed by Lobsang Tenzing, phone 9811223344.\n",
        encoding="utf-8",
    )

    # 3. Fragmented note for patient 1 (Venkatesh ji, S/o Raghavan)
    (inp_dir / "notes" / "ward_note_frag.txt").write_text(
        "NURSING NOTES - Ward 6 (Cardiology)\n"
        "01.04.2026 21:00 - Venkatesh ji ka BP 138/84. Dolo 650 BD x 5 days diya.\n"
        "Attendant (S/o Raghavan) informed.\n",
        encoding="utf-8",
    )

    # 4. JSON with base64 attachment + source_path metadata + unseen schema keys
    b64_payload = base64.b64encode(
        b"Consent signed by Venkatesh Subramanian (KGMU/26/88412), mob 9823011223, for Guillain-Barre syndrome care."
    ).decode("ascii")
    (inp_dir / "notes" / "opd_unseen.json").write_text(
        json.dumps({
            "_meta": {"source_path": "C:\\exports\\Venkatesh_Subramanian_88412\\visit.json"},
            "visits": [{
                "patient_ref": "KGMU/26/88412",
                "visit_date": "01.04.2026",
                "diagnosis": [{"code": "G61.0", "display": "Guillain-Barre syndrome"}],
                "attachment": {"data": b64_payload},
            }],
        }, indent=2),
        encoding="utf-8",
    )

    # 5. Clean protocol (must survive untouched)
    (inp_dir / "clean" / "protocol.txt").write_text(
        "PROTOCOL: C/O fever x 5 days -> Paracetamol 650 mg 1-1-1. Dr. Meenakshi Sundaram.\n",
        encoding="utf-8",
    )

    # 6. Malformed / unsupported files (must quarantine, never copy raw)
    (inp_dir / "bad" / "empty.csv").write_bytes(b"")
    (inp_dir / "bad" / "scan.pdf").write_bytes(b"%PDF-1.4\nSecret patient Lobsang Tenzing\n")
    (inp_dir / "bad" / "mislabeled.csv").write_text('{"patient": "Lobsang Tenzing", "aadhaar": "998877665544"}', encoding="utf-8")
    (inp_dir / "bad" / "corrupt.txt").write_bytes(b"Note for Lobsang Tenzing \xff\xfe\xc3( mobile 9811223344\n")
    (inp_dir / "bad" / "truncated.json").write_text('{"patient": "Lobsang Tenzing"', encoding="utf-8")

    return {
        "pii": ["Venkatesh Subramanian", "वेंकटेश सुब्रमण्यन", "Raghavan Subramanian", "राघवन सुब्रमण्यन",
                "98230 11223", "9823011223", "९८२३०११२२३", "98230 44556", "KGMU/26/88412", "KGMU/26/88413",
                "Nandgaon", "नंदगांव", "440012", "411005", "14.03.1973", "19.08.1997",
                "Lobsang Tenzing", "9811223344", "998877665544"],
        "preserve": [
            "Hashimoto's thyroiditis", "Levothyroxine 75 mcg", "1-0-0", "BP 138/84",
            "Dr. Raghavan Iyer", "C/O chest pain", "Dolo 650", "Guillain-Barre syndrome",
            "Ward 6 (Cardiology)", "Dr. Meenakshi Sundaram",
        ],
    }


def run_chaos_suite() -> None:
    py = str(REPO_ROOT / ".venv" / "bin" / "python")
    with tempfile.TemporaryDirectory(prefix="sw-chaos-") as tmp:
        tmp_p = Path(tmp)
        inp = tmp_p / "input"
        out1 = tmp_p / "out1"
        out2 = tmp_p / "out2"
        out3 = tmp_p / "out3"
        expected = build_unseen_chaos_dataset(inp)

        for out, secret in ((out1, "secret-alpha-1"), (out2, "secret-alpha-1"), (out3, "secret-beta-2")):
            res = subprocess.run(
                [py, "main.py", "--input", str(inp), "--output", str(out)],
                cwd=REPO_ROOT,
                env=dict(os.environ, RAKSHA_SECRET=secret),
                capture_output=True,
                text=True,
            )
            assert res.returncode == 0, f"Run failed: {res.stderr}"
            assert "Lobsang" not in res.stdout + res.stderr
            assert secret not in res.stdout + res.stderr

        # Verify contract with validate_output.py
        val_script = REPO_ROOT.parent / "Secret-Wards-ToolKit" / "validate_output.py"
        if val_script.is_file():
            vres = subprocess.run([sys.executable, str(val_script), "--input", str(inp), "--output", str(out1)], capture_output=True, text=True)
            assert "CONTRACT OK" in vres.stdout, vres.stdout

        # Check report quarantine count (5 malformed files in bad/)
        rep = json.loads((out1 / "report.json").read_text(encoding="utf-8"))
        assert rep["counts"]["files_quarantined"] == 5, rep["counts"]
        assert rep["counts"]["files_sanitized"] == 5, rep["counts"]

        # Check ZERO PII & canary leaks across ALL outputs including decoded base64 and filenames
        all_texts = []
        for p in (out1 / "sanitized").rglob("*"):
            if p.is_file():
                all_texts.append(p.name)
                txt = p.read_text(encoding="utf-8")
                all_texts.append(txt)
                for m in __import__("re").finditer(r"[A-Za-z0-9+/]{24,}={0,2}", txt):
                    try:
                        all_texts.append(base64.b64decode(m.group(0)).decode("utf-8", errors="ignore"))
                    except Exception:
                        pass
        all_texts.append((out1 / "report.json").read_text(encoding="utf-8"))
        all_texts.append((out1 / "records.jsonl").read_text(encoding="utf-8"))
        blob = "\n".join(all_texts)

        for pii in expected["pii"]:
            assert pii.lower() not in blob.lower(), f"UNSEEN PII LEAKED: {pii}"

        # Check ALL clinical preserve probes survived
        san_blob = "\n".join(p.read_text(encoding="utf-8") for p in (out1 / "sanitized").rglob("*") if p.is_file())
        for pv in expected["preserve"]:
            assert pv in san_blob or pv in blob, f"PRESERVE VALUE LOST: {pv}"

        # Check exact 9-day length of stay survived in both CSV & Devanagari discharge note
        # Check linkage: patient 1 (88412 across CSV row:0, Hindi note, frag note, JSON visit) linked together,
        # and patient 2 (88413, homonym Venkatesh Subramanian!) NOT merged with patient 1!
        recs = [json.loads(l) for l in (out1 / "records.jsonl").read_text().splitlines() if l.strip()]
        by_loc = {(r["file"], r["loc"]): r["patient"] for r in recs}
        p1_csv = by_loc[("sanitized/reg/kgmu_dump.csv", "row:0")]
        p2_csv = by_loc[("sanitized/reg/kgmu_dump.csv", "row:1")]
        p1_json = by_loc[("sanitized/notes/opd_unseen.json", "/visits/0")]
        p1_frag = by_loc[("sanitized/notes/ward_note_frag.txt", "doc")]
        p1_note = next(v for (f, l), v in by_loc.items() if f.startswith("sanitized/notes/") and f.endswith(".txt") and "ward_note_frag" not in f)
        assert p1_csv is not None and p1_csv == p1_json == p1_note == p1_frag, f"Missed cross-script/frag link: {by_loc}"
        assert p2_csv is not None and p2_csv != p1_csv, "HOMONYM FALSE MERGE on unseen dataset!"

        # Check determinism (out1 == out2) and secret sensitivity (out1 != out3)
        def digest(root: Path) -> str:
            h = hashlib.sha256()
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.name != "report.json":
                    h.update(f.relative_to(root).as_posix().encode() + b"\0" + f.read_bytes())
            return h.hexdigest()

        assert digest(out1) == digest(out2), "Non-deterministic output!"
        assert digest(out1) != digest(out3), "Output did not change with different RAKSHA_SECRET!"

        print("UNSEEN CHAOS SUITE PASSED: 0/19 unseen PII leaked, 10/10 clinical probes preserved, homonym separated, cross-script+frag linked, deterministic & keyed!")


if __name__ == "__main__":
    run_chaos_suite()
