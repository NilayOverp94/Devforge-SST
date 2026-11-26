#!/usr/bin/env python3
"""Automated unit & integration tests for the offline Raksha Kavach GUI server."""
from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gui  # noqa: E402


class TestRakshaGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = gui.ReusableThreadingServer(("127.0.0.1", 0), gui.RakshaHandler)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _get(self, path: str) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(self.base_url + path, timeout=30) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, exc.read().decode("utf-8", errors="replace")
            finally:
                exc.close()

    def _post_json(self, path: str, payload: dict) -> tuple[int, dict]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read().decode("utf-8"))
            finally:
                exc.close()

    def test_static_assets_and_offline_guarantees(self) -> None:
        st_html, html = self._get("/")
        self.assertEqual(st_html, 200)
        self.assertIn("Raksha Kavach", html)
        for forbidden in ("https://", "http://", "fonts.googleapis", "cdn.", "jsdelivr", "unpkg"):
            self.assertNotIn(forbidden, html)

        st_css, css = self._get("/static/styles.css")
        self.assertEqual(st_css, 200)
        self.assertIn("--navy", css)

        st_js, js = self._get("/static/app.js")
        self.assertEqual(st_js, 200)
        self.assertIn("bootstrap", js)

        # Static path traversal must be rejected
        st_trav, _ = self._get("/static/../main.py")
        self.assertEqual(st_trav, 404)

    def test_health_defaults_and_policy_roundtrip(self) -> None:
        st, raw_health = self._get("/api/health")
        self.assertEqual(st, 200)
        health = json.loads(raw_health)
        self.assertTrue(health["ok"])
        self.assertTrue(health["offline"])
        self.assertIn("run-upload", health.get("capabilities", []))

        st_unk, unk_body = self._get("/api/no-such-route")
        self.assertEqual(st_unk, 404)
        self.assertIn("Unknown API endpoint", json.loads(unk_body)["error"])

        st_unk_post, unk_post_body = self._post_json("/api/no-such-route", {})
        self.assertEqual(st_unk_post, 404)
        self.assertIn("Unknown API endpoint", unk_post_body["error"])

        st_def, raw_def = self._get("/api/defaults")
        self.assertEqual(st_def, 200)
        defs = json.loads(raw_def)
        self.assertIsInstance(defs["policy"], dict)

        orig_policy_bytes = gui.POLICY_PATH.read_bytes()
        try:
            st_pol, pol_res = self._post_json("/api/policy", {"policy": defs["policy"]})
            self.assertEqual(st_pol, 200)
            self.assertTrue(pol_res["ok"])
        finally:
            gui.POLICY_PATH.write_bytes(orig_policy_bytes)

    def test_sandbox_hindi_meaning_trap_and_quarantine(self) -> None:
        canary = "ULTRA_SECRET_CANARY_987654"
        hindi_text = (
            "मरीज़ का नाम: रीना निषाद  आयु: 65 वर्ष / महिला\n"
            "मोबाइल: ७५६००१००३४\n"
            "Medications: Cap Doxycycline 100 mg 1-0-1 x 15 days\n"
            "Review with Dr. Ramesh Srivastava."
        )
        st, res = self._post_json("/api/sandbox", {"text": hindi_text, "format": "text", "secret": canary})
        self.assertEqual(st, 200)
        self.assertEqual(res["status"], "sanitized")
        self.assertGreater(len(res["pii_spans"]), 0)
        self.assertGreater(len(res["protected_spans"]), 0)
        self.assertIn("Doxycycline 100 mg", res["sanitized_text"])
        self.assertIn("Dr. Ramesh Srivastava", res["sanitized_text"])
        self.assertNotIn("रीना निषाद", res["sanitized_text"])
        self.assertNotIn(canary, json.dumps(res))

        meaning_text = (
            "Patient Name: Manoj Yadav Age/Sex: 65 Yrs/M\n"
            "C/O Ramesh Yadav, Ward 8, Sitapur - 261699\n"
            "C/O: fever for 4 days. K/C/O Parkinson's disease.\n"
            "Dr. Ramesh Srivastava reviewed."
        )
        st2, res2 = self._post_json("/api/sandbox", {"text": meaning_text, "format": "text", "secret": canary})
        self.assertEqual(st2, 200)
        self.assertEqual(res2["status"], "sanitized")
        self.assertIn("Parkinson's disease", res2["sanitized_text"])
        self.assertIn("fever for 4 days", res2["sanitized_text"])
        self.assertIn("Dr. Ramesh Srivastava", res2["sanitized_text"])
        self.assertNotIn("Manoj Yadav", res2["sanitized_text"])

        st3, res3 = self._post_json("/api/sandbox", {"text": "{broken json", "format": "json", "secret": canary})
        self.assertEqual(st3, 200)
        self.assertEqual(res3["status"], "quarantined")
        self.assertTrue(res3["reason"])

    def test_batch_run_inspector_containment_and_verify(self) -> None:
        dev_in = gui.TOOLKIT_DIR / "dev" / "input"
        if not dev_in.is_dir():
            self.skipTest("Secret-Wards-ToolKit/dev/input not present")

        canary = "GUI_BATCH_CANARY_KEY_42"
        with tempfile.TemporaryDirectory(prefix="raksha-gui-test-") as tmp:
            out_dir = Path(tmp) / "out"

            # Reject output directory overlapping input directory
            st_bad, _ = self._post_json(
                "/api/run", {"input_dir": str(dev_in), "output_dir": str(dev_in), "secret": canary}
            )
            self.assertEqual(st_bad, 400)

            st_run, run_res = self._post_json(
                "/api/run", {"input_dir": str(dev_in), "output_dir": str(out_dir), "secret": canary}
            )
            self.assertEqual(st_run, 200)
            self.assertTrue(run_res["ok"])
            self.assertNotIn(canary, json.dumps(run_res))
            self.assertGreater(run_res["cluster_count"], 0)
            self.assertGreater(run_res["report"]["counts"]["files_sanitized"], 0)

            st_res, raw_loaded = self._get(
                f"/api/results?output_dir={urllib.parse.quote(str(out_dir))}"
            )
            self.assertEqual(st_res, 200)
            loaded = json.loads(raw_loaded)
            self.assertTrue(loaded["exists"])
            self.assertEqual(loaded["cluster_count"], run_res["cluster_count"])

            san_entry = next(f for f in loaded["report"]["files"] if f["status"] == "sanitized")
            qs = urllib.parse.urlencode({
                "output_dir": str(out_dir),
                "rel": san_entry["output"],
                "input_dir": str(dev_in),
                "sha256": san_entry["input_sha256"],
                "include_raw": "1",
            })
            st_f, raw_f = self._get(f"/api/file?{qs}")
            self.assertEqual(st_f, 200)
            f_data = json.loads(raw_f)
            self.assertIsNotNone(f_data["sanitized_content"])
            self.assertIsNotNone(f_data["raw_content"])

            # Path traversal & arbitrary file read defenses
            st_missing, _ = self._get("/api/file?rel=README.md")
            self.assertEqual(st_missing, 400)

            qs_trav = urllib.parse.urlencode({"output_dir": str(out_dir), "rel": "../main.py"})
            st_t, raw_t = self._get(f"/api/file?{qs_trav}")
            self.assertEqual(st_t, 200)
            self.assertIsNone(json.loads(raw_t)["sanitized_content"])

            qs_nonout = urllib.parse.urlencode({"output_dir": str(REPO_ROOT), "rel": "README.md"})
            st_no, raw_no = self._get(f"/api/file?{qs_nonout}")
            self.assertEqual(st_no, 200)
            self.assertIsNone(json.loads(raw_no)["sanitized_content"])

            for kind in ("validate", "check_dev", "chaos", "eval_attacks"):
                st_v, v_res = self._post_json(
                    "/api/verify",
                    {"kind": kind, "input_dir": str(dev_in), "output_dir": str(out_dir)},
                )
                self.assertEqual(st_v, 200)
                self.assertTrue(v_res["ok"], f"{kind} failed: {v_res}")

    def test_upload_files_folder_quarantine_and_zip_download(self) -> None:
        canary = "UPLOAD_CANARY_KEY_555"
        hindi_note = (
            "मरीज़ का नाम: रीना निषाद  आयु: 65 वर्ष / महिला\n"
            "मोबाइल: ७५६००१००३४\n"
            "Medications: Cap Doxycycline 100 mg 1-0-1 x 15 days\n"
            "Review with Dr. Ramesh Srivastava.\n"
        ).encode("utf-8")
        csv_bytes = (
            "patient_name,age,mobile,diagnosis\n"
            "Reena Nishad,65,7560010034,Urinary tract infection\n"
        ).encode("utf-8")
        bad_json = b"{broken: json"

        with tempfile.TemporaryDirectory(prefix="raksha-gui-up-test-") as tmp:
            out_dir = Path(tmp) / "upload_out"

            # Reject path traversal, Windows drive paths, and empty uploads
            st_empty, _ = self._post_json(
                "/api/run-upload", {"files": [], "output_dir": str(out_dir), "secret": canary}
            )
            self.assertEqual(st_empty, 400)

            for bad_rel in ("../../evil.txt", "C:/evil.txt", "."):
                st_bad, _ = self._post_json(
                    "/api/run-upload",
                    {
                        "files": [{"rel_path": bad_rel, "content_b64": base64.b64encode(b"hi").decode()}],
                        "output_dir": str(out_dir),
                        "secret": canary,
                    },
                )
                self.assertEqual(st_bad, 400)

            st_up, up_res = self._post_json(
                "/api/run-upload",
                {
                    "files": [
                        {"rel_path": "notes/hindi_note.txt", "content_b64": base64.b64encode(hindi_note).decode()},
                        {"rel_path": "opd/visits.csv", "content_b64": base64.b64encode(csv_bytes).decode()},
                        {"rel_path": "broken/bad.json", "content_b64": base64.b64encode(bad_json).decode()},
                    ],
                    "output_dir": str(out_dir),
                    "secret": canary,
                },
            )
            self.assertEqual(st_up, 200)
            self.assertTrue(up_res["ok"])
            self.assertEqual(up_res["report"]["counts"]["files_total"], 3)
            self.assertEqual(up_res["report"]["counts"]["files_sanitized"], 2)
            self.assertEqual(up_res["report"]["counts"]["files_quarantined"], 1)
            self.assertNotIn(canary, json.dumps(up_res))

            # Verify validate gives clean 400 if no disk input_dir is provided
            st_val_err, _ = self._post_json(
                "/api/verify", {"kind": "validate", "input_dir": "", "output_dir": str(out_dir)}
            )
            self.assertEqual(st_val_err, 400)

            # Verify ZIP download endpoint bundles sanitized/, report.json, records.jsonl cleanly
            zip_url = f"{self.base_url}/api/download-zip?output_dir={urllib.parse.quote(str(out_dir))}"
            with urllib.request.urlopen(zip_url, timeout=30) as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("application/zip", resp.headers.get("Content-Type", ""))
                zbytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
                names = set(zf.namelist())
                self.assertIn("report.json", names)
                self.assertIn("records.jsonl", names)
                self.assertTrue(any(n.startswith("sanitized/") and n.endswith(".txt") for n in names))
                self.assertTrue(any(n.startswith("sanitized/") and n.endswith(".csv") for n in names))
                self.assertFalse(any("bad.json" in n for n in names))


if __name__ == "__main__":
    unittest.main(verbosity=2)
