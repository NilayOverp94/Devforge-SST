#!/usr/bin/env python3
"""Raksha Kavach Formal GUI Console — 100% offline local web server (Python stdlib only)."""
from __future__ import annotations

import argparse
import base64
import binascii
import csv
import hashlib
import http.server
import io
import json
import mimetypes
import os
import secrets
import socket
import socketserver
import subprocess
import sys
import tempfile
import urllib.parse
import webbrowser
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.detection import PIIDetector
from src.ingestion import parse_file
from src.linkage import extract_clues_from_file, infer_file_date_hints, link_all_records
from src.privacy import build_privacy_profiles, sanitize_file
from src.security import KeyRing

REPO_ROOT = Path(__file__).resolve().parent
GUI_DIR = REPO_ROOT / "gui"
POLICY_PATH = REPO_ROOT / "config" / "policy.yaml"
TOOLKIT_DIR = REPO_ROOT.parent / "Secret-Wards-ToolKit"
_AUTO_SESSION_KEY = os.environ.get("RAKSHA_SECRET") or secrets.token_hex(32)


def _load_policy_dict() -> dict[str, Any]:
    if not POLICY_PATH.is_file():
        return {}
    data = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _sanitize_policy_updates(raw: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    int_bounds = {
        "age_band_years": (1, 50),
        "rare_age_band_years": (5, 50),
        "high_risk_age_band_years": (5, 50),
        "elderly_cap_min_age": (60, 100),
        "pediatric_widen_max_age": (1, 18),
        "date_shift_min_days": (1, 730),
        "date_shift_max_days": (1, 1095),
        "min_group_size": (1, 50),
        "rare_condition_threshold": (1, 20),
        "pin_prefix_digits": (0, 6),
        "linkage_max_age_diff": (0, 10),
        "max_json_depth": (10, 1000),
        "pseudonym_length": (4, 12),
    }
    float_bounds = {
        "linkage_name_threshold": (0.5, 1.0),
        "linkage_relative_threshold": (0.5, 1.0),
        "linkage_village_threshold": (0.5, 1.0),
    }
    bool_keys = {"suppress_dob", "generalize_rare_district", "linkage_require_corroboration_for_name", "encrypt_intermediate_vault"}
    str_keys = {"default_state_label", "pseudonym_prefix"}

    for k, (lo, hi) in int_bounds.items():
        if k in raw:
            val = int(raw[k])
            out[k] = max(lo, min(hi, val))
    if out.get("date_shift_min_days", 1) > out.get("date_shift_max_days", 540):
        out["date_shift_min_days"] = out["date_shift_max_days"]
    for k, (lo, hi) in float_bounds.items():
        if k in raw:
            val = float(raw[k])
            out[k] = round(max(lo, min(hi, val)), 3)
    for k in bool_keys:
        if k in raw:
            out[k] = bool(raw[k])
    for k in str_keys:
        if k in raw and isinstance(raw[k], str) and raw[k].strip():
            out[k] = raw[k].strip()[:40]
    return out


def _safe_resolve_under(base_dir: Path, rel_path: str) -> Path | None:
    try:
        base_res = base_dir.resolve()
        cand = (base_res / rel_path).resolve()
        if cand.is_relative_to(base_res) and cand.is_file():
            return cand
    except Exception:
        pass
    return None


def _safe_rel_upload_path(rel_path: str) -> Path | None:
    """Validate browser-supplied relative upload path without allowing traversal or absolute paths."""
    cleaned = rel_path.replace("\\", "/").strip().lstrip("/")
    if not cleaned or "\x00" in cleaned:
        return None
    parts = [p for p in cleaned.split("/") if p]
    if not parts:
        return None
    for part in parts:
        if part in (".", "..") or ":" in part:
            return None
    return Path(*parts)


def _run_pipeline_subprocess(
    inp_dir: Path,
    out_dir: Path,
    secret: str,
    overrides: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run main.py offline in a subprocess with bounded policy and in-memory secret."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eff_policy = _sanitize_policy_updates(overrides or {}, _load_policy_dict())

    with tempfile.TemporaryDirectory(prefix="raksha-gui-pol-") as tmp:
        pol_file = Path(tmp) / "policy.yaml"
        pol_file.write_text(yaml.safe_dump(eff_policy, sort_keys=False), encoding="utf-8")

        env = dict(os.environ)
        env["RAKSHA_SECRET"] = secret or _AUTO_SESSION_KEY
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "main.py"),
                "--input", str(inp_dir),
                "--output", str(out_dir),
                "--policy", str(pol_file),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )

    if proc.returncode != 0:
        return 500, {
            "ok": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()[-1000:],
        }

    summary = _summarize_output(out_dir)
    summary["ok"] = True
    summary["cli_summary"] = proc.stdout.strip()
    return 200, summary


def _summarize_output(output_dir: Path) -> dict[str, Any]:
    report_path = output_dir / "report.json"
    records_path = output_dir / "records.jsonl"
    if not report_path.is_file():
        return {"exists": False}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unlinked: list[dict[str, Any]] = []
    total_records = 0

    if records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            total_records += 1
            psn = rec.get("patient")
            if psn:
                clusters[psn].append(rec)
            else:
                unlinked.append(rec)

    cluster_list = []
    for psn in sorted(clusters, key=lambda p: (-len(clusters[p]), p)):
        items = clusters[psn]
        files_touched = sorted({it["file"] for it in items})
        cluster_list.append({
            "patient": psn,
            "record_count": len(items),
            "file_count": len(files_touched),
            "files": files_touched[:12],
            "records": items[:50],
        })

    return {
        "exists": True,
        "output_dir": str(output_dir),
        "report": report,
        "cluster_count": len(cluster_list),
        "unlinked_count": len(unlinked),
        "total_records": total_records,
        "clusters": cluster_list,
        "unlinked_sample": unlinked[:50],
    }


def _run_sandbox_snippet(
    text: str,
    fmt: str,
    secret: str,
    policy_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not text.strip():
        return {"error": "Input snippet is empty."}

    policy = _sanitize_policy_updates(policy_overrides or {}, _load_policy_dict())
    ext_map = {"text": ".txt", "csv": ".csv", "tsv": ".tsv", "json": ".json"}
    ext = ext_map.get(fmt, ".txt")

    with tempfile.TemporaryDirectory(prefix="raksha-gui-sbx-") as tmp:
        tmp_dir = Path(tmp)
        fpath = tmp_dir / f"snippet{ext}"
        fpath.write_text(text, encoding="utf-8")

        inf = parse_file(
            fpath,
            tmp_dir,
            max_json_depth=int(policy.get("max_json_depth", 200)),
            max_file_bytes=int(policy.get("max_file_bytes", 104857600)),
        )
        if inf.quarantine_reason:
            return {
                "status": "quarantined",
                "reason": inf.quarantine_reason,
                "sha256": inf.sha256,
                "sanitized_text": "",
                "pii_spans": [],
                "protected_spans": [],
                "records": [],
                "entity_counts": {},
            }

        detector = PIIDetector()
        keyring = KeyRing(
            secret=(secret or "sandbox-preview").encode("utf-8"),
            pseudo_prefix=str(policy.get("pseudonym_prefix", "PSN")),
            pseudo_len=int(policy.get("pseudonym_length", 6)),
        )
        date_hints, modal_year = infer_file_date_hints([inf])
        hint = date_hints.get(inf.rel_path, "dmy")
        clues = extract_clues_from_file(inf, detector=detector, date_hint=hint, modal_year=modal_year)
        unit_to_psn, psn_clusters, ref_lookup = link_all_records(clues, keyring=keyring, policy=policy)
        _, unit_profiles = build_privacy_profiles(clues, unit_to_psn, psn_clusters, keyring=keyring, policy=policy)

        entity_counts: dict[str, int] = {}
        stats: dict[str, int] = {"records_generalized": 0}
        sanitized_text, records_out = sanitize_file(
            inf=inf,
            date_hint=hint,
            modal_year=modal_year,
            clues_list=clues,
            unit_to_psn=unit_to_psn,
            unit_profiles=unit_profiles,
            ref_lookup=ref_lookup,
            detector=detector,
            keyring=keyring,
            entity_counts=entity_counts,
            stats=stats,
        )

        # Collect highlighted spans for plain text or JSON text preview
        pii_spans = []
        prot_spans = []
        if fmt == "text":
            prot_spans = [{"start": s, "end": e, "text": text[s:e]} for s, e in detector.protected_spans(text)]
            rc = clues[0] if clues else None
            spans = detector.detect_text(
                text,
                extra_patient_names=rc.patient_names if rc else None,
                extra_relative_names=rc.relative_names if rc else None,
                allow_pin_scan=True,
            )
            pii_spans = [
                {"start": sp.start, "end": sp.end, "type": sp.pii_type, "role": sp.role, "text": sp.surface}
                for sp in spans
            ]

        return {
            "status": "sanitized",
            "format": inf.fmt,
            "sha256": inf.sha256,
            "sanitized_text": sanitized_text,
            "pii_spans": pii_spans,
            "protected_spans": prot_spans,
            "records": records_out,
            "entity_counts": entity_counts,
            "generalized_records": stats["records_generalized"],
        }


class RakshaHandler(http.server.BaseHTTPRequestHandler):
    server_version = "RakshaKavachGUI/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress HTTP logs carrying query strings or filenames to uphold strict PII-free logging
        pass

    def _send_json(self, payload: Any, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404, "Not found")
            return
        ctype, _ = mimetypes.guess_type(str(path))
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self, max_bytes: int = 5 * 1024 * 1024) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > max_bytes:
            raise ValueError(f"Payload exceeds maximum allowed size ({max_bytes} bytes).")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send_file(GUI_DIR / "index.html")
            return

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = _safe_resolve_under(GUI_DIR, rel)
            if target:
                self._send_file(target)
            else:
                self.send_error(404)
            return

        if path == "/api/health":
            self._send_json({
                "ok": True,
                "offline": True,
                "repo": str(REPO_ROOT),
                "capabilities": ["run", "run-upload", "download-zip", "sandbox", "verify"],
            })
            return

        if path == "/api/defaults":
            dev_input = TOOLKIT_DIR / "dev" / "input"
            out_default = REPO_ROOT / ".gui_output"
            self._send_json({
                "input_dir": str(dev_input) if dev_input.is_dir() else str(REPO_ROOT),
                "toolkit_input_dir": str(dev_input) if dev_input.is_dir() else "",
                "output_dir": str(out_default),
                "toolkit_present": TOOLKIT_DIR.is_dir(),
                "toolkit_dir": str(TOOLKIT_DIR),
                "policy": _load_policy_dict(),
                "has_existing_output": (out_default / "report.json").is_file(),
            })
            return

        if path == "/api/policy":
            self._send_json({"policy": _load_policy_dict()})
            return

        if path == "/api/results":
            out_dir = Path(qs.get("output_dir", [str(REPO_ROOT / ".gui_output")])[0]).expanduser().resolve()
            self._send_json(_summarize_output(out_dir))
            return

        if path == "/api/file":
            out_str = qs.get("output_dir", [""])[0].strip()
            rel = qs.get("rel", [""])[0].strip()
            if not out_str or not rel:
                self._send_json({"error": "Both output_dir and rel are required."}, status=400)
                return

            out_dir = Path(out_str).expanduser().resolve()
            inp_dir_str = qs.get("input_dir", [""])[0].strip()
            sha = qs.get("sha256", [""])[0].strip()
            reveal_raw = qs.get("include_raw", ["0"])[0] == "1"

            san_file = _safe_resolve_under(out_dir, rel)
            if san_file is not None:
                allowed = (
                    san_file.is_relative_to(out_dir / "sanitized")
                    or san_file == out_dir / "report.json"
                    or san_file == out_dir / "records.jsonl"
                )
                if not allowed:
                    san_file = None
            san_content = san_file.read_text(encoding="utf-8", errors="replace")[:200000] if san_file else None

            raw_content = None
            raw_rel = None
            if reveal_raw and inp_dir_str and sha:
                inp_dir = Path(inp_dir_str).expanduser().resolve()
                if inp_dir.is_dir():
                    for cand in sorted(p for p in inp_dir.rglob("*") if p.is_file()):
                        raw_bytes = cand.read_bytes()
                        if hashlib.sha256(raw_bytes).hexdigest() == sha:
                            raw_rel = cand.relative_to(inp_dir).as_posix()
                            if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
                                raw_content = raw_bytes.decode("utf-16", errors="replace")[:200000]
                            else:
                                raw_content = raw_bytes.decode("utf-8", errors="replace")[:200000]
                            break

            # Parse CSV preview rows when applicable
            csv_rows = None
            if san_file and san_file.suffix.lower() in (".csv", ".tsv") and san_content is not None:
                delim = "\t" if san_file.suffix.lower() == ".tsv" else ","
                try:
                    csv_rows = list(csv.reader(io.StringIO(san_content), delimiter=delim))[:100]
                except Exception:
                    csv_rows = None

            self._send_json({
                "sanitized_path": rel,
                "sanitized_content": san_content,
                "csv_rows": csv_rows,
                "raw_rel": raw_rel,
                "raw_content": raw_content,
            })
            return

        if path == "/api/download-zip":
            out_str = qs.get("output_dir", [""])[0].strip()
            if not out_str:
                self._send_json({"error": "output_dir is required."}, status=400)
                return
            out_dir = Path(out_str).expanduser().resolve()
            if not (out_dir / "report.json").is_file():
                self._send_json({"error": "No sanitized release found in output_dir."}, status=404)
                return

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                san_root = out_dir / "sanitized"
                if san_root.is_dir():
                    for fpath in sorted(p for p in san_root.rglob("*") if p.is_file()):
                        zf.write(fpath, arcname=fpath.relative_to(out_dir).as_posix())
                for meta_name in ("report.json", "records.jsonl"):
                    meta_path = out_dir / meta_name
                    if meta_path.is_file():
                        zf.write(meta_path, arcname=meta_name)

            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="raksha-sanitized-release.zip"')
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)
            return

        if path.startswith("/api/"):
            self._send_json({"error": f"Unknown API endpoint: {path}"}, status=404)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        max_body = 64 * 1024 * 1024 if path == "/api/run-upload" else 5 * 1024 * 1024
        try:
            body = self._read_json_body(max_bytes=max_body)
        except ValueError as exc:
            self._send_json({"error": f"Invalid JSON payload: {exc}"}, status=400)
            return

        if path == "/api/policy":
            updated = _sanitize_policy_updates(body.get("policy") or body, _load_policy_dict())
            POLICY_PATH.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
            self._send_json({"ok": True, "policy": updated})
            return

        if path == "/api/run":
            inp_str = str(body.get("input_dir", "")).strip()
            out_str = str(body.get("output_dir", "")).strip()
            secret = str(body.get("secret", ""))
            overrides = body.get("policy_overrides") or {}
            if not inp_str or not out_str:
                self._send_json({"error": "Both input_dir and output_dir are required."}, status=400)
                return

            inp_dir = Path(inp_str).expanduser().resolve()
            out_dir = Path(out_str).expanduser().resolve()
            if not inp_dir.is_dir():
                self._send_json({"error": f"Input directory does not exist: {inp_dir}"}, status=400)
                return
            if out_dir == inp_dir or out_dir.is_relative_to(inp_dir):
                self._send_json({"error": "Output directory must be separate from read-only input directory."}, status=400)
                return

            status, payload = _run_pipeline_subprocess(inp_dir, out_dir, secret, overrides)
            self._send_json(payload, status=status)
            return

        if path == "/api/run-upload":
            files = body.get("files")
            out_str = str(body.get("output_dir", "")).strip()
            secret = str(body.get("secret", ""))
            overrides = body.get("policy_overrides") or {}
            if not isinstance(files, list) or not files:
                self._send_json({"error": "Select at least one file or folder to process."}, status=400)
                return
            if len(files) > 1000:
                self._send_json({"error": "Upload batch exceeds 1,000 files limit."}, status=400)
                return
            if not out_str:
                self._send_json({"error": "Output Release Directory is required."}, status=400)
                return

            out_dir = Path(out_str).expanduser().resolve()
            with tempfile.TemporaryDirectory(prefix="raksha-gui-up-") as tmp_in_str:
                tmp_in = Path(tmp_in_str).resolve()
                if out_dir == tmp_in or out_dir.is_relative_to(tmp_in):
                    self._send_json({"error": "Invalid output directory."}, status=400)
                    return

                total_decoded = 0
                for item in files:
                    if not isinstance(item, dict):
                        self._send_json({"error": "Invalid file entry in upload payload."}, status=400)
                        return
                    rel_raw = str(item.get("rel_path", ""))
                    b64_raw = str(item.get("content_b64", ""))
                    safe_rel = _safe_rel_upload_path(rel_raw)
                    if safe_rel is None:
                        self._send_json({"error": f"Unsafe or invalid relative file path: {rel_raw!r}"}, status=400)
                        return
                    try:
                        raw_bytes = base64.b64decode(b64_raw, validate=True)
                    except (binascii.Error, ValueError):
                        self._send_json({"error": f"Corrupt base64 upload payload for {safe_rel.as_posix()}"}, status=400)
                        return

                    total_decoded += len(raw_bytes)
                    if total_decoded > 50 * 1024 * 1024:
                        self._send_json({"error": "Total selected files exceed 50 MB upload limit."}, status=400)
                        return

                    dest = (tmp_in / safe_rel).resolve()
                    if not dest.is_relative_to(tmp_in):
                        self._send_json({"error": "Path escapes staging boundary."}, status=400)
                        return
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(raw_bytes)

                status, payload = _run_pipeline_subprocess(tmp_in, out_dir, secret, overrides)
            self._send_json(payload, status=status)
            return

        if path == "/api/sandbox":
            text = str(body.get("text", ""))
            fmt = str(body.get("format", "text"))
            secret = str(body.get("secret", "sandbox-demo"))
            overrides = body.get("policy_overrides") or {}
            result = _run_sandbox_snippet(text=text, fmt=fmt, secret=secret, policy_overrides=overrides)
            self._send_json(result)
            return

        if path == "/api/verify":
            kind = str(body.get("kind", "validate"))
            inp_str = str(body.get("input_dir", "")).strip()
            out_str = str(body.get("output_dir", "")).strip()
            cmd: list[str]
            cwd = REPO_ROOT

            if kind == "validate":
                script = TOOLKIT_DIR / "validate_output.py"
                if not script.is_file():
                    self._send_json({"error": "Secret-Wards-ToolKit/validate_output.py not found."}, status=404)
                    return
                if not inp_str or not Path(inp_str).expanduser().is_dir() or not out_str:
                    self._send_json({
                        "error": "Switch Tab 1 to 'Local Folder Path' with an existing Input Directory to run validate_output.py.",
                    }, status=400)
                    return
                cmd = [sys.executable, str(script), "--input", inp_str, "--output", out_str]
            elif kind == "check_dev":
                script = TOOLKIT_DIR / "check_dev.py"
                if not script.is_file():
                    self._send_json({"error": "Secret-Wards-ToolKit/check_dev.py not found."}, status=404)
                    return
                if not out_str or not (Path(out_str).expanduser() / "report.json").is_file():
                    self._send_json({"error": "Run the firewall first so report.json exists in output_dir."}, status=400)
                    return
                cmd = [sys.executable, str(script), "--output", out_str, "--show", "15"]
            elif kind == "chaos":
                cmd = [sys.executable, str(REPO_ROOT / "tests" / "test_chaos.py")]
            elif kind == "eval_attacks":
                script = REPO_ROOT / "tests" / "eval_attacks.py"
                truth = TOOLKIT_DIR / "dev" / "truth" / "truth.json"
                reg = TOOLKIT_DIR / "dev" / "attacker" / "registry.csv"
                kf = TOOLKIT_DIR / "dev" / "attacker" / "known_facts.jsonl"
                raw_atk = TOOLKIT_DIR / "dev" / "truth" / "raw_attack.json"
                if not all(p.is_file() for p in (script, truth, reg, kf)):
                    self._send_json({"error": "Dev truth/attacker files not found alongside repo."}, status=404)
                    return
                cmd = [
                    sys.executable, str(script),
                    "--output", out_str, "--truth", str(truth),
                    "--registry", str(reg), "--known-facts", str(kf),
                    "--raw-attack", str(raw_atk),
                ]
            elif kind == "check_submission":
                script = TOOLKIT_DIR / "check_submission.py"
                if not script.is_file():
                    self._send_json({"error": "Secret-Wards-ToolKit/check_submission.py not found."}, status=404)
                    return
                cmd = [sys.executable, str(script), "--repo", str(REPO_ROOT), "--run"]
            else:
                self._send_json({"error": f"Unknown verification kind: {kind}"}, status=400)
                return

            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
            self._send_json({
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "output": (proc.stdout + ("\n" + proc.stderr if proc.stderr else "")).strip(),
            })
            return

        if path.startswith("/api/"):
            self._send_json({"error": f"Unknown API endpoint: {path}"}, status=404)
            return
        self.send_error(404)


class ReusableThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def pick_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def main() -> int:
    ap = argparse.ArgumentParser(description="Raksha Kavach Local Offline GUI Console")
    ap.add_argument("--port", type=int, default=8573, help="Localhost port (default: 8573)")
    ap.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = ap.parse_args()

    port = pick_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    srv = ReusableThreadingServer(("127.0.0.1", port), RakshaHandler)
    print(f"Raksha Kavach Console running offline at {url} (Press Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Raksha Kavach Console.")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
