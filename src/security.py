"""Cryptographic primitives, encrypted at-rest vault, PII-free logging, and safe naming."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import sys
from pathlib import Path
from typing import Callable

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

try:
    import psutil
except ImportError:  # pragma: no cover - optional fallback
    psutil = None


_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


class KeyRing:
    """Derives domain-separated subkeys from RAKSHA_SECRET without ever exposing it."""

    def __init__(self, secret: bytes | None = None, pseudo_prefix: str = "PSN", pseudo_len: int = 6) -> None:
        raw_secret = os.environ.get("RAKSHA_SECRET", "").encode("utf-8") if secret is None else secret
        if not raw_secret:
            raw_secret = b"raksha-default-ephemeral-key"
        self._pseudo_prefix = pseudo_prefix
        self._pseudo_len = pseudo_len
        self._pseudo_key = self._derive(raw_secret, b"raksha-pseudonym-v1")
        self._shift_key = self._derive(raw_secret, b"raksha-date-shift-v1")
        self._tag_key = self._derive(raw_secret, b"raksha-safe-tag-v1")
        self._vault_key = self._derive(raw_secret, b"raksha-vault-v1")
        self._assigned: dict[str, str] = {}
        self._reverse: dict[str, str] = {}

    @staticmethod
    def _derive(master: bytes, info: bytes) -> bytes:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"secret-wards-hkdf-salt-v1",
            info=info,
        )
        return hkdf.derive(master)

    def pseudonym(self, canonical_key: str) -> str:
        """Issue deterministic, non-reversible HMAC-SHA256 Crockford pseudonym."""
        if canonical_key in self._assigned:
            return self._assigned[canonical_key]
        salt_idx = 0
        while True:
            msg = canonical_key if salt_idx == 0 else f"{canonical_key}|{salt_idx}"
            digest = hmac.new(self._pseudo_key, msg.encode("utf-8"), hashlib.sha256).digest()
            n = int.from_bytes(digest[:6], "big")
            code = "".join(_CROCKFORD[(n >> (5 * i)) & 31] for i in range(self._pseudo_len))
            candidate = f"{self._pseudo_prefix}-{code}"
            owner = self._reverse.get(candidate)
            if owner is None or owner == canonical_key:
                self._assigned[canonical_key] = candidate
                self._reverse[candidate] = canonical_key
                return candidate
            salt_idx += 1

    def date_shift_days(self, entity_key: str, min_days: int = 180, max_days: int = 540) -> int:
        """Deterministic non-zero per-patient day offset in [-max_days, -min_days] U [min_days, max_days]."""
        if max_days < 1:
            return 1
        min_d = max(1, min(min_days, max_days))
        width = max(1, max_days - min_d + 1)
        digest = hmac.new(self._shift_key, entity_key.encode("utf-8"), hashlib.sha256).digest()
        sign = -1 if (digest[0] & 1) else 1
        mag = min_d + (int.from_bytes(digest[1:5], "big") % width)
        return sign * mag

    def safe_tag(self, value: str, length: int = 8) -> str:
        """Short non-reversible HMAC hex tag for anonymized filenames/headers."""
        digest = hmac.new(self._tag_key, value.encode("utf-8"), hashlib.sha256).hexdigest()
        return digest[:length]

    def vault_key(self) -> bytes:
        return self._vault_key


class EncryptedVault:
    """In-memory AES-256-GCM encrypted vault for buffering raw/intermediate payloads."""

    def __init__(self, key: bytes) -> None:
        self._aesgcm = AESGCM(key)
        self._counter = 0
        self._store: dict[str, tuple[bytes, bytes]] = {}

    def seal(self, data: bytes) -> str:
        self._counter += 1
        token = f"v_{self._counter}"
        nonce = hashlib.sha256(f"{token}".encode("utf-8")).digest()[:12]
        ct = self._aesgcm.encrypt(nonce, data, token.encode("utf-8"))
        self._store[token] = (nonce, ct)
        return token

    def unseal(self, token: str) -> bytes:
        nonce, ct = self._store[token]
        return self._aesgcm.decrypt(nonce, ct, token.encode("utf-8"))

    def purge(self) -> None:
        self._store.clear()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def peak_rss_mb() -> float:
    rss_vals = []
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_vals.append(r / 1048576.0 if sys.platform == "darwin" else r / 1024.0)
    except Exception:
        pass
    if psutil is not None:
        try:
            rss_vals.append(psutil.Process().memory_info().rss / 1048576.0)
        except Exception:
            pass
    return round(max(rss_vals) if rss_vals else 0.0, 2)


def safe_reason(exc: Exception) -> str:
    """Map an exception to a strictly PII-free short reason code."""
    if isinstance(exc, ValueError):
        msg = str(exc)
        if msg and re.fullmatch(r"[a-z0-9_]{2,48}", msg):
            return msg
        return "validation_error"
    return re.sub(r"[^a-zA-Z0-9_]", "_", type(exc).__name__)[:48]


def sanitize_filename(
    rel_path: str,
    file_sha256: str,
    keyring: KeyRing,
    has_pii_fn: Callable[[str], bool],
    pseudonym: str | None = None,
) -> tuple[str, bool]:
    """Detect and safely rename any filename/path segment carrying PII."""
    parts = Path(rel_path).parts
    if not parts:
        return rel_path, False

    new_parts = []
    changed = False
    for idx, part in enumerate(parts):
        is_file = idx == len(parts) - 1
        if not is_file:
            if has_pii_fn(part.replace("_", " ").replace("-", " ")):
                new_parts.append(f"dir_{keyring.safe_tag(part, 8)}")
                changed = True
            else:
                new_parts.append(part)
            continue

        stem = Path(part).stem
        ext = Path(part).suffix
        probe = stem.replace("_", " ").replace("-", " ")
        # Detect raw digit sequences >= 6 (ignore hex bundle UUIDs like bundle_7dfbd411)
        is_hex_bundle = bool(re.fullmatch(r"(?i)bundle[_-][0-9a-f]{6,}", stem))
        has_digits_id = (not is_hex_bundle) and bool(re.search(r"(?<![0-9a-fA-F])\d{6,}(?![0-9a-fA-F])", stem))
        has_mrn_pat = bool(
            re.search(r"(?i)(?:^|[_-])(?:uhid|mrn|ipd|adm|ip|snj|jjh|dhh)[_-]?\d{2,}", stem)
            or re.search(r"(?:^|[_-])[A-Z]{2,6}[-_](?:20\d{2}[-_])?\d{4,}", stem)
        )
        if has_digits_id or has_mrn_pat or has_pii_fn(probe) or has_pii_fn(stem):
            prefix = "doc"
            # Preserve generic non-PII category prefix if clean
            clean_tokens = []
            for tok in re.split(r"[_-]+", stem):
                if not tok:
                    continue
                if re.search(r"\d", tok) or has_pii_fn(tok):
                    break
                if tok.lower() in {
                    "discharge", "summary", "nursing", "report", "rx", "opd",
                    "lab", "note", "notes", "sanjeevani", "jeevanjyoti", "arogya",
                    "dhhardoi", "ds", "bundle", "whatsapp", "forward", "claims",
                }:
                    clean_tokens.append(tok.lower())
                else:
                    break
            if clean_tokens:
                prefix = "_".join(clean_tokens[:2])
            tag = pseudonym if pseudonym else keyring.safe_tag(file_sha256, 8)
            suffix_tag = keyring.safe_tag(file_sha256, 4) if pseudonym else ""
            new_stem = f"{prefix}_{tag}_{suffix_tag}".rstrip("_")
            new_parts.append(f"{new_stem}{ext}")
            changed = True
        else:
            new_parts.append(part)

    return "/".join(new_parts), changed
