/* Raksha Kavach Offline Console Frontend (Zero External Dependencies) */
const state = {
  defaults: null,
  policy: {},
  results: null,
  selectedFile: null,
  selectedCluster: null,
  sourceMode: "upload",
  selectedFiles: [],
  uploadedRawBySha: {},
};

const PRESETS = {
  hindi: {
    format: "text",
    text:
`Jeevan Jyoti Charitable Hospital, Barabanki
डिस्चार्ज सारांश

मरीज़ का नाम: रीना निषाद    आयु: 65 वर्ष / महिला
पिता का नाम: अजय निषाद
पंजीकरण सं.: JJH-0076928    आईपीडी सं.: IP/26/00117
पता: मकान नं. 398, ग्राम भीमनगर, तहसील कादीपुर, जिला सुल्तानपुर - 228076
मोबाइल: ७५६००१००३४
भर्ती की तारीख: 11-09-2026
छुट्टी की तारीख: 16-09-2026

निदान: पेशाब में संक्रमण (N39.0);
मरीज़ को 2 दिन से बुखार था। BP 105/73, Pulse 106/min।
Medications on discharge:
 1. Cap Doxycycline 100 mg 1-0-1 x 15 days
Review with Dr. Ramesh Srivastava. Attendant: Ajay Nishad, Mob. 7560010034`
  },
  meaning: {
    format: "text",
    text:
`AROGYA SEVA SADAN, SITAPUR
Patient Name: Manoj Yadav        Age/Sex: 65 Yrs/M
UHID: UHID-010623        S/o Ramesh Yadav
Address (attendant): C/O Ramesh Yadav, Ward 8, Sitapur - 261699
C/O: fever for 4 days, generalised weakness.
K/C/O addison's disease & Parkinson's disease.
Ward: 4 (Medicine)
Rx: Tab Ramipril 5 mg 0-0-1 x 30 days.
Dr. Ramesh Srivastava reviewed the patient. Referred by ASHA worker.`
  },
  whatsapp: {
    format: "text",
    text:
`[30/05/26, 9:32 pm] Kamal Bhaiya: <attachment: report.pdf>
Forwarded
*Lifeline Pathology*
Report of Asha ji  48 Y / F
Haemoglobin 14.0 g/dL | Random Blood Sugar 245 mg/dL | Serum Creatinine 2.5 mg/dL
Dr ne bola Dolo 650 BD x 5 days lena hai
Koi problem ho to call karo 8069773173
Visitor log note: Zorawar Qadri (attendant, bed 12), mob 6910522119`
  },
  json: {
    format: "json",
    text: JSON.stringify({
      "_export_meta": {
        "hospital": "District Hospital Hardoi",
        "source_path": "D:\\HIS\\exports\\opd\\Sunita_Patel_084999\\visits.json"
      },
      "visits": [{
        "visit_id": "OPD/26/05/00097",
        "patient_ref": "DHH2026-084999",
        "visit_date": "2026-05-20",
        "doctor": "Dr. Ramesh Srivastava",
        "complaints": "5 दिन से पेट दर्द",
        "history": "K/C/O Bell's palsy",
        "rx": [{"drug": "Azithromycin", "strength": "500 mg", "frequency": "1-0-0", "duration": "7 days"}],
        "patient": {"name": "Sunita Patel", "age_sex": "53/F", "mobile": "+91-8599253525"},
        "address_note": "C/O Ramesh Patel, Ward 1, Mohalla Chowk Kotwali",
        "attachment": {"data": "UmVmZXJyYWw6IFN1bml0YSBQYXRlbCAoREhIMjAyNi0wODQ5OTkpLCBtb2JpbGUgODU5OTI1MzUyNSwgUmVmZXJyZWQgZm9yIEJlbGwncyBwYWxzeS4="}
      }]
    }, null, 2)
  }
};

function esc(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function parseApiResponse(res, url) {
  const raw = await res.text();
  try {
    const data = raw ? JSON.parse(raw) : {};
    if (!res.ok && !data.error) data.error = `HTTP ${res.status}`;
    return data;
  } catch {
    if (res.status === 404) {
      return {
        error: `Endpoint ${url} returned HTTP 404 (older gui.py process running). Press Ctrl+C in terminal and restart '.venv/bin/python gui.py --port 8573'.`,
      };
    }
    return { error: `Server returned HTTP ${res.status} (non-JSON response).` };
  }
}

async function apiGet(url) {
  const res = await fetch(url);
  return parseApiResponse(res, url);
}

async function apiPost(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseApiResponse(res, url);
}

const TAB_TITLES = {
  overview: ["Run & Executive Audit Overview", "Ingest heterogeneous clinical exports, enforce privacy policy, and inspect release integrity."],
  files: ["File Audit & Quarantine Manifest", "Every input file indexed strictly by SHA-256 with rename tracking and fail-safe quarantine reasons."],
  linkage: ["Cross-Script Identity Linkage Explorer", "Verify deterministic keyed HMAC pseudonyms across CSV, TSV, JSON, FHIR R4, and free-text notes."],
  inspector: ["Sanitized Document Inspector", "Inspect UTF-8 cleaned records, age bands, shifted dates, and optional local input comparison."],
  sandbox: ["Live Multilingual De-identification Sandbox", "Test English, Devanagari, and Hinglish snippets with live PII span & clinical guard highlighting."],
  policy: ["Privacy Policy & Verification Suite", "Configure k-anonymity thresholds in config/policy.yaml and run official verification checks."],
};

function switchTab(tab) {
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${tab}`));
  const [title, sub] = TAB_TITLES[tab] || ["Raksha Kavach", ""];
  document.getElementById("page-title").textContent = title;
  document.getElementById("page-subtitle").textContent = sub;
}

function showBanner(id, msg, ok = true) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.className = `banner ${ok ? "banner-ok" : "banner-err"}`;
}

function setStages(mode) {
  document.querySelectorAll("#pipeline-stages .stage").forEach(el => {
    el.classList.remove("running", "done");
    if (mode === "running") el.classList.add("running");
    if (mode === "done") el.classList.add("done");
  });
}

function renderDashboard(data) {
  state.results = data;
  const zipBtn = document.getElementById("btn-download-zip");
  if (zipBtn) zipBtn.disabled = !(data && data.exists);
  if (!data || !data.exists) return;
  const resultsEl = document.getElementById("overview-results");
  if (resultsEl) resultsEl.classList.remove("hidden");

  const rep = data.report || {};
  const counts = rep.counts || {};
  document.getElementById("kpi-total").textContent = counts.files_total ?? 0;
  document.getElementById("kpi-sanitized").textContent = counts.files_sanitized ?? 0;
  document.getElementById("kpi-quarantined").textContent = counts.files_quarantined ?? 0;
  document.getElementById("kpi-identifiers").textContent = counts.identifiers_detected ?? 0;
  document.getElementById("kpi-clusters").textContent = data.cluster_count ?? 0;
  document.getElementById("kpi-generalized").textContent = counts.records_generalized ?? 0;
  document.getElementById("kpi-time").textContent = `${(rep.timing || {}).seconds ?? 0}s`;
  document.getElementById("kpi-rss").textContent = `${(rep.resources || {}).peak_rss_mb ?? 0} MB`;
  document.getElementById("kpi-sub-files").textContent = `${counts.records_processed ?? 0} records exported`;
  document.getElementById("kpi-sub-linked").textContent = `${counts.records_linked ?? 0} linked / ${data.unlinked_count ?? 0} null`;

  // Entity bar chart
  const entities = Object.entries(rep.entities || {}).sort((a, b) => b[1] - a[1]);
  const chart = document.getElementById("entity-chart");
  if (!entities.length) {
    chart.innerHTML = `<div class="empty-state">Zero entities reported.</div>`;
  } else {
    const maxVal = Math.max(...entities.map(e => e[1]), 1);
    chart.innerHTML = entities.map(([k, v]) => `
      <div class="bar-row">
        <span class="bar-label" title="${esc(k)}">${esc(k)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, (v / maxVal) * 100).toFixed(1)}%"></div></div>
        <span class="bar-num">${v}</span>
      </div>
    `).join("");
  }

  // Quarantine breakdown
  const qFiles = (rep.files || []).filter(f => f.status === "quarantined");
  const qBox = document.getElementById("quarantine-summary");
  if (!qFiles.length) {
    qBox.innerHTML = `<div class="empty-state">All input files parsed cleanly; 0 files quarantined.</div>`;
  } else {
    const byReason = {};
    qFiles.forEach(f => { byReason[f.reason || "unknown"] = (byReason[f.reason || "unknown"] || 0) + 1; });
    const entries = Object.entries(byReason).sort((a, b) => b[1] - a[1]);
    const maxQ = Math.max(...entries.map(e => e[1]), 1);
    qBox.innerHTML = entries.map(([r, c]) => `
      <div class="bar-row">
        <span class="bar-label">${esc(r)}</span>
        <div class="bar-track"><div class="bar-fill bar-fill-warn" style="width:${((c / maxQ) * 100).toFixed(1)}%"></div></div>
        <span class="bar-num">${c}</span>
      </div>
    `).join("") + `<p class="hint" style="margin-top:10px">Quarantined SHA-256 digests are recorded in <code>report.json</code> and strictly excluded from <code>sanitized/</code>.</p>`;
  }

  renderFilesTable();
  renderClusters();
}

function renderFilesTable() {
  const rep = (state.results || {}).report;
  const tbody = document.getElementById("files-tbody");
  if (!rep || !rep.files) return;

  const q = document.getElementById("file-search").value.trim().toLowerCase();
  const st = document.getElementById("file-status-filter").value;

  const rows = rep.files.filter(f => {
    if (st !== "all" && f.status !== st) return false;
    if (!q) return true;
    return (f.input_sha256 || "").toLowerCase().includes(q)
      || (f.output || "").toLowerCase().includes(q)
      || (f.reason || "").toLowerCase().includes(q);
  });

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-cell">No matching files.</td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map((f, idx) => {
    const isSan = f.status === "sanitized";
    const badge = isSan ? `<span class="badge badge-ok">sanitized</span>` : `<span class="badge badge-warn">quarantined</span>`;
    const action = isSan
      ? `<button class="btn btn-ghost btn-sm inspect-btn" data-out="${esc(f.output)}" data-sha="${esc(f.input_sha256)}">Inspect</button>`
      : `<span class="hint">Isolated</span>`;
    return `
      <tr>
        <td>${idx + 1}</td>
        <td class="mono" title="${esc(f.input_sha256)}">${esc((f.input_sha256 || "").slice(0, 14))}&hellip;</td>
        <td>${badge}</td>
        <td class="mono">${esc(f.output || "null")}</td>
        <td class="mono">${esc(f.reason || "—")}</td>
        <td>${action}</td>
      </tr>
    `;
  }).join("");

  tbody.querySelectorAll(".inspect-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      openInspector(btn.dataset.out, btn.dataset.sha);
      switchTab("inspector");
    });
  });
}

function renderClusters() {
  const data = state.results;
  if (!data) return;
  const q = document.getElementById("cluster-search").value.trim().toLowerCase();
  const clusters = (data.clusters || []).filter(c =>
    !q || c.patient.toLowerCase().includes(q) || c.files.some(f => f.toLowerCase().includes(q))
  );
  document.getElementById("cluster-summary-bar").innerHTML =
    `<strong>${data.cluster_count}</strong> distinct pseudonym clusters &middot; <strong>${data.total_records - data.unlinked_count}</strong> linked records &middot; <strong>${data.unlinked_count}</strong> unlinked/null records`;

  const listEl = document.getElementById("cluster-list");
  if (!clusters.length) {
    listEl.innerHTML = `<div class="empty-state">No clusters match your search.</div>`;
    return;
  }

  listEl.innerHTML = clusters.map(c => `
    <div class="cluster-item ${state.selectedCluster === c.patient ? "active" : ""}" data-psn="${esc(c.patient)}">
      <div>
        <div class="mono" style="font-weight:700;color:#0f2942">${esc(c.patient)}</div>
        <div class="hint">${c.file_count} files &middot; ${esc(c.files[0] || "")}</div>
      </div>
      <span class="badge badge-info">${c.record_count} recs</span>
    </div>
  `).join("");

  listEl.querySelectorAll(".cluster-item").forEach(item => {
    item.addEventListener("click", () => {
      const c = clusters.find(x => x.patient === item.dataset.psn);
      state.selectedCluster = item.dataset.psn;
      renderClusters();
      renderClusterDetail(c);
    });
  });
}

function renderClusterDetail(cluster) {
  if (!cluster) return;
  document.getElementById("cluster-detail-title").textContent = `Cluster ${cluster.patient} (${cluster.record_count} records)`;
  const body = document.getElementById("cluster-detail-body");
  body.innerHTML = `
    <table class="data-table">
      <thead><tr><th>File</th><th>Locator</th><th>Status</th><th>Action</th></tr></thead>
      <tbody>
        ${cluster.records.map(r => `
          <tr>
            <td class="mono">${esc(r.file)}</td>
            <td class="mono">${esc(r.loc)}</td>
            <td><span class="badge badge-neutral">${esc(r.status || "sanitized")}</span></td>
            <td><button class="btn btn-ghost btn-sm open-rec-btn" data-file="${esc(r.file)}">View</button></td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
  body.querySelectorAll(".open-rec-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const repFiles = ((state.results || {}).report || {}).files || [];
      const match = repFiles.find(x => x.output === btn.dataset.file);
      openInspector(btn.dataset.file, match ? match.input_sha256 : "");
      switchTab("inspector");
    });
  });
}

async function openInspector(relOut, sha) {
  state.selectedFile = { rel: relOut, sha: sha || "" };
  const outDir = document.getElementById("out-dir").value.trim();
  const inpDir = document.getElementById("inp-dir").value.trim();
  const includeRaw = document.getElementById("chk-compare-raw").checked ? "1" : "0";

  document.getElementById("inspector-meta").textContent = `${relOut} (SHA-256: ${(sha || "n/a").slice(0, 16)}...)`;
  const qs = new URLSearchParams({ output_dir: outDir, rel: relOut, input_dir: inpDir, sha256: sha || "", include_raw: includeRaw });
  const data = await apiGet(`/api/file?${qs.toString()}`);

  const sanView = document.getElementById("sanitized-view");
  if (data.csv_rows && data.csv_rows.length) {
    const [hdr, ...rows] = data.csv_rows;
    sanView.innerHTML = `<table class="data-table"><thead><tr>${hdr.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${
      rows.map(r => `<tr>${r.map(c => `<td>${highlightSanitized(c)}</td>`).join("")}</tr>`).join("")
    }</tbody></table>`;
  } else {
    sanView.innerHTML = `<pre class="code-pre" style="padding:0;max-height:none">${highlightSanitized(data.sanitized_content || "Unable to read file.")}</pre>`;
  }

  const grid = document.getElementById("inspector-grid");
  const rawPane = document.getElementById("raw-pane");
  let rawText = data.raw_content;
  if (includeRaw === "1" && rawText === null && sha && state.uploadedRawBySha[sha]) {
    rawText = decodeRawBytes(state.uploadedRawBySha[sha]);
  }
  if (includeRaw === "1" && rawText !== null && rawText !== undefined) {
    grid.className = "inspector-split";
    rawPane.classList.remove("hidden");
    document.getElementById("raw-view").textContent = rawText;
  } else {
    grid.className = "inspector-single";
    rawPane.classList.add("hidden");
  }
}

function highlightSanitized(text) {
  return esc(text)
    .replace(/\bPSN-[0-9A-Z]{6}\b/g, m => `<span class="tok-psn">${m}</span>`)
    .replace(/\[[A-Z_]{3,20}\]/g, m => `<span class="tok-redacted">${m}</span>`);
}

function renderPolicyForm(policy) {
  state.policy = policy || {};
  const fields = [
    ["age_band_years", "Standard Age Band (yrs)", "number"],
    ["high_risk_age_band_years", "High-Risk Age Band (yrs)", "number"],
    ["elderly_cap_min_age", "Elderly Cap Min Age", "number"],
    ["min_group_size", "k-Anonymity Min Group Size", "number"],
    ["rare_condition_threshold", "Rare Condition Threshold", "number"],
    ["date_shift_min_days", "Date Shift Min Days", "number"],
    ["date_shift_max_days", "Date Shift Max Days", "number"],
    ["pin_prefix_digits", "PIN Prefix Digits Kept", "number"],
    ["max_json_depth", "Max JSON Depth Guard", "number"],
    ["default_state_label", "Generalized Region Label", "text"],
    ["suppress_dob", "Suppress Exact DOB", "bool"],
    ["generalize_rare_district", "Coarsen Sparse/Rare Districts", "bool"],
    ["encrypt_intermediate_vault", "AES-256-GCM In-Memory Vault", "bool"],
  ];

  document.getElementById("policy-form").innerHTML = fields.map(([k, label, kind]) => {
    const val = state.policy[k];
    if (kind === "bool") {
      return `<label class="policy-check"><input type="checkbox" data-pkey="${k}" ${val ? "checked" : ""}> ${esc(label)}</label>`;
    }
    return `<div class="policy-field"><label>${esc(label)}</label><input type="${kind}" data-pkey="${k}" value="${esc(val ?? "")}"></div>`;
  }).join("");
}

function collectPolicyFromForm() {
  const out = {};
  document.querySelectorAll("#policy-form [data-pkey]").forEach(el => {
    const k = el.dataset.pkey;
    if (el.type === "checkbox") out[k] = el.checked;
    else if (el.type === "number") out[k] = Number(el.value);
    else out[k] = el.value;
  });
  return out;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function arrayBufferToBase64(buf) {
  const bytes = new Uint8Array(buf);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

async function sha256Hex(buf) {
  if (!window.crypto || !window.crypto.subtle) return "";
  try {
    const hashBuf = await window.crypto.subtle.digest("SHA-256", buf);
    return Array.from(new Uint8Array(hashBuf)).map(b => b.toString(16).padStart(2, "0")).join("");
  } catch {
    return "";
  }
}

function decodeRawBytes(bytes) {
  if (!bytes || !bytes.length) return "";
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe) {
    return new TextDecoder("utf-16le", { fatal: false }).decode(bytes).slice(0, 200000);
  }
  if (bytes.length >= 2 && bytes[0] === 0xfe && bytes[1] === 0xff) {
    return new TextDecoder("utf-16be", { fatal: false }).decode(bytes).slice(0, 200000);
  }
  return new TextDecoder("utf-8", { fatal: false }).decode(bytes).slice(0, 200000);
}

function setSourceMode(mode) {
  state.sourceMode = mode;
  document.getElementById("btn-mode-upload").classList.toggle("active", mode === "upload");
  document.getElementById("btn-mode-path").classList.toggle("active", mode === "path");
  document.getElementById("source-upload-panel").classList.toggle("hidden", mode !== "upload");
  document.getElementById("source-path-panel").classList.toggle("hidden", mode !== "path");
}

function stripCommonTopFolder(items) {
  if (!items.length) return items;
  const firstSegs = items.map(it => it.rel_path.split("/")[0]);
  const allHaveSlash = items.every(it => it.rel_path.includes("/"));
  if (allHaveSlash && firstSegs.every(s => s === firstSegs[0])) {
    return items.map(it => ({
      ...it,
      rel_path: it.rel_path.slice(firstSegs[0].length + 1),
    }));
  }
  return items;
}

function setSelectedFiles(items) {
  const cleaned = stripCommonTopFolder(items.filter(it => it.rel_path && it.file));
  const seen = new Set();
  state.selectedFiles = [];
  for (const it of cleaned) {
    if (!seen.has(it.rel_path)) {
      seen.add(it.rel_path);
      state.selectedFiles.push(it);
    }
  }
  renderSelectedFiles();
}

function renderSelectedFiles() {
  const box = document.getElementById("selected-files-box");
  const clearBtn = document.getElementById("btn-clear-files");
  const sumEl = document.getElementById("selected-files-summary");
  const listEl = document.getElementById("selected-files-list");
  const files = state.selectedFiles;

  if (!files.length) {
    box.classList.add("hidden");
    clearBtn.classList.add("hidden");
    sumEl.textContent = "";
    listEl.innerHTML = "";
    return;
  }

  const totalBytes = files.reduce((acc, f) => acc + (f.file.size || 0), 0);
  box.classList.remove("hidden");
  clearBtn.classList.remove("hidden");
  sumEl.textContent = `${files.length} file${files.length === 1 ? "" : "s"} selected (${formatBytes(totalBytes)}) — ready to sanitize`;
  const preview = files.slice(0, 60).map(f =>
    `<div class="file-chip" title="${esc(f.rel_path)}">${esc(f.rel_path)} <span>(${esc(formatBytes(f.file.size || 0))})</span></div>`
  ).join("");
  const extra = files.length > 60 ? `<div class="file-chip">+${files.length - 60} more</div>` : "";
  listEl.innerHTML = preview + extra;
}

async function readEntryRecursive(entry, prefix = "") {
  if (entry.isFile) {
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    return [{ rel_path: `${prefix}${entry.name}`, file }];
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    const out = [];
    while (true) {
      const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
      if (!batch.length) break;
      for (const child of batch) {
        out.push(...(await readEntryRecursive(child, `${prefix}${entry.name}/`)));
      }
    }
    return out;
  }
  return [];
}

async function collectDroppedFiles(dataTransfer) {
  const items = Array.from(dataTransfer.items || []);
  const entries = items.map(it => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
  if (entries.length) {
    const all = [];
    for (const ent of entries) {
      all.push(...(await readEntryRecursive(ent, "")));
    }
    return all;
  }
  return Array.from(dataTransfer.files || []).map(file => ({
    rel_path: file.webkitRelativePath || file.name,
    file,
  }));
}

async function runBatchFirewall() {
  const btn = document.getElementById("btn-run-firewall");
  const topBtn = document.getElementById("btn-top-run");
  const outDir = document.getElementById("out-dir").value.trim();
  const secret = "";
  const policyOverrides = collectPolicyFromForm();

  if (!outDir) {
    showBanner("run-banner", "Please specify an Output Release Directory.", false);
    return;
  }

  btn.disabled = topBtn.disabled = true;
  setStages("running");
  document.getElementById("run-banner").classList.add("hidden");

  try {
    let data;
    if (state.sourceMode === "upload") {
      if (!state.selectedFiles.length) {
        setStages("");
        showBanner("run-banner", "Select files or a folder first (or switch to 'Local Folder Path').", false);
        return;
      }
      const totalBytes = state.selectedFiles.reduce((acc, f) => acc + (f.file.size || 0), 0);
      if (totalBytes > 50 * 1024 * 1024) {
        setStages("");
        showBanner("run-banner", "Selected files exceed the 50 MB browser upload limit; use 'Local Folder Path' for huge directories.", false);
        return;
      }
      state.uploadedRawBySha = {};
      const payloadFiles = [];
      for (const item of state.selectedFiles) {
        const buf = await item.file.arrayBuffer();
        const hex = await sha256Hex(buf);
        if (hex) state.uploadedRawBySha[hex] = new Uint8Array(buf);
        payloadFiles.push({
          rel_path: item.rel_path,
          content_b64: arrayBufferToBase64(buf),
        });
      }
      data = await apiPost("/api/run-upload", {
        files: payloadFiles,
        output_dir: outDir,
        secret,
        policy_overrides: policyOverrides,
      });
    } else {
      const inpDir = document.getElementById("inp-dir").value.trim();
      if (!inpDir) {
        setStages("");
        showBanner("run-banner", "Enter a Local Input Directory Path, or switch to 'Select Files / Folder'.", false);
        return;
      }
      data = await apiPost("/api/run", {
        input_dir: inpDir,
        output_dir: outDir,
        secret,
        policy_overrides: policyOverrides,
      });
    }

    if (!data.ok) {
      setStages("");
      showBanner("run-banner", data.error || data.stderr || "Pipeline execution failed.", false);
      return;
    }
    setStages("done");
    showBanner("run-banner", `Completed cleanly: ${data.cli_summary}`, true);
    renderDashboard(data);
  } catch (err) {
    setStages("");
    showBanner("run-banner", `Error: ${err.message}`, false);
  } finally {
    btn.disabled = topBtn.disabled = false;
  }
}

async function runSandbox() {
  const text = document.getElementById("sbx-input").value;
  const fmt = document.getElementById("sbx-format").value;
  const secret = document.getElementById("sbx-secret").value;
  const res = await apiPost("/api/sandbox", { text, format: fmt, secret, policy_overrides: collectPolicyFromForm() });

  if (res.error) {
    document.getElementById("sbx-output").textContent = res.error;
    return;
  }
  if (res.status === "quarantined") {
    document.getElementById("sbx-annotated").innerHTML = `<span class="badge badge-warn">QUARANTINED: ${esc(res.reason)}</span>`;
    document.getElementById("sbx-output").textContent = "File quarantined safely; nothing exported to sanitized/.";
    document.getElementById("sbx-meta").textContent = `SHA-256: ${res.sha256}`;
    return;
  }

  // Render annotated input with non-overlapping PII & Protected Clinical spans
  if (fmt === "text") {
    const events = [];
    (res.pii_spans || []).forEach(s => events.push({ start: s.start, end: s.end, kind: "pii", label: s.type }));
    (res.protected_spans || []).forEach(s => {
      const overlaps = events.some(e => s.start < e.end && s.end > e.start);
      if (!overlaps) events.push({ start: s.start, end: s.end, kind: "prot", label: "CLINICAL" });
    });
    events.sort((a, b) => a.start - b.start);
    let html = "", cur = 0;
    for (const ev of events) {
      if (ev.start < cur) continue;
      html += esc(text.slice(cur, ev.start));
      const cls = ev.kind === "pii" ? "tag-pii" : "tag-prot";
      html += `<span class="${cls}" title="${esc(ev.label)}">${esc(text.slice(ev.start, ev.end))}</span>`;
      cur = ev.end;
    }
    html += esc(text.slice(cur));
    document.getElementById("sbx-annotated").innerHTML = html;
  } else {
    document.getElementById("sbx-annotated").textContent = text;
  }

  document.getElementById("sbx-output").innerHTML = highlightSanitized(res.sanitized_text || "");
  const psns = (res.records || []).map(r => r.patient || "null").join(", ");
  const counts = Object.entries(res.entity_counts || {}).map(([k, v]) => `${k}: ${v}`).join(" | ") || "none";
  document.getElementById("sbx-meta").innerHTML =
    `<strong>Pseudonym(s):</strong> <span class="mono">${esc(psns)}</span> &middot; <strong>Detected:</strong> ${esc(counts)}`;
}

async function runVerification(kind) {
  const outEl = document.getElementById("verify-output");
  const stEl = document.getElementById("verify-status");
  stEl.textContent = `Running ${kind}...`;
  outEl.textContent = "Executing offline verification check...";
  const data = await apiPost("/api/verify", {
    kind,
    input_dir: document.getElementById("inp-dir").value,
    output_dir: document.getElementById("out-dir").value,
  });
  if (data.error) {
    stEl.innerHTML = `<span class="badge badge-warn">ERROR</span> ${esc(data.error)}`;
    outEl.textContent = data.error;
    return;
  }
  stEl.innerHTML = data.ok
    ? `<span class="badge badge-ok">PASSED</span> Verification check <code>${esc(kind)}</code> exited 0.`
    : `<span class="badge badge-warn">FAILED</span> Exit code ${data.returncode}`;
  outEl.textContent = data.output || "(no output)";
}

async function bootstrap() {
  document.querySelectorAll(".nav-btn").forEach(btn => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));
  document.getElementById("btn-run-firewall").addEventListener("click", runBatchFirewall);
  document.getElementById("btn-top-run").addEventListener("click", () => { switchTab("overview"); runBatchFirewall(); });
  document.getElementById("btn-mode-upload").addEventListener("click", () => setSourceMode("upload"));
  document.getElementById("btn-mode-path").addEventListener("click", () => setSourceMode("path"));

  const pickFiles = document.getElementById("pick-files");
  const pickFolder = document.getElementById("pick-folder");
  document.getElementById("btn-pick-files").addEventListener("click", () => pickFiles.click());
  document.getElementById("btn-pick-folder").addEventListener("click", () => pickFolder.click());
  document.getElementById("btn-clear-files").addEventListener("click", () => {
    pickFiles.value = "";
    pickFolder.value = "";
    state.selectedFiles = [];
    renderSelectedFiles();
  });
  pickFiles.addEventListener("change", () => {
    setSelectedFiles(Array.from(pickFiles.files || []).map(file => ({ rel_path: file.name, file })));
  });
  pickFolder.addEventListener("change", () => {
    setSelectedFiles(Array.from(pickFolder.files || []).map(file => ({
      rel_path: file.webkitRelativePath || file.name,
      file,
    })));
  });

  const dropZone = document.getElementById("drop-zone");
  dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", async e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const items = await collectDroppedFiles(e.dataTransfer);
    if (items.length) setSelectedFiles(items);
  });

  document.getElementById("btn-download-zip").addEventListener("click", () => {
    const outDir = document.getElementById("out-dir").value.trim();
    if (!outDir) return;
    window.location.assign(`/api/download-zip?output_dir=${encodeURIComponent(outDir)}`);
  });
  document.getElementById("btn-use-dev").addEventListener("click", () => {
    if (state.defaults && state.defaults.toolkit_input_dir) {
      document.getElementById("inp-dir").value = state.defaults.toolkit_input_dir;
    }
  });
  document.getElementById("btn-load-existing").addEventListener("click", async () => {
    const outDir = document.getElementById("out-dir").value.trim();
    const res = await apiGet(`/api/results?output_dir=${encodeURIComponent(outDir)}`);
    if (!res.exists) showBanner("run-banner", `No report.json found in ${outDir}`, false);
    else {
      showBanner("run-banner", `Loaded existing output from ${outDir}`, true);
      renderDashboard(res);
    }
  });
  document.getElementById("file-search").addEventListener("input", renderFilesTable);
  document.getElementById("file-status-filter").addEventListener("change", renderFilesTable);
  document.getElementById("cluster-search").addEventListener("input", renderClusters);
  document.getElementById("chk-compare-raw").addEventListener("change", () => {
    if (state.selectedFile) openInspector(state.selectedFile.rel, state.selectedFile.sha);
  });
  document.getElementById("btn-reload-doc").addEventListener("click", () => {
    if (state.selectedFile) openInspector(state.selectedFile.rel, state.selectedFile.sha);
  });
  document.querySelectorAll(".preset-btn").forEach(b => {
    b.addEventListener("click", () => {
      const p = PRESETS[b.dataset.preset];
      if (!p) return;
      document.getElementById("sbx-format").value = p.format;
      document.getElementById("sbx-input").value = p.text;
      runSandbox();
    });
  });
  document.getElementById("btn-run-sandbox").addEventListener("click", runSandbox);
  document.getElementById("btn-save-policy").addEventListener("click", async () => {
    const res = await apiPost("/api/policy", { policy: collectPolicyFromForm() });
    if (res.ok) {
      renderPolicyForm(res.policy);
      showBanner("policy-banner", "Saved updated privacy policy to config/policy.yaml.", true);
    } else {
      showBanner("policy-banner", res.error || "Failed to save policy.", false);
    }
  });
  document.querySelectorAll(".verify-btn").forEach(b => b.addEventListener("click", () => runVerification(b.dataset.kind)));

  // Load initial defaults & health
  const health = await apiGet("/api/health");
  const hBadge = document.getElementById("health-badge");
  const hasUploadCap = Array.isArray(health.capabilities) && health.capabilities.includes("run-upload");
  if (health.ok && hasUploadCap) {
    hBadge.className = "badge badge-ok";
    hBadge.textContent = "Engine Ready (Offline)";
  } else if (health.ok) {
    hBadge.className = "badge badge-warn";
    hBadge.textContent = "Restart gui.py Needed";
    showBanner("run-banner", "An older gui.py server is running on this port. Press Ctrl+C in terminal and restart '.venv/bin/python gui.py --port 8573' to activate Select Files / Folder.", false);
  }
  const defs = await apiGet("/api/defaults");
  state.defaults = defs;
  document.getElementById("inp-dir").value = "";
  document.getElementById("out-dir").value = defs.output_dir || "";
  if (defs.toolkit_input_dir) {
    document.getElementById("btn-use-dev").classList.remove("hidden");
  }
  renderPolicyForm(defs.policy || {});
}

document.addEventListener("DOMContentLoaded", bootstrap);
