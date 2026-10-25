"use strict";

const $ = (sel) => document.querySelector(sel);
const OUT_W = 1080;

const state = {
  sys: null,
  job: null,
  cues: [],
  style: null,
  ro: null,
  dirty: false,
  saveTimer: null,
  pollTimer: null,
  lastTaskKey: "",
};

// ---------- helpers ----------

async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.json);
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = r.status + " " + r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch (_) {}
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}

function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isErr ? " err" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), isErr ? 6000 : 2500);
}

function show(el, on) { (typeof el === "string" ? $(el) : el).classList.toggle("hidden", !on); }

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const fmtTime = (t) => {
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
};

// Same rule as the server: Thai marks above/below the line take no width.
const displayWidth = (s) => s.replace(/[\p{Mn}\p{Me}\p{Cf}]/gu, "").length;
const maxLineWidth = () => Math.max(6, Math.floor((OUT_W - 2 * state.style.margin_h) / (state.style.size * 0.58)));

// ---------- system status ----------

async function loadSystem() {
  try {
    state.sys = await api("/api/system");
  } catch (e) {
    toast("ติดต่อเซิร์ฟเวอร์ไม่ได้: " + e.message, true);
    return;
  }
  const s = state.sys;
  const chip = (ok, label) => `<span class="chip ${ok ? "ok" : "bad"}">${label}</span>`;
  $("#chips").innerHTML =
    chip(s.whisper.ok || s.stt_cloud.ok, "ถอดเสียง " + (s.whisper.ok ? "✓" : s.stt_cloud.ok ? "☁" : "✗")) +
    chip(s.llm.ok, "AI ตรวจคำ " + (s.llm.ok ? "✓" : "✗")) +
    chip(s.render.mode === "vaapi", s.render.mode === "vaapi" ? "Render iGPU" : "Render CPU");

  const models = s.whisper.models.length ? s.whisper.models : [s.whisper.current];
  $("#model").innerHTML =
    `<optgroup label="Whisper บน iGPU (NAS)">` +
    models.map((m) => `<option value="${esc(m)}" ${m === s.whisper.current ? "selected" : ""}>${esc(m)}</option>`).join("") +
    `</optgroup>` +
    (s.stt_cloud.models.length
      ? `<optgroup label="Cloud API">` + s.stt_cloud.models.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join("") + `</optgroup>`
      : "");

  $("#llm-model").innerHTML = llmOptions(s.llm);
  $("#proofread").disabled = !s.llm.ok;
  $("#btn-proofread").disabled = !s.llm.ok;
  if (!s.llm.ok) $("#proofread").checked = false;

  $("#st-font").innerHTML = s.fonts.map((f) => `<option value="${esc(f.family)}">${esc(f.family)}</option>`).join("");
  for (const f of s.fonts) {
    try {
      const face = new FontFace(f.family, `url(/fonts/${encodeURIComponent(f.file)})`, { weight: "700" });
      document.fonts.add(face);
      face.load().catch(() => {});
    } catch (_) {}
  }
  if (!state.style) state.style = { ...s.defaults.style };
  if (!state.ro) state.ro = { ...s.defaults.render_options };
  syncStyleInputs();
}

// <optgroup> per provider that is ready, default model selected.
function llmOptions(l) {
  const groups = l.providers.filter((p) => p.ok && p.models.length);
  const all = groups.flatMap((p) => p.models);
  let html = groups.map((p) => `<optgroup label="${esc(p.name)}">` +
    p.models.map((m) => `<option value="${esc(m)}" ${m === l.default_model ? "selected" : ""}>${esc(m)}</option>`).join("") +
    "</optgroup>").join("");
  if (l.default_model && !all.includes(l.default_model)) {
    html = `<option value="${esc(l.default_model)}" selected>${esc(l.default_model)}</option>` + html;
  }
  return html;
}

// ---------- step 1: upload ----------

$("#file").addEventListener("change", () => {
  const f = $("#file").files[0];
  $("#file-label").textContent = f ? `${f.name} (${(f.size / 1048576).toFixed(1)} MB)` : "แตะเพื่อเลือกวิดีโอจากเครื่อง";
  $(".file-pick").classList.toggle("has-file", !!f);
  $("#btn-upload").disabled = !f;
});

$("#btn-upload").addEventListener("click", () => {
  const f = $("#file").files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  fd.append("hint", $("#hint").value);
  const xhr = new XMLHttpRequest();
  const bar = $("#upload-progress");
  show(bar, true);
  $("#btn-upload").disabled = true;
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.firstElementChild.style.width = (e.loaded / e.total * 100) + "%"; };
  xhr.onload = () => {
    show(bar, false);
    $("#btn-upload").disabled = false;
    let data;
    try { data = JSON.parse(xhr.responseText); } catch (_) { data = { detail: xhr.statusText }; }
    if (xhr.status >= 300) return toast(data.detail || "อัปโหลดไม่สำเร็จ", true);
    state.style = { ...state.sys.defaults.style, ...data.style };
    setJob(data, true);
    loadRecent();
    $("#step-transcribe").scrollIntoView({ behavior: "smooth" });
  };
  xhr.onerror = () => { show(bar, false); $("#btn-upload").disabled = false; toast("อัปโหลดไม่สำเร็จ (เครือข่าย)", true); };
  xhr.open("POST", "/api/jobs");
  xhr.send(fd);
});

async function loadRecent() {
  try {
    const list = await api("/api/jobs");
    show("#recent-box", list.length > 0);
    $("#recent").innerHTML = list.map((j) =>
      `<li><a href="#job=${j.id}">${esc(j.filename || j.id)}</a><span>${new Date(j.created * 1000).toLocaleString("th-TH", { dateStyle: "short", timeStyle: "short" })}${j.has_output ? " ✅" : ""}</span></li>`
    ).join("");
  } catch (_) {}
}

async function openJob(id) {
  try {
    const job = await api(`/api/jobs/${id}`);
    state.style = { ...state.sys.defaults.style, ...job.style };
    state.ro = { ...state.sys.defaults.render_options, ...job.render_options };
    setJob(job, true);
  } catch (e) {
    toast(e.message, true);
    history.replaceState(null, "", location.pathname);
  }
}

// ---------- job state ----------

function setJob(job, replaceCues) {
  if (job.now) clockSkew = job.now - Date.now() / 1000;
  const firstLoad = !state.job || state.job.id !== job.id;
  state.job = job;
  if (firstLoad && !location.hash.startsWith("#bench")) history.replaceState(null, "", `#job=${job.id}`);
  if (replaceCues) {
    state.cues = (job.cues || []).map((c) => ({ ...c }));
    state.dirty = false;
    renderCues();
  }
  const info = job.info;
  $("#clip-info").innerHTML =
    `<img src="/api/jobs/${job.id}/thumb.jpg" alt="" onerror="this.remove()">` +
    `<div><b>${esc(job.filename || "")}</b><br>${info.width}×${info.height} · ${info.fps}fps · ${info.duration.toFixed(1)} วิ` +
    `${info.hdr ? " · HDR" : ""}${info.has_audio ? "" : " · ไม่มีเสียง"}</div>`;
  if (firstLoad) {
    $("#hint").value = job.hint || "";
    const v = $("#video");
    v.src = `/api/jobs/${job.id}/source`;
  }
  show("#step-transcribe", true);
  show("#step-edit", true);
  show("#step-render", true);
  syncStyleInputs();
  renderSuggestions();
  renderTask();
  renderResult();
}

function taskBox(name) { return name === "render" ? $("#task-render") : $("#task-transcribe"); }

const NODE_ICON = { pending: "", running: "", done: "✓", skipped: "–", warn: "!", error: "✕" };

// Server clock minus phone clock, so elapsed times are right even if the phone's clock is off.
let clockSkew = 0;
const serverNow = () => Date.now() / 1000 + clockSkew;

function fmtDur(sec) {
  if (sec == null || sec < 0) return "";
  if (sec < 60) return sec.toFixed(1) + " วิ";
  return `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")} นาที`;
}

function renderTask() {
  const t = state.job.task || {};
  const running = t.state === "queued" || t.state === "running";
  $("#btn-transcribe").disabled = running;
  $("#btn-render").disabled = running;
  $("#btn-proofread").disabled = running || !(state.sys && state.sys.llm.ok);
  for (const n of ["transcribe", "render"]) show(taskBox(n), false);
  if (!t.name || t.state === "idle") return;
  paintTask(taskBox(t.name), t);
  if (t.state === "error" && t.name === "render") loadLog();
  if (running) schedulePoll();
}

// Draw a task as a pipeline of nodes into one of the .task boxes.
function paintTask(box, t) {
  show(box, true);
  const pct = t.state === "done" ? 1 : t.progress || 0;
  box.querySelector(".progress > div").style.width = pct * 100 + "%";
  box.querySelector(".pct").textContent = Math.round(pct * 100) + "%";
  box.querySelector(".msg").textContent =
    t.state === "queued" ? "⏳ รอคิว" : t.state === "done" ? "✅ เสร็จแล้ว" : t.state === "error" ? "❌ ไม่สำเร็จ" : t.message;
  const total = box.querySelector(".total-time");
  total.dataset.s = t.started || "";
  total.dataset.e = t.ended || "";

  box.querySelector(".nodes").innerHTML = (t.steps || []).map((n) => {
    const showBar = n.state === "running";
    const indeterminate = showBar && !(n.progress > 0);
    return `<li class="node ${n.state}">
      <div class="dot">${NODE_ICON[n.state] || ""}</div>
      <div class="body">
        <div class="node-top">
          <span class="label">${esc(n.label)}</span>
          <span class="dev ${esc(n.device)}">${esc(n.device)}</span>
          <span class="time" data-s="${n.started || ""}" data-e="${n.ended || ""}"></span>
        </div>
        ${n.detail ? `<div class="detail">${esc(n.detail)}</div>` : ""}
        ${showBar ? `<div class="mini ${indeterminate ? "indeterminate" : ""}"><div style="width:${(n.progress || 0) * 100}%"></div></div>` : ""}
      </div>
    </li>`;
  }).join("");

  const err = box.querySelector(".task-error");
  err.textContent = t.state === "error" ? t.error || t.message : "";
  show(err, t.state === "error" && !!(t.error || t.message));
  tickTimes();
}

// Live elapsed time on every running node, without waiting for the next poll.
function tickTimes() {
  document.querySelectorAll(".task [data-s]").forEach((el) => {
    const s = parseFloat(el.dataset.s);
    if (!s) { el.textContent = ""; return; }
    const e = parseFloat(el.dataset.e) || serverNow();
    el.textContent = fmtDur(e - s);
  });
}
setInterval(tickTimes, 250);

function schedulePoll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(poll, 1000);
}

async function poll() {
  if (!state.job) return;
  let job;
  try {
    job = await api(`/api/jobs/${state.job.id}`);
  } catch (e) {
    return schedulePoll();
  }
  const t = job.task || {};
  const key = `${t.name}:${t.state}`;
  const justFinished = key !== state.lastTaskKey && t.state === "done";
  state.lastTaskKey = key;
  // Fresh transcript replaces the local cue list; otherwise keep the user's edits.
  setJob(job, justFinished && t.name === "transcribe");
  if (justFinished && t.name === "transcribe") {
    toast(`ได้ซับ ${job.cues.length} ท่อน`);
    $("#step-edit").scrollIntoView({ behavior: "smooth" });
  }
  if (justFinished && t.name === "render") {
    toast("เรนเดอร์เสร็จแล้ว 🎉");
    $("#result").scrollIntoView({ behavior: "smooth" });
  }
  if (justFinished && t.name === "proofread") {
    const n = (job.suggestions || []).length;
    toast(job.proofread_error ? job.proofread_error : n ? `AI แนะนำแก้ ${n} จุด` : "AI ไม่พบคำผิด", !!job.proofread_error);
  }
}

// ---------- step 2: transcribe ----------

$("#proofread").addEventListener("change", () => show("#llm-model-wrap", $("#proofread").checked));

$("#btn-transcribe").addEventListener("click", async () => {
  if (state.cues.length && !confirm("ถอดเสียงใหม่จะแทนที่ซับที่แก้ไว้ทั้งหมด ทำต่อไหม?")) return;
  try {
    const job = await api(`/api/jobs/${state.job.id}/transcribe`, {
      json: {
        hint: $("#hint").value,
        model: $("#model").value || null,
        vad: $("#vad").checked,
        proofread: $("#proofread").checked,
        llm_model: $("#llm-model").value || null,
      },
    });
    state.lastTaskKey = "";
    setJob(job, false);
  } catch (e) { toast(e.message, true); }
});

// ---------- step 3: cue editor ----------

function cueWarn(text) {
  const lines = text.split("\n").filter((l) => l.trim());
  const w = maxLineWidth();
  if (lines.length > 2) return "เกิน 2 บรรทัด";
  if (lines.some((l) => displayWidth(l) > w)) return `บรรทัดยาวเกิน ~${w} ตัว (ระบบจะตัดบรรทัดให้อัตโนมัติ)`;
  return "";
}

function renderCues() {
  const ol = $("#cues");
  ol.innerHTML = state.cues.map((c, i) => `
    <li class="cue" data-i="${i}">
      <div class="cue-times">
        <span>#${i + 1}</span>
        <input type="number" inputmode="decimal" step="0.1" min="0" data-f="start" value="${c.start.toFixed(2)}">
        <span class="sep">→</span>
        <input type="number" inputmode="decimal" step="0.1" min="0" data-f="end" value="${c.end.toFixed(2)}">
        <span>วิ</span>
      </div>
      <textarea rows="2" data-f="text">${esc(c.text)}</textarea>
      <div class="warn-line">${esc(cueWarn(c.text))}</div>
      <div class="cue-actions">
        <button class="btn icon" data-a="play" title="เล่นท่อนนี้">▶</button>
        <button class="btn icon" data-a="set-start" title="ตั้งเวลาเริ่ม = เวลาวิดีโอตอนนี้">⏱เริ่ม</button>
        <button class="btn icon" data-a="set-end" title="ตั้งเวลาจบ = เวลาวิดีโอตอนนี้">⏱จบ</button>
        <button class="btn icon" data-a="add" title="เพิ่มท่อนต่อจากนี้">＋</button>
        <button class="btn icon" data-a="del" title="ลบ">🗑</button>
      </div>
    </li>`).join("");
  show("#cues-empty", state.cues.length === 0);
  updateOverlay();
}

$("#cues").addEventListener("input", (e) => {
  const li = e.target.closest(".cue");
  if (!li) return;
  const c = state.cues[+li.dataset.i];
  const f = e.target.dataset.f;
  if (f === "text") {
    c.text = e.target.value;
    li.querySelector(".warn-line").textContent = cueWarn(c.text);
  } else if (f) {
    const v = parseFloat(e.target.value);
    if (!isNaN(v)) c[f] = Math.max(0, v);
  }
  markDirty();
  updateOverlay();
});

$("#cues").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-a]");
  if (!btn) return;
  const i = +btn.closest(".cue").dataset.i;
  const c = state.cues[i];
  const v = $("#video");
  switch (btn.dataset.a) {
    case "play":
      v.currentTime = c.start;
      v.play();
      stopAt = c.end;
      break;
    case "set-start":
      c.start = +v.currentTime.toFixed(2);
      if (c.end <= c.start) c.end = +(c.start + 1.5).toFixed(2);
      renderCues(); markDirty();
      break;
    case "set-end":
      c.end = Math.max(+v.currentTime.toFixed(2), c.start + 0.2);
      renderCues(); markDirty();
      break;
    case "add": {
      const next = state.cues[i + 1];
      const start = +(c.end + 0.05).toFixed(2);
      const end = +(next ? Math.max(Math.min(start + 2, next.start - 0.05), start + 0.5) : start + 2).toFixed(2);
      state.cues.splice(i + 1, 0, { start, end, text: "" });
      renderCues(); markDirty();
      focusCue(i + 1);
      break;
    }
    case "del":
      if (c.text.trim() && !confirm(`ลบท่อน "${c.text.slice(0, 30)}"?`)) return;
      state.cues.splice(i, 1);
      renderCues(); markDirty();
      break;
  }
});

function focusCue(i) {
  const ta = document.querySelector(`.cue[data-i="${i}"] textarea`);
  if (ta) { ta.scrollIntoView({ behavior: "smooth", block: "center" }); ta.focus({ preventScroll: true }); }
}

$("#btn-add-now").addEventListener("click", () => {
  const t = +$("#video").currentTime.toFixed(2);
  const next = state.cues.find((c) => c.start > t);
  const end = +(next ? Math.max(Math.min(t + 2, next.start - 0.05), t + 0.5) : t + 2).toFixed(2);
  state.cues.push({ start: t, end, text: "" });
  state.cues.sort((a, b) => a.start - b.start);
  renderCues(); markDirty();
  focusCue(state.cues.findIndex((c) => c.start === t && c.text === ""));
});

$("#btn-proofread").addEventListener("click", async () => {
  try {
    await flushSave();
    const job = await api(`/api/jobs/${state.job.id}/proofread`, { json: { llm_model: $("#llm-model").value || null } });
    state.lastTaskKey = "";
    setJob(job, false);
    toast("ส่งให้ AI ตรวจแล้ว รอสักครู่…");
  } catch (e) { toast(e.message, true); }
});

// AI suggestions: matched by cue id AND the original text, so stale ones disappear.
const flat = (s) => s.split("\n").map((l) => l.trim()).join("");

function liveSuggestions() {
  return (state.job.suggestions || []).filter((s) => {
    const c = state.cues[s.id - 1];
    return c && flat(c.text) === s.before;
  });
}

function renderSuggestions() {
  const list = liveSuggestions();
  show("#suggestions", list.length > 0);
  $("#sg-list").innerHTML = list.map((s) => `
    <li data-id="${s.id}">#${s.id} <del>${esc(s.before)}</del><br>→ <ins>${esc(s.after)}</ins>
      <div class="sg-actions"><button class="btn small" data-a="ok">✓ ใช้</button><button class="btn small ghost" data-a="no">✗ ไม่ใช้</button></div>
    </li>`).join("");
  const err = state.job.proofread_error;
  $("#proofread-error").textContent = err || "";
  show("#proofread-error", !!err);
}

function applySuggestion(s) {
  const c = state.cues[s.id - 1];
  if (c) c.text = s.after;
}

function dropSuggestion(id) {
  state.job.suggestions = (state.job.suggestions || []).filter((s) => s.id !== id);
}

$("#sg-list").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-a]");
  if (!btn) return;
  const id = +btn.closest("li").dataset.id;
  const s = state.job.suggestions.find((x) => x.id === id);
  if (btn.dataset.a === "ok" && s) applySuggestion(s);
  dropSuggestion(id);
  renderCues(); renderSuggestions(); markDirty();
});

$("#btn-accept-all").addEventListener("click", () => {
  for (const s of liveSuggestions()) applySuggestion(s);
  state.job.suggestions = [];
  renderCues(); renderSuggestions(); markDirty();
});

// ---------- autosave ----------

function editPayload() {
  return {
    cues: state.cues.map(({ start, end, text }) => ({ start, end, text })),
    hint: $("#hint").value,
    style: state.style,
    render_options: state.ro,
  };
}

function markDirty() {
  state.dirty = true;
  $("#save-state").textContent = "ยังไม่บันทึก…";
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(flushSave, 1500);
}

async function flushSave() {
  clearTimeout(state.saveTimer);
  if (!state.dirty || !state.job) return;
  const t = state.job.task || {};
  if (t.state === "queued" || t.state === "running") return;
  try {
    const job = await api(`/api/jobs/${state.job.id}/cues`, { method: "PUT", json: editPayload() });
    state.dirty = false;
    state.job.cues = job.cues;
    $("#save-state").textContent = "บันทึกแล้ว ✓";
  } catch (e) {
    $("#save-state").textContent = "บันทึกไม่สำเร็จ";
  }
}

// ---------- live preview overlay ----------

let stopAt = null;

function currentCue(t) {
  return state.cues.find((c) => t >= c.start && t < c.end);
}

function updateOverlay() {
  const v = $("#video");
  const ov = $("#overlay");
  const span = ov.firstElementChild;
  const st = state.style;
  if (!st) return;
  const pw = $("#preview").clientWidth || 360;
  const k = pw / OUT_W;
  const t = v.currentTime || 0;
  const cue = currentCue(t);
  span.textContent = cue ? cue.text : "";
  span.style.visibility = cue && cue.text.trim() ? "visible" : "hidden";
  ov.style.bottom = st.margin_v * k + "px";
  ov.style.padding = `0 ${st.margin_h * k}px`;
  ov.style.fontFamily = `"${st.font}", sans-serif`;
  ov.style.fontSize = st.size * k + "px";
  ov.style.color = st.color;
  ov.classList.toggle("box", !!st.box);
  span.style.webkitTextStroke = st.box ? "0" : `${st.outline * 2 * k}px ${st.outline_color}`;
  span.style.textShadow = st.box || !st.shadow ? "none" : `${st.shadow * k}px ${st.shadow * k}px 0 rgba(0,0,0,.55)`;
  if (st.box) span.style.background = hexA(st.outline_color, 0.65);
  else span.style.background = "none";
  $("#preview").classList.toggle("crop", state.ro && state.ro.fit === "crop");

  document.querySelectorAll(".cue.active").forEach((el) => el.classList.remove("active"));
  if (cue) {
    const el = document.querySelector(`.cue[data-i="${state.cues.indexOf(cue)}"]`);
    if (el) el.classList.add("active");
  }
}

function hexA(hex, a) {
  const h = hex.replace("#", "");
  return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`;
}

(function tick() {
  const v = $("#video");
  if (stopAt !== null && v.currentTime >= stopAt) { v.pause(); stopAt = null; }
  if (!v.paused) updateOverlay();
  requestAnimationFrame(tick);
})();
$("#video").addEventListener("seeked", updateOverlay);
$("#video").addEventListener("pause", () => { stopAt = null; updateOverlay(); });
window.addEventListener("resize", updateOverlay);

// ---------- style controls ----------

function syncStyleInputs() {
  const st = state.style;
  if (!st) return;
  $("#st-font").value = st.font;
  $("#st-size").value = st.size; $("#st-size-v").textContent = st.size;
  $("#st-outline").value = st.outline; $("#st-outline-v").textContent = st.outline;
  $("#st-margin").value = st.margin_v; $("#st-margin-v").textContent = st.margin_v + "px";
  $("#st-color").value = st.color.toLowerCase();
  $("#st-outline-color").value = st.outline_color.toLowerCase();
  $("#st-box").checked = !!st.box;
  document.querySelectorAll(".swatches").forEach((sw) => {
    const cur = (st[sw.dataset.target] || "").toUpperCase();
    sw.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v.toUpperCase() === cur));
  });
  if (state.ro) {
    $("#ro-fit").value = state.ro.fit;
    $("#ro-sharpen").value = state.ro.sharpen;
    $("#ro-denoise").value = state.ro.denoise;
  }
  updateOverlay();
}

function styleChanged() {
  syncStyleInputs();
  document.querySelectorAll(".cue").forEach((li) => {
    li.querySelector(".warn-line").textContent = cueWarn(state.cues[+li.dataset.i].text);
  });
  markDirty();
}

$("#st-font").addEventListener("change", (e) => { state.style.font = e.target.value; styleChanged(); });
$("#st-size").addEventListener("input", (e) => { state.style.size = +e.target.value; styleChanged(); });
$("#st-outline").addEventListener("input", (e) => { state.style.outline = +e.target.value; styleChanged(); });
$("#st-margin").addEventListener("input", (e) => { state.style.margin_v = +e.target.value; styleChanged(); });
$("#st-color").addEventListener("input", (e) => { state.style.color = e.target.value.toUpperCase(); styleChanged(); });
$("#st-outline-color").addEventListener("input", (e) => { state.style.outline_color = e.target.value.toUpperCase(); styleChanged(); });
$("#st-box").addEventListener("change", (e) => { state.style.box = e.target.checked; styleChanged(); });
document.querySelectorAll(".swatches").forEach((sw) => sw.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  e.preventDefault();
  state.style[sw.dataset.target] = b.dataset.v;
  styleChanged();
}));
for (const [id, key] of [["#ro-fit", "fit"], ["#ro-sharpen", "sharpen"], ["#ro-denoise", "denoise"]]) {
  $(id).addEventListener("change", (e) => { state.ro[key] = e.target.value; styleChanged(); });
}
$("#hint").addEventListener("input", () => { if (state.job) markDirty(); });

// ---------- step 4: render ----------

$("#btn-render").addEventListener("click", async () => {
  if (!state.cues.some((c) => c.text.trim()) && !confirm("ยังไม่มีซับเลย จะเรนเดอร์เฉพาะภาพให้ชัดขึ้นไหม?")) return;
  clearTimeout(state.saveTimer);
  try {
    const job = await api(`/api/jobs/${state.job.id}/render`, { json: editPayload() });
    state.dirty = false;
    $("#save-state").textContent = "บันทึกแล้ว ✓";
    state.lastTaskKey = "";
    show("#result", false);
    show("#log-box", false);
    setJob(job, false);
  } catch (e) { toast(e.message, true); }
});

function renderResult() {
  const job = state.job;
  const has = !!job.output && (job.task || {}).state !== "running";
  show("#result", has);
  if (!has) return;
  const url = `/api/jobs/${job.id}/output?v=${Math.round(job.updated || 0)}`;
  const ov = $("#out-video");
  if (ov.dataset.src !== url) { ov.src = url; ov.dataset.src = url; }
  $("#btn-download").href = `/api/jobs/${job.id}/output?download=1`;
  const m = job.render_meta || {};
  const mode = { gpu: "iGPU (VAAPI)", "gpu-swdec": "iGPU (VAAPI, ถอดรหัสด้วย CPU)", cpu: "CPU (libx264)" }[m.variant] || m.variant;
  $("#result-meta").textContent = m.seconds ? `เรนเดอร์ด้วย ${mode} · ${m.seconds} วิ · ${(m.size / 1048576).toFixed(1)} MB` : "";
}

async function loadLog() {
  try {
    const r = await fetch(`/api/jobs/${state.job.id}/log`);
    const txt = await r.text();
    $("#log").textContent = txt;
    show("#log-box", !!txt);
  } catch (_) {}
}

// ---------- tabs ----------

function showView(name) {
  show("#view-edit", name === "edit");
  show("#view-bench", name === "bench");
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("on", b.dataset.view === name));
  if (name === "bench") initBench();
}

document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  showView(b.dataset.view);
  if (b.dataset.view === "bench") history.replaceState(null, "", bench.run ? `#bench=${bench.run.id}` : "#bench");
  else history.replaceState(null, "", state.job ? `#job=${state.job.id}` : location.pathname);
  window.scrollTo({ top: 0 });
}));

// ---------- model comparison ----------

const bench = { testset: null, run: null, timer: null, inited: false };

async function initBench() {
  if (!state.sys) await loadSystem();
  const s = state.sys;
  // model checkboxes (keep what the user ticked when re-rendering)
  const box = $("#bench-models");
  const ticked = new Set([...box.querySelectorAll("input:checked")].map((i) => i.value));
  if (!bench.inited) ticked.add(s.llm.default_model);
  box.innerHTML = s.llm.providers.map((p) => {
    const head = `<div class="prov">${esc(p.name)} <span class="${p.ok ? "ok" : "bad"}">${esc(p.status)}</span></div>`;
    const items = p.ok ? p.models.map((m) =>
      `<label><input type="checkbox" value="${esc(m)}" ${ticked.has(m) ? "checked" : ""}> ${esc(m)}</label>`).join("") : "";
    return head + items;
  }).join("");
  $("#btn-bench").disabled = !s.llm.ok;

  if (!bench.testset) {
    try { bench.testset = await api("/api/bench/testset"); } catch (_) { bench.testset = { items: [], context: "" }; }
    $("#bench-testset").innerHTML = bench.testset.items.map((it) => {
      const wrong = it.text !== it.expected;
      return `<li>${wrong ? `<del>${esc(it.text)}</del> → <ins>${esc(it.expected)}</ins>` : esc(it.text)} <span class="kind">(${esc(it.kind)})</span></li>`;
    }).join("");
  }

  // test source: built-in set or the subtitles of a recent job
  let jobsList = [];
  try { jobsList = (await api("/api/jobs")).filter((j) => j.cues > 0); } catch (_) {}
  const sel = $("#bench-source");
  const cur = sel.value || "builtin";
  sel.innerHTML = `<option value="builtin">ชุดทดสอบมาตรฐาน (${bench.testset.items.length} ท่อน, มีเฉลย)</option>` +
    jobsList.map((j) => `<option value="${j.id}">ซับจาก ${esc(j.filename || j.id)} (${j.cues} ท่อน, ไม่มีเฉลย)</option>`).join("");
  sel.value = [...sel.options].some((o) => o.value === cur) ? cur : "builtin";
  if (!bench.inited) $("#bench-context").value = bench.testset.context;
  bench.inited = true;
  loadBenchHistory();
}

$("#bench-source").addEventListener("change", async () => {
  const v = $("#bench-source").value;
  if (v === "builtin") { $("#bench-context").value = bench.testset.context; return; }
  try { $("#bench-context").value = (await api(`/api/jobs/${v}`)).hint || ""; } catch (_) {}
});

$("#btn-bench").addEventListener("click", async () => {
  const models = [...document.querySelectorAll("#bench-models input:checked")].map((i) => i.value);
  if (!models.length) return toast("เลือกโมเดลอย่างน้อย 1 ตัว", true);
  if (models.length > 8) return toast("เลือกได้สูงสุด 8 ตัว", true);
  try {
    const run = await api("/api/bench", { json: { models, source: $("#bench-source").value, context: $("#bench-context").value } });
    history.replaceState(null, "", `#bench=${run.id}`);
    setBench(run);
  } catch (e) { toast(e.message, true); }
});

async function openBench(id) {
  try { setBench(await api(`/api/jobs/${id}`)); } catch (e) { toast(e.message, true); }
}

function setBench(run) {
  if (run.now) clockSkew = run.now - Date.now() / 1000;
  bench.run = run;
  const t = run.task || {};
  const running = t.state === "queued" || t.state === "running";
  $("#btn-bench").disabled = running || !(state.sys && state.sys.llm.ok);
  if (t.name) paintTask($("#task-bench"), t);
  renderBenchResults(run);
  clearTimeout(bench.timer);
  if (running) {
    bench.timer = setTimeout(async () => {
      try { setBench(await api(`/api/jobs/${run.id}`)); } catch (_) { setBench(run); }
    }, 1000);
  } else {
    loadBenchHistory();
  }
}

function renderBenchResults(run) {
  const results = run.results || {};
  const models = run.models.filter((m) => results[m]);
  show("#bench-results", models.length > 0);
  if (!models.length) return;
  const keyed = run.items.every((it) => "expected" in it);
  $("#bench-meta").textContent = `${run.source_name} · ${run.items.length} ท่อน` + (run.context ? ` · บริบท: ${run.context}` : "");

  // summary: best score first, then fastest
  const rows = models.map((m) => ({ m, r: results[m] }));
  rows.sort((a, b) => (b.r.ok - a.r.ok) || ((b.r.score?.score ?? 0) - (a.r.score?.score ?? 0)) || ((a.r.wall_s ?? 1e9) - (b.r.wall_s ?? 1e9)));
  const best = rows.find((x) => x.r.ok);
  const head = keyed
    ? "<tr><th>โมเดล</th><th>คะแนน</th><th>แก้ถูก<span class=sub>ผิด · เกิน</span></th><th>เวลา<span class=sub>tok/s</span></th></tr>"
    : "<tr><th>โมเดล</th><th>แนะนำแก้</th><th>ถูกตัดทิ้ง</th><th>เวลา<span class=sub>tok/s</span></th></tr>";
  $("#bench-summary").innerHTML = head + rows.map(({ m, r }) => {
    const notes = [];
    if (r.stats?.load_s >= 0.5) notes.push(`โหลดโมเดล ${r.stats.load_s} วิ`);
    if (r.stats?.thinking) notes.push("คิดก่อนตอบ");
    if (r.attempts > 1) notes.push(`ตอบผิดรูปแบบ ${r.attempts - 1} ครั้ง (ลองใหม่แล้วผ่าน)`);
    if (r.rejected_rewrites) notes.push(`ถูกตัดทิ้ง ${r.rejected_rewrites} (เขียนใหม่เกิน)`);
    const name = `${best && best.m === m && rows.length > 1 ? "🏆 " : ""}${esc(m)}${notes.length ? `<span class="sub">${esc(notes.join(" · "))}</span>` : ""}`;
    if (!r.ok) return `<tr><td>${name}</td><td class="err" colspan="3">❌ ${esc(r.error || "ล้มเหลว")}</td></tr>`;
    const time = `${r.wall_s} วิ`;
    const tps = r.stats?.tokens_per_s || "–";
    if (keyed) {
      const sc = r.score;
      return `<tr class="${best && best.m === m ? "best" : ""}"><td>${name}</td>
        <td><b>${sc.score}</b><span class="score-bar"><i style="width:${sc.score}%"></i></span></td>
        <td>${sc.fixed}/${sc.errors}<span class="sub">${sc.wrong_edits ?? 0} · ${sc.over_edits}</span></td><td>${time}<span class="sub">${tps}</span></td></tr>`;
    }
    return `<tr><td>${name}</td><td>${r.suggestions.length}</td><td>${r.rejected_rewrites || 0}</td><td>${time}<span class="sub">${tps}</span></td></tr>`;
  }).join("");

  // per cue: what each model ended up with
  const norm = (x) => (x || "").replace(/\s+/g, "");
  $("#bench-cues").innerHTML = run.items.map((it) => {
    const per = models.map((m) => {
      const r = results[m];
      if (!r.ok) return `<li class="bad"><span class="m">${esc(m)}</span><span class="r">–</span></li>`;
      const fin = r.finals[String(it.id)] ?? it.text;
      const changed = norm(fin) !== norm(it.text);
      let cls, label;
      if (keyed) {
        const good = norm(fin) === norm(it.expected);
        cls = good ? "ok" : "bad";
        label = (good ? "✓ " : "✗ ") + (changed ? fin : "ไม่แก้") + (!good && changed && it.text !== it.expected ? " (แก้ผิด)" : "");
      } else {
        cls = changed ? "changed" : "same";
        label = changed ? "✎ " + fin : "ไม่แก้";
      }
      return `<li class="${cls}"><span class="m">${esc(m)}</span><span class="r">${esc(label)}</span></li>`;
    }).join("");
    return `<li class="bench-cue">
      <div class="head"><span>#${it.id}</span><span>${esc(it.kind || "")}</span></div>
      <div class="orig">${esc(it.text)}</div>
      ${keyed && it.expected !== it.text ? `<div class="expect">เฉลย: ${esc(it.expected)}</div>` : ""}
      ${keyed && it.expected === it.text ? `<div class="expect">เฉลย: ถูกอยู่แล้ว ไม่ต้องแก้</div>` : ""}
      <ul>${per}</ul>
    </li>`;
  }).join("");
}

async function loadBenchHistory() {
  try {
    const list = await api("/api/bench");
    $("#bench-history").innerHTML = list.length
      ? list.map((r) => `<li><a href="#bench=${r.id}">${esc(r.models.join(", "))}</a><span>${esc(r.source_name || "")} · ${new Date(r.created * 1000).toLocaleString("th-TH", { dateStyle: "short", timeStyle: "short" })}${r.state === "running" || r.state === "queued" ? " ⏳" : ""}</span></li>`).join("")
      : `<li><span>ยังไม่เคยทดสอบ</span></li>`;
  } catch (_) {}
}

// ---------- boot ----------

window.addEventListener("hashchange", () => {
  const b = location.hash.match(/bench(?:=([a-z0-9]+))?/);
  if (b) { showView("bench"); if (b[1] && (!bench.run || bench.run.id !== b[1])) openBench(b[1]); return; }
  const m = location.hash.match(/job=([a-z0-9]+)/);
  if (m && (!state.job || state.job.id !== m[1])) { showView("edit"); openJob(m[1]); }
});
window.addEventListener("beforeunload", () => { if (state.dirty) flushSave(); });

(async function boot() {
  await loadSystem();
  loadRecent();
  const b = location.hash.match(/bench(?:=([a-z0-9]+))?/);
  if (b) {
    showView("bench");
    if (b[1]) openBench(b[1]);
    return;
  }
  const m = location.hash.match(/job=([a-z0-9]+)/);
  if (m && state.sys) openJob(m[1]);
})();
