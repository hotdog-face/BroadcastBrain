/* BroadcastBrain dashboard.
 *
 * Vanilla JS, no build step, no dependencies. Polls the JSON API on a timer
 * and re-renders. At the scale this UI operates on (a few hundred rows) that
 * is simpler and more robust than incremental DOM updates, and it means the
 * page recovers on its own if the server restarts mid-run.
 */

const REFRESH_MS = 2500;
const SLOW_REFRESH_MS = 15000;

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- helpers */

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${path} -> ${response.status}`);
  return response.json();
}

function duration(seconds) {
  if (!seconds || seconds < 0) return "0s";
  if (seconds >= 3600) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${String(m).padStart(2, "0")}m`;
  }
  if (seconds >= 60) {
    const m = Math.floor(seconds / 60);
    return `${m}m ${String(Math.floor(seconds % 60)).padStart(2, "0")}s`;
  }
  return `${Math.round(seconds)}s`;
}

function bytes(value) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let index = 0;
  let size = value;
  while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
  return `${size.toFixed(size < 10 ? 1 : 0)} ${units[index]}`;
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function stat(label, value) {
  return `<div class="stat"><div class="value">${escapeHtml(value)}</div>
          <div class="label">${escapeHtml(label)}</div></div>`;
}

function clipCard(clip) {
  const thumb = clip.has_thumbnail
    ? `<img loading="lazy" src="/thumb/${clip.id}" alt="">`
    : `<div class="noimg">no thumbnail</div>`;
  const meta = [clip.network, clip.air_year, duration(clip.duration_s)]
    .filter(Boolean).join(" · ");
  return `<article class="clip-card" data-clip="${clip.id}">
    ${thumb}
    <div class="body">
      <div class="title" title="${escapeHtml(clip.title || "")}">${escapeHtml(clip.title || "Untitled")}</div>
      <div class="sub">${escapeHtml(clip.category)} ${clip.confidence != null ? `(${clip.confidence.toFixed(2)})` : ""}</div>
      <div class="sub">${escapeHtml(meta)}</div>
    </div>
  </article>`;
}

function bindCards(container) {
  container.querySelectorAll("[data-clip]").forEach((element) => {
    element.addEventListener("click", () => openClip(element.dataset.clip));
  });
}

/* ------------------------------------------------------------- rendering  */

async function refreshFast() {
  try {
    const [status, queue, system] = await Promise.all([
      api("/api/status"), api("/api/queue"), api("/api/system"),
    ]);
    renderStats(status);
    renderQueue(queue);
    renderSystem(system);
    renderCategories(status);
    $("header-meta").textContent =
      `v${status.version} · ${status.instance} · era ${status.era[0]}–${status.era[1]}`;
  } catch (error) {
    $("header-meta").textContent = `offline — ${error.message}`;
  }
}

async function refreshSlow() {
  try {
    const [clips, events, duplicates, capabilities] = await Promise.all([
      api("/api/clips?limit=24"), api("/api/events?limit=40"),
      api("/api/duplicates?limit=12"), api("/api/capabilities"),
    ]);
    renderRecent(clips);
    renderEvents(events);
    renderDuplicates(duplicates);
    renderCapabilities(capabilities);
  } catch (error) {
    console.warn("slow refresh failed", error);
  }
}

function renderStats(status) {
  const clips = status.clips || {};
  $("stats").innerHTML = [
    stat("clips", clips.total_clips ?? 0),
    stat("unique", clips.unique_clips ?? 0),
    stat("alternates", clips.alternates ?? 0),
    stat("unknown", clips.unknown_clips ?? 0),
    stat("runtime", duration(clips.total_duration_s || 0)),
    stat("on disk", bytes(clips.total_bytes || 0)),
    stat("queued", (status.queue?.pending ?? 0) + (status.queue?.running ?? 0)),
    stat("dead jobs", status.queue?.dead ?? 0),
  ].join("");
}

function renderQueue(queue) {
  const counts = queue.counts || {};
  const pill = $("queue-pill");
  pill.textContent = `${counts.running || 0} running · ${counts.pending || 0} pending`;
  pill.className = "pill " + (counts.dead ? "bad" : counts.running ? "good" : "");

  const running = queue.running || [];
  if (!running.length) {
    $("queue-running").innerHTML = `<div class="empty">idle — nothing in flight</div>`;
  } else {
    $("queue-running").innerHTML = `<table><thead><tr>
        <th>job</th><th>worker</th><th>progress</th></tr></thead><tbody>` +
      running.map((job) => `<tr>
        <td>${escapeHtml(job.kind)} <span class="muted mono">#${job.id}</span><br>
            <span class="muted">${escapeHtml(job.note || "")}</span></td>
        <td class="mono muted">${escapeHtml(job.worker || "")}</td>
        <td style="width:130px">
          <div class="bar"><span style="width:${Math.round((job.progress || 0) * 100)}%"></span></div>
        </td></tr>`).join("") + `</tbody></table>`;
  }

  const failures = queue.failures || [];
  if (failures.length) {
    $("queue-running").innerHTML += `<h2 style="margin-top:14px">Failures</h2>` +
      `<table><tbody>` + failures.slice(0, 8).map((job) => `<tr>
        <td><span class="pill bad">${escapeHtml(job.state)}</span> ${escapeHtml(job.kind)}</td>
        <td class="muted">${escapeHtml((job.error || "").slice(0, 120))}</td>
      </tr>`).join("") + `</tbody></table>`;
  }
}

function renderSystem(system) {
  const gpu = system.gpu || {};
  const rows = [];
  if (system.cpu_percent != null) rows.push(meter("CPU", system.cpu_percent));
  if (system.memory_percent != null) rows.push(meter("Memory", system.memory_percent));
  if (gpu.available) {
    rows.push(meter(`GPU (${gpu.vendor})`, gpu.utilization_pct || 0));
    if (gpu.memory_total_mb) {
      rows.push(meter("GPU memory", (gpu.memory_used_mb / gpu.memory_total_mb) * 100));
    }
  } else {
    rows.push(`<div class="muted" style="margin:6px 0">no GPU detected — CPU only</div>`);
  }
  const storage = system.storage || {};
  rows.push(`<div class="kv" style="margin-top:10px">
    <dt>library</dt><dd>${bytes(storage.library_bytes || 0)}</dd>
    <dt>free space</dt><dd>${storage.free_gb ?? "?"} GB</dd>
    <dt>path</dt><dd class="mono muted">${escapeHtml(storage.path || "")}</dd></div>`);
  $("system").innerHTML = rows.join("");
}

function meter(label, percent) {
  const value = Math.max(0, Math.min(100, percent || 0));
  return `<div style="margin:6px 0">
    <div style="display:flex;justify-content:space-between;font-size:12px">
      <span class="muted">${escapeHtml(label)}</span><span>${value.toFixed(0)}%</span></div>
    <div class="bar"><span style="width:${value}%"></span></div></div>`;
}

function renderCategories(status) {
  const categories = status.clips?.by_category || {};
  const entries = Object.entries(categories).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    $("categories").innerHTML = `<div class="empty">nothing categorized yet</div>`;
    return;
  }
  const max = entries[0][1] || 1;
  $("categories").innerHTML = `<table><tbody>` + entries.map(([name, count]) => `<tr>
      <td><a href="#" data-search="${escapeHtml(name)}">${escapeHtml(name)}</a></td>
      <td style="width:45%"><div class="bar"><span style="width:${(count / max) * 100}%"></span></div></td>
      <td class="num">${count}</td></tr>`).join("") + `</tbody></table>`;

  $("categories").querySelectorAll("[data-search]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      $("q").value = link.dataset.search;
      runSearch();
    });
  });
}

function renderRecent(payload) {
  const clips = payload.clips || [];
  $("recent").innerHTML = clips.length
    ? clips.map(clipCard).join("")
    : `<div class="empty">no clips yet — drop files into Incoming and run <span class="mono">bb run</span></div>`;
  bindCards($("recent"));
}

function renderEvents(payload) {
  const events = payload.events || [];
  $("events").innerHTML = events.length
    ? events.map((event) => `<div class="${escapeHtml(event.level)}">
        <time>${escapeHtml(event.at.slice(11, 19))}</time>
        <strong>${escapeHtml(event.kind)}</strong> ${escapeHtml(event.message)}</div>`).join("")
    : `<div class="empty">no activity recorded yet</div>`;
}

function renderDuplicates(payload) {
  const groups = payload.groups || [];
  const stats = payload.statistics || {};
  if (!groups.length) {
    $("duplicates").innerHTML = `<div class="empty">no duplicates found yet</div>`;
    return;
  }
  $("duplicates").innerHTML =
    `<div class="muted" style="margin-bottom:8px">${stats.duplicate_groups || 0} group(s),
     ${stats.duplicate_members || 0} member(s) — alternates are kept, never deleted</div>` +
    `<table><tbody>` + groups.map((group) => `<tr>
      <td><a href="#" data-clip-link="${group.canonical?.id || ""}">
        ${escapeHtml(group.canonical?.title || group.canonical?.id?.slice(0, 8) || "?")}</a></td>
      <td class="muted">${escapeHtml(group.method)}</td>
      <td class="num">${group.members}</td></tr>`).join("") + `</tbody></table>`;

  $("duplicates").querySelectorAll("[data-clip-link]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      if (link.dataset.clipLink) openClip(link.dataset.clipLink);
    });
  });
}

function renderCapabilities(payload) {
  const analyzers = payload.analyzers || [];
  const missing = (payload.backends || []).filter((b) => !b.available);
  $("capabilities").innerHTML = `<table><thead><tr>
      <th>analyzer</th><th>version</th><th>backend</th><th>what it does</th></tr></thead><tbody>` +
    analyzers.map((a) => `<tr>
      <td>${escapeHtml(a.name)}</td>
      <td class="mono muted">${escapeHtml(a.version)}</td>
      <td><span class="pill ${a.available ? "good" : "warn"}">${a.available ? "ready" : "not installed"}</span></td>
      <td class="muted">${escapeHtml(a.description || "")}</td></tr>`).join("") +
    `</tbody></table>` +
    (missing.length
      ? `<div class="muted" style="margin-top:10px">${missing.length} backend(s) missing —
         clips are recorded as <span class="mono">skipped</span>, and
         <span class="mono">bb reanalyze --missing</span> backfills them once installed.</div>`
      : "");
}

/* ------------------------------------------------------------------ search */

async function runSearch() {
  const query = $("q").value.trim();
  if (!query) { $("panel-results").hidden = true; return; }
  $("panel-results").hidden = false;
  $("results").innerHTML = `<div class="empty">searching…</div>`;
  try {
    const payload = await api(`/api/search?q=${encodeURIComponent(query)}&limit=48`);
    $("search-explain").textContent = payload.explanation || "";
    $("results").innerHTML = payload.hits.length
      ? payload.hits.map((hit) => clipCard({
          id: hit.clip_id, title: hit.title, category: hit.category,
          confidence: hit.confidence, network: hit.network,
          air_year: hit.air_year, duration_s: hit.duration_s,
          has_thumbnail: Boolean(hit.thumbnail),
        })).join("")
      : `<div class="empty">no matches</div>`;
    bindCards($("results"));
  } catch (error) {
    $("results").innerHTML = `<div class="empty">search failed: ${escapeHtml(error.message)}</div>`;
  }
}

/* ------------------------------------------------------------ clip dialog */

let currentClipId = null;

async function openClip(clipId) {
  currentClipId = clipId;
  const dialog = $("clip-dialog");
  $("dlg-content").innerHTML = `<div class="empty">loading…</div>`;
  dialog.showModal();
  try {
    const clip = await api(`/api/clip/${clipId}`);
    $("dlg-title").textContent = clip.title || clipId.slice(0, 8);
    $("dlg-content").innerHTML = renderClipDetail(clip);
  } catch (error) {
    $("dlg-content").innerHTML = `<div class="empty">could not load clip: ${escapeHtml(error.message)}</div>`;
  }
}

function renderClipDetail(clip) {
  const evidence = (clip.evidence || []).map((item) =>
    `<div class="evidence">${escapeHtml(item.category)} +${Number(item.weight).toFixed(2)} — ${escapeHtml(item.reason)}</div>`
  ).join("") || `<div class="muted">no classification evidence recorded</div>`;

  const analyses = (clip.analyses || []).map((a) =>
    `<tr><td>${escapeHtml(a.analyzer)}</td>
     <td><span class="pill ${a.status === "ok" ? "good" : a.status === "skipped" ? "warn" : "bad"}">${escapeHtml(a.status)}</span></td>
     <td class="mono muted">${escapeHtml(a.version)}</td>
     <td class="muted">${escapeHtml(a.error || a.backend || "")}</td></tr>`
  ).join("");

  const entities = (clip.entities || []).map((e) =>
    `<span class="pill">${escapeHtml(e.name)} <span class="muted">${escapeHtml(e.kind)}</span></span>`
  ).join(" ") || `<span class="muted">none</span>`;

  const tags = (clip.tags || []).map((t) => `<span class="pill">${escapeHtml(t)}</span>`).join(" ")
    || `<span class="muted">none</span>`;

  return `
    ${clip.has_media ? `<video controls preload="metadata" src="/media/${clip.id}"></video>` : ""}
    <dl class="kv" style="margin-top:14px">
      <dt>category</dt><dd>${escapeHtml(clip.category)} ${clip.subcategory ? `/ ${escapeHtml(clip.subcategory)}` : ""}
        <span class="muted">(confidence ${Number(clip.confidence).toFixed(2)})</span></dd>
      <dt>duration</dt><dd>${duration(clip.duration_s)}</dd>
      <dt>network</dt><dd>${escapeHtml(clip.network || "—")}</dd>
      <dt>year</dt><dd>${escapeHtml(clip.air_year || "—")}</dd>
      <dt>resolution</dt><dd>${clip.width || "?"}×${clip.height || "?"} · quality ${Number(clip.quality_score).toFixed(2)}</dd>
      <dt>id</dt><dd class="mono muted">${escapeHtml(clip.id)}</dd>
    </dl>
    ${clip.summary ? `<p style="margin-top:14px">${escapeHtml(clip.summary)}</p>` : ""}
    <h2 style="margin-top:16px">Tags</h2><div>${tags}</div>
    <h2 style="margin-top:16px">Entities</h2><div>${entities}</div>
    <h2 style="margin-top:16px">Why this classification</h2>${evidence}
    <h2 style="margin-top:16px">Analyzers</h2>
    <table><tbody>${analyses}</tbody></table>
    ${clip.transcript ? `<h2 style="margin-top:16px">Transcript</h2>
      <div class="text-block">${escapeHtml(clip.transcript)}</div>` : ""}
    ${clip.ocr ? `<h2 style="margin-top:16px">On-screen text</h2>
      <div class="text-block">${escapeHtml(clip.ocr)}</div>` : ""}
  `;
}

/* -------------------------------------------------------------- bootstrap */

$("search-btn").addEventListener("click", runSearch);
$("q").addEventListener("keydown", (event) => { if (event.key === "Enter") runSearch(); });
$("dlg-close").addEventListener("click", () => $("clip-dialog").close());

$("dlg-reanalyze").addEventListener("click", async () => {
  if (!currentClipId) return;
  const button = $("dlg-reanalyze");
  button.disabled = true;
  button.textContent = "queued";
  try {
    await api(`/api/clip/${currentClipId}/reanalyze`, { method: "POST" });
  } finally {
    setTimeout(() => { button.disabled = false; button.textContent = "Re-run AI"; }, 2500);
  }
});

$("dlg-similar").addEventListener("click", async () => {
  if (!currentClipId) return;
  $("clip-dialog").close();
  $("q").value = `similar to ${currentClipId}`;
  runSearch();
});

refreshFast();
refreshSlow();
setInterval(refreshFast, REFRESH_MS);
setInterval(refreshSlow, SLOW_REFRESH_MS);
