"use strict";

// ---------- api ----------

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", "X-Jarvis-Studio": "1" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty body */ }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

const $ = (id) => document.getElementById(id);
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, "");
    else if (v !== false && v != null) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid);
  return n;
}

let toastTimer;
function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2400);
}

let meta = { media_keys: {}, builtins: [], reserved: [] };
let live = {};
let gestures = [];

// ---------- live hand ----------

const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];
const view = { yaw: 0, pitch: 0, drag: null };
const canvas = $("hand");
const ctx = canvas.getContext("2d");
let dirty = true;

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function fitCanvas() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  dirty = true;
}

function project(p, w, h) {
  // hand_shape: metres, hand-centred, x right / y down / z away in the camera
  // image. Mirror x so it moves like a mirror, then orbit by the drag angles.
  let [x, y, z] = [-p[0], p[1], p[2]];
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  [x, z] = [x * cy + z * sy, -x * sy + z * cy];
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  [y, z] = [y * cp - z * sp, y * sp + z * cp];
  const scale = Math.min(w, h) / 0.24;
  const persp = 0.6 / (0.6 + z);
  return [w / 2 + x * scale * persp, h / 2 + y * scale * persp, z];
}

function draw() {
  if (!dirty) return requestAnimationFrame(draw);
  dirty = false;
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const hand = live.hand;
  $("no-hand").hidden = !!hand;
  if (hand) {
    const on = !!live.active;
    const color = on ? cssVar("--signal") : cssVar("--bone");
    const pts = hand.map((p) => project(p, w, h));
    const unit = Math.min(w, h) / 160;
    ctx.lineCap = "round";
    ctx.strokeStyle = color;
    for (const [a, b] of BONES) {
      // Nearer bones (smaller z) draw more solid, a cheap depth cue.
      ctx.globalAlpha = 0.7 + 0.3 * Math.max(0, Math.min(1, 0.5 - (pts[a][2] + pts[b][2]) * 6));
      ctx.lineWidth = unit * 1.8;
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    pts.forEach(([x, y, z], i) => {
      const r = unit * (i % 4 === 0 ? 2.6 : 2.0) * (0.6 / (0.6 + z));
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  requestAnimationFrame(draw);
}

canvas.addEventListener("pointerdown", (e) => {
  view.drag = { x: e.clientX, y: e.clientY, yaw: view.yaw, pitch: view.pitch };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!view.drag) return;
  view.yaw = view.drag.yaw + (e.clientX - view.drag.x) / 120;
  view.pitch = Math.max(-1.4, Math.min(1.4, view.drag.pitch + (e.clientY - view.drag.y) / 120));
  dirty = true;
});
canvas.addEventListener("pointerup", () => { view.drag = null; });
canvas.addEventListener("dblclick", () => { view.yaw = 0; view.pitch = 0; dirty = true; });

// ---------- live readouts ----------

let recordingSeconds = 1.5;

function renderLive() {
  dirty = true;
  const name = $("active");
  const sub = $("active-sub");
  if (live.active) {
    name.textContent = live.active;
    name.classList.add("on");
  } else {
    name.textContent = live.pinch === "dragging" ? "Dragging" : live.pinch === "pending" ? "Pinch" : "No gesture";
    name.classList.toggle("on", live.pinch === "dragging" || live.pinch === "pending");
  }
  if (live.recording) sub.textContent = "Cursor and gestures are paused while recording";
  else if (live.active === "fist") sub.textContent = live.scroll ? "Tilt to scroll" : "Scroll mode";
  else if (live.candidate && live.candidate !== live.active) sub.textContent = `Hold for ${live.candidate}`;
  else sub.innerHTML = "&nbsp;";

  const tilt = $("tilt");
  tilt.hidden = !live.scroll;
  if (live.scroll) {
    const p = live.scroll.pitch;
    $("tilt-dot").style.left = `${50 + Math.max(-1, Math.min(1, p / 40)) * 50}%`;
    $("tilt-deg").textContent = `${p >= 0 ? "+" : ""}${p.toFixed(0)}°`;
  }

  const meters = $("meters");
  const scores = live.scores || {};
  const names = Object.keys(scores).sort();
  meters.replaceChildren(...names.map((n) => {
    const s = scores[n];
    return el("div", { class: `meter${live.candidate === n ? " lead" : ""}` },
      el("span", { class: "meter-name", title: n }, n),
      el("div", { class: "meter-track" }, el("div", { class: "meter-fill", style: `width:${Math.round(s * 100)}%` })),
      el("span", { class: "meter-val" }, `${Math.round(s * 100)}`));
  }));

  const starts = (live.events || []).filter((e) => e.phase === "start").slice(-8);
  $("events").replaceChildren(...starts.map((e) => el("li", {}, e.name)));

  renderOverlay();
  document.querySelectorAll(".g").forEach((row) => {
    row.classList.toggle("is-active", row.dataset.name === live.active);
  });
}

function renderOverlay() {
  const rec = live.recording;
  const ov = $("overlay");
  ov.hidden = !rec;
  if (!rec) return;
  ov.classList.toggle("rec", rec.phase !== "countdown");
  if (rec.phase === "countdown") {
    $("overlay-num").textContent = Math.max(1, Math.ceil(rec.remaining));
    $("overlay-text").textContent = rec.kind === "take" ? "Get into the gesture" : "Get ready to move normally";
    $("overlay-fill").style.width = "0%";
  } else {
    $("overlay-num").textContent = "";
    $("overlay-text").textContent = rec.kind === "take" ? "Hold it" : "Move like you normally would";
    const done = 1 - rec.remaining / recordingSeconds;
    $("overlay-fill").style.width = `${Math.round(Math.max(0, Math.min(1, done)) * 100)}%`;
  }
}

function connect() {
  const es = new EventSource("/api/stream");
  es.onmessage = (m) => {
    $("conn").hidden = true;
    const prevActive = live.active;
    live = JSON.parse(m.data);
    renderLive();
    if (prevActive !== live.active) dirty = true;
  };
  es.onerror = () => { $("conn").hidden = false; };
}

// ---------- binding editor ----------

const ACTION_TYPES = [
  ["media", "Media"], ["keys", "Keys"], ["text", "Type text"],
  ["shell", "Shell"], ["macro", "Macro"], ["none", "Nothing"],
];
const STEP_TYPES = [["keys", "Keys"], ["media", "Media"], ["text", "Type text"], ["shell", "Shell"], ["wait", "Wait"]];

const CODE_KEYS = {
  Space: "space", Enter: "return", Tab: "tab", Escape: "esc", Backspace: "delete",
  Delete: "forwarddelete", ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
  Home: "home", End: "end", PageUp: "pageup", PageDown: "pagedown",
  Minus: "-", Equal: "=", BracketLeft: "[", BracketRight: "]", Backslash: "\\",
  Semicolon: ";", Quote: "'", Comma: ",", Period: ".", Slash: "/", Backquote: "`",
};

function comboFromEvent(e) {
  let key = null;
  if (/^Key[A-Z]$/.test(e.code)) key = e.code.slice(3).toLowerCase();
  else if (/^Digit\d$/.test(e.code)) key = e.code.slice(5);
  else if (/^F\d{1,2}$/.test(e.code)) key = e.code.toLowerCase();
  else key = CODE_KEYS[e.code] || null;
  if (!key) return null;
  const mods = [];
  if (e.metaKey) mods.push("cmd");
  if (e.ctrlKey) mods.push("ctrl");
  if (e.altKey) mods.push("option");
  if (e.shiftKey) mods.push("shift");
  return [...mods, key].join("+");
}

function comboInput(value, onInput) {
  // Pressing a combo with a modifier captures it; plain typing also works,
  // for combos the system takes first (cmd+space never reaches the page).
  const input = el("input", {
    class: "combo", value: value || "", placeholder: "Press keys, or type cmd+shift+4",
    "aria-label": "Key combo", autocomplete: "off", spellcheck: "false",
  });
  input.addEventListener("keydown", (e) => {
    if (!(e.metaKey || e.ctrlKey || e.altKey)) return;
    const combo = comboFromEvent(e);
    if (!combo) return;
    e.preventDefault();
    input.value = combo;
    onInput(combo);
  });
  input.addEventListener("input", () => onInput(input.value.trim()));
  input.addEventListener("focus", () => input.classList.add("capturing"));
  input.addEventListener("blur", () => input.classList.remove("capturing"));
  return input;
}

function mediaSelect(value, onInput) {
  const sel = el("select", { "aria-label": "Media key" },
    Object.entries(meta.media_keys).map(([k, label]) =>
      el("option", { value: k, selected: k === value }, label[0].toUpperCase() + label.slice(1))));
  sel.addEventListener("change", () => onInput(sel.value));
  return sel;
}

function textInput(value, placeholder, onInput, label) {
  const input = el("input", { value: value || "", placeholder, "aria-label": label, autocomplete: "off", spellcheck: "false" });
  input.addEventListener("input", () => onInput(input.value));
  return input;
}

function defaultFor(type) {
  return {
    media: { type: "media", key: "play_pause" },
    keys: { type: "keys", combo: "" },
    text: { type: "text", text: "" },
    shell: { type: "shell", command: "" },
    macro: { type: "macro", steps: [{ type: "keys", combo: "" }] },
    none: { type: "none" },
    wait: { type: "wait", seconds: 0.2 },
  }[type];
}

function stepField(step, changed) {
  switch (step.type) {
    case "keys": return comboInput(step.combo, (v) => { step.combo = v; changed(); });
    case "media": return mediaSelect(step.key, (v) => { step.key = v; changed(); });
    case "text": return textInput(step.text, "Text to type", (v) => { step.text = v; changed(); }, "Text to type");
    case "shell": return textInput(step.command, "jarvis-dj next", (v) => { step.command = v; changed(); }, "Shell command");
    case "wait": {
      const n = el("input", { type: "number", min: "0", max: "10", step: "0.1", value: step.seconds, "aria-label": "Seconds to wait" });
      n.addEventListener("input", () => { step.seconds = parseFloat(n.value); changed(); });
      return n;
    }
  }
  return el("span");
}

function bindingEditor(root, initial) {
  let spec = initial && initial.type ? JSON.parse(JSON.stringify(initial)) : defaultFor("media");
  let repeat = spec.repeat_s || 0;
  delete spec.repeat_s;
  let errors = [];
  let timer;

  const errBox = el("div", { class: "be-errors", role: "alert" });

  function current() {
    const out = JSON.parse(JSON.stringify(spec));
    if (repeat && !["macro", "none"].includes(spec.type)) out.repeat_s = repeat;
    return out;
  }

  function validateSoon() {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      try {
        const r = await api("POST", "/api/actions/validate", { action: current() });
        errors = r.errors;
      } catch (e) { errors = [e.message]; }
      errBox.textContent = errors.join("; ");
    }, 250);
  }

  function fields() {
    switch (spec.type) {
      case "media":
        return [el("div", { class: "be-field" }, el("label", {}, "Key"), mediaSelect(spec.key, (v) => { spec.key = v; validateSoon(); }))];
      case "keys":
        return [el("div", { class: "be-field" }, el("label", {}, "Key combo"), comboInput(spec.combo, (v) => { spec.combo = v; validateSoon(); }))];
      case "text":
        return [el("div", { class: "be-field" }, el("label", {}, "Text"), textInput(spec.text, "on my way", (v) => { spec.text = v; validateSoon(); }, "Text to type"))];
      case "shell":
        return [el("div", { class: "be-field" }, el("label", {}, "Command (runs in your shell, as you)"),
          textInput(spec.command, "jarvis-dj next", (v) => { spec.command = v; validateSoon(); }, "Shell command"))];
      case "macro": {
        const list = el("ol", { class: "macro" });
        const redraw = () => {
          list.replaceChildren(...spec.steps.map((step, i) => {
            const typeSel = el("select", { "aria-label": `Step ${i + 1} type` },
              STEP_TYPES.map(([v, l]) => el("option", { value: v, selected: v === step.type }, l)));
            typeSel.addEventListener("change", () => { spec.steps[i] = defaultFor(typeSel.value); redraw(); validateSoon(); });
            const remove = el("button", { type: "button", class: "quiet", "aria-label": `Remove step ${i + 1}`,
              onclick: () => { spec.steps.splice(i, 1); redraw(); validateSoon(); } }, "Remove");
            return el("li", {}, typeSel, stepField(step, validateSoon), remove);
          }));
        };
        redraw();
        const add = el("button", { type: "button", onclick: () => { spec.steps.push(defaultFor("keys")); redraw(); validateSoon(); } }, "Add step");
        return [list, el("div", {}, add)];
      }
      case "none":
        return [el("p", { class: "muted" }, "This gesture does nothing when you make it.")];
    }
    return [];
  }

  function render() {
    const seg = el("div", { class: "seg", role: "group", "aria-label": "Action type" },
      ACTION_TYPES.map(([t, label]) => el("button", {
        type: "button", "aria-pressed": String(spec.type === t),
        onclick: () => { if (spec.type !== t) { spec = defaultFor(t); render(); validateSoon(); } },
      }, label)));
    const repeatBox = el("input", { type: "checkbox", checked: !!repeat, id: `rep-${Math.random().toString(36).slice(2)}` });
    const repeatS = el("input", { type: "number", min: "0.05", step: "0.05", value: repeat || 0.3, "aria-label": "Repeat every seconds" });
    repeatBox.addEventListener("change", () => { repeat = repeatBox.checked ? parseFloat(repeatS.value) : 0; validateSoon(); });
    repeatS.addEventListener("input", () => { if (repeatBox.checked) { repeat = parseFloat(repeatS.value); validateSoon(); } });
    const repeatRow = ["macro", "none"].includes(spec.type) ? null :
      el("label", { class: "be-repeat", for: repeatBox.id }, repeatBox, "Repeat while held, every", repeatS, "s");
    const tryBtn = spec.type === "none" ? null : el("button", {
      type: "button",
      onclick: async () => {
        try { await api("POST", "/api/actions/test", { action: current() }); toast("Ran it"); }
        catch (e) { errBox.textContent = e.message; }
      },
    }, "Try it");
    root.replaceChildren(el("div", { class: "be" }, seg, ...fields(), repeatRow, el("div", { class: "row" }, tryBtn), errBox));
  }

  render();
  validateSoon();
  return { get: current, hasErrors: () => errors.length > 0 };
}

// ---------- gesture list ----------

async function loadGestures() {
  const r = await api("GET", "/api/gestures");
  gestures = r.gestures;
  renderList();
}

const CHECK_NAMES = { recognizable: "Recognizable", repeatable: "Repeatable", distinct: "Distinct" };

function renderList() {
  $("gesture-list").replaceChildren(...gestures.map((g) => {
    const actions = el("div", { class: "g-actions" });
    if (!g.reserved) actions.append(el("button", { onclick: () => openBind(g) }, "Bind"));
    if (g.kind === "custom") {
      actions.append(el("button", { onclick: () => startAdd(g.name, true) }, "Retrain"));
      const del = el("button", { class: "danger" }, "Delete");
      let armed = false;
      del.addEventListener("click", async () => {
        if (!armed) {
          armed = true;
          del.textContent = "Confirm delete";
          setTimeout(() => { armed = false; del.textContent = "Delete"; }, 3000);
          return;
        }
        try { await api("DELETE", `/api/gestures/${encodeURIComponent(g.name)}`); toast(`Deleted ${g.name}`); loadGestures(); }
        catch (e) { toast(e.message); }
      });
      actions.append(del);
    }
    const quality = g.kind === "custom" && Object.keys(g.checks).length
      ? el("div", { class: "g-quality" }, Object.entries(CHECK_NAMES).map(([k, label]) =>
          el("span", { class: `v ${(g.checks[k] || {}).verdict || ""}`, title: `${label}: ${(g.checks[k] || {}).score ?? "?"} / 100` }, label)))
      : null;
    return el("li", { class: `g${g.name === live.active ? " is-active" : ""}`, "data-name": g.name },
      el("div", { class: "g-name" }, g.name, el("span", { class: "g-kind" }, g.kind === "custom" ? "custom" : "built in")),
      g.info ? el("p", { class: "g-info" }, g.info) : null,
      el("p", { class: "g-bind" }, g.reserved ? "Does " : "Bound to ", el("b", {}, g.binding_text)),
      quality,
      actions);
  }));
}

let bindTarget = null;
let bindEditor = null;
function openBind(g) {
  bindTarget = g.name;
  $("bind-title").textContent = `When you make ${g.name}`;
  $("bind-error").textContent = "";
  bindEditor = bindingEditor($("bind-editor"), g.binding);
  $("bind-dialog").showModal();
}
$("bind-save").addEventListener("click", async (e) => {
  e.preventDefault();
  try {
    await api("POST", `/api/gestures/${encodeURIComponent(bindTarget)}/binding`, { action: bindEditor.get() });
    $("bind-dialog").close();
    toast(`Saved binding for ${bindTarget}`);
    loadGestures();
  } catch (err) { $("bind-error").textContent = err.message; }
});

// ---------- add flow ----------

const add = { name: null, replace: false, takes: [], background: null, report: null, busy: false, editor: null };

function showView(which) {
  $("view-list").hidden = which !== "list";
  $("view-add").hidden = which !== "add";
}

function startAdd(name, replace) {
  Object.assign(add, { name: null, replace: !!replace, takes: [], background: null, report: null, busy: false });
  $("name-input").value = name || "";
  $("name-error").textContent = "";
  $("report").replaceChildren();
  $("save-error").textContent = "";
  $("bg-status").textContent = "";
  $("add-title").textContent = replace ? `Retrain ${name}` : "New gesture";
  const existing = replace ? gestures.find((g) => g.name === name) : null;
  add.editor = bindingEditor($("add-binding"), existing && existing.binding);
  showView("add");
  if (replace) createCandidate();
  else $("name-input").focus();
  renderSteps();
}

async function createCandidate() {
  const name = $("name-input").value.trim();
  try {
    await api("POST", "/api/candidate", { name, replace: add.replace });
    add.name = name;
    $("name-error").textContent = "";
    $("name-input").disabled = true;
    renderSteps();
  } catch (e) { $("name-error").textContent = e.message; }
}

function renderSteps() {
  const enough = add.takes.length >= 3;
  const reached = !add.name ? 1 : !enough ? 2 : !add.report ? 4 : add.report.verdict === "fail" ? 4 : 5;
  document.querySelectorAll(".step").forEach((s) => {
    const n = Number(s.dataset.step);
    const open = n <= reached || (n === 3 && enough);
    s.classList.toggle("locked", !open);
    s.classList.toggle("current", n === reached);
    s.classList.toggle("done", n < reached && open);
  });
  $("name-input").disabled = !!add.name;
  $("name-form").querySelector("button").hidden = !!add.name;
  $("take-btn").disabled = add.busy || !add.name;
  $("bg-btn").disabled = add.busy || !add.name;
  $("check-btn").disabled = add.busy || !enough;
  $("save-btn").disabled = add.busy || !add.report || add.report.verdict === "fail";
  $("save-note").textContent = add.report && add.report.verdict === "warn" ? "It has warnings; you can still save it." : "";
  $("takes").replaceChildren(...add.takes.map((t, i) =>
    el("li", { class: `take${t.tracked < 0.9 ? " low" : ""}` },
      `Take ${i + 1}, ${Math.round(t.tracked * 100)}% tracked`,
      el("button", { class: "quiet", "aria-label": `Delete take ${i + 1}`, onclick: () => deleteTake(i) }, "Redo"))));
}

async function record(kind) {
  if (add.busy) return;
  add.busy = true;
  renderSteps();
  recordingSeconds = kind === "take" ? 1.5 : 10;
  try {
    const r = await api("POST", "/api/candidate/record", { kind, countdown_s: 3, seconds: recordingSeconds });
    if (kind === "take") add.takes.push({ tracked: r.tracked });
    else {
      add.background = { tracked: r.tracked };
      $("bg-status").textContent = `Recorded, ${Math.round(r.tracked * 100)}% tracked.`;
    }
    add.report = null;
    $("report").replaceChildren();
  } catch (e) { toast(e.message); }
  add.busy = false;
  renderSteps();
}

async function deleteTake(i) {
  try {
    await api("DELETE", `/api/candidate/takes/${i + 1}`);
    add.takes.splice(i, 1);
    add.report = null;
    $("report").replaceChildren();
    renderSteps();
  } catch (e) { toast(e.message); }
}

const VERDICT_LINE = {
  pass: "Good to go.",
  warn: "Usable, with something to look at.",
  fail: "Not yet. Fix what's flagged and check again.",
};

async function check() {
  if (add.busy || add.takes.length < 3) return;
  add.busy = true;
  renderSteps();
  try {
    const r = await api("POST", "/api/candidate/assess");
    add.report = r;
    $("report").replaceChildren(
      el("p", { class: `verdict-line ${r.verdict}` }, VERDICT_LINE[r.verdict]),
      ...Object.entries(CHECK_NAMES).map(([k, label]) => {
        const c = r.checks[k];
        return el("div", { class: `check ${c.verdict}` },
          el("div", {}, el("div", { class: "check-name" }, label), el("div", { class: "check-verdict" }, `${c.verdict}, ${c.score} / 100`)),
          el("div", { class: "check-bar" }, el("div", { class: "check-fill", style: `width:${c.score}%` })),
          el("p", { class: "check-text" }, c.summary),
          c.hint ? el("p", { class: "check-hint" }, c.hint) : null);
      }));
  } catch (e) { toast(e.message); }
  add.busy = false;
  renderSteps();
}

async function save() {
  if (add.editor.hasErrors()) { $("save-error").textContent = "Fix the binding first."; return; }
  try {
    const action = add.editor.get();
    await api("POST", "/api/candidate/save", { action });
    toast(`Saved ${add.name}`);
    showView("list");
    loadGestures();
  } catch (e) { $("save-error").textContent = e.message; }
}

$("add-btn").addEventListener("click", () => startAdd("", false));
$("add-cancel").addEventListener("click", () => { showView("list"); loadGestures(); });
$("name-form").addEventListener("submit", (e) => { e.preventDefault(); createCandidate(); });
$("take-btn").addEventListener("click", () => record("take"));
$("bg-btn").addEventListener("click", () => record("background"));
$("check-btn").addEventListener("click", check);
$("save-btn").addEventListener("click", save);

document.addEventListener("keydown", (e) => {
  if ($("view-add").hidden || $("bind-dialog").open) return;
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === "Space" && add.name) { e.preventDefault(); record("take"); }
  else if (e.code === "Enter" && add.takes.length >= 3) { e.preventDefault(); check(); }
});

// ---------- boot ----------

window.addEventListener("resize", fitCanvas);
(async function boot() {
  fitCanvas();
  requestAnimationFrame(draw);
  try { meta = await api("GET", "/api/meta"); } catch (_) { /* stream shows the error */ }
  connect();
  loadGestures().catch((e) => toast(e.message));
})();
