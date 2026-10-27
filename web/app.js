// Клиент сканирования: камера -> поток маленьких кадров на сервер -> подсказки и команды «снимай».
// Полные кадры уходят отдельным запросом, когда сервер сказал, что кадр хорош.

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"),
  title: $("step-title"),
  text: $("step-text"),
  hint: $("hint"),
  fill: $("meter-fill"),
  start: $("start"),
  voice: $("voice"),
  flip: $("flip"),
  force: $("force"),
  skip: $("skip"),
  status: $("status"),
};

const STREAM_WIDTH = 640;
const STREAM_INTERVAL_MS = 200;
const HINT_REPEAT_MS = 8000;

const state = {
  facing: "environment", // основная камера; переключается кнопкой
  stream: null,
  socket: null,
  scanId: "",
  timer: null,
  inFlight: false,
  uploading: false,
  stepKey: "",
  audio: null,
  wake: null,
  spoken: new Map(),
};

const streamCanvas = document.createElement("canvas");

function loadToken() {
  const url = new URL(location.href);
  const fromUrl = url.searchParams.get("token");
  if (fromUrl) {
    try { localStorage.setItem("stylist.token", fromUrl); } catch (_) { /* приватный режим */ }
    url.searchParams.delete("token");
    history.replaceState(null, "", url.pathname + url.search + url.hash);
    return fromUrl;
  }
  try { return localStorage.getItem("stylist.token") || ""; } catch (_) { return ""; }
}
const TOKEN = loadToken();

function api(path, options = {}) {
  const headers = { Authorization: `Bearer ${TOKEN}`, ...(options.headers || {}) };
  return fetch(path, { ...options, headers });
}

function setStatus(text) { els.status.textContent = text; }

// ---- голос и звук --------------------------------------------------------

function unlockAudio() {
  try {
    state.audio = state.audio || new (window.AudioContext || window.webkitAudioContext)();
    state.audio.resume();
  } catch (_) { /* звук не обязателен */ }
}

function beep() {
  if (!state.audio) return;
  const osc = state.audio.createOscillator();
  const gain = state.audio.createGain();
  osc.frequency.value = 880;
  gain.gain.value = 0.2;
  osc.connect(gain).connect(state.audio.destination);
  osc.start();
  osc.stop(state.audio.currentTime + 0.15);
}

function say(text, { urgent = false } = {}) {
  if (!els.voice.checked || !text || !("speechSynthesis" in window)) return;
  const synth = window.speechSynthesis;
  if (urgent) {
    synth.cancel();
  } else {
    if (synth.speaking) return;
    const last = state.spoken.get(text) || 0;
    if (Date.now() - last < HINT_REPEAT_MS) return;
  }
  state.spoken.set(text, Date.now());
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "ru-RU";
  synth.speak(utterance);
}

// ---- камера --------------------------------------------------------------

async function startCamera() {
  stopCamera();
  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: state.facing }, width: { ideal: 3840 }, height: { ideal: 2160 } },
  });
  els.video.srcObject = state.stream;
  els.video.classList.toggle("mirrored", state.facing === "user");
  await els.video.play();
}

function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
}

async function requestWake() {
  try { state.wake = await navigator.wakeLock.request("screen"); } catch (_) { /* не везде есть */ }
}

// ---- сеанс ---------------------------------------------------------------

async function startScan() {
  els.start.disabled = true;
  setStatus("Включаю камеру…");
  try {
    unlockAudio();
    await startCamera();
    const response = await api("/api/scan", { method: "POST" });
    if (response.status === 401) throw new Error("Сервер не принял токен: откройте ссылку с ?token=… из терминала");
    if (!response.ok) throw new Error(`Сервер ответил ${response.status}`);
    state.scanId = (await response.json()).scan_id;
    openSocket();
    requestWake();
  } catch (error) {
    els.start.disabled = false;
    setStatus(explain(error));
  }
}

function explain(error) {
  if (error && error.name === "NotAllowedError") return "Нет доступа к камере: разрешите её в настройках Safari для этого сайта.";
  return error && error.message ? error.message : String(error);
}

function openSocket() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${scheme}//${location.host}/ws/scan/${state.scanId}?token=${encodeURIComponent(TOKEN)}`);
  state.socket = socket;
  socket.onopen = () => {
    setStatus("Связь с сервером есть");
    els.force.hidden = false;
    els.skip.hidden = false;
    state.timer = setInterval(sendFrame, STREAM_INTERVAL_MS);
  };
  socket.onmessage = (event) => onMessage(JSON.parse(event.data));
  socket.onclose = () => {
    clearInterval(state.timer);
    if (state.socket === socket) setStatus("Связь с сервером потеряна. Нажмите «Начать скан» ещё раз.");
    els.start.disabled = false;
    els.force.hidden = true;
    els.skip.hidden = true;
  };
}

function sendFrame() {
  const socket = state.socket;
  const video = els.video;
  if (state.inFlight || state.uploading || !socket || socket.readyState !== WebSocket.OPEN || !video.videoWidth) return;
  streamCanvas.width = STREAM_WIDTH;
  streamCanvas.height = Math.round((STREAM_WIDTH * video.videoHeight) / video.videoWidth);
  streamCanvas.getContext("2d").drawImage(video, 0, 0, streamCanvas.width, streamCanvas.height);
  state.inFlight = true;
  streamCanvas.toBlob((blob) => {
    if (blob && socket.readyState === WebSocket.OPEN) socket.send(blob);
    else state.inFlight = false;
  }, "image/jpeg", 0.7);
}

function onMessage(message) {
  state.inFlight = false;
  if (message.type === "error") { setStatus(message.message); return; }
  render(message);
  if (message.capture && message.step) capture(message.step);
}

function render(message) {
  if (message.done) {
    els.title.textContent = "Скан завершён";
    els.text.textContent = "Все шаги пройдены. Снимки лежат на сервере в папке data/scans.";
    els.hint.textContent = "";
    els.fill.style.width = "100%";
    say("Скан завершён", { urgent: true });
    finish();
    return;
  }
  const step = message.step;
  if (step.key !== state.stepKey) {
    state.stepKey = step.key;
    els.title.textContent = `${step.index + 1} из ${step.total}. ${step.title}`;
    els.text.textContent = step.instruction;
    say(`${step.title}. ${step.instruction}`, { urgent: true });
  }
  const tip = message.hints[0];
  els.hint.textContent = tip ? tip.text : message.waiting ? "Снимаю…" : message.steady > 0 ? "Хорошо, держите" : "";
  els.fill.style.width = `${Math.min(100, (message.steady / message.need) * 100)}%`;
  if (tip) say(tip.text);
}

function finish() {
  clearInterval(state.timer);
  if (state.socket) state.socket.close();
  state.socket = null;
  stopCamera();
  els.start.disabled = false;
  els.force.hidden = true;
  els.skip.hidden = true;
}

// ---- полные кадры --------------------------------------------------------

function grabFull() {
  const video = els.video;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.95));
}

async function upload(stepKey, blob, frame) {
  const response = await api(`/api/scan/${state.scanId}/${stepKey}?frame=${frame}`, {
    method: "PUT",
    body: blob,
    headers: { "Content-Type": "image/jpeg" },
  });
  if (!response.ok) throw new Error(`Кадр не принят (${response.status})`);
  return response.json();
}

async function capture(step) {
  state.uploading = true;
  beep();
  try {
    for (let frame = 0; frame < step.frames; frame += 1) {
      const blob = await grabFull();
      setStatus(step.frames > 1 ? `Кадр ${frame + 1} из ${step.frames}` : "Отправляю снимок…");
      await upload(step.key, blob, frame);
      if (frame + 1 < step.frames) await new Promise((resolve) => setTimeout(resolve, step.interval * 1000));
    }
    setStatus("Снимок принят");
  } catch (error) {
    setStatus(explain(error));
  } finally {
    state.uploading = false;
  }
}

// ---- кнопки --------------------------------------------------------------

els.start.addEventListener("click", startScan);
els.skip.addEventListener("click", () => state.socket && state.socket.send(JSON.stringify({ cmd: "skip" })));
els.force.addEventListener("click", () => state.socket && state.socket.send(JSON.stringify({ cmd: "force" })));
els.flip.addEventListener("click", async () => {
  state.facing = state.facing === "environment" ? "user" : "environment";
  els.flip.textContent = `Камера: ${state.facing === "environment" ? "основная" : "фронтальная"}`;
  if (state.stream) {
    try { await startCamera(); } catch (error) { setStatus(explain(error)); }
  }
});
