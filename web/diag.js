// Что реально умеет Safari на этом телефоне. Ничего не отправляет на сервер, кроме проверки связи.

const out = document.getElementById("out");
const lines = [];

function log(key, value) { lines.push(`${key}: ${value}`); out.textContent = lines.join("\n"); }
function section(title) { lines.push("", `== ${title} ==`); out.textContent = lines.join("\n"); }
const json = (value) => JSON.stringify(value);

const RESOLUTIONS = [[3840, 2160], [1920, 1080], [1280, 720]];

async function probeCamera(facing) {
  section(`Камера: ${facing === "environment" ? "основная" : "фронтальная"}`);
  for (const [width, height] of RESOLUTIONS) {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { exact: facing }, width: { ideal: width }, height: { ideal: height } },
      });
    } catch (error) {
      log(`запрос ${width}x${height}`, `ошибка ${error.name} (с точной камерой)`);
      try { // запасной вариант: камера «желательно», как в основном приложении
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: facing }, width: { ideal: width }, height: { ideal: height } },
        });
      } catch (retry) {
        log(`запрос ${width}x${height} (запасной)`, `ошибка ${retry.name}`);
        continue;
      }
    }
    const track = stream.getVideoTracks()[0];
    const s = track.getSettings();
    log(`запрос ${width}x${height}`, `получено ${s.width}x${s.height}, ${s.frameRate || "?"} к/с`);
    if (width === RESOLUTIONS[0][0]) {
      const caps = track.getCapabilities ? track.getCapabilities() : {};
      log("возможности трека (ключи)", Object.keys(caps).join(", ") || "нет getCapabilities");
      log("фонарик (torch)", "torch" in caps ? "есть" : "нет");
      log("зум (zoom)", "zoom" in caps ? json(caps.zoom) : "нет");
      log("фокус (focusMode/focusDistance)", `${json(caps.focusMode || null)} / ${json(caps.focusDistance || null)}`);
    }
    stream.getTracks().forEach((t) => t.stop());
    await new Promise((resolve) => setTimeout(resolve, 300)); // iOS любит паузу между открытиями камеры
  }
}

async function run() {
  lines.length = 0;
  section("Устройство");
  log("браузер", navigator.userAgent);
  log("безопасный контекст (HTTPS)", window.isSecureContext);
  log("экран", `${screen.width}x${screen.height}, масштаб ${window.devicePixelRatio}`);
  log("ядер процессора", navigator.hardwareConcurrency || "?");

  section("Возможности браузера");
  log("getUserMedia", !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia));
  log("ImageCapture", "ImageCapture" in window);
  log("MediaRecorder", "MediaRecorder" in window);
  log("WebGPU", "gpu" in navigator);
  log("OffscreenCanvas", "OffscreenCanvas" in window);
  log("WebSocket", "WebSocket" in window);
  log("Wake Lock (экран не гаснет)", "wakeLock" in navigator);
  log("вибрация", "vibrate" in navigator);
  const synth = "speechSynthesis" in window;
  log("голос (speechSynthesis)", synth);
  if (synth) {
    const voices = speechSynthesis.getVoices().filter((v) => v.lang.toLowerCase().startsWith("ru"));
    log("русских голосов", voices.length ? voices.map((v) => v.name).join(", ") : "0 (список может подгрузиться позже)");
  }
  if (navigator.mediaDevices && navigator.mediaDevices.getSupportedConstraints) {
    log("поддерживаемые ограничения камеры", Object.keys(navigator.mediaDevices.getSupportedConstraints()).join(", "));
  }

  section("Связь с сервером");
  try {
    const started = performance.now();
    const response = await fetch("/api/health", { cache: "no-store" });
    log("/api/health", `${response.status}, ${Math.round(performance.now() - started)} мс`);
  } catch (error) {
    log("/api/health", `ошибка ${error.message}`);
  }

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    await probeCamera("environment");
    await probeCamera("user");
  }
  section("Готово");
}

document.getElementById("run").addEventListener("click", run);
document.getElementById("copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(out.textContent);
  } catch (_) {
    const range = document.createRange();
    range.selectNodeContents(out);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
  }
});
