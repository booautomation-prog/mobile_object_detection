const video = document.getElementById("video");
const canvas = document.getElementById("snapshotCanvas");
const cameraEmpty = document.getElementById("cameraEmpty");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const classifyState = document.getElementById("classifyState");
const bestClass = document.getElementById("bestClass");
const bestConfidence = document.getElementById("bestConfidence");
const matchesList = document.getElementById("matchesList");
const fpsText = document.getElementById("fpsText");
const startButton = document.getElementById("startButton");
const switchButton = document.getElementById("switchButton");
const overlaySwitchButton = document.getElementById("overlaySwitchButton");

let model = null;
let stream = null;
let facingMode = "environment";
let classifying = false;
let lastFrameTime = 0;
let frameCounter = 0;
let lastFpsAt = performance.now();

function setStatus(mode, text) {
  statusDot.className = `status-dot ${mode || ""}`.trim();
  statusText.textContent = text;
}

function setClassifyState(mode, text) {
  classifyState.className = `classify-state ${mode || ""}`.trim();
  classifyState.querySelector("span:last-child").textContent = text;
}

function setButtons(cameraOn) {
  startButton.disabled = !model;
  startButton.innerHTML = cameraOn
    ? `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h12a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Zm2 3v8h8V8H8Z"/></svg>Stop Camera`
    : `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5 7.2 7.4H4a2 2 0 0 0-2 2V17a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9.4a2 2 0 0 0-2-2h-3.2L15 5H9Zm3 12a4 4 0 1 1 0-8 4 4 0 0 1 0 8Z"/></svg>Start Camera`;
  switchButton.disabled = !cameraOn;
  overlaySwitchButton.disabled = !cameraOn;
}

function clearPredictions(message = "Predictions will appear here.") {
  bestClass.textContent = stream ? "Looking for object" : "Waiting for camera";
  bestConfidence.textContent = "Confidence: --";
  fpsText.textContent = "-- fps";
  matchesList.innerHTML = `<div class="empty-match">${escapeHtml(message)}</div>`;
}

function renderPredictions(predictions) {
  if (!predictions.length) {
    clearPredictions("No prediction yet.");
    return;
  }

  const [top] = predictions;
  bestClass.textContent = titleCase(top.className.split(",")[0]);
  bestConfidence.textContent = `Confidence: ${formatPercent(top.probability)}`;

  matchesList.innerHTML = predictions
    .map((prediction) => {
      const label = titleCase(prediction.className.split(",")[0]);
      const score = formatPercent(prediction.probability);
      const width = Math.round(prediction.probability * 100);
      return `
        <div class="match-row">
          <span class="match-name" title="${escapeHtml(prediction.className)}">${escapeHtml(label)}</span>
          <span class="match-track" aria-hidden="true">
            <span class="match-fill" style="width: ${width}%"></span>
          </span>
          <span class="match-score">${score}</span>
        </div>
      `;
    })
    .join("");
}

function titleCase(value) {
  return value.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "\"": "&quot;",
      "'": "&#039;",
    };
    return entities[char];
  });
}

async function waitForLibraries() {
  const started = performance.now();
  while ((!window.tf || !window.mobilenet) && performance.now() - started < 15000) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  if (!window.tf || !window.mobilenet) {
    throw new Error("TensorFlow.js or MobileNet did not load. Check internet/CDN access.");
  }
}

async function loadModel() {
  setStatus("busy", "Loading");
  setClassifyState("", "Loading model");
  setButtons(false);

  try {
    await waitForLibraries();
    model = await mobilenet.load({ version: 2, alpha: 1.0 });
    setStatus("", "Ready");
    setClassifyState("ready", "Ready");
    setButtons(false);
  } catch (error) {
    setStatus("error", "Model error");
    setClassifyState("error", "Model not loaded");
    matchesList.innerHTML = `<div class="empty-match">${escapeHtml(error.message)}</div>`;
  }
}

async function startCamera() {
  if (!model) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("error", "No camera API");
    setClassifyState("error", "Camera not supported");
    clearPredictions("This browser does not support camera access.");
    return;
  }

  stopCamera(false);
  setStatus("busy", "Starting");
  setClassifyState("", "Starting camera");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280 },
        height: { ideal: 960 },
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();
    video.classList.add("active");
    cameraEmpty.classList.add("hidden");
    setButtons(true);
    setStatus("busy", "Classifying");
    setClassifyState("", "Classifying");
    clearPredictions("Point the camera at an object.");
    classifying = true;
    classifyLoop();
  } catch (error) {
    stream = null;
    video.classList.remove("active");
    cameraEmpty.classList.remove("hidden");
    setButtons(false);
    setStatus("error", "Camera blocked");
    setClassifyState("error", "Camera blocked");
    clearPredictions(error.message);
  }
}

function stopCamera(resetStatus = true) {
  classifying = false;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  video.classList.remove("active");
  cameraEmpty.classList.remove("hidden");
  setButtons(false);

  if (resetStatus && model) {
    setStatus("", "Ready");
    setClassifyState("ready", "Ready");
    clearPredictions();
  }
}

async function switchCamera() {
  facingMode = facingMode === "environment" ? "user" : "environment";
  await startCamera();
}

async function classifyLoop(now = performance.now()) {
  if (!classifying || !stream || !model) {
    return;
  }

  if (now - lastFrameTime >= 650 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    lastFrameTime = now;
    try {
      const predictions = await model.classify(video, 5);
      renderPredictions(predictions);
      updateFps();
    } catch (error) {
      setStatus("error", "Classify error");
      setClassifyState("error", "Classify error");
      matchesList.innerHTML = `<div class="empty-match">${escapeHtml(error.message)}</div>`;
    }
  }

  requestAnimationFrame(classifyLoop);
}

function updateFps() {
  frameCounter += 1;
  const now = performance.now();
  const elapsed = now - lastFpsAt;
  if (elapsed >= 2000) {
    fpsText.textContent = `${((frameCounter * 1000) / elapsed).toFixed(1)} fps`;
    frameCounter = 0;
    lastFpsAt = now;
  }
}

startButton.addEventListener("click", () => {
  if (stream) {
    stopCamera(true);
  } else {
    startCamera();
  }
});

switchButton.addEventListener("click", switchCamera);
overlaySwitchButton.addEventListener("click", switchCamera);

window.addEventListener("beforeunload", () => stopCamera(false));

clearPredictions();
loadModel();
