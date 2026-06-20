const video = document.getElementById("video");
const canvas = document.getElementById("snapshotCanvas");
const boxCanvas = document.getElementById("boxCanvas");
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

const MODEL_URL = "yolov8n.onnx";
const MODEL_INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;
const MAX_DETECTIONS = 20;
const ORT_WASM_PATH = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

const COCO_LABELS = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush",
];

let session = null;
let stream = null;
let facingMode = "environment";
let hasSnapshot = false;
let isBusy = false;

function setStatus(mode, text) {
  statusDot.className = `status-dot ${mode || ""}`.trim();
  statusText.textContent = text;
}

function setClassifyState(mode, text) {
  classifyState.className = `classify-state ${mode || ""}`.trim();
  classifyState.querySelector("span:last-child").textContent = text;
}

function iconPath(type) {
  const paths = {
    camera: "M9 5 7.2 7.4H4a2 2 0 0 0-2 2V17a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9.4a2 2 0 0 0-2-2h-3.2L15 5H9Zm3 12a4 4 0 1 1 0-8 4 4 0 0 1 0 8Z",
    snap: "M12 18.5a5.5 5.5 0 1 0 0-11 5.5 5.5 0 0 0 0 11Zm0-2.4a3.1 3.1 0 1 1 0-6.2 3.1 3.1 0 0 1 0 6.2ZM4 6h3l1.6-2h6.8L17 6h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2Z",
    retake: "M12 5a7 7 0 0 0-6.3 4H3.6A9 9 0 0 1 19 6.4V3h2v7h-7V8h3.5A7 7 0 1 0 19 15h2.1A9 9 0 1 1 12 5Z",
  };
  return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[type]}"/></svg>`;
}

function setButtons() {
  const modelReady = Boolean(session);
  startButton.disabled = !modelReady || isBusy;

  if (stream) {
    startButton.innerHTML = `${iconPath("snap")}Snap & Detect`;
  } else if (hasSnapshot) {
    startButton.innerHTML = `${iconPath("retake")}Retake`;
  } else {
    startButton.innerHTML = `${iconPath("camera")}Open Camera`;
  }

  switchButton.disabled = !stream || isBusy;
  overlaySwitchButton.disabled = !stream || isBusy;
}

function clearDetections(message = "YOLO detections will appear here.") {
  bestClass.textContent = hasSnapshot ? "Ready to detect" : "No snapshot yet";
  bestConfidence.textContent = hasSnapshot ? "Tap Snap & Detect after opening camera" : "Take one photo to detect objects";
  fpsText.textContent = "single shot";
  matchesList.innerHTML = `<div class="empty-match">${escapeHtml(message)}</div>`;
  clearBoxes();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
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

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function titleCase(value) {
  return value.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

async function waitForOrt() {
  const started = performance.now();
  while (!window.ort && performance.now() - started < 15000) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  if (!window.ort) {
    throw new Error("ONNX Runtime Web did not load. Check internet/CDN access.");
  }
}

async function loadModel() {
  setStatus("busy", "Loading");
  setClassifyState("", "Loading YOLO");
  setButtons();

  try {
    await waitForOrt();
    ort.env.wasm.wasmPaths = ORT_WASM_PATH;
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    setStatus("", "Ready");
    setClassifyState("ready", "Ready");
    clearDetections("Open camera, then take one snapshot.");
  } catch (error) {
    setStatus("error", "Model missing");
    setClassifyState("error", "YOLO not loaded");
    bestClass.textContent = "YOLO model not loaded";
    bestConfidence.textContent = "Add models/yolov8n.onnx to the repo";
    matchesList.innerHTML = `<div class="empty-match">${escapeHtml(error.message)}</div>`;
  } finally {
    setButtons();
  }
}

async function openCamera() {
  if (!session || isBusy) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("error", "No camera API");
    setClassifyState("error", "Camera not supported");
    clearDetections("This browser does not support camera access.");
    return;
  }

  resetSnapshot(false);
  stopStream();
  setStatus("busy", "Starting");
  setClassifyState("", "Starting camera");
  setButtons();

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
    setStatus("", "Ready");
    setClassifyState("ready", "Ready");
    clearDetections("Frame your object, then tap Snap & Detect.");
  } catch (error) {
    stream = null;
    video.classList.remove("active");
    cameraEmpty.classList.remove("hidden");
    setStatus("error", "Camera blocked");
    setClassifyState("error", "Camera blocked");
    clearDetections(error.message);
  } finally {
    setButtons();
  }
}

function stopStream() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  video.classList.remove("active");
}

function resetSnapshot(showEmpty = true) {
  hasSnapshot = false;
  canvas.classList.remove("active");
  boxCanvas.classList.remove("active");
  canvas.getContext("2d").clearRect(0, 0, canvas.width || 1, canvas.height || 1);
  clearBoxes();
  document.querySelector(".viewfinder").classList.remove("has-snapshot");
  if (showEmpty) {
    cameraEmpty.classList.remove("hidden");
  }
}

async function switchCamera() {
  if (isBusy) {
    return;
  }
  facingMode = facingMode === "environment" ? "user" : "environment";
  await openCamera();
}

async function captureAndDetect() {
  if (!stream || !session || isBusy) {
    return;
  }

  isBusy = true;
  setButtons();
  setStatus("busy", "Detecting");
  setClassifyState("", "Running YOLO");
  clearDetections("Running one YOLO pass on this snapshot...");

  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  canvas.width = width;
  canvas.height = height;
  boxCanvas.width = width;
  boxCanvas.height = height;
  canvas.getContext("2d").drawImage(video, 0, 0, width, height);
  canvas.classList.add("active");
  boxCanvas.classList.add("active");
  document.querySelector(".viewfinder").classList.add("has-snapshot");
  hasSnapshot = true;
  stopStream();

  try {
    const started = performance.now();
    const detections = await runYolo(canvas);
    const elapsed = Math.round(performance.now() - started);
    drawDetections(detections);
    renderDetections(detections, elapsed);
    setStatus("", "Done");
    setClassifyState("ready", "Snapshot done");
  } catch (error) {
    setStatus("error", "Detect error");
    setClassifyState("error", "Detect error");
    bestClass.textContent = "Detection failed";
    bestConfidence.textContent = "Check the YOLO model file";
    matchesList.innerHTML = `<div class="empty-match">${escapeHtml(error.message)}</div>`;
    clearBoxes();
  } finally {
    isBusy = false;
    setButtons();
  }
}

async function runYolo(sourceCanvas) {
  const prepared = prepareYoloInput(sourceCanvas);
  const inputName = session.inputNames[0];
  const outputMap = await session.run({ [inputName]: prepared.tensor });
  const output = outputMap[session.outputNames[0]];
  const decoded = decodeYoloOutput(output, prepared);
  return nonMaxSuppression(decoded, IOU_THRESHOLD).slice(0, MAX_DETECTIONS);
}

function prepareYoloInput(sourceCanvas) {
  const inputCanvas = document.createElement("canvas");
  inputCanvas.width = MODEL_INPUT_SIZE;
  inputCanvas.height = MODEL_INPUT_SIZE;
  const inputCtx = inputCanvas.getContext("2d");
  const sourceWidth = sourceCanvas.width;
  const sourceHeight = sourceCanvas.height;
  const scale = Math.min(MODEL_INPUT_SIZE / sourceWidth, MODEL_INPUT_SIZE / sourceHeight);
  const resizedWidth = Math.round(sourceWidth * scale);
  const resizedHeight = Math.round(sourceHeight * scale);
  const padX = Math.floor((MODEL_INPUT_SIZE - resizedWidth) / 2);
  const padY = Math.floor((MODEL_INPUT_SIZE - resizedHeight) / 2);

  inputCtx.fillStyle = "rgb(114, 114, 114)";
  inputCtx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  inputCtx.drawImage(sourceCanvas, 0, 0, sourceWidth, sourceHeight, padX, padY, resizedWidth, resizedHeight);

  const imageData = inputCtx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const pixels = imageData.data;
  const input = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  const area = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

  for (let i = 0; i < area; i += 1) {
    input[i] = pixels[i * 4] / 255;
    input[i + area] = pixels[i * 4 + 1] / 255;
    input[i + area * 2] = pixels[i * 4 + 2] / 255;
  }

  return {
    tensor: new ort.Tensor("float32", input, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]),
    sourceWidth,
    sourceHeight,
    scale,
    padX,
    padY,
  };
}

function decodeYoloOutput(output, meta) {
  const data = output.data;
  const dims = output.dims;
  let boxCount;
  let valueCount;
  let getValue;

  if (dims.length === 3 && dims[1] < dims[2]) {
    valueCount = dims[1];
    boxCount = dims[2];
    getValue = (boxIndex, valueIndex) => data[valueIndex * boxCount + boxIndex];
  } else if (dims.length === 3) {
    boxCount = dims[1];
    valueCount = dims[2];
    getValue = (boxIndex, valueIndex) => data[boxIndex * valueCount + valueIndex];
  } else if (dims.length === 2) {
    boxCount = dims[0];
    valueCount = dims[1];
    getValue = (boxIndex, valueIndex) => data[boxIndex * valueCount + valueIndex];
  } else {
    throw new Error(`Unsupported YOLO output shape: ${dims.join("x")}`);
  }

  const detections = [];
  const classStart = 4;
  const classCount = Math.min(valueCount - classStart, COCO_LABELS.length);

  for (let i = 0; i < boxCount; i += 1) {
    const cx = getValue(i, 0);
    const cy = getValue(i, 1);
    const width = getValue(i, 2);
    const height = getValue(i, 3);
    let bestScore = 0;
    let classIndex = 0;

    for (let classOffset = 0; classOffset < classCount; classOffset += 1) {
      const score = getValue(i, classStart + classOffset);
      if (score > bestScore) {
        bestScore = score;
        classIndex = classOffset;
      }
    }

    if (bestScore < CONFIDENCE_THRESHOLD) {
      continue;
    }

    const x1 = (cx - width / 2 - meta.padX) / meta.scale;
    const y1 = (cy - height / 2 - meta.padY) / meta.scale;
    const x2 = (cx + width / 2 - meta.padX) / meta.scale;
    const y2 = (cy + height / 2 - meta.padY) / meta.scale;
    const clamped = clampBox(x1, y1, x2, y2, meta.sourceWidth, meta.sourceHeight);

    if (clamped.width <= 1 || clamped.height <= 1) {
      continue;
    }

    detections.push({
      label: COCO_LABELS[classIndex] || `class ${classIndex}`,
      classIndex,
      score: bestScore,
      ...clamped,
    });
  }

  return detections;
}

function clampBox(x1, y1, x2, y2, maxWidth, maxHeight) {
  const left = Math.max(0, Math.min(maxWidth, x1));
  const top = Math.max(0, Math.min(maxHeight, y1));
  const right = Math.max(0, Math.min(maxWidth, x2));
  const bottom = Math.max(0, Math.min(maxHeight, y2));
  return {
    x: left,
    y: top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  };
}

function nonMaxSuppression(detections, iouThreshold) {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const selected = [];

  for (const detection of sorted) {
    const overlaps = selected.some((kept) => kept.classIndex === detection.classIndex && iou(kept, detection) > iouThreshold);
    if (!overlaps) {
      selected.push(detection);
    }
  }

  return selected;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function drawDetections(detections) {
  const ctx = boxCanvas.getContext("2d");
  ctx.clearRect(0, 0, boxCanvas.width, boxCanvas.height);
  const lineWidth = Math.max(3, boxCanvas.width / 260);
  const fontSize = Math.max(18, Math.round(boxCanvas.width / 36));
  ctx.lineWidth = lineWidth;
  ctx.font = `700 ${fontSize}px Arial`;
  ctx.textBaseline = "top";

  detections.forEach((detection) => {
    const label = `${titleCase(detection.label)} ${formatPercent(detection.score)}`;
    const textWidth = ctx.measureText(label).width;
    const labelHeight = fontSize + 10;
    const labelY = Math.max(0, detection.y - labelHeight);

    ctx.strokeStyle = "#0f8f7a";
    ctx.fillStyle = "rgba(15, 143, 122, 0.14)";
    ctx.fillRect(detection.x, detection.y, detection.width, detection.height);
    ctx.strokeRect(detection.x, detection.y, detection.width, detection.height);

    ctx.fillStyle = "#0f8f7a";
    ctx.fillRect(detection.x, labelY, textWidth + 12, labelHeight);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, detection.x + 6, labelY + 5);
  });
}

function clearBoxes() {
  const ctx = boxCanvas.getContext("2d");
  ctx.clearRect(0, 0, boxCanvas.width || 1, boxCanvas.height || 1);
}

function renderDetections(detections, elapsedMs) {
  fpsText.textContent = `${elapsedMs} ms`;

  if (!detections.length) {
    bestClass.textContent = "No object detected";
    bestConfidence.textContent = `YOLO finished in ${elapsedMs} ms`;
    matchesList.innerHTML = `<div class="empty-match">No objects above ${Math.round(CONFIDENCE_THRESHOLD * 100)}% confidence.</div>`;
    return;
  }

  const top = detections[0];
  bestClass.textContent = titleCase(top.label);
  bestConfidence.textContent = `${detections.length} object${detections.length === 1 ? "" : "s"} detected. Top confidence: ${formatPercent(top.score)}`;

  matchesList.innerHTML = detections
    .map((detection) => {
      const label = titleCase(detection.label);
      const score = formatPercent(detection.score);
      const width = Math.round(detection.score * 100);
      return `
        <div class="match-row">
          <span class="match-name" title="${escapeHtml(label)}">${escapeHtml(label)}</span>
          <span class="match-track" aria-hidden="true">
            <span class="match-fill" style="width: ${width}%"></span>
          </span>
          <span class="match-score">${score}</span>
        </div>
      `;
    })
    .join("");
}

startButton.addEventListener("click", () => {
  if (stream) {
    captureAndDetect();
  } else {
    openCamera();
  }
});

switchButton.addEventListener("click", switchCamera);
overlaySwitchButton.addEventListener("click", switchCamera);

window.addEventListener("beforeunload", () => stopStream());

clearDetections();
loadModel();
