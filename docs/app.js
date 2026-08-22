// Treja detection POC — everything below runs on-device, in this browser tab.
// Preprocessing constants copied exactly from the model's mmdetection config
// (rtmdet_s_8xb32-300e_coco.py data_preprocessor), verified against the real
// checkpoint during export. Order is [B, G, R] because the model was trained
// with bgr_to_rgb=False.
const MEAN_BGR = [103.53, 116.28, 123.675];
const STD_BGR = [57.375, 57.12, 58.395];
const INPUT_SIZE = 640;
const SCORE_THR = 0.3;
const IOU_THR = 0.65; // matches model.test_cfg.nms.iou_threshold from training config
const NUM_CLASSES = 80;

const video = document.getElementById('video');
const photoCanvas = document.getElementById('photo');
const captureBtn = document.getElementById('captureBtn');
const retakeBtn = document.getElementById('retakeBtn');
const statusEl = document.getElementById('status');
const listEl = document.getElementById('list');
const progressBar = document.getElementById('progressBar');
const logBox = document.getElementById('logBox');
const copyLogBtn = document.getElementById('copyLogBtn');

let session = null;
let stream = null;

// ---- on-page debug log (mobile browsers have no accessible console) ----
function log(msg) {
  const t = new Date().toISOString().split('T')[1].replace('Z', '');
  logBox.textContent += `[${t}] ${msg}\n`;
  logBox.scrollTop = logBox.scrollHeight;
}
window.addEventListener('error', (e) => log('WINDOW ERROR: ' + e.message));
window.addEventListener('unhandledrejection', (e) => log('UNHANDLED PROMISE: ' + (e.reason?.stack || e.reason)));
copyLogBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(logBox.textContent);
    copyLogBtn.textContent = 'Copied!';
    setTimeout(() => (copyLogBtn.textContent = 'Copy log'), 1500);
  } catch (e) {
    log('Clipboard failed: ' + e.message + ' — select the text manually instead.');
  }
});

function setStatus(msg) { statusEl.textContent = msg; }
function setProgress(pct) { progressBar.style.width = Math.max(0, Math.min(100, pct)) + '%'; }

async function initCamera() {
  log('Requesting camera permission…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 1280 } },
      audio: false,
    });
    video.srcObject = stream;
    log('Camera stream acquired OK.');
  } catch (err) {
    log('CAMERA FAILED: ' + err.name + ' — ' + err.message);
    setStatus('Camera error: ' + err.message);
    throw err;
  }
}

// Fetch a URL manually (instead of letting ort.InferenceSession.create fetch it
// internally) so we can show real download progress and log exactly where it fails.
async function fetchWithProgress(url, onProgress) {
  log(`Fetching ${url} …`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  const total = Number(res.headers.get('content-length')) || 0;
  log(`${url}: content-length = ${total ? (total / 1e6).toFixed(1) + ' MB' : 'unknown'}`);
  if (!res.body || !total) {
    const buf = await res.arrayBuffer();
    log(`${url}: downloaded ${(buf.byteLength / 1e6).toFixed(1)} MB (no progress stream available)`);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received / total);
  }
  const buf = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { buf.set(c, offset); offset += c.length; }
  log(`${url}: download complete (${(received / 1e6).toFixed(1)} MB).`);
  return buf.buffer;
}

async function initModel() {
  ort.env.wasm.wasmPaths = './';
  ort.env.wasm.numThreads = 1; // GitHub Pages can't set COOP/COEP headers threaded WASM needs
  ort.env.logLevel = 'verbose';
  log('ort env configured (wasmPaths=./, numThreads=1). ort version: ' + (ort.env.versions?.web || 'unknown'));

  const modelBuffer = await fetchWithProgress('./model.onnx', (frac) => {
    setProgress(frac * 100);
    setStatus(`Downloading model… ${(frac * 100).toFixed(0)}%`);
  });

  log('Creating InferenceSession (this step compiles the WASM — can take a while on first run)…');
  setStatus('Compiling model on-device (this can take up to a minute the first time)…');
  const t0 = performance.now();
  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  log(`InferenceSession created in ${((performance.now() - t0) / 1000).toFixed(1)}s.`);
  log('Input names: ' + JSON.stringify(session.inputNames) + ' | Output names: ' + JSON.stringify(session.outputNames));
}

async function boot() {
  log('Boot started. UA: ' + navigator.userAgent);
  setStatus('Starting camera…');
  try {
    await initCamera();
  } catch (e) {
    captureBtn.textContent = 'Camera error — see log';
    return;
  }
  setStatus('Loading model (~55MB total, first time only)…');
  try {
    await initModel();
  } catch (e) {
    log('MODEL LOAD FAILED: ' + (e.stack || e.message || e));
    setStatus('Model failed to load — see debug log below, tap "Copy log" and send it over.');
    captureBtn.textContent = 'Model error — see log';
    return;
  }
  setProgress(100);
  setStatus('Ready.');
  captureBtn.disabled = false;
  captureBtn.textContent = 'Take Photo';
  log('Boot complete. Ready for capture.');
}
boot();

captureBtn.addEventListener('click', async () => {
  captureBtn.disabled = true;
  captureBtn.textContent = 'Analyzing…';
  setStatus('Running detection on this device…');

  const vw = video.videoWidth, vh = video.videoHeight;
  photoCanvas.width = vw;
  photoCanvas.height = vh;
  const ctx = photoCanvas.getContext('2d');
  ctx.drawImage(video, 0, 0, vw, vh);

  video.style.display = 'none';
  photoCanvas.style.display = 'block';
  retakeBtn.style.display = 'block';

  const t0 = performance.now();
  const detections = await runDetection(photoCanvas, ctx);
  const ms = (performance.now() - t0).toFixed(0);

  drawDetections(ctx, detections);
  renderList(detections);
  setStatus(`Done in ${ms} ms — ${detections.length} object${detections.length === 1 ? '' : 's'} found.`);
  captureBtn.textContent = 'Take Photo';
});

retakeBtn.addEventListener('click', () => {
  video.style.display = 'block';
  photoCanvas.style.display = 'none';
  retakeBtn.style.display = 'none';
  listEl.innerHTML = '<div id="empty">Take a photo to see results here.</div>';
  setStatus('Ready.');
  captureBtn.disabled = false;
  captureBtn.textContent = 'Take Photo';
});

// Letterbox-resize the captured photo to 640x640 (pad value 114), matching
// the exact preprocessing used when the model was exported/verified.
function preprocess(canvas) {
  const srcW = canvas.width, srcH = canvas.height;
  const scale = Math.min(INPUT_SIZE / srcH, INPUT_SIZE / srcW);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);

  const tmp = document.createElement('canvas');
  tmp.width = INPUT_SIZE;
  tmp.height = INPUT_SIZE;
  const tctx = tmp.getContext('2d');
  tctx.fillStyle = 'rgb(114,114,114)';
  tctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  tctx.drawImage(canvas, 0, 0, srcW, srcH, 0, 0, newW, newH);

  const imgData = tctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const chw = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const plane = INPUT_SIZE * INPUT_SIZE;
  for (let p = 0; p < plane; p++) {
    const r = imgData[p * 4 + 0];
    const g = imgData[p * 4 + 1];
    const b = imgData[p * 4 + 2];
    // model expects BGR order (bgr_to_rgb=False)
    chw[0 * plane + p] = (b - MEAN_BGR[0]) / STD_BGR[0];
    chw[1 * plane + p] = (g - MEAN_BGR[1]) / STD_BGR[1];
    chw[2 * plane + p] = (r - MEAN_BGR[2]) / STD_BGR[2];
  }
  return { tensorData: chw, scale };
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union <= 0 ? 0 : inter / union;
}

function classAwareNMS(candidates) {
  const byClass = new Map();
  for (const c of candidates) {
    if (!byClass.has(c.label)) byClass.set(c.label, []);
    byClass.get(c.label).push(c);
  }
  const kept = [];
  for (const group of byClass.values()) {
    group.sort((a, b) => b.score - a.score);
    const active = group.slice();
    while (active.length) {
      const best = active.shift();
      kept.push(best);
      for (let i = active.length - 1; i >= 0; i--) {
        if (iou(best.box, active[i].box) > IOU_THR) active.splice(i, 1);
      }
    }
  }
  kept.sort((a, b) => b.score - a.score);
  return kept;
}

async function runDetection(canvas) {
  const { tensorData, scale } = preprocess(canvas);
  const inputTensor = new ort.Tensor('float32', tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const outputs = await session.run({ image: inputTensor });
  const boxes = outputs.boxes.data;   // [1, 8400, 4]
  const scores = outputs.scores.data; // [1, 8400, 80]
  const numBoxes = outputs.boxes.dims[1];

  const candidates = [];
  for (let i = 0; i < numBoxes; i++) {
    let bestScore = -1, bestLabel = -1;
    const base = i * NUM_CLASSES;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const s = scores[base + c];
      if (s > bestScore) { bestScore = s; bestLabel = c; }
    }
    if (bestScore > SCORE_THR) {
      const bi = i * 4;
      // map from 640x640 letterboxed space back to the original photo's pixels
      const box = [
        boxes[bi + 0] / scale,
        boxes[bi + 1] / scale,
        boxes[bi + 2] / scale,
        boxes[bi + 3] / scale,
      ];
      candidates.push({ box, score: bestScore, label: bestLabel });
    }
  }
  return classAwareNMS(candidates);
}

function drawDetections(ctx, detections) {
  ctx.lineWidth = Math.max(2, photoCanvas.width / 300);
  ctx.font = `${Math.max(14, photoCanvas.width / 45)}px -apple-system, sans-serif`;
  ctx.textBaseline = 'bottom';
  for (const d of detections) {
    const [x1, y1, x2, y2] = d.box;
    ctx.strokeStyle = '#4f8cff';
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `${COCO_CLASSES[d.label]} ${(d.score * 100).toFixed(0)}%`;
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = '#4f8cff';
    ctx.fillRect(x1, Math.max(0, y1 - 22), textW + 10, 22);
    ctx.fillStyle = '#0b0c10';
    ctx.fillText(label, x1 + 5, Math.max(20, y1));
  }
}

function renderList(detections) {
  if (!detections.length) {
    listEl.innerHTML = '<div id="empty">No objects detected. Try retaking with better lighting/distance.</div>';
    return;
  }
  listEl.innerHTML = detections
    .map(
      (d) => `<div class="item"><span class="name">${COCO_CLASSES[d.label]}</span><span class="score">${(d.score * 100).toFixed(0)}%</span></div>`
    )
    .join('');
}
