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

let session = null;
let stream = null;

function setStatus(msg) { statusEl.textContent = msg; }

async function initCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 1280 } },
      audio: false,
    });
    video.srcObject = stream;
  } catch (err) {
    setStatus('Camera error: ' + err.message + ' (needs HTTPS + camera permission)');
    throw err;
  }
}

async function initModel() {
  ort.env.wasm.wasmPaths = './';
  // GitHub Pages can't set the COOP/COEP headers multi-threaded WASM needs
  // (SharedArrayBuffer), so force single-threaded — fine for one photo at a time.
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create('./model.onnx', {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

async function boot() {
  setStatus('Starting camera…');
  await initCamera();
  setStatus('Loading model (~40MB, first time only)…');
  await initModel();
  setStatus('Ready.');
  captureBtn.disabled = false;
  captureBtn.textContent = 'Take Photo';
}
boot().catch((e) => console.error(e));

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
