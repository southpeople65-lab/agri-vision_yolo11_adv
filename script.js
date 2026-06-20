// ─────────────────────────────────────────────────────────────
//  AgriVision AI — Live Object Detection (TFLite)  •  ROST srl
// ─────────────────────────────────────────────────────────────

const video    = document.getElementById('webcam');
const canvas   = document.getElementById('canvas');
const ctx      = canvas.getContext('2d');
const app      = document.getElementById('app');

// UI
const statusDot   = document.getElementById('statusDot');
const statusText  = document.getElementById('statusText');
const fpsEl       = document.getElementById('fps');
const resultsEl   = document.getElementById('results');
const countEl     = document.getElementById('count');
const shutterBtn  = document.getElementById('shutterBtn');
const shutterIcon = document.getElementById('shutterIcon');
const flipBtn     = document.getElementById('flipBtn');

ctx.imageSmoothingEnabled = true;
tflite.setWasmPath('./libs/');

// ─── Classi in ordine alfabetico da Colab ───
// "others" al posto di "ashcan": se è un falso positivo non importa, lo vedo come ostacolo
const CLASSI = [
    "others", "car", "person", "pole", "stump", "tree", "warning_column"
];

const ACCENT          = '#22c55e';
const SCORE_THRESHOLD = 0.20;
const IOU_THRESHOLD   = 0.45;
const MAX_DETECTIONS  = 6;
const NUM_CLASSES     = CLASSI.length;
const NUM_BOXES       = 8400;

// ─── Stato app ───
let model    = null;
let running  = true;
let facing   = 'environment';
let stream   = null;

// FPS
let lastFrame = performance.now();
let fpsSmoothed = 0;
let lastFpsPaint = 0;

let lastSig = '';   // firma risultati per evitare ridisegni inutili del pannello

// ─── Stato interfaccia ───
function setStatus(text, color) {
    statusText.textContent = text;
    statusText.style.color = color;
    statusDot.style.background = color;
}

function syncCanvasSize() {
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
        canvas.width  = Math.round(rect.width);
        canvas.height = Math.round(rect.height);
    }
}

// ─── Avvio ───
async function startApp() {
    try {
        setStatus('MODELLO', '#f59e0b');
        countEl.textContent = 'Caricamento...';
        model = await tflite.loadTFLiteModel('./assets/models/best_float32.tflite');

        setStatus('CAMERA', '#f59e0b');
        await openCamera(facing);

        const begin = () => {
            setStatus('LIVE', ACCENT);
            running = true;
            app.classList.remove('paused');
            loop();
        };
        if (video.readyState >= 2) begin();
        else video.addEventListener('loadeddata', begin, { once: true });

    } catch (err) {
        setStatus('ERRORE', '#ef4444');
        resultsEl.innerHTML = `<div class="res-empty" style="color:#fca5a5;">Errore: ${err.message}</div>`;
        countEl.textContent = '';
    }
}

async function openCamera(mode) {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: mode },
        audio: false
    });
    video.srcObject = stream;
    video.setAttribute('playsinline', true);
    await video.play();
}

// ─── Ciclo di predizione ───
function loop() {
    if (!running) return;
    predict().finally(() => {
        if (running) requestAnimationFrame(loop);
    });
}

async function predict() {
    let input = null;
    try {
        // FPS
        const now = performance.now();
        const dt = now - lastFrame;
        lastFrame = now;
        if (dt > 0) fpsSmoothed = fpsSmoothed ? fpsSmoothed * 0.85 + (1000 / dt) * 0.15 : 1000 / dt;
        if (now - lastFpsPaint > 500) {
            fpsEl.textContent = Math.round(fpsSmoothed) + ' FPS';
            lastFpsPaint = now;
        }

        syncCanvasSize();

        input = tf.browser.fromPixels(video)
            .resizeBilinear([640, 640])
            .toFloat()
            .div(255.0)
            .expandDims(0);

        const result = model.predict(input);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (result) {
            const data = await result.data();
            result.dispose();

            const candidati = [];
            const scaleX = canvas.width;
            const scaleY = canvas.height;

            // 1. Estrazione candidati
            for (let b = 0; b < NUM_BOXES; b++) {
                let maxScore = 0, classId = -1;
                for (let c = 0; c < NUM_CLASSES; c++) {
                    const score = data[(4 + c) * NUM_BOXES + b];
                    if (score > maxScore) { maxScore = score; classId = c; }
                }
                if (maxScore > SCORE_THRESHOLD) {
                    const cx = data[0 * NUM_BOXES + b];
                    const cy = data[1 * NUM_BOXES + b];
                    const w  = data[2 * NUM_BOXES + b];
                    const h  = data[3 * NUM_BOXES + b];
                    candidati.push({
                        xmin: (cx - w / 2) * scaleX,
                        ymin: (cy - h / 2) * scaleY,
                        width:  w * scaleX,
                        height: h * scaleY,
                        score: maxScore,
                        classId
                    });
                }
            }

            // 2. Ordinamento
            candidati.sort((a, b) => b.score - a.score);

            // 3. Non-Maximum Suppression (NMS)
            const scelti = [];
            for (const box of candidati) {
                let sovrapposto = false;
                for (const scelto of scelti) {
                    const xA = Math.max(box.xmin, scelto.xmin);
                    const yA = Math.max(box.ymin, scelto.ymin);
                    const xB = Math.min(box.xmin + box.width,  scelto.xmin + scelto.width);
                    const yB = Math.min(box.ymin + box.height, scelto.ymin + scelto.height);
                    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
                    if (interArea > 0) {
                        const iou = interArea / (box.width * box.height + scelto.width * scelto.height - interArea);
                        if (iou > IOU_THRESHOLD) { sovrapposto = true; break; }
                    }
                }
                if (!sovrapposto) scelti.push(box);
            }

            const topN = scelti.slice(0, MAX_DETECTIONS);

            // 4. Disegno HUD sul canvas
            for (const box of topN) drawDetection(box);

            // 5. Pannello risultati
            updatePanel(topN);
        }
    } catch (err) {
        setStatus('ERRORE', '#ef4444');
        resultsEl.innerHTML = `<div class="res-empty" style="color:#fca5a5;">Errore predizione: ${err.message}</div>`;
    } finally {
        if (input) input.dispose();
    }
}

// ─── Disegno box stile HUD (angoli + etichetta) ───
function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

function drawDetection(box) {
    const { xmin, ymin, width: w, height: h } = box;
    const corner = Math.max(10, Math.min(28, Math.min(w, h) * 0.22));

    // bordo tenue
    ctx.strokeStyle = 'rgba(34,197,94,0.28)';
    ctx.lineWidth = 1.5;
    roundRect(xmin, ymin, w, h, 6);
    ctx.stroke();

    // angoli marcati
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    const c = (px, py, dx, dy) => {
        ctx.beginPath();
        ctx.moveTo(px, py + dy * corner);
        ctx.lineTo(px, py);
        ctx.lineTo(px + dx * corner, py);
        ctx.stroke();
    };
    c(xmin, ymin, 1, 1);
    c(xmin + w, ymin, -1, 1);
    c(xmin, ymin + h, 1, -1);
    c(xmin + w, ymin + h, -1, -1);

    // etichetta
    const pct = Math.round(box.score * 100);
    const label = `${CLASSI[box.classId]} ${pct}%`;
    ctx.font = '700 13px -apple-system, "Segoe UI", Roboto, sans-serif';
    const tw = ctx.measureText(label).width;
    const chipH = 20;
    const chipW = tw + 14;
    let ly = ymin - chipH - 4;
    if (ly < 0) ly = ymin + 4;
    ctx.fillStyle = ACCENT;
    roundRect(xmin, ly, chipW, chipH, 5);
    ctx.fill();
    ctx.fillStyle = '#04210f';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, xmin + 7, ly + chipH / 2 + 0.5);
}

// ─── Pannello risultati dinamico ───
function updatePanel(topN) {
    const sig = topN.map(b => b.classId + ':' + Math.round(b.score * 100)).join('|');
    if (sig === lastSig) return;       // niente cambiamenti → no re-render
    lastSig = sig;

    if (topN.length === 0) {
        countEl.textContent = 'Scansione...';
        resultsEl.innerHTML = '<div class="res-empty">Inquadra un elemento per iniziare il rilevamento</div>';
        return;
    }

    countEl.textContent = topN.length + (topN.length === 1 ? ' elemento' : ' elementi');
    resultsEl.innerHTML = topN.map(b => {
        const pct  = Math.round(b.score * 100);
        const name = CLASSI[b.classId];
        return `<div class="res-row">
            <span class="res-dot"></span>
            <span class="res-name">${name}</span>
            <span class="res-bar"><i style="width:${pct}%"></i></span>
            <span class="res-pct">${pct}%</span>
        </div>`;
    }).join('');
}

// ─── Controlli ───
shutterBtn.addEventListener('click', () => {
    if (!model) return;
    running = !running;
    if (running) {
        shutterIcon.className = 'icon-stop';
        app.classList.remove('paused');
        setStatus('LIVE', ACCENT);
        video.play().catch(() => {});
        loop();
    } else {
        shutterIcon.className = 'icon-play';
        app.classList.add('paused');
        setStatus('PAUSA', '#64748b');
        video.pause();
    }
});

flipBtn.addEventListener('click', async () => {
    facing = (facing === 'environment') ? 'user' : 'environment';
    try {
        setStatus('CAMERA', '#f59e0b');
        await openCamera(facing);
        if (running) { setStatus('LIVE', ACCENT); loop(); }
    } catch (err) {
        setStatus('ERRORE', '#ef4444');
    }
});

startApp();
