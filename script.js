const video = document.getElementById('webcam');
const output = document.getElementById('output');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

ctx.imageSmoothingEnabled = true;

tflite.setWasmPath('./libs/');

// ─── Le tue  classi in perfetto ordine alfabetico da Colab ───
// ho inserito others al posto di ashcan in modo che se è un falso positivo non importa. mi vede ostacolo
const CLASSI = [
    "others", "car", "person", "pole", "stump", "tree", "warning_column"
];

const LABEL_FONT      = 'bold 15px monospace';
const LABEL_HEIGHT    = 22;
const BOX_COLOR       = '#22c55e';
const SCORE_THRESHOLD = 0.20; 
const IOU_THRESHOLD   = 0.45;
const MAX_DETECTIONS  = 6;
const NUM_CLASSES     = CLASSI.length; 
const NUM_BOXES       = 8400;

function syncCanvasSize() {
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
        canvas.width  = Math.round(rect.width);
        canvas.height = Math.round(rect.height);
    }
}

async function startApp() {
    output.innerText = "Caricamento Modello AgriVision...";
    try {
        const model = await tflite.loadTFLiteModel('./assets/models/best_float32.tflite');
        output.innerText = "Modello caricato. Configurazione flusso video...";

        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        });

        video.srcObject = stream;
        video.setAttribute('playsinline', true);

        const runTracking = () => {
            output.innerText = "AgriVision Attivo: Scansione in tempo reale";
            predict(model);
        };

        if (video.readyState >= 2) {
            runTracking();
        } else {
            video.addEventListener('loadeddata', runTracking);
        }

        await video.play();

    } catch (err) {
        output.innerHTML = `<span style="color:#ef4444;">Errore Inizializzazione: ${err.message}</span>`;
    }
}

async function predict(model) {
    let input = null;
    try {
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
            let topClassId = -1;
            let topScore   = 0;

            const scaleX = canvas.width;
            const scaleY = canvas.height;

            // 1. Estrazione geometrica dei candidati
            for (let b = 0; b < NUM_BOXES; b++) {
                let maxScore = 0;
                let classId  = -1;

                for (let c = 0; c < NUM_CLASSES; c++) {
                    const score = data[(4 + c) * NUM_BOXES + b];
                    if (score > maxScore) {
                        maxScore = score;
                        classId  = c;
                    }
                }

                if (maxScore > topScore) {
                    topScore   = maxScore;
                    topClassId = classId;
                }

                if (maxScore > SCORE_THRESHOLD) {
                    const cx = data[0 * NUM_BOXES + b];
                    const cy = data[1 * NUM_BOXES + b];
                    const w  = data[2 * NUM_BOXES + b];
                    const h  = data[3 * NUM_BOXES + b];

                    const xmin = (cx - w / 2) * scaleX;
                    const ymin = (cy - h / 2) * scaleY;
                    const boxW = w * scaleX;
                    const boxH = h * scaleY;

                    candidati.push({ xmin, ymin, width: boxW, height: boxH, score: maxScore, classId });
                }
            }

            // 2. Ordinamento
            candidati.sort((a, b) => b.score - a.score);

            // 3. Algoritmo Non-Maximum Suppression (NMS)
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
                        if (iou > IOU_THRESHOLD) {
                            sovrapposto = true;
                            break;
                        }
                    }
                }
                if (!sovrapposto) scelti.push(box);
            }

            const topN = scelti.slice(0, MAX_DETECTIONS);

            // 4. Rendering grafico sul Canvas
            ctx.font = LABEL_FONT;
            for (const box of topN) {
                const pct       = Math.round(box.score * 100);
                const etichetta = `${CLASSI[box.classId]} ${pct}%`;
                const labelW    = ctx.measureText(etichetta).width;

                ctx.strokeStyle = BOX_COLOR;
                ctx.lineWidth   = 3;
                ctx.strokeRect(box.xmin, box.ymin, box.width, box.height);

                const labelY = box.ymin > LABEL_HEIGHT ? box.ymin - LABEL_HEIGHT : box.ymin;
                ctx.fillStyle = BOX_COLOR;
                ctx.fillRect(box.xmin, labelY, labelW + 10, LABEL_HEIGHT);

                ctx.fillStyle = '#ffffff';
                ctx.fillText(etichetta, box.xmin + 5, labelY + LABEL_HEIGHT - 5);
            }

            // 5. Aggiornamento interfaccia testuale
            if (topN.length > 0) {
                output.innerHTML = `Rilevati <span style="color:#22c55e;font-weight:bold;">${topN.length}</span> elementi`;
            } else if (topClassId !== -1 && topScore > 0.15) {
                output.innerHTML = `Analisi: Rilevato ${CLASSI[topClassId]} con confidenza bassa (${Math.round(topScore * 100)}%)`;
            } else {
                output.innerHTML = "Scansione AgriVision attiva... Inquadra un elemento";
            }
        }
    } catch (err) {
        output.innerHTML = `<span style="color:#ef4444;">Errore ciclo predizione: ${err.message}</span>`;
        if (input) input.dispose();
        return;
    }

    if (input) input.dispose();
    requestAnimationFrame(() => predict(model));
}

startApp();
