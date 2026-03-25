/* ============================================================
   Diamond Pushup Counter – Client-side logic
   Uses MediaPipe Tasks Vision (Pose Landmarker) in the browser.
   ============================================================ */

import {
    PoseLandmarker,
    FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ── DOM refs ────────────────────────────────────────────────
const video        = document.getElementById("webcam");
const canvas       = document.getElementById("overlay");
const ctx          = canvas.getContext("2d");
const repCountEl   = document.getElementById("rep-count");
const angleValueEl = document.getElementById("angle-value");
const stateBadge   = document.getElementById("state-badge");
const recordBtn    = document.getElementById("record-btn");
const resetBtn     = document.getElementById("reset-btn");
const saveBtn      = document.getElementById("save-btn");
const statusEl     = document.getElementById("status");
const cameraSelect = document.getElementById("camera-select");

// ── Pose landmark indices ───────────────────────────────────
const L_SHOULDER = 11;
const L_ELBOW    = 13;
const L_WRIST    = 15;
const R_SHOULDER = 12;
const R_ELBOW    = 14;
const R_WRIST    = 16;

// ── Thresholds ──────────────────────────────────────────────
const DOWN_ANGLE = 100;  // elbow angle < this → "down"
const UP_ANGLE   = 130;  // elbow angle > this → "up" (count++)

// ── State ───────────────────────────────────────────────────
let poseLandmarker   = null;
let webcamStream     = null;   // always-on camera stream
let poseReady        = false;  // both webcam + model loaded
let isRecording      = false;  // video recording + rep counting active
let animFrameId      = null;
let repCount         = 0;
let isDown           = false;
let lastVideoTime    = -1;

// ── MediaRecorder state ─────────────────────────────────────
let mediaRecorder    = null;
let recordedChunks   = [];
let savedBlobUrl     = null;   // URL for the last recorded video

// ── Offscreen recording canvas (video + rep count, no skeleton) ──
const recCanvas = document.createElement("canvas");
const recCtx    = recCanvas.getContext("2d");

// ── Skeleton drawing config ─────────────────────────────────
const ARM_CONNECTIONS = [
    [L_SHOULDER, L_ELBOW],
    [L_ELBOW,    L_WRIST],
    [R_SHOULDER, R_ELBOW],
    [R_ELBOW,    R_WRIST],
];
const JOINT_INDICES = [L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST];

// Colors
const CLR_UP   = "#00ff88";
const CLR_DOWN = "#ff3366";

// ═════════════════════════════════════════════════════════════
//  Angle calculation  (3-D world landmarks → degrees)
// ═════════════════════════════════════════════════════════════
function angleBetween(a, b, c) {
    const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
    const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };

    const dot   = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
    const magBA = Math.hypot(ba.x, ba.y, ba.z);
    const magBC = Math.hypot(bc.x, bc.y, bc.z);

    if (magBA === 0 || magBC === 0) return 180;

    const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
    return Math.acos(cosAngle) * (180 / Math.PI);
}

// ═════════════════════════════════════════════════════════════
//  MediaPipe initialisation  (runs once on page load)
// ═════════════════════════════════════════════════════════════
async function initPoseLandmarker() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );

        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath:
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                delegate: "GPU",
            },
            runningMode: "VIDEO",
            numPoses: 1,
        });

        statusEl.textContent = "Ready — press record to start";
        statusEl.classList.add("ready");
    } catch (err) {
        console.error("Failed to load PoseLandmarker:", err);
        statusEl.textContent = "Error loading AI model. Check console.";
        statusEl.classList.add("error");
    }
}

// ═════════════════════════════════════════════════════════════════
//  Webcam – starts on page load, stays on
// ═════════════════════════════════════════════════════════════════
async function startWebcam(deviceId) {
    // Stop previous stream if switching cameras
    if (webcamStream) {
        webcamStream.getTracks().forEach((t) => t.stop());
    }

    try {
        const constraints = { audio: false };
        if (deviceId) {
            constraints.video = { deviceId: { exact: deviceId } };
        } else {
            constraints.video = { facingMode: "user" };
        }

        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = webcamStream;

        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                canvas.width  = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve();
            };
        });
    } catch (err) {
        console.error("Webcam error:", err);
        statusEl.textContent = `Webcam error: ${err.message}`;
        statusEl.classList.add("error");
    }
}

// ═════════════════════════════════════════════════════════════════
//  Camera enumeration
// ═════════════════════════════════════════════════════════════════
async function populateCameras() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((d) => d.kind === "videoinput");

    cameraSelect.innerHTML = "";
    cameras.forEach((cam, i) => {
        const opt   = document.createElement("option");
        opt.value   = cam.deviceId;
        opt.text    = cam.label || `Camera ${i + 1}`;
        cameraSelect.appendChild(opt);
    });

    // Pre-select the currently active camera
    if (webcamStream) {
        const activeTrack = webcamStream.getVideoTracks()[0];
        const activeId    = activeTrack?.getSettings().deviceId;
        if (activeId) cameraSelect.value = activeId;
    }
}

cameraSelect.addEventListener("change", async () => {
    await startWebcam(cameraSelect.value);
    // Update overlay canvas size to new camera resolution
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
});

// ═════════════════════════════════════════════════════════════
//  MediaRecorder helpers
// ═════════════════════════════════════════════════════════════
function startMediaRecorder() {
    // Revoke previous blob URL if any
    if (savedBlobUrl) {
        URL.revokeObjectURL(savedBlobUrl);
        savedBlobUrl = null;
    }
    recordedChunks = [];

    // Size the offscreen canvas to match the webcam
    recCanvas.width  = video.videoWidth;
    recCanvas.height = video.videoHeight;

    // Capture the composited canvas stream (video + rep count)
    const recStream = recCanvas.captureStream(30);

    // Choose a supported MIME type
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm";

    mediaRecorder = new MediaRecorder(recStream, { mimeType });

    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: mimeType });
        savedBlobUrl = URL.createObjectURL(blob);
        // Show save button
        saveBtn.classList.remove("hidden");
    };

    mediaRecorder.start();
}

function stopMediaRecorder() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
}

// ═════════════════════════════════════════════════════════════
//  Skeleton drawing (minimal – just the arms)
// ═════════════════════════════════════════════════════════════
function drawSkeleton(landmarks) {
    const color = isDown ? CLR_DOWN : CLR_UP;
    const w = canvas.width;
    const h = canvas.height;

    ctx.save();
    // mirror to match the CSS-mirrored video
    ctx.translate(w, 0);
    ctx.scale(-1, 1);

    // Lines
    ctx.strokeStyle = color;
    ctx.lineWidth   = 4;
    ctx.lineCap     = "round";
    ctx.shadowColor = color;
    ctx.shadowBlur  = 10;

    for (const [i, j] of ARM_CONNECTIONS) {
        const a = landmarks[i];
        const b = landmarks[j];
        ctx.beginPath();
        ctx.moveTo(a.x * w, a.y * h);
        ctx.lineTo(b.x * w, b.y * h);
        ctx.stroke();
    }

    // Joints
    ctx.shadowBlur = 0;
    for (const idx of JOINT_INDICES) {
        const lm = landmarks[idx];
        const x  = lm.x * w;
        const y  = lm.y * h;

        // outer glow ring
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = color + "33";
        ctx.fill();

        // solid center
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle   = color;
        ctx.strokeStyle = "#fff";
        ctx.lineWidth   = 2;
        ctx.fill();
        ctx.stroke();
    }

    ctx.restore();
}

// ═════════════════════════════════════════════════════════════
//  Main detection loop
// ═════════════════════════════════════════════════════════════
function detect(timestamp) {
    if (!poseReady) return;

    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && timestamp !== lastVideoTime) {
        lastVideoTime = timestamp;

        const results = poseLandmarker.detectForVideo(video, performance.now());

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.worldLandmarks && results.worldLandmarks.length > 0) {
            const world  = results.worldLandmarks[0];
            const screen = results.landmarks[0];

            // ─── Elbow angles (3-D) ─────────────────────────
            const leftAngle  = angleBetween(world[L_SHOULDER], world[L_ELBOW], world[L_WRIST]);
            const rightAngle = angleBetween(world[R_SHOULDER], world[R_ELBOW], world[R_WRIST]);
            const avg        = (leftAngle + rightAngle) / 2;

            // update HUD
            angleValueEl.textContent = `${Math.round(avg)}°`;

            // ─── State machine (only count during recording) ─
            if (isRecording) {
                if (!isDown && avg < DOWN_ANGLE) {
                    isDown = true;
                    stateBadge.textContent = "DOWN";
                    stateBadge.classList.add("down");
                } else if (isDown && avg > UP_ANGLE) {
                    isDown = false;
                    repCount += 1;
                    repCountEl.textContent = repCount;
                    stateBadge.textContent = "UP";
                    stateBadge.classList.remove("down");

                    // pulse animation
                    repCountEl.classList.remove("pulse");
                    void repCountEl.offsetWidth;
                    repCountEl.classList.add("pulse");
                }
            }

            // draw arm skeleton onto screen canvas (not recorded)
            drawSkeleton(screen);

            // ─── Composite recording frame (video + rep count) ──
            if (isRecording) {
                const rw = recCanvas.width;
                const rh = recCanvas.height;
                recCtx.save();
                // mirror video so recording matches the selfie view
                recCtx.translate(rw, 0);
                recCtx.scale(-1, 1);
                recCtx.drawImage(video, 0, 0, rw, rh);
                recCtx.restore();

                // Rep count – top-right corner
                const fontSize = Math.round(rh * 0.07);
                recCtx.font = `900 ${fontSize}px Inter, system-ui, sans-serif`;
                recCtx.textAlign    = "right";
                recCtx.textBaseline = "top";
                // shadow for legibility
                recCtx.shadowColor   = "rgba(0,0,0,0.7)";
                recCtx.shadowBlur    = 8;
                recCtx.shadowOffsetX = 2;
                recCtx.shadowOffsetY = 2;
                recCtx.fillStyle = "#00ff88";
                recCtx.fillText(repCount.toString(), rw - 24, 24);
                recCtx.shadowBlur = 0;
            }
        }
    }

    animFrameId = requestAnimationFrame(detect);
}

// ═════════════════════════════════════════════════════════════
//  Button handlers
// ═════════════════════════════════════════════════════════════
recordBtn.addEventListener("click", () => {
    if (!isRecording) {
        // ── START RECORDING ──
        if (!poseReady) {
            statusEl.textContent = "Still loading — please wait…";
            return;
        }

        // Hide save button from any previous recording
        saveBtn.classList.add("hidden");

        isRecording = true;
        isDown      = false;
        recordBtn.classList.add("active");
        statusEl.textContent = "Recording — tracking active";
        statusEl.classList.remove("ready", "error");

        // Start video recording (raw webcam only, no skeleton)
        startMediaRecorder();
    } else {
        // ── STOP RECORDING ──
        isRecording = false;

        // Stop video recording (triggers onstop → shows save btn)
        stopMediaRecorder();

        recordBtn.classList.remove("active");
        stateBadge.textContent = "UP";
        stateBadge.classList.remove("down");
        statusEl.textContent = "Stopped — save your recording or record again";
        statusEl.classList.remove("ready");
    }
});

resetBtn.addEventListener("click", () => {
    repCount = 0;
    isDown   = false;
    repCountEl.textContent = "0";
    stateBadge.textContent = "UP";
    stateBadge.classList.remove("down");
    statusEl.textContent = isRecording
        ? "Counter reset — tracking active"
        : "Counter reset — press record to start";
});

saveBtn.addEventListener("click", () => {
    if (!savedBlobUrl) return;

    const a = document.createElement("a");
    a.href     = savedBlobUrl;
    a.download = `pushups_${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    statusEl.textContent = "Video saved!";
    statusEl.classList.add("ready");
});

// ═════════════════════════════════════════════════════════════
//  Boot sequence – webcam + AI model load in parallel
// ═════════════════════════════════════════════════════════════
Promise.all([startWebcam(), initPoseLandmarker()]).then(async () => {
    if (poseLandmarker && webcamStream) {
        poseReady = true;
        // Populate camera dropdown (labels available after getUserMedia grant)
        await populateCameras();
        // Start continuous pose detection loop (skeleton always visible)
        animFrameId = requestAnimationFrame(detect);
        console.log("Webcam and PoseLandmarker ready — pose detection running.");
    }
});
