/**
 * Phase 2 frontend — talks to the Phase 1 FastAPI backend.
 *
 * Modes:
 *   detect   → POST /detect   → canvas boxes (score > 0.7 already filtered server-side)
 *   caption  → POST /caption  → sentence
 *   classify → POST /classify → prediction + confidence bar (P(dog) on a cat←→dog axis)
 *
 * Local Phase 2: API at http://127.0.0.1:8000
 * Docker Phase 3: same-origin /api proxy (nginx) — auto-detected below.
 */

(() => {
  "use strict";

  // API base: ?api= override → <meta name="api-base"> → localhost:8000 (Phase 2 default).
  // Docker (Phase 3) will change the meta content to "/api" (nginx reverse proxy).
  const params = new URLSearchParams(window.location.search);
  const metaApi = document.querySelector('meta[name="api-base"]')?.content?.trim();
  const API = params.get("api") || metaApi || "http://127.0.0.1:8000";

  const SCORE_MIN = 0.7; // belt-and-suspenders; backend already filters

  const MODE_META = {
    detect: {
      hint: "DETR (facebook/detr-resnet-50) — draws boxes for objects with score > 0.7.",
      run: "Run detection",
      busy: "Detecting objects… (CPU, a few seconds)",
      path: "/detect",
    },
    caption: {
      hint: "BLIP (Salesforce/blip-image-captioning-base) — generates a short English caption.",
      run: "Run caption",
      busy: "Generating caption… (CPU, a few seconds)",
      path: "/caption",
    },
    classify: {
      hint: "Cat vs Dog — my from-scratch CNN (catdog.pt), trained on 275 images.",
      run: "Run classification",
      busy: "Classifying…",
      path: "/classify",
    },
  };

  // --- DOM ---
  const fileInput = document.getElementById("file-input");
  const dropzone = document.getElementById("dropzone");
  const dropzoneEmpty = document.getElementById("dropzone-empty");
  const previewWrap = document.getElementById("preview-wrap");
  const previewImg = document.getElementById("preview-img");
  const canvas = document.getElementById("detect-canvas");
  const fileNameEl = document.getElementById("file-name");
  const clearBtn = document.getElementById("clear-btn");
  const runBtn = document.getElementById("run-btn");
  const runLabel = document.getElementById("run-label");
  const runSpinner = document.getElementById("run-spinner");
  const modeHint = document.getElementById("mode-hint");
  const statusEl = document.getElementById("status");
  const errorEl = document.getElementById("error");
  const apiBaseLabel = document.getElementById("api-base-label");

  const detectPlaceholder = document.getElementById("detect-placeholder");
  const detectList = document.getElementById("detect-list");
  const captionPlaceholder = document.getElementById("caption-placeholder");
  const captionText = document.getElementById("caption-text");
  const classifyPlaceholder = document.getElementById("classify-placeholder");
  const classifyCard = document.getElementById("classify-card");
  const classifyPrediction = document.getElementById("classify-prediction");
  const classifyConfidence = document.getElementById("classify-confidence");
  const classifyBar = document.getElementById("classify-bar");
  const pDogNote = document.getElementById("p-dog-note");

  let currentFile = null;
  let objectUrl = null;
  let mode = "detect";
  let busy = false;
  /** @type {{ detections: any[], image_width: number, image_height: number } | null} */
  let lastDetect = null;

  apiBaseLabel.textContent = API;

  // --- Helpers ---
  function setError(message) {
    if (!message) {
      errorEl.classList.add("hidden");
      errorEl.textContent = "";
      return;
    }
    errorEl.textContent = message;
    errorEl.classList.remove("hidden");
  }

  function setStatus(text, isBusy = false) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("is-busy", Boolean(isBusy));
  }

  function setBusy(on) {
    busy = on;
    runBtn.disabled = on || !currentFile;
    runSpinner.classList.toggle("hidden", !on);
    fileInput.disabled = on;
    clearBtn.disabled = on || !currentFile;
  }

  function resetResults() {
    lastDetect = null;
    detectPlaceholder.classList.remove("hidden");
    detectList.classList.add("hidden");
    detectList.innerHTML = "";
    captionPlaceholder.classList.remove("hidden");
    captionText.classList.add("hidden");
    captionText.textContent = "";
    classifyPlaceholder.classList.remove("hidden");
    classifyCard.classList.add("hidden");
    canvas.classList.add("hidden");
    previewImg.classList.remove("hidden");
    setError("");
    setStatus("");
  }

  function showPreview(file) {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    previewImg.onload = () => {
      // Keep canvas sized later when we draw; for now show the photo.
      canvas.classList.add("hidden");
      previewImg.classList.remove("hidden");
    };
    previewImg.src = objectUrl;
    dropzoneEmpty.classList.add("hidden");
    previewWrap.classList.remove("hidden");
    dropzone.classList.add("has-image");
    fileNameEl.textContent = file.name;
    clearBtn.disabled = false;
    runBtn.disabled = busy;
    resetResults();
  }

  function clearFile() {
    currentFile = null;
    fileInput.value = "";
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    previewImg.removeAttribute("src");
    previewWrap.classList.add("hidden");
    dropzoneEmpty.classList.remove("hidden");
    dropzone.classList.remove("has-image");
    fileNameEl.textContent = "No file selected";
    clearBtn.disabled = true;
    runBtn.disabled = true;
    resetResults();
  }

  function setMode(next) {
    mode = next;
    document.querySelectorAll(".tab").forEach((tab) => {
      const active = tab.dataset.mode === mode;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll(".result-pane").forEach((pane) => {
      pane.classList.toggle("is-active", pane.dataset.pane === mode);
    });
    const meta = MODE_META[mode];
    modeHint.textContent = meta.hint;
    runLabel.textContent = meta.run;

    if (mode === "detect" && lastDetect) {
      // Re-show the overlay if we already ran detection on this image.
      drawDetections(lastDetect.detections, lastDetect.image_width, lastDetect.image_height);
    } else {
      canvas.classList.add("hidden");
      previewImg.classList.remove("hidden");
    }
  }

  /**
   * Draw the image + boxes on the canvas.
   * Server boxes are in original pixel space (image_width × image_height).
   * We size the canvas to the *displayed* image box and scale coordinates.
   */
  function drawDetections(detections, imageWidth, imageHeight) {
    const img = previewImg;
    // Use the laid-out size of the <img> as the canvas CSS + bitmap size.
    const displayW = img.clientWidth;
    const displayH = img.clientHeight;
    if (!displayW || !displayH) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(displayW * dpr);
    canvas.height = Math.round(displayH * dpr);
    canvas.style.width = `${displayW}px`;
    canvas.style.height = `${displayH}px`;

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, displayW, displayH);
    ctx.drawImage(img, 0, 0, displayW, displayH);

    const sx = displayW / imageWidth;
    const sy = displayH / imageHeight;

    const visible = detections.filter((d) => d.score > SCORE_MIN);

    visible.forEach((det, i) => {
      const [x, y, w, h] = det.box;
      const rx = x * sx;
      const ry = y * sy;
      const rw = w * sx;
      const rh = h * sy;

      // Distinct but readable stroke per box
      const hue = (i * 47) % 360;
      ctx.strokeStyle = `hsl(${hue} 70% 45%)`;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(rx, ry, rw, rh);

      const label = `${det.label} ${(det.score * 100).toFixed(0)}%`;
      ctx.font = "600 13px DM Sans, system-ui, sans-serif";
      const padX = 6;
      const padY = 4;
      const textW = ctx.measureText(label).width;
      const boxH = 20;
      let labelY = ry - boxH - 2;
      if (labelY < 0) labelY = ry + 2;

      ctx.fillStyle = `hsl(${hue} 70% 35% / 0.92)`;
      ctx.fillRect(rx, labelY, textW + padX * 2, boxH);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, rx + padX, labelY + boxH - padY - 1);
    });

    previewImg.classList.add("hidden");
    canvas.classList.remove("hidden");

    // Side list
    detectPlaceholder.classList.add("hidden");
    detectList.classList.remove("hidden");
    detectList.innerHTML = "";
    if (!visible.length) {
      detectList.innerHTML = "<li>No detections above score 0.7.</li>";
      return;
    }
    visible
      .slice()
      .sort((a, b) => b.score - a.score)
      .forEach((det) => {
        const li = document.createElement("li");
        li.innerHTML = `<span>${escapeHtml(det.label)}</span><span class="score">${(det.score * 100).toFixed(1)}%</span>`;
        detectList.appendChild(li);
      });
  }

  function showCaption(text) {
    captionPlaceholder.classList.add("hidden");
    captionText.classList.remove("hidden");
    captionText.textContent = text;
  }

  function showClassify(data) {
    classifyPlaceholder.classList.add("hidden");
    classifyCard.classList.remove("hidden");

    const pred = data.prediction;
    classifyPrediction.textContent = pred;
    classifyPrediction.classList.toggle("is-cat", pred === "cat");
    classifyPrediction.classList.toggle("is-dog", pred === "dog");
    classifyConfidence.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

    // Bar marker sits at P(dog): 0% = certain cat, 100% = certain dog.
    const pDog = data.p_dog;
    classifyBar.style.left = `${Math.min(100, Math.max(0, pDog * 100))}%`;
    pDogNote.textContent = `P(dog) = ${pDog.toFixed(4)}  ·  threshold 0.5`;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function runCurrentMode() {
    if (!currentFile || busy) return;
    const meta = MODE_META[mode];
    setError("");
    setBusy(true);
    setStatus(meta.busy, true);

    const body = new FormData();
    body.append("file", currentFile);

    try {
      const res = await fetch(`${API}${meta.path}`, {
        method: "POST",
        body,
      });

      let payload = null;
      const raw = await res.text();
      try {
        payload = JSON.parse(raw);
      } catch {
        payload = { detail: raw || res.statusText };
      }

      if (!res.ok) {
        const detail = payload.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : detail?.message || detail?.error || `Request failed (${res.status})`;
        throw new Error(msg);
      }

      if (mode === "detect") {
        lastDetect = {
          detections: payload.detections || [],
          image_width: payload.image_width,
          image_height: payload.image_height,
        };
        drawDetections(lastDetect.detections, lastDetect.image_width, lastDetect.image_height);
        setStatus(`Found ${(payload.detections || []).length} object(s) with score > 0.7.`);
      } else if (mode === "caption") {
        // Caption mode uses the plain preview image (no canvas overlay).
        canvas.classList.add("hidden");
        previewImg.classList.remove("hidden");
        showCaption(payload.caption || "");
        setStatus("Caption ready.");
      } else {
        canvas.classList.add("hidden");
        previewImg.classList.remove("hidden");
        showClassify(payload);
        setStatus("Classification ready.");
      }
    } catch (err) {
      const message =
        err.name === "TypeError"
          ? `Could not reach API at ${API}. Is the backend running?`
          : err.message || String(err);
      setError(message);
      setStatus("");
    } finally {
      setBusy(false);
    }
  }

  // --- Events ---
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    currentFile = file;
    showPreview(file);
  });

  clearBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    clearFile();
  });

  runBtn.addEventListener("click", () => runCurrentMode());

  // Drag & drop
  ["dragenter", "dragover"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (!file || !file.type.startsWith("image/")) {
      setError("Please drop a JPEG, PNG, or WebP image.");
      return;
    }
    currentFile = file;
    showPreview(file);
  });

  // Clicking the preview (when has-image) opens the file picker again.
  previewWrap.addEventListener("click", () => {
    if (!busy) fileInput.click();
  });

  // Redraw boxes on resize if canvas is visible
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    if (canvas.classList.contains("hidden")) return;
    clearTimeout(resizeTimer);
    // We don't keep last payload globally — resize just keeps CSS; full redraw needs re-run.
    // Acceptable for Phase 2; user can click Run again after a big resize.
  });

  setMode("detect");
})();
