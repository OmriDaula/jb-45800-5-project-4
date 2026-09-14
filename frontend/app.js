/**
 * Frontend — talks to the FastAPI backend.
 *
 * Modes (Cat vs Dog is default — core deliverable):
 *   classify → POST /classify → prediction + confidence bar
 *   detect   → POST /detect   → canvas boxes (server already filtered by threshold)
 *   caption  → POST /caption  → sentence
 */

(() => {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const metaApi = document.querySelector('meta[name="api-base"]')?.content?.trim();
  const API = params.get("api") || metaApi || "http://127.0.0.1:8000";

  const MODE_META = {
    classify: {
      hint: "My trained model — from-scratch CNN (catdog.pt) on 275 images. Classifies cat vs dog.",
      run: "Run classification",
      busy: "Classifying…",
      path: "/classify",
    },
    detect: {
      hint: "Pretrained extension — DETR (facebook/detr-resnet-50). Bounding boxes from the server.",
      run: "Run detection",
      busy: "Detecting objects… (CPU, a few seconds)",
      path: "/detect",
    },
    caption: {
      hint: "Pretrained extension — BLIP (Salesforce/blip-image-captioning-base). Short English caption.",
      run: "Run caption",
      busy: "Generating caption… (CPU, a few seconds)",
      path: "/caption",
    },
  };

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
  const tabs = Array.from(document.querySelectorAll(".tab"));

  let currentFile = null;
  let objectUrl = null;
  let mode = "classify";
  let busy = false;
  /** @type {{ detections: any[], image_width: number, image_height: number, threshold?: number } | null} */
  let lastDetect = null;

  apiBaseLabel.textContent = API;

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
    tabs.forEach((tab) => {
      const active = tab.dataset.mode === mode;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
      tab.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll(".result-pane").forEach((pane) => {
      const active = pane.dataset.pane === mode;
      pane.classList.toggle("is-active", active);
      pane.hidden = !active;
    });
    const meta = MODE_META[mode];
    modeHint.textContent = meta.hint;
    runLabel.textContent = meta.run;

    if (mode === "detect" && lastDetect) {
      drawDetections(
        lastDetect.detections,
        lastDetect.image_width,
        lastDetect.image_height,
        lastDetect.threshold
      );
    } else {
      canvas.classList.add("hidden");
      previewImg.classList.remove("hidden");
    }
  }

  /**
   * Draw image + boxes. Server already filtered by threshold — render all returned detections.
   */
  function drawDetections(detections, imageWidth, imageHeight, threshold) {
    const img = previewImg;
    const displayW = img.clientWidth;
    const displayH = img.clientHeight;
    if (!displayW || !displayH || !imageWidth || !imageHeight) return;

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
    const list = detections || [];

    list.forEach((det, i) => {
      const [x, y, w, h] = det.box;
      const rx = x * sx;
      const ry = y * sy;
      const rw = w * sx;
      const rh = h * sy;

      const hue = (i * 47) % 360;
      ctx.strokeStyle = `hsl(${hue} 70% 45%)`;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(rx, ry, rw, rh);

      const label = `${det.label} ${(det.score * 100).toFixed(0)}%`;
      ctx.font = "600 13px system-ui, sans-serif";
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

    detectPlaceholder.classList.add("hidden");
    detectList.classList.remove("hidden");
    detectList.innerHTML = "";
    if (!list.length) {
      const t = threshold != null ? threshold : "?";
      detectList.innerHTML = `<li>No detections above score ${t}.</li>`;
      return;
    }
    list
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
      const res = await fetch(`${API}${meta.path}`, { method: "POST", body });
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
          threshold: payload.threshold,
        };
        drawDetections(
          lastDetect.detections,
          lastDetect.image_width,
          lastDetect.image_height,
          lastDetect.threshold
        );
        const t = payload.threshold != null ? payload.threshold : "";
        setStatus(
          `Found ${(payload.detections || []).length} object(s)` +
            (t !== "" ? ` (score > ${t}).` : ".")
        );
      } else if (mode === "caption") {
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

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => setMode(tab.dataset.mode));
    tab.addEventListener("keydown", (e) => {
      const i = tabs.indexOf(tab);
      let next = -1;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (i + 1) % tabs.length;
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (i - 1 + tabs.length) % tabs.length;
      if (e.key === "Home") next = 0;
      if (e.key === "End") next = tabs.length - 1;
      if (next < 0) return;
      e.preventDefault();
      tabs[next].focus();
      setMode(tabs[next].dataset.mode);
    });
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

  previewWrap.addEventListener("click", () => {
    if (!busy) fileInput.click();
  });

  // Debounced redraw using stored detections — no extra API call.
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    if (!lastDetect || mode !== "detect") return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      drawDetections(
        lastDetect.detections,
        lastDetect.image_width,
        lastDetect.image_height,
        lastDetect.threshold
      );
    }, 100);
  });

  setMode("classify");
})();
