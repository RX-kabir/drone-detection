const dropzone    = document.getElementById("dropzone");
const fileInput   = document.getElementById("fileInput");
const previewBox  = document.getElementById("previewBox");
const previewImg  = document.getElementById("previewImg");
const previewVid  = document.getElementById("previewVid");
const clearBtn    = document.getElementById("clearBtn");
const detectBtn   = document.getElementById("detectBtn");
const confSlider  = document.getElementById("confSlider");
const confVal     = document.getElementById("confVal");
const progressWrap= document.getElementById("progressWrap");
const resultsCard = document.getElementById("resultsCard");

let selectedFile = null;

// ── Confidence slider ──────────────────────────────────────
confSlider.addEventListener("input", () => {
  confVal.textContent = parseFloat(confSlider.value).toFixed(2);
});

// ── Drag & drop ───────────────────────────────────────────
dropzone.addEventListener("dragover", e => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () =>
  dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── File handler ──────────────────────────────────────────
function handleFile(file) {
  selectedFile = file;
  const url    = URL.createObjectURL(file);
  const isVid  = file.type.startsWith("video");

  dropzone.style.display  = "none";
  previewBox.style.display = "block";

  if (isVid) {
    previewImg.style.display = "none";
    previewVid.style.display = "block";
    previewVid.src = url;
  } else {
    previewVid.style.display = "none";
    previewImg.style.display = "block";
    previewImg.src = url;
  }

  detectBtn.disabled = false;
}

// ── Clear ──────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src  = "";
  previewVid.src  = "";
  previewBox.style.display  = "none";
  dropzone.style.display    = "block";
  detectBtn.disabled        = true;
  resultsCard.style.display = "none";
  progressWrap.style.display= "none";
});

// ── Detect ─────────────────────────────────────────────────
detectBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  detectBtn.disabled         = true;
  progressWrap.style.display = "block";
  resultsCard.style.display  = "none";

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("conf", confSlider.value);

  try {
    const response = await fetch("/detect", {
      method: "POST",
      body:   formData,
    });
    const data = await response.json();

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    showResults(data);

  } catch (err) {
    alert("Request failed: " + err.message);
  } finally {
    detectBtn.disabled         = false;
    progressWrap.style.display = "none";
  }
});

// ── Show Results ──────────────────────────────────────────
function showResults(data) {
  resultsCard.style.display = "block";

  document.getElementById("humanCount").textContent = data.humans ?? 0;
  document.getElementById("carCount").textContent   = data.cars   ?? 0;
  document.getElementById("totalCount").textContent = data.total  ?? (data.humans + data.cars);
  document.getElementById("timeMs").textContent     = data.time_ms ?? "–";

  const resultImg = document.getElementById("resultImg");
  const resultVid = document.getElementById("resultVid");
  const dlBtn     = document.getElementById("downloadBtn");

  dlBtn.href = data.result_url;

  if (data.type === "video") {
    resultImg.style.display = "none";
    resultVid.style.display = "block";
    resultVid.src = data.result_url;
    document.getElementById("timeMs").textContent = `${data.frames}f`;
  } else {
    resultVid.style.display = "none";
    resultImg.style.display = "block";
    resultImg.src = data.result_url + "?t=" + Date.now();
  }

  // Breakdown pills
  const breakdownWrap = document.getElementById("breakdownWrap");
  const breakdownGrid = document.getElementById("breakdownGrid");
  if (data.breakdown && Object.keys(data.breakdown).length > 0) {
    breakdownGrid.innerHTML = "";
    Object.entries(data.breakdown)
      .sort((a,b) => b[1]-a[1])
      .forEach(([cls, cnt]) => {
        const pill = document.createElement("div");
        pill.className = "breakdown-pill";
        pill.innerHTML = `${cls}: <span>${cnt}</span>`;
        breakdownGrid.appendChild(pill);
      });
    breakdownWrap.style.display = "block";
  }

  resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
}