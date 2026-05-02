/* ─────────────────────────────────────────────────────
   main.js  –  ObesityAI  Frontend Logic
   ───────────────────────────────────────────────────── */

const ACTIVITY_LABELS = {
  0: "Tidak Aktif",
  1: "Kurang Aktif",
  2: "Cukup Aktif",
  3: "Aktif",
  4: "Sangat Aktif"
};

const PROB_COLORS = {
  "Kurus":      "#63b3ed",
  "Normal":     "#68d391",
  "Kegemukan":  "#f6ad55",
  "Obesitas":   "#fc8181"
};

/* ── BMI live preview ── */
function calcBMI() {
  const t = parseFloat(document.getElementById("tinggi").value);
  const b = parseFloat(document.getElementById("berat").value);
  const el = document.getElementById("bmiLive");
  const val = document.getElementById("bmiVal");
  if (t > 0 && b > 0) {
    const bmi = (b / ((t / 100) ** 2)).toFixed(2);
    val.textContent = bmi;
    el.style.display = "block";
  } else {
    el.style.display = "none";
  }
}

document.getElementById("tinggi").addEventListener("input", calcBMI);
document.getElementById("berat").addEventListener("input", calcBMI);

/* ── Slider label ── */
const slider = document.getElementById("aktivitas");
const sliderVal = document.getElementById("aktivitasVal");
if (slider) {
  slider.addEventListener("input", () => {
    const v = parseInt(slider.value);
    sliderVal.textContent = `${v} — ${ACTIVITY_LABELS[v]}`;
    const pct = (v / 4) * 100;
    slider.style.background =
      `linear-gradient(to right, var(--accent) ${pct}%, var(--surface2) ${pct}%)`;
  });
}

/* ── PREDICT ── */
async function predict() {
  const btn   = document.getElementById("predictBtn");
  const btext = btn.querySelector(".btn-text");
  const bicon = btn.querySelector(".btn-icon");
  const bload = btn.querySelector(".btn-loading");

  // Validate inputs
  const fields = ["usia","tinggi","berat","kalori","tidur"];
  for (const id of fields) {
    const el = document.getElementById(id);
    if (!el.value || isNaN(parseFloat(el.value))) {
      el.focus();
      el.style.borderColor = "#fc8181";
      el.style.boxShadow   = "0 0 0 3px rgba(252,129,129,.2)";
      setTimeout(() => {
        el.style.borderColor = "";
        el.style.boxShadow   = "";
      }, 1500);
      return;
    }
  }

  // Show loading
  btext.style.display = "none";
  bicon.style.display = "none";
  bload.style.display = "flex";
  btn.disabled = true;

  const keluarga = document.querySelector('input[name="keluarga"]:checked').value;

  const payload = {
    usia:      parseFloat(document.getElementById("usia").value),
    tinggi:    parseFloat(document.getElementById("tinggi").value),
    berat:     parseFloat(document.getElementById("berat").value),
    aktivitas: parseFloat(document.getElementById("aktivitas").value),
    kalori:    parseFloat(document.getElementById("kalori").value),
    tidur:     parseFloat(document.getElementById("tidur").value),
    keluarga:  parseFloat(keluarga)
  };

  try {
    const res  = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.success) {
      renderResult(data);
    } else {
      alert("Terjadi kesalahan: " + data.error);
    }
  } catch(e) {
    alert("Gagal menghubungi server.");
  } finally {
    btext.style.display = "";
    bicon.style.display = "";
    bload.style.display = "none";
    btn.disabled = false;
  }
}

/* ── RENDER RESULT ── */
function renderResult(d) {
  document.getElementById("placeholder").style.display = "none";
  const rc = document.getElementById("resultContent");
  rc.style.display = "flex";

  // Badge
  document.getElementById("riskIcon").textContent  = d.icon;
  const lbl = document.getElementById("riskLabel");
  lbl.textContent  = d.result;
  lbl.style.color  = d.color;
  document.getElementById("riskConf").textContent  = `Keyakinan model: ${d.confidence}%`;

  // BMI bar
  document.getElementById("bmiResult").textContent = d.bmi;
  const bmiPct = Math.min(Math.max(((d.bmi - 14) / (40 - 14)) * 100, 2), 98);
  document.getElementById("bmiMarker").style.left = bmiPct + "%";

  // Probability bars
  const probContainer = document.getElementById("probBars");
  probContainer.innerHTML = "";
  const sorted = Object.entries(d.probability).sort((a,b) => b[1]-a[1]);
  sorted.forEach(([cls, pct]) => {
    const row = document.createElement("div");
    row.className = "prob-bar-row";
    row.innerHTML = `
      <div class="prob-label">${cls}</div>
      <div class="prob-track">
        <div class="prob-fill" style="width:0%;background:${PROB_COLORS[cls] || '#667eea'}"></div>
      </div>
      <div class="prob-pct" style="color:${PROB_COLORS[cls] || '#667eea'}">${pct.toFixed(1)}%</div>
    `;
    probContainer.appendChild(row);
    // Animate bar
    setTimeout(() => {
      row.querySelector(".prob-fill").style.width = pct + "%";
    }, 80);
  });

  // Description
  document.getElementById("descBox").textContent = d.desc;

  // Tips
  const tipsList = document.getElementById("tipsList");
  tipsList.innerHTML = "";
  d.tips.forEach(t => {
    const li = document.createElement("li");
    li.textContent = t;
    tipsList.appendChild(li);
  });

  // Scroll to result
  document.getElementById("resultPanel").scrollIntoView({ behavior:"smooth", block:"start" });
}

/* ── RESET ── */
function resetForm() {
  document.getElementById("placeholder").style.display    = "flex";
  document.getElementById("resultContent").style.display  = "none";
}
