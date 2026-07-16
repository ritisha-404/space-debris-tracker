const objectListEl = document.getElementById("object-list");
const riskBtn = document.getElementById("risk-btn");
const riskOutputEl = document.getElementById("risk-output");
const ctx = document.getElementById("trajectoryChart");

let debrisData = [];
let chart = null;
let activeId = null;

async function loadDebris() {
  const res = await fetch("/api/debris");
  debrisData = await res.json();

  objectListEl.innerHTML = "";
  debrisData.forEach((obj) => {
    const chip = document.createElement("div");
    chip.className = "object-chip";
    chip.textContent = obj.id;
    chip.onclick = () => selectObject(obj.id);
    chip.dataset.id = obj.id;
    objectListEl.appendChild(chip);
  });

  if (debrisData.length > 0) {
    selectObject(debrisData[0].id);
  }
}

function selectObject(id) {
  activeId = id;
  document.querySelectorAll(".object-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.id === id);
  });

  const obj = debrisData.find((o) => o.id === id);
  if (!obj) return;

  const xs = obj.trajectory.map((p) => p[0]);
  const ys = obj.trajectory.map((p) => p[1]);
  const points = xs.map((x, i) => ({ x, y: ys[i] }));

  if (chart) {
    chart.data.datasets[0].data = points;
    chart.data.datasets[0].label = id;
    chart.update();
  } else {
    chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: id,
            data: points,
            borderColor: "#4da3ff",
            backgroundColor: "rgba(77,163,255,0.15)",
            fill: true,
            tension: 0.3,
            pointRadius: 3,
          },
        ],
      },
      options: {
        scales: {
          x: { title: { display: true, text: "X (km)" }, ticks: { color: "#8b93a7" } },
          y: { title: { display: true, text: "Y (km)" }, ticks: { color: "#8b93a7" } },
        },
        plugins: {
          legend: { labels: { color: "#e6e9f2" } },
        },
      },
    });
  }
}

async function runRiskCheck() {
  riskOutputEl.textContent = "Running screening...";
  const res = await fetch("/api/risk");
  const data = await res.json();

  if (data.error) {
    riskOutputEl.textContent = `Error: ${data.error}`;
    return;
  }

  if (data.close_approaches.length === 0) {
    riskOutputEl.textContent =
      `No close approaches predicted within ${data.threshold_km} km threshold.`;
    return;
  }

  riskOutputEl.innerHTML = data.close_approaches
    .map(
      (f) =>
        `<div class="risk-flag">⚠ ${f.object_a} ↔ ${f.object_b}: ${f.predicted_distance_km} km</div>`
    )
    .join("");
}

riskBtn.addEventListener("click", runRiskCheck);
loadDebris();
