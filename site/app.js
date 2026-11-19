"use strict";

const names = ["Low", "Medium", "High"];
const colors = ["#d84e55", "#dca43d", "#28a084"];
let data = null;
let activeRun = null;
let revealed = 0;
let timer = null;

const el = (id) => document.getElementById(id);
const percent = (n) => (n * 100).toFixed(1) + "%";

function stop() {
  if (timer !== null) clearInterval(timer);
  timer = null;
  el("play").textContent = "Play";
}

function switchRun(id) {
  stop();
  activeRun = data.runs.find((run) => String(run.id) === String(id));
  revealed = 0;
  el("seek").max = String(activeRun.points.length);
  update();
}

function drawTimeline(points) {
  const canvas = el("timeline");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);

  const left = 76;
  const right = 10;
  const visible = Math.min(72, points.length);
  const first = Math.max(0, Math.min(points.length - visible, revealed - 36));
  const last = first + visible;
  const cell = (width - left - right) / visible;
  const tops = [34, 103];
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "#62728a";
  ctx.fillText("Measured", 0, tops[0] + 19);
  ctx.fillText("Predicted", 0, tops[1] + 19);

  for (let i = first; i < last; i++) {
    const x = left + (i - first) * cell;
    const point = points[i];
    for (let row = 0; row < 2; row++) {
      ctx.fillStyle = i < revealed
        ? colors[row === 0 ? point.actual : point.predicted]
        : "#edf2f8";
      ctx.fillRect(x + 1, tops[row], Math.max(cell - 2, 1), 42);
    }
    if (i < revealed && point.actual !== point.predicted) {
      ctx.strokeStyle = "#b83742";
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, tops[1], Math.max(cell - 2, 1), 42);
    }
    if (i === revealed) {
      ctx.strokeStyle = "#2456b7";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, tops[0] - 4, Math.max(cell, 3), 120);
    }
  }
  ctx.fillStyle = "#62728a";
  ctx.fillText("Window " + points[first].next_seq, left, 174);
  const label = "Window " + points[last - 1].next_seq;
  ctx.fillText(label, Math.max(left, width - right - ctx.measureText(label).width), 174);
}

function update() {
  const points = activeRun.points;
  const total = points.length;
  const pending = revealed < total ? points[revealed] : null;
  const previous = revealed > 0 ? points[revealed - 1] : null;
  el("seek").value = String(revealed);
  el("progress-text").textContent = revealed + " of " + total +
    " next-window predictions revealed";
  if (revealed) {
    const correct = points.slice(0, revealed).filter((p) => p.predicted === p.actual).length;
    el("running-accuracy").textContent =
      "This recording so far: " + correct + "/" + revealed + " correct";
  } else {
    el("running-accuracy").textContent = "";
  }
  el("step").disabled = revealed === total;
  el("play").disabled = revealed === total;

  el("throughput").textContent = pending ? pending.current_mbps.toLocaleString() : "—";
  const edges = data.meta.edges_mbps;
  const current = pending
    ? pending.current_mbps < edges[0] ? 0 : pending.current_mbps < edges[1] ? 1 : 2
    : null;
  el("observed-band").textContent =
    pending ? "Current band: " + names[current] : "Recording complete";
  el("window").textContent = pending
    ? pending.seq + " → " + pending.next_seq : "—";
  el("nr-status").textContent = pending ? pending.nr_status : "—";
  el("rsrp").textContent = pending && pending.nr_rsrp !== null
    ? pending.nr_rsrp + " dBm" : "Unavailable";

  const badge = el("prediction");
  badge.className = "band-pill " + (pending ? names[pending.predicted].toLowerCase() : "neutral");
  badge.textContent = pending ? names[pending.predicted] : "Complete";
  el("confidence").textContent = pending
    ? "Model confidence: " + percent(pending.confidence) : "All outcomes revealed";
  el("baseline").textContent = pending
    ? "Last-band baseline: " + names[pending.baseline] : "—";

  const outcome = el("outcome");
  outcome.className = "outcome";
  if (previous) {
    const good = previous.predicted === previous.actual;
    outcome.classList.add(good ? "good" : "bad");
    outcome.textContent = "Window " + previous.next_seq + " measured " +
      previous.next_mbps + " Mbps (" + names[previous.actual] + "). " +
      (good ? "Prediction correct." : "Prediction incorrect.");
  } else {
    outcome.textContent = "The measured next band stays hidden until you reveal it.";
  }
  drawTimeline(points);
}

function fillStaticResults() {
  const test = data.scores.test;
  el("model-f1").textContent = test.model.macro_f1.toFixed(3);
  el("baseline-f1").textContent = test.baseline.macro_f1.toFixed(3);
  el("test-count").textContent = test.rows.toLocaleString();
  el("run-count").textContent = "Across " + test.runs.length + " held-out runs";
  const edges = data.meta.edges_mbps;
  el("thresholds").textContent =
    "Bands derived from training data: Low < " + edges[0] +
    " Mbps; Medium " + edges[0] + " to < " + edges[1] +
    " Mbps; High ≥ " + edges[1] + " Mbps.";

  const body = el("matrix-body");
  body.replaceChildren();
  test.model.confusion_matrix.forEach((row, i) => {
    const tr = document.createElement("tr");
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = names[i];
    tr.appendChild(label);
    row.forEach((count, j) => {
      const td = document.createElement("td");
      td.textContent = count.toLocaleString();
      if (i === j) td.className = "diagonal";
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  const recall = el("recall-bars");
  recall.replaceChildren();
  names.forEach((name) => {
    const value = test.model.per_class[name].recall;
    const row = document.createElement("div");
    row.className = "recall-row";
    const label = document.createElement("span");
    label.textContent = name;
    const track = document.createElement("div");
    track.className = "track";
    const fill = document.createElement("div");
    fill.className = "fill " + name.toLowerCase();
    fill.style.width = (100 * value) + "%";
    track.appendChild(fill);
    const score = document.createElement("strong");
    score.textContent = percent(value);
    row.append(label, track, score);
    recall.appendChild(row);
  });
  el("transition-score").textContent =
    "Band changes in the test set: " + test.transition_windows.toLocaleString() +
    " windows; model accuracy on those changes: " +
    (test.transition_accuracy === null ? "unavailable" : percent(test.transition_accuracy)) + ".";

  const select = el("run-select");
  data.runs.forEach((run) => {
    const option = document.createElement("option");
    option.value = String(run.id);
    option.textContent = "Run " + run.id + " (" + run.points.length + " predictions)";
    select.appendChild(option);
  });
  switchRun(data.runs[0].id);
}

el("run-select").addEventListener("change", (event) => switchRun(event.target.value));
el("step").addEventListener("click", () => {
  if (revealed < activeRun.points.length) {
    revealed++;
    update();
  }
});
el("reset").addEventListener("click", () => {
  stop();
  revealed = 0;
  update();
});
el("seek").addEventListener("input", (event) => {
  stop();
  revealed = Number(event.target.value);
  update();
});
el("play").addEventListener("click", () => {
  if (timer !== null) return stop();
  timer = setInterval(() => {
    if (revealed < activeRun.points.length) {
      revealed++;
      update();
    }
    if (revealed === activeRun.points.length) stop();
  }, Number(el("speed").value));
  el("play").textContent = "Pause";
});
el("speed").addEventListener("change", () => {
  if (timer !== null) {
    stop();
    el("play").click();
  }
});
el("failure").addEventListener("click", () => {
  stop();
  const index = activeRun.points.findIndex(
    (p) => p.actual !== p.baseline && p.predicted !== p.actual
  );
  const fallback = activeRun.points.findIndex((p) => p.predicted !== p.actual);
  if (index < 0 && fallback < 0) {
    el("error").textContent = "No model error in this recording. Try another run.";
    el("error").hidden = false;
    return;
  }
  el("error").hidden = true;
  revealed = (index >= 0 ? index : fallback) + 1;
  update();
});
window.addEventListener("resize", () => {
  if (activeRun) drawTimeline(activeRun.points);
});

fetch("results.json")
  .then((response) => {
    if (!response.ok) throw new Error("Could not load results.json");
    return response.json();
  })
  .then((result) => {
    data = result;
    if (!data.runs || !data.runs.length) throw new Error("No test runs in results.json");
    fillStaticResults();
  })
  .catch((error) => {
    el("error").textContent = error.message +
      ". Run train.py and open this folder through a local HTTP server.";
    el("error").hidden = false;
  });
