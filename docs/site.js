const nav = document.querySelector(".page-nav");
const navLinks = Array.from(document.querySelectorAll(".page-nav a"));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

function updateNavigation() {
  if (!nav) return;

  nav.classList.toggle("visible", window.scrollY > 320);

  const current = sections
    .filter((section) => section.getBoundingClientRect().top < window.innerHeight * 0.38)
    .pop();

  navLinks.forEach((link) => {
    const target = document.querySelector(link.getAttribute("href"));
    link.classList.toggle("active", target === current);
  });
}

document.addEventListener("scroll", updateNavigation, { passive: true });
window.addEventListener("load", updateNavigation);

const resultState = {
  benchmark: "alfworld",
  scale: "4b",
  expanded: new Set(),
};

const resultLabels = {
  alfworld: {
    title: "ALFWorld",
    seen: ["Pick&Place", "Transform", "Examine"],
    unseen: ["Pick&Place", "Transform", "Examine"],
  },
  search: {
    title: "Search",
    seen: ["NQ", "HotpotQA"],
    unseen: ["PopQA", "TriviaQA", "2WikiMHQA", "MuSiQue", "Bamboogle"],
  },
};

const resultData = {
  alfworld: {
    "4b": [
      { type: "Baseline", method: "ReAct", seen: 37.1, unseen: 32.8, overall: 35.0, seenDetail: [54.7, 28.1, 28.6], unseenDetail: [30.8, 28.9, 66.7] },
      { type: "w/ Evolve", method: "Best", seen: 54.5, unseen: 52.7, overall: 53.6, seenDetail: [60.5, 44.6, 79.5], unseenDetail: [45.5, 58.2, 55.6] },
      { type: "w/ Skill", method: "Best", seen: 51.0, unseen: 63.4, overall: 57.5, seenDetail: [61.6, 43.1, 61.5], unseenDetail: [50.4, 68.0, 74.1] },
      { type: "w/ RL", method: "GRPO", seen: 74.8, unseen: 75.9, overall: 75.3, seenDetail: [93.8, 62.3, 53.8], unseenDetail: [86.2, 72.0, 68.5] },
      { type: "w/ RL", method: "MemRL", seen: 81.2, unseen: 79.1, overall: 80.2, seenDetail: [89.8, 75.8, 71.8], unseenDetail: [78.9, 76.6, 94.4] },
      { type: "w/ RL", method: "EvolveR", seen: 76.7, unseen: 77.9, overall: 77.3, seenDetail: [70.6, 85.2, 61.5], unseenDetail: [69.9, 77.9, 94.4] },
      { type: "w/ RL", method: "INSPO", seen: 76.4, unseen: 79.6, overall: 78.0, seenDetail: [78.0, 77.6, 61.5], unseenDetail: [82.9, 87.0, 83.3] },
      { type: "w/ RL", method: "SkillRL", seen: 85.7, unseen: 82.1, overall: 83.9, seenDetail: [89.8, 85.9, 61.5], unseenDetail: [69.9, 86.9, 88.9] },
      { type: "w/ RL", method: "ReSkill", seen: 90.0, unseen: 89.6, overall: 89.8, seenDetail: [91.6, 88.7, 76.9], unseenDetail: [87.4, 89.6, 96.3], highlight: true },
    ],
    "8b": [
      { type: "Baseline", method: "ReAct", seen: 59.3, unseen: 64.9, overall: 62.0, seenDetail: [73.4, 44.8, 64.3], unseenDetail: [69.2, 65.0, 66.7] },
      { type: "w/ Evolve", method: "Best", seen: 66.2, unseen: 65.7, overall: 65.9, seenDetail: [79.1, 58.3, 61.5], unseenDetail: [57.7, 76.4, 57.4] },
      { type: "w/ Skill", method: "Best", seen: 66.4, unseen: 74.6, overall: 70.7, seenDetail: [79.1, 56.4, 64.1], unseenDetail: [76.4, 80.4, 72.2] },
      { type: "w/ RL", method: "GRPO", seen: 80.7, unseen: 81.6, overall: 81.1, seenDetail: [93.8, 72.1, 66.7], unseenDetail: [82.1, 81.8, 79.6] },
      { type: "w/ RL", method: "MemRL", seen: 83.1, unseen: 81.6, overall: 82.3, seenDetail: [86.4, 78.5, 84.6], unseenDetail: [82.1, 83.3, 75.9] },
      { type: "w/ RL", method: "EvolveR", seen: 83.3, unseen: 82.8, overall: 83.1, seenDetail: [88.7, 79.8, 74.4], unseenDetail: [70.7, 89.2, 83.3] },
      { type: "w/ RL", method: "INSPO", seen: 83.2, unseen: 87.8, overall: 85.5, seenDetail: [90.4, 79.2, 71.8], unseenDetail: [88.1, 89.2, 81.5] },
      { type: "w/ RL", method: "SkillRL", seen: 89.0, unseen: 82.6, overall: 85.8, seenDetail: [95.5, 82.8, 87.2], unseenDetail: [82.9, 78.3, 94.4] },
      { type: "w/ RL", method: "ReSkill", seen: 90.2, unseen: 95.3, overall: 92.7, seenDetail: [96.5, 81.4, 94.9], unseenDetail: [99.3, 91.2, 100.0], highlight: true },
    ],
  },
  search: {
    "4b": [
      { type: "Baseline", method: "ReAct", seen: 25.8, unseen: 27.8, overall: 27.2, seenDetail: [21.3, 30.3], unseenDetail: [29.4, 40.7, 30.7, 9.3, 30.4] },
      { type: "w/ Evolve", method: "Best", seen: 32.9, unseen: 35.4, overall: 34.5, seenDetail: [30.4, 35.6], unseenDetail: [39.4, 57.3, 34.9, 11.1, 41.3] },
      { type: "w/ Skill", method: "Best", seen: 33.6, unseen: 34.9, overall: 34.5, seenDetail: [31.6, 36.0], unseenDetail: [42.1, 55.1, 30.9, 10.8, 42.1] },
      { type: "w/ RL", method: "GRPO", seen: 50.2, unseen: 42.0, overall: 44.6, seenDetail: [48.0, 52.4], unseenDetail: [42.9, 66.7, 40.9, 18.3, 40.3] },
      { type: "w/ RL", method: "MemRL", seen: 50.1, unseen: 41.9, overall: 44.2, seenDetail: [48.7, 51.4], unseenDetail: [48.2, 65.3, 35.8, 15.9, 44.3] },
      { type: "w/ RL", method: "EvolveR", seen: 49.6, unseen: 43.1, overall: 45.0, seenDetail: [49.3, 49.8], unseenDetail: [46.6, 65.7, 43.4, 16.7, 43.2] },
      { type: "w/ RL", method: "INSPO", seen: 51.1, unseen: 42.3, overall: 45.1, seenDetail: [48.6, 53.6], unseenDetail: [46.5, 63.0, 44.0, 15.8, 42.4] },
      { type: "w/ RL", method: "SkillRL", seen: 51.2, unseen: 42.4, overall: 45.1, seenDetail: [51.0, 51.3], unseenDetail: [48.5, 65.0, 39.7, 15.3, 44.8] },
      { type: "w/ RL", method: "ReSkill", seen: 52.6, unseen: 45.4, overall: 47.6, seenDetail: [51.6, 53.7], unseenDetail: [49.7, 66.9, 45.7, 18.3, 47.5], highlight: true },
    ],
    "8b": [
      { type: "Baseline", method: "ReAct", seen: 31.1, unseen: 32.8, overall: 32.2, seenDetail: [30.2, 32.0], unseenDetail: [37.2, 52.1, 30.8, 9.8, 35.2] },
      { type: "w/ Evolve", method: "Best", seen: 32.9, unseen: 34.1, overall: 33.8, seenDetail: [29.5, 36.2], unseenDetail: [37.5, 57.2, 27.5, 11.9, 40.3] },
      { type: "w/ Skill", method: "Best", seen: 36.0, unseen: 35.5, overall: 35.2, seenDetail: [32.8, 39.2], unseenDetail: [42.6, 57.0, 30.6, 11.2, 40.8] },
      { type: "w/ RL", method: "GRPO", seen: 49.9, unseen: 43.1, overall: 45.3, seenDetail: [48.0, 51.9], unseenDetail: [51.8, 63.7, 40.7, 16.2, 43.5] },
      { type: "w/ RL", method: "MemRL", seen: 51.6, unseen: 43.8, overall: 46.2, seenDetail: [48.3, 54.8], unseenDetail: [49.8, 65.7, 41.2, 19.0, 42.1] },
      { type: "w/ RL", method: "EvolveR", seen: 51.2, unseen: 44.5, overall: 46.6, seenDetail: [49.3, 53.1], unseenDetail: [51.2, 65.7, 43.9, 16.7, 45.6] },
      { type: "w/ RL", method: "INSPO", seen: 51.9, unseen: 42.3, overall: 45.0, seenDetail: [48.7, 55.2], unseenDetail: [48.2, 64.7, 39.4, 16.9, 42.1] },
      { type: "w/ RL", method: "SkillRL", seen: 52.4, unseen: 45.3, overall: 47.5, seenDetail: [48.7, 56.1], unseenDetail: [47.2, 64.7, 47.4, 20.8, 48.3] },
      { type: "w/ RL", method: "ReSkill", seen: 53.7, unseen: 48.0, overall: 49.8, seenDetail: [49.0, 58.3], unseenDetail: [52.3, 68.7, 47.6, 22.6, 50.4], highlight: true },
    ],
  },
};

function formatScore(value) {
  return `${value.toFixed(1)}%`;
}

function renderDetailBlock(title, labels, values) {
  return `
    <div class="detail-block">
      <h4>${title}</h4>
      <div class="detail-grid">
        ${labels.map((label, index) => `
          <div class="detail-metric">
            <span>${label}</span>
            <strong>${formatScore(values[index])}</strong>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderResultsTable() {
  const mount = document.getElementById("results-table");
  if (!mount) return;

  const rows = resultData[resultState.benchmark][resultState.scale];
  const labels = resultLabels[resultState.benchmark];
  const grpo = rows.find((row) => row.method === "GRPO")?.overall ?? 0;
  const body = rows.map((row) => {
    const key = `${resultState.benchmark}-${resultState.scale}-${row.type}-${row.method}`;
    const expanded = resultState.expanded.has(key);
    const gain = row.overall - grpo;
    const gainClass = gain >= 0 ? "gain-positive" : "gain-negative";
    const gainText = row.method === "GRPO" ? "0.0" : `${gain > 0 ? "+" : ""}${gain.toFixed(1)}`;

    return `
      <tr class="${row.highlight ? "highlight" : ""}">
        <td>
          <div class="method-cell">
            <button class="expand-row" type="button" data-expand-key="${key}" aria-expanded="${expanded}" aria-label="${expanded ? "Hide" : "Show"} ${row.method} details">${expanded ? "−" : "+"}</button>
            <div>
              <span class="method-name">${row.method}</span>
              <span class="method-type">${row.type}</span>
            </div>
          </div>
        </td>
        <td>${formatScore(row.seen)}</td>
        <td>${formatScore(row.unseen)}</td>
        <td><strong>${formatScore(row.overall)}</strong></td>
        <td class="${gainClass}">${gainText}</td>
      </tr>
      ${expanded ? `
        <tr class="details-row">
          <td colspan="5">
            <div class="details-panel">
              ${renderDetailBlock(`${labels.title} seen`, labels.seen, row.seenDetail)}
              ${renderDetailBlock(`${labels.title} unseen`, labels.unseen, row.unseenDetail)}
            </div>
          </td>
        </tr>
      ` : ""}
    `;
  }).join("");

  mount.innerHTML = `
    <table class="results-table">
      <thead>
        <tr>
          <th>Method</th>
          <th>Seen</th>
          <th>Unseen</th>
          <th>Overall</th>
          <th>Δ vs GRPO</th>
        </tr>
      </thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

document.querySelectorAll("[data-result-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    const group = button.dataset.resultTab;
    resultState[group] = button.dataset.value;
    resultState.expanded.clear();

    document.querySelectorAll(`[data-result-tab="${group}"]`).forEach((candidate) => {
      candidate.classList.toggle("active", candidate === button);
    });

    renderResultsTable();
  });
});

document.getElementById("results-table")?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-expand-key]");
  if (!button) return;

  const key = button.dataset.expandKey;
  if (resultState.expanded.has(key)) {
    resultState.expanded.delete(key);
  } else {
    resultState.expanded.add(key);
  }

  renderResultsTable();
});

renderResultsTable();
