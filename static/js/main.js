/* =============================================================================
   AI 大模型评测平台 — 交互脚本
   ============================================================================= */

// Chart.js 全局默认配置
Chart.defaults.color = '#a1a1aa';
Chart.defaults.borderColor = '#2a2a2e';
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

// ---------------------------------------------------------------------------
// 工具函数
// ---------------------------------------------------------------------------
function getColor(index) {
  const colors = ['#6366f1', '#22c55e', '#f59e0b', '#3b82f6', '#ef4444', '#a855f7', '#ec4899'];
  return colors[index % colors.length];
}

// ---------------------------------------------------------------------------
// 仪表盘 — 全局雷达图
// ---------------------------------------------------------------------------
function renderRadarChart(canvasId, models, benchmarks) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = benchmarks.map(b => b.name);
  const datasets = models.map((m, i) => ({
    label: m.name,
    data: labels.map(l => m.scores[benchmarks.find(b => b.name === l).id]),
    borderColor: m.color || getColor(i),
    backgroundColor: (m.color || getColor(i)) + '20',
    borderWidth: 2,
    pointRadius: 3,
    pointBackgroundColor: m.color || getColor(i),
  }));

  new Chart(ctx, {
    type: 'radar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { padding: 16, usePointStyle: true, pointStyle: 'circle' },
        },
      },
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          ticks: { stepSize: 20, font: { size: 10 }, backdropColor: 'transparent' },
          grid: { color: '#2a2a2e' },
          angleLines: { color: '#2a2a2e' },
          pointLabels: { font: { size: 11, weight: '500' } },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// 仪表盘 — 模型平均分柱状图
// ---------------------------------------------------------------------------
function renderBarChart(canvasId, models) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = models.map(m => m.name);
  const avgs = models.map(m => {
    const vals = Object.values(m.scores);
    return Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 10) / 10;
  });
  const colors = models.map(m => m.color);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: '平均分',
        data: avgs,
        backgroundColor: colors.map(c => c + '80'),
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 4,
        barPercentage: 0.55,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: { beginAtZero: true, max: 100, grid: { color: '#2a2a2e' } },
        x: { grid: { display: false } },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// 模型页 — 各 Benchmark 分组柱状图
// ---------------------------------------------------------------------------
function renderGroupedBar(canvasId, models, benchmarks) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = benchmarks.map(b => b.name);
  const datasets = models.map((m, i) => ({
    label: m.name,
    data: labels.map(l => m.scores[benchmarks.find(b => b.name === l).id]),
    backgroundColor: m.color + 'cc',
    borderColor: m.color,
    borderWidth: 1,
    borderRadius: 3,
    barPercentage: 0.7,
    categoryPercentage: 0.7,
  }));

  new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { padding: 14, usePointStyle: true, pointStyle: 'rectRounded' },
        },
      },
      scales: {
        y: { beginAtZero: true, max: 100, grid: { color: '#2a2a2e' } },
        x: { grid: { display: false } },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Benchmark 页 — 各模型在该 Benchmark 上的得分对比
// ---------------------------------------------------------------------------
function renderBenchmarkChart(canvasId, benchmarkId, models) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const filtered = models.filter(m => m.scores[benchmarkId] !== undefined);
  const labels = filtered.map(m => m.name);
  const data = filtered.map(m => m.scores[benchmarkId]);
  const colors = filtered.map(m => m.color);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: '得分',
        data,
        backgroundColor: colors.map(c => c + '90'),
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 5,
        barPercentage: 0.5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.parsed.x} / 100`,
          },
        },
      },
      scales: {
        x: { beginAtZero: true, max: 100, grid: { color: '#2a2a2e' } },
        y: { grid: { display: false } },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// 对比页 — 双轴对比图
// ---------------------------------------------------------------------------
function renderCompareChart(canvasId, modelA, modelB, benchmarks) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = benchmarks.map(b => b.name);
  const dataA = labels.map(l => modelA.scores[benchmarks.find(b => b.name === l).id]);
  const dataB = labels.map(l => modelB.scores[benchmarks.find(b => b.name === l).id]);

  new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [
        {
          label: modelA.name,
          data: dataA,
          borderColor: modelA.color,
          backgroundColor: modelA.color + '20',
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: modelA.color,
        },
        {
          label: modelB.name,
          data: dataB,
          borderColor: modelB.color,
          backgroundColor: modelB.color + '20',
          borderWidth: 2,
          pointRadius: 4,
          pointBackgroundColor: modelB.color,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { padding: 16, usePointStyle: true, pointStyle: 'circle', font: { size: 12 } },
        },
      },
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          ticks: { stepSize: 20, font: { size: 10 }, backdropColor: 'transparent' },
          grid: { color: '#2a2a2e' },
          angleLines: { color: '#2a2a2e' },
          pointLabels: { font: { size: 11, weight: '500' } },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Benchmark 页 — 点击展开详情
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.benchmark-card').forEach(card => {
    card.addEventListener('click', function () {
      const id = this.dataset.benchmarkId;
      const target = document.getElementById('benchmark-detail-' + id);
      if (!target) return;
      const isOpen = target.style.display !== 'none';
      // 关闭所有
      document.querySelectorAll('.benchmark-detail').forEach(el => el.style.display = 'none');
      target.style.display = isOpen ? 'none' : 'block';
      // 滚动到
      if (!isOpen) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
});
