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
// 对比页 — 能力维度雷达图（按 category 聚合）
// ---------------------------------------------------------------------------
function renderCompareChart(canvasId, modelA, modelB, benchmarks) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // 按 category 聚合各模型分数
  function aggByCategory(model, benchmarks) {
    var groups = {};
    var counts = {};
    Object.keys(model.scores || {}).forEach(function(bid) {
      var bm = benchmarks.find(function(b) { return b.id === bid; });
      if (!bm) return;
      var cat = bm.category || '其他';
      if (!groups[cat]) { groups[cat] = 0; counts[cat] = 0; }
      groups[cat] += model.scores[bid];
      counts[cat]++;
    });
    var result = {};
    Object.keys(groups).forEach(function(cat) {
      result[cat] = Math.round((groups[cat] / counts[cat]) * 10) / 10;
    });
    return { scores: result, counts: counts };
  }

  var aggA = aggByCategory(modelA, benchmarks);
  var aggB = aggByCategory(modelB, benchmarks);

  // 合并两个模型的维度标签 & 排序
  var allDims = {};
  Object.keys(aggA.scores).forEach(function(k) { allDims[k] = true; });
  Object.keys(aggB.scores).forEach(function(k) { allDims[k] = true; });
  // 自定义排序
  var dimOrder = ['知识理解', '数学推理', '代码能力', '推理能力', '综合能力', '中文专项', '医疗专业', '安全合规', '多模态', '多语言', '长上下文', '红队/对抗', '自定义', '其他'];
  var labels = Object.keys(allDims).sort(function(a,b) {
    var ia = dimOrder.indexOf(a); if (ia < 0) ia = 99;
    var ib = dimOrder.indexOf(b); if (ib < 0) ib = 99;
    return ia - ib;
  });

  var dataA = labels.map(function(l) { return aggA.scores[l] !== undefined ? aggA.scores[l] : null; });
  var dataB = labels.map(function(l) { return aggB.scores[l] !== undefined ? aggB.scores[l] : null; });

  // 为每个模型创建径向渐变背景
  function makeGradient(color) {
    var g = ctx.createRadialGradient(0, 0, 0, 0, 0, 120);
    var c = color || '#6366f1';
    g.addColorStop(0, c + '55');
    g.addColorStop(0.6, c + '25');
    g.addColorStop(1, c + '08');
    return g;
  }

  new Chart(ctx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [
        {
          label: modelA.name,
          data: dataA,
          borderColor: modelA.color,
          backgroundColor: makeGradient(modelA.color),
          borderWidth: 2.5,
          pointRadius: 4,
          pointHoverRadius: 7,
          pointBackgroundColor: modelA.color,
          pointBorderColor: '#0a0a0b',
          pointBorderWidth: 2,
        },
        {
          label: modelB.name,
          data: dataB,
          borderColor: modelB.color,
          backgroundColor: makeGradient(modelB.color),
          borderWidth: 2.5,
          pointRadius: 4,
          pointHoverRadius: 7,
          pointBackgroundColor: modelB.color,
          pointBorderColor: '#0a0a0b',
          pointBorderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: {
        duration: 1000,
        easing: 'easeOutQuart'
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            padding: 20,
            usePointStyle: true,
            pointStyle: 'circle',
            font: { size: 13, weight: '600' },
            color: '#c4c4cc',
          }
        },
        tooltip: {
          backgroundColor: '#1a1a1e',
          titleColor: '#f4f4f5',
          bodyColor: '#a1a1aa',
          borderColor: '#2a2a2e',
          borderWidth: 1,
          padding: 12,
          cornerRadius: 8,
          displayColors: true,
          callbacks: {
            label: function(ctx) {
              var modelName = ctx.dataset.label;
              var avg = ctx.parsed.r;
              // 找该维度的 benchmark 数量
              var dim = ctx.label;
              var count = 0;
              var bms = benchmarks.filter(function(b) { return (b.category || '其他') === dim; });
              bms.forEach(function(b) {
                var model = modelA.name === modelName ? modelA : modelB;
                if (model.scores[b.id] !== undefined) count++;
              });
              return modelName + ': ' + avg + '% (' + count + ' 项)';
            }
          }
        },
      },
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          backgroundColor: 'rgba(26, 26, 30, 0.4)',
          grid: {
            color: [
              'rgba(255,255,255,0.04)',
              'rgba(255,255,255,0.08)',
              'rgba(255,255,255,0.05)',
              'rgba(255,255,255,0.09)',
              'rgba(255,255,255,0.04)',
            ],
            circular: true,
          },
          angleLines: { color: 'rgba(255,255,255,0.07)' },
          ticks: {
            stepSize: 20,
            font: { size: 9 },
            color: '#6b6b76',
            backdropColor: 'transparent',
            display: false,
          },
          pointLabels: {
            font: { size: 12, weight: '600', family: '-apple-system, sans-serif' },
            color: '#c4c4cc',
          }
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
