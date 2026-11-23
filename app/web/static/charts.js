// Chart helpers. Colors come from CSS custom properties so light/dark mode stay in sync.
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function baseOptions() {
  const grid = css('--grid'), text = css('--text-2');
  Chart.defaults.color = text;
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: 'nearest', intersect: false },
    plugins: { legend: { display: false }, tooltip: { padding: 10, displayColors: true } },
    scales: {
      x: { grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 0 } },
      y: { grid: { color: grid }, border: { display: false } },
    },
  };
}

async function priceChart(canvas, symbol, horizon = 'daily') {
  const res = await (await fetch(`/api/ticker/${encodeURIComponent(symbol)}/history?days=180`)).json();
  if (!res.prices.length) { canvas.parentElement.innerHTML = '<div class="empty">No price data yet.</div>'; return; }
  const labels = res.prices.map(p => p.date);
  const closeByDate = Object.fromEntries(res.prices.map(p => [p.date, p.close]));
  const preds = res.predictions.filter(p => p.horizon === horizon);
  const groups = { right: [], wrong: [], pending: [] };
  for (const p of preds) {
    const y = closeByDate[p.target_date] ?? p.base_close * (1 + (p.expected_move_pct || 0) / 100);
    const key = p.correct === 1 ? 'right' : p.correct === 0 ? 'wrong' : 'pending';
    groups[key].push({ x: p.target_date, y, p });
    if (!(p.target_date in closeByDate)) labels.push(p.target_date);
  }
  labels.sort();
  const uniq = [...new Set(labels)];
  const marker = (data, label, color) => ({
    type: 'scatter', label, data, backgroundColor: color, borderColor: css('--surface'), borderWidth: 2,
    pointStyle: 'triangle', pointRadius: 7, pointHoverRadius: 9,
    rotation: data.map(d => d.p.direction === 'UP' ? 0 : 180),
  });
  const opts = baseOptions();
  opts.plugins.tooltip.callbacks = {
    label: (ctx) => {
      if (!ctx.raw || !ctx.raw.p) return `Close ${ctx.parsed.y.toFixed(2)}`;
      const p = ctx.raw.p;
      const res = p.correct === 1 ? 'right' : p.correct === 0 ? 'wrong' : 'pending';
      return `${p.direction} call, ${(p.confidence * 100).toFixed(0)}% confidence (${res})`;
    },
  };
  new Chart(canvas, {
    data: {
      labels: uniq,
      datasets: [
        { type: 'line', label: 'Close', data: uniq.map(d => closeByDate[d] ?? null), borderColor: css('--series-1'),
          borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, spanGaps: true, tension: 0.1 },
        marker(groups.right, 'Right call', css('--good')),
        marker(groups.wrong, 'Wrong call', css('--bad')),
        marker(groups.pending, 'Pending call', css('--pending')),
      ],
    },
    options: opts,
  });
}

async function intradayChart(canvas, symbol, baseClose, direction) {
  const res = await (await fetch(`/api/ticker/${encodeURIComponent(symbol)}/intraday`)).json();
  if (!res.bars || !res.bars.length) {
    canvas.parentElement.innerHTML = `<div class="empty">${res.error ? 'Could not reach Yahoo Finance for live prices.' : 'No trades yet today (market closed or not open yet).'}</div>`;
    return null;
  }
  const labels = res.bars.map(b => b.t.slice(11, 16));
  const last = res.bars[res.bars.length - 1].close;
  const opts = baseOptions();
  opts.plugins.tooltip.callbacks = { label: (ctx) => `${ctx.dataset.label} ${ctx.parsed.y.toFixed(2)}` };
  new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Price', data: res.bars.map(b => b.close), borderColor: css('--series-1'), borderWidth: 2,
          pointRadius: 0, tension: 0.1 },
        { label: 'Base close', data: labels.map(() => baseClose), borderColor: css('--muted'), borderWidth: 1.5,
          borderDash: [5, 4], pointRadius: 0 },
      ],
    },
    options: opts,
  });
  return { last, change: (last / baseClose - 1) * 100 };
}

function rolling(series, window) {
  // Hit rate over the last `window` dates, weighted by number of calls.
  const out = [];
  for (let i = 0; i < series.length; i++) {
    const slice = series.slice(Math.max(0, i - window + 1), i + 1);
    const n = slice.reduce((a, s) => a + s.n, 0);
    const hits = slice.reduce((a, s) => a + s.rate * s.n, 0);
    out.push(n ? (hits / n) * 100 : null);
  }
  return out;
}

async function accuracyCharts(lineCanvas, calibCanvas, horizon, days) {
  const res = await (await fetch(`/api/accuracy/series?horizon=${horizon}&days=${days}`)).json();
  const keys = ['sp500', 'europe', 'bel20'];
  const colors = { sp500: css('--series-1'), europe: css('--series-2'), bel20: css('--series-3') };
  const allDates = [...new Set(keys.flatMap(k => res.series[k].map(s => s.d)))].sort();
  if (!allDates.length) {
    lineCanvas.parentElement.innerHTML = '<div class="empty">No scored predictions yet.</div>';
    calibCanvas.parentElement.innerHTML = '<div class="empty">No scored predictions yet.</div>';
    return;
  }
  const win = horizon === 'daily' ? 10 : 4;
  const datasets = keys.map(k => {
    const byDate = Object.fromEntries(res.series[k].map(s => [s.d, s]));
    const series = allDates.map(d => byDate[d] || { n: 0, rate: 0 });
    return { label: res.labels[k], data: rolling(series, win), borderColor: colors[k], backgroundColor: colors[k],
             borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, spanGaps: true, tension: 0.2 };
  });
  datasets.push({ label: 'Coin flip (50%)', data: allDates.map(() => 50), borderColor: css('--muted'),
                  borderWidth: 1, borderDash: [4, 4], pointRadius: 0 });
  const opts = baseOptions();
  opts.interaction = { mode: 'index', intersect: false };
  opts.scales.y.min = 0; opts.scales.y.max = 100;
  opts.scales.y.ticks = { callback: (v) => v + '%' };
  opts.plugins.tooltip.callbacks = { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y == null ? '–' : ctx.parsed.y.toFixed(1) + '%'}` };
  new Chart(lineCanvas, { type: 'line', data: { labels: allDates, datasets }, options: opts });

  const cal = res.calibration.filter(c => c.n > 0);
  const copts = baseOptions();
  copts.interaction = { mode: 'nearest', intersect: true };
  copts.scales.x = { type: 'linear', min: 50, max: 100, title: { display: true, text: 'Stated confidence' },
                     grid: { display: false }, ticks: { callback: (v) => v + '%' } };
  copts.scales.y.min = 0; copts.scales.y.max = 100;
  copts.scales.y.title = { display: true, text: 'Actually right' };
  copts.scales.y.ticks = { callback: (v) => v + '%' };
  copts.plugins.tooltip.callbacks = { label: (ctx) => ctx.raw.n ? `Said ${ctx.raw.x.toFixed(0)}%, right ${ctx.raw.y.toFixed(0)}% (n=${ctx.raw.n})` : '' };
  new Chart(calibCanvas, {
    type: 'scatter',
    data: { datasets: [
      { label: 'Perfect calibration', type: 'line', data: [{ x: 50, y: 50 }, { x: 100, y: 100 }],
        borderColor: css('--muted'), borderDash: [4, 4], borderWidth: 1, pointRadius: 0 },
      { label: 'Claude', data: cal.map(c => ({ x: c.conf * 100, y: c.rate * 100, n: c.n })),
        backgroundColor: css('--series-1'), borderColor: css('--surface'), borderWidth: 2,
        pointRadius: cal.map(c => Math.max(5, Math.min(14, Math.sqrt(c.n)))), pointHoverRadius: 10 },
    ] },
    options: copts,
  });
}
