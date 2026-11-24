// Chart helpers. Colors come from CSS custom properties so light/dark mode stay in sync.
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// API address. The local server answers /api/... with query strings; the static GitHub Pages
// build stores each answer as a file: /api/ticker/KBC.BR/chart?range=1d -> <root>/api/ticker/KBC.BR/chart-1d.json
function apiUrl(path, params = {}) {
  const site = window.SITE || { root: '', static: false };
  if (!site.static) {
    const q = new URLSearchParams(params).toString();
    return path + (q ? '?' + q : '');
  }
  const vals = Object.values(params);
  return site.root + path + (vals.length ? '-' + vals.join('-') : '') + '.json';
}

// Link to a page of this site (same rules as url() in app/web/main.py).
function pageUrl(path) {
  const site = window.SITE || { root: '', static: false };
  return site.static ? site.root + path.replace(/\/?$/, '/') : path;
}

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
  const res = await (await fetch(apiUrl(`/api/ticker/${encodeURIComponent(symbol)}/history`, { days: 180 }))).json();
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

// Price chart with 1D / 1W / 1M buttons, drawn from the prices the daily jobs store.
function rangeChart(canvas, symbol, seg, note, foot) {
  let chart = null, seq = 0;
  const buttons = [...seg.querySelectorAll('button[data-range]')];
  const fmtTime = (ms) => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const fmtDayTime = (ms) => new Date(ms).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' });
  const fmtDay = (ms) => new Date(ms).toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' });

  async function show(range) {
    const mine = ++seq;
    buttons.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.range === range)));
    try { localStorage.setItem('chartRange', range); } catch (e) { /* storage unavailable */ }
    const res = await (await fetch(apiUrl(`/api/ticker/${encodeURIComponent(symbol)}/chart`, { range }))).json();
    if (mine !== seq) return;               // a newer click won
    if (chart) { chart.destroy(); chart = null; }
    if (!res.points.length) {
      canvas.style.display = 'none';
      note.textContent = '';
      foot.textContent = 'No prices stored for this stock yet. They are saved by the morning and evening jobs.';
      return;
    }
    canvas.style.display = '';
    const intraday = res.kind === 'intraday';
    const labels = res.points.map(p => intraday
      ? (range === '1d' ? fmtTime(p.t) : fmtDayTime(p.t))
      : fmtDay(Date.parse(p.d + 'T12:00:00')));
    const up = res.change_pct == null ? null : res.change_pct >= 0;
    const color = up == null ? css('--series-1') : up ? css('--good') : css('--bad');
    document.getElementById('pc-swatch')?.style.setProperty('background', color);
    const datasets = [{ label: 'Price', data: res.points.map(p => p.close), borderColor: color, borderWidth: 2,
                        pointRadius: intraday ? 0 : 2, pointHoverRadius: 4, tension: 0.1 }];
    if (res.base != null) {
      datasets.push({ label: 'Previous close', data: labels.map(() => res.base), borderColor: css('--muted'),
                      borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, pointHoverRadius: 0 });
    }
    const opts = baseOptions();
    opts.interaction = { mode: 'index', intersect: false };
    opts.scales.x.ticks.maxTicksLimit = range === '1d' ? 8 : 6;
    opts.plugins.tooltip.callbacks = { label: (ctx) => `${ctx.dataset.label} ${ctx.parsed.y.toFixed(2)}` };
    chart = new Chart(canvas, { type: 'line', data: { labels, datasets }, options: opts });

    const span = { '1d': 'last trading day', '1w': 'last 5 trading days', '1m': 'last month' }[range];
    note.textContent = res.change_pct == null ? `· ${res.last.toFixed(2)}`
      : `· ${res.last.toFixed(2)} (${res.change_pct >= 0 ? '+' : ''}${res.change_pct.toFixed(2)}% over the ${span})`;
    const lastAt = intraday ? fmtDayTime(res.points.at(-1).t) : res.points.at(-1).d;
    foot.textContent = (intraday ? '15-minute prices' : 'Daily closes')
      + (range !== '1m' && !intraday ? ' (no 15-minute prices stored yet)' : '')
      + `, saved by the morning and evening jobs; last price ${lastAt}. Times are in your local time.`;
  }

  buttons.forEach(b => b.addEventListener('click', () => show(b.dataset.range)));
  let start = '1d';
  try { start = localStorage.getItem('chartRange') || '1d'; } catch (e) { /* storage unavailable */ }
  show(buttons.some(b => b.dataset.range === start) ? start : '1d');
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
  const res = await (await fetch(apiUrl('/api/accuracy/series', { horizon, days }))).json();
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

// ---------- Size of the move: rolling move accuracy + predicted vs actual scatter ----------

async function moveCharts(lineCanvas, scatterCanvas, horizon, days) {
  const [series, moves] = await Promise.all([
    fetch(apiUrl('/api/accuracy/series', { horizon, days })).then(r => r.json()),
    fetch(apiUrl('/api/accuracy/moves', { horizon, days })).then(r => r.json()),
  ]);
  const empty = (c) => { c.parentElement.innerHTML = '<div class="empty">No scored predictions with an expected move yet.</div>'; };
  if (!moves.points.length) { empty(lineCanvas); empty(scatterCanvas); return; }

  // Rolling move accuracy per market, weighted by the number of calls per date.
  const keys = ['sp500', 'europe', 'bel20'];
  const colors = { sp500: css('--series-1'), europe: css('--series-2'), bel20: css('--series-3') };
  const allDates = [...new Set(keys.flatMap(k => series.series[k].map(s => s.d)))].sort();
  const win = horizon === 'daily' ? 10 : 4;
  const datasets = keys.map(k => {
    const byDate = Object.fromEntries(series.series[k].map(s => [s.d, s]));
    const pts = allDates.map(d => {
      const s = byDate[d];
      return s && s.move_n ? { n: s.move_n, rate: s.move_acc } : { n: 0, rate: 0 };
    });
    return { label: series.labels[k], data: rolling(pts, win), borderColor: colors[k], backgroundColor: colors[k],
             borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, spanGaps: true, tension: 0.2 };
  });
  const opts = baseOptions();
  opts.interaction = { mode: 'index', intersect: false };
  opts.scales.y.min = 0; opts.scales.y.max = 100;
  opts.scales.y.ticks = { callback: (v) => v + '%' };
  opts.plugins.tooltip.callbacks = { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y == null ? '–' : ctx.parsed.y.toFixed(1) + '%'}` };
  new Chart(lineCanvas, { type: 'line', data: { labels: allDates, datasets }, options: opts });

  // Scatter: x = predicted move, y = actual move. Symmetric axes, clipped at the 98th percentile.
  const pts = moves.points.map(p => ({ x: p.exp, y: p.act, p }));
  const abs = pts.flatMap(p => [Math.abs(p.x), Math.abs(p.y)]).sort((a, b) => a - b);
  const lim = Math.max(1, Math.ceil(abs[Math.floor(abs.length * 0.98)] || 1));
  const wrongShade = {
    id: 'wrongShade',
    beforeDatasetsDraw(chart) {
      const { ctx, chartArea: a, scales: { x, y } } = chart;
      const x0 = x.getPixelForValue(0), y0 = y.getPixelForValue(0);
      ctx.save();
      ctx.fillStyle = css('--bad'); ctx.globalAlpha = 0.07;
      ctx.fillRect(x0, y0, a.right - x0, a.bottom - y0);   // predicted up, went down
      ctx.fillRect(a.left, a.top, x0 - a.left, y0 - a.top); // predicted down, went up
      ctx.restore();
    },
  };
  const sopts = baseOptions();
  sopts.interaction = { mode: 'nearest', intersect: true };
  sopts.scales.x = { type: 'linear', min: -lim, max: lim, grid: { color: css('--grid') },
                     title: { display: true, text: 'Predicted move' }, ticks: { callback: (v) => v + '%' } };
  sopts.scales.y = { min: -lim, max: lim, grid: { color: css('--grid') }, border: { display: false },
                     title: { display: true, text: 'Actual move' }, ticks: { callback: (v) => v + '%' } };
  sopts.plugins.tooltip.callbacks = {
    label: (ctx) => {
      const p = ctx.raw.p; if (!p) return '';
      return `${p.symbol} ${p.d}: expected ${p.exp >= 0 ? '+' : ''}${p.exp.toFixed(2)}%, actual ${p.act >= 0 ? '+' : ''}${p.act.toFixed(2)}%`;
    },
  };
  new Chart(scatterCanvas, {
    type: 'scatter',
    data: { datasets: [
      { label: 'Exactly right', type: 'line', data: [{ x: -lim, y: -lim }, { x: lim, y: lim }],
        borderColor: css('--muted'), borderDash: [4, 4], borderWidth: 1, pointRadius: 0 },
      { label: 'Calls', data: pts, backgroundColor: css('--series-1') + '66', borderWidth: 0,
        pointRadius: pts.length > 1500 ? 2 : 3, pointHoverRadius: 6 },
    ] },
    options: sopts,
    plugins: [wrongShade],
  });
}

// ---------- Learning page: hit rate against trivial baselines ----------

async function learningChart(canvas) {
  const rows = await (await fetch(apiUrl('/api/learning/daily'))).json();
  if (!rows.length) { canvas.parentElement.innerHTML = '<div class="empty">No scored days yet.</div>'; return; }
  const pct = (k) => rows.map(r => r[k] == null ? null : r[k] * 100);
  const line = (label, key, color, dash) => ({ label, data: pct(key), borderColor: color, backgroundColor: color,
    borderWidth: dash ? 1.5 : 2, borderDash: dash || [], pointRadius: rows.length > 40 ? 0 : 3, pointHoverRadius: 5,
    tension: 0.2, spanGaps: true });
  const opts = baseOptions();
  opts.interaction = { mode: 'index', intersect: false };
  opts.scales.y.min = 0; opts.scales.y.max = 100;
  opts.scales.y.ticks = { callback: (v) => v + '%' };
  opts.plugins.tooltip.callbacks = {
    label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y == null ? '–' : ctx.parsed.y.toFixed(1) + '%'}`,
    afterBody: (items) => `Playbook ${rows[items[0].dataIndex].version}`,
  };
  new Chart(canvas, { type: 'line', data: { labels: rows.map(r => r.date), datasets: [
    line('Your hit rate', 'hit', css('--series-1')),
    line('Always UP', 'always_up', css('--muted'), [5, 4]),
    line('Everyone follows your lean', 'follow_majority', css('--series-2'), [2, 3]),
    line('Stock selection (beats the market)', 'rel_hit', css('--series-3')),
  ] }, options: opts });
}
