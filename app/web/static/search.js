// Ticker search with autocomplete: type a ticker or (part of) a company name.
(function () {
  const form = document.getElementById('ticker-search');
  if (!form) return;
  const input = form.querySelector('input');
  const list = form.querySelector('[role="listbox"]');
  let tickers = null, loading = null, items = [], active = -1;

  // Lower-case and strip accents, so "nestle" finds Nestlé and "moet" finds Moët.
  const norm = (s) => (s || '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function load() {
    loading = loading || fetch('/api/tickers').then((r) => r.json()).then((rows) => {
      tickers = rows.map((t) => ({ ...t, _sym: norm(t.symbol), _name: norm(t.name) }));
      return tickers;
    }).catch(() => { loading = null; return []; });
    return loading;
  }

  // Best matches first: exact ticker, ticker prefix, name word prefix, then anything containing it.
  function rank(t, q) {
    const base = t._sym.split('.')[0];
    if (t._sym === q || base === q) return 0;
    if (t._sym.startsWith(q)) return 1;
    if (t._name.startsWith(q)) return 2;
    if (t._name.split(/[\s\-&.,()/]+/).some((w) => w.startsWith(q))) return 3;
    if (t._sym.includes(q) || t._name.includes(q)) return 4;
    return -1;
  }

  function search(q) {
    q = norm(q.trim());
    if (!q) return tickers.slice();                       // empty box: browse the whole list
    return tickers.map((t) => [rank(t, q), t]).filter(([r]) => r >= 0)
      .sort((a, b) => a[0] - b[0] || a[1].name.localeCompare(b[1].name)).map(([, t]) => t);
  }

  function render() {
    const q = input.value.trim();
    items = search(q).slice(0, q ? 50 : 600);
    active = q && items.length ? 0 : -1;
    if (!items.length) {
      list.innerHTML = `<li class="none">No stock matches “${esc(q)}”</li>`;
    } else {
      const head = q ? '' : `<li class="none">All ${items.length} stocks · type to filter</li>`;
      list.innerHTML = head + items.map((t, i) => `
        <li role="option" id="opt-${i}" data-i="${i}" aria-selected="${i === active}">
          <span class="sym">${esc(t.symbol)}</span>
          <span class="nm">${esc(t.name)}<small>${esc(t.universes.join(' · '))}</small></span>
          <span class="px">${t.close != null ? t.close.toFixed(2) : ''}</span>
        </li>`).join('');
    }
    open(true);
    highlight();
  }

  function highlight() {
    list.querySelectorAll('[role="option"]').forEach((li) => {
      const on = Number(li.dataset.i) === active;
      li.setAttribute('aria-selected', String(on));
      if (on) li.scrollIntoView({ block: 'nearest' });
    });
    input.setAttribute('aria-activedescendant', active >= 0 ? `opt-${active}` : '');
  }

  function open(on) {
    list.hidden = !on;
    input.setAttribute('aria-expanded', String(on));
  }

  function go(symbol) { location = '/ticker/' + encodeURIComponent(symbol); }

  input.addEventListener('focus', async () => { await load(); render(); });
  input.addEventListener('input', async () => { await load(); render(); });
  input.addEventListener('keydown', (e) => {
    if (list.hidden && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) { render(); return; }
    if (e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(active + 1, items.length - 1); highlight(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(active - 1, 0); highlight(); }
    else if (e.key === 'Escape') { open(false); }
  });
  list.addEventListener('mousedown', (e) => {          // mousedown fires before the input loses focus
    const li = e.target.closest('[role="option"]');
    if (li) { e.preventDefault(); go(items[Number(li.dataset.i)].symbol); }
  });
  input.addEventListener('blur', () => setTimeout(() => open(false), 100));

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const typed = input.value.trim();
    if (!typed) return;
    await load();
    if (active >= 0 && items[active]) return go(items[active].symbol);
    const hits = tickers && search(typed);
    go(hits && hits.length ? hits[0].symbol : typed.toUpperCase());
  });
})();
