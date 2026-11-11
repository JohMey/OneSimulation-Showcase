// Client-side search over search-index.json. No dependencies, no tracking.
(function () {
  var root = document.documentElement.dataset.root || '';
  var input = document.getElementById('q');
  var box = document.getElementById('results');
  if (!input || !box) return;
  var entries = null, loading = null, active = -1;

  function load() {
    if (entries) return Promise.resolve(entries);
    if (!loading) {
      loading = fetch(root + 'search-index.json').then(function (r) { return r.json(); }).then(function (data) {
        entries = data.map(function (e) {
          return { e: e, title: e.t.toLowerCase(), text: (e.t + ' ' + e.p + ' ' + e.x).toLowerCase() };
        });
        return entries;
      });
    }
    return loading;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; });
  }

  function search(query) {
    var words = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    var hits = [];
    for (var i = 0; i < entries.length; i++) {
      var item = entries[i], score = 0, ok = true;
      for (var w = 0; w < words.length; w++) {
        var word = words[w];
        if (item.text.indexOf(word) < 0) { ok = false; break; }
        if (item.title === word) score += 50;
        else if (item.title.indexOf(word) === 0) score += 20;
        else if (item.title.indexOf(word) >= 0) score += 10;
        else score += 1;
      }
      if (ok) hits.push({ score: score - item.title.length / 200, e: item.e });
    }
    hits.sort(function (a, b) { return b.score - a.score; });
    return hits.slice(0, 12);
  }

  function render(hits, query) {
    active = -1;
    if (!query.trim()) { box.className = ''; box.innerHTML = ''; return; }
    if (!hits.length) { box.innerHTML = '<p>No results for “' + escapeHtml(query) + '”.</p>'; box.className = 'open'; return; }
    box.innerHTML = hits.map(function (h) {
      return '<a role="option" href="' + root + h.e.u + '"><strong>' + escapeHtml(h.e.t) + '</strong><small>' +
        escapeHtml(h.e.p + (h.e.x ? ' · ' + h.e.x : '')) + '</small></a>';
    }).join('');
    box.className = 'open';
  }

  input.addEventListener('focus', load);
  input.addEventListener('input', function () {
    var query = input.value;
    load().then(function () { if (input.value === query) render(search(query), query); });
  });
  input.addEventListener('keydown', function (event) {
    var links = box.querySelectorAll('a');
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      if (!links.length) return;
      event.preventDefault();
      if (active >= 0) links[active].classList.remove('active');
      active = (active + (event.key === 'ArrowDown' ? 1 : -1) + links.length) % links.length;
      links[active].classList.add('active');
      links[active].scrollIntoView({ block: 'nearest' });
    } else if (event.key === 'Enter' && links.length) {
      event.preventDefault();
      window.location.href = links[active >= 0 ? active : 0].getAttribute('href');
    } else if (event.key === 'Escape') {
      box.className = '';
      input.blur();
    }
  });
  document.addEventListener('click', function (event) {
    if (!box.contains(event.target) && event.target !== input) box.className = '';
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === '/' && document.activeElement !== input && !/input|textarea/i.test(document.activeElement.tagName)) {
      event.preventDefault();
      input.focus();
    }
  });
})();
