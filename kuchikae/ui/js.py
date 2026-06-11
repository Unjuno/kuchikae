"""Kuchikae UI — PTT JavaScript snippet."""

PTT_HTML = """
<div id="ptt-container">
  <button id="ptt-btn"
    class="ptt-idle"
    onmousedown="pttStart(event)"
    onmouseup="pttStop()"
    onmouseleave="pttStop()"
    ontouchstart="pttStart(event)"
    ontouchend="pttStop()">
    <span id="ptt-label">押して話す</span>
  </button>
  <div id="ptt-hint">ボタンを押しながら話す、離すと自動変換</div>
</div>
<script>
let pttState = 0;
let pttTimer = null;

function findBtn(wrap, selector) {
  if (!wrap) return null;
  let b = wrap.querySelector(selector);
  if (b) return b;
  b = wrap.querySelector('button');
  if (b) return b;
  const all = document.querySelectorAll('#simple-audio-wrap button, [id*="simple-audio"] button');
  return all.length > 0 ? all[0] : null;
}

function pttStart(e) {
  if (e) e.preventDefault();
  if (pttState === 1) return;
  pttState = 1;
  const btn = document.getElementById('ptt-btn');
  const label = document.getElementById('ptt-label');
  btn.className = 'ptt-recording';
  label.textContent = '話し終えたら離す';
  document.getElementById('ptt-hint').textContent = '録音中…';

  const wrap = document.getElementById('simple-audio-wrap');
  const recBtn = findBtn(wrap, '.record-button');
  if (recBtn) { recBtn.click(); return; }
  const btns = wrap ? wrap.querySelectorAll('button') : [];
  for (const b of btns) {
    b.click(); break;
  }
}

function pttStop() {
  if (pttState !== 1) return;
  pttState = 0;
  const btn = document.getElementById('ptt-btn');
  const label = document.getElementById('ptt-label');
  btn.className = 'ptt-idle';
  label.textContent = '押して話す';
  document.getElementById('ptt-hint').textContent = '変換中…';

  const wrap = document.getElementById('simple-audio-wrap');
  const stopBtn = findBtn(wrap, '.stop-button');
  if (stopBtn) { stopBtn.click(); return; }
  const btns = wrap ? wrap.querySelectorAll('button') : [];
  for (const b of btns) {
    b.click(); break;
  }

  if (pttTimer) clearTimeout(pttTimer);
  pttTimer = setTimeout(() => {
    const h = document.getElementById('ptt-hint');
    if (h) h.textContent = 'ボタンを押しながら話す、離すと自動変換';
  }, 5000);
}
</script>
"""
