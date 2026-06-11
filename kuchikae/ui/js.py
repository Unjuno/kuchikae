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
  return null;
}

function findAudioButton(wrap, candidates) {
  if (!wrap) return null;
  const buttons = Array.from(wrap.querySelectorAll("button"));
  return buttons.find((b) => {
    const haystack = [
      b.textContent || "",
      b.getAttribute("aria-label") || "",
      b.getAttribute("title") || "",
      b.className || "",
    ].join(" ").toLowerCase();
    return candidates.some((c) => haystack.includes(c));
  }) || null;
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
  const recBtn = findAudioButton(wrap, ["record", "録音", "start", "microphone", "mic"]);
  if (recBtn) {
    recBtn.click();
    return;
  }
  const hint = document.getElementById('ptt-hint');
  if (hint) hint.textContent = '録音ボタンが見つかりません。通常モードで録音してください。';
  pttState = 0;
  btn.className = 'ptt-idle';
  label.textContent = '押して話す';
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
  const stopBtn = findAudioButton(wrap, ["stop", "停止", "done", "完了"]);
  if (stopBtn) {
    stopBtn.click();
    return;
  }
  const hint = document.getElementById('ptt-hint');
  if (hint) hint.textContent = '停止ボタンが見つかりません。通常モードで録音してください。';

  if (pttTimer) clearTimeout(pttTimer);
  pttTimer = setTimeout(() => {
    const h = document.getElementById('ptt-hint');
    if (h) h.textContent = 'ボタンを押しながら話す、離すと自動変換';
  }, 5000);
}
</script>
"""
