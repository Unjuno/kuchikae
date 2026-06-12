"""Kuchikae UI — PTT JavaScript snippet."""

PTT_HTML = """
<div id="ptt-container" role="region" aria-label="音声録音コントロール">
  <button id="ptt-btn" class="ptt-idle" aria-label="押して話す" aria-pressed="false" type="button">
    <span id="ptt-label">押して話す</span>
  </button>
  <div id="ptt-hint" aria-live="polite">ボタンを押しながら話す、離すと自動変換</div>
</div>
"""

PTT_JS = r"""
(function () {
  let pttState = 0;

  function setHint(text) {
    const hint = document.getElementById('ptt-hint');
    if (hint) hint.textContent = text;
  }

  function setButtonState(recording) {
    const btn = document.getElementById('ptt-btn');
    const label = document.getElementById('ptt-label');
    if (!btn || !label) return;
    if (recording) {
      btn.className = 'ptt-recording';
      btn.setAttribute('aria-label', '録音中');
      btn.setAttribute('aria-pressed', 'true');
      label.textContent = '話し終えたら離す';
    } else {
      btn.className = 'ptt-idle';
      btn.setAttribute('aria-label', '押して話す');
      btn.setAttribute('aria-pressed', 'false');
      label.textContent = '押して話す';
    }
  }

  function findAudioWrap() {
    return document.getElementById('simple-audio-wrap');
  }

  function findControlButton(wrap, name) {
    if (!wrap) return null;
    const target = name.toLowerCase();
    return Array.from(wrap.querySelectorAll('button')).find((b) => {
      const text = (b.textContent || '').trim().toLowerCase();
      const aria = (b.getAttribute('aria-label') || '').toLowerCase();
      const title = (b.getAttribute('title') || '').toLowerCase();
      return text === target || text.includes(target) || aria.includes(target) || title.includes(target);
    }) || null;
  }

  function clickNativeControl(name) {
    const wrap = findAudioWrap();
    const btn = findControlButton(wrap, name);
    if (!btn) return false;
    btn.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, cancelable: true, pointerType: 'mouse' }));
    btn.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, cancelable: true, pointerType: 'mouse' }));
    btn.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, view: window }));
    btn.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true, view: window }));
    btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
    return true;
  }

  function startRecording() {
    if (pttState === 1) return;
    pttState = 1;
    setButtonState(true);
    setHint('録音中…');
    if (!clickNativeControl('record')) {
      pttState = 0;
      setButtonState(false);
      setHint('録音ボタンが見つかりませんでした。');
    }
  }

  function stopRecording() {
    if (pttState !== 1) return;
    pttState = 0;
    setButtonState(false);
    setHint('変換中…');
    if (!clickNativeControl('stop')) {
      setHint('停止ボタンが見つかりませんでした。');
      return;
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (pttState === 0) {
        startRecording();
      } else {
        stopRecording();
      }
    }
  }

  function attachPTTHandlers() {
    const btn = document.getElementById('ptt-btn');
    if (!btn || btn.dataset.pttBound === '1') return;
    btn.dataset.pttBound = '1';
    btn.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      startRecording();
    }, { passive: false });
    btn.addEventListener('pointerup', (e) => {
      e.preventDefault();
      stopRecording();
    });
    btn.addEventListener('pointercancel', stopRecording);
    btn.addEventListener('pointerleave', stopRecording);
    btn.addEventListener('keydown', handleKeyDown);
  }

  attachPTTHandlers();
  setTimeout(attachPTTHandlers, 100);
  setTimeout(attachPTTHandlers, 500);
})();
"""
