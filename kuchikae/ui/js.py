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

  function findRecordButton(wrap) {
    if (!wrap) return null;
    const buttons = Array.from(wrap.querySelectorAll('button'));
    for (const btn of buttons) {
      const svg = btn.querySelector('svg');
      if (svg) {
        const paths = Array.from(svg.querySelectorAll('path'));
        const hasMicPath = paths.some(p => {
          const d = p.getAttribute('d') || '';
          return d.includes('M12') && (d.includes('12 14') || d.includes('12 16'));
        });
        if (hasMicPath) return btn;
      }
    }
    for (const btn of buttons) {
      const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
      const title = (btn.getAttribute('title') || '').toLowerCase();
      if (aria.includes('record') || aria.includes('録音') || title.includes('record') || title.includes('録音')) {
        return btn;
      }
    }
    if (buttons.length > 0) return buttons[0];
    return null;
  }

  function findStopButton(wrap) {
    if (!wrap) return null;
    const buttons = Array.from(wrap.querySelectorAll('button'));
    for (const btn of buttons) {
      const svg = btn.querySelector('svg');
      if (svg) {
        const rects = Array.from(svg.querySelectorAll('rect'));
        if (rects.length > 0) return btn;
      }
    }
    for (const btn of buttons) {
      const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
      const title = (btn.getAttribute('title') || '').toLowerCase();
      if (aria.includes('stop') || aria.includes('停止') || title.includes('stop') || title.includes('停止')) {
        return btn;
      }
    }
    if (buttons.length > 1) return buttons[1];
    return null;
  }

  function clickButton(btn) {
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
    const wrap = findAudioWrap();
    const btn = findRecordButton(wrap);
    if (!btn || !clickButton(btn)) {
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
    const wrap = findAudioWrap();
    const btn = findStopButton(wrap);
    if (!btn) {
      setHint('停止ボタンが見つかりませんでした。');
      return;
    }
    clickButton(btn);
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
