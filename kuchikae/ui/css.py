"""Kuchikae UI — CSS styles."""

CSS = """
:root {
  --bg-primary: #18181B;
  --bg-secondary: #252528;
  --bg-tertiary: #27272A;
  --bg-accent: #1F1135;
  --text-primary: #F4F4F5;
  --text-secondary: #E4E4E7;
  --text-muted: #A1A1AA;
  --border-color: #3F3F46;
  --accent-primary: #7C3AED;
  --accent-secondary: #5B21B6;
  --accent-light: #C4B5FD;
  --accent-dark: #3B0764;
}

body {
  background: var(--bg-primary) !important;
  color: var(--text-secondary) !important;
}

gradio-app, .gradio-app {
  background: transparent !important;
}

.gradio-container {
  max-width: 760px !important;
  margin: 0 auto !important;
  padding: 24px 16px !important;
}

.main > .wrap {
  background: var(--bg-secondary) !important;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, .3);
  color: var(--text-secondary) !important;
}

#title {
  text-align: center;
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 20px;
}

.tabs {
  margin-bottom: 0;
}

.tabs button {
  border: none !important;
  background: transparent !important;
  color: var(--text-muted) !important;
  font-size: 13px;
  padding: 8px 16px !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  transition: all .15s ease;
}

.tabs button.selected {
  color: var(--accent-light) !important;
  border-bottom-color: var(--accent-primary) !important;
  background: transparent !important;
}

.tabs button:hover {
  color: var(--text-secondary) !important;
}

.tab-nav {
  background: transparent !important;
  border-bottom: 1px solid var(--border-color) !important;
  margin-bottom: 16px;
  justify-content: center;
}

#template-select {
  margin-bottom: 12px;
}

#template-select > span {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  display: block;
  margin-bottom: 6px;
}

#template-select .wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  padding: 0 !important;
}

#template-select label {
  padding: 4px 12px !important;
  min-height: 28px !important;
  font-size: 12px;
  border-radius: 6px;
  border: 1px solid var(--border-color);
  background: var(--bg-tertiary);
  cursor: pointer;
  transition: all .15s ease;
  color: var(--text-secondary);
}

#template-select label.selected {
  background: #2D1B4E;
  border-color: var(--accent-primary);
  color: var(--accent-light);
  font-weight: 600;
}

#template-select label input {
  display: none;
}

#run-btn {
  background: var(--accent-primary) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px;
  padding: 8px 24px !important;
  font-size: 14px;
  font-weight: 600;
  transition: all .15s ease;
  margin: 8px 0;
}

#run-btn:hover {
  opacity: .9;
}

#run-btn:focus {
  outline: 2px solid var(--accent-light);
  outline-offset: 2px;
}

#text-compare, #normal-text-compare, #simple-text-compare {
  gap: 12px;
}

#text-compare > span, #normal-text-compare > span, #simple-text-compare > span {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: .04em;
  display: block;
}

#source-text > span, #transformed-text > span {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: .04em;
}

#audio-input-wrap {
  background: var(--bg-tertiary);
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 8px;
  border: 1px solid var(--border-color);
}

#audio-input-wrap .block {
  background: transparent !important;
  box-shadow: none !important;
}

#audio-input-wrap label, #output-audio label {
  background: transparent !important;
  color: var(--text-muted) !important;
}

#audio-input-wrap .record-button {
  background: #2D1B4E !important;
  color: var(--accent-light) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
}

#audio-input-wrap select {
  background: var(--bg-tertiary) !important;
  border-radius: 6px;
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
}

#audio-input-wrap .icon-button-wrapper {
  background: transparent !important;
}

#audio-input-wrap .stop-button, #audio-input-wrap .resume-button, #audio-input-wrap .pause-button {
  color: var(--text-secondary) !important;
}

#audio-input-wrap .block select {
  background: var(--bg-tertiary) !important;
  border-radius: 6px;
  border: 1px solid var(--border-color);
}

#source-text > .block, #transformed-text > .block {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}

#source-text textarea {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.6;
  resize: none;
}

#transformed-text textarea {
  background: var(--bg-accent);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.6;
  resize: none;
}

#template-select > .block {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}

#output-audio > .block {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}

#output-audio {
  margin-top: 4px;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 12px;
  background: var(--bg-tertiary);
}

#output-audio audio {
  height: 44px !important;
  margin: 0 auto;
  background: var(--bg-tertiary);
  border-radius: 8px;
}

#status {
  font-size: 12px;
  color: var(--text-muted);
  text-align: center;
  min-height: 16px;
  margin-top: 4px;
}

#simple-status {
  font-size: 12px;
  color: var(--text-muted);
  text-align: center;
  min-height: 16px;
  margin-top: 4px;
}

#prompt-box textarea {
  border-radius: 8px;
  border: 1px solid var(--border-color);
  padding: 8px 12px;
  font-size: 13px;
  color: var(--text-secondary);
  background: var(--bg-tertiary);
}

#prompt-box textarea:focus {
  border-color: var(--accent-primary);
  background: var(--bg-tertiary);
  outline: none;
  box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2);
}

#simple-audio-wrap {
  background: var(--bg-tertiary);
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 8px;
  border: 1px solid var(--border-color);
  position: relative;
}

#simple-audio-wrap .block {
  background: transparent !important;
  box-shadow: none !important;
}

#simple-audio-wrap label {
  color: var(--text-muted) !important;
}

#simple-audio-wrap .record-button {
  background: #2D1B4E !important;
  color: var(--accent-light) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 8px !important;
}

#simple-template-select {
  margin-bottom: 12px;
}

#simple-template-select > span {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  display: block;
  margin-bottom: 6px;
}

#simple-template-select .wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  padding: 0 !important;
}

#simple-template-select label {
  padding: 4px 12px !important;
  min-height: 28px !important;
  font-size: 12px;
  border-radius: 6px;
  border: 1px solid var(--border-color);
  background: var(--bg-tertiary);
  cursor: pointer;
  transition: all .15s ease;
  color: var(--text-secondary);
}

#simple-template-select label.selected {
  background: #2D1B4E;
  border-color: var(--accent-primary);
  color: var(--accent-light);
  font-weight: 600;
}

#simple-template-select label input {
  display: none;
}

#simple-src > .block, #simple-trf > .block {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}

#simple-src textarea {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 14px;
  color: var(--text-primary);
  line-height: 1.5;
  resize: none;
}

#simple-trf textarea {
  background: linear-gradient(135deg, rgba(124, 58, 237, 0.08) 0%, rgba(91, 33, 182, 0.12) 100%);
  border: 1px solid rgba(124, 58, 237, 0.3);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 14px;
  color: var(--accent-light);
  line-height: 1.5;
  resize: none;
}

#simple-output-audio > .block {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}

#simple-output-audio {
  margin-top: 12px;
  border: 1px solid rgba(124, 58, 237, 0.3);
  border-radius: 12px;
  padding: 16px;
  background: linear-gradient(135deg, rgba(124, 58, 237, 0.05) 0%, transparent 100%);
}

#simple-output-audio audio {
  height: 44px !important;
  margin: 0 auto;
  border-radius: 8px;
}

#simple-output-audio label {
  color: var(--accent-light) !important;
  font-weight: 600;
}

#ptt-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 16px 0 20px;
  margin: 8px 0;
  background: linear-gradient(180deg, rgba(124, 58, 237, 0.05) 0%, transparent 100%);
  border-radius: 16px;
}

#ptt-btn {
  width: 140px;
  height: 140px;
  border-radius: 9999px;
  border: 2px solid var(--accent-primary);
  background: radial-gradient(circle at 30% 30%, #8B5CF6, var(--accent-secondary) 70%, var(--accent-dark) 100%);
  color: #FFFFFF;
  box-shadow: 0 12px 32px rgba(91, 33, 182, 0.4), 0 0 0 4px rgba(124, 58, 237, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  user-select: none;
  touch-action: none;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.02em;
  transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
}

#ptt-btn:hover {
  filter: brightness(1.1);
  box-shadow: 0 14px 36px rgba(91, 33, 182, 0.5), 0 0 0 6px rgba(124, 58, 237, 0.15);
}

#ptt-btn:active {
  transform: scale(0.96);
  box-shadow: 0 8px 20px rgba(91, 33, 182, 0.3);
}

#ptt-btn.ptt-recording {
  transform: scale(0.96);
  background: radial-gradient(circle at 30% 30%, #EF4444, #DC2626 70%, #991B1B 100%);
  border-color: #EF4444;
  box-shadow: 0 8px 20px rgba(239, 68, 68, 0.4), 0 0 0 4px rgba(239, 68, 68, 0.2);
  animation: ptt-pulse 1.2s ease-in-out infinite;
}

@keyframes ptt-pulse {
  0%, 100% { box-shadow: 0 8px 20px rgba(239, 68, 68, 0.4), 0 0 0 4px rgba(239, 68, 68, 0.2); }
  50% { box-shadow: 0 8px 20px rgba(239, 68, 68, 0.6), 0 0 0 8px rgba(239, 68, 68, 0.1); }
}

#ptt-btn:focus {
  outline: 3px solid var(--accent-light);
  outline-offset: 3px;
}

#ptt-label {
  display: block;
  max-width: 100px;
  line-height: 1.3;
}

#ptt-hint {
  color: var(--text-muted);
  font-size: 11px;
  text-align: center;
  opacity: 0.8;
}

#simple-status {
  font-size: 12px;
  color: var(--text-muted);
  text-align: center;
  min-height: 20px;
  margin-top: 8px;
  padding: 8px 12px;
  background: rgba(124, 58, 237, 0.08);
  border-radius: 8px;
  transition: all 0.2s ease;
}

#simple-status.processing {
  color: var(--accent-light);
  background: rgba(124, 58, 237, 0.15);
}

input[type="text"]:focus,
input[type="number"]:focus,
select:focus,
textarea:focus {
  outline: none;
  border-color: var(--accent-primary) !important;
  box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2);
}

button:focus-visible {
  outline: 2px solid var(--accent-light);
  outline-offset: 2px;
}

@media (max-width: 640px) {
  .gradio-container {
    padding: 16px 10px !important;
  }

  .main > .wrap {
    padding: 16px;
  }

  #template-select .wrap {
    gap: 3px;
  }

  #template-select label {
    padding: 3px 8px !important;
    font-size: 11px;
  }

  #text-compare, #normal-text-compare, #simple-text-compare {
    flex-direction: column !important;
  }

  #text-compare > *, #normal-text-compare > *, #simple-text-compare > * {
    min-width: 0 !important;
  }

  #ptt-btn {
    width: 140px;
    height: 140px;
    font-size: 16px;
  }

  #run-btn {
    width: 100%;
    padding: 12px 24px !important;
  }
}
"""
