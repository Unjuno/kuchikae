"""Kuchikae UI — CSS styles."""

CSS = """
body { background: #18181B !important; color: #E4E4E7 !important; }
gradio-app, .gradio-app { background: transparent !important; }
.gradio-container { max-width: 760px !important; margin: 0 auto !important; padding: 24px 16px !important; }
.main > .wrap { background: #252528 !important; border-radius: 16px; padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.3); color: #E4E4E7 !important; }
#title { text-align: center; font-size: 22px; font-weight: 700; color: #F4F4F5; margin-bottom: 20px; }
.tabs { margin-bottom: 0; }
.tabs button { border: none !important; background: transparent !important; color: #A1A1AA !important; font-size: 13px; padding: 8px 16px !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; transition: all .15s ease; }
.tabs button.selected { color: #C4B5FD !important; border-bottom-color: #7C3AED !important; background: transparent !important; }
.tabs button:hover { color: #D4D4D8 !important; }
.tab-nav { background: transparent !important; border-bottom: 1px solid #3F3F46 !important; margin-bottom: 16px; justify-content: center; }
#template-select { margin-bottom: 12px; }
#template-select > span { font-size: 12px; font-weight: 600; color: #A1A1AA; display: block; margin-bottom: 6px; }
#template-select .wrap.svelte-e4x47i { display: flex; flex-wrap: wrap; gap: 4px; padding: 0 !important; }
#template-select label.svelte-19qdtil { padding: 4px 12px !important; min-height: 28px !important; font-size: 12px; border-radius: 6px; border: 1px solid #3F3F46; background: #27272A; cursor: pointer; transition: all .15s ease; color: #D4D4D8; }
#template-select label.svelte-19qdtil.selected { background: #2D1B4E; border-color: #7C3AED; color: #C4B5FD; font-weight: 600; }
#template-select label.svelte-19qdtil input { display: none; }
#run-btn { background: #7C3AED !important; color: white !important; border: none !important; border-radius: 8px; padding: 8px 24px !important; font-size: 14px; font-weight: 600; transition: all .15s ease; margin: 8px 0; }
#run-btn:hover { opacity: .9; }
#text-compare { gap: 12px; }
#text-compare > span { font-size: 11px; font-weight: 600; color: #A1A1AA; text-transform: uppercase; letter-spacing: .04em; display: block; }
#source-text > span, #transformed-text > span { font-size: 11px; font-weight: 600; color: #A1A1AA; text-transform: uppercase; letter-spacing: .04em; }
#audio-input-wrap { background: #27272A; border-radius: 12px; padding: 12px; margin-bottom: 8px; border: 1px solid #3F3F46; }
#audio-input-wrap .block { background: transparent !important; box-shadow: none !important; }
#audio-input-wrap label, #output-audio label { background: transparent !important; color: #A1A1AA !important; }
#audio-input-wrap .record-button { background: #2D1B4E !important; color: #C4B5FD !important; border: 1px solid #3F3F46 !important; border-radius: 8px !important; }
#audio-input-wrap select { background: #27272A !important; border-radius: 6px; border: 1px solid #3F3F46; color: #D4D4D8; }
#audio-input-wrap .icon-button-wrapper { background: transparent !important; }
#audio-input-wrap .stop-button, #audio-input-wrap .resume-button, #audio-input-wrap .pause-button { color: #D4D4D8 !important; }
#audio-input-wrap .block select { background: #27272A !important; border-radius: 6px; border: 1px solid #3F3F46; }
#source-text > .block, #transformed-text > .block { background: transparent !important; box-shadow: none !important; border: none !important; }
#source-text textarea { background: #27272A; border: 1px solid #3F3F46; border-radius: 8px; padding: 8px 12px; font-size: 13px; color: #E4E4E7; line-height: 1.6; resize: none; }
#transformed-text textarea { background: #1F1135; border: 1px solid #3F3F46; border-radius: 8px; padding: 8px 12px; font-size: 13px; color: #E4E4E7; line-height: 1.6; resize: none; }
#template-select > .block { background: transparent !important; box-shadow: none !important; border: none !important; }
#output-audio > .block { background: transparent !important; box-shadow: none !important; border: none !important; }
#output-audio { margin-top: 4px; border: 1px solid #3F3F46; border-radius: 12px; padding: 12px; background: #27272A; }
#output-audio audio { height: 44px !important; margin: 0 auto; background: #27272A; border-radius: 8px; }
#status { font-size: 12px; color: #A1A1AA; text-align: center; min-height: 16px; margin-top: 4px; }
#simple-status { font-size: 12px; color: #A1A1AA; text-align: center; min-height: 16px; margin-top: 4px; }
.block.svelte-1plpy97 { background: transparent !important; }
.label-wrap.svelte-e5lyqv { color: #D4D4D8 !important; }
#prompt-box textarea { border-radius: 8px; border: 1px solid #3F3F46; padding: 8px 12px; font-size: 13px; color: #E4E4E7; background: #27272A; }
#prompt-box textarea:focus { border-color: #7C3AED; background: #27272A; }

#simple-audio { background: #27272A; border-radius: 12px; padding: 12px; margin-bottom: 8px; border: 1px solid #3F3F46; }
#simple-audio .block { background: transparent !important; box-shadow: none !important; }
#simple-audio label { color: #A1A1AA !important; }
#simple-audio .record-button { background: #2D1B4E !important; color: #C4B5FD !important; border: 1px solid #3F3F46 !important; border-radius: 8px !important; }
#simple-src > .block, #simple-trf > .block { background: transparent !important; box-shadow: none !important; border: none !important; }
#simple-src textarea { background: #27272A; border: 1px solid #3F3F46; border-radius: 8px; padding: 8px 12px; font-size: 13px; color: #E4E4E7; line-height: 1.6; resize: none; }
#simple-trf textarea { background: #1F1135; border: 1px solid #3F3F46; border-radius: 8px; padding: 8px 12px; font-size: 13px; color: #E4E4E7; line-height: 1.6; resize: none; }
#simple-output-audio > .block { background: transparent !important; box-shadow: none !important; border: none !important; }
#simple-output-audio { margin-top: 4px; border: 1px solid #3F3F46; border-radius: 12px; padding: 12px; background: #27272A; }
#simple-output-audio audio { height: 44px !important; margin: 0 auto; background: #27272A; border-radius: 8px; }

@media (max-width: 640px) {
  .gradio-container { padding: 16px 10px !important; }
  .main > .wrap { padding: 16px; }
  #template-select .wrap.svelte-e4x47i { gap: 3px; }
  #template-select label.svelte-19qdtil { padding: 3px 8px !important; font-size: 11px; }
  #text-compare { flex-direction: column !important; }
  #text-compare > * { min-width: 0 !important; }
}
"""
