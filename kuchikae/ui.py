"""Kuchikae UI — Gradio view layer (View Isolation)."""

from __future__ import annotations

import os
import tempfile
from typing import Generator

import gradio as gr
import soundfile as sf

from kuchikae.pipeline import KuchikaePipeline
from kuchikae.types import TextTransformPrompt

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

#simple-audio-wrap .block { background: transparent !important; box-shadow: none !important; }
#simple-audio-wrap label { display: none !important; }
#simple-audio-wrap select,
#simple-audio-wrap .block select { display: none !important; }
#simple-audio-wrap { position: absolute !important; top: -9999px !important; left: -9999px !important; opacity: 0 !important; pointer-events: none !important; border: none !important; padding: 0 !important; margin: 0 !important; background: transparent !important; }
#simple-audio-wrap button { pointer-events: auto !important; }

#simple-src > .block, #simple-trf > .block { background: transparent !important; box-shadow: none !important; border: none !important; }
#simple-src textarea { background: #27272A; border: 1px solid #3F3F46; border-radius: 8px; padding: 8px 12px; font-size: 13px; color: #E4E4E7; line-height: 1.6; resize: none; }
#simple-trf textarea { background: #1F1135; border: 1px solid #3F3F46; border-radius: 8px; padding: 8px 12px; font-size: 13px; color: #E4E4E7; line-height: 1.6; resize: none; }
#simple-output-audio > .block { background: transparent !important; box-shadow: none !important; border: none !important; }
#simple-output-audio { margin-top: 4px; border: 1px solid #3F3F46; border-radius: 12px; padding: 12px; background: #27272A; }
#simple-output-audio audio { height: 44px !important; margin: 0 auto; background: #27272A; border-radius: 8px; }

#ptt-container { text-align: center; padding: 32px 16px; margin-bottom: 8px; }
#ptt-btn { width: 140px; height: 140px; border-radius: 50%; border: none; cursor: pointer; transition: all .15s ease; display: inline-flex; align-items: center; justify-content: center; font-size: 18px; font-weight: 700; line-height: 1.3; margin: 0 auto; -webkit-user-select: none; user-select: none; touch-action: manipulation; }
#ptt-btn.ptt-idle { background: #7C3AED; color: white; box-shadow: 0 4px 20px rgba(124,58,237,.4); }
#ptt-btn.ptt-idle:hover { transform: scale(1.05); background: #6D28D9; }
#ptt-btn.ptt-idle:active { transform: scale(0.95); }
#ptt-btn.ptt-recording { background: #DC2626; color: white; box-shadow: 0 0 0 8px rgba(220,38,38,.25); animation: ptt-pulse 1s ease-in-out infinite; transform: scale(0.95); }
@keyframes ptt-pulse { 0%,100% { box-shadow: 0 0 0 8px rgba(220,38,38,.25); } 50% { box-shadow: 0 0 0 16px rgba(220,38,38,.15); } }
#ptt-btn.ptt-recording:hover { background: #B91C1C; }
#ptt-hint { margin-top: 12px; color: #A1A1AA; font-size: 13px; transition: color .15s ease; }

@media (max-width: 640px) {
  .gradio-container { padding: 16px 10px !important; }
  .main > .wrap { padding: 16px; }
  #template-select .wrap.svelte-e4x47i { gap: 3px; }
  #template-select label.svelte-19qdtil { padding: 3px 8px !important; font-size: 11px; }
  #text-compare { flex-direction: column !important; }
  #text-compare > * { min-width: 0 !important; }
}
"""

TEMPLATES = {
    "自然に": "内容、数字、日時、固有名詞、否定条件は保ちつつ、言い回しを自然な日本語に変換してください。",
    "丁寧に": "次のテキストを「です・ます」調の丁寧な言葉遣いに変換してください。",
    "柔らかく": "次のテキストを柔らかく丁寧な表現に変換してください。",
    "短く": "次のテキストを簡潔に短く要約・変換してください。内容や固有名詞は変えないでください。",
    "カスタム": "",
}


def normalize_audio_path(audio_input):
    if audio_input is None:
        return None
    if isinstance(audio_input, tuple):
        sr, data = audio_input
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, data, sr)
            return tmp.name
    if isinstance(audio_input, dict):
        path = audio_input.get("orig_name") or audio_input.get("name") or audio_input.get("path")
        return str(path) if path else None
    path = getattr(audio_input, "path", None) or getattr(audio_input, "orig_name", None) or getattr(audio_input, "name", None)
    if path:
        return str(path)
    result = str(audio_input)
    return result if os.path.isfile(result) else None


def run_simple(
    pipeline: KuchikaePipeline,
    audio_input,
    live_streaming: bool = False,
) -> Generator:
    path = normalize_audio_path(audio_input)
    if path is None:
        yield gr.update(), "", "", ""
        return

    prompt = TextTransformPrompt(instruction=TEMPLATES["自然に"])
    stream_fn = pipeline.process_stream_live if live_streaming else pipeline.process_stream

    for status, src, txt, aud in stream_fn(path, prompt):
        if status == "DONE":
            yield aud, src, txt, "言い直しました"
        elif status == "VOX":
            yield gr.update(value=None), src, txt, "変換中..."
        elif status == "TXT":
            yield gr.update(value=None), src, txt, "変換中..."
        elif status == "STT_PARTIAL":
            yield gr.update(value=None), src, "", "文字起こし中..."
        else:
            yield gr.update(value=None), "", "", "音声認識中..."


def create_app(pipeline: KuchikaePipeline, default_prompt: TextTransformPrompt, live_streaming: bool = False) -> gr.Blocks:

    import logging
    logger = logging.getLogger("kuchikae.ui")

    def run(audio_input, template_name: str, custom_prompt: str) -> Generator:
        path = normalize_audio_path(audio_input)
        if path is None:
            raise gr.Error("音声を録音またはアップロードしてください")

        if template_name == "カスタム" and custom_prompt.strip():
            prompt_text = custom_prompt
        elif template_name in TEMPLATES:
            prompt_text = TEMPLATES[template_name]
        else:
            prompt_text = TEMPLATES["自然に"]

        prompt = TextTransformPrompt(instruction=prompt_text)
        stream_fn = pipeline.process_stream_live if live_streaming else pipeline.process_stream

        for status, src, txt, aud in stream_fn(path, prompt):
            if status == "DONE":
                yield aud, src, txt, "言い直しました"
            elif status == "VOX":
                yield gr.update(value=None), src, txt, "変換中..."
            elif status == "TXT":
                yield gr.update(value=None), src, txt, "変換中..."
            elif status == "STT_PARTIAL":
                yield gr.update(value=None), src, "", "文字起こし中..."
            else:
                yield gr.update(value=None), "", "", "音声認識中..."

    def on_template_change(template_name: str):
        if template_name == "カスタム":
            return gr.update()
        text = TEMPLATES.get(template_name, "")
        return gr.update(value=text)

    with gr.Blocks(title="Kuchikae") as demo:
        with gr.Column(elem_classes="wrap"):
            gr.HTML('<div id="title">Kuchikae</div>')

            with gr.Tabs(elem_classes="tabs"):
                with gr.Tab("通常"):
                    template = gr.Radio(
                        elem_id="template-select",
                        label="どう言い換える？",
                        choices=list(TEMPLATES.keys()),
                        value="自然に",
                    )

                    audio_input = gr.Audio(
                        elem_id="audio-input-wrap",
                        label="元の音声",
                        sources=["microphone", "upload"],
                        type="filepath",
                    )

                    run_btn = gr.Button("言い直す", elem_id="run-btn")

                    with gr.Row(elem_id="text-compare"):
                        source_text = gr.Textbox(
                            elem_id="source-text",
                            label="聞き取った内容",
                            lines=4,
                            interactive=False,
                        )
                        transformed_text = gr.Textbox(
                            elem_id="transformed-text",
                            label="言い直した内容",
                            lines=4,
                            interactive=False,
                        )

                    output_audio = gr.Audio(
                        elem_id="output-audio",
                        label="言い直し音声",
                        type="filepath",
                        autoplay=True,
                    )

                    status = gr.HTML(elem_id="status", value="", visible=True)

                    with gr.Accordion("詳細プロンプト", open=False):
                        text_prompt = gr.Textbox(
                            elem_id="prompt-box",
                            label="LLMに渡す言い換え指示",
                            value=default_prompt.instruction,
                            lines=3,
                        )

                    template.change(on_template_change, inputs=[template], outputs=[text_prompt])
                    run_btn.click(
                        run,
                        inputs=[audio_input, template, text_prompt],
                        outputs=[output_audio, source_text, transformed_text, status],
                    )

                with gr.Tab("簡易"):
                    simple_audio = gr.Audio(
                        elem_id="simple-audio-wrap",
                        label="",
                        sources=["microphone"],
                        type="filepath",
                        visible=True,
                    )

                    gr.HTML("""
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
let pttState = 0; // 0=idle, 1=recording
let pttTimer = null;

function pttStart(e) {
  if (e) e.preventDefault();
  if (pttState === 1) return;
  pttState = 1;
  const btn = document.getElementById('ptt-btn');
  const label = document.getElementById('ptt-label');
  btn.className = 'ptt-recording';
  label.textContent = '話し終えたら離す';
  document.getElementById('ptt-hint').textContent = '録音中…';

  // Click Gradio's Record button
  const wrap = document.getElementById('simple-audio-wrap');
  if (!wrap) return;
  const btns = wrap.querySelectorAll('button');
  for (const b of btns) {
    if (b.textContent.trim() === 'Record') { b.click(); break; }
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

  // Click Gradio's Stop button
  const wrap = document.getElementById('simple-audio-wrap');
  if (!wrap) return;
  const btns = wrap.querySelectorAll('button');
  for (const b of btns) {
    if (b.textContent.trim() === 'Stop') { b.click(); break; }
  }

  // Reset hint after timeout
  if (pttTimer) clearTimeout(pttTimer);
  pttTimer = setTimeout(() => {
    const h = document.getElementById('ptt-hint');
    if (h) h.textContent = 'ボタンを押しながら話す、離すと自動変換';
  }, 5000);
}
</script>
""")

                    with gr.Row(elem_id="text-compare"):
                        simple_source = gr.Textbox(
                            elem_id="simple-src",
                            label="聞き取った内容",
                            lines=4,
                            interactive=False,
                        )
                        simple_transformed = gr.Textbox(
                            elem_id="simple-trf",
                            label="言い直した内容",
                            lines=4,
                            interactive=False,
                        )

                    simple_output = gr.Audio(
                        elem_id="simple-output-audio",
                        label="言い直し音声",
                        type="filepath",
                        autoplay=True,
                    )

                    simple_status = gr.HTML(elem_id="simple-status", value="", visible=True)

                    simple_audio.change(
                        lambda a: run_simple(pipeline, a, live_streaming),
                        inputs=[simple_audio],
                        outputs=[simple_output, simple_source, simple_transformed, simple_status],
                    )

    return demo
