"""Human-facing hints for common failures."""

from __future__ import annotations


def hint_for_error(stage: str, exc: Exception) -> str:
    msg = str(exc)
    if "faster-whisper" in msg or "faster_whisper" in msg:
        return "real STT backend が未インストールです。`uv sync --extra real` を実行してください。"
    if "irodori_tts" in msg:
        return "Irodori-TTS が未インストールです。`uv sync --extra real` を実行してください。"
    if "OPENAI_API_KEY" in msg:
        return "GPT text backend を使うには OPENAI_API_KEY を設定してください。"
    if "Audio file not found" in msg:
        return "録音ファイル path が backend に渡っていません。Gradio Audio の type='filepath' と normalize_audio_path() を確認してください。"
    if "Unsupported format" in msg:
        return "対応形式は wav/mp3/m4a/flac です。録音形式または拡張子を確認してください。"
    if "Audio too long" in msg:
        return "音声が長すぎます。UI 制限を超えています。benchmark では制限 override を使ってください。"
    if stage == "stt" and "DummySTTBackend" in msg:
        return "STT backend が dummy です。allow_dummy_backends を切るか real backend を有効化してください。"
    return ""
