# Kuchikae v0.1.1 — 手動動作確認リスト

テスト日: ________  確認者: ________  結果: ✅ / ❌ / ⏭

---

## 1. インストール (Zip 展開)

| # | テスト項目 | 手順 | 期待結果 | 結果 |
|---|-----------|------|---------|------|
| 1.1 | Zip ダウンロード | `curl -sL https://github.com/Unjuno/kuchikae/archive/refs/tags/v0.1.1.tar.gz \| tar xz` | エラーなく展開される | ☐ |
| 1.2 | ディレクトリ構造 | `ls kuchikae-0.1.1/` | Makefile, pyproject.toml, kuchikae/ 等が存在 | ☐ |
| 1.3 | Makefile 存在 | `ls Makefile` | Makefile が存在 | ☐ |
| 1.4 | version 確認 | `uv run python -c "from kuchikae import __version__; print(__version__)"` | `0.1.1` と表示 | ☐ |
| 1.5 | 依存関係インストール | `make install` (＝ `uv sync`) | エラーなく完了、`Completed` と表示 | ☐ |

## 2. CLI

| # | テスト項目 | 手順 | 期待結果 | 結果 |
|---|-----------|------|---------|------|
| 2.1 | `--help` | `uv run kuchikae --help` | serve / doctor / setup-models が表示される | ☐ |
| 2.2 | `serve --help` | `uv run kuchikae serve --help` | --dummy / --real / --streaming / --port が表示 | ☐ |
| 2.3 | `doctor` | `uv run kuchikae doctor` | Core dependencies OK, Models 一覧が表示 | ☐ |
| 2.4 | `doctor` 正常終了 | `uv run kuchikae doctor && echo "EXIT=0"` | `EXIT=0` と表示（exit code 0） | ☐ |
| 2.5 | `setup-models --help` | `uv run kuchikae setup-models --help` | --all / --stt / --tts / --emotion / --repair が表示 | ☐ |

## 3. Dummy serve (重要)

| # | テスト項目 | 手順 | 期待結果 | 結果 |
|---|-----------|------|---------|------|
| 3.1 | サーバー起動 | `uv run kuchikae serve --dummy --port 17861` | `Running on local URL: http://127.0.0.1:17861` | ☐ |
| 3.2 | HTTP 応答 | `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:17861` | `200` が返る | ☐ |
| 3.3 | Gradio UI 表示 | ブラウザで http://127.0.0.1:17861 を開く | 録音ボタンとテキスト変換UIが表示される | ☐ |
| 3.4 | タイトル表示 | 画面上部のタイトル | 「Kuchikae」と表示 | ☐ |
| 3.5 | 音声認識モード選択 | Radio ボタン | 速い / バランス / 高精度 の3択が表示 | ☐ |
| 3.6 | テンプレートカテゴリー | Radio ボタン | 標準 / 実験: / 実験強: / カスタム が表示 | ☐ |
| 3.7 | テンプレート選択 | カテゴリー「標準」を選ぶ | Dropdown に「自然に」「かしこまって」等が表示 | ☐ |
| 3.8 | 声の出し方 | Radio ボタン | auto / natural / calm / bright / slow_clear が表示 | ☐ |
| 3.9 | ファイル録音 | 「録音」または「ファイルを選択」で音声入力 | waveform が表示される | ☐ |
| 3.10 | 変換実行 | 音声入力後「変換」または自動実行 | 認識テキスト・変換テキスト・音声が順次表示 | ☐ |
| 3.11 | 認識テキスト表示 | 変換後、認識結果エリア | `[Dummy STT] ...` と表示（dummyモード） | ☐ |
| 3.12 | 変換テキスト表示 | 変換結果エリア | `[transformed ...]` と表示 | ☐ |
| 3.13 | 音声出力再生 | 音声プレーヤー | 再生可能な音声ファイル（dummy音声） | ☐ |
| 3.14 | 音声分析表示 | 声の印象エリア | 分析結果または `unknown` が表示 | ☐ |
| 3.15 | サーバー停止 | Ctrl+C | プロセスが終了、port解放 | ☐ |
| 3.16 | port 変更 | `--port 17862` で起動しアクセス | `http://127.0.0.1:17862` でUI表示 | ☐ |

## 4. Web UI — テンプレート動作

| # | テスト項目 | 手順 | 期待結果 | 結果 |
|---|-----------|------|---------|------|
| 4.1 | 実験: カテゴリー | 「実験:」を選択 | Dropdown に実験用テンプレートが表示、黄色い警告が表示 | ☐ |
| 4.2 | 実験強: カテゴリー | 「実験強:」を選択 | 赤い警告が表示、「OK」クリックでDropdown表示 | ☐ |
| 4.3 | カスタムプロンプト | 「カスタム」+ 自由入力 | 入力したプロンプトで変換が動作 | ☐ |
| 4.4 | 声の出し方 preset | 「calm」を選択し変換 | calm スタイルで音声出力 | ☐ |
| 4.5 | 声の出し方 auto | 「auto」で変換 | 音声感情分析に基づいて自動スタイル | ☐ |
| 4.6 | voice_analysis HTML | 声の印象エリア | HTML で感情分析結果表示 | ☐ |

## 5. Doctor — モデル状態 (このマシンで)

| # | テスト項目 | 手順 | 期待結果 | 結果 |
|---|-----------|------|---------|------|
| 5.1 | doctor 実行 | `uv run kuchikae doctor` | 全モデル状態表示 | ☐ |
| 5.2 | faster-whisper | doctor の Models セクション | `[OK] faster-whisper → ...` | ☐ |
| 5.3 | irodori-tts | doctor の Models セクション | `[MISSING]` または `[OK]`（環境依存） | ☐ |
| 5.4 | irodori-codec | doctor の Models セクション | `[OK] irodori-codec → ...` | ☐ |
| 5.5 | audio-emotion | doctor の Models セクション | `[OK] audio-emotion → ...` | ☐ |
| 5.6 | Ollama status | doctor の Ollama セクション | `Running (N models)` | ☐ |

## 6. model setup

| # | テスト項目 | 手順 | 期待結果 | 結果 |
|---|-----------|------|---------|------|
| 6.1 | setup-models (basic) | `uv run kuchikae setup-models` | STT + TTS + codec がダウンロードされる | ☐ |
| 6.2 | setup-models --all | `uv run kuchikae setup-models --all` | 上記 + 感情モデルがダウンロード | ☐ |
| 6.3 | setup-models --repair | `uv run kuchikae setup-models --repair` | force_download で再DL | ☐ |
| 6.4 | setup-models --stt | `uv run kuchikae setup-models --stt` | STT のみDL | ☐ |
| 6.5 | 再ダウンロード耐性 | 2回目 setup-models | `already cached` と表示、エラーなし | ☐ |

## 7. Real serve（このマシンで可能な場合）

| # | テスト項目 | 手順 | 期待結果 | 結果 |
|---|-----------|------|---------|------|
| 7.1 | real deps install | `uv sync --extra real` | torch / transformers / faster-whisper 等インストール | ☐ |
| 7.2 | real serve 起動 | `uv run kuchikae serve --real --port 17863` | サーバーが起動 | ☐ |
| 7.3 | real STT | 音声入力 → 変換 | 実際の文字起こし結果が表示（`[Dummy STT]` ではない） | ☐ |
| 7.4 | streaming serve | `uv run kuchikae serve --real --streaming --port 17864` | streaming mode で起動 | ☐ |

## 8. 異常系

| # | テスト項目 | 手順 | 期待結果 | 結果 |
|---|-----------|------|---------|------|
| 8.1 | 不正な音声ファイル | 空ファイル / 対応外フォーマットを入力 | エラーメッセージが表示、クラッシュしない | ☐ |
| 8.2 | 接続なしで起動 | オフラインで `serve --dummy` | dummyはオフラインでも動作 | ☐ |
| 8.3 | port 衝突 | 既に使用中のportで起動 | エラーメッセージが表示 | ☐ |
| 8.4 | Ollama 未起動で real serve | Ollama停止状態で `serve --real` | エラーハンドリング、起動はする | ☐ |

## 9. Makefile コマンド

| # | テスト項目 | 手順 | 期待結果 | 結果 |
|---|-----------|------|---------|------|
| 9.1 | `make run` | 新規ディレクトリで `make run` | install + dummy serve が一発で起動 | ☐ |
| 9.2 | `make doctor` | `make doctor` | doctor が実行される | ☐ |
| 9.3 | `make test` | `make test` | pytest が実行され PASS のみ | ☐ |
| 9.4 | `make check-all` | `make check-all` | ruff → mypy → pip-audit → compileall → pytest が全て PASS | ☐ |
| 9.5 | `make clean` | `make clean` | キャッシュディレクトリが削除される | ☐ |

---

## 結果サマリ

| カテゴリ | 全テスト数 | 通過 | 失敗 | スキップ |
|---------|-----------|------|------|---------|
| 1. インストール | 5 | ☐ | ☐ | ☐ |
| 2. CLI | 5 | ☐ | ☐ | ☐ |
| 3. Dummy serve | 16 | ☐ | ☐ | ☐ |
| 4. UI テンプレート | 6 | ☐ | ☐ | ☐ |
| 5. Doctor | 6 | ☐ | ☐ | ☐ |
| 6. Model setup | 5 | ☐ | ☐ | ☐ |
| 7. Real serve | 4 | ☐ | ☐ | ☐ |
| 8. 異常系 | 4 | ☐ | ☐ | ☐ |
| 9. Makefile | 5 | ☐ | ☐ | ☐ |
| **合計** | **56** | ☐ | ☐ | ☐ |

**総評:** _________________________________________________________________
