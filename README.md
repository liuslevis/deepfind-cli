# deepfind-cli

Minimal multi-agent research CLI in Python.

## Install

```bash
python3 -m pip install -e .
```

Install optional ASR dependencies only when you need Bilibili transcription:

```bash
python3 -m pip install -e ".[media]"
```

Pre-download the ASR model on Windows (PowerShell) to avoid first-run delay:

```powershell
hf download Qwen/Qwen3-ASR-1.7B --repo-type model
```

Optional tools:

- [twitter-cli](https://github.com/jackwener/twitter-cli)
- [xiaohongshu-cli](https://github.com/jackwener/xiaohongshu-cli)
- [bilibili-cli](https://github.com/jackwener/bilibili-cli)

Install optional CLIs (WSL):

```bash
uv tool install bilibili-cli
uv tool install git+https://github.com/jackwener/xiaohongshu-cli.git
uv tool install git+https://github.com/jackwener/twitter-cli.git
```

## Env

The CLI auto-loads `.env` from the repo root.

```bash
cp .env.example .env
```

Minimal `.env`:

```bash
QWEN_API_KEY=...
QWEN_MODEL_NAME=qwen3-max
```

Optional:

```bash
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
TWITTER_CLI_BIN=twitter
XHS_CLI_BIN=xhs
BILI_BIN=bili
ASR_MODEL=Qwen/Qwen3-ASR-1.7B
DEEPFIND_AUDIO_DIR=audio
DEEPFIND_TOOL_TIMEOUT=90
```

## Run

```bash
python3 -m deepfind.cli "小红书上的博主：刘小鸭的AI日记 都有什么内容？ 他有多少粉丝?" --num-agent 2
python3 -m deepfind.cli "same query" --num-agent 2 --quiet
python3 -m deepfind.cli "same query" --viewer frogmouth
```

Flags:

- `query`: required
- `--num-agent`: `1..4`
- `--max-iter-per-agent`: default `50`
- `--quiet`: disable formatted progress output
- `--viewer`: `auto|plain|frogmouth`, default `auto`

For prettier terminal Markdown:

```bash
pipx install frogmouth
```

## How It Works

- Lead agent splits the query into a few tasks.
- Sub-agents call local tools such as `xhs_search_user`, `xhs_user`, `xhs_user_posts`, `xhs_read`, `twitter_search`, `twitter_read`, and `bili_transcribe`.
- Lead agent merges the results into one answer.

Qwen is used through the OpenAI-compatible `chat.completions` API.

## Bilibili Transcription Tool

`bili_transcribe` is available to sub-agents and accepts either a Bilibili video URL
or a raw `BV...` ID.
It returns transcript text only (no summary generation).

Setup (WSL):

```bash
uv tool install bilibili-cli
bili status
```

Artifacts:

- Segments: `audio/<BVID>/seg_*`
- Transcript: `audio/transcripts/<BVID>.txt`

## Test

```bash
python3 -m unittest discover -s tests -v
```
