# deepfind-cli

Minimal multi-agent research CLI in Python.

## Install

```bash
python3 -m pip install -e .
```

Optional tools:

- [twitter-cli](https://github.com/jackwener/twitter-cli)
- [xiaohongshu-cli](https://github.com/jackwener/xiaohongshu-cli)

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
DEEPFIND_TOOL_TIMEOUT=90
```

## Run

```bash
python3 -m deepfind.cli "小红书上的博主：刘小鸭的AI日记 都有什么内容？ 他有多少粉丝?" --num-agent 2
```

Flags:

- `query`: required
- `--num-agent`: `1..4`
- `--max-iter-per-agent`: default `50`

## How It Works

- Lead agent splits the query into a few tasks.
- Sub-agents call local tools such as `xhs_search_user`, `xhs_user`, `xhs_user_posts`, `xhs_read`, `twitter_search`, and `twitter_read`.
- Lead agent merges the results into one answer.

Qwen is used through the OpenAI-compatible `chat.completions` API.

## Test

```bash
python3 -m unittest discover -s tests -v
```
