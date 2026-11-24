# DeepFind Web 回归测试技能

使用 Playwright 从真实浏览器页面发送 `nvidia`，验证前端、认证、后端、SSE 和模型调用。
不要只运行单元测试或直接调用 API；最终必须从页面读取回答。

## 前置条件

在仓库根目录执行。你需要：

- Python Playwright 已安装，并且 Chromium 可启动；
- `web` 依赖已安装；
- `.venv` 已安装项目 Python 依赖；
- `.env` 包含可用的 `DEEPFIND_WEB_TOKEN` 和至少一个模型 API Key；
- 前端监听 `http://127.0.0.1:5173`；
- 后端监听 `http://127.0.0.1:8000`。

先检查环境：

```bash
python3 -m playwright --version
opencli --version
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:5173/
```

不要输出 `.env`、访问令牌或 API Key。

## 1. 基础页面 E2E

基础测试固定选择 Qwen，避免默认模型端点的临时 DNS 或兼容性问题影响页面回归判断。

成功标准：

1. 登录页接受 `.env` 中的 `DEEPFIND_WEB_TOKEN`；
2. 聊天输入框可见；
3. 模型切换为 Qwen；
4. 输入并发送 `nvidia`；
5. 最后一条 assistant 消息结束 pending 状态；
6. 回答非空，并包含 `NVIDIA`、`nvidia` 或 `英伟达`；
7. 页面没有 `.message__error` 或 `.composer__error`；
8. 回答不是 `The run ended before a final answer was produced.`。

执行：

```bash
python3 - <<'PY'
import re
from pathlib import Path

from playwright.sync_api import sync_playwright


def env_value(name: str) -> str:
    text = Path(".env").read_text(encoding="utf-8")
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(.*?)\s*$", text)
    if not match:
        return ""
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value


token = env_value("DEEPFIND_WEB_TOKEN")
console_errors: list[str] = []
failed_requests: list[str] = []

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on(
        "requestfailed",
        lambda request: failed_requests.append(
            f"{request.method} {request.url}: {request.failure}"
        ),
    )

    page.goto(
        "http://127.0.0.1:5173/?e2e=nvidia",
        wait_until="domcontentloaded",
        timeout=30_000,
    )
    page.wait_for_function(
        "() => Boolean(document.querySelector('input[type=password]') "
        "|| document.querySelector('#chat-input'))",
        timeout=30_000,
    )

    password = page.locator('input[type="password"]')
    if password.is_visible():
        if not token:
            raise RuntimeError(
                "DEEPFIND_WEB_TOKEN is required but missing from .env"
            )
        password.fill(token)
        password.press("Enter")

    composer = page.locator("#chat-input")
    composer.wait_for(state="visible", timeout=30_000)

    model_button = page.locator('button[aria-label="Model"]')
    model_button.wait_for(state="visible", timeout=30_000)
    for _ in range(6):
        if model_button.inner_text().strip() == "Qwen":
            break
        model_button.click()
    if model_button.inner_text().strip() != "Qwen":
        raise RuntimeError(
            f"Could not select Qwen; current model is "
            f"{model_button.inner_text().strip()}"
        )

    composer.fill("nvidia")
    page.locator('button[type="submit"]').click()

    assistant = page.locator('[data-message-role="assistant"]').last
    assistant.wait_for(state="visible", timeout=30_000)
    page.wait_for_function(
        """() => {
          const cards = document.querySelectorAll(
            '[data-message-role="assistant"]'
          );
          const card = cards[cards.length - 1];
          if (!card || card.classList.contains("message--pending")) {
            return false;
          }
          const body = card.querySelector(".message__body");
          return Boolean(body && body.textContent && body.textContent.trim());
        }""",
        timeout=240_000,
    )

    body = assistant.locator(".message__body").inner_text().strip()
    errors: list[str] = []
    for locator in (
        assistant.locator(".message__error"),
        page.locator(".composer__error"),
    ):
        if locator.count() and locator.is_visible():
            errors.append(locator.inner_text().strip())

    normalized = body.lower()
    relevant = "nvidia" in normalized or "英伟达" in body
    placeholder = "run ended before a final answer was produced" in normalized
    passed = bool(body) and relevant and not placeholder and not errors

    print("E2E_RESULT=PASS" if passed else "E2E_RESULT=FAIL")
    print("MODEL=Qwen")
    print(f"ANSWER_CHARS={len(body)}")
    print("ANSWER_BEGIN")
    print(body[:4000])
    print("ANSWER_END")
    if errors:
        print("PAGE_ERRORS=" + " | ".join(errors))
    if console_errors:
        print("CONSOLE_ERRORS=" + " | ".join(console_errors[:10]))
    if failed_requests:
        print("FAILED_REQUESTS=" + " | ".join(failed_requests[:10]))

    browser.close()
    raise SystemExit(0 if passed else 1)
PY
```

只有输出 `E2E_RESULT=PASS` 才算通过。保留回答文本，便于判断回答是否相关，而不是只检查
HTTP 200。

## 2. opencli 工具链检查

基础 E2E 可以在搜索工具不可用时仍由模型生成回答，因此你必须单独检查 opencli：

```bash
opencli doctor
opencli profile list
```

完整工具链成功标准：

- Daemon 是 `[OK]`；
- Extension 是 `[OK]`；
- Connectivity 是 `[OK]`；
- 默认 profile 是当前 connected profile。

如果 connected profile 和 default profile 不一致，先向用户确认，再切换：

```bash
opencli profile use <connected-profile>
opencli doctor
```

不要擅自选择一个未连接的 profile。完成切换后，重新运行基础页面 E2E。

## 3. macOS 自签名证书错误

如果页面显示 `Connection error.`，先直接验证模型：

```bash
.venv/bin/python - <<'PY'
from deepfind.config import Settings

settings = Settings.from_env(require_api_key=False).with_qwen_remote()
response = settings.new_client().chat.completions.create(
    model=settings.model,
    messages=[{"role": "user", "content": "Reply with OK only."}],
    max_tokens=8,
)
print((response.choices[0].message.content or "").strip())
PY
```

如果异常链包含 `CERTIFICATE_VERIFY_FAILED` 和
`self-signed certificate in certificate chain`，不要禁用 TLS 验证。生成临时 CA 合并包：

```bash
mkdir -p tmp
CA_FILE="$PWD/tmp/reg-test-ca.pem"
.venv/bin/python -m certifi > "$CA_FILE"
security find-certificate -a -p >> "$CA_FILE"
SSL_CERT_FILE="$CA_FILE" .venv/bin/python - <<'PY'
from deepfind.config import Settings

settings = Settings.from_env(require_api_key=False).with_qwen_remote()
response = settings.new_client().chat.completions.create(
    model=settings.model,
    messages=[{"role": "user", "content": "Reply with OK only."}],
    max_tokens=8,
)
print((response.choices[0].message.content or "").strip())
PY
```

输出 `OK` 后，用同一个 CA 环境启动后端：

```bash
SSL_CERT_FILE="$PWD/tmp/reg-test-ca.pem" \
  .venv/bin/deepfind-web --host 127.0.0.1 --port 8000
```

然后重新运行页面 E2E。测试完成并停止后端后，可删除明确的临时文件：

```bash
rm "$PWD/tmp/reg-test-ca.pem"
```

## 4. 高级回归

基础测试通过后，再按需要执行：

1. 不切换模型，验证默认 DeepSeek；
2. 依次切换 Mimo、MiniMax 和 GLM；
3. 打开 Tools 菜单，确认页面没有展示后端未注册的工具；
4. 运行 `npm test -- --run` 和 `npm run build`；
5. 检查浏览器 console、失败请求和后端日志。

默认模型失败但 Qwen 通过时，不要把问题归类为页面回归；记录为对应模型端点或网络问题。
Qwen 也失败时，再检查 CA、API Key、DNS 和代理配置。
