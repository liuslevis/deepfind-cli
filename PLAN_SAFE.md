# DeepFind-CLI 公网暴露安全加固方案

> ⚠️ **结论：当前状态下，绝对不能暴露到公网。** 没有任何认证机制，任何人都能调用全部 API、读取所有聊天记录、通过 SSRF 探测内网、甚至借助路径穿越读取 `.env` 中的 API Key。

---

## 一、威胁模型

你的场景：手机通过公网访问 `http://localhost:5173`（Vite dev server → FastAPI 8000）。

攻击者只需知道你的公网 IP + 端口，即可：
- 调用所有 API（无认证）
- 读取 / 删除你全部对话记录
- 通过 `/api/files?path=` 读取服务器任意文件（含 `.env`、SSH key 等）
- 通过 `web_fetch` / `browser_fetch` 探测你的内网服务
- 消耗你的 Qwen / MiMo / MiniMax API 额度（无限速）
- 通过 Vite dev server 获取完整前端源码和 source map

---

## 二、漏洞清单

### 🔴 CRITICAL — 必须在暴露公网前修复

| # | 漏洞 | 位置 | 说明 |
|---|------|------|------|
| C1 | **零认证** | `web_api.py:42-50` | 所有端点无需任何凭证即可调用。CORS 设置为 `allow_origins=["*"]`。 |
| C2 | **路径穿越读取任意文件** | `web_api.py:109-119` | `/api/files?path=C:\Users\david\.env` 可读取 API Key。虽然检查了 `relative_to(root)`, 但 symlink 和 `..` 在 Windows 下有绕过风险。 |
| C3 | **API Key 泄露** | `config.py:48-62`, `.env` | 路径穿越一旦成功，`.env` 中全部密钥（Qwen/MiMo/MiniMax/Gemini）全部泄露。 |
| C4 | **SSRF — 任意 URL 抓取** | `tools.py:1165-1242`, `web_fetch.py:84-94` | `web_fetch` 和 `browser_fetch` 可请求任意 URL，包括 `http://127.0.0.1:11434`、`http://192.168.x.x` 等内网地址，无 IP 黑名单。 |
| C5 | **聊天数据全局可读** | `chat_store.py:31-97` | 所有对话存储在 `tmp/web/chats/`，无用户隔离。任何人可列出、读取、删除所有对话。 |
| C6 | **Vite Dev Server 暴露源码** | `vite.config.ts:6-13` | dev 模式下暴露完整 TypeScript 源码、source map，攻击者可分析前端逻辑。 |

### 🟠 HIGH — 暴露后被利用风险高

| # | 漏洞 | 位置 | 说明 |
|---|------|------|------|
| H1 | **无速率限制** | `web_api.py` 全局 | 攻击者可无限调用 `/api/chats/{id}/messages/stream`，每次触发 LLM 调用 + 工具链，快速耗尽 API 额度和服务器资源。 |
| H2 | **请求体无大小限制** | `web_api.py:85-107` | `SendMessageRequest.content` 无长度限制，可发送数 MB 消息造成内存/磁盘 DoS。 |
| H3 | **browser_fetch headless=false** | `tools.py:1202-1242` | LLM agent 可指定 `headless=False` 打开可见浏览器窗口，在服务器上弹出 GUI。 |
| H4 | **子进程错误信息泄露** | `tools.py` 各处 | 错误信息包含完整命令行、stderr、文件路径等，暴露服务器内部结构。 |
| H5 | **无 HTTPS** | 全局 | HTTP 明文传输，手机在公共 WiFi 下所有内容（含 API Key header）可被嗅探。 |

### 🟡 MEDIUM — 应尽快修复

| # | 漏洞 | 位置 | 说明 |
|---|------|------|------|
| M1 | **缺少安全响应头** | `web_api.py` | 无 CSP、X-Frame-Options、HSTS、X-Content-Type-Options 等头。 |
| M2 | **子进程 URL 参数未校验** | `xhs_transcribe.py:58-121`, `youtube_audio_transcribe.py` | ffmpeg/yt-dlp 的 URL 参数未做格式校验，特殊字符可能被解释为命令行参数。 |
| M3 | **浏览器 profile 持久化** | `browser_fetch.py:22-24` | Playwright profile 存在 `tmp/browser_profile/`，累积 cookie/缓存，隐私风险。 |
| M4 | **chat_id 路径注入** | `chat_store.py:83-84` | `_path()` 直接将 `chat_id` 拼接为文件名（`f"{chat_id}.json"`），未校验 `../` 等特殊字符。 |
| M5 | **无审计日志** | `web_api.py` | 无 API 调用日志，被攻击后无法追溯。 |

### 🟢 LOW — 可后续优化

| # | 漏洞 | 位置 | 说明 |
|---|------|------|------|
| L1 | 无证书 pinning | `web_fetch.py`, `gen_img.py` | API Key 通过 HTTPS 发送但无 pinning |
| L2 | 外部 CLI 工具无完整性校验 | `tools.py:2469-2492` | `shutil.which()` 查找 PATH 上的可执行文件，无签名/版本校验 |
| L3 | 生成的 slides HTML 无 CSP | `gen_slides.py` | 虽然用了 `html.escape()`，但生成的 HTML 无 CSP header |

---

## 三、加固方案

### Phase 1: 最小可用安全（暴露公网前必须完成）

#### 1.1 添加 Token 认证

在 FastAPI 中添加 Bearer Token 认证中间件，所有 `/api/*` 端点必须携带 token。

```
# .env 新增
DEEPFIND_WEB_TOKEN=<随机生成的长 token>
```

```python
# web_api.py 添加依赖
from fastapi import Depends, Header

def _require_auth(authorization: str = Header(...)) -> None:
    expected = os.environ.get("DEEPFIND_WEB_TOKEN", "")
    if not expected:
        raise HTTPException(503, "DEEPFIND_WEB_TOKEN not configured")
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(401, "invalid token")
```

前端在 localStorage 存 token，每次请求带上 `Authorization: Bearer <token>` header。

#### 1.2 收紧 CORS

```python
allow_origins=["https://your-domain.com"]  # 替换为实际域名
# 或不暴露 CORS，前后端同源部署
```

#### 1.3 修复路径穿越

```python
# web_service.py - resolve_file_path 加固
def resolve_file_path(self, raw_path: str) -> Path:
    # 只允许相对路径
    if os.path.isabs(raw_path):
        raise ValueError("absolute paths not allowed")
    path = (self._repo_root / raw_path).resolve()
    # 严格检查 resolve 后仍在 repo 内
    if not str(path).startswith(str(self._repo_root.resolve())):
        raise ValueError("path escapes repository root")
    # 禁止 symlink
    if path.is_symlink():
        raise ValueError("symlinks not allowed")
    # 禁止敏感文件
    BLOCKED = {".env", ".git", "pyproject.toml", "uv.lock"}
    if path.name in BLOCKED or any(part.startswith(".") for part in path.parts):
        raise ValueError("access denied")
    return path
```

#### 1.4 SSRF 防护

```python
# web_fetch.py 添加 URL 校验
import ipaddress
from urllib.parse import urlparse

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
]

def validate_fetch_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme: {parsed.scheme}")
    import socket
    for addr_info in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        ip = ipaddress.ip_address(addr_info[4][0])
        for network in _BLOCKED_NETWORKS:
            if ip in network:
                raise ValueError(f"access to private network blocked: {ip}")
```

#### 1.5 chat_id 校验

```python
# chat_store.py
import re
_CHAT_ID_RE = re.compile(r"^chat_[a-f0-9]{32}$")

def _path(self, chat_id: str) -> Path:
    if not _CHAT_ID_RE.match(chat_id):
        raise ValueError(f"invalid chat_id: {chat_id}")
    return self.root / f"{chat_id}.json"
```

#### 1.6 使用生产构建 + HTTPS

```bash
# 构建前端静态文件
cd web && npm run build

# 用 HTTPS 反代（推荐 Caddy，自动 Let's Encrypt）
# Caddyfile:
your-domain.com {
    reverse_proxy 127.0.0.1:8000
}

# 启动后端（不暴露 8000 端口到公网）
deepfind-web --host 127.0.0.1 --port 8000
```

**绝对不要把 `npm run dev` (Vite dev server) 暴露到公网。**

### Phase 2: 加固（暴露后尽快完成）

#### 2.1 速率限制

```python
# 安装: pip install slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/chats/{chat_id}/messages/stream")
@limiter.limit("10/minute")
def stream_message(...): ...

@app.post("/api/chats")
@limiter.limit("30/minute")
def create_chat(...): ...
```

#### 2.2 请求体大小限制

```python
# SendMessageRequest 加验证
class SendMessageRequest(BaseModel):
    content: str = Field(..., max_length=10000)
    mode: ChatMode
    model_target: ModelTarget = "qwen"
```

#### 2.3 安全响应头

```python
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=()"
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

app.add_middleware(SecurityHeadersMiddleware)
```

#### 2.4 隐藏内部错误详情

```python
# tools.py _run() 方法
# 生产环境下不返回 stderr 和完整命令行
if os.environ.get("DEEPFIND_ENV") == "production":
    return {"ok": False, "tool": tool, "error": "tool execution failed"}
```

#### 2.5 browser_fetch 禁止 headless=false

```python
# tools.py browser_fetch
def browser_fetch(self, url: str, prompt: str, headless: bool = True) -> dict:
    # 公网部署时强制 headless
    headless = True  # 忽略 LLM 传来的值
```

### Phase 3: 长期改进

| 项目 | 说明 |
|------|------|
| OAuth2 / 密码登录 | 多用户场景需要完整的用户系统 |
| 对话隔离 | 每个用户只能访问自己的对话 |
| API 额度管理 | 按用户跟踪和限制 LLM API 调用次数 |
| 审计日志 | 记录所有 API 调用的 IP、时间、参数 |
| 子进程沙箱 | 用 seccomp / AppArmor 限制子进程权限 |
| 定期密钥轮换 | 定期更换 API Key 和 Web Token |

---

## 四、快速部署建议（最小安全配置）

```
                    ┌──────────────┐
  手机 ──HTTPS──▶  │  Caddy 反代   │ ──HTTP──▶ FastAPI :8000
                    │  (自动证书)    │           (仅监听 127.0.0.1)
                    │  + 限速       │           (Token 认证)
                    └──────────────┘           (静态文件: web/dist)
```

1. `npm run build` 构建前端 → `web/dist/`
2. FastAPI 挂载 `web/dist` 静态文件（已支持）
3. Caddy 反代 + 自动 HTTPS
4. `.env` 添加 `DEEPFIND_WEB_TOKEN=<random>`
5. 防火墙只开放 443 端口

---

## 五、总结

| 优先级 | 必须修 | 工作量 |
|--------|--------|--------|
| 🔴 暴露前 | Token 认证 + CORS + 路径穿越 + SSRF + 生产构建 + HTTPS | ~2-3 天 |
| 🟠 暴露后 | 速率限制 + 请求大小限制 + 安全头 + 错误隐藏 | ~1 天 |
| 🟡 持续 | 审计日志 + 用户系统 + 沙箱 | 持续迭代 |
