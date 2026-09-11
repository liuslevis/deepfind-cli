---
title: Coding Tool 规范
description: 在一次性容器沙箱中执行 Python 编码任务的最小实现规范。
---

# Coding Tool

## 1. 目标

为 DeepFind 增加一个模型可调用的 `coding(query, context)` 工具。工具在独立临时目录 `sandbox/{task_name}` 中完成 Python 编码任务，可以创建文件、运行 Python、安装 PyPI 依赖，并返回结构化结果。

首版只解决一个问题：**让模型安全地运行一次 Python 编码任务，而不污染宿主 Python、文件系统、凭据或网络环境。**

## 2. 核心原则

1. **容器才是安全边界。** 子进程、工作目录和虚拟环境只能防止依赖污染，不能替代容器隔离。
2. **默认拒绝。** 容器运行时不可用、隔离参数无法生效或路径校验失败时，任务直接失败；不得自动退化为宿主执行。
3. **最小权限。** 容器以非 root 用户运行，默认无网络、无宿主凭据、无额外 Linux capabilities，且只能写当前任务目录。
4. **单任务单目录。** 每次调用创建新的不可预测目录，不复用其他任务的文件或 Python 环境。
5. **最小产品面。** 首版不提供通用终端、持久容器、远程执行、多语言运行时或权限审批系统。

## 3. 用户接口

### 3.1 工具签名

`coding(query: str, context: str | None = None) -> CodingResult`

- `query`：必填，描述要完成的编码任务。
- `context`：可选，只包含完成任务所需的文本背景。它不是文件路径，也不自动授予宿主文件访问权。

输入限制：

- `query` 去除首尾空白后不能为空。
- `query` 最大 20,000 字符。
- `context` 最大 100,000 字符。
- 输入仅作为数据写入任务请求，不拼接为宿主 shell 命令。

### 3.2 返回值

`CodingResult` 至少包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `ok` | `bool` | 任务是否成功完成 |
| `task_id` | `str` | 服务生成的任务标识，格式 `task_<uuid4 hex>`，与 webapp 中 `chat_`、`msg_` 前缀风格一致 |
| `status` | `str` | `completed`、`failed`、`timed_out` 或 `cancelled` |
| `answer` | `str` | 面向模型或用户的最终说明 |
| `artifacts` | `list[Artifact]` | 任务产物清单 |
| `commands` | `list[CommandResult]` | 容器内命令记录 |
| `error` | `CodingError \| null` | 失败原因 |
| `duration_ms` | `int` | 总执行时间 |

`Artifact` 只返回相对于任务目录的路径、文件大小和媒体类型，不返回宿主绝对路径。

`CommandResult` 包含命令、退出码、耗时，以及截断后的标准输出和标准错误。输出不得包含注入到运行器内部的控制数据。

## 4. 任务目录

宿主目录结构：

```text
sandbox/
  task_<uuid4-hex>/
    request.json
    workspace/
    result.json
    logs/
```

- `task_id` 格式为 `task_<uuid4-hex>`（例：`task_9f3c7a1b2d4e4f56a1b3c4d5e6f70819`），与 `deepfind/chat_store.py` 中 `chat_<uuid4-hex>`、`deepfind/web_service.py` 中 `msg_<uuid4-hex>` 保持一致的前缀风格。
- `task_id` 是目录名和安全边界的唯一来源；uuid4 提供 122 位随机，足以在个人单机场景下防止路径猜测和并发冲突。
- 展示层若需要人类可读标签，由 UI/CLI 从 query 截断得到，不写入路径、日志键、缓存键或安全决策。
- 任务根目录 (`DEEPFIND_CODING_ROOT`) 默认为工作目录下 `sandbox/`。项目为个人单机运行，接受此路径在 git 工作树内的风险；仓库应在 `.gitignore` 中排除 `sandbox/`，避免误提交产物。
- 所有路径必须先做规范化，再验证其仍位于本次任务根目录内。
- 拒绝绝对路径、`..`、符号链接逃逸、junction/reparse point 逃逸和硬链接越界。
- 容器只挂载 `workspace/`，不得挂载仓库根目录、用户主目录、系统临时目录或容器运行时 socket。
- 任务完成后默认保留产物一段可配置时间，再由后台清理；清理只能删除已验证属于 `sandbox/` 的具体任务目录。

## 5. 容器安全基线

首版支持 Docker 或 Podman，使用固定摘要的 Python 基础镜像。运行参数必须满足：

- 非 root 用户，固定容器内 UID/GID。
- 根文件系统只读。
- 仅 `/workspace` 和私有临时目录可写。
- `/tmp` 使用容器私有 `tmpfs`，设置容量和执行限制。
- 默认禁用网络。
- 丢弃全部 Linux capabilities。
- 启用 `no-new-privileges`。
- 使用固化在仓库中的 seccomp allowlist（基于运行时默认裁剪 `unshare`、`keyctl`、`ptrace`、`bpf`、`perf_event_open` 等 Python 任务不需要的调用）。加载失败必须以 `sandbox_unavailable` 拒绝启动，不得回退到运行时默认。
- 禁止 privileged 模式、host PID、host IPC、host network 和设备映射。
- 不挂载 Docker/Podman socket。
- 不透传宿主环境变量；只使用显式白名单变量。
- 不透传 SSH agent、Git credentials、云凭据、API key、浏览器资料或用户主目录。
- 设置 CPU、内存、进程数、文件大小、打开文件数和总运行时间上限。

建议默认值：

| 资源 | 默认值 |
|---|---|
| 总超时 | 120 秒 |
| CPU | 1 核 |
| 内存 | 512 MiB |
| 进程数 | 64 |
| 单文件大小 | 32 MiB |
| 全部产物 | 128 MiB |
| stdout + stderr | 各 1 MiB |
| 最大并发任务数 | 4 |

若运行时不支持磁盘配额（如部分 rootless Podman 场景），必须以 tmpfs 大小上限替代整体产物限制，不得静默放行。

达到最大并发时，新任务直接返回 `resource_limit`，不做无界排队。

Windows 和 macOS 上必须通过 Docker Desktop、Podman Machine 或等价 Linux VM 后端运行。不得把 Windows ACL、macOS 文件权限或 Python 虚拟环境单独宣称为安全沙箱。

### 5.1 启动期能力探测与预热

服务进程启动时（而非每次任务调用时）执行一次探测：

- 校验运行时可用、镜像 digest 存在、非 root、只读 rootfs、seccomp、caps drop、no-new-privileges 等参数全部生效。
- 结果连同镜像 digest、运行时版本一并缓存；镜像 digest 或运行时版本变化时缓存必须失效。
- 探测失败一律以 `sandbox_unavailable` 拒绝注册工具，不得延迟到首次调用时才发现。

任务运行时的第 7 节 step 5 只做轻量校验（缓存命中即可），避免每次调用重复 200–500ms 的探测开销。

### 5.2 镜像预置虚拟环境模板

镜像内需要预置一个 `/opt/venv-template` 目录（构建期生成）。任务需要独立 venv 时通过复制模板获得，替代 `python -m venv` 的冷启动。模板属于镜像的一部分，与镜像同 digest 校验，安全属性等价。

## 6. Python 与依赖安装

- 容器镜像提供固定 Python 大版本和基础工具，并预置 `/opt/venv-template` 供任务复制。
- Phase 1 默认断网，任务不需要额外依赖时**不创建** `.venv`，直接使用镜像 Python，避免无谓的冷启动开销。
- 当任务需要隔离依赖或需要写入 site-packages 时，通过复制 `/opt/venv-template` 到 `/workspace/.venv` 获得独立环境，而不是运行 `python -m venv`。
- `python` 和 `pip` 必须解析到该虚拟环境。
- 禁止 `--user` 安装和写入全局 site-packages。
- `HOME`、`TMPDIR`、pip cache 和工具 cache 均指向任务内或容器私有临时目录。
- 不读取宿主 `pip.conf`、`.pypirc`、`.netrc` 或 keyring。
- 安装命令必须使用参数数组启动，不通过 `shell=True`。
- 默认断网时，依赖只能来自镜像预装包或显式配置的只读 wheelhouse。

网络安装是后续可选能力，不属于默认模式。若启用，必须：

1. 由服务配置允许，而不是由 query 自行开启。
2. 只允许访问配置的包索引。
3. 设置下载大小、安装时间和依赖数量限制。
4. 不向容器暴露私有索引凭据；需要私有源时使用短期、任务级凭据。
5. 在结果中记录安装的包名和版本。

## 7. 执行流程

一次调用按以下顺序执行：

1. 校验 `query` 和 `context`。
2. 生成 `task_id`、安全 `task_name` 和任务目录。
3. 原子创建目录，并拒绝已存在路径。
4. 写入只读的 `request.json`。
5. 检查容器运行时、固定镜像和所需安全能力。
6. 使用安全基线启动一次性容器。
7. 容器内代理读取请求，在 `/workspace` 中创建代码、安装允许的依赖并执行验证。
8. 容器写出 `result.json` 和产物。
9. 宿主校验结果大小、JSON schema、路径归属和符号链接。
10. 停止并删除容器，返回 `CodingResult`。

无论成功、失败或超时，都必须执行容器清理。清理失败应记录为独立错误，不得把失败任务改报为成功。

## 8. 容器内代理

容器内代理保持极简：

- 输入：`request.json`。
- 可用动作：写工作区文件、创建任务虚拟环境、安装允许的 Python 包、运行参数化命令、读取命令结果。
- 输出：`result.json`。
- 不直接访问 DeepFind 的宿主进程。
- 不持有容器管理权限。
- 不允许嵌套启动容器。

首版可使用单个 Python runner 实现固定循环，不需要独立 agent 服务、消息队列或数据库。

## 9. DeepFind 集成

在现有 `Toolset` 中增加唯一工具名 `coding`：

- 在工具 catalog 中暴露 `query` 和可选 `context`。
- 遵循现有 `enabled_tools` 过滤。
- 返回现有工具调用层可序列化的字典。
- Web 端通过现有 `selected_tools` 机制启用或禁用。
- chat 模式继续禁用全部工具，不为 `coding` 添加例外。
- CLI 的 `--list-tools` 自动展示该工具。

建议模块边界：

```text
deepfind/
  coding.py          # 输入校验、任务生命周期、结果模型
  coding_runtime.py  # Docker/Podman 参数构造与执行
  tools.py           # 仅注册和适配 coding 工具
```

不要把容器生命周期、路径校验和结果解析堆入 `tools.py`。

## 10. 错误模型

错误必须包含稳定的 `code` 和安全的 `message`：

| code | 场景 |
|---|---|
| `invalid_input` | query/context 不合法 |
| `invalid_path` | 路径越界或链接逃逸 |
| `runtime_unavailable` | Docker/Podman 不可用 |
| `sandbox_unavailable` | 安全参数或隔离能力无法生效 |
| `image_unavailable` | 固定镜像不可用 |
| `dependency_denied` | 依赖来源或网络策略不允许 |
| `resource_limit` | CPU、内存、磁盘或进程限制触发 |
| `timed_out` | 超过任务时限 |
| `execution_failed` | 容器内任务失败 |
| `invalid_result` | result.json 或产物不可信 |
| `cleanup_failed` | 容器或任务目录清理失败 |

不得把完整宿主命令行、环境变量、绝对路径或容器运行时内部信息返回给模型。详细诊断仅写入受控日志。

## 11. 威胁模型

必须防御：

- 恶意 query 诱导读取宿主文件或凭据。
- 通过 `../`、绝对路径、符号链接、junction 或硬链接逃逸工作区。
- Python 包安装污染宿主解释器或缓存。
- 依赖安装脚本执行恶意代码。
- fork bomb、无限循环、内存耗尽、磁盘填满和超量日志。
- 通过并发拉起大量任务耗尽宿主 CPU、内存、PID 或磁盘。
- 通过网络外传数据或下载第二阶段 payload。
- 访问容器运行时 socket 后控制宿主。
- 复用目录导致跨任务数据泄漏。
- 将容器内路径或输出误当作可信数据。

不承诺防御：

- 容器运行时或宿主内核的零日漏洞。
- 用户明确开启网络后，允许目标返回的恶意内容。
- 宿主管理员或同等权限进程读取任务目录。

## 12. 可观测性

每次任务记录：

- `task_id`、状态、开始/结束时间和耗时。
- 镜像摘要和运行时类型，不记录敏感运行时配置。
- 容器退出码和资源限制触发原因。
- 安装包名与版本。
- 产物相对路径、大小和内容哈希。

日志不得记录完整 `context`，默认只记录长度和哈希。输出进入日志前必须做大小限制和控制字符处理。

## 13. 配置

首版只提供以下服务端配置：

| 配置 | 含义 |
|---|---|
| `DEEPFIND_CODING_ENABLED` | 是否注册 coding 工具，默认关闭 |
| `DEEPFIND_CODING_RUNTIME` | `docker` 或 `podman` |
| `DEEPFIND_CODING_IMAGE` | 固定镜像引用，生产环境必须含 digest |
| `DEEPFIND_CODING_ROOT` | 任务根目录，默认工作目录下 `sandbox/`，仓库应在 `.gitignore` 中排除 |
| `DEEPFIND_CODING_TIMEOUT` | 总超时 |
| `DEEPFIND_CODING_MAX_CONCURRENT` | 最大并发任务数，默认 4，超限返回 `resource_limit` |
| `DEEPFIND_CODING_NETWORK` | 是否允许受限包安装网络，默认关闭 |
| `DEEPFIND_CODING_RETENTION` | 任务目录保留时长，默认 `0`（任务完成立即清理），仅调试时显式打开 |

未知配置值必须报错，不使用隐式宽松默认值。

## 14. 验收标准

功能验收：

- 工具可以创建并运行一个只依赖 Python 标准库的程序。
- 成功结果包含 answer、命令记录和产物清单。
- 失败、超时和资源限制返回稳定错误码。
- `selected_tools=["coding"]` 可单独启用该工具。

隔离验收：

- 容器内无法写入任务目录外的挂载路径。
- 容器内看不到宿主主目录、仓库根目录和敏感环境变量。
- 默认无法建立外部网络连接。
- `pip install` 不改变宿主 Python、用户 site-packages 或宿主 cache。
- 尝试挂载 socket、使用 privileged 模式或缺失安全参数时启动失败。
- 路径穿越、符号链接、junction/reparse point 和硬链接逃逸测试失败关闭。
- 超时后容器不再运行，且没有孤儿子进程。
- 两个并发任务不能读取彼此的目录。

## 15. 实施阶段

### Phase 1：最小安全版本

- Docker/Podman 单次容器，启动期完成一次能力探测与预热。
- 固定 Python 镜像，内含预置 `/opt/venv-template`。
- 默认预装 numpy pandas Scikit-learn  SciPy Statsmodels Seaborn
- 项目自带 seccomp allowlist。
- 默认断网。
- 标准库和镜像预装依赖。
- 并发上限、结构化结果、资源限制、路径校验和清理。
- 任务根目录默认位于工作目录下 `sandbox/`（个人单机场景），仓库需将其加入 `.gitignore`。
- Toolset、CLI 和 Web 工具选择接入。

### Phase 2：受限依赖安装

- 只读 wheelhouse。
- 可选包索引 allowlist。
- 安装审计与下载限制。

### Phase 3：体验增强

- 任务取消。
- 进度事件。
- 产物预览。
- 可配置保留和手动删除。

任何阶段都不得以“先在宿主运行，后续再加沙箱”作为实现路径。
