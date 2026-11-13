# Bloomberg 视频每日总结 → Kindle SOP

跟踪表见本文件末尾「转译记录」，每次执行完都要更新它。

**已配置每天 07:00（Asia/Shanghai）自动执行**（DSH schedule `schedule-cfc66597-68ac-4ea1-8e50-f8d60e1da84f`，
daily）。自动任务就是按本文件第 1→5 节走一遍；它触发时依赖：
Chrome 开着且 opencli 扩展 connected、`.env` 里的 SMTP/Kindle 配置有效。
任一项失效，任务会中止并列出已生成但未发出的 PDF，手动按第 4 节补发即可。

## 0. 必读：当前环境已知限制

先读一遍，别在这几个坑上浪费时间：

1. **取视频列表用 opencli 抓页面，不用 `bili user-videos`。**
   `bili user-videos` 依赖登录态（`~/.bilibili-cli/credential.json` 的 `sessdata`
   经常是空的），一未登录就 412，`-n 30` 也容易 412。
   opencli 走你已登录的 Chrome（扩展已装好、`opencli doctor` 显示 connected），
   从投稿页第一页直接读卡片，还能拿到相对发布时间。见第 1 节。
2. **标题里的 `YYMMDD` 是节目日期，不是上架日期。** 261107 的节目可能在 11/08 上架
   （批量上传）。opencli 拿到的相对时间才是上架时间；两者不一致**很正常**，
   `fetch_uploads.py` 只会在整页都没人用这个日期时才标「日期冲突」。
   判断「今天有没有新视频」用**上架时间**。
3. **`send_kindle` 工具在当前宿主机上必定失败。** 它在进程临时目录
   （`%TEMP%\dsh-*\deepfind-kindle-*`）里渲染 PDF 时权限被拒：
   `Failed to render summary.md as PDF: [Errno 13] Permission denied`。
   改标题、改文件名、换成 `.txt` 都没用，`kindle_pdf.convert_to_pdf` 的兜底
   也会因为同级子目录写不进去而失败。**不要在这个错误上反复重试。**
   要发 Kindle 就用「4. 发送」里的直发脚本：在工作目录里渲染 PDF，再调
   `send_kindle._deliver`，SMTP 参数和标题规则完全一样。
4. **`bili_summary` 有 `summary.md` 缓存。** 同一个 BV 调第二次会直接返回
   `cached: true`，不会重新转写。需要重做时才传 `refresh=True`。
5. **QQ 邮箱 SMTP 授权码会失效/限频。** 2026-11-09 21:50 起返回
   `535 Login fail … login frequency limited`，11/10 上午换新授权码后恢复
   （`235 Authentication successful`）。每封邮件最多重试 1 次、间隔 60 秒，再失败就停并记录待补发。
   判断方法：直接用 `smtplib.SMTP_SSL('smtp.qq.com',465).login(...)` 打一次，
   `535` 是账号侧问题（换授权码/解限频），超时或断连才是网络问题。

## 1. 取最近 3 天的视频（opencli）

今天是 D 日，就取 D、D-1、D-2 三天（比如 11/10 → 11/10、11/9、11/8）。

```powershell
& ".\.venv\Scripts\python.exe" `
  ".\skill\scripts\fetch_uploads.py"
```

它会：
- `opencli browser bili open https://space.bilibili.com/670833898/upload/video`，
  读**第一页**所有 `.bili-video-card`：标题（`title` 属性，完整原文）、BV 链接、
  相对时间（`15小时前` / `昨天 07:56` / `2天前` / `11月6日`）、时长；
- 按页面相对时间定上架日期，`YYMMDD` 只作兜底；整页其他视频都没用过的标题日期
  才标 `<-- 日期冲突`；
- 只保留 D-2 ~ D 内的条目，写进 `tmp/bili_uploads.json`，格式和
  `bili user-videos --json` 兼容（另外多了 `published_date` / `relative` /
  `date_conflict` 字段）。

opencli 的两个坑（脚本已经处理，改脚本时注意）：
- 要用 `opencli browser <session> eval <js>`，`<session>` 是位置参数，必须放在
  `eval` 前面；`--window` 只能放在 `browser` 之后、session 之前；
- JS 会被 PowerShell 按行拆开，**必须写成单行、只用单引号、用 IIFE 包住并
  `return JSON.stringify(...)`**，否则报 `SyntaxError: Unexpected end of input`。

`bili` 只用来补元数据（可选），例如 `bili video BVxxxx --json` 拿时长/作者。

**定时任务只跑到这里。** 输出里第一行就是最新上架的视频：

- 有一条上架日期是今天（D 日）就说明今天有新视频；
- 一条都没有，就说明今天还没上传，**不要**把旧的当新的处理；
- opencli 连不上就 `opencli doctor` 看扩展是否 connected，再不行才退回
  `bili user-videos 670833898 -n 15 --json`（前提是 `bili status` 是登录态）。

筛选规则：
- 只留**上架日期**在 D、D-1、D-2 内的条目（`fetch_uploads.py` 已经过滤好）；
- 已在本文件「转译记录」里、且「转译」「发送」都打勾的，跳过；
- 转译过但没发送的，只补发送（见第 4 步），不重新转写；
- 拿不准的逐条看 `relative`；**别猜**。

## 2. 转写 + 生成摘要

标准路径是 **web CLI**（它会把过程落到 Web UI，并走和 UI 一样的 agent 流程）：

```powershell
& ".\.venv\Scripts\deepfind-chat.exe" "总结视频【261109 Bloomberg Weekend】并发送 kindle https://www.bilibili.com/video/BV1AKWx6BEKp/"
```

要点：
- 后端在 `http://127.0.0.1:8000`，要先确认 `/api/health` 可用：
  `Invoke-WebRequest http://127.0.0.1:8000/api/health`
- **后端 token 必须和 `.env` 里的 `DEEPFIND_WEB_TOKEN` 一致。** 健康检查返回
  `requires_token: true`、而 `/api/chats` 返回 401，就说明这个后端是用别的
  token 起的：停掉它（`netstat -ano | Select-String ':8000\s+.*LISTENING'`
  拿到 PID → `Stop-Process`），再用 `.venv\Scripts\deepfind-web.exe --host
  127.0.0.1 --port 8000` 重启，让它重新读 `.env`。**不要自己编一个 token。**
- `bili_summary` 输出一般 3–10 分钟一条，直接跑会超时；用后台 job 跑，完成后读
  job 输出。日志里的中文是 GBK 乱码，不用管。
- web CLI 跑到 `send_kindle` 报 `[Errno 13] Permission denied` 是正常的
  （见第 0 节第 3 条）。关键看 `bili_summary` 是否 `ok`，并确认
  `assets/video/bili/<BV>/text/summary.md` 已生成。
- 一条视频一个 chat，把 chat id 记下来（`Chat ready in the UI: chat_xxx`）。

批量兜底（后端不可用、或一次要跑好几条时）：直接调工具再自己发，参数和 CLI 一样，
参考 `tmp/run_three.py`：

```python
from deepfind import tools
from deepfind.config import Settings
toolset = tools.Toolset(Settings.from_env())
print(toolset.bili_summary("BV1AKWx6BEKp", title="261109 Bloomberg Weekend"))
```

## 3. 检查摘要

发之前先抽查 `summary.md`：
- 第一行必须是 `# <标题>`；
- 至少要有「摘要」「要点」这类小节；
- 正文日期不能和标题日期冲突（例如标题写 `261107`、正文写非农是周五 11/6，
  就该按正文判断真实期号）。冲突时**先问**，确认后再发；
- 发现 `We need ...`、`Need ...` 这类思考残留，一律重跑 `refresh=True`。

## 4. 发送 Kindle

用工作目录渲染 + 直发脚本，一条视频对应一份 PDF，同时把 Markdown 存一份到
`assets/kd/`：

```powershell
& ".\.venv\Scripts\python.exe" `
  ".\tmp\send_summaries_kindle.py" BV1AKWx6BEKp
```

**SMTP 会限频。** 2026-11-09 晚上 QQ 邮箱返回
`535 Login fail. … login frequency limited`，白天完全正常。规则：
- 每封邮件最多重试 **1 次**，间隔 60 秒；再失败就停，别连续重试；
- 失败时 PDF 和 `assets/kd` 存档都已生成，等限频解除后直接重跑同一条命令补发即可
  （`send_one` 会重新渲染，不会重复转写）；
- 判断是不是限频：单独跑一次 `smtplib.SMTP_SSL('smtp.qq.com',465).login(...)`，
  报 `535` 就是账号侧限流，报超时/断连才是网络问题。

规则：
- **PDF 文件名 = 邮件标题 = Kindle 标题**，统一用 `MM/DD BB <节目名>`：
  - `261109 Bloomberg Weekend` → `11/09 BB Weekend`
  - `261107 Bloomberg Wall street week` → `11/07 BB Wall Street Week`
  - `261107 Bloomberg Money` → `11/07 BB Money`
  - `261107 Bloomberg Tech …` → `11/07 BB Tech …`（节目标题里多出来的修饰语保留）
  - `261109 Bloomberg｜The Asia Trade` → `11/09 BB Asia Trade`（全角 `｜` 是
    Bloomberg 分隔符，会被折叠；`The China Show` → `China Show` 同理）
  - 不是 Bloomberg 频道的（如 `261108 CNN News - Anderson Cooper 360`）→
    `11/08 BB CNN News - Anderson Cooper 360`（前缀仍用 `BB`，节目名照原样）
  - 标题不是 `YYMMDD` 开头的（如 `Bloomberg Surveillance 1106`）→ 原样去掉 `Bloomberg`
    → `Surveillance 1106`，**不要编造日期**；
  - 转换逻辑在 `tmp/send_summaries_kindle.py` 的 `kindle_title()`；只想看映射结果时：
    `python tmp/send_summaries_kindle.py --names "261107 Bloomberg Money"`。
- 收件人 `KINDLE_EMAIL`，发件人 `SMTP_USER`，`smtp.qq.com:465` 走 SSL，全部从 `.env` 读；
- 成功标志：脚本返回 `"ok": true`，并且 `assets/kd/<stem>.md` 已写入；
- 如果 `tmp/send_summaries_kindle.py` 不在了，用 `deepfind.kindle_pdf._convert_to_pdf`
  （在工作目录里渲染）加 `deepfind.send_kindle._deliver` 补一个；
- 整个流程**跑完自己核对，不要只信 agent 的口头汇报**：agent 常说「已发送」，
  实际 SMTP 根本没走。以 SMTP 返回结果为准。

## 5. 核对并更新跟踪表

```powershell
# 一条命令：读 tmp/bili_uploads.json，回填标题缓存，检查每条视频的转译/发送状态
& ".\.venv\Scripts\python.exe" `
  ".\skill\scripts\verify_videos.py" `
  --from-uploads ".\tmp\bili_uploads.json"
```

`verify_videos.py` 只读检查 `<BV>/text/` 和 `assets/kd/`，判断「转译」「发送」：
- 转译 = 有 `summary.md`；
- 发送 = 有 PDF **且** 有对应 `assets/kd/*.md` 存档，且 PDF 时间不早于存档太多
  （防止把旧 PDF 误判成已发）；
- `--from-uploads` 会把 `tmp/bili_uploads.json`（opencli 抓的页面）里的标题写进
  `tmp/bili_titles.json`；没有 `YYMMDD` 前缀的条目会继承上一条更新的 Bloomberg
  视频日期——所以别把列表打乱顺序；
- 已知盲区 1：`assets/kd` 是从 11/9 才开始写的，更早单独发过的不在存档里，
  会显示「否」，需要人工确认后手动改表；
- 已知盲区 2：PDF 在存档之前生成，所以一旦重新生成了 PDF（比如换了标题），
  「发送」会掉成「否」，提醒你补发一次——这是有意的保守行为；
- 已知盲区 3：归档名有新旧两套（`Bloomberg 261107 …` 与 `11/08 BB …`），
  脚本两套都认，但存档名不改，表格里以脚本结果为准；
- 已知盲区 4：同一节目重发过多次时，脚本只认**最新**那份 PDF 与其后写入的存档；
  标题换过名（如 Wall street week 的「纪要」版）会显示「否」，需人工确认后手动标注。

把结果贴回本文件末尾的表格（新的放最上面，旧的往下沉，只保留最近两周的，更早的删掉）。

## 6. 转译记录

<!-- 由 `verify_videos.py --from-uploads tmp/bili_uploads.json` 生成；Kindle 标题用新规范，上传标题留作对账 -->

| Kindle 标题 | 上传标题 | BID 链接 | 转译 | 发送 |
|---|---|---|---|---|
| 11/12 BB CNN News -The Lead | 261112 CNN News -The Lead | https://www.bilibili.com/video/BV1DyWR6PELf | 是 | 是 2026-11-13 09:20 |
| 11/12 BB China Show | 261112 Bloomberg｜The China Show P1 | https://www.bilibili.com/video/BV1fpWX66EBH | 是 | 是 2026-11-13 09:27 |
| 11/12 BB China Show | 261112 Bloomberg｜The China Show | https://www.bilibili.com/video/BV1qTW966EEG | 是 | 是 2026-11-13 09:27 |
| 11/12 BB Asia Trade | 261112 Bloomberg｜The Asia Trade | https://www.bilibili.com/video/BV1qTW966EDw | 是 | 是 2026-11-13 09:32 |
| 11/12 BB Balance of Power | 261112 Bloomberg Balance of Power | https://www.bilibili.com/video/BV1qTW966Ez1 | 是 | 是 2026-11-13 09:37 |
| 11/12 BB ABC World News Tonight【Cut黄】 | 261112 ABC World News Tonight【Cut黄】 | https://www.bilibili.com/video/BV1HzW96UEtR | 是 | 是 2026-11-13 09:39 |
| 11/12 BB Surveillance 财经观察 P2 | 261112 Bloomberg Surveillance 财经观察 P2 | https://www.bilibili.com/video/BV1nkW962EsU | 是 | 是 2026-11-13 09:45 |
| 11/12 BB Surveillance 财经观察 | 261112 Bloomberg Surveillance 财经观察 | https://www.bilibili.com/video/BV1tEWS62EzZ | 是 | 是 2026-11-13 09:50 |
| 11/12 BB NBC Morning News NOW | 261112 NBC  Morning News NOW | https://www.bilibili.com/video/BV1nBWQ6rE3P | 是 | 是 2026-11-13 09:57 |
| 11/12 BB Tech ：CoreWeave平息AI恐惧,Anthropic IPO需求增加 | 261112 Bloomberg Tech ：CoreWeave平息AI恐惧，Anthropic IPO需求增加 | https://www.bilibili.com/video/BV1KTWX6QE2f | 是 | 是 2026-11-13 10:01（PDF/邮件标题里全角冒号被折叠成空格） |
| 11/12 BB CNN News Central | 261112 CNN News Central | https://www.bilibili.com/video/BV1cEWX65EBC | 是 | 是 2026-11-13 10:05 |
| 11/11 BB China Show | 261111 Bloomberg｜The China Show | https://www.bilibili.com/video/BV1wNW26fEb3 | 是 | 是 2026-11-12 07:11 |
| 11/11 BB CNN News -Anderson Cooper 360 | 261111 CNN News -Anderson Cooper 360 | https://www.bilibili.com/video/BV1uaW263ECn | 是 | 是 2026-11-12 07:14 |
| 11/11 BB Asia Trade | 261111 Bloomberg｜The Asia Trade | https://www.bilibili.com/video/BV1fUWy6DEv3 | 是 | 是 2026-11-12 07:24 |
| 11/11 BB Tech | 261111 Bloomberg Tech | https://www.bilibili.com/video/BV1XAW163EMt | 是 | 是 2026-11-12 07:28 |
| 11/11 BB ABC World News Tonight【中英】 | 261111 ABC World News Tonight【中英】 | https://www.bilibili.com/video/BV131W16kEkH | 是 | 是 2026-11-12 07:31 |
| 11/11 BB ABC World News Tonight | 261111 ABC World News Tonight | https://www.bilibili.com/video/BV1WMWj6LEaS | 是 | 是 2026-11-12 07:33 |
| 11/11 BB Balance of Power | 261111 Bloomberg Balance of Power | https://www.bilibili.com/video/BV1f7Wj6MEQy | 是 | 是 2026-11-12 07:39 |
| 11/10 BB Brief | 261110 Bloomberg Brief | https://www.bilibili.com/video/BV1ndWj6VEd3 | 是 | 是 2026-11-12 07:43 |
| 11/11 BB Surveillance 财经观察 | 261111 Bloomberg Surveillance 财经观察 | https://www.bilibili.com/video/BV1m6Wj6pEUY | 是 | 是 2026-11-12 07:56 |
| 11/11 BB CNN News Central | 261111 CNN News Central | https://www.bilibili.com/video/BV1KZWV65EkA | 是 | 是 2026-11-12 08:00 |

<!-- 2026-11-13 09:16 手工补跑（当天 07:00 定时任务没跑成：本会话 schedule 列表里已没有
     schedule-cfc66597…，tmp/bili_uploads.json 也停在 07:00:20 没往下走，原因未确认）。
     抓取结果：页面 40 条，D-2 ~ D 窗口（11/11 ~ 11/13）共 21 条；11/13 当天新上架 9 条，
     加上 11/12 上架但没处理的 2 条（Tech BV1KTWX6QE2f、CNN News Central BV1cEWX65EBC），
     共 11 条全部新转写并发出（09:20 ~ 10:05，每条 SMTP attempts=1，没有 535）。
     11/11 那批上轮已发，未重跑。批量脚本：`tmp/run_batch_1113.py`，日志 `tmp/run_batch_1113.log`。
     注意：`run_batch.py` 模式的邮件/PDF 标题取自 `kindle_title(上传标题)`，和 send_summaries_kindle.py
     取自 summary H1 略有出入；Tech 那条的全角「：」被 `kindle_title` 折叠成空格：
     `1112 BB Tech CoreWeave平息AI恐惧,Anthropic IPO需求增加`。
     11/10 那批（China Show/Tech/ABC/AC360/NBC）已滑出 D-2 窗口，移到下方历史记录。
     表格时区为 Asia/Shanghai；「发送」时间以 assets/kd 存档写入时间为准。 -->

<!-- 2026-11-12 07:00 定时任务：页面 40 条，11/12 新上架 10 条（约 11 小时音频），全部转译并发出。
     CNN News Central（BV1KZWV65EkA）首次 `bili audio` 下载 90 秒超时，单独重试后成功（08:00）。
     其中 Tech（BV1KTWX6QE2f）与 CNN News Central（BV1cEWX65EBC）在 07:00 扫描时页面还显示
     「22小时前」，被记成 11/13 上架，实际是 11/12 pre-09:00 发布；11/13 补跑时按 11/12 处理。
     11/09 那批已滑出 D-2 窗口，移到下方历史记录。
     表格时区为 Asia/Shanghai；「发送」时间以 assets/kd 存档写入时间为准。 -->

### 历史记录（已滑出最近 3 天窗口）

| Kindle 标题 | 上传标题 | BID 链接 | 转译 | 发送 |
|---|---|---|---|---|
| 11/10 BB China Show | 261110 Bloomberg｜The China Show | https://www.bilibili.com/video/BV1SsWg6uEQN | 是 | 是 2026-11-11 07:15 |
| 11/10 BB Tech | 261110 Bloomberg Tech | https://www.bilibili.com/video/BV1SsWg6uEa8 | 是 | 是 2026-11-11 07:19 |
| 11/10 BB ABC World News Tonight | 261110 ABC World News Tonight | https://www.bilibili.com/video/BV1TyWT6fENe | 是 | 是 2026-11-11 07:21 |
| 11/10 BB CNN News -Anderson Cooper 360 | 261110 CNN News -Anderson Cooper 360 | https://www.bilibili.com/video/BV1tsWM6aEyt | 是 | 是 2026-11-11 07:24 |
| 11/10 BB NBC Morning News NOW | 261110 NBC  Morning News NOW | https://www.bilibili.com/video/BV1hYWN6bEUt | 是 | 是 2026-11-11 07:31 |
| 11/09 BB Asia Trade | 261109 Bloomberg｜The Asia Trade | https://www.bilibili.com/video/BV1aVWb68Edc | 是 | 是 2026-11-10 08:58 |
| 11/09 BB China Show | 261109 Bloomberg｜The China Show | https://www.bilibili.com/video/BV1nXWx6KEKb | 是 | 是 2026-11-10 09:08 |
| 11/09 BB 采访：米塔尔与印度电信崛起 | 261109 Bloomberg 采访：米塔尔与印度电信崛起 | https://www.bilibili.com/video/BV1UhWW6kEtN | 是 | 是 2026-11-10 09:11 |
| 11/09 BB CNN News -Anderson Cooper 360 | 261109 CNN News -Anderson Cooper 360 | https://www.bilibili.com/video/BV1uJWb6xEYC | 是 | 是 2026-11-10 09:15 |
| 11/09 BB Weekend | 261109 Bloomberg Weekend | https://www.bilibili.com/video/BV1AKWx6BEKp | 是 | 是 2026-11-09 09:40 |
| 11/08 BB Wall Street Week | 261107 Bloomberg Wall street week | https://www.bilibili.com/video/BV18jW76ZE4f | 是 | 是 2026-11-09 09:44 |
| 11/08 BB Money | 261107 Bloomberg Money | https://www.bilibili.com/video/BV1TPWE6dEKV | 是 | 是 2026-11-09 08:59 |
| 11/08 BB Weekend | 261107 Bloomberg Weekend | https://www.bilibili.com/video/BV1oK446TEW9 | 是 | 是 2026-11-09 08:59 |
| 11/08 BB CNN News -Anderson Cooper 360 | 261108 CNN News -Anderson Cooper 360 | https://www.bilibili.com/video/BV18jW76ZEBv | 是 | 是 2026-11-09 09:44 |
| 11/08 BB CNN News -The Source | 261108 CNN News -The Source | https://www.bilibili.com/video/BV1AQ4x6mEgg | 是 | 是 2026-11-10 08:49 |
| 11/07 BB Balance of Power | 261107 Bloomberg Balance of Power | https://www.bilibili.com/video/BV1sv4764Ewu | 是 | 是 2026-11-10 09:20 |
| 11/07 BB Surveillance | 261107 Bloomberg Surveillance | https://www.bilibili.com/video/BV1gE4L68EFE | 是 | 是 2026-11-09 08:38（存档名为旧标题，脚本核对不到） |
| 11/07 BB Tech | 261107 Bloomberg Tech | https://www.bilibili.com/video/BV1Pv4764EAS | 是 | 是 2026-11-09 08:59（存档名为旧标题，脚本核对不到） |

<!-- Surveillance / Tech 是 11/9 早上手工发的，当时 SMTP 确认成功，但 assets/kd 用的是
     旧标题（Bloomberg 261107 …），verify_videos.py 的判定看不到，所以手工补录在此。 -->

### 其他有用路径

- 原始转录：`assets/video/bili/<BV>/text/transcript.txt`
- 摘要：`assets/video/bili/<BV>/text/summary.md`
- Kindle 存档：`assets/kd/<Kindle 标题>.md`（旧文件是 `Bloomberg <上传标题>.md`）
- 本次抓取：`tmp/bili_uploads.json`；页面抽取 JS：`tmp/extract_uploads.js`
- 聊天记录：`tmp/web/chats/chat_*.json`
- 标题缓存：`tmp/bili_titles.json`
