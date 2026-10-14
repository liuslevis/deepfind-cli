# Physical Intelligence π 系列模型概览

> 更新日期：2026-09-23  
> 范围：π0、π0.5、π\*0.6、π0.7。这里的“微调数据”统一包括 supervised post-training、任务专用微调以及 RL / 自主交互数据。公开资料没有给出的数据量标为“未披露”，不作推测。

## TL;DR

π 系列的演进主线不是单纯扩大模型，而是逐步解决机器人基础模型的四个瓶颈：

1. **π0：会做**——用跨机器人、大规模示范数据和 flow matching，把预训练 VLM 变成能输出高频连续动作的通用 VLA。
2. **π0.5：会泛化和规划**——把机器人数据、Web 多模态数据、高层子任务标注和语言指导统一共训；同一个模型先生成语义子任务，再生成低层动作，因此能在没见过的新家完成长时程任务。
3. **π\*0.6：会靠实践变熟练**——在更强的 π0.6 上引入 Recap：示范学习打底，人工纠错处理策略实际遇到的错误，再用自主执行经验和奖励优化可靠性与吞吐量。
4. **π0.7：会组合、可操控**——通过语言子任务、质量/速度/错误元数据、视觉子目标和控制模式等丰富条件，把人类视频、失败轨迹、自主数据和 RL 专家经验统一吸收进单个通用模型，出现初步的组合泛化，并达到任务专用专家的水平。

## 对比表

| 模型 | 训练数据（基础训练 / 预训练） | 微调、后训练与强化学习数据 | 模型结构 | 模型特性 / 一句话结论 |
|---|---|---|---|---|
| **π0** | 从 Internet-scale 图文预训练的 **PaliGemma 3B** 初始化；机器人混合数据超过 **10,000 小时**。内部数据约 **903M timesteps**、**68 个复杂任务**、**7 种机器人配置**（博客按另一统计口径称 8 个机器人），另混入 OXE / Bridge v2 / DROID 等开源数据；OXE 涉及 22 种机器人。训练 mixture 中开源数据采样权重约 9.1%。 | 对具体下游任务使用高质量、策略一致的示范数据做 supervised post-training；简单任务约 **5 小时**，最复杂任务可达 **100+ 小时**。示例包括洗衣折叠、收餐桌和纸箱组装。 | **3.3B VLA**：3B PaliGemma VLM + 从零训练的约 300M action expert。图像、语言和本体状态作为输入；action expert 用 **conditional flow matching** 一次预测 **50-step action chunk**，10 次去噪，可支持最高约 50 Hz 的连续控制。整体类似双专家 Transformer：VLM 处理图文，action expert 处理状态与动作。 | 首次建立 π 系列的核心范式：**跨 embodiment 预训练 + 高质量任务后训练 + 连续动作生成**。优势是灵巧、多机器人通用；不足是复杂任务仍常依赖任务专用微调，长时程语义规划较弱。 |
| **π0.5** | 在 π0 基础上做异构 co-training。数据包括：约 **400 小时、约 100 个家庭环境**的移动双臂机器人数据（MM）；多家庭中的固定单/双臂数据（ME）；实验室跨 embodiment 数据与扩展 OXE（CE）；人工标注的高层子任务和 bounding box（HL）；Web 图像描述、VQA、目标定位数据（WD）。第一阶段 **97.6%** 的样本并非“移动机器人做家务”的直接目标域数据。 | 两阶段训练：先用 FAST 把动作离散化，做 **280k steps** 自回归预训练；再做 **80k steps** post-training，引入随机初始化的 flow-matching action expert。后训练使用筛选后的成功且较短的 MM + ME 轨迹、相关 HL、WD，以及人类逐步发出语义子任务的 verbal instruction（VI）；不再使用实验室 CE 数据。 | 沿用约 **3B PaliGemma + 300M action expert**，但统一支持离散文本/FAST action token 与连续 flow-matching 动作。推理是层级式的：同一模型先自回归生成高层语义子任务，再以该子任务为条件，经 10 次去噪输出 **50-step / 约 1 秒**的连续动作块。 | 核心突破是 **开放世界环境泛化**：可在未参与训练的新家中，仅凭高层指令完成 10–15 分钟的清理、整理等任务。Web 数据主要补语义和新物体识别，跨机器人/多环境数据主要补物理技能泛化。 |
| **π\*0.6** | 基座 **π0.6** 大体继承 π0.5 的数据配方：内部与外部跨 embodiment 数据、多家庭移动/固定机器人数据、高层子任务预测，以及含 bounding box、keypoint 等任务的 Web 多模态数据；总规模未披露。π\*0.6 首先把普通 imitation 预训练替换为 **Recap 的 offline RL 预训练**。 | 每个目标任务先用人工示范微调，再收集真实机器人运行数据：① 专家在人机运行中接管并提供纠错；② 策略自主执行的成功、失败及奖励反馈。Recap 训练 value function 做 credit assignment，以 value change（advantage）标注好/坏动作，并训练 advantage-conditioned VLA；目标任务包括咖啡制作、混合衣物折叠和纸箱组装。具体数据量未披露。 | π0.6 是约 **5B 级**层级式 VLA：Gemma 3 4B VLM（含约 400M 视觉编码器）+ 约 860M action expert；同时预测 FAST 离散动作与 flow-matching 连续动作。采用 **Knowledge Insulation**，action expert 的梯度不回传 VLM；支持提示元数据。π\*0.6 在该策略上增加 **value / advantage 条件化的 Recap 训练**，而不是另起一种 VLA 主干。 | 从“模仿人类”转向 **示范 + 纠错 + 自主练习**。重点不是更广的零样本泛化，而是任务专用的稳定性、速度与恢复能力：困难任务吞吐量可翻倍，失败率可降低 2 倍以上，咖啡任务成功率超过 90%，可连续运行数小时。 |
| **π0.7** | 数据范围进一步扩大：多平台、多环境的高/低质量示范；大量策略评测产生的自主轨迹；人工干预；开源机器人数据；第一视角人类视频；Web 目标定位、属性预测、VQA、纯文本和视频语言任务。特别纳入失败、有明显错误的成功轨迹，以及 π\*0.6 RL 训练/评测经验，并用元数据区分质量和策略。总小时数、episode 数和各类比例未披露。用于通用性评测的自主数据明确从训练中排除。 | 主模型以 **out-of-the-box、无任务专用 post-training** 为目标；把 RL 专家的自主数据蒸馏回通用模型。面对全新任务，可由人类只用逐步语言进行 coaching，再用这些 coaching episodes **仅微调同架构的高层语言策略**，由其自动生成子任务；不需要额外低层遥操作示范。视觉子目标 world model 另用高质量分段机器人/人类视频，以及开源图像编辑、视频数据训练。 | **约 5B VLA**：Gemma 3 4B VLM（含约 400M vision encoder）+ 860M flow-matching action expert，并加入 **MEM-style 视频历史编码器**。最多输入 4 路相机、每路 6 帧历史和 3 张视觉子目标；输出 50-step action chunk，通常 5 次去噪。提示可组合任务/子任务语言、速度/质量/错误 metadata、joint/EE 控制模式和视觉子目标。视觉子目标由独立的、基于 **BAGEL 14B** 的轻量 world model 生成。 | 关键能力是 **可操控的组合泛化**：单个通用模型能吸收不同质量和策略的数据，开箱表现可匹配或超过 π\*0.6 任务专家；能组合旧技能操作训练中未示范的新电器，并把折衣技能零样本迁移到没有该任务数据的新机器人 embodiment。 |

## 关键演进

| 维度 | π0 → π0.5 | π0.5 → π\*0.6 | π\*0.6 → π0.7 |
|---|---|---|---|
| **数据观** | 从“多机器人示范”扩展为“机器人 + Web + 高层语义 + 语言指导”的异构课程。 | 从离线人工示范扩展为策略自己产生的分布内错误、人工纠错和带奖励的自主经验。 | 不再只保留优质轨迹；用速度、质量、错误等 context 显式解歧，从而安全利用失败和混合质量数据。 |
| **训练目标** | 从单纯低层动作生成，扩展为高层文本推理与低层连续控制联合学习。 | 从行为克隆扩展为 offline / real-world RL，优化成功率和吞吐量。 | 把多个 RL 专家的经验蒸馏回一个通用策略，主要依靠 prompting 而不是每任务微调。 |
| **结构重点** | 离散 FAST 预训练 + flow matching 后训练；同一模型分层推理。 | 更大 Gemma 3 主干、Knowledge Insulation、元数据条件化，再叠加 advantage conditioning。 | 在 π0.6 上增加时序记忆、视觉子目标和丰富多模态 prompt；另配 world model 生成子目标。 |
| **主要能力** | 新环境长时程泛化。 | 已知高难任务的可靠、快速、持续运行。 | 新任务、新组合、新 embodiment 的开箱泛化，并保持专家级熟练度。 |

## 阅读结论与注意事项

- **“预训练”在几代模型里的含义不同。** π0 主要指多机器人行为预训练；π0.5 将 FAST 离散动作、Web 任务和高层语义统一进自回归预训练；π\*0.6 的第一阶段改为 offline RL；π0.7 则强调在丰富 context 下吸收混合质量数据。
- **π\*0.6 是 specialist 路线，π0.7 是把 specialist 能力重新汇总到 generalist。** 两者不是简单的版本替代：前者证明真实机器人 RL 能显著优化任务表现，后者证明这些经验可以被单一通用模型蒸馏和条件化调用。
- **视觉子目标不等于 π0.7 主模型自己“想象”。** 子目标图由独立的 BAGEL-based world model 生成，再作为 π0.7 的条件输入；不用视觉子目标时，π0.7 也可仅靠语言和 metadata 运行。
- **“没见过任务”需谨慎理解。** π0.7 的组合泛化通常意味着没有该“任务 × 场景 × embodiment”的直接示范，但组成技能、相近物体、Web 语义或其他机器人上的相关经验可能存在。例如论文声称的是 UR5e 双臂系统没有折衣训练数据，而不是整个训练集完全没有折衣数据。
- **后续版本的完整数据规模没有公开。** π0.5 只明确披露目标域移动数据约 400 小时；π0.6 / π\*0.6 和 π0.7 没有公开总小时数或完整 mixture 比例，因此不能据此做严格的数据规模横向比较。

## 来源

### π0

- [博客：π0](https://www.pi.website/blog/pi0)
- [论文：π0: A Vision-Language-Action Flow Model for General Robot Control](https://www.pi.website/download/pi0.pdf)

### π0.5

- [博客：π0.5](https://www.pi.website/blog/pi05)
- [论文：π0.5: A Vision-Language-Action Model with Open-World Generalization](https://www.pi.website/download/pi05.pdf)

### π0.6 / π\*0.6

- [博客：π\*0.6 与 Recap](https://www.pi.website/blog/pistar06)
- [π0.6 Model Card](https://website.pi-asset.com/pi06star/PI06_model_card.pdf)

### π0.7

- [博客：π0.7](https://www.pi.website/blog/pi07)
- [论文：π0.7: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities](https://www.pi.website/download/pi07.pdf)
