# LeRobot Diffusion Policy 模型架构

> 源码仓库：[huggingface/lerobot](https://github.com/huggingface/lerobot)  
> 主要实现：[modeling_diffusion.py](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/modeling_diffusion.py)  
> 配置定义：[configuration_diffusion.py](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/configuration_diffusion.py)  
> 本文基于上述提交中的 LeRobot 实现整理。图中的维度采用默认配置：`n_obs_steps=2`、`horizon=64`、`n_action_steps=32`、`down_dims=(512, 1024, 2048)`。

## 1. 整体架构

```mermaid
flowchart LR
    subgraph OBS["观测窗口：最近 n_obs_steps 帧"]
        S["机器人状态<br/>(B, S, state_dim)"]
        I["多相机 RGB 图像<br/>(B, S, N, C, H, W)"]
        E["环境状态（可选）<br/>(B, S, env_dim)"]
    end

    subgraph VE["视觉编码器（每个相机独立或共享）"]
        PRE["Resize / Crop（可选）"]
        RES["ResNet backbone<br/>默认 ResNet-18，去掉 avgpool/fc"]
        SS["Spatial Softmax<br/>默认 32 个关键点"]
        FC["Flatten → Linear → ReLU<br/>每相机输出 64 维"]
        PRE --> RES --> SS --> FC
    end

    I --> PRE
    S --> CAT
    FC --> CAT
    E --> CAT
    CAT["按特征拼接，再展平时间维<br/>global_cond: (B, S × cond_dim)"]

    T["扩散时间步 t"] --> PE["Sinusoidal Embedding<br/>→ MLP → 128 维"]
    CAT --> COND
    PE --> COND
    COND["拼接条件<br/>FiLM condition"]

    XT["带噪动作轨迹 x_t<br/>(B, horizon, action_dim)"] --> UNET["Conditional 1D U-Net"]
    COND -. "注入每个残差块" .-> UNET
    UNET --> PRED["预测 ε 或干净动作 x₀<br/>(B, horizon, action_dim)"]
```

全局条件的构成为：

```text
global_cond =
    flatten_time(
        concat(
            robot_state,
            image_feature_camera_1,
            ...,
            image_feature_camera_N,
            optional_environment_state
        )
    )
```

若每个相机的视觉特征维度为 `64`，则 U-Net 收到的观测条件维度为：

```text
global_cond_dim =
    n_obs_steps × (state_dim + num_cameras × 64 + optional_env_dim)

FiLM_cond_dim = diffusion_step_embed_dim + global_cond_dim
```

对应源码：

- [`DiffusionModel.__init__`](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/modeling_diffusion.py#L191-L232)：计算条件维度，创建 RGB Encoder、U-Net 和噪声调度器。
- [`_prepare_global_conditioning`](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/modeling_diffusion.py#L270-L306)：编码多相机图像，拼接状态并展平观测时间维。
- [`DiffusionRgbEncoder`](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/modeling_diffusion.py#L474-L560)：图像预处理、ResNet、Spatial Softmax 和输出投影。

## 2. Conditional 1D U-Net

动作序列的时间轴被当作 1D 卷积的空间轴。输入先从 `(B, T, action_dim)` 转换为 `(B, action_dim, T)`，经过 U-Net 后再转换回来。

```mermaid
flowchart TB
    X["x_t<br/>(B, action_dim, 64)"]

    D1["Down Block 1<br/>2 × Conditional ResBlock<br/>action_dim → 512<br/>长度 64"]
    DS1["Conv1d k=3, stride=2<br/>长度 64 → 32"]

    D2["Down Block 2<br/>2 × Conditional ResBlock<br/>512 → 1024<br/>长度 32"]
    DS2["Conv1d k=3, stride=2<br/>长度 32 → 16"]

    D3["Down Block 3<br/>2 × Conditional ResBlock<br/>1024 → 2048<br/>长度 16"]

    M["Middle<br/>2 × Conditional ResBlock<br/>2048 通道，长度 16"]

    C1["Concat Skip：2048 + 2048"]
    U1["Up Block 1<br/>2 × Conditional ResBlock<br/>4096 → 1024"]
    US1["ConvTranspose1d k=4, stride=2<br/>长度 16 → 32"]

    C2["Concat Skip：1024 + 1024"]
    U2["Up Block 2<br/>2 × Conditional ResBlock<br/>2048 → 512"]
    US2["ConvTranspose1d k=4, stride=2<br/>长度 32 → 64"]

    OUT["Conv1d Block 512 → 512<br/>1×1 Conv：512 → action_dim"]
    Y["ε̂ 或 x̂₀<br/>(B, 64, action_dim)"]

    X --> D1 --> DS1 --> D2 --> DS2 --> D3 --> M --> C1 --> U1 --> US1 --> C2 --> U2 --> US2 --> OUT --> Y
    D3 -. "skip" .-> C1
    D2 -. "skip" .-> C2

    COND["同一全局条件<br/>[time_embedding ; observations]"]
    COND -. "FiLM" .-> D1
    COND -. "FiLM" .-> D2
    COND -. "FiLM" .-> D3
    COND -. "FiLM" .-> M
    COND -. "FiLM" .-> U1
    COND -. "FiLM" .-> U2
```

默认配置下，`horizon=64`，三层 `down_dims` 对应两个实际降采样操作，时间长度变化为 `64 → 32 → 16`，随后恢复为 `16 → 32 → 64`。代码会要求 `horizon` 能被 `2 ** len(down_dims)` 整除，约束比实际两次 stride-2 降采样更严格。

### Conditional Residual Block

```mermaid
flowchart LR
    X["输入 x"] --> C1["Conv1d → GroupNorm → Mish"]
    C1 --> FILM["FiLM 调制"]
    COND["全局条件"] --> CE["Mish → Linear"]
    CE --> SCALE["拆分 scale 与 bias<br/>默认开启 scale modulation"]
    SCALE --> FILM
    FILM --> C2["Conv1d → GroupNorm → Mish"]
    X --> RC["Identity 或 1×1 Conv"]
    C2 --> ADD["残差相加"]
    RC --> ADD
    ADD --> Y["输出"]
```

默认 FiLM 形式为：

```text
h = ConvBlock1(x)
(scale, bias) = Linear(Mish(condition))
h = scale * h + bias
output = ConvBlock2(h) + ResidualProjection(x)
```

注意这里源码直接使用 `scale * h + bias`，没有写成常见的 `(1 + scale) * h + bias`。

对应源码：

- [`DiffusionConditionalUnet1d`](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/modeling_diffusion.py#L628-L766)
- [`DiffusionConditionalResidualBlock1d`](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/modeling_diffusion.py#L768-L830)

## 3. 视觉编码器

```mermaid
flowchart LR
    IMG["RGB 图像<br/>(B, C, H, W)"]
    IMG --> R["Resize（可选）"]
    R --> C["训练：RandomCrop<br/>推理：CenterCrop<br/>均为可选"]
    C --> RN["ResNet-18 卷积主干"]
    RN --> FM["二维特征图<br/>(B, channels, h, w)"]
    FM --> KP["1×1 Conv<br/>映射为 32 张关键点热图"]
    KP --> SM["每张热图执行 spatial softmax"]
    SM --> XY["期望坐标 (x, y)<br/>(B, 32, 2)"]
    XY --> F["Flatten → Linear(64,64) → ReLU"]
    F --> V["图像特征<br/>(B, 64)"]
```

`SpatialSoftmax` 不输出普通的全局池化向量，而是把每张特征热图转为归一化图像坐标中的期望位置，因此 32 个关键点产生 `32 × 2 = 64` 维特征。

## 4. 训练流程

```mermaid
flowchart TB
    OBS["n_obs_steps 帧观测"] --> ENC["构造 global_cond"]
    A0["真实动作轨迹 x₀<br/>(B, horizon, action_dim)"]
    EPS["采样高斯噪声 ε"]
    STEP["为 batch 中每个样本随机采样 t"]
    A0 --> ADD["Noise Scheduler.add_noise"]
    EPS --> ADD
    STEP --> ADD
    ADD --> XT["x_t"]
    XT --> U["Conditional 1D U-Net"]
    ENC --> U
    STEP --> U
    U --> P["模型预测"]
    EPS --> TARGET["target"]
    A0 --> TARGET
    P --> LOSS["MSE Loss"]
    TARGET --> LOSS
```

目标由 `prediction_type` 决定：

| `prediction_type` | U-Net 学习目标 |
|---|---|
| `epsilon`（默认） | 加到动作轨迹上的噪声 `ε` |
| `sample` | 原始动作轨迹 `x₀` |

如果启用 `do_mask_loss_for_padding`，复制填充的动作位置不会参与损失；默认关闭，以保持与原始 Diffusion Policy 实现一致。

训练核心见 [`DiffusionModel.compute_loss`](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/modeling_diffusion.py#L335-L400)。

## 5. 推理与动作执行

```mermaid
flowchart TB
    Q["维护最近 n_obs_steps 帧观测队列"]
    Q --> COND["编码 global_cond"]
    N["初始化高斯噪声动作轨迹<br/>(B, horizon, action_dim)"]
    N --> LOOP["对 scheduler 的每个时间步 t"]
    COND --> LOOP
    LOOP --> UNET["U-Net 预测 ε̂ 或 x̂₀"]
    UNET --> SCH["DDPM / DDIM scheduler.step<br/>x_t → x_(t-1)"]
    SCH --> CHECK{"还有时间步？"}
    CHECK -- "是" --> LOOP
    CHECK -- "否" --> FULL["完整去噪动作轨迹<br/>horizon 步"]
    FULL --> SLICE["截取 [n_obs_steps-1 :<br/>n_obs_steps-1+n_action_steps]"]
    SLICE --> AQ["动作队列：默认 32 步"]
    AQ --> EXEC["逐步输出动作；队列耗尽后重新规划"]
```

默认情况下：

```text
观测使用区间：动作轨迹索引 0..1       （n_obs_steps = 2）
执行动作区间：动作轨迹索引 1..32      （n_action_steps = 32）
预测总区间：  动作轨迹索引 0..63      （horizon = 64）
```

也就是说，模型虽然生成 64 步动作，但每轮只缓存并执行从“当前时刻”开始的 32 步。动作缓存耗尽后，才使用最新观测窗口再次运行完整扩散采样。

对应源码：

- [`DiffusionPolicy.select_action`](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/modeling_diffusion.py#L125-L161)：维护观测/动作队列。
- [`DiffusionModel.conditional_sample`](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/modeling_diffusion.py#L234-L268)：从高斯噪声开始迭代去噪。
- [`DiffusionModel.generate_actions`](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/diffusion/modeling_diffusion.py#L308-L333)：从完整轨迹中截取实际执行的动作块。

## 6. 默认关键配置

| 模块 | 默认值 | 含义 |
|---|---:|---|
| `n_obs_steps` | `2` | 条件观测窗口长度 |
| `horizon` | `64` | 一次扩散生成的动作轨迹长度 |
| `n_action_steps` | `32` | 每次规划后缓存执行的动作数 |
| `vision_backbone` | `resnet18` | RGB 图像主干网络 |
| `spatial_softmax_num_keypoints` | `32` | 每个相机提取的空间关键点数 |
| `use_separate_rgb_encoder_per_camera` | `True` | 每个相机使用独立视觉编码器 |
| `down_dims` | `(512, 1024, 2048)` | 1D U-Net 各下采样阶段通道数 |
| `kernel_size` | `5` | U-Net 残差块卷积核大小 |
| `n_groups` | `8` | GroupNorm 分组数 |
| `diffusion_step_embed_dim` | `128` | 扩散时间步嵌入维度 |
| `use_film_scale_modulation` | `True` | FiLM 同时产生 scale 和 bias |
| `noise_scheduler_type` | `DDPM` | 默认噪声调度器，也支持 DDIM |
| `num_train_timesteps` | `100` | 训练前向扩散步数 |
| `beta_schedule` | `squaredcos_cap_v2` | beta 调度方式 |
| `prediction_type` | `epsilon` | 默认预测噪声 |
| `num_inference_steps` | `None` | 未设置时等于训练扩散步数，即 100 |

## 7. 一句话总结

LeRobot 的 Diffusion Policy 是一个**以最近多帧状态和视觉关键点为全局条件、通过 FiLM 调制的 1D U-Net**：训练时对动作轨迹加噪并学习去噪目标，推理时从随机动作序列出发，经 DDPM/DDIM 多步反向扩散生成整段动作，再截取当前时刻开始的一段动作执行。
