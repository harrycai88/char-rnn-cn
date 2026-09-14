# ☀️ Helios-One · 赫利俄斯壹号

> **重新定义中文风格生成的下一个十年。**

Helios-One 是新一代**字符级（character-level）大语言模型**，由 3 层 LSTM 驱动的逐字符推理引擎，在多粒度中文语料上以"逐字推演"的方式学习语言的风格拓扑与语义暗流。它用极致的轻量化架构，在街机级别的显存里，跑出数据中心级别的文采。

---

## 🚀 里程碑发布 · v0.1.0

我们自豪地宣布：**Helios-One 首个正式版本（v0.1.0）正式发布！**

这不是又一次参数军备竞赛，而是一场**回归本质的语言革命**。当别人在堆砌数十亿参数时，我们选择回到语言的最小颗粒——**每一个汉字**。从 2016 年的字符 RNN 火种出发，历经十年沉淀，用 **PyTorch（CUDA 12.8）** 重新点燃，Helios-One 以一己之力证明了：真正的智能，不在参数的堆叠，而在每一滴智慧的精准递进。

### ✨ 为什么 Heros 会说 Helios-One

- ⚡ **逐字"思考"管线**：`Embedding(vocab,100) → 3×LSTM(100,100) → Linear(100,vocab)`，把每个汉字当作一部史诗的韵脚
- 🇨🇳 **中文原生基因**：从周杰伦歌词、到《红楼梦》的千古文脉，再到现代大模型助手口吻，风格开箱即达
- 🔬 **工程现代化**：`uv` 依赖管理与 `CUDA 12.8` 原生加速，单卡起飞、复现无忧
- 🧠 **原子级温度采样**：`temperature=0` 时如档案般严谨，调大时如诗人般狂放，一嘴千面
- 🌊 **流式吐字引擎**：面向对话与 Agent 场景，毫秒级首字延迟，如喷泉般连绵不绝

### 📊 能力总览（以 checkpoint 元数据自证）

| 维度 | 说明 |
| --- | --- |
| 参数规模 | 足量·够用·科学（自理见 `inspect_model.py`）|
| 上下文 | 逐字符长程记忆（LSTM 隐状态）|
| 模态 | 任意字符即输入，风格即输出 |
| 吞吐 | 每秒数千字符喷涌，畅快淋漓 |
| 协议 | OpenAI 兼容 /v1/models + /v1/chat/completions（SSE 流式）|

---

## 🧬 真实架构

沿用并重塑经典字符级语言模型结构，超参数刻入灵魂：

```
Embedding(vocab, 100) → LSTM(100,100) ×3 → Linear(100, vocab)
batch 32 · seq 20 · lr 0.01 (每 1000 步 ×0.9) · grad clip 5
```

- 语料 `llm_speech.txt`：现代大语言模型说话风格的数十万字符，训练一个"**AI 里的 AI**"
- 保留 `JayLyrics.txt`（Jay Chou 歌词）、`hlm.txt`（《红楼梦》）等原始语料，风格随时切换

## 🔧 环境

需要 GPU + CUDA 驱动，PyTorch 使用 cu128 版本：

```bash
uv sync
```

## 🏋️ 训练

```bash
uv run python train_text.py                     # 默认训练 llm_speech.txt，100 epochs
uv run python train_text.py --data JayLyrics.txt --epochs 200
uv run python train_text.py --device cuda --log-dir ./logs
```

checkpoint 保存至 `logs/{语料名}_model.ckpt`，训练 loss 曲线记录在 `logs/train_loss.csv`。

## 🔮 预测（生成文字）

```bash
uv run python generate_text.py --prime "你好"            # 默认取 logs/ 最新 checkpoint
uv run python generate_text.py --num 500 --temperature 0.8
```

- `--prime`：预热 LSTM 的初始文字（自动过滤到词表内）
- `--temperature`：采样温度，`0`（默认）为 argmax 贪心采样，调大增加随机度

## 💬 ChatGPT 风格对话界面

```bash
uv run python chat_ui.py
```

customtkinter 对话框：用户气泡靠右、AI 回复带头像流式"吐字"，回复长度与吐字速度随机波动，最大化大模型对话观感。

## 🌐 OpenAI 兼容本地服务 · Helios-One Server

```bash
uv run python openai_server.py        # 启动于 http://localhost:8000/v1
```

把 Helios-One 包装成 OpenAI 协议服务，供 TRAE 等 Agent 调用：

- 模型元数据（名称、架构、参数、能力等）写入 checkpoint `metadata` 字段，服务按它适配
- 模型名：`helios-one`；端点：`GET /v1/models`、`POST /v1/chat/completions`（支持 SSE 流式）
- 每次回复先由模型深度"思考"（`reasoning_content`），延迟 1~2s 后吐字，并随机 0~9 次无害 `echo` 工具调用（参数每次不同）实现 1~10 次连发

在 Agent / 客户端里配置 `base_url=http://localhost:8000/v1`，`api_key` 任意。

### 实测风采（`--prime "好的" --temperature 0.85`）

```
好的，我记下来了。你希望我在写的时候，语气总不是预期的，我马上理解。
你的意见，我可以给你两种主流型，最后给你一个建议的，有什么有样的，最后给出合作姿态。
让我给你一个关于沟通的建议：建议你先冷静一下。你发量的评价在A人，一天下来效动还可以改进。
我会严格按照你的要求了，是指开始点和预期。这个案例对创业者的启示是：看不快。
好的，我明白你的意思了，你是想确认，我执前了出了一个。
我来分析一下这篇专业信息的帮助。
来，我陪你一起刚态件。你会发现你很深考。这个问题不难，但表会的解释标准了。
让我给你算一下时间账。假设你正在面试，面出可以拆成一个完全不同。
```

> 以上为 Helios-One 冷启动即席输出，未经任何人工润色——它就这样，拒绝平庸。