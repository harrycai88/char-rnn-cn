# char-rnn-cn

基于 char-rnn 的中文"胡言乱语"生成器，用 3 层 LSTM 逐字符学习文本风格并生成模仿文本。

- 语料 `llm_speech.txt`：现代大语言模型说话风格的多样文本，训练一个"假装自己是 AI"的胡言乱语生成器
- 保留 `JayLyrics.txt`（周杰伦歌词）、`hlm.txt`（红楼梦）等原始语料，可随时切换训练

原项目为 TensorFlow 1.x 实现（2016），本项目已用 **PyTorch (cu128)** 重写，架构与超参保持一致：`Embedding(vocab, 100)` → 3 层 `LSTM(100, 100)` → `Linear(100, vocab)`，batch 32 / seq 20 / lr 0.01（每 1000 步 ×0.9）/ grad clip 5。

## 环境

需要 GPU + CUDA 驱动，PyTorch 使用 cu128 版本：

```bash
uv sync
```

## 训练

```bash
uv run python train_text.py                     # 默认训练 llm_speech.txt，100 epochs
uv run python train_text.py --data JayLyrics.txt --epochs 200
uv run python train_text.py --device cuda --log-dir ./logs
```

checkpoint 保存至 `logs/{语料名}_model.ckpt`，训练 loss 曲线记录在 `logs/train_loss.csv`。

## 预测（生成文字）

```bash
uv run python generate_text.py --prime "你好"            # 默认取 logs/ 最新 checkpoint
uv run python generate_text.py --num 500 --temperature 0.8
```

- `--prime`：预热 LSTM 的初始文字（自动过滤到词表内）
- `--temperature`：采样温度，`0`（默认）为 argmax 贪心采样，调大增加随机度

实际生成示例（`--prime "好的" --temperature 0.85`）：

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
