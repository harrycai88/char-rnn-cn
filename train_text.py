#!/usr/bin/python
# -*- coding: utf-8 -*-
"""char-rnn 中文胡言乱语生成器 - 训练脚本（PyTorch 重写）

忠实还原原 TensorFlow 1.x 版架构与语义：
- nn.Embedding(vocab, 100) -> 3 层 LSTM(100, 100) -> nn.Linear(100, vocab)
- batch 32 / seq 20 / lr 0.01（每 1000 步衰减 0.9）/ grad clip 5 / 100 epochs
- 每步零初始状态，整段序列求交叉熵损失
"""

import argparse
import os
import time

import torch
import torch.nn as nn


class HParam:
    batch_size = 32
    n_epoch = 100
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.9
    grad_clip = 5
    state_size = 100
    num_layers = 3
    seq_length = 20
    log_dir = './logs'


class DataGenerator:
    """读文本、建排序词表、pointer 式 next_batch（与原逻辑一致）"""

    def __init__(self, datafiles, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        with open(datafiles, encoding='utf-8') as f:
            self.data = f.read()

        self.total_len = len(self.data)
        self.words = list(set(self.data))
        self.words.sort()
        self.vocab_size = len(self.words)
        print('Vocabulary Size: ', self.vocab_size)

        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}

        self._pointer = 0

    def char2id(self, c):
        return self.char2id_dict[c]

    def id2char(self, id):
        return self.id2char_dict[id]

    def next_batch(self):
        x_batches = []
        y_batches = []
        for _ in range(self.batch_size):
            if self._pointer + self.seq_length + 1 >= self.total_len:
                self._pointer = 0
            bx = self.data[self._pointer: self._pointer + self.seq_length]
            by = self.data[self._pointer + 1: self._pointer + self.seq_length + 1]
            self._pointer += self.seq_length

            x_batches.append([self.char2id(c) for c in bx])
            y_batches.append([self.char2id(c) for c in by])

        return x_batches, y_batches


class Model(nn.Module):
    """3 层 LSTM char-RNN"""

    def __init__(self, vocab_size, state_size=100, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, state_size)
        self.lstm = nn.LSTM(state_size, state_size, num_layers=num_layers)
        self.fc = nn.Linear(state_size, vocab_size)

    def forward(self, x, state=None):
        # x: (seq, batch)
        emb = self.embedding(x)
        outputs, last_state = self.lstm(emb, state)
        logits = self.fc(outputs)  # (seq, batch, vocab)
        return logits, last_state


def train(data, model, args, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    os.makedirs(args.log_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.log_dir, '{}_model.ckpt'.format(os.path.splitext(os.path.basename(args.data))[0]))
    loss_csv = os.path.join(args.log_dir, 'train_loss.csv')

    max_iter = args.n_epoch * \
        (data.total_len // args.seq_length) // args.batch_size
    print('max_iter: ', max_iter)

    steps_done = 0
    with open(loss_csv, 'w', encoding='utf-8') as f:
        f.write('step,loss\n')

        for _ in range(max_iter):
            # 与原版一致：每步从零初始状态开始
            state = None
            lr = args.learning_rate * \
                (args.decay_rate ** (steps_done // args.decay_steps))
            for g in optimizer.param_groups:
                g['lr'] = lr

            x_batch, y_batch = data.next_batch()
            x = torch.tensor(x_batch, dtype=torch.long, device=device).t()  # (seq, batch)
            y = torch.tensor(y_batch, dtype=torch.long, device=device)  # (batch, seq)

            logits, _ = model(x, state)
            # logits: (seq, batch, vocab)，与 y(batch, seq) 对齐后再展平求损失
            loss = criterion(logits.permute(1, 0, 2).reshape(-1, data.vocab_size),
                             y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if steps_done % 10 == 0:
                print('Step:{}/{}, training_loss:{:4f}'.format(
                    steps_done, max_iter, loss.item()))
                f.write('{},{}\n'.format(steps_done, loss.item()))
            if steps_done % 2000 == 0 or (steps_done + 1) == max_iter:
                _save(ckpt_path, model, data, args)
            steps_done += 1


def _save(ckpt_path, model, data, args):
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': data.words,
        'data_file': args.data,
        'state_size': args.state_size,
        'num_layers': args.num_layers,
    }, ckpt_path)
    print('Checkpoint saved: {}'.format(ckpt_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='llm_speech.txt')
    parser.add_argument('--epochs', type=int, default=HParam.n_epoch)
    parser.add_argument('--log-dir', default=HParam.log_dir)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seq-length', type=int, default=HParam.seq_length)
    parser.add_argument('--batch-size', type=int, default=HParam.batch_size)
    parser.add_argument('--state-size', type=int, default=HParam.state_size)
    parser.add_argument('--layers', type=int, default=HParam.num_layers)
    args = parser.parse_args()
    args.n_epoch = args.epochs
    # 用 HParam 补齐其余超参默认值
    for k, v in vars(HParam).items():
        if not k.startswith('_') and not hasattr(args, k):
            setattr(args, k, v)

    start = time.time()
    data = DataGenerator(args.data, args)
    model = Model(data.vocab_size,
                  state_size=args.state_size,
                  num_layers=args.layers)
    train(data, model, args, args.device)
    print('Training finished in {:.1f}s'.format(time.time() - start))
