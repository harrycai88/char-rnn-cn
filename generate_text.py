#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Helios-One 字符级大语言模型 - 生成脚本（PyTorch cu128 实现）"""

import argparse
import glob
import os
import sys

import torch
import torch.nn.functional as F

from train_text import DataGenerator, HParam, Model


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu')
    return ckpt


def generate(prime, data, model, args, device):
    model.to(device)
    model.eval()

    # 预热：用 prime 前面所有字符驱动 LSTM
    state = None
    with torch.no_grad():
        for word in prime[:-1]:
            x = torch.tensor([[data.char2id(word)]], dtype=torch.long, device=device)
            _, state = model(x, state)

        word = prime[-1]
        output = prime
        for _ in range(args.num):
            x = torch.tensor([[data.char2id(word)]], dtype=torch.long, device=device)
            logits, state = model(x, state)
            if args.temperature <= 0:
                # 与原版一致的 argmax 采样
                word = data.id2char(logits[0, 0].argmax().item())
            else:
                probs = F.softmax(logits[0, 0] / args.temperature, dim=0)
                word = data.id2char(torch.multinomial(probs, 1).item())
            print(word, end='')
            sys.stdout.flush()
            output += word
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None,
                        help='模型 checkpoint 路径，默认取 logs/ 下最新')
    parser.add_argument('--data', default=None,
                        help='语料文件，默认读取 checkpoint 中记录的语料')
    parser.add_argument('--prime', default='你好')
    parser.add_argument('--num', type=int, default=500)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--log-dir', default=HParam.log_dir)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.checkpoint is None:
        ckpts = sorted(glob.glob(os.path.join(args.log_dir, '*_model.ckpt')),
                       key=os.path.getmtime)
        if not ckpts:
            sys.exit('未找到 checkpoint，请先运行 train_text.py')
        args.checkpoint = ckpts[-1]
        print('Using checkpoint: {}'.format(args.checkpoint))

    ckpt = load_checkpoint(args.checkpoint)

    class _Args:
        seq_length = 1
        batch_size = 1
    data_args = _Args()
    data = DataGenerator(args.data or ckpt['data_file'], data_args)

    model = Model(data.vocab_size,
                  state_size=ckpt['state_size'],
                  num_layers=ckpt['num_layers'])
    model.load_state_dict(ckpt['model_state_dict'])

    # prime 自动过滤到词表内
    prime = ''.join(c for c in args.prime if c in data.char2id_dict)
    if not prime:
        prime = data.id2char(0)
    print('Prime: {}'.format(prime))

    generate(prime, data, model, args, args.device)
    print()
