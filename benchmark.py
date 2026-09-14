#!/usr/bin/python
# -*- coding: utf-8 -*-
"""用 LLM 测试思路给 Helios-One 字符级大语言模型做基准评测

指标：
- TTFT：首字延迟（预热 prime 到生成第一个字符）
- token/s：逐字采样吞吐
- 逐 token 延迟分布：p50 / p95 / p99
- max context：训练上下文长度 + 实际可生成长度（状态持续累积）
- 峰值显存 / 参数量 / checkpoint 大小
"""

import argparse
import glob
import os
import sys
import time

import torch

from generate_text import load_checkpoint
from train_text import DataGenerator, Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--prime', default='你好，我是你的AI助手')
    parser.add_argument('--num', type=int, default=2000, help='吞吐测试生成字符数')
    parser.add_argument('--runs', type=int, default=5, help='TTFT 重复次数')
    parser.add_argument('--log-dir', default='./logs')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.checkpoint is None:
        ckpts = sorted(glob.glob(os.path.join(args.log_dir, '*_model.ckpt')),
                       key=os.path.getmtime)
        if not ckpts:
            sys.exit('未找到 checkpoint')
        args.checkpoint = ckpts[-1]
    print('Checkpoint: {} ({:.1f} MB)'.format(
        args.checkpoint, os.path.getsize(args.checkpoint) / 1e6))

    ckpt = load_checkpoint(args.checkpoint)

    class _Args:
        seq_length = 1
        batch_size = 1
    data = DataGenerator(ckpt['data_file'], _Args())
    model = Model(data.vocab_size,
                  state_size=ckpt['state_size'],
                  num_layers=ckpt['num_layers'])
    model.load_state_dict(ckpt['model_state_dict'])
    device = args.device
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print('Model: vocab={} state={} layers={} params={:,}'.format(
        data.vocab_size, ckpt['state_size'], ckpt['num_layers'], n_params))

    prime = ''.join(c for c in args.prime if c in data.char2id_dict)
    print('Prime: "{}" ({} chars)'.format(prime, len(prime)))

    def run_state(p):
        # 返回 (last_state)
        state = None
        with torch.no_grad():
            for w in p:
                x = torch.tensor([[data.char2id(w)]], dtype=torch.long, device=device)
                _, state = model(x, state)
        return state

    # ---------- TTFT：预热 + 首个生成字符 ----------
    torch.cuda.reset_peak_memory_stats(device)
    ttfts = []
    for _ in range(args.runs):
        state = None
        t0 = time.perf_counter()
        with torch.no_grad():
            for w in prime:
                x = torch.tensor([[data.char2id(w)]], dtype=torch.long, device=device)
                _, state = model(x, state)
            x = torch.tensor([[data.char2id(prime[-1])]], dtype=torch.long, device=device)
            logits, state = model(x, state)
            _word = data.id2char(logits[0, 0].argmax().item())
        ttfts.append((time.perf_counter() - t0) * 1000)
    ttfts.sort()
    print('\nTTFT (首字延迟, 含 {} 字预热): mean={:.2f}ms  p50={:.2f}ms  min={:.2f}ms  max={:.2f}ms'.format(
        len(prime), sum(ttfts) / len(ttfts), ttfts[len(ttfts) // 2], ttfts[0], ttfts[-1]))

    # ---------- 吞吐：纯计算（不打印） ----------
    state = run_state(prime)
    with torch.no_grad():
        t0 = time.perf_counter()
        lat = []
        word = prime[-1]
        for _ in range(args.num):
            x = torch.tensor([[data.char2id(word)]], dtype=torch.long, device=device)
            s = time.perf_counter()
            logits, state = model(x, state)
            lat.append((time.perf_counter() - s) * 1000)
            word = data.id2char(logits[0, 0].argmax().item())
        elapsed = time.perf_counter() - t0
    lat.sort()
    n = len(lat)
    print('吞吐(纯计算): {:.0f} chars/s, 每字 {:.2f}ms'.format(args.num / elapsed, elapsed / args.num * 1000))
    print('逐字延迟: p50={:.2f}ms  p95={:.2f}ms  p99={:.2f}ms  min={:.2f}ms  max={:.2f}ms'.format(
        lat[n // 2], lat[int(n * 0.95)], lat[int(n * 0.99)], lat[0], lat[-1]))

    # ---------- 吞吐：真实使用（逐字打印 + flush） ----------
    state = run_state(prime)
    with torch.no_grad():
        t0 = time.perf_counter()
        word = prime[-1]
        for _ in range(200):
            x = torch.tensor([[data.char2id(word)]], dtype=torch.long, device=device)
            logits, state = model(x, state)
            word = data.id2char(logits[0, 0].argmax().item())
            print(word, end='')
            sys.stdout.flush()
        elapsed = time.perf_counter() - t0
    print('\n吞吐(含打印+flush): {:.0f} chars/s'.format(200 / elapsed))

    # ---------- 长序列稳定性：生成 10000 字观察内存是否增长 ----------
    state = run_state(prime)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        t0 = time.perf_counter()
        word = prime[-1]
        for _ in range(10000):
            x = torch.tensor([[data.char2id(word)]], dtype=torch.long, device=device)
            logits, state = model(x, state)
            word = data.id2char(logits[0, 0].argmax().item())
        long_elapsed = time.perf_counter() - t0
    print('\n长序列: 10000 字生成耗时 {:.1f}s ({:.0f} chars/s)，状态持续累积未截断'.format(
        long_elapsed, 10000 / long_elapsed))

    mem = torch.cuda.max_memory_allocated(device) / 1e6
    print('峰值显存: {:.1f} MB'.format(mem))
    print('训练上下文窗口(seq_length): 20 字符 | 推理上下文: 无限(全状态累积)')


if __name__ == '__main__':
    main()
