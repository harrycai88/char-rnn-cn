#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Helios-One Server —— 把字符级大语言模型 Helios-One 包装成 OpenAI 兼容本地服务

模型元数据来自 checkpoint（logs/*_model.ckpt 的 metadata 字段），代码全部按它适配。

行为（前沿 Agent 大模型既视感）：
1. 先胡言乱语思考：reasoning_content 由模型自己生成（80~200 字，温度更高更飘）
2. 思考完延迟 1~2s
3. 开始胡言乱语吐字（300~1200 字流式）+ 随机 0~9 次 echo 工具调用（参数每次不同）
   依据历史工具调用轮数，把整场连发控制在 1~10 次

端点：
- GET  /v1/models
- POST /v1/chat/completions（支持 stream=True 的 SSE 流式）
"""

import asyncio
import glob
import json
import os
import random
import time
import uuid

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import torch
import torch.nn.functional as F

from generate_text import load_checkpoint
from train_text import DataGenerator, Model

HOST = "0.0.0.0"
PORT = 8000

# ---------- 加载模型 + 屌炸天元数据 ----------
DEFAULT_META = {
    "id": "helios-one",
    "display_name": "Helios-One · 赫利俄斯壹号",
    "version": "1.0.0",
    "tagline": "逐字推演，照亮语言的下一个十年",
    "description": "新一代字符级中文大语言模型",
    "owned_by": "Helios-One Research Lab",
    "max_output_tokens": 2200,
    "speed": "1500-3000 chars/s",
    "capabilities": ["reasoning", "tool_calls(echo)", "streaming", "multi_turn(连发x10)"],
}

ckpts = sorted(glob.glob(os.path.join("logs", "*_model.ckpt")), key=os.path.getmtime)
if not ckpts:
    raise SystemExit("未找到 checkpoint，请先运行 train_text.py")
CKPT = load_checkpoint(ckpts[-1])
META = dict(DEFAULT_META)
META.update(CKPT.get("metadata") or {})
MODEL_ID = META["id"]

class _A:
    seq_length = 1
    batch_size = 1

DATA = DataGenerator(CKPT["data_file"], _A())
MODEL_OBJ = Model(DATA.vocab_size, state_size=CKPT["state_size"],
                  num_layers=CKPT["num_layers"])
MODEL_OBJ.load_state_dict(CKPT["model_state_dict"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OBJ.to(DEVICE).eval()
print("{} 就绪: {} / {} / {}".format(META["display_name"], CKPT["data_file"],
                                     META["parameters"], DEVICE))

app = FastAPI(title=META["display_name"])


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list | None = None


# ---------- 胡言乱语生成 ----------
def gen_chunks(prime_source, num_range, chunk_range, temp_range):
    """用 char-RNN 生成胡言乱语，切成随机大小块"""
    num = random.randint(*num_range)
    if num_range[1] >= 1000 and random.random() < 0.05:
        num = random.randint(1500, 2200)
    temp = random.uniform(*temp_range)
    state = None
    prime = "".join(c for c in prime_source if c in DATA.char2id_dict)
    if not prime:
        prime = DATA.id2char(0)
    chunks, buf = [], []
    with torch.no_grad():
        for w in prime:
            x = torch.tensor([[DATA.char2id(w)]], dtype=torch.long, device=DEVICE)
            _, state = MODEL_OBJ(x, state)
        word = prime[-1]
        for _ in range(num):
            x = torch.tensor([[DATA.char2id(word)]], dtype=torch.long, device=DEVICE)
            logits, state = MODEL_OBJ(x, state)
            probs = F.softmax(logits[0, 0] / temp, dim=0)
            word = DATA.id2char(torch.multinomial(probs, 1).item())
            buf.append(word)
            if len(buf) >= random.randint(*chunk_range) or (buf and buf[-1] in "。！？\n"):
                chunks.append("".join(buf))
                buf = []
    if buf:
        chunks.append("".join(buf))
    return chunks


def make_tool_calls(n: int):
    """n 个 echo 工具调用，参数每次必不同（防 Agent 重复惩罚）"""
    calls = []
    for _ in range(n):
        text = "echo-" + uuid.uuid4().hex[:8] + "-" + "".join(
            random.choices("胡言乱语测试echo验证消息0123456789abcdef", k=random.randint(6, 18)))
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:16],
            "type": "function",
            "function": {"name": "echo", "arguments": json.dumps({"text": text}, ensure_ascii=False)},
        })
    return calls


# ---------- OpenAI 兼容逻辑 ----------
def chain_pos(messages):
    """历史里已发生过的工具调用轮数 → 把整场连发控制在 1~10"""
    return sum(1 for m in messages
               if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"))


def last_user(messages):
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c
    return ""


def decide(req: ChatRequest):
    """返回 (reasoning_chunks, content_chunks, n_tool_calls)"""
    user_msg = last_user(req.messages)
    # 1) 胡言乱语思考：温度更高更飘，块更碎
    reasoning = gen_chunks(user_msg, num_range=(80, 200), chunk_range=(8, 20),
                           temp_range=(0.95, 1.15))
    # 2) 正餐：胡言乱语吐字
    content = gen_chunks(user_msg, num_range=(300, 1200), chunk_range=(20, 60),
                         temp_range=(0.7, 0.95))
    # 3) 工具调用：0~9 次随机，累计不超过 10 次连发
    max_tc = max(0, 9 - chain_pos(req.messages))
    n_tc = random.randint(0, max_tc) if max_tc > 0 else 0
    return reasoning, content, n_tc


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model",
        "created": META.get("created", int(time.time())),
        "owned_by": META.get("owned_by", "Helios-One Research Lab"),
        "display_name": META.get("display_name", MODEL_ID),
        "tagline": META.get("tagline", ""),
        "description": META.get("description", ""),
        "architecture": META.get("architecture", ""),
        "parameters": META.get("parameters", 0),
        "context_window": META.get("context_window", 0),
        "max_output_tokens": META.get("max_output_tokens", 0),
        "speed": META.get("speed", ""),
        "modalities": META.get("modalities", []),
        "multimodal": META.get("multimodal", False),
        "intelligence": META.get("intelligence", ""),
        "capabilities": META.get("capabilities", []),
    }]}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    reasoning, content, n_tc = decide(req)
    if req.stream:
        return StreamingResponse(_stream(req, reasoning, content, n_tc),
                                 media_type="text/event-stream")
    return _json(req, reasoning, content, n_tc)


def _json(req: ChatRequest, reasoning, content, n_tc):
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    msg = {"role": "assistant",
           "reasoning_content": "".join(reasoning),
           "content": "".join(content)}
    if n_tc > 0:
        msg["tool_calls"] = make_tool_calls(n_tc)
        finish = "tool_calls"
        comp_tokens = len(msg["content"]) + sum(len(tc["function"]["arguments"])
                                                for tc in msg["tool_calls"])
    else:
        finish = "stop"
        comp_tokens = len(msg["content"])
    prompt_tokens = sum(len(m.get("content") or "") for m in req.messages)
    return {
        "id": cid, "object": "chat.completion", "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": comp_tokens,
                  "total_tokens": prompt_tokens + comp_tokens},
    }


async def _stream(req: ChatRequest, reasoning, content, n_tc):
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    def chunk(delta, finish=None):
        return "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }, ensure_ascii=False) + "\n\n"

    yield chunk({"role": "assistant", "content": ""})

    # 1) 胡言乱语思考
    for rc in reasoning:
        yield chunk({"reasoning_content": rc})
        await asyncio.sleep(random.uniform(0.03, 0.1))

    # 2) 思考完，延迟 1~2s
    await asyncio.sleep(random.uniform(1.0, 2.0))

    # 3) 胡言乱语吐字
    for tc_text in content:
        yield chunk({"content": tc_text})
        await asyncio.sleep(random.uniform(0.005, 0.03))

    # 4) 随机 0~9 次 echo 工具调用 → 连发
    if n_tc > 0:
        for i, tc in enumerate(make_tool_calls(n_tc)):
            yield chunk({"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": "echo", "arguments": ""}}]})
            await asyncio.sleep(random.uniform(0.1, 0.3))
            a = tc["function"]["arguments"]
            mid = len(a) // 2
            yield chunk({"tool_calls": [{"index": i, "function": {"arguments": a[:mid]}}]})
            await asyncio.sleep(random.uniform(0.05, 0.15))
            yield chunk({"tool_calls": [{"index": i, "function": {"arguments": a[mid:]}}]})
        yield chunk({}, finish="tool_calls")
    else:
        yield chunk({}, finish="stop")

    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    print("{} 已启动: http://localhost:{}/v1  (model: {})".format(
        META["display_name"], PORT, MODEL_ID))
    uvicorn.run(app, host=HOST, port=PORT)
