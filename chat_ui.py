#!/usr/bin/python
# -*- coding: utf-8 -*-
"""ChatGPT 风格聊天界面（customtkinter）+ Helios-One 流式生成

- 深色 ChatGPT 风格界面：用户气泡靠右绿色，AI 回复带头像靠左
- 模型逐字生成，UI 以波动节奏"吐字"：高速连吐 + 偶尔停顿 + 首字稍慢
- 每次回复长度与速度在随机范围内波动，尽量像真人 LLM 对话
"""

import glob
import os
import queue
import random
import sys
import threading
import time

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#343541"        # 聊天区背景
HEADER = "#202123"    # 顶栏
PANEL = "#40414F"     # 输入框
TEXT = "#ECECF1"      # 正文
MUTED = "#8E8EA0"     # 次要文字
ACCENT = "#10a37f"    # ChatGPT 绿
FONT = ("Microsoft YaHei UI", 13)


class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI 助手 · Helios-One 字符级大模型版")
        self.geometry("900x700")
        self.minsize(640, 480)
        self.configure(fg_color=BG)

        self.model = None
        self.data = None
        self.device = "cpu"

        self.msg_queue = queue.Queue()
        self.busy = False
        self.current_label = None
        self.current_text = ""

        self._build_ui()
        # 后台加载模型（torch 导入较慢，避免阻塞窗口显示）
        threading.Thread(target=self._load_model, daemon=True).start()
        self.after(5, self._poll)

    # ---------------- UI ----------------
    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color=HEADER, corner_radius=0)
        header.pack(fill="x")
        ctk.CTkLabel(header, text="AI 助手",
                     font=("Microsoft YaHei UI", 16, "bold"),
                     text_color=TEXT).pack(side="left", padx=14, pady=10)
        self.status = ctk.CTkLabel(header, text="正在加载模型…",
                                   font=("Microsoft YaHei UI", 12),
                                   text_color=MUTED)
        self.status.pack(side="right", padx=14)
        ctk.CTkButton(header, text="清空", width=56, height=28,
                      fg_color="transparent", hover_color="#2A2B32",
                      text_color=MUTED, border_width=0,
                      command=self.clear_chat).pack(side="right", padx=(0, 8))

        self.scroll = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0)
        self.scroll.pack(fill="both", expand=True)

        self._assistant_bubble("你好，我是你的 AI 助手。随便说点什么，我会假装认真回答。")

        bottom = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        bottom.pack(fill="x", padx=12, pady=(0, 12))
        self.entry = ctk.CTkEntry(bottom, placeholder_text="给 AI 助手发消息…",
                                  fg_color=PANEL, border_width=0,
                                  text_color=TEXT, height=40, font=FONT)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self.send())
        self.send_btn = ctk.CTkButton(bottom, text="发送", width=80, height=40,
                                      fg_color=ACCENT, hover_color="#0E8C68",
                                      text_color="white", command=self.send,
                                      font=("Microsoft YaHei UI", 13, "bold"))
        self.send_btn.pack(side="left", padx=(8, 0))

    def _user_bubble(self, text):
        row = ctk.CTkFrame(self.scroll, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=5)
        bubble = ctk.CTkFrame(row, fg_color=ACCENT, corner_radius=14)
        bubble.pack(side="right", anchor="e")
        ctk.CTkLabel(bubble, text=text, text_color="white",
                     wraplength=520, font=FONT, justify="left").pack(padx=12, pady=8)
        self._scroll_bottom()

    def _assistant_bubble(self, text=None):
        row = ctk.CTkFrame(self.scroll, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=5)
        avatar = ctk.CTkFrame(row, width=34, height=34,
                              corner_radius=17, fg_color=ACCENT)
        avatar.pack(side="left", padx=(4, 8))
        avatar.pack_propagate(False)
        ctk.CTkLabel(avatar, text="AI", text_color="white",
                     font=("Microsoft YaHei UI", 11, "bold")).place(relx=0.5, rely=0.5, anchor="center")
        body = ctk.CTkFrame(row, fg_color="transparent")
        body.pack(side="left", fill="x", expand=True)
        label = ctk.CTkLabel(body, text=text or "", text_color=TEXT,
                             wraplength=560, font=FONT, justify="left", anchor="w")
        label.pack(fill="x")
        self._scroll_bottom()
        return label

    def _scroll_bottom(self):
        self.scroll._parent_canvas.yview_moveto(1.0)

    # ---------------- 交互 ----------------
    def send(self):
        if self.busy:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._user_bubble(text)
        self.busy = True
        self.send_btn.configure(state="disabled")
        threading.Thread(target=self._assistant_worker, args=(text,), daemon=True).start()

    def clear_chat(self):
        for w in self.scroll.winfo_children():
            w.destroy()
        self._assistant_bubble("你好，我是你的 AI 助手。随便说点什么，我会假装认真回答。")

    # ---------------- 模型 ----------------
    def _load_model(self):
        try:
            import torch
            from generate_text import load_checkpoint
            from train_text import DataGenerator, Model

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            ckpts = sorted(glob.glob(os.path.join("logs", "*_model.ckpt")),
                           key=os.path.getmtime)
            if not ckpts:
                raise RuntimeError("未找到 checkpoint，请先运行 train_text.py")
            ckpt = load_checkpoint(ckpts[-1])

            class _A:
                seq_length = 1
                batch_size = 1
            data = DataGenerator(ckpt["data_file"], _A())
            model = Model(data.vocab_size, state_size=ckpt["state_size"],
                          num_layers=ckpt["num_layers"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device).eval()

            self.data, self.model = data, model
            n = sum(p.numel() for p in model.parameters())
            self.status.configure(text="模型就绪 · {:.1f}M 参数 · {} · 极速流式".format(
                n / 1e6, self.device.upper()))
        except Exception as e:
            self.status.configure(text="模型加载失败: {}".format(e))

    def _assistant_worker(self, prompt):
        self.msg_queue.put(("thinking",))
        time.sleep(random.uniform(0.2, 0.8))  # 极速模型，思考也快
        try:
            import torch
            import torch.nn.functional as F
            if self.model is None:
                raise RuntimeError("模型未加载完成")

            state = None
            prime = ''.join(c for c in prompt if c in self.data.char2id_dict)
            if not prime:
                prime = self.data.id2char(0)
            # 内容多一些：长度随机 300~1200 字，偶尔更长
            length = random.randint(300, 1200)
            if random.random() < 0.05:
                length = random.randint(1500, 2200)
            temp = random.uniform(0.7, 0.95)

            buf = []
            with torch.no_grad():
                for w in prime:
                    x = torch.tensor([[self.data.char2id(w)]],
                                     dtype=torch.long, device=self.device)
                    _, state = self.model(x, state)
                word = prime[-1]
                self.msg_queue.put(("start",))
                for _ in range(length):
                    x = torch.tensor([[self.data.char2id(word)]],
                                     dtype=torch.long, device=self.device)
                    logits, state = self.model(x, state)
                    probs = F.softmax(logits[0, 0] / temp, dim=0)
                    word = self.data.id2char(torch.multinomial(probs, 1).item())
                    buf.append(word)
                    # 按随机块大小吐出（20~60 字），块间偶尔停顿 → 速度波动
                    if len(buf) >= random.randint(20, 60) or (buf and buf[-1] in "。！？\n"):
                        self.msg_queue.put(("chunk", "".join(buf)))
                        buf = []
                        if random.random() < 0.12:
                            self.msg_queue.put(("pause", random.randint(20, 100)))
                        elif random.random() < 0.03:
                            self.msg_queue.put(("pause", random.randint(120, 250)))
            if buf:
                self.msg_queue.put(("chunk", "".join(buf)))
        except Exception as e:
            self.msg_queue.put(("chunk", "\n\n[出错] {}".format(e)))
        finally:
            self.msg_queue.put(("done",))

    # ---------------- 消息泵 ----------------
    def _poll(self):
        try:
            item = self.msg_queue.get_nowait()
        except queue.Empty:
            delay = 10
        else:
            kind = item[0]
            if kind == "thinking":
                self.current_label = self._assistant_bubble("正在输入…")
                delay = 10
            elif kind == "start":
                self.current_text = ""
                self.current_label.configure(text="")
                delay = 10
            elif kind == "chunk":
                self.current_text += item[1]
                self.current_label.configure(text=self.current_text)
                delay = 10
            elif kind == "pause":
                delay = item[1]
            elif kind == "done":
                self.busy = False
                self.send_btn.configure(state="normal")
                self.current_label = None
                delay = 10
            self._scroll_bottom()
        self.after(delay, self._poll)


if __name__ == "__main__":
    ChatApp().mainloop()
