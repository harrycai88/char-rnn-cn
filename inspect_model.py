#!/usr/bin/python
# -*- coding: utf-8 -*-
"""标准 OpenAI SDK 读取本地 Helios-One 的模型元数据

快速体检：`uv run python inspect_model.py` 输出模型 id / display_name / 架构 / 能力等元数据。

用法：先启动服务（uv run python openai_server.py），再运行本脚本：
    uv run python inspect_model.py
默认连 http://localhost:8000/v1，可用环境变量覆盖：
    OPENAI_BASE_URL / OPENAI_API_KEY
"""

import os

from openai import OpenAI

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "sk-local")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

print("== GET /v1/models ==")
resp = client.models.list()
for m in resp.data:
    print("id              :", m.id)
    print("object          :", m.object)
    print("created         :", m.created)
    print("owned_by        :", m.owned_by)
    # 服务端返回的扩展元数据字段
    for k, v in (getattr(m, "model_extra", None) or {}).items():
        print("{:<15}: {}".format(k, v))
    print("-" * 44)
