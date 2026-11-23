#!/usr/bin/env python3
"""本地 RAG 问答的公共库：切块、向量化、检索、问答。

两个后端：
- Ollama 后端（默认）：向量用 nomic-embed-text，问答用 qwen2.5:1.5b，
  全部跑在自己电脑上，不花钱、不联网。
- Mock 后端（--mock）：词袋向量 + 引用原文回答，只用来在没装 Ollama
  的电脑上跑通流程，不代表真实效果。
只用 Python 标准库 + numpy。
"""
import json
import math
import re
import urllib.request
import urllib.error

import numpy as np

CHUNK_SIZE = 400


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """按段落切块：段落拼到接近 chunk_size 就切一段；超长段落硬切。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    for para in paras:
        while len(para) > chunk_size:
            chunks.append(para[:chunk_size])
            para = para[chunk_size:]
        if len(current) + len(para) + 1 > chunk_size and current:
            chunks.append(current)
            current = para
        else:
            current = (current + "\n" + para) if current else para
    if current:
        chunks.append(current)
    return chunks


def cosine_top_k(query_vec, matrix, k=3):
    """余弦相似度取前 k 个，返回 [(下标, 相似度), ...]（相似度从高到低）。"""
    q = np.asarray(query_vec, dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []
    m = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(m, axis=1)
    sims = (m @ q) / (norms * q_norm + 1e-9)
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order if norms[i] > 0]


class OllamaEmbedder:
    """调本地 Ollama 的 /api/embeddings 接口做向量化。"""

    def __init__(self, model="nomic-embed-text", host="http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def embed(self, texts):
        vectors = []
        for text in texts:
            payload = json.dumps(
                {"model": self.model, "prompt": text}
            ).encode("utf-8")
            req = urllib.request.Request(
                self.host + "/api/embeddings", data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as e:
                raise RuntimeError(
                    "连不上 Ollama 服务。请先确认：1) 已安装 Ollama；"
                    "2) 已执行 ollama serve；3) 已执行 ollama pull "
                    f"{self.model}。原始错误：{e}"
                ) from e
            vectors.append(data["embedding"])
        return np.array(vectors, dtype=float)


class OllamaChat:
    """调本地 Ollama 的 /api/generate 接口做问答。"""

    def __init__(self, model="qwen2.5:1.5b", host="http://localhost:11434",
                 timeout=180):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def answer(self, question, contexts):
        context_text = "\n---\n".join(contexts)
        prompt = (
            "请只根据下面提供的资料回答问题，不要编造资料里没有的内容。\n\n"
            f"资料：\n{context_text}\n\n问题：{question}\n\n回答："
        )
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.host + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"连不上 Ollama 服务：{e}") from e
        return data.get("response", "")


def _tokenize(text):
    """简单分词：按中文字符和英文单词切，全部小写。"""
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


class MockEmbedder:
    """词袋向量（演示用）：先 fit 建词表，再把文本变成词频向量。
    能按关键词重合度做检索，足够在没装 Ollama 时验证流程。"""

    def __init__(self):
        self.vocab = []
        self.index = {}

    def fit(self, texts):
        words = set()
        for text in texts:
            words.update(_tokenize(text))
        self.vocab = sorted(words)
        self.index = {w: i for i, w in enumerate(self.vocab)}

    def embed(self, texts):
        rows = []
        for text in texts:
            vec = np.zeros(len(self.vocab), dtype=float)
            for word in _tokenize(text):
                if word in self.index:
                    vec[self.index[word]] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            rows.append(vec)
        return np.array(rows, dtype=float)

    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    @classmethod
    def load_vocab(cls, path):
        obj = cls()
        with open(path, "r", encoding="utf-8") as f:
            obj.vocab = json.load(f)
        obj.index = {w: i for i, w in enumerate(obj.vocab)}
        return obj


class MockChat:
    """演示用问答：直接引用最相关的一段原文，不生成新内容。"""

    def answer(self, question, contexts):
        if not contexts:
            return "资料里没有相关内容，无法回答。"
        text = contexts[0]
        if len(text) > 200:
            cut = text.rfind("。", 0, 200)
            text = text[:cut + 1] if cut > 0 else text[:200]
        return "根据资料，相关内容如下：\n" + text
