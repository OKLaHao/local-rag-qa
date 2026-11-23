#!/usr/bin/env python3
"""问答：根据问题检索最相关的文档片段，让本地模型基于资料回答。

用法：
    python3 ask.py --question "RAG 是哪三个步骤？"
    python3 ask.py --mock --question "怎么用 Ollama 拉一个模型？"
    python3 ask.py --question "..." --top-k 5
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rag import cosine_top_k, MockChat, MockEmbedder, OllamaChat, OllamaEmbedder


def main():
    parser = argparse.ArgumentParser(description="本地 RAG 问答")
    parser.add_argument("--question", required=True, help="要问的问题")
    parser.add_argument("--index", default="index", help="索引目录")
    parser.add_argument("--top-k", type=int, default=3, help="取几段相关资料")
    parser.add_argument("--mock", action="store_true",
                        help="用词袋向量 + 引用原文回答（不用装 Ollama）")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                        help="Ollama 向量化模型")
    parser.add_argument("--chat-model", default="qwen2.5:1.5b",
                        help="Ollama 问答模型")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                        help="Ollama 服务地址")
    args = parser.parse_args()

    index_dir = Path(args.index)
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    matrix = np.load(index_dir / "embeddings.npy")

    if args.mock:
        embedder = MockEmbedder.load_vocab(index_dir / "vocab.json")
        chat = MockChat()
    else:
        embedder = OllamaEmbedder(model=args.embed_model,
                                   host=args.ollama_host)
        chat = OllamaChat(model=args.chat_model, host=args.ollama_host)

    q_vec = embedder.embed([args.question])[0]
    hits = cosine_top_k(q_vec, matrix, k=args.top_k)
    print(f"问题：{args.question}\n")
    print("检索到的资料：")
    contexts = []
    for rank, (idx, score) in enumerate(hits, start=1):
        chunk = chunks[idx]
        contexts.append(chunk["text"])
        print(f"  [{rank}] {chunk['source']}（相似度 {score:.3f}）")
    print()
    answer = chat.answer(args.question, contexts)
    print("回答：")
    print(answer)


if __name__ == "__main__":
    main()
