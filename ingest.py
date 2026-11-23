#!/usr/bin/env python3
"""建索引：读 docs/ 里的 .md/.txt 文档，切块、向量化，存到 index/。

用法：
    python3 ingest.py                          # 用 Ollama 向量化
    python3 ingest.py --mock                   # 用词袋向量（不用装 Ollama）
    python3 ingest.py --docs my_docs --index my_index
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rag import chunk_text, MockEmbedder, OllamaEmbedder


def load_docs(docs_dir):
    """读入文档目录，返回 [(文件名, 全文), ...]。"""
    docs = []
    for path in sorted(Path(docs_dir).glob("*")):
        if path.suffix.lower() in (".md", ".txt") and path.is_file():
            docs.append((path.name, path.read_text(encoding="utf-8")))
    if not docs:
        raise SystemExit(f"在 {docs_dir} 里没找到 .md 或 .txt 文档")
    return docs


def main():
    parser = argparse.ArgumentParser(description="RAG 建索引")
    parser.add_argument("--docs", default="docs", help="文档目录")
    parser.add_argument("--index", default="index", help="索引输出目录")
    parser.add_argument("--mock", action="store_true",
                        help="用词袋向量代替 Ollama（不用装 Ollama）")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                        help="Ollama 向量化模型")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                        help="Ollama 服务地址")
    args = parser.parse_args()

    docs = load_docs(args.docs)
    chunks = []
    for filename, text in docs:
        for chunk in chunk_text(text):
            chunks.append({"source": filename, "text": chunk})
    print(f"文档 {len(docs)} 个，切成 {len(chunks)} 块")

    if args.mock:
        embedder = MockEmbedder()
        embedder.fit([c["text"] for c in chunks])
        vectors = embedder.embed([c["text"] for c in chunks])
    else:
        embedder = OllamaEmbedder(model=args.embed_model,
                                   host=args.ollama_host)
        vectors = embedder.embed([c["text"] for c in chunks])
    print(f"向量化完成：{vectors.shape[0]} 条，维度 {vectors.shape[1]}")

    out = Path(args.index)
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out / "embeddings.npy", vectors)
    if args.mock:
        embedder.save_vocab(out / "vocab.json")
    print(f"索引已保存到 {out}/（chunks.json、embeddings.npy"
          + ("、vocab.json" if args.mock else "") + ")")


if __name__ == "__main__":
    main()
