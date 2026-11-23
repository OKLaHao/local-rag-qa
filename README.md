# 本地 RAG 问答

给一堆文档建索引，然后问问题，让本地大模型只根据资料回答。
全部跑在自己电脑上，不花钱，资料不上传。

## 它是怎么工作的

```
文档（docs/） → 切块 → 向量化 → 索引（index/）
问题 → 向量化 → 找最相关的 3 段 → 模型基于资料回答
```

三个核心概念：

- **切块**：把长文档切成一段一段（默认每段不超过 400 字）。
- **向量化**：把每段文字变成一串数字（向量），意思接近的段落，向量也接近。
- **检索**：问题也变成向量，算它和每段的余弦相似度，取最高的几段给模型当参考。

## 快速开始

需要 Python 3.9 以上，和一个 numpy：

```bash
pip install -r requirements.txt
```

### 不用装模型也能跑（演示模式）

```bash
# 1. 建索引（词袋向量，演示用）
python3 ingest.py --mock

# 2. 提问
python3 ask.py --mock --question "RAG 是哪三个步骤？"
python3 ask.py --mock --question "怎么用 Ollama 拉一个模型？"
python3 ask.py --mock --question "向量检索用什么方法算相似度？"
```

演示模式下，回答是直接引用最相关的一段原文，不生成新内容，
用来验证检索流程对不对。

### 用本地真实模型跑

1. 安装 Ollama：https://ollama.com
2. 拉两个模型（免费，跑在自己电脑上）：
   ```bash
   ollama pull qwen2.5:1.5b
   ollama pull nomic-embed-text
   ```
3. 建索引、提问：
   ```bash
   python3 ingest.py
   python3 ask.py --question "RAG 是哪三个步骤？"
   ```

## 换成自己的文档

把 `.md` 或 `.txt` 文档放进 `docs/` 目录，重新跑一遍：

```bash
python3 ingest.py --mock        # 或不加 --mock，用真实模型
python3 ask.py --mock --question "你的问题"
```

## 参数

### ingest.py（建索引）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--docs` | docs | 文档目录 |
| `--index` | index | 索引输出目录 |
| `--mock` | 关闭 | 用词袋向量代替 Ollama |
| `--embed-model` | nomic-embed-text | Ollama 向量化模型 |
| `--ollama-host` | http://localhost:11434 | Ollama 服务地址 |

### ask.py（问答）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--question` | 必填 | 要问的问题 |
| `--index` | index | 索引目录 |
| `--top-k` | 3 | 取几段相关资料 |
| `--mock` | 关闭 | 用词袋向量 + 引用原文回答 |
| `--embed-model` | nomic-embed-text | Ollama 向量化模型 |
| `--chat-model` | qwen2.5:1.5b | Ollama 问答模型 |
| `--ollama-host` | http://localhost:11434 | Ollama 服务地址 |

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `rag.py` | 公共库：切块、余弦检索、Ollama/Mock 的向量与问答客户端 |
| `ingest.py` | 建索引：读文档、切块、向量化、保存索引 |
| `ask.py` | 问答：检索相关片段、调用模型回答 |
| `docs/` | 示例文档：RAG 简介、Ollama 简介、向量数据库简介 |
| `index/` | 建索引后生成的目录（chunks.json、embeddings.npy），不进 Git |
