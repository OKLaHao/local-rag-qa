# Ollama 使用简介

Ollama 是一个在自己电脑上运行大模型的工具，免费，数据不出本机。

安装 Ollama 之后，先拉一个模型，例如：
ollama pull qwen2.5:1.5b

拉完之后可以直接在终端里和模型对话：
ollama run qwen2.5:1.5b

Ollama 还提供本地 HTTP 接口，默认地址是 http://localhost:11434。
程序可以通过这个接口调用模型，例如生成接口是 /api/generate，
向量化接口是 /api/embeddings。
