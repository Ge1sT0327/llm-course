# 下载指南 - 智能知识库Agent 项目

## 硬件需求
- RTX 3070 8GB ✅ (可微调Qwen3.7-1.7B, 4-bit量化)
- 硬盘空间: ~20GB
- 内存: 16GB+

---

## 1. 模型下载 (HuggingFace)

### 方式一: 直接下载 (需要科学上网)
```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 基础模型 - 用于5.1微调
huggingface-cli download Qwen/Qwen3.7-1.7B-Instruct --local-dir ./models/Qwen3.7-1.7B-Instruct

# Embedding模型 - 用于3.1/3.3向量检索
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ./models/bge-small-zh-v1.5
```

### 方式二: 国内镜像 (推荐, 速度快10倍)
```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 基础模型 (~3GB)
huggingface-cli download Qwen/Qwen3.7-1.7B-Instruct --local-dir ./models/Qwen3.7-1.7B-Instruct

# Embedding模型 (~380MB)
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ./models/bge-small-zh-v1.5
```

### 方式三: ModelScope (国内, 无需代理)
```bash
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3.7-1.7B-Instruct', cache_dir='./models')
snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='./models')
"
```

---

## 2. 数据集下载

### 微调数据集 - Alpaca-zh (中文指令数据)
```bash
# 从GitHub下载
git clone https://github.com/hikariming/alpaca_chinese_dataset.git ./data/alpaca-zh

# 或者直接用Python下载
python -c "
from datasets import load_dataset
ds = load_dataset('silk-road/alpaca-data-gpt4-chinese', split='train')
ds.select(range(1000)).to_json('./data/alpaca_zh_1k.json', force_ascii=False)
"
```

### 知识库文档 (用于RAG)
- 自备: 技术文档、产品手册、FAQ等 50-100篇
- 或使用维基百科中文dump (可选)
- 我们会生成样例文档用于实验

---

## 3. 工具安装

### Ollama (本地模型部署)
- 下载: https://ollama.com/download/windows
- 安装后拉取模型:
```bash
ollama pull qwen3:1.7b
```

### Docker Desktop (可选, 用于5.2容器化部署)
- 下载: https://www.docker.com/products/docker-desktop/

---

## 4. Python依赖

```bash
# 核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# LLM & Agent框架
pip install openai langchain langgraph langchain-openai

# RAG & 向量检索
pip install faiss-cpu sentence-transformers chromadb rank-bm25

# 微调
pip install transformers peft datasets accelerate bitsandbytes

# 部署
pip install fastapi uvicorn

# 工具
pip install python-dotenv pillow numpy pandas tqdm
```

---

## 5. 下载清单汇总

| # | 项目 | 大小 | 用途 | 优先级 |
|---|------|------|------|--------|
| 1 | Qwen3.7-1.7B-Instruct | ~3GB | 5.1微调基座 | ⭐⭐⭐ 必需 |
| 2 | bge-small-zh-v1.5 | ~380MB | 3.1/3.3向量化 | ⭐⭐⭐ 必需 |
| 3 | Alpaca-zh (1K条) | ~5MB | 5.1微调数据 | ⭐⭐ 实验3前下载 |
| 4 | Python依赖 | ~2GB | 全部实验 | ⭐⭐⭐ 必需 |
| 5 | Ollama | ~500MB | 5.2本地部署 | ⭐ 实验5前下载 |
| 6 | 知识库文档 | 自备 | 3.2/3.3 RAG | ⭐ 实验3前准备 |

---

## 6. 快速开始 (一次性全部下载)

```bash
# 1. 安装Python依赖
pip install torch openai langchain langgraph langchain-openai faiss-cpu sentence-transformers chromadb rank-bm25 transformers peft datasets accelerate bitsandbytes fastapi uvicorn python-dotenv pillow numpy pandas tqdm huggingface_hub

# 2. 下载模型 (国内镜像)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3.7-1.7B-Instruct --local-dir ./models/Qwen3.7-1.7B-Instruct
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ./models/bge-small-zh-v1.5

# 3. 下载数据集
python -c "
from datasets import load_dataset
ds = load_dataset('silk-road/alpaca-data-gpt4-chinese', split='train')
ds.select(range(1000)).to_json('./data/alpaca_zh_1k.json', force_ascii=False)
print('Dataset downloaded: data/alpaca_zh_1k.json')
"

# 4. 验证
python -c "
from config import verify_config; verify_config()
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print(f'Embedding dim: {m.get_sentence_embedding_dimension()}')
print('All set!')
"
```
