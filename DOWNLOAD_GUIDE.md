# 环境配置与模型下载指南

## 硬件需求

| 实验范围 | 最低配置 | 推荐配置 |
|---------|---------|---------|
| 1.4-2.3 (API实验) | 任意电脑, 无需GPU | CPU + 8GB RAM |
| 3.x (RAG实验) | CPU + 8GB RAM | 16GB RAM |
| 4.1-4.3 (Agent实验) | CPU + 8GB RAM | - |
| 5.1 (LoRA微调) | GPU 8GB+ VRAM | RTX 3070/4090, 24GB |
| 5.2 (本地部署) | GPU 4GB+ | 8GB+ VRAM |
| 6.x (工程化) | CPU + 8GB RAM | - |

## 方式一：本地环境

### Python依赖

```bash
# 核心依赖 (所有实验)
pip install openai python-dotenv

# RAG实验 (3.x)
pip install faiss-cpu sentence-transformers chromadb rank-bm25

# Agent实验 (4.x)
pip install langchain langgraph langchain-openai

# 微调实验 (5.1, 需要GPU)
pip install torch transformers peft datasets accelerate bitsandbytes

# 一键安装全部
pip install openai python-dotenv langchain langgraph langchain-openai \
  faiss-cpu sentence-transformers chromadb rank-bm25 \
  transformers peft datasets accelerate bitsandbytes
```

> **注意**: 如果 `sentence-transformers` 和 `transformers` 版本冲突, 使用:
> ```bash
> pip install transformers==4.51.0 sentence-transformers -i https://pypi.org/simple
> ```

### 模型下载

**方式A: HuggingFace 直接下载**
```bash
pip install huggingface_hub

# BGE Embedding 模型 (380MB, RAG实验必需)
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ./models/bge-small-zh-v1.5

# Qwen3 微调基座 (3GB, 实验5.1需要)
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ./models/Qwen3-1.7B-Instruct
```

**方式B: 国内镜像加速**
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ./models/bge-small-zh-v1.5
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ./models/Qwen3-1.7B-Instruct
```

**方式C: ModelScope (国内用户推荐)**
```bash
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='./models')
snapshot_download('Qwen/Qwen3-1.7B', cache_dir='./models')
"
```

### API Key 配置

```bash
cp .env.example .env
# 编辑 .env 文件, 至少填入一个 API Key
# 推荐 DeepSeek (免费注册, 成本最低): https://platform.deepseek.com
```

**.env 文件示例**:
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
ZHIPU_API_KEY=xxxxxxxxxxxxxxxx
```

### 验证安装

```bash
python -c "
from config import verify_config; verify_config()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print(f'BGE OK: dim={model.get_sentence_embedding_dimension()}')
import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
"
```

## 方式二：云GPU平台 (推荐用于微调实验)

### AutoDL 配置

```bash
# 1. 克隆项目
git clone https://github.com/<your-repo>/llm-course.git
cd llm-course

# 2. 安装依赖 (AutoDL 镜像可能缺少部分包)
pip install openai python-dotenv -i https://pypi.org/simple
pip install faiss-cpu sentence-transformers -i https://pypi.org/simple
pip install peft datasets -i https://pypi.org/simple

# 3. 升级 PyTorch (AutoDL 预装版本可能过旧)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -U
pip install transformers==4.51.0 -i https://pypi.org/simple

# 4. 配置 API Key
echo "DEEPSEEK_API_KEY=sk-你的key" > .env

# 5. 下载模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ./models/bge-small-zh-v1.5
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ./models/Qwen3-1.7B-Instruct

# 6. 如果模型已缓存, 可开启离线模式加速
export HF_HUB_OFFLINE=1
```

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `sentence-transformers` 安装失败 | PyTorch 版本过旧 (<2.4) | `pip install torch -U` |
| `transformers` 不支持 `qwen3` 架构 | transformers 版本过旧 | `pip install transformers>=4.51.0` |
| `faiss-cpu` 找不到 | 镜像源不包含 | `pip install faiss-cpu -i https://pypi.org/simple` |
| BGE 模型下载慢 | 默认从 huggingface.co 下载 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| 模型已下载但实验仍慢 | 每次加载都尝试联网验证 | 设置 `HF_HUB_OFFLINE=1` |
| RTX 3070 8GB 微调OOM | 模型+优化器超出显存 | 使用 `device_map="auto"` + `torch_dtype=torch.float16` |

## 模型清单

| 模型 | 大小 | 用途 | 实验 |
|------|------|------|------|
| BAAI/bge-small-zh-v1.5 | 380MB | 文本向量化 | 3.1, 3.3, 6.3 |
| Qwen/Qwen3-1.7B | 3GB | LoRA微调基座 | 5.1 |
| DeepSeek V4 API | - | LLM推理 (云端) | 全部 |
| Qwen3.7-Max API | - | LLM推理 (云端) | 可选 |

## Ollama 本地部署 (可选, 实验5.2)

```bash
# 安装: https://ollama.com/download
# 拉取模型:
ollama pull qwen3:1.7b

# 部署自定义微调模型:
ollama create kb-agent-v1 -f Modelfile
ollama serve
```
