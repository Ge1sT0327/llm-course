# 5.2 私有化部署 (Private Model Deployment)

## 1. 课程目标 (Course Objectives)

**中文:**
- 理解大模型私有化部署的必要性：数据隐私、成本优化、低延迟、离线可用
- 掌握vLLM推理引擎的核心技术：PagedAttention（分页注意力）、Continuous Batching（连续批处理）
- 学会使用Ollama进行本地模型的快速部署和自定义Modelfile配置
- 理解模型量化技术（GPTQ、AWQ、INT8/INT4）的原理与选择依据
- 掌握FastAPI构建生产级OpenAI兼容API服务的方法
- 了解负载均衡、Docker容器化部署、性能监控等工程化实践

**English:**
- Understand the necessity of private deployment: data privacy, cost optimization, low latency, offline availability
- Master vLLM's core technologies: PagedAttention, Continuous Batching
- Learn to deploy local models with Ollama and customize Modelfile configurations
- Understand model quantization (GPTQ, AWQ, INT8/INT4) principles and selection criteria
- Master building production-grade OpenAI-compatible API services with FastAPI
- Learn engineering practices: load balancing, Docker containerization, performance monitoring

## 2. 背景介绍 (Background)

Cloud-based API services (such as OpenAI's ChatGPT API or Alibaba's Bailian platform) offer convenience but come with significant limitations for enterprise use cases. Data privacy regulations (GDPR, China's Cybersecurity Law, PIPL) often prohibit sending sensitive data to external servers. Per-token pricing models make large-scale usage economically unviable for many organizations. Network latency and API rate limits create reliability concerns for real-time applications.

Private deployment of large language models addresses these challenges by keeping both the model and data within the organization's infrastructure. This approach offers data sovereignty, predictable costs (hardware investment vs. per-call fees), sub-100ms latency (no network round-trip), and the ability to customize the model through fine-tuning.

The landscape of private deployment tools has matured significantly. vLLM, developed at UC Berkeley, introduced PagedAttention -- a technique inspired by virtual memory management that dramatically improves GPU memory utilization for KV cache storage. Combined with Continuous Batching, vLLM achieves 15-20x throughput improvements over naive Hugging Face inference.

For simpler use cases, Ollama provides a one-command deployment experience ("ollama run qwen:7b") that abstracts away all the complexity of model loading, quantization, and inference. For production services, FastAPI combined with OpenAI-compatible API formats has become the de facto standard, enabling seamless migration between cloud and private deployments.

## 3. 基础概念 (Basic Concepts)

### 3.1 vLLM核心架构

```
┌────────────────────────────────────────────────────────────┐
│                    vLLM Inference Engine                    │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              SCHEDULER (调度器)                       │   │
│  │  ┌──────────────┐    ┌──────────────────────┐      │   │
│  │  │ REQUEST QUEUE│───>│ CONTINUOUS BATCHING  │      │   │
│  │  │ (请求队列)   │    │  动态添加/移除请求     │      │   │
│  │  └──────────────┘    └──────────────────────┘      │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        │                                     │
│                        v                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PAGED ATTENTION (分页注意力)             │   │
│  │                                                       │   │
│  │  GPU MEMORY LAYOUT:                                   │   │
│  │  ┌──────┬──────┬──────┬──────┬──────┬──────┐        │   │
│  │  │Page1 │Page2 │Page3 │Page4 │Page5 │Page6 │        │   │
│  │  │      │      │      │      │      │      │        │   │
│  │  │Req1  │Req1  │Req3  │Req2  │Req1  │Req3  │        │   │
│  │  │K,V   │K,V   │K,V   │K,V   │K,V   │K,V   │        │   │
│  │  └──────┴──────┴──────┴──────┴──────┴──────┘        │   │
│  │                                                       │   │
│  │  LOGICAL -> PHYSICAL MAPPING:                         │   │
│  │    Req1: [Page1, Page2, Page5]                       │   │
│  │    Req2: [Page4]                                     │   │
│  │    Req3: [Page3, Page6]                              │   │
│  │                                                       │   │
│  │  BENEFITS:                                            │   │
│  │    - 无内存碎片 (所有页面都在使用)                     │   │
│  │    - 短序列少分配, 长序列多分配                        │   │
│  │    - 页面可GPU-CPU交换                                │   │
│  │    - 支持更大批次 → 更高吞吐                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Continuous Batching vs Static Batching

```
  STATIC BATCHING (problem):
  ┌──────────────────────────────────────────┐
  │ t=0:  [Req1] [Req2] [Req3] [Req4]       │
  │ t=1:  [Req1] [Req2] [DONE] [Req4]       │ ← Req3完成, 槽位浪费
  │ t=2:  [Req1] [Req2] [---- ] [Req4]      │
  │ t=3:  [DONE] [Req2] [---- ] [Req4]      │ ← Req1完成, 两个槽位浪费
  │                                          │
  │ GPU利用率: ~50%                           │
  └──────────────────────────────────────────┘

  CONTINUOUS BATCHING (solution):
  ┌──────────────────────────────────────────┐
  │ t=0:  [Req1] [Req2] [Req3] [Req4]       │
  │ t=1:  [Req1] [Req2] [Req5] [Req4]       │ ← Req3完成→立即插入Req5
  │ t=2:  [Req6] [Req2] [Req5] [Req4]       │ ← Req1完成→立即插入Req6
  │ t=3:  [Req6] [Req7] [Req5] [Req4]       │ ← Req2完成→立即插入Req7
  │                                          │
  │ GPU利用率: ~98%                           │
  │ 吞吐提升: 6-10x                           │
  └──────────────────────────────────────────┘
```

### 3.3 量化技术对比

```
    ORIGINAL        BIT WIDTH        MEMORY (7B model)
    FP32            ──── 32bit ────  28.0 GB  │
                                               │  2x compression
    FP16/BF16       ──── 16bit ────  14.0 GB  │
                                               │  4x compression
    INT8            ────  8bit ────   7.0 GB  │
                                               │  8x compression
    INT4 (GPTQ/AWQ) ────  4bit ────   3.5 GB  │

    ┌──────────────┬──────────┬──────────┬──────────────┐
    │ Method       │ Bits     │ Memory   │ Quality Loss │
    ├──────────────┼──────────┼──────────┼──────────────┤
    │ FP32         │ 32       │ 28.0 GB  │ 0%           │
    │ BF16         │ 16       │ 14.0 GB  │ ~0%          │
    │ INT8         │ 8        │ 7.0 GB   │ <0.5%        │
    │ INT4 AWQ     │ 4        │ 3.5 GB   │ 1-2%         │
    │ INT4 GPTQ    │ 4        │ 3.5 GB   │ 1-3%         │
    └──────────────┴──────────┴──────────┴──────────────┘
```

### 3.4 部署架构层级

```
  LEVEL 1: SINGLE MACHINE (单机部署)
  ┌──────────────────────────────────────────┐
  │  Client ──> FastAPI ──> vLLM/Ollama      │
  │                                          │
  │  GPU: RTX 4090 x1                        │
  │  Model: Qwen3.7-8B (INT4)                  │
  │  Concurrency: 1-10                       │
  └──────────────────────────────────────────┘

  LEVEL 2: MULTI-GPU (多卡部署)
  ┌──────────────────────────────────────────┐
  │  Client ──> Nginx ──> FastAPI x2         │
  │               │         vLLM (TP=2)       │
  │                                          │
  │  GPU: RTX 4090 x2 / A10 x4              │
  │  Model: Qwen-14B (FP16)                  │
  │  Concurrency: 10-100                     │
  └──────────────────────────────────────────┘

  LEVEL 3: CLUSTER (集群部署)
  ┌──────────────────────────────────────────┐
  │  Client ──> CDN ──> K8s Cluster          │
  │    ├─ FastAPI Pods (hpa auto-scale)      │
  │    ├─ vLLM Pods (TP=4/8, A100)           │
  │    ├─ Redis Cluster (cache)              │
  │    └─ Kafka (async queue)                │
  │                                          │
  │  GPU: A100/H100 x8+                      │
  │  Model: Qwen-72B (AWQ)                   │
  │  Concurrency: 100-1000+                  │
  └──────────────────────────────────────────┘
```

## 4. 环境准备 (Environment Setup)

### 4.1 本实验依赖

```bash
# 基础依赖（实验脚本可运行）
pip install numpy

# 运行演示
python run.py
```

### 4.2 生产环境依赖

```bash
# vLLM推理引擎
pip install vllm

# FastAPI服务
pip install fastapi uvicorn

# Ollama本地部署
# 访问 https://ollama.ai/download 下载安装

# Docker部署
# 安装Docker + NVIDIA Container Toolkit
```

### 4.3 API服务启动（生产参考）

```bash
# vLLM OpenAI兼容服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.7-8B \
    --gpu-memory-utilization 0.9

# FastAPI服务
uvicorn api_server:app --host 0.0.0.0 --port 8000

# Ollama服务
ollama serve  # 默认监听 http://localhost:11434
```

## 5. 实践项目 (Practice Project)

### 5.1 项目结构

```
5.2_私有化部署/
├── run.py                    # 主演示脚本 (~700行)
├── api_server.py             # 自动生成的FastAPI服务代码
├── 课程章节内容.md             # 详细课程讲义
├── 5.2_私有化部署.ipynb      # Jupyter Notebook
└── README.md                 # 本文档
```

### 5.2 演示模块

| 部分 | 内容 | 说明 |
|------|------|------|
| 第1部分 | vLLM批量推理模拟 | Continuous Batching vs Sequential对比 |
| 第2部分 | FastAPI服务代码生成 | 自动生成生产级API服务 |
| 第3部分 | 性能基准测试 | 4/8/16/32请求对比 |
| 第4部分 | 量化方案对比 | 7种量化方案x5种模型大小 |
| 第5部分 | 部署架构推荐 | 小/中/大规模部署方案 |

## 6. 实验步骤 (Experiment Steps)

### Step 1: vLLM Continuous Batching模拟

```python
class VLLMSimulator:
    """vLLM风格连续批处理推理引擎"""
    def __init__(self, model_name="Qwen3.7-8B",
                 max_batch_size=32, max_seq_len=8192):
        self.active_requests = []
        self.requests_queue = []

    def step_batch(self) -> list:
        """执行一个推理批次步骤"""
        # 填充batch到最大容量
        while self.requests_queue and len(self.active_requests) < self.max_batch_size:
            self.active_requests.append(self.requests_queue.pop(0))

        completed = []
        for req in self.active_requests:
            if random.random() < 0.15:  # 15%概率完成
                completed.append(req.request_id)

        # 移除已完成的请求（为新请求腾出空间）
        self.active_requests = [r for r in self.active_requests
                                if r.request_id not in completed]
        return completed
```

### Step 2: FastAPI服务代码生成

实验脚本自动生成一个完整的、生产就绪的FastAPI推理服务代码（约200行），包含：

- OpenAI兼容的 `/v1/chat/completions` 和 `/v1/completions` 端点
- `/v1/models` 和 `/v1/health` 管理端点
- CORS中间件、请求日志中间件
- 异步并发控制（asyncio.Semaphore）
- 流式输出支持（Server-Sent Events）
- Prometheus监控指标（总请求数、活跃请求数、错误数）

```python
# 生成的api_server.py关键结构
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    async with request_semaphore:  # 并发控制
        metrics["total_requests"] += 1
        if req.stream:
            return StreamingResponse(
                engine.generate_stream(...),
                media_type="text/event-stream"
            )
        content = engine.generate(req.messages, ...)
        return JSONResponse({
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
        })
```

### Step 3: 性能基准测试

```python
def run_benchmark():
    request_counts = [4, 8, 16, 32]
    # 对比: 顺序推理 vs 批量推理
    # 输出: P50/P95/P99延迟, 吞吐量(tokens/s), 加速比

# 示例输出:
#   方法         请求数    总耗时(s)    吞吐(tok/s)    P95延迟(ms)
#   顺序         4         2.150        0.9           520
#   批量         4         0.820        4.9           95
#   顺序         16        8.500        1.9           1820
#   批量         16        1.200        13.3          180
```

## 7. 实验结果 (Experiment Results)

### 7.1 vLLM批量推理模拟

运行 `python run.py` 的**真实输出摘要**：

```
============================================================
  第5.2章 - 私有化部署技术
  演示脚本
============================================================

============================================================
第1部分: vLLM 连续批处理推理模拟
============================================================

[模拟场景]
  模型: Qwen3.7-8B
  GPU显存: 24 GB
  最大批次: 32
  最大吞吐: 2000 tokens/s
  并发请求: 10 个

[Continuous Batching 执行统计]
  完成请求数:    10
  总耗时:        0.42s
  平均延迟:      0.04s/请求

[与顺序推理对比]
  顺序推理预估:  3.50s
  批量推理预估:  0.42s
  加速比:        8.33x

[关键技术特性]
  PagedAttention: KV Cache分页管理，显存利用率提升2-4x
  Continuous Batching: 动态添加/移除请求，GPU利用率接近100%
  Prefix Caching: 共享前缀自动缓存，减少重复计算
  量化支持: AWQ/GPTQ量化，显存占用降低50-75%
```

### 7.2 性能基准测试对比

```
方法          请求数   总耗时(s)   吞吐(tok/s)   P95延迟(ms)
────────────────────────────────────────────────────────
顺序          4        2.150       0.9          520
批量          4        0.820       4.9          95
顺序          8        4.300       1.9          1010
批量          8        0.950       8.4          150
顺序          16       8.500       1.9          1820
批量          16       1.200       13.3         180
顺序          32       17.00       1.9          3600
批量          32       1.650       19.4         210

[性能对比总结]
  4请求:  批量推理吞吐提升 5.4x
  8请求:  批量推理吞吐提升 4.4x
  16请求: 批量推理吞吐提升 7.0x
  32请求: 批量推理吞吐提升 10.3x
```

### 7.3 量化方案分析

```
方法      精度    压缩比   1.7B(GB) 8B(GB)  14B(GB) 32B(GB) 72B(GB)
─────────────────────────────────────────────────────────────
FP32      32bit   1.0x    3.0      14.0     28.0    64.0    144.0
FP16      16bit   2.0x    1.5      7.0      14.0    32.0    72.0
INT8      8bit    4.0x    0.8      3.5      7.0     16.0    36.0
INT4      4bit    8.0x    0.5      1.8      3.5     8.0     18.0
AWQ       4bit    8.0x    0.5      1.8      3.5     8.0     18.0
GPTQ      4bit    8.0x    0.5      1.8      3.5     8.0     18.0

[消费级GPU部署参考]
  GPU            显存     推荐模型
  RTX 3060      12GB     Qwen3.7-8B (INT4), 1.7B (FP16)
  RTX 3090/4090 24GB     Qwen3.7-14B (INT4), 8B (FP16)
  RTX 4070 Ti   12GB     Qwen3.7-8B (INT4), 1.7B (FP16)
  A100          40/80GB  Qwen3.7-72B (INT4), 32B (FP16)
```

### 7.4 部署架构推荐

实验脚本展示了三种规模的部署架构：

**方案A（小规模）**: 单机RTX 4090 + llama.cpp/Ollama + FastAPI，支持1-10并发，适合小团队
**方案B（中规模）**: 多卡vLLM + Redis缓存 + Nginx负载均衡，支持10-100并发，适合中小企业
**方案C（大规模）**: K8s集群 + Tensor Parallelism + Kafka + Prometheus/Grafana，支持100-1000+并发

## 8. 结果分析 (Result Analysis)

本次实验通过模拟和数据对比，全面展示了私有化部署的核心技术价值。以下进行深入分析。

**Continuous Batching的吞吐革命。** 实验数据显示，在32个请求的场景下，批量推理相比顺序推理实现了10.3倍的吞吐提升。这个提升来自两个机制：(1) GPU利用率从~50%提升到~98%，因为完成的请求立即被新请求替换；(2) 批次处理分摊了模型加载和KV Cache管理的固定开销。在实际部署中，这种提升意味着：一台带vLLM的RTX 4090可以处理原本需要10台机器的工作量，大幅降低硬件成本。这也是为什么几乎所有的生产级LLM推理服务（无论是云服务商还是私有部署）都采用了Continuous Batching技术。

**量化技术的选择策略。** 实验对比了7种量化方案在不同模型大小下的显存占用。关键洞察是：AWQ（Activation-aware Weight Quantization）在4bit量化下保持了最好的精度（精度损失<2%），因为它考虑了激活值中的离群通道（outlier channels）。对于大多数生产场景，推荐使用AWQ INT4作为默认方案——它将72B模型的显存占用从144GB降到18GB，使8张A100-80G可行。如果需要最高精度且显存充裕（如A100-80G），使用BF16；如果显存极度受限（如RTX 3060 12GB），使用INT4 GPTQ（兼容性更好）。

**FastAPI服务的工程化价值。** 实验自动生成的api_server.py包含了一个完整的生产级服务的所有要素：OpenAI兼容性确保现有客户端代码无需修改；CORS中间件支持跨域Web访问；Semaphore并发控制防止GPU过载；SSE流式输出改善用户体验；健康检查和Prometheus指标支撑运维监控。这个200行的生成代码可以直接作为企业级服务的骨架，替换其中的模拟推理引擎为vLLM或Ollama后端即可投入生产。

**部署架构的渐进式演进。** 实验展示了从小到大的三种部署架构，实际企业应从方案A（单机）开始验证业务可行性，确认需求后再升级到方案B（多卡）或方案C（集群）。关键决策点：并发用户数超过50时考虑多卡部署；日均API调用超过100万次时考虑K8s自动扩缩容；SLA要求99.9%以上时需要跨可用区冗余部署。不建议初期过度设计（Over-engineering），简单的FastAPI + vLLM已经可以应对大部分中小规模场景。

**成本对比分析。** 假设日均10万次API调用，每次平均500 tokens：使用云API（如通义千问API，0.008元/1K tokens）月度成本约12000元；而使用RTX 4090私有部署（电费+硬件折旧），月度成本约3000元，节省75%。如果日均调用量达到100万次，节省可达90%以上。但私有部署需要一次性硬件投入（RTX 4090约15000元）和持续的运维人力，需要根据实际业务规模综合评估。

## 9. 扩展学习 (Extended Learning)

### 9.1 进阶方向

**vLLM实际部署实战** -- 在真实GPU服务器上部署vLLM，测试不同模型（Qwen3.7-8B/14B/72B）、不同量化方案（FP16/INT8/INT4 AWQ）、不同并发参数下的实际性能。使用locust进行压力测试，建立性能基线。

**Kubernetes + vLLM集群** -- 使用K8s部署vLLM推理集群，实现基于HPA（水平Pod自动扩缩容）的弹性伸缩。根据GPU利用率和请求队列长度自动增减推理Pod数量。集成Prometheus Operator + Grafana Dashboard实现可视化监控。

**多模型路由网关** -- 构建一个API网关，根据请求特征（长度、复杂度、预算）自动路由到不同的后端模型。例如，简单问候路由到Qwen3.7-1.7B，代码生成路由到DeepSeek-Coder，复杂推理路由到Qwen3.7-72B。使用LiteLLM作为多模型代理。

**GPU推理优化进阶** -- 学习FlashAttention-2、TensorRT-LLM、vLLM quantization、Speculative Decoding（推测解码）等更底层的GPU优化技术。了解如何编写CUDA Kernel来进一步加速推理。

**模型安全与合规检查** -- 在推理服务中集成内容安全审核（通义内容安全API）、PII过滤、速率限制、审计日志等安全模块。确保私有部署满足GDPR和《个人信息保护法》的合规要求。

### 9.2 推荐资源

- vLLM官方文档: https://docs.vllm.ai/
- Ollama官方: https://ollama.ai/
- FastAPI文档: https://fastapi.tiangolo.com/
- GPTQ论文: "GPTQ: Accurate Post-Training Quantization" (Frantar et al., 2023)
- AWQ论文: "AWQ: Activation-aware Weight Quantization" (Lin et al., 2023)
- PagedAttention论文: "Efficient Memory Management for Large Language Model Serving" (Kwon et al., 2023)
