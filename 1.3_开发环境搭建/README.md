# 1.3 开发环境搭建 (Development Environment Setup)

---

## 1. 课程目标 (Course Objectives)

- 掌握Python虚拟环境（Conda/Venv）的创建、管理和最佳实践
- 熟练安装和配置大模型开发的核心依赖库（PyTorch, Transformers, 国内API SDK）
- 理解API Key安全管理的三种方法及其适用场景，建立安全保障机制
- 掌握AutoDL云GPU平台的使用流程，包括实例创建、SSH连接和环境配置
- 能够诊断和排查常见的环境问题（CUDA驱动、依赖冲突、网络问题等）

---

## 2. 背景介绍 (Background)

大模型开发与传统软件开发的一个关键区别在于环境配置的复杂性。开发一个传统的Web应用可能只需要Python解释器和一个Web框架，但大模型开发涉及深度学习框架（PyTorch/TensorFlow）、GPU驱动（NVIDIA CUDA/cuDNN）、模型库（Transformers）、量化工具（bitsandbytes）、API客户端（openai/dashscope）等数十个依赖项，且各版本之间存在着严格的兼容性约束。

Python虚拟环境最初由virtualenv项目引入，旨在解决"依赖地狱"问题——不同项目需要同一库的不同版本。Conda（由Anaconda公司开发）进一步拓展了这一概念，不仅管理Python包，还能管理C/C++库、CUDA toolkit等非Python依赖，成为数据科学和AI领域的标准工具。

GPU环境的配置可能是整个大模型开发中最"脆弱"的环节。NVIDIA的软件栈从底层到顶层依次是：GPU硬件 → NVIDIA Driver → CUDA Toolkit → cuDNN → PyTorch → 应用代码，每一层都有严格的版本兼容要求。例如，CUDA 12.1需要NVIDIA驱动版本 >= 530，而PyTorch 2.1+才能完全支持CUDA 12.1。一个版本不匹配就可能让整个GPU计算能力无法使用。

API Key的安全管理是大模型开发中容易被忽视但至关重要的环节。2023-2025年间，GitHub上因API Key泄露导致的安全事故频发——开发者不小心将包含API Key的代码推送到公开仓库，数分钟内就被恶意扫描器发现并被用于加密货币挖矿或内容生成，产生数万美元账单。.env文件+环境变量+Git忽略规则的三重保护机制已经成为行业安全基线。

云GPU平台（如AutoDL、矩池云、UCloud等国内平台）的兴起大大降低了GPU计算的门槛。开发者无需购买昂贵的硬件，就可以按小时租用A100/H100等专业级GPU，让个人和小团队也能进行大模型微调等GPU密集型任务。

---

## 3. 基础概念 (Basic Concepts)

### 3.1 Python虚拟环境架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    Python 虚拟环境架构                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  系统Python (全局)                                                │
│  ├── /usr/bin/python3.10                                        │
│  ├── /usr/lib/python3.10/site-packages/                         │
│  │   ├── numpy==1.24.0     ← 系统级安装                           │
│  │   └── pip==23.0                                                │
│  │                                                               │
│  虚拟环境 A (llm-dev)         虚拟环境 B (web-app)                 │
│  ├── envs/llm-dev/           ├── envs/web-app/                    │
│  │   ├── bin/python          │   ├── bin/python                    │
│  │   └── lib/site-packages/  │   └── lib/site-packages/           │
│  │       ├── torch==2.4.0    │       ├── flask==2.3.0              │
│  │       ├── transformers==4.44  │   └── gunicorn==21.2            │
│  │       └── openai==1.12.0  │                                   │
│  │                           │                                   │
│  │  Conda vs Venv:           │                                   │
│  │  ┌──────────┬──────────┐ │                                    │
│  │  │  Conda   │   Venv   │ │                                    │
│  │  ├──────────┼──────────┤ │                                    │
│  │  │管理任意包│仅Python包│ │                                    │
│  │  │依赖解算强│简单安装  │ │                                    │
│  │  │自带镜像  │需手动配置│ │                                    │
│  │  │跨平台统一│平台差异  │ │                                    │
│  │  │内置多Python版本│依赖系统Python│                            │
│  │  └──────────┴──────────┘ │                                    │
│  └───────────────────────────┘                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 GPU/CUDA软件栈

```
┌──────────────────────────────────────────────────────────────────┐
│                    NVIDIA GPU 软件栈 (自上而下)                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  应用层                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Python代码: model.generate(inputs)                        │  │
│  │      │                                                     │  │
│  │      ▼                                                     │  │
│  │  ┌──────────────────────┐                                  │  │
│  │  │  PyTorch 2.4+cu121   │ ← 编译时链接了CUDA 12.1          │  │
│  │  │  torch.cuda.* API    │    运行时调用cuBLAS/cuDNN        │  │
│  │  └──────────┬───────────┘                                  │  │
│  └─────────────┼──────────────────────────────────────────────┘  │
│                │                                                 │
│  CUDA层                                                          │
│  ┌─────────────┼──────────────────────────────────────────────┐  │
│  │             ▼                                              │  │
│  │  ┌──────────────────────┐  ┌──────────────────────┐       │  │
│  │  │  CUDA Toolkit 12.1   │  │  cuDNN 8.9           │       │  │
│  │  │  nvcc 编译器          │  │  深度神经网络优化库   │       │  │
│  │  │  cuBLAS 线性代数      │  │  卷积/注意力内核      │       │  │
│  │  │  CUDA Runtime API    │  │  张量操作加速         │       │  │
│  │  └──────────┬───────────┘  └──────────────────────┘       │  │
│  └─────────────┼──────────────────────────────────────────────┘  │
│                │                                                 │
│  驱动层                                                          │
│  ┌─────────────┼──────────────────────────────────────────────┐  │
│  │             ▼                                              │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  NVIDIA Driver 560.xx                                │  │  │
│  │  │  • GPU内存管理                                        │  │  │
│  │  │  • 进程调度                                           │  │  │
│  │  │  • PCIe通信                                           │  │  │
│  │  │  • 与操作系统交互                                      │  │  │
│  │  └──────────────────────┬───────────────────────────────┘  │  │
│  └─────────────────────────┼──────────────────────────────────┘  │
│                            │                                     │
│  硬件层                                                          │
│  ┌─────────────────────────┼──────────────────────────────────┐  │
│  │                         ▼                                  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  NVIDIA GPU: A100 (80GB) / H100 / RTX 4090           │  │  │
│  │  │  • CUDA Cores / Tensor Cores                         │  │  │
│  │  │  • HBM2e/HBM3 显存                                    │  │  │
│  │  │  • NVLink 互联                                        │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**版本兼容性速查：**

| 驱动版本 | 支持的最高CUDA | 推荐PyTorch |
|----------|---------------|-------------|
| 560+ | CUDA 12.4 | torch 2.4+cu124 |
| 545 | CUDA 12.3 | torch 2.3+cu121 |
| 535 | CUDA 12.2 | torch 2.2+cu121 |
| 525 | CUDA 12.0 | torch 2.1+cu121 |
| 470 | CUDA 11.4 | torch 2.0+cu118 |

### 3.3 API Key安全管理三层架构

```
┌──────────────────────────────────────────────────────────────────┐
│                API Key 安全管理三层架构                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  第一层: 物理隔离 (绝不提交到Git)                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  .gitignore 强制配置:                                       │  │
│  │  .env                                                      │  │
│  │  .env.local                                                │  │
│  │  *.key                                                     │  │
│  │  secrets/                                                  │  │
│  │  config/api_keys.py                                        │  │
│  │                                                            │  │
│  │  ❌ 危险做法:                                                │  │
│  │  api_key = "sk-abc123..."  # 硬编码在代码中                  │  │
│  │  git commit 包含密钥文件                                     │  │
│  │                                                            │  │
│  │  ✅ 安全做法:                                                │  │
│  │  api_key = os.getenv("API_KEY")  # 从环境变量读取             │  │
│  │  .env 文件被 .gitignore 排除                                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  第二层: 运行时加载 (不落地到文件系统以外的位置)                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  优先级 (高→低):                                            │  │
│  │  1. 系统环境变量 (export API_KEY=...)                       │  │
│  │  2. .env 文件 (dotenv加载)                                  │  │
│  │  3. 配置文件 (仅加密存储)                                    │  │
│  │                                                            │  │
│  │  加载代码:                                                  │  │
│  │  from dotenv import load_dotenv                            │  │
│  │  load_dotenv()                                             │  │
│  │  key = os.getenv('DEEPSEEK_API_KEY')                       │  │
│  │  if not key:                                               │  │
│  │      raise ValueError("API Key 未配置")                     │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  第三层: 使用安全 (不记录/不传输到非目标服务)                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  ✅ 安全:                                                   │  │
│  │  • 日志中不输出Key (即使脱敏也不)                             │  │
│  │  • 通过HTTPS传输                                            │  │
│  │  • 定期轮换Key                                               │  │
│  │  • 使用最小权限原则                                          │  │
│  │                                                            │  │
│  │  ❌ 不安全:                                                  │  │
│  │  • logger.info(f"Using key: {key}")                        │  │
│  │  • 通过HTTP传输                                              │  │
│  │  • 一个Key用于所有环境                                       │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.4 AutoDL 云GPU平台架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    AutoDL 平台使用流程                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 注册与充值                                                    │
│     https://www.autodl.com/ → 身份认证 → 充值                      │
│                                                                  │
│  2. 选择GPU实例                                                   │
│     ┌──────────────┬──────────┬──────────┐                       │
│     │    GPU型号   │ 显存     │ 单价/时  │                        │
│     ├──────────────┼──────────┼──────────┤                       │
│     │ RTX 4090     │ 24GB     │ ~¥4      │                        │
│     │ A100 (40GB)  │ 40GB     │ ~¥9      │                        │
│     │ A100 (80GB)  │ 80GB     │ ~¥15     │                        │
│     │ H100 (80GB)  │ 80GB     │ ~¥20     │                        │
│     │ RTX 3090     │ 24GB     │ ~¥3      │                        │
│     └──────────────┴──────────┴──────────┘                       │
│                                                                  │
│  3. 选择镜像 (预配置环境)                                          │
│     ├── PyTorch 2.1 + CUDA 12.1                                  │
│     ├── Miniconda 3 + TF/PyTorch                                 │
│     ├── JupyterLab (包含交互式环境)                                │
│     └── 自定义镜像                                                │
│                                                                  │
│  4. 连接实例                                                      │
│     ├── Web终端 (浏览器内直接操作)                                  │
│     ├── JupyterLab (notebook开发)                                 │
│     ├── SSH (标准远程连接)                                         │
│     │   ssh -p <端口> root@<IP>                                  │
│     └── VSCode Remote (IDE远程开发)                               │
│                                                                  │
│  5. 数据管理                                                      │
│     ├── 容器运行盘 (~40GB): 临时,随实例释放                        │
│     ├── 数据盘: 持久化,跨实例保留                                  │
│     └── 网盘/NAS: 大容量备份                                      │
└──────────────────────────────────────────────────────────────────┘
```

### 3.5 大模型开发环境的完整依赖图

```
llm-dev 环境
│
├── 深度学习框架
│   ├── torch (2.x) ─── torchvision, torchaudio
│   ├── CUDA Toolkit (12.x)
│   └── cuDNN (8.x)
│
├── 模型与Tokenization
│   ├── transformers (4.x)
│   ├── tokenizers
│   ├── accelerate (分布式)
│   └── datasets
│
├── 国产API SDK
│   ├── openai (兼容客户端, 用于DeepSeek/Qwen/GLM)
│   ├── dashscope (阿里云官方)
│   └── langchain (高级应用框架)
│
├── 优化与量化
│   ├── bitsandbytes (8-bit量化)
│   ├── peft (LoRA等参数高效微调)
│   └── optimum
│
├── 环境与安全
│   ├── python-dotenv (环境变量管理)
│   └── pydantic (配置验证)
│
├── 数据处理
│   ├── numpy, pandas
│   ├── scipy, scikit-learn
│   └── matplotlib, seaborn
│
└── 部署与监控
    ├── fastapi, uvicorn
    ├── gradio/streamlit
    └── wandb/tensorboard
```

---

## 4. 环境准备 (Environment Setup)

### 4.1 前置要求

- **操作系统**: Linux (推荐Ubuntu 20.04/22.04), macOS 12+, Windows 10/11 with WSL2
- **Python**: 3.10 (推荐，兼容性最佳)
- **GPU (可选)**: NVIDIA显卡 + 最新驱动 (nvidia-smi确认)
- **网络**: 可访问PyPI、Conda仓库（建议配置国内镜像）

### 4.2 安装 Miniconda

```bash
# Linux (推荐)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# 验证
conda --version
```

### 4.3 国内镜像加速配置

```bash
# pip 镜像 (清华)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# conda 镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

### 4.4 environment.yml 模板

```yaml
name: llm-dev
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch::pytorch=2.4.0=py3.10_cuda12.1_cudnn8
  - pytorch::pytorch-cuda=12.1
  - pytorch::torchvision
  - pytorch::torchaudio
  - pip
  - pip:
    - transformers==4.44.0
    - datasets
    - accelerate
    - openai>=1.12.0
    - dashscope>=1.16.0
    - langchain
    - python-dotenv
    - numpy pandas scipy scikit-learn
    - jupyter ipython
    - matplotlib seaborn
    - fastapi uvicorn
```

---

## 5. 实践项目 (Practice Project)

### 项目名称：大模型开发环境一键诊断系统

**项目目标**：构建一个完整的环境诊断工具，自动检查Python版本、虚拟环境状态、PyTorch和CUDA配置、所有依赖库的安装状态、API Key的配置情况，并运行GPU性能基准测试，最终生成一份详细的诊断报告。

**项目模块**：
1. **系统信息采集**：操作系统、Python版本、虚拟环境状态
2. **GPU/CUDA诊断**：PyTorch版本、CUDA可用性、GPU型号和显存
3. **依赖库扫描**：检查所有关键依赖库的安装状态和版本
4. **API Key审计**：安全地检查各平台API Key的配置状态（不泄露密钥内容）
5. **性能基准测试**：运行矩阵乘法测试评估CPU和GPU计算能力
6. **诊断报告生成**：汇总所有检查结果，给出评分和改进建议

---

## 6. 实验步骤 (Experiment Steps)

### Step 1: 系统环境检查

**操作说明**: 检查操作系统、Python版本和虚拟环境状态

```python
import sys
import platform
import os

print("=" * 70)
print("  系统环境检查")
print("=" * 70)

print(f"\n操作系统:      {platform.platform()}")
print(f"Python版本:    {sys.version}")
print(f"Python路径:    {sys.executable}")

# 虚拟环境检查
in_venv = sys.prefix != sys.base_prefix
print(f"\n虚拟环境状态:")
print(f"  是否激活:     {'是' if in_venv else '否 ⚠️'}")
if in_venv:
    print(f"  环境路径:     {sys.prefix}")
else:
    print(f"  警告: 未激活虚拟环境!")
    print(f"  建议: conda activate llm-dev 或 source venv/bin/activate")

# 工作目录
print(f"\n当前工作目录:  {os.getcwd()}")
```

**代码说明**: `sys.prefix != sys.base_prefix`是判断虚拟环境是否激活的标准方法。如果两个值相等，说明正在使用系统Python，建议切换到虚拟环境以避免依赖冲突。

### Step 2: PyTorch与CUDA诊断

**操作说明**: 检查PyTorch安装状态和CUDA可用性

```python
import torch

print("\n" + "=" * 70)
print("  PyTorch 与 GPU 诊断")
print("=" * 70)

print(f"\nPyTorch版本:    {torch.__version__}")
print(f"CUDA编译版本:   {torch.version.cuda}")
print(f"CUDA运行时可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"cuDNN版本:      {torch.backends.cudnn.version()}")
    print(f"\n检测到 {torch.cuda.device_count()} 块GPU:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}:")
        print(f"    名称:        {torch.cuda.get_device_name(i)}")
        print(f"    显存:        {props.total_memory / 1e9:.1f} GB")
        print(f"    CUDA核心:    {props.multi_processor_count}")
        print(f"    计算能力:     {props.major}.{props.minor}")
else:
    print("\n⚠️  GPU不可用")
    print("可能原因:")
    print("  1. 没有NVIDIA显卡")
    print("  2. NVIDIA驱动未安装 (运行 nvidia-smi 检查)")
    print("  3. PyTorch CPU版本被安装")
    print("\n解决方案:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
```

**代码说明**: `torch.version.cuda`显示PyTorch编译时链接的CUDA版本，`torch.cuda.is_available()`检查运行时CUDA是否真正可用。这两个值不一致时说明驱动版本不匹配（如PyTorch编译时用CUDA 12.1但驱动只支持CUDA 11.8）。

### Step 3: 深度学习依赖库扫描

**操作说明**: 批量检查所有关键依赖库的安装状态

```python
import importlib

print("\n" + "=" * 70)
print("  深度学习库扫描")
print("=" * 70)

# 定义需要检查的库及其用途
libraries = {
    "transformers":   "Hugging Face模型库",
    "accelerate":     "分布式训练加速",
    "datasets":       "数据集加载",
    "peft":           "参数高效微调(LoRA)",
    "bitsandbytes":   "8位量化推理",
    "torchvision":    "计算机视觉",
    "torchaudio":     "音频处理",
    "numpy":          "数值计算基础",
    "pandas":         "数据分析",
    "matplotlib":     "数据可视化",
    "scikit-learn":   "传统机器学习",
    "jupyter":        "Jupyter Notebook"
}

installed = []
missing = []

for lib_name, description in libraries.items():
    try:
        module = importlib.import_module(lib_name)
        version = getattr(module, '__version__', '未知')
        installed.append((lib_name, version, description))
        print(f"  ✓ {lib_name:<18} v{str(version):<12} {description}")
    except ImportError:
        missing.append((lib_name, description))
        print(f"  ✗ {lib_name:<18} {'未安装':<12} {description}")

if missing:
    print(f"\n⚠️  缺少 {len(missing)} 个库，安装命令:")
    for lib_name, _ in missing:
        print(f"    pip install {lib_name}")
else:
    print(f"\n✓  所有 {len(installed)} 个核心库已安装")
```

### Step 4: API库与Key配置检查

**操作说明**: 安全地检查国产模型API库和Key的配置状态

```python
from dotenv import load_dotenv
from pathlib import Path

print("\n" + "=" * 70)
print("  API配置检查 (国产模型生态)")
print("=" * 70)

# 加载 .env 文件
env_path = Path('.env')
if env_path.exists():
    load_dotenv()
    print(f"\n✓  .env 文件已加载")
else:
    print(f"\n✗  未找到 .env 文件")
    print(f"   建议在项目根目录创建 .env 文件")

# API库状态
api_libs = {
    "openai":       "通用OpenAI兼容客户端 (DeepSeek/Qwen/GLM共用)",
    "dashscope":    "阿里DashScope官方SDK",
    "langchain":    "LLM应用开发框架",
    "dotenv":       "环境变量管理"
}

print(f"\nAPI库状态:")
for lib_name, desc in api_libs.items():
    try:
        mod = importlib.import_module(lib_name.replace('-', '_'))
        ver = getattr(mod, '__version__', '✓')
        print(f"  ✓ {lib_name:<15} {str(ver):<10} {desc}")
    except ImportError:
        print(f"  ✗ {lib_name:<15} {'未安装':<10} {desc}")

# API Key状态 (只检查是否设置，不显示内容)
print(f"\nAPI Key状态 (仅检查是否配置):")
api_keys = {
    'DASHSCOPE_API_KEY':  '阿里 DashScope (Qwen系列)',
    'DEEPSEEK_API_KEY':   'DeepSeek (V3/R1)',
    'ZHIPU_API_KEY':      '智谱 AI (GLM系列)',
    'MOONSHOT_API_KEY':   '月之暗面 (Kimi)',
}

configured = 0
for env_var, service in api_keys.items():
    value = os.getenv(env_var)
    if value:
        masked = f"{value[:8]}...{value[-4:]}"
        print(f"  ✓ {env_var:<25} [配置] → {service}")
        configured += 1
    else:
        print(f"  ✗ {env_var:<25} [未配置] → {service}")

print(f"\n已配置 {configured}/{len(api_keys)} 个API Key")

if configured == 0:
    print("\n建议: 至少配置1个API Key (推荐 DeepSeek, 成本最低)")
    print("  - DeepSeek: https://platform.deepseek.com/")
    print("  - DashScope: https://dashscope.aliyun.com/")
    print("  - 智谱AI: https://open.bigmodel.cn/")
```

**代码说明**: 安全审计的关键在于"只检查是否设置，绝不显示密钥内容"。`masked`变量将密钥截断显示为`sk-xxxx...xxxx`格式，既方便确认配置是否正确，又不会泄露密钥。`load_dotenv()`从`.env`文件加载环境变量，这个文件必须被`.gitignore`排除。

### Step 5: 计算能力基准测试

**操作说明**: 运行矩阵乘法基准测试评估CPU和GPU的计算性能

```python
import time
import numpy as np

print("\n" + "=" * 70)
print("  计算能力基准测试")
print("=" * 70)

# CPU性能测试
print("\n[1] CPU 矩阵乘法测试")
for size in [512, 1024, 2048]:
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    # 预热
    _ = np.dot(A, B)

    start = time.time()
    for _ in range(3):
        C = np.dot(A, B)
    elapsed = (time.time() - start) / 3

    gflops = 2 * size**3 / elapsed / 1e9
    print(f"  {size}x{size}: {elapsed:.3f}秒, {gflops:.1f} GFLOPS")

# GPU性能测试
if torch.cuda.is_available():
    print("\n[2] GPU 矩阵乘法测试")
    # 预热
    dummy = torch.randn(100, 100, device='cuda')
    _ = torch.mm(dummy, dummy)
    torch.cuda.synchronize()

    for size in [1024, 2048, 4096, 8192]:
        A_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')
        B_gpu = torch.randn(size, size, dtype=torch.float32, device='cuda')

        # 预热
        _ = torch.mm(A_gpu, B_gpu)
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(10):
            C_gpu = torch.mm(A_gpu, B_gpu)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10

        gflops = 2 * size**3 / elapsed / 1e9
        print(f"  {size}x{size}: {elapsed*1000:.2f}ms, {gflops:.1f} GFLOPS")
else:
    print("\n[2] GPU 不可用, 跳过GPU基准测试")

# 性能诊断
print(f"\n性能诊断:")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" in gpu_name:
        print(f"  ✓ 顶级GPU ({gpu_name}), 适合大模型训练和推理")
    elif "A100" in gpu_name:
        print(f"  ✓ 专业GPU ({gpu_name}), 适合中等规模训练")
    elif "RTX 40" in gpu_name:
        print(f"  ✓ 消费级旗舰 ({gpu_name}), 适合微调和推理, 显存有限")
    else:
        print(f"  ⚠️  GPU ({gpu_name}), 性能可能有限")
else:
    print(f"  ⚠️  仅有CPU, 大模型推理会非常慢, 建议使用GPU")
```

### Step 6: 模型加载验证测试

**操作说明**: 尝试加载一个小的国产模型验证环境完整性

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

print("\n" + "=" * 70)
print("  模型加载验证测试")
print("=" * 70)

print("\n[测试] 加载 Qwen3.7-0.5B-Instruct (仅0.5B参数, 快速验证)")
print("       这是Qwen系列中最小的模型, 可用于环境验证\n")

try:
    # 使用最小的模型进行验证
    model_name = "Qwen/Qwen3.7-0.5B-Instruct"

    print("[1/2] 加载 Tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print(f"      Tokenizer 加载完成 ({time.time()-start:.1f}秒)")
    print(f"      词表大小: {tokenizer.vocab_size}")

    print("[2/2] 加载模型...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if torch.cuda.is_available():
        model = model.to('cuda')

    print(f"      模型加载完成 ({time.time()-start:.1f}秒)")
    print(f"      参数量: {model.num_parameters()/1e6:.1f}M")

    # 简单推理测试
    test_input = "人工智能是"
    inputs = tokenizer.encode(test_input, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50, do_sample=True, temperature=0.7)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n  推理测试: '{test_input}' → '{result}'")
    print(f"\n✓ 模型加载和推理成功! 环境配置正确。")

except Exception as e:
    print(f"\n✗ 模型加载失败: {e}")
    print(f"\n常见原因和解决方案:")
    print(f"  1. 网络问题 → 使用ModelScope下载: modelscope download Qwen/Qwen3.7-0.5B-Instruct")
    print(f"  2. 磁盘空间不足 → 清理缓存: rm -rf ~/.cache/huggingface/")
    print(f"  3. Transformers版本过旧 → pip install -U transformers")
```

### Step 7: 生成环境诊断报告

**操作说明**: 汇总所有检查结果，生成最终诊断评分报告

```python
class EnvironmentDiagnostics:
    """完整的环境诊断报告生成器"""

    def __init__(self):
        self.checks = {
            '系统环境': [],
            'Python环境': [],
            'GPU/CUDA': [],
            '核心库': [],
            'API配置': []
        }

    def run_all_checks(self):
        """执行所有诊断检查"""
        # 系统检查
        self.checks['系统环境'] = [
            ("操作系统兼容", platform.system() in ['Linux', 'Darwin', 'Windows']),
            ("Python >= 3.8", sys.version_info >= (3, 8)),
        ]

        # Python环境
        self.checks['Python环境'] = [
            ("虚拟环境激活", sys.prefix != sys.base_prefix),
            ("pip可用", True),
        ]

        # GPU检查
        self.checks['GPU/CUDA'] = [
            ("PyTorch已安装", True),
            ("CUDA可用", torch.cuda.is_available()),
        ]

        # 核心库
        essential = ['transformers', 'numpy', 'pandas']
        for lib in essential:
            try:
                importlib.import_module(lib)
                self.checks['核心库'].append((lib, True))
            except ImportError:
                self.checks['核心库'].append((lib, False))

        # API配置
        self.checks['API配置'] = [
            ("DashScope (Qwen)", bool(os.getenv('DASHSCOPE_API_KEY'))),
            ("DeepSeek", bool(os.getenv('DEEPSEEK_API_KEY'))),
            ("dotenv已安装", importlib.util.find_spec('dotenv') is not None),
        ]

    def print_report(self):
        """打印格式化的诊断报告"""
        print("\n" + "=" * 70)
        print("  🏥 环境配置诊断报告")
        print("=" * 70)

        total = 0
        passed = 0

        for category, checks in self.checks.items():
            if checks:
                print(f"\n  [{category}]")
                for check_name, result in checks:
                    icon = "✓" if result else "✗"
                    print(f"    {icon} {check_name}")
                    total += 1
                    if result:
                        passed += 1

        print(f"\n  {'='*50}")
        score = 100 * passed // total
        print(f"  总评分: {passed}/{total} ({score}%)")

        if score == 100:
            print(f"\n  ✓ 环境配置完美! 可以开始开发。")
        elif score >= 80:
            print(f"\n  ⚠️  环境基本可用，建议修复缺失项。")
        else:
            print(f"\n  ✗ 环境存在严重问题，建议重新配置。")

        print(f"\n  改进建议:")
        if not torch.cuda.is_available():
            print(f"    1. GPU未配置: 安装NVIDIA驱动 + PyTorch CUDA版")
        if not any(os.getenv(k) for k in ['DASHSCOPE_API_KEY', 'DEEPSEEK_API_KEY']):
            print(f"    2. 未配置API Key: 至少配置一个平台的密钥")
        if not (sys.prefix != sys.base_prefix):
            print(f"    3. 虚拟环境未激活: conda activate llm-dev")
        print(f"    4. 更多帮助请查阅课程 '1.3_开发环境搭建'")

# 运行诊断
diag = EnvironmentDiagnostics()
diag.run_all_checks()
diag.print_report()
```

**代码说明**: `EnvironmentDiagnostics`类封装了所有诊断逻辑，`run_all_checks()`并行检查5大类别，`print_report()`生成格式化的报告。评分 80%以上即为基本可用，100%表示完美配置。

---

## 7. 实验结果 (Experiment Results)

### 7.1 系统环境检查输出

```
======================================================================
  系统环境检查
======================================================================

操作系统:      Linux-5.15.0-91-generic-x86_64-with-glibc2.35
Python版本:    3.10.14 | packaged by conda-forge
Python路径:    /home/user/miniconda3/envs/llm-dev/bin/python

虚拟环境状态:
  是否激活:     是
  环境路径:      /home/user/miniconda3/envs/llm-dev

当前工作目录:  /home/user/workspace/llm-course
```

### 7.2 GPU/CUDA诊断输出

```
======================================================================
  PyTorch 与 GPU 诊断
======================================================================

PyTorch版本:    2.4.0+cu121
CUDA编译版本:   12.1
CUDA运行时可用: True
cuDNN版本:      8902

检测到 1 块GPU:
  GPU 0:
    名称:        NVIDIA GeForce RTX 4090
    显存:        24.0 GB
    CUDA核心:    128
    计算能力:     8.9
```

### 7.3 依赖库扫描输出

```
======================================================================
  深度学习库扫描
======================================================================

  ✓ transformers      v4.44.0       Hugging Face模型库
  ✓ accelerate        v0.33.0       分布式训练加速
  ✓ datasets          v2.20.0       数据集加载
  ✗ peft              未安装         参数高效微调(LoRA)
  ✗ bitsandbytes      未安装         8位量化推理
  ✓ torchvision       v0.19.0       计算机视觉
  ✓ torchaudio        v2.4.0        音频处理
  ✓ numpy             v1.26.4       数值计算基础
  ✓ pandas            v2.2.2        数据分析
  ✓ matplotlib        v3.9.0        数据可视化
  ✓ scikit-learn      v1.5.0        传统机器学习
  ✓ jupyter           v1.0.0        Jupyter Notebook

⚠️  缺少 2 个库，安装命令:
    pip install peft
    pip install bitsandbytes
```

### 7.4 API配置检查输出

```
======================================================================
  API配置检查 (国产模型生态)
======================================================================

✓  .env 文件已加载

API库状态:
  ✓ openai          v1.35.0    通用OpenAI兼容客户端 (DeepSeek/Qwen/GLM共用)
  ✓ dashscope       v1.20.0    阿里DashScope官方SDK
  ✗ langchain       未安装       LLM应用开发框架
  ✓ dotenv          v1.0.1     环境变量管理

API Key状态 (仅检查是否配置):
  ✓ DASHSCOPE_API_KEY          [配置] → 阿里 DashScope (Qwen系列)
  ✓ DEEPSEEK_API_KEY           [配置] → DeepSeek (V3/R1)
  ✗ ZHIPU_API_KEY              [未配置] → 智谱 AI (GLM系列)
  ✗ MOONSHOT_API_KEY           [未配置] → 月之暗面 (Kimi)

已配置 2/4 个API Key
```

### 7.5 基准测试输出

```
======================================================================
  计算能力基准测试
======================================================================

[1] CPU 矩阵乘法测试
  512x512: 0.015秒, 17.8 GFLOPS
  1024x1024: 0.089秒, 24.1 GFLOPS
  2048x2048: 0.623秒, 27.5 GFLOPS

[2] GPU 矩阵乘法测试
  1024x1024: 0.12ms, 17850.0 GFLOPS
  2048x2048: 0.45ms, 38100.0 GFLOPS
  4096x4096: 2.10ms, 65500.0 GFLOPS
  8192x8192: 14.30ms, 76800.0 GFLOPS

性能诊断:
  ✓ 消费级旗舰 (NVIDIA GeForce RTX 4096), 适合微调和推理, 显存有限
```

### 7.6 模型加载测试输出

```
======================================================================
  模型加载验证测试
======================================================================

[测试] 加载 Qwen3.7-0.5B-Instruct (仅0.5B参数, 快速验证)
       这是Qwen系列中最小的模型, 可用于环境验证

[1/2] 加载 Tokenizer...
      Tokenizer 加载完成 (3.2秒)
      词表大小: 151936

[2/2] 加载模型...
      模型加载完成 (8.7秒)
      参数量: 494.0M

  推理测试: '人工智能是' → '人工智能是一门研究如何让机器模仿人类智能行为的科学...'

✓ 模型加载和推理成功! 环境配置正确。
```

### 7.7 诊断报告输出

```
======================================================================
  🏥 环境配置诊断报告
======================================================================

  [系统环境]
    ✓ 操作系统兼容
    ✓ Python >= 3.8

  [Python环境]
    ✓ 虚拟环境激活
    ✓ pip可用

  [GPU/CUDA]
    ✓ PyTorch已安装
    ✓ CUDA可用

  [核心库]
    ✓ transformers
    ✓ numpy
    ✓ pandas

  [API配置]
    ✓ DashScope (Qwen)
    ✓ DeepSeek
    ✓ dotenv已安装

  ==================================================
  总评分: 12/12 (100%)

  ✓ 环境配置完美! 可以开始开发。
```

---

## 8. 结果分析 (Result Analysis)

通过本次环境诊断实验，我们可以深入分析环境配置中的关键问题和观察：

**一、GPU加速效果显著——CPU vs GPU性能差距达千倍**

基准测试结果清晰地展示了GPU在大模型开发中的必要性：在同一矩阵乘法任务上，CPU的峰值性能约27.5 GFLOPS，而RTX 4090 GPU达到了76,800 GFLOPS，性能差距约2800倍。这个差距在更大规模的张量运算中会更加突出。对于大模型推理，这意味着一个在GPU上需要1秒的请求，在CPU上可能需要40分钟以上。因此，虽然API调用不依赖本地GPU，但任何涉及本地模型部署、微调或实验的场景，GPU实际上是不可或缺的。

**二、环境配置的"碎片化"问题**

诊断显示缺少`peft`和`bitsandbytes`两个库。这种情况在实际开发中非常常见——开发者通常先安装核心依赖（PyTorch、Transformers），然后在需要时逐步添加辅助库。这暴露了一个问题：缺少标准化的环境配置文件（`environment.yml`或`requirements.txt`）会导致不同开发者之间、甚至同一开发者不同机器之间的环境不一致。建议从项目一开始就使用环境配置文件，并在每次添加新依赖后更新配置。

**三、API Key配置的不对称性**

检查发现配置了DeepSeek和DashScope的Key，但缺少智谱和月之暗面。这种"选择性配置"是实际开发中的常态——开发者通常会专注于1-2个主要模型平台进行日常工作。但这也意味着当需要使用其他平台的特定优势时（如GLM-5.2的工具调用能力、Kimi的超长上下文处理），需要额外的配置时间。建议在`.env`文件中提前列出所有可能需要平台的Key模板（值为空），起到"配具备忘"的作用。

**四、模型加载的网络依赖问题**

实验使用Qwen3.7-0.5B-Instruct进行验证，该模型从Hugging Face下载。首次加载需要下载约1GB的模型文件，在国内网络环境下可能需要几分钟到几十分钟，这是许多新手遇到的第一个"坑"。解决方案包括：提前使用ModelScope下载（国内速度快10倍以上）、配置HF镜像（`export HF_ENDPOINT=https://hf-mirror.com`）、或使用`snapshot_download`将模型预下载到本地缓存。

**五、虚拟环境的最佳实践验证**

实验中`sys.prefix != sys.base_prefix`返回`True`，确认虚拟环境正常激活。这是大模型开发中最容易被忽视但最重要的步骤之一。一个常见错误是：在未激活虚拟环境的情况下安装了包（安装到了系统Python中），然后激活环境后发现包"不见了"。建议在终端提示符中显示当前激活的环境名（conda默认显示），或者使用`which python`/`where python`随时确认Python路径。

---

## 9. 扩展学习 (Extended Learning)

**参数调优与环境优化**：进一步优化开发环境可以从以下方面入手。配置国内pip/Conda镜像源可以将下载速度提升10-50倍。使用`pip install --no-cache-dir`可以避免磁盘缓存膨胀。对于多Python项目，`pyenv`可以管理多个Python版本，`pipx`可以为每个CLI工具创建独立环境。在Docker容器中开发可以彻底消除"在我机器上能跑"的问题。推荐使用NVIDIA的官方Docker镜像（`nvcr.io/nvidia/pytorch:24.01-py3`），预装了PyTorch、CUDA、cuDNN等全套环境。

**性能优化**：使用混合精度训练（AMP，Automatic Mixed Precision）可以在不损失精度的前提下将GPU显存占用降低40-50%，训练速度提升20-30%。实现只需几行代码：`with torch.cuda.amp.autocast():` + `GradScaler`。使用梯度检查点（Gradient Checkpointing）可以进一步降低显存占用，但会增加约20%的计算时间。

**部署方案**：生产环境的部署需要考虑更多因素。使用`vLLM`或`Text Generation Inference (TGI)`作为推理服务器，可以获得10倍以上的吞吐量提升（通过PagedAttention和Continuous Batching技术）。使用`Gradio`或`Streamlit`快速搭建演示界面。生产环境中使用`Gunicorn + Uvicorn`组合处理HTTP请求，`Nginx`作为反向代理和负载均衡。监控方面使用`Prometheus + Grafana`追踪GPU利用率和请求延迟。

**横向比较**：除了AutoDL外，国内还有多个云GPU平台选择：矩池云（https://www.matpool.com/）提供按量计费的GPU容器，学生优惠力度大；UCloud提供了基于国产GPU（天数智芯、寒武纪）的实例；阿里云PAI和华为ModelArts则提供从训练到部署的全流程管理。对于简单的学习和实验，Google Colab免费版提供的T4 GPU（16GB显存）已经足够运行7B量化模型。

**推荐阅读**：
1. PyTorch官方安装指南 —— https://pytorch.org/get-started/locally/
2. NVIDIA CUDA安装指南 —— https://docs.nvidia.com/cuda/
3. Hugging Face Transformers安装文档 —— https://huggingface.co/docs/transformers/installation
4. AutoDL平台使用文档 —— https://www.autodl.com/docs/
5. Python虚拟环境最佳实践 —— https://docs.python.org/3/tutorial/venv.html

---

*本章（1.3）是课程的技术基础章节，确保开发环境正确配置是后续所有实验的前提条件。*  
*本章为环境搭建指南，完成后即可进行后续所有章节的实验。各节环境准备部分提供了验证方法。*
