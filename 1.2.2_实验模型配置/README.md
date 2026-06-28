# 1.2.2 实验模型配置 (Model Configuration)

---

## 1. 课程目标

- 掌握全局模型配置模块的设计与使用，理解"一次配置、全局复用"的工程原则
- 能够独立完成 `.env` 文件的创建，配置 DeepSeek、Qwen、GLM 等主流国产模型的 API Key
- 熟练使用 `get_client()` 工厂函数和 `verify_config()` 诊断工具
- 理解 OpenAI 兼容接口的统一抽象模式，能够在不同模型提供商之间自由切换
- 掌握新增模型提供商的注册方法

---

## 2. 背景介绍

在大模型应用开发中，API Key 管理是最基础也最容易出问题的环节。常见问题包括：

- **重复配置**：每节课、每个实验都要写一遍 `load_dotenv()` + `OpenAI(api_key=..., base_url=...)`，大量重复代码
- **Key 泄露**：硬编码在代码中的 Key 被不小心提交到 Git
- **切换困难**：想从 DeepSeek 换到 Qwen，需要修改多处代码

本模块将这些重复逻辑抽象为一个统一的 `config` 模块，包含：

- **提供商注册表** (`providers.py`)：集中管理所有模型的 base_url、model ID、pricing 等信息，2026年6月最新数据
- **统一客户端** (`client.py`)：封装 OpenAI 兼容接口，支持 `chat()`、`chat_stream()`、Function Calling
- **诊断工具** (`verify.py`)：一键检测所有 API Key 配置状态，输出格式化报告

该模块遵循"配置与代码分离"原则：API Key 存储在 `.env` 文件中（已在 `.gitignore` 中排除），代码通过环境变量读取。

---

## 3. 基础概念

### 3.1 模块架构

```
1.2.2_实验模型配置/
├── README.md              ← 本文档
├── .env.example           ← 配置模板（可提交Git）
├── images/                ← 配图
├── screenshots/           ← 截图
├── references/            ← 参考资料
├── __init__.py             ← 包导出
├── providers.py            ← 7个提供商注册表
├── client.py               ← 统一LLMClient
└── verify.py               ← 诊断工具
```

### 3.2 数据流

```
.env 文件 ──→ os.environ ──→ providers.py (读取) ──→ client.py (创建客户端) ──→ 实验代码
                                                          │
                                                    OpenAI 兼容接口
                                                    ↓
                                              DeepSeek / Qwen / GLM / ...
```

### 3.3 支持的提供商

| ID | 模型 | API端点 | Key环境变量 | 定价 (输入/输出, 元/1M tokens) |
|---|---|---|---|---|
| deepseek | DeepSeek V4 | api.deepseek.com | DEEPSEEK_API_KEY | 1 / 2 |
| deepseek-r1 | DeepSeek-R1-0528 | api.deepseek.com | DEEPSEEK_API_KEY | 4 / 16 |
| qwen | Qwen3.7-Max | dashscope.aliyuncs.com | DASHSCOPE_API_KEY | 20 / 60 |
| qwen-plus | Qwen3.7-Plus | dashscope.aliyuncs.com | DASHSCOPE_API_KEY | 2 / 6 |
| glm | GLM-5.2 | open.bigmodel.cn | ZHIPU_API_KEY | 50 / 50 |
| kimi | Kimi K2.7 | api.moonshot.cn | MOONSHOT_API_KEY | 12 / 12 |
| doubao | 豆包 Seed 2.0 Pro | ark.cn-beijing.volces.com | DOUBAO_API_KEY | 0.8 / 2 |

---

## 4. 环境准备

### 4.1 安装依赖

```bash
pip install openai
```

### 4.2 配置 API Key

**方式一（推荐）：创建 .env 文件**

```bash
# 复制模板
cp 1.2.2_实验模型配置/.env.example .env

# 编辑 .env，填入你的 Key
# 至少配置一个即可
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
```

**方式二：设置环境变量**

```bash
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
```

### 4.3 验证配置

```python
from config import verify_config
verify_config()
```

---

## 5. 实践项目

**项目名称：全局模型配置系统**

使用 `config` 模块完成：

1. 在 `.env` 文件中配置至少一个提供商的 API Key
2. 运行 `verify_config()` 验证配置状态
3. 使用 `get_client()` 自动获取可用客户端
4. 测试 `client.chat()` 基础对话
5. 对比两个不同提供商的回答质量和响应速度
6. 尝试 `client.chat_stream()` 流式输出

---

## 6. 实验步骤

### Step 1：环境检查与配置诊断

```python
# 添加项目根目录到 Python 路径（Jupyter 中需要）
import sys
sys.path.insert(0, '..')

from config import verify_config, list_providers

# 列出所有已注册的提供商
print("已注册提供商:", list_providers())

# 运行配置诊断
report = verify_config()
print(f"\n配置状态: {report['status']}")
print(f"已配置: {report['available_count']}/{report['total_count']}")
```

### Step 2：获取客户端并测试对话

```python
from config import get_client

# 自动选择第一个可用的提供商
client = get_client()
print(f"使用模型: {client.name} ({client.model})")

# 测试基础对话
result = client.chat("用一句话介绍什么是大语言模型")
print(f"回答: {result['content']}")
print(f"Token消耗: 输入{result['usage']['input']}, 输出{result['usage']['output']}")
```

### Step 3：指定提供商对比

```python
# 对比 DeepSeek 和 Qwen 的回答
questions = [
    "解释什么是MoE架构",
    "用Python写一个快速排序",
]

for provider_id in ["deepseek", "qwen-plus"]:
    try:
        client = get_client(provider_id)
        print(f"\n=== {client.name} ===")
        for q in questions:
            result = client.chat(q, temperature=0.3, max_tokens=300)
            print(f"  Q: {q}")
            print(f"  A: {result['content'][:150]}...")
            print(f"  Token: {result['usage']['output']}")
    except ValueError as e:
        print(f"  {provider_id}: 未配置 - {e}")
```

### Step 4：流式输出

```python
client = get_client()  # 自动选择

print("流式输出:")
for token in client.chat_stream("讲一个关于AI的冷笑话"):
    print(token, end='', flush=True)
print()
```

### Step 5：新增提供商（扩展演示）

```python
# 演示如何临时添加一个自定义提供商
from config.providers import PROVIDERS

# 查看当前所有提供商
for pid, info in PROVIDERS.items():
    print(f"  {pid}: {info['name']} -> {info['base_url']}")

# 如需新增，编辑 config/providers.py 添加条目即可
```

---

## 7. 实验结果

运行 `verify_config()` 的真实输出：

```
============================================================
  模型配置诊断报告
============================================================
  项目路径: G:\CLAUDE CODE_PROJECT\SHIXI\course
  Python:   3.12.0
  .env文件: 已找到
  OpenAI库: 1.35.0

  已配置: 6/7 个提供商
  提供商           模型                    状态         Key预览
  ---------------- ---------------------- ---------- --------------------
  deepseek        deepseek-chat           OK         sk-6****4727
  deepseek-r1     deepseek-reasoner       OK         sk-6****4727
  qwen            qwen3.7-max             OK         sk-w****2Mr8
  qwen-plus       qwen3.7-plus            OK         sk-w****2Mr8
  glm             glm-5.2                 OK         当前****key
  kimi            kimi-k2.7               OK         当前****key
  doubao          doubao-seed-2.0-pro     --         未配置

  >>> 运行: from config import get_client
      client = get_client()  # 自动选择可用提供商
============================================================
```

---

## 8. 结果分析

**为什么需要全局配置模块？**

在本次实验之前，每节课都需要独立配置 API Key，存在三个核心问题：

1. **代码重复**：每个 Notebook 都要写 `load_dotenv()`、`OpenAI(api_key=..., base_url=...)` 等样板代码，违反了 DRY 原则
2. **维护成本高**：当模型升级（如 Qwen3 → Qwen3.7）或 API 端点变更时，需要逐个修改所有 Notebook
3. **配置错误难以排查**：没有统一的诊断工具，当某个实验报"API Key 未找到"时，学生不知道是哪个环节出了问题

`config` 模块通过**集中管理 + 统一接口**解决了这些问题。所有模型信息集中在 `providers.py` 一个文件中，一次更新即可影响所有实验。`verify_config()` 提供了可视化的诊断报告，让学生一目了然地看到哪些 Key 已配置、哪些缺失。

**设计模式的价值**

该模块体现了两个重要的软件工程原则：

- **工厂模式**：`get_client()` 根据参数自动创建对应的 `LLMClient` 实例，调用者无需关心底层实现细节
- **配置与代码分离**：敏感信息（API Key）存储在 `.env` 文件中，代码通过环境变量读取，避免了硬编码和 Git 泄露风险

---

## 9. 扩展学习

**多模型路由**：在实际生产项目中，可以在 `get_client()` 基础上实现智能路由——根据任务复杂度、预算限制、延迟要求等条件，自动选择最合适的模型。例如：简单翻译用 `qwen-plus`（快速便宜），代码生成用 `deepseek`（代码能力强），复杂推理用 `deepseek-r1`。

**配置加密**：`.env` 文件以明文存储 API Key，在企业环境中不够安全。可以使用 `python-dotenv` + 加密库（如 `cryptography`）实现加密存储，或接入企业的密钥管理服务（如阿里云 KMS、HashiCorp Vault）。

**类型安全**：当前模块使用字典存储提供商信息，可以进一步使用 `dataclass` 或 `Pydantic` 模型增强类型安全性，在 IDE 中获得自动补全和类型检查支持。

**MCP 协议集成**：2026 年 MCP (Model Context Protocol) 已成为 Agent 工具接入的标准协议。`config` 模块可以进一步扩展，支持 MCP 服务器的注册和管理——统一管理 LLM API 和 MCP Tool 的配置。

---

*本章（1.2.2）与 1.2.1 模型生态与选型形成互补：1.2.1 解决"选什么模型"的问题，1.2.2 解决"如何统一配置和使用"的问题。*
*所有后续章节的实验将直接使用本模块获取 LLM 客户端。*
