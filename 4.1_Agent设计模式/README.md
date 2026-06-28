# 4.1 Agent设计模式 (Agent Design Patterns)

## 1. 课程目标 (Course Objectives)

**中文:**
- 理解AI Agent的核心概念与设计原理，区分Agent系统与传统对话系统的本质差异
- 掌握ReAct（推理+行动）模式的实现机制，能独立构建带工具调用的ReAct Agent
- 掌握Plan-and-Solve（先规划后执行）模式，理解复杂任务分解的方法论
- 实现Agent记忆系统（短期/长期/工作记忆），理解记忆在Agent决策中的作用
- 建立Agent安全边界（工具白名单、输入校验、速率限制），防范恶意输入
- 能够为实际业务场景选择合适的设计模式，并评估安全性

**English:**
- Understand the core concepts and design principles of AI Agents, distinguishing them from traditional dialogue systems
- Master the ReAct (Reasoning + Acting) pattern implementation and build tool-calling Agents
- Master the Plan-and-Solve pattern, decompose complex tasks into executable steps
- Implement Agent memory systems (short-term/long-term/working memory)
- Establish Agent security boundaries (tool whitelisting, input validation, rate limiting)
- Select appropriate design patterns for real-world business scenarios

## 2. 背景介绍 (Background)

Traditional LLM applications follow a simple question-answer pattern: the user asks, the model responds. While effective for simple conversations, this approach fails when facing complex, multi-step tasks that require tool usage, reasoning, and planning.

The concept of an "Agent" emerged from the need for AI systems that can autonomously perceive their environment, make decisions, take actions, and improve through feedback. In the context of Large Language Models, an Agent is an LLM-powered system capable of:

1. **Perception** -- gathering information from the environment and context
2. **Reasoning** -- using the LLM's capabilities to analyze problems
3. **Planning** -- decomposing complex tasks into executable steps
4. **Tool Use** -- calling external tools (APIs, databases, code execution)
5. **Reflection** -- evaluating results and improving strategies

The landmark paper "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2023) established the foundational pattern that most modern Agent frameworks implement. The key insight is that interleaving reasoning traces with action steps significantly improves task success rates compared to reasoning-only or acting-only approaches.

In the Chinese AI ecosystem, domestic models like Qwen (通义千问) and DeepSeek provide robust function-calling capabilities that make Agent implementation practical and cost-effective. This chapter focuses on building Agent systems using these domestic models, with all examples using the DashScope (阿里云灵积) API for Qwen models and DeepSeek's API.

## 3. 基础概念 (Basic Concepts)

### 3.1 Agent的四大核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    USER / ENVIRONMENT                     │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │ Input                                  │
│                         v                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              1. LLM CORE (大模型核心)                      │   │
│  │    - Reasoning & Thinking (推理与思考)                     │   │
│  │    - Decision Making (决策制定)                            │   │
│  │    - Action Planning (行动规划)                            │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│         ┌───────────────┼───────────────┐                       │
│         v               v               v                        │
│  ┌───────────┐  ┌───────────┐  ┌───────────────┐               │
│  │ 2. TOOL   │  │ 3. MEMORY │  │ 4. PLANNING   │               │
│  │   SYSTEM  │  │   SYSTEM  │  │   & REASONING │               │
│  │           │  │           │  │               │               │
│  │ - Search  │  │ - Short   │  │ - Decompose   │               │
│  │ - Calc    │  │ - Long    │  │ - Schedule    │               │
│  │ - DB      │  │ - Working │  │ - Track       │               │
│  │ - API     │  │ - Vector  │  │ - Adapt       │               │
│  └─────┬─────┘  └─────┬─────┘  └───────┬───────┘               │
│        v              v                v                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   OUTPUT / ACTION                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 ReAct模式循环 (ReAct Loop)

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   THOUGHT    │────>│    ACTION    │────>│ OBSERVATION  │
│   (思考)     │     │   (行动)     │     │   (观察)     │
│              │<────│              │<────│              │
│ "我需要搜索  │     │ Tool: search │     │ Result: ...  │
│  相关资料"   │     │ query="AI"   │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                            │
       │              循环直到得出最终答案              │
       │                                            │
       └────────────  FINAL ANSWER  <───────────────┘
                     (最终答案)
```

### 3.3 Plan-and-Solve模式

```
          PHASE 1: PLANNING (规划阶段)
    ┌─────────────────────────────────────┐
    │  User Goal → LLM → Step-by-step Plan │
    │                                        │
    │  Step 1: Search information on X       │
    │  Step 2: Query database for Y          │
    │  Step 3: Calculate statistics on Z     │
    │  Step 4: Synthesize results            │
    └─────────────────────────────────────┘
                       │
                       v
          PHASE 2: EXECUTION (执行阶段)
    ┌─────────────────────────────────────┐
    │  For each Step:                       │
    │    1. Select appropriate tool         │
    │    2. Execute with parameters         │
    │    3. Collect intermediate result     │
    │    4. Verify against dependencies     │
    │    5. Adjust plan if needed           │
    └─────────────────────────────────────┘
                       │
                       v
          PHASE 3: SYNTHESIS (综合阶段)
    ┌─────────────────────────────────────┐
    │  Collect all results → LLM →          │
    │  Generate comprehensive final answer  │
    └─────────────────────────────────────┘
```

### 3.4 Multi-Agent协作架构对比

```
  Master-Slave (主从模式)         Debate (辩论模式)
  ┌──────────────┐              ┌─────┐ ┌─────┐ ┌─────┐
  │ Master Agent │              │ Ag1 │ │ Ag2 │ │ Ag3 │
  └──────┬───────┘              └──┬──┘ └──┬──┘ └──┬──┘
    ┌─────┼─────┐                  │       │       │
    v     v     v                  └───┬───┴───┬───┘
  [W1]  [W2]  [W3]                    │       │
  Worker Agents                   Consensus (共识)
```

### 3.5 记忆系统架构

```
┌───────────────────────────────────────────────┐
│              MEMORY SYSTEM                     │
├───────────┬───────────────┬───────────────────┤
│ SHORT-TERM│ WORKING       │ LONG-TERM         │
│ (短期记忆) │ (工作记忆)     │ (长期记忆)         │
├───────────┼───────────────┼───────────────────┤
│ Storage:  │ Storage:      │ Storage:          │
│ List/Dict │ Dict/Stack    │ Vector DB         │
│           │               │                   │
│ Capacity: │ Capacity:     │ Capacity:         │
│ 20-50 msgs│ Task-specific │ Unlimited         │
│           │               │                   │
│ Speed:    │ Speed:        │ Speed:            │
│ Fast      │ Fast          │ Moderate          │
│           │               │                   │
│ Purpose:  │ Purpose:      │ Purpose:          │
│ Recent    │ Current task  │ Historical        │
│ dialogue  │ state & vars  │ knowledge         │
└───────────┴───────────────┴───────────────────┘
```

## 4. 环境准备 (Environment Setup)

### 4.1 依赖安装

```bash
# 安装必要的Python依赖
pip install openai  # DashScope API使用OpenAI兼容接口

# 可选依赖（用于生产环境）
pip install redis          # 长期记忆存储
pip install faiss-cpu      # 向量检索（持久化记忆）
pip install sentence-transformers  # 嵌入模型
```

### 4.2 配置DashScope API Key（通义千问）

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "sk-your-dashscope-api-key"

# Linux / macOS
export DASHSCOPE_API_KEY="sk-your-dashscope-api-key"

# 获取API Key: https://dashscope.console.aliyun.com/
```

### 4.3 环境验证

```python
# 测试API连接
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen3.7-plus",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
```

> **注意:** 本实验的安全测试（场景3）不需要API Key即可运行。ReAct和Plan-Solve演示需要API Key。

## 5. 实践项目 (Practice Project)

### 5.1 项目概览

本项目构建一个完整的Agent设计模式实验平台，包含三大核心模块：

1. **工具系统 (Tool System)** -- 搜索工具、计算器工具、数据库查询工具的注册与执行
2. **Agent引擎 (Agent Engine)** -- ReAct Agent（Thought-Action-Observation循环）和 Plan-Solve Agent（先规划后执行）
3. **安全边界 (Security Boundary)** -- 工具白名单、输入模式校验、速率限制

### 5.2 实验脚本结构

```
4.1_Agent设计模式/
├── run.py                    # 主实验脚本
├── 课程章节内容.md             # 详细课程讲义
├── 4.11_Agent设计模式.ipynb   # Jupyter Notebook
└── README.md                 # 本文档
```

### 5.3 核心技术点

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM引擎 | DashScope Qwen3.7-Plus | 使用OpenAI兼容接口，性价比最优 |
| 工具注册 | Python class继承 | Tool基类 + run() + to_schema() |
| 解析器 | 正则表达式 | 解析LLM输出的Action/Action Input |
| 记忆系统 | @dataclass + dict | 短期对话历史管理 |
| 安全 | 模式匹配 + 白名单 | 禁止危险操作、超长输入截断 |

## 6. 实验步骤 (Experiment Steps)

### Step 1: 构建工具系统

```python
# 工具基类定义
class Tool:
    """所有工具的抽象基类"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, **kwargs) -> str:
        """执行工具，子类必须覆写"""
        raise NotImplementedError

    def to_schema(self) -> dict:
        """生成OpenAI function-calling兼容的工具定义"""
        raise NotImplementedError

# 搜索工具实现
class SearchTool(Tool):
    _KNOWLEDGE = {
        "人工智能": "人工智能是计算机科学的分支...",
        "Agent": "AI Agent是能自主感知环境、做出决策的智能体。",
        # ...更多知识条目
    }

    def run(self, query: str = "") -> str:
        """全文匹配 + 部分匹配搜索"""
        results = []
        for key, value in self._KNOWLEDGE.items():
            if key in query:
                results.append(f"  [{key}] {value}")
        return "\n".join(results) if results else "未找到相关结果"

# 安全计算器工具（使用安全命名空间，禁止危险操作）
class CalculatorTool(Tool):
    def run(self, expression: str = "") -> str:
        safe_ns = {
            "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt,
            "pi": math.pi, "e": math.e, "abs": abs, "round": round,
            # 只暴露安全的数学函数，不暴露os/sys/subprocess
        }
        result = eval(expression, {"__builtins__": {}}, safe_ns)
        return f"{expression} = {result}"
```

### Step 2: 实现ReAct Agent循环

```python
class ReActAgent:
    """ReAct Agent -- Thought-Action-Observation循环"""

    def run(self, user_query: str) -> str:
        self.memory.add_user_message(user_query)

        for iteration in range(1, self.max_iterations + 1):
            # Step 1: 调用LLM获取Thought + Action
            response = call_llm(self.memory.get_context())

            # Step 2: 解析Action
            tool_name, params, thought = self._parse_action(response)

            # Step 3: 检查是否到达最终答案
            if tool_name == "final_answer":
                return params.get("answer", response)

            # Step 4: 安全检查
            if not self.security.check_tool(tool_name):
                continue

            # Step 5: 执行工具
            tool = self.tools.get(tool_name)
            result = tool.run(**params)

            # Step 6: 记录观察结果到记忆
            self.memory.add_observation(result)

        return "达到最大迭代次数"

    def _parse_action(self, text: str):
        """解析LLM输出中的Action和Final Answer"""
        fa_match = re.search(r"Final\s*Answer\s*:\s*(.+)", text, re.I | re.DOTALL)
        if fa_match:
            return "final_answer", {"answer": fa_match.group(1).strip()}, text

        action_match = re.search(r"Action\s*:\s*(\w+)", text, re.I)
        input_match = re.search(r"Action\s*Input\s*:\s*(.+)", text, re.I | re.DOTALL)

        if action_match:
            return action_match.group(1).strip().lower(), {"query": input_match.group(1).strip()}, None
        return None, None, text
```

### Step 3: 建立安全边界

```python
class SecurityGuard:
    """安全守卫 -- 工具白名单 + 输入校验"""

    ALLOWED_TOOLS = {"search", "calculator", "database"}  # 白名单

    # 危险模式（防注入）
    FORBIDDEN_PATTERNS = [
        r"__import__", r"os\.", r"subprocess", r"eval\s*\(",
        r"exec\s*\(", r"open\s*\(", r"file\(", r"\.write",
        r"rm\s+-rf", r"/bin/", r"import\s+os", r"import\s+sys",
    ]

    def check_tool(self, tool_name: str) -> bool:
        """检查工具是否在白名单中"""
        if tool_name not in self.ALLOWED_TOOLS:
            print(f"  [安全] 拒绝: 工具 '{tool_name}' 不在白名单中")
            return False
        return True

    def sanitize_input(self, text: str) -> tuple[bool, str]:
        """检查输入是否包含危险模式"""
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"[安全] 检测到危险模式: {pattern}"
        if len(text) > 2000:
            return False, "[安全] 输入过长 (最大2000字符)"
        return True, text
```

## 7. 实验结果 (Experiment Results)

### 7.1 安全边界测试（无需API Key，可直接运行）

以下是从 `python run.py` 捕获的**真实控制台输出**（编码问题导致的乱码已尽可能保留原始输出）：

```
======================================================================
  第4.1章 Agent设计模式 — 实验演示
  模型: qwen3.7-plus  (DashScope / Qwen)
  时间: 2026-06-12 18:52:34
======================================================================

[提示] 未检测到 DASHSCOPE_API_KEY 环境变量
  场景3 (安全边界测试) 不需要 API, 将直接运行
  场景1和场景2 将使用模拟模式运行


######################################################################
# 场景3: 安全边界验证 — 拦截危险请求
######################################################################

[测试] 合法搜索请求
  工具=search, 参数=人工智能
  => 安全检查通过, 允许执行

[测试] 非法工具调用
  工具=delete_all, 参数=drop table
  [安全] 拒绝: 工具 'delete_all' 不在白名单中
  => 已拦截: 工具 'delete_all' 不在白名单

[测试] 合法计算请求
  工具=calculator, 参数=2 + 2
  => 安全检查通过, 允许执行

[测试] 危险输入注入
  工具=search, 参数=__import__('os').system('rm -rf /')
  => 已拦截: [安全] 检测到危险模式: __import__

[测试] 危险表达式
  工具=calculator, 参数=open('/etc/passwd')
  => 已拦截: [安全] 检测到危险模式: open\s*\(

======================================================================
[跳过] 场景1和场景2需要 LLM API, 请设置 DASHSCOPE_API_KEY 后重试
  获取API Key: https://dashscope.console.aliyun.com/
======================================================================

======================================================================
  所有实验场景运行完毕
======================================================================
```

### 7.2 安全测试结果分析

| 测试用例 | 工具 | 参数 | 预期结果 | 实际结果 |
|---------|------|------|---------|---------|
| 合法搜索 | search | 人工智能 | 通过 | PASS - 安全检查通过 |
| 非法工具 | delete_all | drop table | 拦截 | PASS - 工具不在白名单 |
| 合法计算 | calculator | 2 + 2 | 通过 | PASS - 安全检查通过 |
| 危险注入 | search | `__import__('os')...` | 拦截 | PASS - 检测到`__import__` |
| 危险表达式 | calculator | `open('/etc/passwd')` | 拦截 | PASS - 检测到`open(` |

**5/5 测试全部通过，安全边界正常工作。**

### 7.3 场景1和场景2说明

场景1（ReAct Agent多工具协同检索）和场景2（Plan-and-Solve Agent复杂问题分步求解）需要设置 `DASHSCOPE_API_KEY` 环境变量后方可运行。API使用通义千问（Qwen3.7-Plus），通过DashScope的OpenAI兼容接口调用。配置方法：

```bash
set DASHSCOPE_API_KEY=sk-your-api-key-here
python run.py
```

## 8. 结果分析 (Result Analysis)

本次实验通过安全边界测试，完整验证了Agent安全防护体系的三个核心层面。以下从多个角度进行深入分析。

**安全白名单机制的有效性。** 实验中使用 `delete_all` 这样的未注册工具名进行调用，SecurityGuard立即拒绝并输出明确的拦截日志。这一机制确保了Agent只能调用开发者预先注册和审核过的工具，从根本上防止了"工具滥用"攻击。在实际生产环境中，工具白名单应该结合RBAC（基于角色的访问控制），针对不同用户角色动态调整可用工具集合。例如，普通用户可能只能使用搜索和计算功能，而管理员才能访问数据库写入操作。

**输入模式校验的全面性。** 实验测试了两种典型攻击模式：`__import__('os').system('rm -rf /')`（Python代码注入）和 `open('/etc/passwd')`（文件系统访问）。正则模式匹配在两种情况下均成功拦截。但必须指出，基于正则的防护是不完备的——攻击者可以通过编码混淆（如Base64编码）、Unicode同形字或分段注入来绕过。生产环境建议采用多层防护：正则预筛选 + 语义理解模型（如使用Qwen自身判断输入是否危险）+ 沙箱隔离执行。特别是对于代码执行场景，应使用Docker容器或gVisor等强隔离机制。

**安全与功能的平衡。** 实验中 `2 + 2` 这样的合法计算请求顺利通过，而危险操作被精确拦截，体现了安全设计的"最小权限原则"（Principle of Least Privilege）。在Agent设计中，开发者常常面临安全性与灵活性之间的权衡。过度严格的安全策略会限制Agent的能力边界，使Agent在某些场景下"束手束脚"；而过于宽松的策略则可能导致严重的安全事故。建议采用"渐进式信任模型"：新用户/新工具从严格限制开始，根据安全审计日志逐步放宽权限。

**记忆系统的安全考量。** 虽然本次实验的记忆系统较为简单（仅维护对话历史），但在实际生产环境中，记忆系统存储了用户交互历史和系统执行轨迹。这些数据可能包含敏感信息，需要加密存储和定期清理。特别是当使用向量数据库作为长期记忆时，要注意检索内容可能被用于间接推断用户的隐私数据。

**生产环境增强建议。** (1) 添加速率限制（Rate Limiting），防止暴力攻击和资源耗尽；(2) 实现Human-in-the-Loop审批机制，对高风险操作要求人工确认；(3) 建立完整的审计日志，记录每一次工具调用和决策过程；(4) 使用内容安全API（如通义千问内容安全服务）对输入输出进行二次审核；(5) 定期进行红队测试（Red Team Testing），主动发现安全漏洞。

## 9. 扩展学习 (Extended Learning)

### 9.1 进阶方向

**Reflection模式（自我反思）** -- 在每次执行后引入自我评估环节，让Agent分析"这个结果是否有效？是否有更好的方法？"，然后根据反思结果调整策略。可通过简单的多轮LLM调用实现。

**Multi-Agent协作** -- 将任务分配给多个专业Agent并行处理。例如，一个"研究Agent"负责搜索信息，一个"分析Agent"负责数据处理，一个"写作Agent"负责生成报告。LangGraph框架对此有良好支持。

**Function Calling原生集成** -- 本实验使用文本解析来提取Action，但Qwen、DeepSeek等国产模型已支持原生的function calling/tool calling功能。建议在生产环境中使用原生API而非正则解析，以提升可靠性和降低解析错误率。

**记忆系统扩展** -- 使用Milvus向量数据库实现长期记忆，支持语义检索。结合sentence-transformers的bge-large-zh-v1.5嵌入模型，可实现高效的中文语义记忆检索。

**持续学习与适应** -- 将用户反馈（点赞/踩/修正）纳入Agent的记忆系统，通过RLHF（基于人类反馈的强化学习）持续优化Agent的决策策略。

### 9.2 推荐资源

- ReAct原始论文: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2023)
- 通义千问Function Calling文档: https://help.aliyun.com/zh/dashscope/
- DeepSeek API文档: https://platform.deepseek.com/api-docs/
- LangGraph官方教程: https://langchain-ai.github.io/langgraph/
- "Building AI Agents with LLMs" -- Anthropic Engineering Blog
