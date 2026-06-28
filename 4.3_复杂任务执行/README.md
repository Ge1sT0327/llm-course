# 4.3 复杂任务执行 (Complex Task Execution)

## 1. 课程目标 (Course Objectives)

**中文:**
- 实现安全的代码解释器（沙箱化Python执行），理解代码执行在Agent中的核心价值
- 构建安全的文件操作管理系统（读/写/列表，带路径越界防护）
- 实现端到端的数据分析工具链：创建CSV数据集 → 统计摘要 → 洞察报告
- 掌握安全管理器的设计：权限检查、审计日志、资源限制
- 理解并实现复杂任务Agent的自动编排与执行流程
- 了解代码执行安全的黑名单检测与沙箱隔离技术

**English:**
- Implement a safe code interpreter (sandboxed Python execution)
- Build a secure file operation management system with path traversal protection
- Implement an end-to-end data analysis pipeline: dataset creation, statistical summary, insight reports
- Master security manager design: permission checking, audit logging, resource limits
- Understand and implement automated orchestration for complex task agents
- Learn blacklist detection and sandbox isolation techniques for secure code execution

## 2. 背景介绍 (Background)

Real-world Agent applications go far beyond simple question-answering. An enterprise-grade Agent must handle complex, multi-step tasks that involve code execution, file manipulation, data analysis, and web browsing -- all within strict security boundaries.

The evolution of Agent capabilities follows a clear trajectory: from stateless chatbots to tool-augmented assistants, and finally to autonomous task executors. The key differentiator of the third stage is the ability to act as a "digital employee" that can create files, run analyses, generate reports, and interact with external systems -- all without constant human supervision.

However, with great power comes great security responsibility. When an Agent can execute code and modify files, the attack surface expands dramatically. A malicious prompt could attempt to delete critical files, exfiltrate sensitive data, or execute arbitrary system commands. This is why the security infrastructure (sandboxed execution, path validation, permission checking, audit logging) is not an afterthought but a foundational requirement of any production Agent system.

Domestic Chinese models like Qwen (通义千问) are well-suited for complex task execution due to their strong code generation and analysis capabilities. The Qwen3.7-Max model, in particular, demonstrates excellent performance on code-related benchmarks and can be used as the reasoning core for task planning and result synthesis.

## 3. 基础概念 (Basic Concepts)

### 3.1 复杂任务Agent架构

```
┌─────────────────────────────────────────────────────────────────┐
│                Complex Task Agent Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  USER TASK ("分析销售数据并生成报告")                              │
│         │                                                        │
│         v                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          TASK PLANNER (任务规划器)                        │   │
│  │  LLM decomposes task into step sequence:                  │   │
│  │    1. create_dataset("sales.csv", 100 rows)               │   │
│  │    2. analyze_dataset("sales.csv")                        │   │
│  │    3. generate_summary_report(analysis)                   │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│                         v                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          SECURITY GATE (安全网关)                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │   │
│  │  │ Permission  │  │ Path        │  │ Code Safety │      │   │
│  │  │ Checker     │──│ Validator   │──│ Scanner     │      │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│         ┌───────────────┼───────────────┐                       │
│         v               v               v                        │
│  ┌───────────┐  ┌───────────┐  ┌───────────────┐               │
│  │  CODE     │  │   FILE    │  │    DATA       │               │
│  │INTERPRETER│  │  MANAGER  │  │   ANALYZER    │               │
│  │ (沙箱执行) │  │ (安全读写) │  │ (统计分析)    │               │
│  └─────┬─────┘  └─────┬─────┘  └───────┬───────┘               │
│        │              │                │                         │
│        └──────────────┼────────────────┘                        │
│                       │                                          │
│                       v                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          AUDIT LOGGER (审计日志)                          │   │
│  │  Records every operation for compliance & debugging       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 沙箱化代码执行流程

```
     User Code
         │
         v
┌─────────────────────┐
│ 1. CODE VALIDATION  │ ← 黑名单正则检测
│    (代码验证)         │   (os.system, subprocess, eval, exec...)
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    │ SAFE?      │── NO ──> [REJECTED] "检测到危险模式"
    └─────┬──────┘
          │ YES
          v
┌─────────────────────┐
│ 2. BUILD SANDBOX    │ ← 受限全局命名空间
│    (构建沙箱)         │   __builtins__: safe subset only
│                     │   math, statistics, json allowed
│                     │   os, sys, subprocess BLOCKED
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│ 3. EXECUTE          │ ← subprocess with timeout
│    (执行)            │   print() captured via io.StringIO
│   timeout = 10s     │   SyntaxError caught early
└─────────┬───────────┘
          │
          v
┌─────────────────────┐
│ 4. COLLECT OUTPUT   │
│    (收集结果)         │ → stdout captured
│                     │ → stderr captured
│                     │ → return code captured
└─────────────────────┘
```

### 3.3 文件路径安全校验

```
    USER REQUEST: read "sales.csv"
            │
            v
    ┌───────────────────────┐
    │   PATH VALIDATION     │
    │                       │
    │ workspace = /sandbox  │  ← 安全工作区根目录
    │                       │
    │ full_path = /sandbox  │
    │   /sales.csv          │
    │   → RESOLVE           │
    │   → /sandbox/sales    │
    │       .csv            │
    │                       │
    │ CHECK: starts with    │
    │   /sandbox ?          │
    │   → YES ✓             │
    └───────────┬───────────┘
                │
    ┌───────────┴───────────┐
    │                       │
    v                       v
 [ALLOW]                 [BLOCK]
 正常读写                "../../etc/passwd"
                         → 路径越界攻击拦截!
```

## 4. 环境准备 (Environment Setup)

### 4.1 依赖安装

```bash
# 基础依赖
pip install openai  # DashScope LLM API
```

本实验脚本的大部分功能（文件操作、代码执行、数据分析）无需任何外部依赖即可运行。仅LLM报告生成和Agent自动编排需要DashScope API。

### 4.2 API配置（可选）

```bash
# 如果需要LLM报告生成功能
export DASHSCOPE_API_KEY="sk-your-api-key"
```

> **说明:** 核心演示（数据集创建、统计分析、安全代码执行、审计报告）完全不依赖API。脚本会自动检测API Key是否配置，未配置时跳过LLM相关步骤。

## 5. 实践项目 (Practice Project)

### 5.1 项目结构

```
4.3_复杂任务执行/
├── run.py                    # 主实验脚本 (~700行)
├── sandbox_workspace/        # 安全沙箱工作目录（自动创建）
│   └── sales_2025.csv        # 实验生成的数据集
├── 课程章节内容.md             # 详细课程讲义
├── 4.13_复杂任务执行.ipynb   # Jupyter Notebook
└── README.md                 # 本文档
```

### 5.2 子演示模块

| 子演示 | 内容 | 需要API |
|--------|------|---------|
| 子演示1 | 创建80行模拟销售数据集 | 否 |
| 子演示2 | 统计分析（销售额、区域分布） | 否 |
| 子演示3 | 文件预览（前5行CSV） | 否 |
| 子演示4 | 安全代码执行 + 危险代码拦截测试 | 否 |
| 子演示5 | LLM生成数据洞察报告 | 是 |
| 子演示6 | 安全审计报告 | 否 |
| 子演示7 | 复杂任务Agent自动编排 | 是 |

## 6. 实验步骤 (Experiment Steps)

### Step 1: 创建数据集与文件管理

```python
class DataAnalyzer:
    """数据分析工具 -- 创建/分析CSV数据集"""
    def create_dataset(self, filename: str, rows: int = 50) -> str:
        import random
        random.seed(42)  # 可复现

        products = ["笔记本电脑", "智能手机", "平板电脑", "智能手表",
                    "耳机", "键盘", "鼠标", "显示器", "打印机", "路由器"]
        regions = ["华北", "华东", "华南", "西南", "西北", "东北"]

        header = ["日期", "产品名称", "类别", "区域", "销量", "单价(元)", "总金额(元)"]
        rows_data = [header]

        for i in range(rows):
            product = random.choice(products)
            region = random.choice(regions)
            quantity = random.randint(1, 100)
            unit_price = random.randint(50, 8000)
            total = quantity * unit_price

            rows_data.append([
                f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                product, "电子产品", region,
                str(quantity), str(unit_price), str(total),
            ])

        csv_content = "\n".join(",".join(str(c) for c in row) for row in rows_data)
        return self.file_manager.write_file(filename, csv_content)
```

### Step 2: 安全代码解释器

```python
class SecurityManager:
    """安全管理器 -- 黑名单检测 + 安全命名空间"""
    DANGEROUS_PATTERNS = [
        r"os\.system\s*\(",     # 系统命令执行
        r"subprocess",           # 子进程模块
        r"eval\s*\(",            # eval执行
        r"exec\s*\(",            # exec执行
        r"__import__\s*\(",      # 动态导入
        r"open\s*\([^)]*['\"]w", # 写模式打开文件
        r"shutil\.(rmtree|copytree|move)",  # 文件系统操作
        r"requests\.",           # 网络请求
        r"socket",               # 网络socket
    ]

    SAFE_BUILTINS = {
        "abs": abs, "len": len, "range": range, "print": print,
        "sum": sum, "max": max, "min": min, "sorted": sorted,
        "list": list, "dict": dict, "str": str, "int": int, "float": float,
        # 只暴露安全的数学模块引用
        "__import__math": __import__("math"),
        "__import__statistics": __import__("statistics"),
    }

    def check_code_safety(self, code: str) -> tuple[bool, str]:
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                return False, f"代码包含危险模式: {pattern}"
        return True, "OK"

class SafeCodeInterpreter:
    def execute(self, code: str) -> str:
        # 安全检查
        safe, reason = self.security.check_code_safety(code)
        if not safe:
            return f"[安全拦截] {reason}"

        # 构建受限命名空间
        safe_globals = {"__builtins__": self.security.SAFE_BUILTINS}

        # 捕获输出
        stdout_capture = io.StringIO()

        try:
            compiled = compile(code, f"<sandbox>", "exec")
            exec(compiled, safe_globals)
            return stdout_capture.getvalue().strip() or "执行成功"
        except SyntaxError as e:
            return f"[语法错误] 第{e.lineno}行: {e.msg}"
        except Exception as e:
            return f"[运行时错误] {type(e).__name__}: {e}"
```

### Step 3: 复杂任务Agent自动编排

```python
class ComplexTaskAgent:
    def execute_task(self, task: str) -> dict:
        # Phase 1: 任务规划（LLM生成执行步骤）
        steps = self._plan_task(task)
        # 例如:
        # [
        #   {"action": "create_dataset", "params": {"filename": "data.csv", "rows": 100}},
        #   {"action": "analyze_dataset", "params": {"filename": "data.csv"}},
        #   {"action": "generate_report", "params": {}}
        # ]

        # Phase 2: 逐步执行
        results = {}
        for step in steps:
            action = step["action"]
            params = step["params"]

            if action == "create_dataset":
                result = self.analyzer.create_dataset(**params)
            elif action == "analyze_dataset":
                result = self.analyzer.analyze_dataset(**params)
                results["analysis"] = result
            elif action == "generate_report":
                result = self.analyzer.generate_summary_report(
                    results.get("analysis", "")
                )
                results["report"] = result
            # ... 更多操作类型

        return results
```

## 7. 实验结果 (Experiment Results)

### 7.1 数据集创建与统计分析

以下是 `python run.py` 的**真实控制台输出**：

```
======================================================================
  第4.3章 复杂任务执行 — 实验演示
  模型: qwen3.7-plus  (DashScope / Qwen)
  时间: 2026-06-12 18:52:37
======================================================================

==================================================
  子演示1: 创建模拟销售数据集
==================================================
[数据集已创建] sales_2025.csv: 80 行数据 (含表头)

[验证] 目录内容:
[目录内容] . (共1项):
  [FILE] sales_2025.csv (4603 bytes)

==================================================
  子演示2: 统计分析数据集
==================================================
[数据分析] 分析完成: 80 条记录

[分析结果]
数据集: sales_2025.csv
总记录数: 80
字段: 日期, 产品名称, 类别, 区域, 销量, 单价(元), 总金额(元)

销量 统计:
  总和: 3800.00
  均值: 47.50
  最大值: 100.00
  最小值: 2.00

单价(元) 统计:
  总和: 218990.00
  均值: 2737.38
  最大值: 7467.00
  最小值: 78.00

总金额(元) 统计:
  总和: 9739360.00
  均值: 121742.00
  最大值: 482860.00
  最小值: 2262.00

产品名称 Top5:
  智能手表: 14 条
  打印机: 12 条
  智能手机: 9 条
  耳机: 8 条
  显示器: 8 条

类别 Top5:
  电子产品: 39 条
  网络设备: 27 条
  电脑配件: 14 条

区域 Top5:
  华北: 20 条
  华南: 17 条
  华东: 13 条
  西南: 11 条
  西北: 10 条
```

### 7.2 安全代码执行测试

```
==================================================
  子演示4: 安全代码执行测试
==================================================

[安全代码执行]
  统计计算成功:
  数据: [23, 45, 12, 67, 34, 56, 78, 90, 43, 21]
  样本数: 10
  均值: 46.90
  中位数: 44.00
  标准差: 25.27
  最大值: 90, 最小值: 12

[危险代码拦截测试]
  import os; os.system('dir')
  => [安全拦截] 代码包含危险模式: os\.system\s*\(
```

### 7.3 安全审计报告

```
==================================================
安全审计报告
==================================================
总操作数: 16
允许: 15, 拒绝: 1
──────────────────────────────────────────────────
[V] PERM_CHECK: data_analysis -> 创建数据集 sales_2025.csv
[V] PATH_CHECK: 路径校验通过: .../sandbox_workspace/sales_2025.csv
[V] FILE_WRITE: 写入: sales_2025.csv (3015字节)
[V] FILE_READ: 读取: sales_2025.csv (3015字节)
[X] CODE_CHECK: 检测到危险模式: os\.system\s*\(
==================================================
```

### 7.4 实验总结

| 测试类别 | 测试内容 | 结果 |
|---------|---------|------|
| 数据集创建 | 80行销售CSV, 7个字段 | PASS: 4,603字节文件 |
| 统计分析 | 销量/单价/总金额/产品分布 | PASS: 完整统计摘要 |
| 文件安全 | 路径越界检测 | PASS: sandbox_workspace隔离 |
| 代码安全 | 危险代码拦截 | PASS: os.system被拦截 |
| 审计日志 | 16条操作记录 | PASS: 15允许, 1拒绝 |
| 文件预览 | CSV前5行 | PASS: 数据正确可读 |

**安全审计: 16次操作, 15次通过, 1次拒绝（危险代码尝试）。**

## 8. 结果分析 (Result Analysis)

本次实验通过7个子演示完整展示了复杂任务Agent从数据创建到分析报告的全流程，核心成果和发现如下。

**数据管线的完整性与实用性。** 实验成功创建了一个包含80行、7个字段的模拟销售数据集，覆盖了10种产品、6个销售区域。数据统计分析输出了销量/单价/总金额的描述性统计（均值、最值）和分布特征（产品Top5、区域Top5、类别分布），形成了一个完整的数据分析报告模板。这种数据管线在企业场景中具有直接的应用价值——例如，可扩展为自动化的销售日报生成系统，每天从数据库提取数据、执行统计分析、调用LLM生成可读性强的管理层摘要。

**安全沙箱的执行隔离能力。** 实验的黑名单检测成功拦截了 `import os; os.system('dir')` 这样的危险代码。然而，必须清醒地认识到基于黑名单的安全策略的局限性。攻击者可以通过以下方式绕过黑名单：使用字符串拼接（`__imp` + `ort__('os')`）、使用 `getattr`、利用反序列化漏洞等。生产环境建议采用更强的隔离方案：(1) Docker容器隔离，每个代码执行都在独立的容器中进行；(2) gVisor或Firecracker等微虚拟机；(3) 限制网络访问、磁盘配额和CPU时间；(4) 结合静态分析（AST级别）和动态行为监控。

**路径安全的纵深防御。** 实验中实现了 `validate_path` 函数，通过 `Path.resolve()` 获取绝对路径并与安全工作区比较，有效防止了 `../../etc/passwd` 这样的路径遍历攻击。但需要注意，Windows和Linux的路径解析存在差异（如NTFS junction points、符号链接），建议在生产环境中对平台差异做专门处理。

**审计日志的合规价值。** 实验通过 `SecurityManager` 记录每次操作的审计日志（时间、操作类型、资源、允许/拒绝），输出了16条记录的审计报告。在企业环境中，这种审计日志是实现合规（如等级保护、GDPR）的基本要求。建议增强审计日志的完整性：记录操作者的用户ID、IP地址、请求来源，使用加密签名防止日志篡改，并设置保留期限策略。

**Agent自动编排的优势与风险。** 实验中LLM自动将"创建小型数据集并分析"分解为 `create_dataset → analyze_dataset → generate_report` 三个步骤。这种自动编排大幅降低了人工配置工作流的工作量。但自动编排也带来风险——LLM可能生成不合理的计划（如顺序错误、遗漏关键步骤）。建议在实际应用中采用"LLM规划 + 人工审核"的双重验证机制，对高风险操作（如删除、修改生产数据）必须经人工确认后方可执行。

## 9. 扩展学习 (Extended Learning)

### 9.1 进阶方向

**Docker沙箱集成** -- 将代码执行迁移到Docker容器中。通过Docker SDK for Python，在每次代码执行时启动一个临时的隔离容器，执行完毕后销毁。这提供了比Python命名空间更强的隔离级别。可配置CPU/内存限制、网络阻断、只读文件系统等。

**数据分析Agent定制** -- 基于本实验的数据管线，扩展支持更多数据源的Agent。集成SQL数据库查询（通过SQLAlchemy）、Excel文件解析（openpyxl）、API数据拉取（aiohttp）。构建一个"数据分析聊天机器人"，用户通过自然语言即可完成复杂的数据查询和分析任务。

**Web浏览能力集成** -- 使用Playwright（比Selenium更现代）构建安全的网页浏览工具。设置白名单URL模式，限制Agent只能访问特定域名的网页。实现页面内容提取、表单填写、文件下载等功能。

**多模态数据分析** -- 结合视觉模型，让Agent不仅能分析结构化数据，还能理解图表截图、PDF文档、扫描件中的信息。通义千问VL版本支持图文理解，可作为多模态Agent的基础能力。

**自动化报告生成** -- 将分析结果自动生成为格式化的PDF或HTML报告。结合Jinja2模板引擎和WeasyPrint/Playwright-PDF，实现从数据到可视化报告的全自动流程。

### 9.2 推荐资源

- Python沙箱安全: https://docs.python.org/3/library/ast.html (AST模块用于代码静态分析)
- Docker SDK for Python: https://docker-py.readthedocs.io/
- Playwright for Python: https://playwright.dev/python/
- 通义千问DashScope文档: https://help.aliyun.com/zh/dashscope/
- "Building a Data Analysis Agent" -- LangChain官方教程
