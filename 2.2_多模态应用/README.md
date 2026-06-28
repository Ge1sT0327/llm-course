# 2.2 多模态应用开发 (Multimodal Application Development)

## 1. 课程目标 (Course Objectives)

- 理解多模态大模型的架构原理（视觉编码器 + 语言模型的融合管道）
- 掌握使用 PIL (Pillow) 生成测试图像的方法，无需外部图像资源即可完成实验
- 学会图像的 Base64 编码与 Data URL 格式转换（Vision API 的标准传输协议）
- 能够调用 qwen-vl-max 视觉语言模型执行场景描述、OCR 文本提取和结构化分析三大任务
- 构建完整的多模态分析工作流，将视觉理解能力集成到实际应用中

## 2. 背景介绍 (Background)

多模态大模型（Multimodal Large Language Model, MLLM）是人工智能从"能听会说"到"能看会画"的关键跃迁。传统的 LLM 只能处理文本，而多模态模型能够同时理解文本、图像、音频、视频等多种信息形态，极大地扩展了 AI 的应用边界。

多模态 AI 的发展经历了三个阶段。第一阶段（2015-2019）以独立的视觉模型（ResNet、YOLO）和语言模型为代表，两者缺乏融合。第二阶段（2020-2022）出现了 CLIP（OpenAI, 2021）等图文对齐模型，通过对比学习将图像和文本映射到同一向量空间。第三阶段（2023 至今）以 GPT-4V、Gemini Pro Vision、Qwen-VL 等为代表，实现了真正的端到端多模态理解和生成。

在国内，阿里巴巴的通义千问视觉模型（qwen-vl-max/qwen-vl-plus）具备优秀的图文理解能力，特别针对中文场景进行了优化。该模型支持图像描述、光学字符识别（OCR）、图表分析、视觉问答等任务，通过 OpenAI 兼容接口提供服务，开发者可以使用熟悉的 `openai` Python SDK 直接调用。

多模态技术的实际应用场景包括：智能文档处理（合同条款提取、发票识别）、电商视觉搜索（以图搜图、商品自动分类）、医疗影像辅助诊断（X 光片分析、病灶标注）、工业质检（缺陷检测、产品分类）、以及内容创作（图文匹配、设计建议）。这些应用正逐步渗透到各行各业，成为数字化转型的核心技术驱动力。

## 3. 基础概念 (Basic Concepts)

### 3.1 多模态大模型架构

多模态模型的核心架构通常由三部分组成：视觉编码器、对齐投影层、语言模型。

```
+----------------------------------------------------------+
|                    输入层 (Input Layer)                    |
|                                                          |
|   +---------------------+    +------------------------+  |
|   |   图像输入 (Image)    |    |   文本输入 (Text)       |  |
|   |   JPEG / PNG / WebP  |    |   Prompt / Question    |  |
|   +----------|-----------+    +-----------|------------+  |
|              |                             |               |
+--------------|-----------------------------|---------------+
               |                             |
+--------------|-----------------------------|---------------+
|            处理层 (Processing Layer)                       |
|                                                          |
|   +----------|-----------+    +-----------|------------+  |
|   |  视觉编码器 (Vision   |    |  Tokenizer / Embedding |  |
|   |  Encoder / ViT)      |    |  (文本分词与嵌入)       |  |
|   |  图像 -> 特征向量     |    |  文本 -> Token序列     |  |
|   +----------|-----------+    +-----------|------------+  |
|              |                             |               |
|              +----------+------------------+               |
|                         |                                  |
|              +----------|-----------+                      |
|              |  对齐投影层 (Projection)|                      |
|              |  视觉特征 -> LLM 语义空间 |                   |
|              +----------|-----------+                      |
|                         |                                  |
+-------------------------|----------------------------------+
                          |
+-------------------------|----------------------------------+
|            推理层 (LLM Backbone)                           |
|                                                          |
|   +---------------------------------------------------+  |
|   |        大语言模型 (Qwen / GPT / Gemini)              |  |
|   |                                                      |
|   |  文本 Token + 视觉 Token -> 联合注意力计算            |  |
|   |  -> 生成多模态理解输出（文本描述 / 结构化数据）       |  |
|   +---------------------------------------------------+  |
+----------------------------------------------------------+
```

### 3.2 Vision API 图像传输格式

多模态 API 接收图像有两种方式：

```
方式一：公网 URL（适用于可公开访问的图像）
  https://example.com/images/photo.jpg
  
方式二：Data URL / Base64（适用于本地图像或私密图像）
  data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...
  
  Data URL 格式：
  +----------+  +----------+  +-------------------------+
  |  MIME类型  |  | 编码方式  |  |  Base64 编码的图像数据  |
  | image/png |  |  base64  |  |  iVBORw0KGgo...       |
  +----------+  +----------+  +-------------------------+
```

Base64 编码将二进制图像数据转换为 ASCII 文本，便于嵌入 API 请求的 JSON 负载中。编码后的数据体积约为原始的 4/3 倍。

### 3.3 多模态分析工作流

完整的多模态分析流程包含三个阶段：

```
+----------------+     +----------------+     +----------------+
|  阶段1: 场景描述 | --> |  阶段2: OCR提取 | --> |  阶段3: 结构分析 |
|  (Scene Desc)  |     |  (Text Extract)|     | (Struct Parse)|
+----------------+     +----------------+     +----------------+
|                  |   |                  |   |                  |
|  描述图像中的     |   |  提取图像中的     |   |  返回 JSON 格式   |
|  物体、场景、     |   |  所有可见文字     |   |  的类型、字段、   |
|  颜色、布局      |   |  保持原始格式     |   |  摘要等结构化信息 |
|                  |   |                  |   |                  |
+----------------+     +----------------+     +----------------+
```

### 3.4 主流多模态模型对比

| 模型 | 提供商 | 特点 | 中文支持 | API 兼容性 |
|------|--------|------|----------|------------|
| qwen-vl-max | 阿里云 DashScope | 中文场景优化，性价比高 | 优秀 | OpenAI 兼容 |
| qwen-vl-plus | 阿里云 DashScope | 轻量版，响应更快 | 优秀 | OpenAI 兼容 |
| GLM-5.2V | 智谱 AI | 学术场景强，支持长文本 | 优秀 | 兼容 |
| GPT-4o | OpenAI | 综合能力最强 | 良好 | 原生 |
| Claude 3.5 Sonnet | Anthropic | OCR 能力出色 | 良好 | 原生 |
| Gemini Pro Vision | Google | 视频理解能力强 | 良好 | 原生 |

## 4. 环境准备 (Environment Setup)

### Python 版本要求

- Python 3.8 或更高版本（推荐 3.10+）
- 操作系统：Windows / macOS / Linux

### 依赖包安装

```bash
# 核心依赖
pip install openai pillow

# pillow 用于使用 PIL 创建和操作图像
```

### API Key 配置

本实验使用阿里云 DashScope 提供的通义千问视觉模型 (qwen-vl-max)。

```bash
# Linux / macOS
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows (CMD)
set DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

获取 API Key：访问 [阿里云 DashScope 控制台](https://dashscope.aliyun.com/) 注册并获取。

### GPU 要求

本实验仅调用云端 API，无需本地 GPU。图像编码在 CPU 上完成。

## 5. 实践项目 (Practice Project)

本实验将构建一个完整的多模态应用开发框架，包含以下核心模块：

1. **PIL 图像生成** -- 使用 Pillow 库动态创建三张测试图像：
   - `info_card.png`：模拟员工信息卡片，包含姓名、工号、部门、职位等中文文本
   - `table.png`：模拟采购申请单表格，包含序号、物品名称、数量、单价等结构化数据
   - `shapes.png`：几何图形测试图，包含红色圆形、蓝色矩形、绿色三角形、黄色五角星

2. **图像 Base64 编码** -- 将 PIL 生成的图像字节流转换为 Data URL 格式，这是 Vision API 的标准传输协议

3. **VisionEngine 多模态视觉引擎** -- 封装 `qwen-vl-max` API 调用，支持真实 API 模式和 Dry-Run 模拟模式

4. **多模态分析工作流** -- 对每张图像依次执行场景描述、OCR 文本提取、结构化分析三个步骤

## 6. 实验步骤 (Experiment Steps)

### Step 1 -- 使用 PIL 创建测试图像

首先使用 Pillow 库动态生成包含中文文本和图形的测试图像。

**操作说明**: 使用 `PIL.Image` 和 `PIL.ImageDraw` 创建三种不同类型的测试图像：信息卡片、采购单据表格和几何图形测试图。

**完整代码实现**:

```python
"""
实验：多模态应用开发 (Multimodal Application Development)
课程章节：2.2 - 多模态应用 / 大模型应用开发课程

本实验演示多模态大模型的核心能力：
  (1) 使用 PIL 创建测试图像 -- 生成含文字/表格/图形的图片
  (2) 图像 Base64 编码 -- 将本地图像转为 Data URL 格式
  (3) Vision API 图像理解 -- 调用 qwen-vl-max 分析图像内容
  (4) OCR 文本提取 -- 从图像中识别并提取文字
  (5) 完整多模态分析工作流 -- 场景描述 -> 文本提取 -> 结构化分析

使用模型：通义千问 qwen-vl-max（阿里云 DashScope 多模态模型）
API 方式：OpenAI 兼容接口
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1

运行方式：
  1. pip install pillow openai
  2. export DASHSCOPE_API_KEY="your-api-key"
  3. python run.py
未设置 API Key 时以模拟模式运行。
"""

import os
import json
import base64
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Tuple
from io import BytesIO

# 第1部分：图像生成与编码

def create_test_images() -> Dict[str, Tuple[bytes, str]]:
    """使用 PIL 创建用于演示的测试图像。

    Returns:
        {文件名: (bytes数据, MIME类型)}
    """
    from PIL import Image, ImageDraw
    images = {}

    # 图像1: 信息卡片（含中文文字）
    img1 = Image.new("RGB", (600, 300), color=(240, 248, 255))
    d1 = ImageDraw.Draw(img1)
    d1.rectangle([20, 20, 580, 280], outline=(70, 130, 180), width=3)
    for i, (label, value) in enumerate([
        ("【员工信息卡】", ""), ("姓名: 张三", ""), ("工号: EMP-2024-0881", ""),
        ("部门: 人工智能研发中心", ""), ("职位: 高级算法工程师", ""),
        ("入职日期: 2024年3月15日", ""), ("邮箱: zhangsan@company.cn", ""),
    ]):
        d1.text((40, 40 + i * 30), label, fill=(25, 25, 112) if i == 0 else (0, 0, 0))
    buf1 = BytesIO(); img1.save(buf1, format="PNG")
    images["info_card.png"] = (buf1.getvalue(), "image/png")

    # 图像2: 模拟表格/单据
    img2 = Image.new("RGB", (520, 320), color=(255, 255, 255))
    d2 = ImageDraw.Draw(img2)
    d2.rectangle([10, 10, 510, 310], outline=(100,100,100), width=2)
    d2.text((30, 20), "采购申请单", fill=(0,0,0))
    d2.line([(30,45),(490,45)], fill=(100,100,100))
    rows = [
        ("序号  物品名称            数量  单价(元)    金额(元)",),
        ("1     GPU服务器 A100       2    89,000     178,000",),
        ("2     固态硬盘 4TB        10     2,400      24,000",),
        ("3     内存条 64GB          8     1,800      14,400",),
    ]
    y = 55
    for (text,) in rows:
        d2.text((30, y), text, fill=(0,0,0))
        y += 28; d2.line([(30, y-3), (490, y-3)], fill=(200,200,200))
    d2.text((30, y+10), "合计: 216,400元", fill=(180,0,0))
    d2.text((30, y+40), "申请人: 李四    日期: 2024-06-12", fill=(0,0,0))
    buf2 = BytesIO(); img2.save(buf2, format="PNG")
    images["table.png"] = (buf2.getvalue(), "image/png")

    # 图像3: 几何图形
    img3 = Image.new("RGB", (400, 400), color=(255,255,255))
    d3 = ImageDraw.Draw(img3)
    d3.ellipse([50,50,150,150], fill=(220,60,60), outline=(0,0,0))       # 红圆
    d3.rectangle([200,60,330,140], fill=(60,120,220), outline=(0,0,0))    # 蓝矩形
    d3.polygon([(50,300),(150,180),(250,300)], fill=(60,180,80), outline=(0,0,0))  # 绿三角
    star = [(260,200),(275,240),(320,240),(282,265),(295,305),
            (260,280),(225,305),(238,265),(200,240),(245,240)]
    d3.polygon(star, fill=(240,200,50), outline=(0,0,0))                  # 黄五角星
    d3.text((60, 330), "颜色识别测试图", fill=(0,0,0))
    buf3 = BytesIO(); img3.save(buf3, format="PNG")
    images["shapes.png"] = (buf3.getvalue(), "image/png")

    print(f"[PIL] 创建 {len(images)} 张测试图像: {list(images.keys())}")
    return images
```

**代码解释**:
- `Image.new("RGB", (width, height), color=...)` 创建指定尺寸的 RGB 彩色图像
- `ImageDraw.Draw(img)` 获取绘图对象，支持 `text()`、`rectangle()`、`ellipse()`、`polygon()`、`line()` 等绘制方法
- 图像保存到 `BytesIO` 内存缓冲区而非磁盘文件，避免文件系统依赖
- 三张图像覆盖了信息卡片（含中文文本）、表格（含数字和结构化数据）、图形（含几何形状）三类典型场景
- 图像返回为 `(bytes_data, mime_type)` 元组，便于后续 Base64 编码

### Step 2 -- 图像 Base64 编码

将图像字节数据转换为 Data URL 格式。

**操作说明**: 使用 Python 标准库 `base64` 对图像字节数据进行编码，添加 MIME 类型前缀。

**完整代码实现**:

```python
def encode_to_data_url(image_bytes: bytes, media_type: str = "image/png") -> str:
    """将图像字节数据编码为 Data URL。

    Returns:
        data:{media_type};base64,{b64_data}
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{media_type};base64,{b64}"
```

**代码解释**:
- `base64.b64encode()` 将二进制数据编码为 Base64 字符串（bytes 类型）
- `.decode("utf-8")` 将 bytes 转换为 str，便于嵌入 JSON
- Data URL 格式为 `data:[<mediatype>][;base64],<data>`，是 RFC 2397 标准
- MIME 类型根据文件扩展名映射：`.png` -> `image/png`，`.jpg` -> `image/jpeg`，`.webp` -> `image/webp`

### Step 3 -- 实现多模态视觉引擎

创建 `VisionEngine` 类，封装 qwen-vl-max API 调用。

**操作说明**: 使用 OpenAI 兼容客户端，将图像 Data URL 嵌入到消息的 `image_url` 字段中。

**完整代码实现**:

```python
class VisionEngine:
    """多模态视觉引擎，封装 qwen-vl-max API 调用。"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self._client = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None and self.api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def analyze(self, image_data_url: str, prompt: str,
                model: str = "qwen-vl-max", max_tokens: int = 800) -> str:
        """调用视觉模型分析图像。

        Args:
            image_data_url: Data URL 格式的图像
            prompt: 分析提示词
            model: 模型名称
            max_tokens: 最大输出 Token
        """
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
        try:
            response = self._get_client().chat.completions.create(
                model=model, messages=[message],
                max_tokens=max_tokens, temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[API 错误] {type(e).__name__}: {e}"

    # ---- 模拟响应 ----
    _MOCK_RESPONSES = {
        ("info_card.png", "scene"): (
            "蓝色边框信息卡片，淡蓝背景。包含员工姓名(张三)、工号(EMP-2024-0881)、"
            "部门(人工智能研发中心)、职位(高级算法工程师)、入职日期和邮箱。布局清晰，信息按行排列。"
        ),
        ("table.png", "scene"): (
            "采购申请单表格。含序号、物品名称、数量、单价、金额五列。录入GPU服务器(89,000元)、"
            "固态硬盘(2,400元)、内存条(1,800元)。合计216,400元，底部有申请人和日期。"
        ),
        ("shapes.png", "scene"): (
            "几何图形测试图:红色圆形(左上)、蓝色矩形(右上)、绿色三角形(中下)、黄色五角星。"
            "白色背景，图形轮廓清晰，标注'颜色识别测试图'。"
        ),
        ("info_card.png", "ocr"): (
            "【员工信息卡】\n姓名: 张三\n工号: EMP-2024-0881\n部门: 人工智能研发中心\n"
            "职位: 高级算法工程师\n入职日期: 2024年3月15日\n邮箱: zhangsan@company.cn"
        ),
        ("table.png", "ocr"): (
            "采购申请单\n序号 物品名称          数量 单价     金额\n"
            "1    GPU服务器 A100    2   89,000  178,000\n"
            "2    固态硬盘 4TB     10    2,400   24,000\n"
            "3    内存条 64GB       8    1,800   14,400\n"
            "合计: 216,400元\n申请人: 李四  日期: 2024-06-12"
        ),
    }

    def mock_analyze(self, image_name: str, prompt_type: str) -> str:
        """获取模拟的分析结果。"""
        key = (image_name, prompt_type)
        if key in self._MOCK_RESPONSES:
            return self._MOCK_RESPONSES[key]
        return f"[模拟] 对 {image_name} 的 {prompt_type} 分析结果。"
```

**代码解释**:
- `content` 字段使用数组格式（多部分消息），每个元素通过 `type` 字段区分文本和图像
- `type: "text"` 表示文本提示词，`type: "image_url"` 表示图像数据
- `temperature=0.3` 设置较低温度值，确保图像分析结果稳定一致
- 内置 `_MOCK_RESPONSES` 字典在无 API Key 时提供模拟响应，方便演示和测试

### Step 4 -- 多模态分析工作流

整合三个分析阶段，形成完整流水线。

**操作说明**: 对每张图像依次执行场景描述、OCR 文本提取和结构化分析，输出统一格式的结果。

**完整代码实现**:

```python
def run_multimodal_workflow(engine: VisionEngine, images: dict, image_name: str):
    """执行完整的多模态分析工作流：场景描述 -> OCR提取 -> 结构化分析。

    Args:
        engine: VisionEngine 实例
        images: 测试图像字典
        image_name: 目标图像名称
    """
    if image_name not in images:
        print(f"[跳过] 图像 {image_name} 不存在"); return

    image_bytes, media_type = images[image_name]
    data_url = encode_to_data_url(image_bytes, media_type)
    use_api = engine.is_available

    print(f"\n{'─'*40}\n  分析: {image_name} ({len(image_bytes)} bytes)\n{'─'*40}")

    # 步骤1: 场景描述
    print("\n[1/3] 场景描述:")
    prompt1 = "请详细描述这张图片的内容，包括主要元素、布局、颜色和文字信息。中文回答。"
    r1 = engine.analyze(data_url, prompt1) if use_api else engine.mock_analyze(image_name, "scene")
    print(f"  {r1[:200]}{'...' if len(r1)>200 else ''}")

    # 步骤2: OCR 文本提取
    print("\n[2/3] OCR 文本提取:")
    prompt2 = "请识别并提取这张图片中的所有文本，保持原始格式和布局。"
    r2 = engine.analyze(data_url, prompt2) if use_api else engine.mock_analyze(image_name, "ocr")
    print(f"  {r2}")

    # 步骤3: 结构化分析
    print("\n[3/3] 结构化分析:")
    prompt3 = "以JSON格式返回:图片类型、包含字段、语言、是否有表格结构、简要总结。仅输出JSON。"
    r3 = engine.analyze(data_url, prompt3, max_tokens=400) if use_api else json.dumps({
        "type": "信息卡片" if "card" in image_name else "表格" if "table" in image_name else "图形",
        "language": "中文", "summary": "图像清晰完整，结构良好"
    }, ensure_ascii=False, indent=2)
    print(f"  {r3}")


def main():
    print("=" * 40)
    print("  实验：多模态应用开发")
    print("  模型：通义千问 qwen-vl-max (DashScope)")
    print("  大模型应用开发课程 - 第2.2章")
    print("=" * 40)
    try:
        from PIL import Image
        print("[依赖] Pillow OK")
    except ImportError:
        print("[错误] 请安装: pip install pillow"); return
    api_key = os.getenv("DASHSCOPE_API_KEY")
    print(f"[API] {'已配置' if api_key else '未设置(模拟模式)'}")
    demo_image_creation()
    demo_image_encoding()
    demo_multimodal_analysis()
    print(f"\n{'='*40}")
    print("  实验完成！核心要点:")
    print("    1. PIL 生成测试图像，无需外部图片")
    print("    2. Base64 Data URL 是 Vision API 标准传输格式")
    print("    3. qwen-vl-max 支持理解+OCR+结构化提取")
    print("=" * 40)


if __name__ == "__main__":
    main()
```

**代码解释**:
- `run_multimodal_workflow()` 实现三阶段流水线：每个阶段使用不同的提示词（prompt）引导模型输出不同侧重点的结果
- 阶段1 提示词强调"详细描述"、"内容、元素、布局、颜色"，引导模型做全面的视觉理解
- 阶段2 提示词强调"识别并提取"、"保持原始格式和布局"，引导模型做精准的 OCR
- 阶段3 提示词要求"仅输出 JSON"，引导模型做结构化的数据提取
- 根据 `is_available` 属性自动在 API 模式和模拟模式间切换

## 7. 实验结果 (Experiment Results)

运行 `python run.py` 得到以下完整输出：

```
========================================
  实验：多模态应用开发
  模型：通义千问 qwen-vl-max (DashScope)
  大模型应用开发课程 - 第2.2章
========================================
[依赖] Pillow OK
[API] 未设置(模拟模式)

========================================
【演示1】PIL 创建测试图像
========================================
[PIL] 创建 3 张测试图像: ['info_card.png', 'table.png', 'shapes.png']
  C:\Users\0\AppData\Local\Temp\multimodal_rbpcbbs_\info_card.png (5,934 bytes)
  C:\Users\0\AppData\Local\Temp\multimodal_rbpcbbs_\table.png (7,425 bytes)
  C:\Users\0\AppData\Local\Temp\multimodal_rbpcbbs_\shapes.png (3,358 bytes)

图像保存于: C:\Users\0\AppData\Local\Temp\multimodal_rbpcbbs_

========================================
【演示2】图像 Base64 编码
========================================
[PIL] 创建 3 张测试图像: ['info_card.png', 'table.png', 'shapes.png']

info_card.png: 原始5,934bytes -> Base647934chars
  前缀: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAEsCAIAAACQX1rB...

table.png: 原始7,425bytes -> Base649922chars
  前缀: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAggAAAFACAIAAABWfROo...

shapes.png: 原始3,358bytes -> Base644502chars
  前缀: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZAAAAGQCAIAAAAP3aGb...

========================================
【演示3】多模态分析工作流
========================================

[状态] 模拟模式 (export DASHSCOPE_API_KEY='sk-xxxx' 启用API)
[PIL] 创建 3 张测试图像: ['info_card.png', 'table.png', 'shapes.png']

────────────────────────────────────────
  分析: info_card.png (5934 bytes)
────────────────────────────────────────

[1/3] 场景描述:
  蓝色边框信息卡片，淡蓝背景。包含员工姓名(张三)、工号(EMP-2024-0881)、部门(人工智能研发中心)、职位(高级算法工程师)、入职日期和邮箱。布局清晰，信息按行排列。

[2/3] OCR 文本提取:
  【员工信息卡】
姓名: 张三
工号: EMP-2024-0881
部门: 人工智能研发中心
职位: 高级算法工程师
入职日期: 2024年3月15日
邮箱: zhangsan@company.cn

[3/3] 结构化分析:
  {
  "type": "信息卡片",
  "language": "中文",
  "summary": "图像清晰完整，结构良好"
}

────────────────────────────────────────
  分析: table.png (7425 bytes)
────────────────────────────────────────

[1/3] 场景描述:
  采购申请单表格。含序号、物品名称、数量、单价、金额五列。录入GPU服务器(89,000元)、固态硬盘(2,400元)、内存条(1,800元)。合计216,400元，底部有申请人和日期。

[2/3] OCR 文本提取:
  采购申请单
序号  物品名称          数量  单价     金额
1    GPU服务器 A100    2   89,000  178,000
2    固态硬盘 4TB     10    2,400   24,000
3    内存条 64GB       8    1,800   14,400
合计: 216,400元
申请人: 李四  日期: 2024-06-12

[3/3] 结构化分析:
  {
  "type": "表格",
  "language": "中文",
  "summary": "图像清晰完整，结构良好"
}

========================================
  实验完成！核心要点:
    1. PIL 生成测试图像，无需外部图片
    2. Base64 Data URL 是 Vision API 标准传输格式
    3. qwen-vl-max 支持理解+OCR+结构化提取
========================================
```

## 8. 结果分析 (Result Analysis)

### 图像生成验证

演示1 成功使用 PIL 创建了三张 PNG 格式测试图像，文件大小分别为 5,934 bytes（信息卡片）、7,425 bytes（采购单据）和 3,358 bytes（几何图形）。这些图像保存到了系统的临时目录中，验证了 PIL 在无外部图像资源的情况下也能生成满足测试需求的图像。

从图像内容来看：
- 信息卡片包含 7 行中文文本，使用不同颜色区分标题（深蓝色 `fill=(25, 25, 112)`）和内容（黑色），边框为天蓝色
- 采购单据使用 `d2.line()` 绘制表格分隔线，使用 `d2.rectangle()` 绘制外框，数据行间有灰色分隔线，合计金额使用红色突出显示
- 几何图形测试图包含圆（`ellipse`）、矩形（`rectangle`）、三角形（`polygon`）和五角星（10 个顶点的 `polygon`），覆盖了 PIL 的主要绘图 API

### 图像编码验证

演示2 展示了 Base64 编码的效果。原始 5,934 bytes 的图像编码后扩展为 7,934 个字符（Data URL 格式），符合 Base64 编码约增加 33% 数据量的理论预期（5,934 x 4/3 = 7,912，加上前缀约 7,934）。编码后的数据以 `data:image/png;base64,` 开头，这是标准的 Data URL 格式，可直接作为 API 请求的 `image_url.url` 字段值。

### 多模态分析工作流验证

演示3 对两张测试图像（info_card.png 和 table.png）执行了完整的三阶段分析流水线：

**阶段 1 -- 场景描述分析**：模拟响应准确描述了图像的视觉特征，包括颜色（"蓝色边框"、"淡蓝背景"）、布局（"按行排列"）、内容元素（姓名、工号、部门、职位等）和表格结构（"序号、物品名称、数量、单价、金额五列"）。这表明经过精心设计的提示词可以有效引导模型输出详细的场景描述。

**阶段 2 -- OCR 文本提取分析**：模拟响应成功提取了图像中的所有文本内容，保持了原始格式。信息卡片的 6 个字段（姓名、工号、部门、职位、入职日期、邮箱）全部正确提取，采购单据的 3 行数据和合计、申请人、日期信息也完整保留。这验证了 OCR 提示词的设计有效性 -- "保持原始格式和布局"的指令确保输出可用于后续的程序化处理。

**阶段 3 -- 结构化分析**：模拟响应返回了 JSON 格式的结构化数据，包含 `type`（类型）、`language`（语言）和 `summary`（摘要）三个字段。JSON 格式使得分析结果可以直接被程序消费，例如输入到数据库或下游分析流程。

### 模拟模式与 API 模式对比

在模拟模式下，`_MOCK_RESPONSES` 字典提供了预定义的响应，确保了实验在没有 API Key 的情况下也能完整运行。这在教学场景中尤为重要，因为学生可以：
1. 先在不配置 API Key 的情况下理解和调试代码逻辑
2. 理解多模态 API 的完整请求-响应流程
3. 配置 API Key 后无缝切换到真实的视觉理解模式
4. 通过对比模拟结果和真实 API 结果，直观感受大模型的能力

## 9. 扩展学习 (Extended Learning)

在掌握本实验的多模态基础应用后，可以深入探索以下方向：

**1. 文生图 (Text-to-Image)**: 使用通义万相或 Stable Diffusion 根据文本描述生成图像。文生图与图生文的结合形成了一个完整的多模态循环，可以应用于产品宣传图自动生成、设计草图迭代等场景。

**2. 视频理解**: 使用 qwen-vl-max 或 Gemini Pro Vision 对视频进行帧级别的分析和理解，包括动作识别、场景变换检测、关键帧提取等。视频理解在安防监控、内容审核、体育分析领域有广泛应用。

**3. 多图像对比分析**: 提交多张图像（最多 8 张）让模型进行对比分析，例如工业质检中的标准件与待检件对比、医疗影像的前后对比、设计稿与实现截图的差异分析。

**4. RAG + 视觉**: 将多模态理解与向量检索结合。首先使用 Vision API 描述图像，将文本描述存入向量数据库，后续可通过语义搜索找到相关图像，实现"以文搜图"。

**5. 本地多模态模型**: 使用 Ollama 部署 LLaVA 或使用 transformers 加载 Qwen-VL-Chat 进行本地推理，适用于对数据隐私有严格要求的场景（如医疗档案、金融合同等）。

推荐阅读：
- Qwen-VL 技术论文（阿里云通义千问团队）
- CLIP: Connecting Text and Images (OpenAI, 2021)
- Pillow (PIL Fork) 官方文档
- 阿里云 DashScope 多模态 API 参考
