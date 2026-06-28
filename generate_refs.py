"""批量生成 references/ 真实参考资料"""
import os

COURSE_ROOT = r"G:\CLAUDE CODE_PROJECT\SHIXI\course"

# ===== 每门课的详细参考资料 =====
REFS = {
    "1.1_大模型技术概览": """# 参考资料 - 大模型技术概览

## 必读论文
- **Attention Is All You Need** (Vaswani et al., 2017) — Transformer原始论文. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Language Models are Few-Shot Learners** (Brown et al., 2020) — GPT-3, 首次论证涌现能力. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- **Training language models to follow instructions** (Ouyang et al., 2022) — RLHF奠基工作. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

## 2026年最新技术报告
- **DeepSeek-V4 Technical Report** (2026.4) — 1.6T MoE, 昇腾910C全栈训练. [github.com/deepseek-ai](https://github.com/deepseek-ai)
- **Qwen3.7 Technical Report** (2026.5) — 235B-A22B MoE, 1M ctx, 多模态. [github.com/QwenLM](https://github.com/QwenLM/Qwen3)
- **GLM-5.2** (2026.6) — 自研GLM架构, Agent T0. [github.com/THUDM](https://github.com/THUDM)

## 视频资源
- Stanford CS324: Large Language Models — [cs324.stanford.edu](https://cs324.stanford.edu)
- Karpathy: Intro to LLMs (1h) — [YouTube](https://www.youtube.com/watch?v=zjkBMFhNj_g)
- 李宏毅 2025 生成式AI — [B站搜索](https://search.bilibili.com)

## 工具
- HuggingFace: [huggingface.co/models](https://huggingface.co/models)
- ModelScope: [modelscope.cn](https://modelscope.cn)
""",

    "1.2_模型生态与选型": """# 参考资料 - 模型生态与选型

## 2026年6月模型对比
- **Open LLM Leaderboard v3**: [huggingface.co/spaces/open-llm-leaderboard](https://huggingface.co/spaces/open-llm-leaderboard)
- **AA Intelligence Index** (2026.6): Qwen3.7-Max #1 (57分). [aiancestral.com](https://aiancestral.com)
- **SWE-bench Verified**: DeepSeek V4 Pro 80.6%

## API平台
- DeepSeek: [platform.deepseek.com/api-docs](https://platform.deepseek.com/api-docs)
- Qwen3.7 (DashScope): [dashscope.aliyun.com](https://dashscope.aliyun.com)
- GLM-5.2: [open.bigmodel.cn](https://open.bigmodel.cn)
- Kimi K2.7: [platform.moonshot.cn](https://platform.moonshot.cn)
- 豆包 Seed 2.0: [console.volcengine.com/ark](https://console.volcengine.com/ark)

## 开源模型
- Qwen3.7: [huggingface.co/Qwen](https://huggingface.co/Qwen)
- DeepSeek V4: [huggingface.co/deepseek-ai](https://huggingface.co/deepseek-ai)
- GLM-5.2: [huggingface.co/THUDM](https://huggingface.co/THUDM)

## 选型指南
- [Qwen vs DeepSeek Full Comparison (2026)](https://qwen-ai.com/vs-deepseek/)
- [国产大模型全面崛起 2026](https://post.smzdm.com/p/a26nxvw7/)
- [Open Source LLMs 2026 Guide](https://www.morphllm.com/open-source-llm)
""",

    "1.4_Prompt_Engineering": """# 参考资料 - Prompt Engineering

## 必读论文
- **Chain-of-Thought Prompting** (Wei et al., 2022) — CoT开创. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- **Self-Consistency Improves CoT** (Wang et al., 2023) — 投票推理. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- **Tree of Thoughts** (Yao et al., 2023) — 多路径思维树. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)

## 2026推荐
- **The Prompt Report** (Schulhoff et al., 2024) — 全面综述
- **OWASP Top 10 for LLM** (2025) — Prompt注入防御
- Anthropic Prompt Engineering Guide: [docs.anthropic.com](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)

## 免费课程
- DeepLearning.AI: ChatGPT Prompt Engineering (1h)
- Anthropic: Prompt Engineering Interactive Tutorial
""",

    "2.3_函数调用与工具使用": """# 参考资料 - Function Calling 与工具使用

## 核心文档
- OpenAI Function Calling: [platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
- Anthropic Tool Use: [docs.anthropic.com/en/docs/build-with-claude/tool-use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- **MCP Specification** (2026): [modelcontextprotocol.io](https://modelcontextprotocol.io)

## 2026工具生态
- MCP Python SDK v2 (beta): `pip install mcp[cli]==2.0.0a1`
- 10,000+ MCP公开服务器, 97M月下载量

## 论文
- **ReAct: Synergizing Reasoning and Acting** (Yao et al., 2023) — [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
""",

    "3.1_向量检索基础": """# 参考资料 - 向量检索基础

## Embedding模型
- **BGE-M3** (BAAI, 2024): 多语言, 8192 tokens. [huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- **bge-small-zh-v1.5**: 轻量中文, 512维. [huggingface.co/BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)

## 向量数据库
- FAISS: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
- Milvus: [milvus.io](https://milvus.io)
- ChromaDB: [trychroma.com](https://www.trychroma.com)

## 教程
- Pinecone: Vector Search Fundamentals
- DeepLearning.AI: Vector Databases Course
""",

    "3.3_高级RAG架构": """# 参考资料 - 高级RAG架构

## 必读论文
- **Self-RAG** (Asai et al., 2023): [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- **Corrective RAG** (Yan et al., 2024): [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)
- **RRF Fusion** (Cormack et al.): [ACM](https://dl.acm.org/doi/10.1145/3340531.3412034)

## RAG评估
- RAGAS: [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas)
- DeepEval: [github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval)

## 2026前沿
- Microsoft GraphRAG: [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)
- BGE-Reranker-v2-m3: [huggingface.co/BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
""",

    "4.1_Agent设计模式": """# 参考资料 - Agent设计模式

## 必读论文
- **ReAct** (Yao et al., 2023): [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- **Plan-and-Solve** (Wang et al., 2023): [arXiv:2305.04091](https://arxiv.org/abs/2305.04091)
- **AutoGen** (Microsoft, 2023): [arXiv:2308.08155](https://arxiv.org/abs/2308.08155)

## 2026书籍与课程
- **AI Agents and Applications** (Manning, 2026.3) — LangChain+LangGraph+MCP
- **MCP Bootcamp** (O'Reilly, 2026) — 2天实战
- Hugging Face MCP Course (免费证书): [huggingface.co/learn/mcp-course](https://huggingface.co/learn/mcp-course)

## 框架
- LangGraph: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- AutoGen: [microsoft.github.io/autogen](https://microsoft.github.io/autogen)
- CrewAI: [docs.crewai.com](https://docs.crewai.com)
""",

    "4.2_框架实战": """# 参考资料 - 框架实战 (LangChain & LangGraph)

## 官方文档
- LangChain: [python.langchain.com](https://python.langchain.com)
- LangGraph: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- MCP Python SDK: [github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)

## 2026书籍
- **AI Agents and Applications** (Manning, 2026) — with LangChain, LangGraph and MCP
- **Building Reliable AI Systems** (Manning, 2026) — Ch7: MCP Integration

## 教程
- DeepLearning.AI: LangChain for LLM Application Development
- DeepLearning.AI: Functions, Tools and Agents with LangChain
""",

    "5.1_高效微调技术": """# 参考资料 - 高效微调技术

## 必读论文
- **LoRA** (Hu et al., 2021): [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **QLoRA** (Dettmers et al., 2023): [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **DPO** (Rafailov et al., 2023): [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

## 微调框架
- LLaMA-Factory: [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- SWIFT (阿里): [github.com/modelscope/swift](https://github.com/modelscope/swift)
- Unsloth (2-5x加速): [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)

## 数据集
- Alpaca中文: [github.com/hikariming/alpaca_chinese_dataset](https://github.com/hikariming/alpaca_chinese_dataset)
- BELLE: [github.com/LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)

## 模型
- Qwen3.7 (微调基座): [huggingface.co/Qwen](https://huggingface.co/Qwen)
""",

    "5.2_私有化部署": """# 参考资料 - 私有化部署

## 部署工具
- vLLM: [docs.vllm.ai](https://docs.vllm.ai) — 高性能LLM推理引擎
- Ollama: [ollama.com](https://ollama.com) — 一键本地部署
- Xinference: [github.com/xorbitsai/inference](https://github.com/xorbitsai/inference)
- SGLang: [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)

## 量化工具
- bitsandbytes: [github.com/bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- GPTQ: [github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
- AWQ: [github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)

## 2026推荐
- vLLM 0.8+ 支持DeepSeek V4 MoE推理优化
- Ollama 0.6+ 国产模型一键部署
""",

    "6.1_性能优化": """# 参考资料 - 性能优化

## 核心技术
- KV Cache优化: FlashAttention-3 (2024)
- 模型量化: AWQ, GPTQ, bitsandbytes 4-bit
- 批处理: Continuous Batching (vLLM)

## 缓存策略
- GPTCache: [github.com/zilliztech/GPTCache](https://github.com/zilliztech/GPTCache)
- Semantic Cache with Redis + Embedding

## 2026优化框架
- FlashMLA (DeepSeek): 35x推理加速
- vLLM Prefix Caching: 自动复用KV Cache
""",

    "6.2_安全与合规": """# 参考资料 - 安全与合规

## 安全框架
- **OWASP Top 10 for LLM Applications** (2025版): [owasp.org](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- NIST AI Risk Management Framework: [nist.gov](https://www.nist.gov/itl/ai-risk-management-framework)

## 注入防御
- Prompt Injection Detection: regex + 语义分类器
- Input Sanitization: 多层过滤
- Output Validation: LLM-as-Judge二次审查

## 隐私合规
- GDPR Art.22: 自动化决策权
- 《生成式人工智能服务管理暂行办法》(2023)
- 《个人信息保护法》PIPL
""",

    "6.3_项目实战": """# 参考资料 - 项目实战

## 开源项目参考
- RAGFlow: [github.com/infiniflow/ragflow](https://github.com/infiniflow/ragflow)
- Dify: [github.com/langgenius/dify](https://github.com/langgenius/dify)
- FastGPT: [github.com/labring/FastGPT](https://github.com/labring/FastGPT)
- MaxKB: [github.com/1Panel-dev/MaxKB](https://github.com/1Panel-dev/MaxKB)

## 生产工具链
- Langfuse (可观测性): [langfuse.com](https://langfuse.com)
- LangSmith (调试追踪): [smith.langchain.com](https://smith.langchain.com)
- Prometheus + Grafana (监控)

## 评估
- RAGAS: [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas)
- DeepEval: [github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval)
""",
}

# 通用模板 (剩余课程)
DEFAULT_TOPICS = {
    "1.3_开发环境搭建": ["Conda/Venv管理", "CUDA/cuDNN兼容性", "HuggingFace镜像", "API Key安全", "AutoDL云GPU"],
    "2.1_对话系统开发": ["Streaming API", "Context Window", "Conversation Memory", "Chat Completions API"],
    "2.2_多模态应用": ["Qwen-VL API", "CLIP", "OCR", "PIL/Pillow", "Base64 Encoding"],
    "3.2_文档处理流水线": ["Document Chunking", "Semantic Splitting", "LangChain Loaders", "Text Cleaning"],
    "4.3_复杂任务执行": ["Code Interpreter", "Sandboxed Execution", "Security Audit", "Multi-step Planning"],
}

for dirname, topics in DEFAULT_TOPICS.items():
    REFS[dirname] = f"""# 参考资料 - {dirname.split("_", 1)[1]}

## 核心主题
{chr(10).join(f'- {t}' for t in topics)}

## 推荐资源
- 相关论文: [arXiv.org](https://arxiv.org) 搜索 `{", ".join(topics[:2])}`
- DeepLearning.AI 免费短课: [deeplearning.ai/short-courses](https://www.deeplearning.ai/short-courses/)
- Hugging Face 教程: [huggingface.co/learn](https://huggingface.co/learn)

## 2026年最新
- **AI Model Roundup June 2026**: [dev.to](https://dev.to/doremonai/ai-model-roundup-june-2026s-biggest-releases-you-need-to-know-355i)
- **MCP协议**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
"""

# 写入所有文件
for dirname, content in REFS.items():
    ref_path = os.path.join(COURSE_ROOT, dirname, "references", "README.md")
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    with open(ref_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"OK: {dirname}/references/README.md")

print(f"\nDone! {len(REFS)} courses updated.")
