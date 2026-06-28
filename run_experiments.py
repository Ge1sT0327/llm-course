"""
手动实验运行脚本 - 逐步执行, 每步保存结果
用法: python run_experiments.py 1    (运行实验1)
       python run_experiments.py all  (运行全部)

每个实验独立, 不会因为一个失败就崩溃
"""
import sys, os, json

sys.path.insert(0, ".")
COURSE_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_1_4():
    """1.4 Prompt Engineering - 提示词模板 + A/B评估 + 结构化输出"""
    from config import get_client
    client = get_client()

    print("=" * 60)
    print("  1.4 Prompt Engineering - Prompt Template Factory")
    print("=" * 60)

    # Step 1: Templates
    templates = {
        "system": "你是智能知识库助手。基于文档回答,引用来源。不确定时标注[存疑]。",
        "rag_qa": "知识库:\n{context}\n\n问题: {question}\n请引用来源编号。",
        "structured": "你只输出纯JSON。不要Markdown标记。格式: {\"answer\": \"...\", \"confidence\": N}",
    }
    print(f"[Step1] {len(templates)} templates loaded")

    # Step 2: A/B test
    print("\n[Step2] A/B Testing:")
    cases = [
        {"name": "知识问答", "question": "什么是MCP协议?",
         "context": "MCP(Model Context Protocol)是AI Agent标准化协议,Linux Foundation托管。"},
        {"name": "无相关内容", "question": "今天天气?",
         "context": "(无相关文档)"},
    ]
    for tc in cases:
        r = client.chat(
            system=templates["system"],
            messages=[{"role": "user", "content": templates["rag_qa"].format(**tc)}],
            temperature=0.3, max_tokens=200
        )
        print(f"  [{tc['name']}] Q: {tc['question']}")
        print(f"    A: {r['content'][:120]}...")
        print(f"    tokens: {r['usage']['output']}")

    # Step 3: Structured output
    print("\n[Step3] Structured JSON:")
    r = client.chat(
        system=templates["structured"],
        messages=[{"role": "user", "content": "分析:这个产品很好用但有点贵。输出JSON。"}],
        temperature=0.1, max_tokens=150
    )
    print(f"  {r['content']}")

    # Save
    os.makedirs("prompts", exist_ok=True)
    with open("prompts/templates.json", "w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)
    print(f"\n[Done] Templates saved -> prompts/templates.json")

def run_2_3():
    """2.3 Tool Registry + Function Calling"""
    from config import get_client
    from agent_project.tools import create_default_registry
    client = get_client()
    tools = create_default_registry()

    print("=" * 60)
    print("  2.3 Function Calling - Tool Registry")
    print("=" * 60)

    print(f"[Tools] {tools.list_tools()}")

    queries = ["Search MCP protocol definition", "Calculate 25*4+100/5", "What time is it?", "Hello, how are you?"]
    for q in queries:
        r = client.chat(
            system="You are an assistant. Use tools when needed.",
            messages=[{"role": "user", "content": q}],
            tools=tools.get_openai_schemas(),
            temperature=0.1, max_tokens=200
        )
        if r.get("tool_calls"):
            tc = r["tool_calls"][0]
            result = tools.execute(tc["name"], tc["arguments"])
            print(f"  [{tc['name']}] {q} -> {result[:60]}")
        else:
            print(f"  [direct] {q} -> {r['content'][:60]}")

def run_3_x():
    """3.x RAG Pipeline - Document ingestion + Hybrid search"""
    from agent_project.pipeline import DocumentPipeline
    from agent_project.hybrid_search import HybridSearcher
    from sentence_transformers import SentenceTransformer

    print("=" * 60)
    print("  3.x RAG Pipeline - Hybrid Search")
    print("=" * 60)

    knowledge = [
        "Transformer uses Self-Attention mechanism for parallel sequence processing (Vaswani et al., 2017).",
        "RAG combines retrieval and generation. 2026 standard: Hybrid+RRF+Rerank+Self-RAG.",
        "DeepSeek V4 (Apr 2026): 1.6T MoE, 37B active, SWE-bench 80.6%, $0.87/M output tokens.",
        "Qwen3.7-Max (May 2026): 235B MoE, 1M context, multimodal, AA Index #1.",
        "MCP is AI Agent standard protocol by Linux Foundation. 10,000+ servers, 97M monthly SDK downloads.",
        "LoRA trains low-rank matrices for efficient fine-tuning. Only 0.09% params trainable.",
        "Hybrid search = BM25 keyword + Vector semantic, RRF fusion, Cross-Encoder rerank.",
    ]

    pipeline = DocumentPipeline(chunk_size=300, chunk_overlap=50)
    pipeline.ingest_texts([{"source": f"doc{i}.md", "content": k} for i, k in enumerate(knowledge)])
    stats = pipeline.get_stats()
    print(f"[Pipeline] {stats['documents']} docs -> {stats['total_chunks']} chunks")

    emb = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    searcher = HybridSearcher(emb)
    searcher.index(pipeline.all_chunks)

    for q in ["What is MCP?", "DeepSeek V4 details", "Best RAG approach?"]:
        results = searcher.search(q, top_k=3)
        a = searcher.self_assess(results)
        print(f"  [{a['grade']}] {q}")
        print(f"    Top: [{results[0]['score']:.4f}] {results[0]['content'][:80]}...")

def run_4_1():
    """4.1 ReAct Agent"""
    from config import get_client
    from agent_project.agent import ReActAgent
    from agent_project.tools import create_default_registry

    client = get_client()
    tools = create_default_registry()

    print("=" * 60)
    print("  4.1 ReAct Agent")
    print("=" * 60)

    react = ReActAgent(client, tools, max_steps=5, verbose=True)
    answer = react.run("Search for MCP protocol info, then calculate sqrt(256)")
    print(f"\n[Final] {answer[:300]}")

def run_6_3():
    """6.3 Final Project - KnowledgeBaseAgent"""
    from config import get_client
    from agent_project.app import KnowledgeBaseAgent
    from sentence_transformers import SentenceTransformer

    print("=" * 60)
    print("  6.3 Final Project - KnowledgeBaseAgent")
    print("=" * 60)

    client = get_client()
    emb = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    knowledge = [
        "Transformer: Self-Attention mechanism, parallel processing (Vaswani 2017).",
        "RAG: Retrieval + Generation. 2026: Hybrid+RRF+Rerank+Self-RAG. Reduces hallucination.",
        "DeepSeek V4: Apr 2026, 1.6T MoE(37B active), SWE-bench 80.6%, ~$1/1M tokens API cost.",
        "Qwen3.7-Max: May 2026, 235B MoE, 1M context, multimodal, AA Index #1 ranked.",
        "MCP: AI Agent protocol, Linux Foundation(Dec 2025). 3 primitives: Tool/Resource/Prompt. 10K+ servers.",
        "LoRA: Low-Rank Adaptation, trains ~0.1% params. RTX 3070 8GB can fine-tune 1.7B model.",
        "Hybrid Search: BM25 + Vector, RRF fusion, Cross-Encoder reranker. RAG standard in 2026.",
    ]

    agent = KnowledgeBaseAgent(client, emb, knowledge)
    print(f"[Agent] Tools: {agent.tools.list_tools()}")

    for q in [
        "What is MCP protocol?",
        "Tell me about DeepSeek V4",
        "What is the best RAG architecture in 2026?"
    ]:
        r = agent.ask(q, verbose=False)
        print(f"\n  Q: {q}")
        print(f"  A: {r['answer'][:250]}...")
        print(f"  Sources: {len(r['sources'])} | Grade: {r['assessment']['grade']}")

    stats = agent.get_stats()
    print(f"\n[Stats] Turns: {stats['turn_count']} | Audits: {stats['audit_entries']}")

# ===== Main =====
EXPERIMENTS = {
    "1": ("1.4 Prompt Engineering", run_1_4),
    "2": ("2.3 Function Calling", run_2_3),
    "3": ("3.x RAG Pipeline", run_3_x),
    "4": ("4.1 ReAct Agent", run_4_1),
    "5": ("6.3 Final Project", run_6_3),
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        for k, (name, _) in EXPERIMENTS.items():
            print(f"  python run_experiments.py {k}    # {name}")
        print(f"  python run_experiments.py all    # 全部运行")
        sys.exit(0)

    choice = sys.argv[1]
    if choice == "all":
        for k, (name, func) in EXPERIMENTS.items():
            print(f"\n{'#' * 60}")
            print(f"# Experiment {k}: {name}")
            print(f"{'#' * 60}\n")
            try:
                func()
            except Exception as e:
                print(f"[ERROR] {e}")
    elif choice in EXPERIMENTS:
        name, func = EXPERIMENTS[choice]
        func()
    else:
        print(f"无效选择: {choice}. 可选: {list(EXPERIMENTS.keys())}")
