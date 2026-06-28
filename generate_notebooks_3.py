"""Generate remaining 5 notebooks: 3.2, 3.3, 4.2, 5.1, 5.2"""
import json, os

ROOT = r"G:\CLAUDE CODE_PROJECT\SHIXI\course"

def nb(cells):
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}

def md(s):
    return {"cell_type":"markdown","metadata":{},"source":[s]}

def code(lines):
    return {"cell_type":"code","metadata":{},"outputs":[],"source":[l+"\n" for l in lines]}

SETUP = code([
    'import sys, os',
    'sys.path.insert(0, "..")',
    'from config import get_client; client = get_client()',
    'from agent_project import *',
    'from agent_project.tools import create_default_registry',
    'print(f"LLM: {client.name} | modules loaded")',
])

def save(d, f, c):
    p = os.path.join(ROOT, d, f)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(nb(c), fh, ensure_ascii=False, indent=1)
    print(f"OK: {d}/{f}")

# ===== 3.2 Document Pipeline =====
save("3.2_文档处理流水线", "实验_文档处理流水线.ipynb", [
    md("# 3.2 Document Pipeline - Build the RAG ingestion engine"),
    md("## Project Role: DocumentPipeline feeds documents into HybridSearcher for the final project"),
    SETUP,
    code([
        'from agent_project.pipeline import DocumentPipeline',
        'pipeline = DocumentPipeline(chunk_size=300, chunk_overlap=50)',
        '',
        'KNOWLEDGE = [',
        '  {"source":"ai_basics.md","content":"# AI Basics\\nAI creates intelligent systems. ML trains models from data. DL uses multi-layer neural networks. CNNs process images, RNNs/Transformers handle sequences."},',
        '  {"source":"llm_guide.md","content":"# LLM Guide\\nLLMs have billions of parameters. Transformer by Vaswani et al. (2017). Self-Attention enables full parallelization. Training: Pretrain->SFT->RLHF."},',
        '  {"source":"rag_design.md","content":"# RAG Design\\nRAG combines retrieval with generation. Components: Document Pipeline, Search Engine, LLM Generator. 2026 standard: Hybrid Search + Rerank + Self-RAG."},',
        '  {"source":"mcp_protocol.md","content":"# MCP Protocol\\nMCP is AI Agent standard protocol. Linux Foundation (Dec 2025). 3 primitives: Tool, Resource, Prompt. 10,000+ servers, 97M monthly SDK downloads."},',
        ']',
        '',
        'pipeline.ingest_texts(KNOWLEDGE)',
        'stats = pipeline.get_stats()',
        'print(f"Pipeline: {stats[\"documents\"]} docs -> {stats[\"total_chunks\"]} chunks ({stats[\"total_chars\"]} chars)")',
        'print(f"Avg chunk: {stats[\"avg_chunk_size\"]:.0f} chars")',
    ]),
    code([
        'print("===== Chunk Quality Analysis =====\\n")',
        'for doc in pipeline.documents:',
        '    sizes = [len(c) for c in doc.chunks]',
        '    print(f"{doc.metadata[\"source\"]}: {len(doc.chunks)} chunks, range [{min(sizes)},{max(sizes)}]")',
        '    for i, c in enumerate(doc.chunks[:2]):',
        '        print(f"  Chunk[{i}]: {c[:80]}...")',
    ]),
    code([
        'from agent_project.hybrid_search import HybridSearcher',
        'searcher = HybridSearcher()',
        'searcher.index(pipeline.all_chunks)',
        '',
        'results = searcher.search("What is RAG?", top_k=3)',
        'print(f"Search test: {len(results)} results")',
        'for r in results:',
        '    print(f"  [{r[\"score\"]:.4f}] {r[\"content\"][:80]}...")',
        'print("\\nConnected: DocumentPipeline -> HybridSearcher -> Final Project!")',
    ]),
])

# ===== 3.3 Advanced RAG =====
save("3.3_高级RAG架构", "实验_高级RAG系统.ipynb", [
    md("# 3.3 Advanced RAG - HybridSearch Engine"),
    md("## Project Role: HybridSearcher is the core retrieval module for the final Agent"),
    SETUP,
    code([
        'from agent_project.hybrid_search import HybridSearcher',
        'searcher = HybridSearcher()',
        '',
        'DOCS = [',
        '    "BM25 excels at exact keyword matching like error codes and product IDs",',
        '    "Vector search handles semantic similarity synonyms and cross-language queries",',
        '    "Hybrid search combines BM25 and vector for best accuracy and recall",',
        '    "RRF fusion merges rankings without tuning using reciprocal rank positions",',
        '    "Cross-Encoder reranking is 10-50x slower than Bi-Encoder but more accurate",',
        '    "Query rewriting expands user queries into multiple variants for better recall",',
        '    "Self-RAG evaluates retrieval quality and triggers fallback when needed",',
        '    "HyDE generates hypothetical document vectors to bridge semantic gaps",',
        '    "DeepSeek V4 achieves 80.6% on SWE-bench code generation benchmark",',
        '    "Qwen3.7-Max has 1M context handling entire books at once",',
        ']',
        'searcher.index(DOCS)',
        'print(f"Hybrid index built: {len(DOCS)} documents")',
        'print(f"Components: BM25 + Vector + RRF + Reranker + Self-Assessment")',
    ]),
    code([
        'print("===== Hybrid vs Vector Comparison =====\\n")',
        'queries = [',
        '    ("BM25 algorithm details", "exact keyword match"),',
        '    ("How to find similar documents?", "semantic search"),',
        '    ("Best retrieval for RAG?", "hybrid scenario"),',
        ']',
        'for q, desc in queries:',
        '    print(f"Query [{desc}]: {q}")',
        '    hr = searcher.search(q, top_k=3, use_hybrid=True, use_rerank=True)',
        '    vr = searcher.search(q, top_k=3, use_hybrid=False, use_rerank=False)',
        '    print(f"  Hybrid: {[r[\"doc_id\"] for r in hr]}  scores={[r[\"score\"] for r in hr]}")',
        '    print(f"  Vector: {[r[\"doc_id\"] for r in vr]}  scores={[r[\"score\"] for r in vr]}")',
        '    print()',
    ]),
    code([
        'print("===== Self-Assessment Test =====\\n")',
        'tests = [("What is BM25?", True), ("Weather forecast?", False)]',
        'for q, expect_good in tests:',
        '    r = searcher.search(q, top_k=5)',
        '    a = searcher.self_assess(r)',
        '    status = "OK" if ((a["grade"] in "AB") == expect_good) else "CHECK"',
        '    print(f"  [{status}] {q}: grade={a[\"grade\"]}, action={a[\"action\"]}")',
        'print("\\nHybridSearcher ready for final project!")',
    ]),
])

# ===== 4.2 LangGraph =====
save("4.2_框架实战", "实验_LangChain与LangGraph实战.ipynb", [
    md("# 4.2 Framework Practice - LangGraph Workflow"),
    md("## Project Role: Build state-machine Agent workflow for the final project"),
    SETUP,
    code([
        'import subprocess, sys',
        'subprocess.check_call([sys.executable, "-m", "pip", "install", "langgraph", "langchain", "langchain-openai", "-q"])',
        'print("langgraph + langchain installed")',
    ]),
    code([
        'from typing import TypedDict, Annotated',
        'import operator',
        '',
        'class AgentState(TypedDict):',
        '    messages: Annotated[list, operator.add]',
        '    next_action: str',
        '    tool_results: list',
        '    final_answer: str',
        '',
        'print("AgentState: messages + next_action + tool_results + final_answer")',
    ]),
    code([
        'from langgraph.graph import StateGraph, END',
        '',
        'def llm_node(state):',
        '    msg = str(state["messages"][-1]) if state["messages"] else "Hello"',
        '    r = client.chat(msg, temperature=0.3, max_tokens=200)',
        '    return {"messages": [r["content"]], "next_action": "done"}',
        '',
        'def router(state):',
        '    content = str(state.get("messages", [""])[-1]).lower()',
        '    if any(kw in content for kw in ["search", "calculate", "tool"]):',
        '        return "call_tool"',
        '    return "finish"',
        '',
        'def tool_node(state):',
        '    return {"tool_results": ["Tool executed"], "next_action": "tool_done"}',
        '',
        'def finish_node(state):',
        '    return {"final_answer": "Analysis complete", "next_action": "end"}',
        '',
        'graph = StateGraph(AgentState)',
        'graph.add_node("llm", llm_node)',
        'graph.add_node("tool", tool_node)',
        'graph.add_node("finish", finish_node)',
        'graph.set_entry_point("llm")',
        'graph.add_conditional_edges("llm", router, {"call_tool": "tool", "finish": "finish"})',
        'graph.add_edge("tool", "llm")',
        'graph.add_edge("finish", END)',
        'app = graph.compile()',
        'print("LangGraph workflow compiled!")',
        'print("Flow: llm -> {tool -> llm} or {finish -> END}")',
    ]),
    code([
        'print("\\n===== Test Workflow =====")',
        'result = app.invoke({"messages": ["Explain what LangGraph is in one sentence"], "next_action": "", "tool_results": [], "final_answer": ""})',
        'print(f"Final: {result.get(\"final_answer\", result.get(\"messages\", [\"N/A\"])[-1])[:200]}")',
    ]),
])

# ===== 5.1 LoRA Fine-tuning =====
save("5.1_高效微调技术", "实验_LoRA微调Qwen.ipynb", [
    md("# 5.1 LoRA Fine-tuning - Customize the Agent's LLM"),
    md("## Prerequisites: GPU 8GB+, model at ./models/Qwen3.7-1.7B-Instruct"),
    code([
        'import torch',
        'print(f"PyTorch: {torch.__version__}")',
        'print(f"CUDA: {torch.cuda.is_available()}")',
        'if torch.cuda.is_available():',
        '    print(f"GPU: {torch.cuda.get_device_name(0)}")',
        '    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB")',
    ]),
    code([
        'from transformers import AutoModelForCausalLM, AutoTokenizer',
        'from peft import LoraConfig, get_peft_model, TaskType',
        '',
        'MODEL_PATH = "./models/Qwen3.7-1.7B-Instruct"',
        'tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)',
        'tokenizer.pad_token = tokenizer.eos_token',
        '',
        'model = AutoModelForCausalLM.from_pretrained(',
        '    MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)',
        'print(f"Model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")',
    ]),
    code([
        'lora_config = LoraConfig(',
        '    r=8, lora_alpha=16,',
        '    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],',
        '    lora_dropout=0.05,',
        '    task_type=TaskType.CAUSAL_LM,',
        '    bias="none",',
        ')',
        'model = get_peft_model(model, lora_config)',
        'model.print_trainable_parameters()',
    ]),
    code([
        'import json',
        'TRAIN_DATA = [',
        '    {"instruction":"What is RAG?","input":"","output":"RAG combines document retrieval with LLM generation for accurate sourced answers."},',
        '    {"instruction":"What is MCP?","input":"","output":"MCP is AI Agent standard protocol by Linux Foundation. 3 primitives: Tool Resource Prompt."},',
        '    {"instruction":"Explain LoRA","input":"","output":"LoRA trains low-rank matrices for efficient fine-tuning. Only ~0.1% params trained."},',
        ']',
        'with open("./data/train_data.json", "w") as f:',
        '    json.dump(TRAIN_DATA, f, ensure_ascii=False, indent=2)',
        'print(f"Training data: {len(TRAIN_DATA)} examples -> ./data/train_data.json")',
        'print("For real training: use alpaca_zh_1k.json (1000 examples), ~30min on RTX 3070")',
    ]),
])

# ===== 5.2 Deployment =====
save("5.2_私有化部署", "实验_模型部署实战.ipynb", [
    md("# 5.2 Model Deployment - Serve the Agent as API"),
    md("## Prerequisites: Ollama installed (ollama.com)"),
    code([
        'print("===== Ollama Deployment =====\\n")',
        'print("1. Install: https://ollama.com/download/windows")',
        'print("2. Pull model: ollama pull qwen3:1.7b")',
        'print("3. Create Modelfile with LoRA adapter")',
        'print("4. Build: ollama create kb-agent-v1 -f Modelfile")',
        'print("5. Serve: ollama serve")',
        'print("6. Test: curl http://localhost:11434/api/generate -d \'{\"model\":\"kb-agent-v1\",\"prompt\":\"Hello\"}\'\\n")',
    ]),
    code([
        'print("===== FastAPI Alternative =====\\n")',
        'print("from fastapi import FastAPI")',
        'print("from agent_project.app import KnowledgeBaseAgent")',
        'print("app = FastAPI()")',
        'print("@app.post(\"/ask\")")',
        'print("async def ask(question: str):")',
        'print("    return agent.ask(question)")',
        'print("# Run: uvicorn deploy.api:app --host 0.0.0.0 --port 8000")',
    ]),
])

print("All 5 remaining notebooks generated!")
