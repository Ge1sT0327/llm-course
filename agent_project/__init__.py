"""
智能知识库Agent - 完整的RAG + Agent系统
通过14个实验逐步构建的最终项目

使用方式:
    from agent_project import KnowledgeBaseAgent
    from config import get_client

    agent = KnowledgeBaseAgent(
        llm_client=get_client(),
        knowledge_docs=["文档1", "文档2", ...],
    )
    result = agent.ask("什么是MCP协议?")
"""
from .app import KnowledgeBaseAgent
from .conversation import ConversationManager
from .hybrid_search import HybridSearcher, BM25Retriever, VectorRetriever, RRF, Reranker
from .agent import ReActAgent, PlanSolveAgent
from .pipeline import DocumentPipeline, DocumentLoader, TextChunker
from .security import SecurityManager
from .tools import ToolRegistry, create_default_registry
