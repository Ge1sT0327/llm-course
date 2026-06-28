"""
智能知识库Agent - 主入口
组装所有模块: 对话管理 + 混合检索 + Agent推理 + 工具调用 + 安全防护
"""
import sys
from pathlib import Path

# 确保能从项目根目录导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from .conversation import ConversationManager
from .hybrid_search import HybridSearcher
from .agent import ReActAgent, PlanSolveAgent
from .tools.registry import ToolRegistry, create_default_registry
from .security import SecurityManager


class KnowledgeBaseAgent:
    """智能知识库Agent - 完整的RAG + Agent系统"""

    def __init__(self, llm_client, embedding_model=None, knowledge_docs: list[str] | None = None):
        """
        初始化Agent

        Args:
            llm_client: LLM客户端 (from config import get_client)
            embedding_model: Embedding模型 (sentence-transformers)
            knowledge_docs: 初始知识库文档列表
        """
        self.llm = llm_client
        self.conversation = ConversationManager(
            system_prompt="你是智能知识库助手。基于知识库内容回答问题。"
        )
        self.searcher = HybridSearcher(embedding_model)
        self.tools = create_default_registry()
        self.security = SecurityManager()
        self.react_agent = ReActAgent(llm_client, self.tools)

        # 索引知识库
        if knowledge_docs:
            self.load_knowledge(knowledge_docs)

    def load_knowledge(self, documents: list[str]) -> None:
        """加载知识库文档并建立索引"""
        self.searcher.index(documents)
        # 将文档信息添加到搜索工具
        self.tools.register(
            "search_knowledge",
            lambda query: self._search_knowledge(query),
            "搜索知识库中的文档内容",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )

    def _search_knowledge(self, query: str) -> str:
        results = self.searcher.search(query, top_k=3)
        if not results:
            return "未找到相关文档"
        return "\n---\n".join(
            f"[文档{r['doc_id']}] (相关度:{r['score']}) {r['content']}"
            for r in results
        )

    def ask(self, question: str, use_agent: bool = False, verbose: bool = True) -> dict:
        """
        向Agent提问

        Args:
            question: 用户问题
            use_agent: 是否使用Agent模式 (多步推理+工具调用)
            verbose: 是否打印详细日志

        Returns:
            {"answer": str, "sources": list, "trace": list}
        """
        # 1. 安全检查
        validation = self.security.validate_input(question)
        if not validation["safe"]:
            return {"answer": f"输入安全检查未通过: {'; '.join(validation['issues'])}",
                    "sources": [], "trace": []}
        question = validation["sanitized"]

        # 2. 添加到对话历史
        self.conversation.add_user_message(question)
        self.security.audit("QUERY", question[:100])

        # 3. 检索相关文档
        retrieval_results = self.searcher.search(question, top_k=5, use_hybrid=True, use_rerank=True)
        assessment = self.searcher.self_assess(retrieval_results)

        if verbose:
            print(f"[检索] {len(retrieval_results)} 条结果, 评估: {assessment['grade']}")

        # 降级策略
        if assessment["action"] in ("rewrite_query", "expand_search"):
            retrieval_results = self.searcher.search(question, top_k=10, use_hybrid=True, use_rerank=False)
            if verbose:
                print(f"[降级] 扩大检索至 {len(retrieval_results)} 条")

        # 4. 构建上下文
        context = "\n".join(
            f"[来源{i+1}] {r['content']}" for i, r in enumerate(retrieval_results[:3])
        ) if retrieval_results else "无相关文档"

        # 5. 生成回答
        if use_agent:
            prompt = f"""基于以下知识库内容回答问题。如果需要, 可以使用工具获取更多信息。

知识库内容:
{context}

问题: {question}"""
            answer = self.react_agent.run(prompt)
            trace = [{"step": s.step_num, "thought": s.thought[:100],
                      "action": s.action, "obs": s.observation[:100]}
                     for s in self.react_agent.trace]
        else:
            messages = [
                {"role": "system", "content": f"基于以下知识库内容回答问题:\n{context}"},
                {"role": "user", "content": question},
            ]
            response = self.llm.chat(messages=messages, temperature=0.3, max_tokens=500)
            answer = response["content"]
            trace = []

        # 6. 记录
        self.conversation.add_assistant_message(answer, {"sources": len(retrieval_results)})
        self.security.audit("RESPONSE", f"len={len(answer)}, sources={len(retrieval_results)}")

        return {
            "answer": answer,
            "sources": retrieval_results[:3],
            "trace": trace,
            "assessment": assessment,
        }

    def batch_ask(self, questions: list[str]) -> list[dict]:
        """批量提问"""
        return [self.ask(q, verbose=False) for q in questions]

    def get_stats(self) -> dict:
        """获取系统统计"""
        conv = self.conversation.to_dict()
        return {
            "turn_count": conv["turn_count"],
            "token_count": conv["token_count"],
            "tools_available": self.tools.list_tools(),
            "audit_entries": len(self.security.get_audit_trail()),
        }
