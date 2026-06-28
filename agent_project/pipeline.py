"""
文档处理管道 - 文档加载、清洗、分块、批量处理
"""
import re
import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """文档对象"""
    content: str
    metadata: dict = field(default_factory=dict)
    chunks: list[str] = field(default_factory=list)

    def __repr__(self):
        return f"Document({self.metadata.get('source', 'unknown')}, {len(self.content)} chars, {len(self.chunks)} chunks)"


class DocumentLoader:
    """文档加载器 - 支持多种格式"""

    @staticmethod
    def load_text(path: str) -> Document:
        """加载纯文本文件"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return Document(content, {"source": path, "format": "txt", "size": len(content)})

    @staticmethod
    def load_markdown(path: str) -> Document:
        """加载Markdown文件, 自动去除markdown标记"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 保留原始markdown用于显示
        cleaned = DocumentLoader._clean_markdown(content)
        return Document(cleaned, {"source": path, "format": "md", "size": len(content),
                                  "raw_size": len(content)})

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """清洗Markdown标记"""
        text = re.sub(r'#{1,6}\s', '', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        return re.sub(r'\n{3,}', '\n\n', text).strip()

    @staticmethod
    def load_directory(directory: str, extensions: tuple = ('.txt', '.md')) -> list[Document]:
        """批量加载目录中的所有文档"""
        docs = []
        for root, _, files in os.walk(directory):
            for fname in sorted(files):
                if fname.endswith(extensions):
                    path = os.path.join(root, fname)
                    try:
                        if fname.endswith('.md'):
                            docs.append(DocumentLoader.load_markdown(path))
                        else:
                            docs.append(DocumentLoader.load_text(path))
                    except Exception as e:
                        print(f"  跳过 {fname}: {e}")
        return docs


class TextChunker:
    """文本分块器 - 支持固定大小和语义分块"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[str]:
        """对文档进行分块, 添加安全边界防止死循环"""
        text = document.content
        if len(text) <= self.chunk_size:
            document.chunks = [text.strip()] if text.strip() else []
            return document.chunks
        chunks = []
        start = 0
        max_iter = len(text) // max(self.chunk_size - self.chunk_overlap, 1) + 2
        for _ in range(max_iter):
            if start >= len(text):
                break
            end = min(start + self.chunk_size, len(text))
            # 尝试在自然断点处切割 (优先中文句号, 兼容英文)
            if end < len(text):
                for sep in ['. ', '\n\n', '\n', ' ', '.']:
                    pos = text.rfind(sep, start, end)
                    if pos > start + self.chunk_size // 3:
                        end = pos + len(sep)
                        break
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 10:
                chunks.append(chunk)
            start = end - self.chunk_overlap
        document.chunks = chunks
        return chunks

    def batch_chunk(self, documents: list[Document]) -> int:
        """批量分块, 返回总块数"""
        total = 0
        for doc in documents:
            total += len(self.chunk(doc))
        return total


class DocumentPipeline:
    """完整文档处理管道: 加载 → 清洗 → 分块 → 向量化"""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.documents: list[Document] = []
        self.all_chunks: list[str] = []

    def ingest_directory(self, directory: str) -> int:
        """从目录批量导入文档"""
        self.documents = self.loader.load_directory(directory)
        total_chunks = self.chunker.batch_chunk(self.documents)
        self.all_chunks = []
        for doc in self.documents:
            self.all_chunks.extend(doc.chunks)
        return len(self.documents)

    def ingest_texts(self, texts: list[dict]) -> int:
        """从文本列表导入, 每条: {content, source}"""
        self.documents = []
        for item in texts:
            doc = Document(item["content"], {"source": item.get("source", "inline")})
            self.chunker.chunk(doc)
            self.documents.append(doc)
        self.all_chunks = []
        for doc in self.documents:
            self.all_chunks.extend(doc.chunks)
        return len(self.documents)

    def get_stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "total_chunks": len(self.all_chunks),
            "avg_chunk_size": sum(len(c) for c in self.all_chunks) / max(len(self.all_chunks), 1),
            "total_chars": sum(len(d.content) for d in self.documents),
        }
