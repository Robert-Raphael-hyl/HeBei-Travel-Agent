from __future__ import annotations
import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


# =========================
# 0) Embedding
# =========================
class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=True
        )

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

# =========================
# 1) 从 hebei_knowledge.txt 解析为条目
# =========================
def load_entries_structured(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到知识库文件：{path}")

    raw = open(path, "r", encoding="utf-8").read().strip()
    if not raw:
        return []

    # 以 --- 作为“组分隔”
    groups = [g.strip() for g in raw.split("---") if g.strip()]

    entries: List[str] = []
    for g in groups:
        parts = re.split(r"(?=【类型】)", g)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if not p.startswith("【类型】"):
                continue
            entries.append(p)

    return entries


def block_to_document(block: str, idx: int) -> Document:
    def extract(field: str) -> str:
        m = re.search(rf"【{field}】([^\n]+)", block)
        return m.group(1).strip() if m else ""

    entry_type = extract("类型")
    city = extract("城市")
    name = extract("名称")
    content = extract("内容")

    page_content = (
        f"【类型】{entry_type}\n"
        f"【城市】{city}\n"
        f"【名称】{name}\n"
        f"【内容】{content}"
    )

    title = f"{entry_type}-{name}".strip("-")

    return Document(
        page_content=page_content,
        metadata={
            "type": entry_type,
            "city": city,
            "name": name,
            "title": title,
            "id": idx
        }
    )


def build_documents_from_txt(path: str) -> List[Document]:
    entries = load_entries_structured(path)
    docs: List[Document] = []
    for i, e in enumerate(entries):
        docs.append(block_to_document(e, i))
    return docs


# =========================
# 2) 构建并保存 FAISS
# =========================
def build_faiss(
    txt_path: str = "hebei_knowledge.txt",
    out_dir: str = "faiss_hebei",
) -> None:
    docs = build_documents_from_txt(txt_path)
    if not docs:
        raise ValueError("知识库 txt 为空或解析失败，无法构建向量库。")

    embeddings = LocalEmbeddings()

    print(f"读取条目数：{len(docs)}")
    print("使用本地 SentenceTransformer Embedding")
    print("开始构建 FAISS（首次会慢一些）...")

    vs = FAISS.from_documents(docs, embeddings)
    os.makedirs(out_dir, exist_ok=True)
    vs.save_local(out_dir)

    print(f"构建完成：已保存到 {out_dir}/")


if __name__ == "__main__":
    build_faiss()
