from __future__ import annotations
import os
from typing import Dict, List, Tuple, Union
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# =========================
# 0) 配置DeepSeek Chat
# =========================
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)
CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

FAISS_DIR = os.getenv("FAISS_DIR", "faiss_hebei")

# =========================
# 0.1) UniAPI
# =========================
UNIAPI_KEY = os.getenv("UNIAPI_KEY")
UNIAPI_BASE = os.getenv("UNIAPI_BASE")
UNIAPI_ENABLED = bool(UNIAPI_KEY and UNIAPI_BASE)

uniapi_client = None
if UNIAPI_ENABLED:
    uniapi_client = OpenAI(api_key=UNIAPI_KEY, base_url=UNIAPI_BASE)

UNIAPI_CHAT_MODEL = os.getenv("UNIAPI_CHAT_MODEL", "gpt-4o-mini")

# =========================
# 1) Embedding
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
# 2) 加载 FAISS 向量库
# =========================
def load_faiss():
    embeddings = LocalEmbeddings()
    if not os.path.isdir(FAISS_DIR):
        raise FileNotFoundError(
            f"未找到向量库目录 {FAISS_DIR}，请先运行 build_faiss_hebei.py"
        )
    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


vectorstore = load_faiss()

# =========================
# 3) 全局状态：对话记忆
# =========================
conversation_memory: Dict[str, List[Tuple[str, str]]] = {}


def get_history_text(user_id: str, last_n: int = 3) -> str:
    history = conversation_memory.get(user_id, [])[-last_n:]
    if not history:
        return "无"
    return "\n".join([f"用户：{q}\n智能体：{a}" for q, a in history])

# =========================
# 4) FAISS 语义检索
# =========================
def retrieve_relevant_knowledge(
    query: str,
    user_id: str,
    top_k: int = 5,
    return_evidence: bool = False
) -> Union[str, Tuple[str, List[dict]]]:
    """
    使用 FAISS + 本地 embedding 进行语义检索
    - 默认返回拼接后的知识内容（字符串）
    - return_evidence=True 时，同时返回 Top-K 命中证据（title等）
    """
    history_text = get_history_text(user_id)
    enhanced_query = f"{query}\n（历史对话：{history_text}）"

    raw_results = vectorstore.similarity_search(enhanced_query, k=top_k * 3)
    if not raw_results:
        if return_evidence:
            return "无相关信息", []
        return "无相关信息"

    filtered = []
    for doc in raw_results:
        title = doc.metadata.get("title", "")
        if title.startswith("城市"):
            continue
        filtered.append(doc)
        if len(filtered) >= top_k:
            break

    final_results = filtered if filtered else raw_results[:top_k]

    evidence = []
    for doc in final_results:
        evidence.append({
            "title": doc.metadata.get("title", doc.metadata.get("name", "未命名")),
            "type": doc.metadata.get("type", ""),
            "city": doc.metadata.get("city", ""),
            "name": doc.metadata.get("name", ""),
            "id": doc.metadata.get("id", None),
        })

    print("\n【向量检索命中 Top-K 条目（过滤后）】")
    for i, e in enumerate(evidence, 1):
        print(f"[命中{i}] {e.get('title')}")
    print("================================\n")

    merged_text = "\n\n".join([doc.page_content for doc in final_results])

    if return_evidence:
        return merged_text, evidence
    return merged_text

# =========================
# 4.1) UniAPI 语言增强
# =========================
def enhance_with_uniapi(answer: str, user_query: str) -> str:
    """
    注意：只做表达增强，不引入新信息、不新增事实。
    UniAPI 失败时自动回退为原始回答。
    """
    if not UNIAPI_ENABLED or not uniapi_client:
        return answer

    prompt = f"""
你是旅游产品的“文案润色助手”。请对下面【原始回答】进行优化，使其更像商业产品的输出：
- 保留原始事实
- 结构更清晰：用小标题 + 分点
- 更“保姆级”：给出操作步骤、注意事项、节奏建议
- 语言更自然更吸引人，但不夸张

用户问题：
{user_query}

原始回答（事实来源于知识库）：
{answer}

只输出润色后的最终回答正文，不要解释。
""".strip()

    try:
        resp = uniapi_client.chat.completions.create(
            model=UNIAPI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        print("[UniAPI 增强失败，已回退为本地回答]")
        print("原因：", e)
        return answer

# =========================
# 5) 核心问答函数
# =========================
def get_hebei_answer(
    user_query: str,
    user_id: str = "default",
    use_llm_enhance: bool = False,
    return_evidence: bool = False
) -> Union[str, Tuple[str, List[dict]]]:
    """
    - use_llm_enhance: True 时启用 UniAPI 表达增强（仅润色）
    - return_evidence: True 时返回 (answer, evidence)
    """
    user_query = user_query.strip()
    if not user_query:
        msg = "😯 你还没输入问题哦！可以问比如“承德避暑山庄门票”“保定驴肉火烧哪家正宗”～"
        return (msg, []) if return_evidence else msg

    too_vague = ["河北旅游", "河北好玩吗", "推荐什么", "怎么玩", "有啥好玩的"]
    if any(word == user_query for word in too_vague):
        msg = (
            "💡 你可以具体问这些哦：\n"
            "1. 景点类：XX景点门票 / 开放时间 / 怎么去\n"
            "2. 美食类：XX城市特色美食 / 推荐店铺\n"
            "3. 行程类：河北X日游（亲子 / 老人 / 情侣）\n"
            "4. 实用类：预约方式 / 避坑指南 / 交通攻略"
        )
        return (msg, []) if return_evidence else msg

    # === FAISS 检索 ===
    if return_evidence:
        relevant_knowledge, evidence = retrieve_relevant_knowledge(
            user_query, user_id=user_id, top_k=5, return_evidence=True
        )
    else:
        relevant_knowledge = retrieve_relevant_knowledge(
            user_query, user_id=user_id, top_k=5, return_evidence=False
        )
        evidence = []

    if relevant_knowledge == "无相关信息":
        msg = "😅 抱歉，我的知识库里暂时没有相关信息，可以换个问法试试～"
        return (msg, evidence) if return_evidence else msg

    # === 回答生成 ===
    history_text = get_history_text(user_id)

    final_prompt = f"""
你是一个【旅游产品级行程规划引擎】，不是聊天机器人。

请根据【知识库内容】，为用户生成一份【可直接执行的河北旅游行程方案】，必须满足以下要求：

【一、整体要求（非常重要）】
- 输出的是“最终可用方案”，不是建议草稿
- 不要提示用户“可以补充”“可再查询”“建议进一步了解”
- 不要把任何工作交给用户
- 假设用户会严格照着你给的内容出行

【二、结构要求（必须严格遵守）】
- 按天输出（Day 1 / Day 2 / Day 3 …）
- 每一天都必须包含以下四个模块（缺一不可）：

1️⃣ 今日行程概览  
   - 城市 / 区域  
   - 主要游览景点（按顺序）

2️⃣ 🎟 门票与必要消费  
   - 明确列出当天涉及景点的门票价格  
   - 若有观光车 / 游船 / 二次消费，需一并列出  
   - 用“约 / 人民币”标注，保持务实

3️⃣ 🚗 交通与移动方式  
   - 城市间或景点间交通方式（高铁 / 大巴 / 自驾 / 市内公交）  
   - 给出可执行的方案（如：高铁 + 市内打车）  
   - 说明大致时间成本或费用区间

4️⃣ ⚠️ 当天执行提醒（产品级）  
   - 排队 / 限流 / 预约  
   - 老人 / 亲子 / 学生注意事项  
   - 时间安排节奏（上午 / 下午 / 晚上）

【三、内容来源约束】
- 所有事实（门票、交通、开放时间）必须来自【知识库内容】
- 禁止编造、不确定信息可用“以景区官方为准”表述
- 若知识库中信息不足，需用“保守方案”而不是留空

【四、语言风格】
- 商业产品说明书风格
- 清晰、有条理、偏“保姆级”
- 不夸张、不营销、不口水

【五、结尾要求】
- 不要提问用户
- 不要让用户继续补充
- 结尾只允许一句总结性说明，例如：
  “以上行程已补齐门票与交通信息，可直接作为出行计划使用。”

【知识库内容如下】
{{relevant_knowledge}}

【用户需求】
{{user_query}}

请直接输出最终行程正文。

""".strip()

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.2,
        max_tokens=900,
    )
    answer = response.choices[0].message.content.strip()

    if use_llm_enhance:
        answer = enhance_with_uniapi(answer=answer, user_query=user_query)

    conversation_memory.setdefault(user_id, [])
    conversation_memory[user_id].append((user_query, answer))
    conversation_memory[user_id] = conversation_memory[user_id][-3:]

    return (answer, evidence) if return_evidence else answer


# =========================
# 6) CLI 入口
# =========================
if __name__ == "__main__":
    print("🎉 河北旅游智能体启动！")
    print("💡 这么近，那么美，周末到河北~~\n")

    USER_ID = "hebei_travel_user_001"

    while True:
        user_input = input("你：").strip()
        if user_input.lower() in ["拜拜", "退出", "结束"]:
            print("智能体：祝你在河北玩得开心！👋")
            conversation_memory.pop(USER_ID, None)
            break

        ans = get_hebei_answer(user_input, USER_ID, use_llm_enhance=False)
        print(f"智能体：{ans}\n")
