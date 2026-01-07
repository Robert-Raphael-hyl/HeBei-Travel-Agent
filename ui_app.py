from __future__ import annotations
import re
import uuid
import streamlit as st
from hebei_agent_faiss_main import get_hebei_answer, UNIAPI_ENABLED

# =========================
def extract_requested_days_from_text(text: str):
    """
    从用户输入中提取 X 日游
    """
    m = re.search(r"(\\d+)\\s*日", text)
    return int(m.group(1)) if m else None

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="河北旅游智能助手",
    page_icon="🧭",
    layout="centered",
)

# =========================
# 全局商业风格 CSS
# =========================
st.markdown(
    """
<style>
/* ===== 全局背景（商业产品质感：渐变 + 轻噪点感）===== */
.stApp {
  background:
    radial-gradient(1200px 600px at 10% 0%, rgba(37,99,235,0.10), transparent 60%),
    radial-gradient(1000px 500px at 90% 10%, rgba(16,185,129,0.10), transparent 55%),
    linear-gradient(180deg, #f6f7fb 0%, #f3f6ff 50%, #f6f7fb 100%);
}

/* ===== 页面容器 ===== */
.block-container {
  max-width: 980px;
  padding-top: 2.6rem;
  padding-bottom: 4rem;
}

/* ===== 顶部 Hero ===== */
.hero {
  position: relative;
  padding: 26px 22px;
  border-radius: 22px;
  background: linear-gradient(135deg, rgba(37,99,235,0.14), rgba(16,185,129,0.10));
  border: 1px solid rgba(255,255,255,0.60);
  box-shadow: 0 18px 50px rgba(0,0,0,0.08);
  overflow: hidden;
  margin-bottom: 18px;
}
.hero:before{
  content:"";
  position:absolute;
  inset:-2px;
  background:
    radial-gradient(260px 260px at 18% 30%, rgba(37,99,235,0.25), transparent 60%),
    radial-gradient(240px 240px at 82% 18%, rgba(16,185,129,0.22), transparent 60%);
  filter: blur(2px);
  opacity: .85;
}
.hero-inner{
  position: relative;
  z-index: 2;
}
.main-title{
  font-size: 40px;
  font-weight: 860;
  letter-spacing: 0.4px;
  margin: 0;
  color: #0f172a;
}
.sub-title{
  margin-top: 8px;
  font-size: 15px;
  color: rgba(15,23,42,0.72);
}
.hero-badges{
  margin-top: 14px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.badge{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding: 7px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 650;
  color: rgba(15,23,42,0.78);
  background: rgba(255,255,255,0.65);
  border: 1px solid rgba(255,255,255,0.60);
  backdrop-filter: blur(10px);
}
.badge-dot{
  width:8px;height:8px;border-radius:999px;
  background:#2563eb;
  box-shadow: 0 0 0 4px rgba(37,99,235,0.12);
}

/* ===== 侧边栏 ===== */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.86);
  border-right: 1px solid rgba(148,163,184,0.35);
}
section[data-testid="stSidebar"] .stMarkdown {
  color: rgba(15,23,42,0.86);
}

/* ===== Chat 气泡 ===== */
.stChatMessage {
  border-radius: 18px;
  padding: 18px 18px;
  margin-bottom: 14px;
  border: 1px solid rgba(148,163,184,0.28);
  box-shadow: 0 14px 40px rgba(0,0,0,0.06);
  background: rgba(255,255,255,0.85);
}
.stChatMessage[data-testid="user"]{
  background: linear-gradient(135deg, rgba(37,99,235,0.92), rgba(30,64,175,0.92));
  color: white;
  border: 1px solid rgba(255,255,255,0.20);
}
.stChatMessage[data-testid="assistant"]{
  background: rgba(255,255,255,0.92);
  color: #0f172a;
}

/* ===== 状态 pill ===== */
.status-pill{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 7px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 10px;
}
.status-local{
  background: rgba(37,99,235,0.12);
  color: #1e3a8a;
  border: 1px solid rgba(37,99,235,0.20);
}
.status-llm{
  background: rgba(16,185,129,0.14);
  color: #065f46;
  border: 1px solid rgba(16,185,129,0.22);
}

/* ===== Day 卡片 ===== */
.day-card{
  background: rgba(255,255,255,0.96);
  border-radius: 18px;
  padding: 18px 18px;
  margin: 14px 0;
  border: 1px solid rgba(148,163,184,0.26);
  box-shadow: 0 16px 44px rgba(0,0,0,0.07);
  position: relative;
  overflow: hidden;
}
.day-card:before{
  content:"";
  position:absolute;
  inset:0;
  border-left: 6px solid rgba(37,99,235,0.90);
}
.day-title{
  font-size: 16px;
  font-weight: 850;
  color: #0f172a;
  margin-bottom: 10px;
  padding-left: 10px;
}
.day-body{
  padding-left: 10px;
  color: rgba(15,23,42,0.86);
  font-size: 14px;
  line-height: 1.55;
}

/* ===== Evidence 证据盒 ===== */
.evidence-box{
  background: rgba(248,250,252,0.9);
  border: 1px dashed rgba(99,102,241,0.40);
  border-radius: 14px;
  padding: 12px 14px;
  font-size: 13px;
  color: rgba(15,23,42,0.78);
}

/* ===== 按钮质感 ===== */
button[kind="secondary"], .stButton>button{
  border-radius: 12px !important;
  padding: 10px 14px !important;
  border: 1px solid rgba(148,163,184,0.35) !important;
  background: rgba(255,255,255,0.86) !important;
}
button[kind="secondary"]:hover, .stButton>button:hover{
  border-color: rgba(99,102,241,0.45) !important;
  background: rgba(238,242,255,0.75) !important;
}
</style>
""",
    unsafe_allow_html=True
)


# =========================
# 把回答拆成 Day 卡片
# =========================
DAY_SPLIT_PATTERN = re.compile(r"(Day\s*\d+\s*[:：])", re.IGNORECASE)


def render_answer_cards(answer: str):
    parts = DAY_SPLIT_PATTERN.split(answer)

    if len(parts) <= 1:
        st.markdown(answer)
        return

    for i in range(1, len(parts), 2):
        day_title = parts[i].strip()
        day_body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        day_body_html = day_body.replace("\n", "<br>")

        st.markdown(
            f"""
            <div class="day-card">
              <div class="day-title">{day_title}</div>
              <div class="day-body">{day_body_html}</div>
            </div>
            """,
            unsafe_allow_html=True
        )



# =========================
# 会话初始化
# =========================
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_evidence" not in st.session_state:
    st.session_state.last_evidence = []


# =========================
# Sidebar：系统控制台
# =========================
with st.sidebar:
    st.markdown("## ⚙️ 系统控制台")

    page = st.radio(
        "导航",
        ["💬 Chat（问答）", "🗺 Roadmap（产品规划）"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🧠 模式设置")

    if UNIAPI_ENABLED:
        use_llm_enhance = st.toggle("启用 UniAPI 语言增强", value=False)
    else:
        use_llm_enhance = False
        st.info("未检测到 UniAPI 配置（.env 中缺少 UNIAPI_KEY/UNIAPI_BASE），当前仅本地模式。")

    if use_llm_enhance:
        st.markdown('<div class="status-pill status-llm">🧠 当前：UniAPI 语言增强</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill status-local">📘 当前：知识库驱动</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 快捷提问")

    demo_questions = [
        "河北3日游怎么安排？",
        "亲子4日游怎么安排？",
        "适合老人去的景点有哪些？",
        "清东陵门票和交通",
        "山海关避坑有哪些？",
    ]
    for q in demo_questions:
        if st.button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})

            answer, evidence = get_hebei_answer(
                q,
                st.session_state.user_id,
                use_llm_enhance=use_llm_enhance,
                return_evidence=True
            )
            st.session_state.last_evidence = evidence
            st.session_state.messages.append({"role": "assistant", "content": answer})

    st.markdown("---")
    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_evidence = []


# =========================
# Roadmap 页面
# =========================
if page == "🗺 Roadmap（产品规划）":
    st.markdown(
        """
<div class="hero">
  <div class="hero-inner">
    <div class="main-title">🗺 商业产品 Roadmap</div>
    <div class="sub-title">从 Demo → 决策型智能体 → 可用产品 → 平台级智能体</div>
    <div class="hero-badges">
      <span class="badge"><span class="badge-dot"></span>可控（知识库驱动）</span>
      <span class="badge"><span class="badge-dot"></span>可解释（Top-K 证据）</span>
      <span class="badge"><span class="badge-dot"></span>可落地（UI + 工程闭环）</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("### Stage 0｜可演示原型（当前）")
    st.markdown("- ✅ TXT → FAISS 向量库\n- ✅ 语义检索 Top-K + 可解释证据\n- ✅ 多轮对话（基础上下文）\n- ✅ Streamlit 商业风格 UI\n- ✅ 可选 UniAPI 表达增强（不改事实）")

    st.markdown("### Stage 1｜决策型智能体（短期）")
    st.markdown("- 🔜 规划类问题进入“多步决策流程”（追问→确认→生成）\n- 🔜 结构化输出（Day 卡片 / 门票卡 / 交通卡 / 预算卡）\n- 🔜 用户画像（亲子/老人/学生/预算/节奏偏好）")

    st.markdown("### Stage 2｜可用产品（中期）")
    st.markdown("- 🔜 推荐解释：为什么推荐 / 为什么不推荐\n- 🔜 多场景入口：行程 / 门票 / 交通 / 避坑\n- 🔜 可导出：生成行程单（Markdown / PDF）")

    st.markdown("### Stage 3｜平台级智能体（长期）")
    st.markdown("- 🔜 接入实时数据：开放时间、票价、天气、拥挤度\n- 🔜 多 Agent 协作：行程 / 预算 / 风险 / 偏好学习\n- 🔜 ToB 文旅局 / 景区咨询导览；ToC 会员与定制")

    st.stop()


# =========================
# Chat 页面
# =========================
st.markdown(
    """
<div class="hero">
  <div class="hero-inner">
    <div class="main-title">🧭 河北旅游智能助手</div>
    <div class="sub-title">商业级输出风格 ｜ 知识库可控 + FAISS 可解释检索（这么近，那么美，周末到河北）</div>
    <div class="hero-badges">
      <span class="badge"><span class="badge-dot"></span>RAG（检索增强）</span>
      <span class="badge"><span class="badge-dot"></span>Top-K 证据可截图</span>
      <span class="badge"><span class="badge-dot"></span>多轮对话</span>
      <span class="badge"><span class="badge-dot"></span>可选 UniAPI 润色</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

for msg in st.session_state.messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        if msg["role"] == "assistant":
            render_answer_cards(msg["content"])
        else:
            st.markdown(msg["content"])

if st.session_state.last_evidence:
    with st.expander("📎 本次回答的 Top-K 检索证据（可截图）", expanded=False):
        lines = []
        for i, e in enumerate(st.session_state.last_evidence, 1):
            title = e.get("title", "未命名")
            city = e.get("city", "")
            typ = e.get("type", "")
            lines.append(f"**[命中{i}]** {title}  ｜ {typ} ｜ {city}")
        st.markdown('<div class="evidence-box">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)

user_input = st.chat_input("请输入问题，例如：河北3日游 / 亲子4日游 / 清东陵门票 / 山海关避坑")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🔍 正在检索知识库并生成答案..."):
        answer, evidence = get_hebei_answer(
            user_input,
            st.session_state.user_id,
            use_llm_enhance=use_llm_enhance,
            return_evidence=True
        )

    st.session_state.last_evidence = evidence
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        render_answer_cards(answer)

    if use_llm_enhance:
        st.caption("🧠 本次回答：已启用 UniAPI 语言增强（仅润色，不新增事实）")
    else:
        st.caption("📘 本次回答：知识库驱动（可控可解释）")
