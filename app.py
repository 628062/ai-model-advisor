"""
AI Model Advisor - RAG over 6 AI chatbots (ChatGPT, Claude, Gemini, DeepSeek, Grok, Perplexity)
Modes: Ask | Compare | Recommend
Run:  streamlit run app.py
"""
import os
import glob
import base64
import streamlit as st
try:
    import streamlit.components.v1 as components
except Exception:
    components = None
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# =====================================================================
# CONFIG
# =====================================================================
DATA_DIR = "data"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_OPTIONS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]

# key: (display name, company, colour, tagline)
BOTS = {
    "chatgpt":    ("ChatGPT",    "OpenAI",     "#10a37f", "All-rounder, biggest ecosystem"),
    "claude":     ("Claude",     "Anthropic",  "#d97757", "Coding and long documents"),
    "gemini":     ("Gemini",     "Google",     "#4285f4", "Free tier, multimodal"),
    "deepseek":   ("DeepSeek",   "DeepSeek",   "#4d6bfe", "Cheapest, open weights"),
    "grok":       ("Grok",       "xAI",        "#9ca3af", "Real-time X data, voice and video"),
    "perplexity": ("Perplexity", "Perplexity", "#20b2aa", "Live web search with citations"),
}
def name(k): return BOTS[k][0]

RULES = """Rules:
- Use ONLY the information in the CONTEXT. Do not use outside knowledge.
- If the answer is not in the context, say: "Sorry, this information is not in my knowledge base."
- Always reply in English.
- Use clean Markdown only (headings, bullets, tables). NEVER use HTML tags like <br>.
- Keep table cells short. Put code in separate code blocks below tables, never inside a table."""

ASK_EXAMPLES = [
    "What models does Claude offer?",
    "Which chatbots have a free API?",
    "GPT-6 Astra vs Sol vs Luna?",
    "How is Perplexity different?",
]
REC_EXAMPLES = [
    "I'm a student learning to code and need a free or very cheap AI API.",
    "I want a customer support chatbot for 10,000 users on a low budget.",
    "I need up-to-date answers with sources for research.",
    "I need to analyze very long PDF documents.",
]

st.set_page_config(page_title="AI Model Advisor", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

# =====================================================================
# STYLES
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton, .stSelectbox, .stTextArea {font-family:'Inter',sans-serif;}
.stApp {background:
    radial-gradient(900px 500px at 100% -5%, rgba(124,92,255,.22), transparent 60%),
    radial-gradient(800px 480px at -10% 105%, rgba(32,178,170,.16), transparent 60%),
    #0b0e22;}
header[data-testid="stHeader"] {background:transparent;}
#MainMenu, footer {visibility:hidden;}
.block-container {padding-top:1.6rem; max-width:1200px;}

/* nav */
.nav {display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem;}
.brand {display:flex;align-items:center;gap:.7rem;font-weight:800;font-size:1.15rem;color:#f0f0fa;}
.logo {width:38px;height:38px;border-radius:11px;display:grid;place-items:center;font-size:1.2rem;
       background:linear-gradient(135deg,#7c5cff,#20b2aa);box-shadow:0 6px 20px rgba(124,92,255,.45);}
.tech span {font-size:.72rem;font-weight:600;color:#b8b8d8;padding:5px 10px;margin-left:6px;border-radius:999px;
       border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.04);}

/* hero */
.badge {display:inline-block;padding:5px 12px;border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:.08em;
       color:#b7a6ff;background:rgba(124,92,255,.14);border:1px solid rgba(124,92,255,.35);}
.title {font-size:3.1rem;font-weight:800;line-height:1.05;margin:.9rem 0 .8rem;letter-spacing:-.02em;color:#f4f4ff;}
.title em {font-style:normal;background:linear-gradient(90deg,#8f73ff,#20c7bb);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sub {font-size:1.05rem;line-height:1.6;color:#b9bad6;max-width:520px;}
.stats {display:flex;gap:10px;margin-top:1.4rem;}
.stat {flex:1;padding:14px 12px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);}
.stat b {display:block;font-size:1.5rem;font-weight:800;color:#fff;}
.stat small {color:#a3a4c4;font-size:.78rem;}

/* bot cards */
.bots {display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:1.6rem 0 1.2rem;}
.bot {padding:14px 12px;border-radius:14px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.08);
      border-top:3px solid var(--c);transition:transform .15s;}
.bot:hover {transform:translateY(-3px);background:rgba(255,255,255,.06);}
.bot .n {font-weight:700;color:#f2f2ff;font-size:.95rem;}
.bot .co {font-size:.72rem;color:#8e90b3;margin-bottom:6px;}
.bot .t {font-size:.75rem;color:#c4c5de;line-height:1.35;}
@media (max-width:900px){.bots{grid-template-columns:repeat(3,1fr);} .title{font-size:2.3rem;}}

/* tabs as pills */
div[data-testid="stTabs"] [role="tablist"] {gap:8px;border-bottom:none;padding:6px;border-radius:14px;
       background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);width:fit-content;}
div[data-testid="stTabs"] button[role="tab"] {padding:8px 20px;border-radius:10px;font-weight:600;}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {background:linear-gradient(90deg,#7c5cff,#5b7cff);color:#fff;}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"], div[data-testid="stTabs"] [data-baseweb="tab-border"] {display:none;}

/* inputs + buttons */
.stTextInput input, .stTextArea textarea {border-radius:12px !important;}
div.stButton > button {border-radius:10px;font-weight:600;}
div.stButton > button[kind="primary"] {background:linear-gradient(90deg,#7c5cff,#5b7cff);border:none;
       box-shadow:0 6px 18px rgba(124,92,255,.35);padding:.55rem 1.6rem;}
div.stButton > button[kind="secondary"] {background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.12);
       font-size:.82rem;font-weight:500;color:#cfd0ea;}
div.stButton > button[kind="secondary"]:hover {border-color:#7c5cff;color:#fff;}

/* answer card */
.ans-head {display:flex;align-items:center;justify-content:space-between;margin-bottom:.3rem;}
.ans-head b {font-size:1rem;color:#fff;}
.ans-head span {font-size:.72rem;color:#a3a4c4;}
.src span {display:inline-block;font-size:.75rem;padding:4px 10px;margin:0 6px 6px 0;border-radius:999px;
       background:rgba(32,178,170,.12);border:1px solid rgba(32,178,170,.35);color:#9fe7e0;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px;}
.section {font-size:1.25rem;font-weight:700;color:#f2f2ff;margin:.4rem 0 .1rem;}
.hint {color:#a3a4c4;font-size:.9rem;margin-bottom:.6rem;}
.foot {text-align:center;color:#7d7fa3;font-size:.8rem;margin-top:2.5rem;}
section[data-testid="stSidebar"] {background:#0f1230;border-right:1px solid rgba(255,255,255,.06);}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# RAG
# =====================================================================
@st.cache_resource(show_spinner="Building knowledge base...")
def build_store():
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        key = os.path.splitext(os.path.basename(path))[0].lower()
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append(Document(page_content=text, metadata={"bot": key, "source": os.path.basename(path)}))
    if not docs:
        return None, 0, 0
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(docs)
    store = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name=EMBED_MODEL))
    return store, len(docs), len(chunks)

def format_context(chunks):
    return "\n\n".join(f"[Source: {c.metadata['source']}]\n{c.page_content}" for c in chunks)

def run_llm(llm, prompt):
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        st.error(f"LLM error: {e}  —  try another model from the sidebar.")
        return None

def render_answer(result, title):
    """result = {'answer': str, 'chunks': [...]}; shown in a card with source pills."""
    if not result:
        return
    chunks = result["chunks"]
    files = list(dict.fromkeys(c.metadata["source"] for c in chunks))
    with st.container(border=True):
        st.markdown(f"<div class='ans-head'><b>✨ {title}</b><span>{len(chunks)} chunks · {len(files)} sources</span></div>",
                    unsafe_allow_html=True)
        st.markdown(result["answer"])
        st.markdown("<div class='src'>" + "".join(f"<span>📄 {f}</span>" for f in files) + "</div>",
                    unsafe_allow_html=True)
        with st.expander("View retrieved context"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**{i}. {c.metadata['source']}**")
                st.caption(c.page_content[:450] + ("..." if len(c.page_content) > 450 else ""))

# =====================================================================
# INTRO VIDEO (autoplays; falls back to muted + "Tap for sound")
# =====================================================================
@st.cache_data
def video_b64(path, mtime):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def intro_player(path):
    b64 = video_b64(path, os.path.getmtime(path))
    html = f"""
<div style="position:relative;width:100%;border-radius:18px;padding:6px;
     background:linear-gradient(135deg,rgba(124,92,255,.7),rgba(32,178,170,.7));box-shadow:0 18px 50px rgba(0,0,0,.45);">
  <video id="v" autoplay playsinline controls style="width:100%;border-radius:13px;display:block;background:#000;">
    <source src="data:video/mp4;base64,{b64}" type="video/mp4">
  </video>
  <button id="b" style="display:none;position:absolute;top:18px;right:18px;padding:7px 14px;border:none;border-radius:999px;
     background:#7c5cff;color:#fff;font:600 13px Inter,sans-serif;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.4);">
     🔊 Tap for sound</button>
</div>
<script>
  const v=document.getElementById('v'), b=document.getElementById('b');
  v.play().catch(()=>{{v.muted=true; v.play(); b.style.display='block';}});
  b.onclick=()=>{{v.muted=false; v.currentTime=0; v.play(); b.style.display='none';}};
</script>"""
    if hasattr(st, "iframe"):                      # newer Streamlit
        st.iframe(html, height=330)
    elif components is not None and hasattr(components, "html"):
        components.html(html, height=330)
    else:
        st.video(path, autoplay=True, muted=True)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = os.getenv("GROQ_API_KEY") or st.text_input("🔑 Groq API key", type="password",
                                                         help="Free key from console.groq.com/keys")
    if api_key:
        st.success("API key connected", icon="✅")
    else:
        st.caption("Get a free key at console.groq.com/keys")
    llm_model = st.selectbox("🧠 LLM model", LLM_OPTIONS, help="If you get an error, try another model")
    st.toggle("🎬 Show intro video", value=True, key="show_intro")

store, n_docs, n_chunks = build_store()
with st.sidebar:
    st.divider()
    st.markdown("### 📚 Knowledge base")
    m1, m2 = st.columns(2)
    m1.metric("Files", n_docs)
    m2.metric("Chunks", n_chunks)
    st.caption("Embeddings: all-MiniLM-L6-v2\n\nVector store: FAISS\n\nData as of 25 Sep 2026")

# =====================================================================
# HEADER + HERO
# =====================================================================
st.markdown("""
<div class="nav">
  <div class="brand"><div class="logo">🤖</div>AI Model Advisor</div>
  <div class="tech"><span>RAG</span><span>LangChain</span><span>FAISS</span><span>Groq</span></div>
</div>""", unsafe_allow_html=True)

show_video = os.path.exists("intro.mp4") and st.session_state.get("show_intro", True)
left, right = st.columns([1.1, 1] if show_video else [1, 0.001], gap="large")
with left:
    st.markdown("""
<div class="badge">RAG-POWERED AI ASSISTANT</div>
<div class="title">Find the <em>right AI</em><br>for every job.</div>
<div class="sub">Ask about, compare and choose between the world's leading AI chatbots. Every answer is grounded in official documentation and shows its sources.</div>
<div class="stats">
  <div class="stat"><b>6</b><small>AI chatbots</small></div>
  <div class="stat"><b>3</b><small>Smart modes</small></div>
  <div class="stat"><b>100%</b><small>Cited answers</small></div>
</div>""", unsafe_allow_html=True)
with right:
    if show_video:
        intro_player("intro.mp4")

st.markdown("<div class='bots'>" + "".join(
    f"<div class='bot' style='--c:{c}'><div class='n'>{n}</div><div class='co'>{co}</div><div class='t'>{t}</div></div>"
    for n, co, c, t in BOTS.values()) + "</div>", unsafe_allow_html=True)

# =====================================================================
# GUARDS
# =====================================================================
if store is None:
    st.error(f"No .txt files found in '{DATA_DIR}/'. Add chatgpt.txt, claude.txt, etc.")
    st.stop()
if not api_key:
    with st.container(border=True):
        st.markdown("#### 👋 Get started in 3 steps")
        st.markdown("1. Get a free API key at **console.groq.com/keys**\n"
                    "2. Paste it in the **sidebar** on the left\n"
                    "3. Ask, compare or get a recommendation")
    st.stop()

llm = ChatGroq(model=llm_model, api_key=api_key, temperature=0.2)
available = [k for k in BOTS if os.path.exists(os.path.join(DATA_DIR, f"{k}.txt"))]
ss = st.session_state
for k in ("ask_res", "cmp_res", "rec_res"):
    ss.setdefault(k, None)

def set_value(key, value):
    ss[key] = value
    ss[key + "_go"] = True

tab_ask, tab_cmp, tab_rec = st.tabs(["💬  Ask", "⚖️  Compare", "🎯  Recommend"])

# ---------------------------------------------------------------- ASK
with tab_ask:
    st.markdown("<div class='section'>Ask anything</div><div class='hint'>Models, APIs, pricing, strengths and more.</div>",
                unsafe_allow_html=True)
    ci, cb = st.columns([5, 1])
    q = ci.text_input("Question", key="ask_q", label_visibility="collapsed",
                      placeholder="e.g. What models does Claude offer, and how are they different?")
    go = cb.button("Ask", type="primary", use_container_width=True, key="ask_btn")
    cols = st.columns(len(ASK_EXAMPLES))
    for col, ex in zip(cols, ASK_EXAMPLES):
        col.button(ex, key=f"ex_{ex}", use_container_width=True, on_click=set_value, args=("ask_q", ex))
    if (go or ss.pop("ask_q_go", False)) and q:
        with st.spinner("Searching the knowledge base..."):
            chunks = store.similarity_search(q, k=5)
            ans = run_llm(llm, f"""You are an expert assistant about AI chatbots and their models/APIs.
{RULES}

CONTEXT:
{format_context(chunks)}

QUESTION: {q}

ANSWER:""")
        ss.ask_res = {"answer": ans, "chunks": chunks, "q": q} if ans else None
    if ss.ask_res:
        render_answer(ss.ask_res, ss.ask_res["q"])

# ---------------------------------------------------------------- COMPARE
with tab_cmp:
    st.markdown("<div class='section'>Compare side by side</div><div class='hint'>Pick two chatbots and an optional focus.</div>",
                unsafe_allow_html=True)
    c1, cvs, c2, c3 = st.columns([3, 0.4, 3, 3])
    a = c1.selectbox("Chatbot A", available, format_func=name, index=0)
    cvs.markdown("<div style='text-align:center;padding-top:2.1rem;font-weight:800;color:#8f73ff'>VS</div>",
                 unsafe_allow_html=True)
    b = c2.selectbox("Chatbot B", available, format_func=name, index=1 if len(available) > 1 else 0)
    aspect = c3.selectbox("Focus on", ["Everything", "Pricing", "Models", "API", "Coding",
                                       "Context window", "Free tier", "Web search", "Strengths and weaknesses"])
    if st.button("Compare", type="primary", key="cmp_btn"):
        if a == b:
            st.warning("Please pick two different chatbots.")
        else:
            focus = "" if aspect == "Everything" else aspect
            query = focus or "overview models features pricing API strengths weaknesses"
            with st.spinner(f"Comparing {name(a)} and {name(b)}..."):
                ca = store.similarity_search(query, k=4, filter={"bot": a})
                cb_ = store.similarity_search(query, k=4, filter={"bot": b})
                ans = run_llm(llm, f"""You compare AI chatbots.
{RULES}

CONTEXT FOR {name(a)}:
{format_context(ca)}

CONTEXT FOR {name(b)}:
{format_context(cb_)}

TASK: Compare {name(a)} vs {name(b)}{' focusing on: ' + focus if focus else ''}.
1. A markdown comparison table (rows = aspects, columns = {name(a)}, {name(b)}). Write "Not in data" where info is missing.
2. Then 3-4 bullet key differences.
3. End with a one-line **Verdict:** who should pick which.""")
            ss.cmp_res = {"answer": ans, "chunks": ca + cb_, "t": f"{name(a)} vs {name(b)}"} if ans else None
    if ss.cmp_res:
        render_answer(ss.cmp_res, ss.cmp_res["t"])

# ---------------------------------------------------------------- RECOMMEND
with tab_rec:
    st.markdown("<div class='section'>Get a recommendation</div><div class='hint'>Describe what you want to build, your budget and needs.</div>",
                unsafe_allow_html=True)
    use_case = st.text_area("Use case", key="rec_q", height=100, label_visibility="collapsed",
                            placeholder="e.g. I'm a student learning to code and need a free or very cheap AI API.")
    cols = st.columns(len(REC_EXAMPLES))
    for col, ex in zip(cols, REC_EXAMPLES):
        col.button(ex[:38] + ("…" if len(ex) > 38 else ""), key=f"rex_{ex}", use_container_width=True,
                   on_click=set_value, args=("rec_q", ex), help=ex)
    go = st.button("Recommend", type="primary", key="rec_btn")
    if (go or ss.pop("rec_q_go", False)) and use_case:
        with st.spinner("Finding the best option..."):
            chunks = store.similarity_search(use_case, k=10)
            ans = run_llm(llm, f"""You are an AI advisor who recommends which AI chatbot/model a user should use.
{RULES}

CONTEXT:
{format_context(chunks)}

USER'S USE CASE: {use_case}

Respond in this format:
### 🏆 Best choice: <chatbot + specific model if known>
Why (3 bullets, tied to the user's needs)
### 🥈 Alternatives
2 alternatives, one line each on when to pick them
### ⚠️ Keep in mind
1-2 limitations or cost notes
Base everything only on the context.""")
        ss.rec_res = {"answer": ans, "chunks": chunks} if ans else None
    if ss.rec_res:
        render_answer(ss.rec_res, "Recommendation")

st.markdown("<div class='foot'>Built with LangChain · FAISS · HuggingFace embeddings · Groq · Streamlit"
            "<br>Data from official provider docs, as of 25 September 2026</div>", unsafe_allow_html=True)
