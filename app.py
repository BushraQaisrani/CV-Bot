"""
CVBot — Upgraded Streamlit UI (Gemini + LangChain + FAISS)
Backend identical to My_own_Bot.ipynb
Run: streamlit run app.py
"""
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ModuleNotFoundError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
CV_PATH = Path("CV.txt")

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="CVBot · Bushra Qadir",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Header ── */
.cv-header {
    background: linear-gradient(135deg, #0d9488 0%, #0891b2 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(13,148,136,0.25);
}
.cv-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff !important;
    margin: 0 0 4px 0;
}
.cv-header p {
    color: #ccfbf1 !important;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}

/* User bubble accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid #0d9488 !important;
    background: #1a2e3b !important;
}

/* Assistant bubble accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #0891b2 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #1e293b !important;
    border: 1px solid #0d9488 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #2dd4bf !important;
    box-shadow: 0 0 0 2px rgba(45,212,191,0.2) !important;
}

/* ── Suggestion chips ── */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;
}
.chip {
    background: #1e3a4a;
    border: 1px solid #0d9488;
    color: #2dd4bf;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s;
}
.chip:hover { background: #0d9488; color: #fff; }

/* ── Sidebar card ── */
.sidebar-card {
    background: #243447;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
.sidebar-card h4 {
    color: #2dd4bf !important;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 10px 0;
}
.sidebar-card p, .sidebar-card li {
    font-size: 0.85rem;
    color: #94a3b8 !important;
    margin: 4px 0;
}
.sidebar-card ul { padding-left: 16px; margin: 0; }

/* ── Stats row ── */
.stat-row {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
}
.stat-box {
    flex: 1;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.stat-box .num {
    font-size: 1.5rem;
    font-weight: 700;
    color: #2dd4bf;
}
.stat-box .label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #475569;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state h3 { color: #64748b; font-size: 1.1rem; margin-bottom: 6px; }
.empty-state p { font-size: 0.85rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0d9488; }

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Backend (identical to notebook) ─────────────────────────────────────────
@st.cache_resource
def build_chain():
    documents = TextLoader(str(CV_PATH), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if os.path.exists("vectorstore/index.faiss"):
        vectorstore = FAISS.load_local(
            "vectorstore", embeddings, allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("vectorstore")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are CVBot, an assistant that answers questions about Bushra Qadir "
         "using ONLY the context below.\n"
         "- Use only facts found in the context; never use outside knowledge.\n"
         "- You MAY combine facts from several context chunks to give a complete answer.\n"
         "- If the answer is genuinely not in the context, say: "
         "\"I don't have that information in the CV.\"\n"
         "- Be concise and quote details (dates, degree, project names) exactly as written.\n\n"
         "Context:\n{context}"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(retriever, qa_chain)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:3rem;'>🤖</div>
        <div style='font-size:1.1rem; font-weight:700; color:#2dd4bf;'>CVBot</div>
        <div style='font-size:0.75rem; color:#64748b;'>Powered by Gemini + LangChain</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-card'>
        <h4>👩‍💻 About</h4>
        <p><b style='color:#e2e8f0'>Bushra Qadir</b></p>
        <p>BS Software Engineering</p>
        <p>UMT Lahore · 2026</p>
    </div>

    <div class='sidebar-card'>
        <h4>🛠 Tech Stack</h4>
        <ul>
            <li>LangChain orchestration</li>
            <li>Gemini 2.5 Flash (LLM)</li>
            <li>Gemini Embeddings</li>
            <li>FAISS vector store</li>
            <li>Streamlit UI</li>
        </ul>
    </div>

    <div class='sidebar-card'>
        <h4>💡 Try asking</h4>
        <ul>
            <li>What degree does Bushra have?</li>
            <li>What ML projects has she built?</li>
            <li>What are her skills?</li>
            <li>Tell me about CycleGo</li>
            <li>What certifications does she have?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style='text-align:center; margin-top:16px;'>
        <span style='font-size:0.7rem; color:#334155;'>
        Answers grounded strictly in CV content
        </span>
    </div>
    """, unsafe_allow_html=True)


# ── Main area ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='cv-header'>
    <h1>📄 CV Assistant</h1>
    <p>Ask anything about Bushra Qadir's education, skills, projects, or experience.<br>
    All answers are grounded strictly in the CV — no hallucination.</p>
</div>
""", unsafe_allow_html=True)

# Stats row
msg_count = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
st.markdown(f"""
<div class='stat-row'>
    <div class='stat-box'>
        <div class='num'>8</div>
        <div class='label'>Projects</div>
    </div>
    <div class='stat-box'>
        <div class='num'>5+</div>
        <div class='label'>Skills</div>
    </div>
    <div class='stat-box'>
        <div class='num'>2026</div>
        <div class='label'>Graduating</div>
    </div>
    <div class='stat-box'>
        <div class='num'>{msg_count}</div>
        <div class='label'>Questions asked</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Key check ────────────────────────────────────────────────────────────────
if "GOOGLE_API_KEY" not in os.environ:
    st.error("⚠️ GOOGLE_API_KEY not found. Add it to your .env file and restart.")
    st.stop()

# ── Load chain ───────────────────────────────────────────────────────────────
with st.spinner("🔗 Loading CV knowledge base..."):
    rag_chain = build_chain()

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class='empty-state'>
        <div class='icon'>💬</div>
        <h3>Start a conversation</h3>
        <p>Ask about education, projects, skills, or experience.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask about Bushra's CV..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching CV..."):
            result = rag_chain.invoke({"input": question})
            answer = result["answer"]
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
