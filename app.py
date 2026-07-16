"""
CVBot — Purple Interactive UI (Gemini + LangChain + FAISS)
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

st.set_page_config(
    page_title="CVBot · Bushra Qadir",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #1a0533 0%, #2d1b69 40%, #1a0533 100%);
    color: #f0e6ff;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2d1b69 0%, #1a0533 100%) !important;
    border-right: 2px solid #7c3aed !important;
}
[data-testid="stSidebar"] * { color: #e9d5ff !important; }

/* Robot logo */
.robot-logo {
    text-align: center;
    padding: 20px 0 10px;
}
.robot-svg { filter: drop-shadow(0 0 16px #a855f7); }

/* Header */
.cv-header {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #ec4899 100%);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(168,85,247,0.4);
    position: relative;
    overflow: hidden;
}
.cv-header::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.cv-header h1 { color: #fff !important; font-size: 2rem; font-weight: 800; margin: 0 0 6px; }
.cv-header p  { color: #fce7f3 !important; font-size: 0.95rem; margin: 0; }

/* Stats */
.stat-row { display: flex; gap: 12px; margin-bottom: 20px; }
.stat-box {
    flex: 1;
    background: rgba(124,58,237,0.2);
    border: 1px solid #7c3aed;
    border-radius: 14px;
    padding: 14px 8px;
    text-align: center;
    transition: transform 0.2s;
}
.stat-box:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(168,85,247,0.3); }
.stat-box .num  { font-size: 1.6rem; font-weight: 800; color: #c084fc; }
.stat-box .icon { font-size: 1.2rem; }
.stat-box .lbl  { font-size: 0.68rem; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.06em; }

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: rgba(45,27,105,0.7) !important;
    border: 1px solid #4c1d95 !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 4px solid #ec4899 !important;
    background: rgba(236,72,153,0.1) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 4px solid #a855f7 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: rgba(45,27,105,0.8) !important;
    border: 2px solid #7c3aed !important;
    border-radius: 14px !important;
    color: #f0e6ff !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #ec4899 !important;
    box-shadow: 0 0 0 3px rgba(236,72,153,0.25) !important;
}

/* Sidebar cards */
.s-card {
    background: rgba(124,58,237,0.15);
    border: 1px solid #5b21b6;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.s-card h4 {
    color: #c084fc !important;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 10px;
}
.s-card p, .s-card li {
    font-size: 0.85rem;
    color: #ddd6fe !important;
    margin: 5px 0;
    line-height: 1.5;
}
.s-card ul { padding-left: 18px; margin: 0; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 56px 24px;
}
.empty-state .big { font-size: 4rem; margin-bottom: 12px; }
.empty-state h3 { color: #a78bfa; font-size: 1.2rem; margin-bottom: 8px; }
.empty-state p  { color: #7c3aed; font-size: 0.9rem; }

/* Clear button */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #ec4899) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(168,85,247,0.5) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a0533; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #a855f7; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Backend (unchanged) ───────────────────────────────────────────────────────
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Cute colourful robot SVG logo
    st.markdown("""
    <div class="robot-logo">
      <svg class="robot-svg" width="110" height="130" viewBox="0 0 110 130" xmlns="http://www.w3.org/2000/svg">
        <!-- Antenna -->
        <line x1="55" y1="18" x2="55" y2="6" stroke="#f472b6" stroke-width="3" stroke-linecap="round"/>
        <circle cx="55" cy="4" r="5" fill="#f472b6"/>
        <!-- Head -->
        <rect x="20" y="18" width="70" height="52" rx="16" fill="#7c3aed"/>
        <rect x="22" y="20" width="66" height="48" rx="14" fill="#6d28d9"/>
        <!-- Eyes -->
        <circle cx="38" cy="38" r="10" fill="#1a0533"/>
        <circle cx="72" cy="38" r="10" fill="#1a0533"/>
        <circle cx="38" cy="38" r="7" fill="#a855f7"/>
        <circle cx="72" cy="38" r="7" fill="#22d3ee"/>
        <circle cx="40" cy="36" r="2.5" fill="white"/>
        <circle cx="74" cy="36" r="2.5" fill="white"/>
        <!-- Mouth -->
        <rect x="34" y="54" width="42" height="8" rx="4" fill="#1a0533"/>
        <rect x="36" y="55" width="8" height="6" rx="2" fill="#f472b6"/>
        <rect x="47" y="55" width="8" height="6" rx="2" fill="#a855f7"/>
        <rect x="58" y="55" width="8" height="6" rx="2" fill="#22d3ee"/>
        <!-- Ears -->
        <rect x="8"  y="28" width="12" height="22" rx="6" fill="#ec4899"/>
        <rect x="90" y="28" width="12" height="22" rx="6" fill="#ec4899"/>
        <rect x="10" y="31" width="8"  height="16" rx="4" fill="#f9a8d4"/>
        <rect x="92" y="31" width="8"  height="16" rx="4" fill="#f9a8d4"/>
        <!-- Neck -->
        <rect x="46" y="70" width="18" height="10" rx="4" fill="#5b21b6"/>
        <!-- Body -->
        <rect x="15" y="80" width="80" height="44" rx="16" fill="#7c3aed"/>
        <rect x="17" y="82" width="76" height="40" rx="14" fill="#6d28d9"/>
        <!-- Chest panel -->
        <rect x="30" y="90" width="50" height="24" rx="8" fill="#1a0533"/>
        <circle cx="42" cy="102" r="6" fill="#f472b6"/>
        <circle cx="55" cy="102" r="6" fill="#a855f7"/>
        <circle cx="68" cy="102" r="6" fill="#22d3ee"/>
        <!-- Arms -->
        <rect x="0"  y="84" width="15" height="32" rx="7" fill="#7c3aed"/>
        <rect x="95" y="84" width="15" height="32" rx="7" fill="#7c3aed"/>
        <circle cx="7"  cy="120" r="7" fill="#ec4899"/>
        <circle cx="103" cy="120" r="7" fill="#ec4899"/>
        <!-- Legs -->
        <rect x="28" y="124" width="20" height="6" rx="3" fill="#5b21b6"/>
        <rect x="62" y="124" width="20" height="6" rx="3" fill="#5b21b6"/>
      </svg>
      <div style="color:#c084fc;font-size:1.2rem;font-weight:800;margin-top:8px;">CVBot 🤖</div>
      <div style="color:#7c3aed;font-size:0.72rem;">Bushra's AI Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div class="s-card">
        <h4>👩‍🎓 Student</h4>
        <p>🧑‍💼 <b>Bushra Qadir</b></p>
        <p>🏫 UMT Lahore</p>
        <p>📚 BS Software Engineering</p>
        <p>🎓 Graduating 2026</p>
    </div>

    <div class="s-card">
        <h4>💻 Tech Skills</h4>
        <ul>
            <li>🐍 Python · C++ · JavaScript</li>
            <li>🧠 TensorFlow · Keras · sklearn</li>
            <li>🔗 LangChain · FAISS · Gemini</li>
            <li>🎨 Streamlit · UML · C4 Model</li>
        </ul>
    </div>

    <div class="s-card">
        <h4>🚀 Projects</h4>
        <ul>
            <li>❤️ Heart Disease Prediction</li>
            <li>📊 Student Performance ML</li>
            <li>🚲 CycleGo Bike-Sharing</li>
            <li>💬 Sentiment Analysis (RNN)</li>
            <li>🤖 This CV Chatbot!</li>
        </ul>
    </div>

    <div class="s-card">
        <h4>💡 Ask me about</h4>
        <ul>
            <li>🎓 Education & courses</li>
            <li>🛠 Skills & tools</li>
            <li>📁 Projects & details</li>
            <li>🏆 Certifications</li>
            <li>🎯 Co-curricular activities</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cv-header">
    <h1>✨ Hi! I'm CVBot</h1>
    <p>
    Ask me anything about <b>Bushra Qadir's</b> CV — her education 🎓, skills 💻,
    projects 🚀, certifications 🏆, and more.<br>
    I only answer from the actual CV — no made-up facts, ever! 🛡️
    </p>
</div>
""", unsafe_allow_html=True)

# Stats
msg_count = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
st.markdown(f"""
<div class="stat-row">
    <div class="stat-box">
        <div class="icon">🚀</div>
        <div class="num">8</div>
        <div class="lbl">Projects</div>
    </div>
    <div class="stat-box">
        <div class="icon">💻</div>
        <div class="num">10+</div>
        <div class="lbl">Skills</div>
    </div>
    <div class="stat-box">
        <div class="icon">🏆</div>
        <div class="num">2</div>
        <div class="lbl">Certifications</div>
    </div>
    <div class="stat-box">
        <div class="icon">💬</div>
        <div class="num">{msg_count}</div>
        <div class="lbl">Questions Asked</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Key check
if "GOOGLE_API_KEY" not in os.environ:
    st.error("⚠️ GOOGLE_API_KEY not found. Add it to your .env file and restart.")
    st.stop()

# Load chain
with st.spinner("🔮 Loading knowledge base..."):
    rag_chain = build_chain()

# Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="big">🤖💜</div>
        <h3>Ready to chat!</h3>
        <p>Try: "What ML projects has Bushra built?" or "What degree does she have?"</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        icon = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"]):
            st.write(f"{icon} {msg['content']}")

# Input
if question := st.chat_input("💬 Ask anything about Bushra's CV..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(f"🧑‍💻 {question}")

    with st.chat_message("assistant"):
        with st.spinner("🔮 Searching CV..."):
            result = rag_chain.invoke({"input": question})
            answer = result["answer"]
        st.write(f"🤖 {answer}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

