"""
CVBot — Streamlit RAG Chatbot (Gemini + LangChain + FAISS)
Matches the logic already verified in My_own_Bot.ipynb
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

@st.cache_resource
def build_chain():
    # Load + split (same as notebook Cells 8, 10)
    documents = TextLoader(str(CV_PATH), encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embeddings + vector store (same as notebook Cells 14, 16)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if os.path.exists("vectorstore/index.faiss"):
        vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("vectorstore")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM + prompt (same as notebook Cells 6, 22)
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

# ---- UI ----
st.set_page_config(page_title="CVBot", page_icon="📄")
st.title("📄 CVBot")
st.caption("Ask questions about Bushra Qadir's CV — answers grounded in the document only.")

if "GOOGLE_API_KEY" not in os.environ:
    st.warning("Set GOOGLE_API_KEY in your .env file before chatting.")

rag_chain = build_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input("Ask about the CV..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    result = rag_chain.invoke({"input": question})
    answer = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
