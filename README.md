# CVBot — RAG Chatbot for My CV

**Student Name:** Bushra Qadir
**Student ID:** [s2024065009@umt.edu.pk]
**Project Title:** RAG-based CV Chatbot using LangChain + Gemini
**Chatbot Name:** CVBot
**Deployment Link:** https://cv-bot-6q5jcub5uwazanuk7qbxej.streamlit.app/

## Description
A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
my CV (`CV.txt`). Built with LangChain for orchestration, Google Gemini for
embeddings (`gemini-embedding-001`) and generation (`gemini-2.5-flash`), and
FAISS as the vector store. The bot answers strictly from CV content and
refuses to answer out-of-scope questions to avoid hallucination.

## How it works
1. **Load & Split** — `TextLoader` + `RecursiveCharacterTextSplitter`
2. **Embed & Store** — `GoogleGenerativeAIEmbeddings` + `FAISS`
3. **Retrieve** — top-3 relevant chunks via `as_retriever()`
4. **Generate** — grounded prompt + `create_retrieval_chain`, `temperature=0`

## Run locally
```bash
pip install -r requirements.txt
```
Create a `.env` file in the project folder with:
GOOGLE_API_KEY=your-gemini-key-here
Then run:
```bash
streamlit run app.py
```

## Files
- `app.py` — Streamlit RAG chatbot
- `CV.txt` — CV used as knowledge source
- `requirements.txt` — dependencies
- `My_own_Bot.ipynb` — development notebook
