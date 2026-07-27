Anshu.AI — Personal AI Assistant

An AI-powered talking portfolio that lets recruiters interactively ask questions about my background, skills, and projects. Built on a RAG pipeline with hybrid search, hybrid memory, and live ResumeIQ integration.

Live: [Railway deployment link]  |  ResumeIQ: agentic-resume-analyzer.netlify.app

What it does
Answers recruiter questions about my skills, projects, education and experience in real time
Uses hybrid search (BM25 + dense retrieval via RRF) for accurate chunk retrieval
Streams responses word-by-word using Server-Sent Events
Integrates ResumeIQ as a live tool — paste a JD and get an ATS match analysis
Maintains conversation context using a hybrid memory system
Architecture
User query
    │
    ├── Input guardrails (prompt injection detection)
    │
    ├── Embedding (all-MiniLM-L6-v2)
    │
    ├── Hybrid Retrieval
    │   ├── Dense: ChromaDB semantic search
    │   ├── Sparse: BM25 keyword search
    │   └── RRF: Reciprocal Rank Fusion
    │
    ├── Hybrid Memory
    │   ├── Running summarization (every 6 turns)
    │   ├── Semantic search on history
    │   └── Recent turns (last 3)
    │
    ├── Gemini 3.5 Flash (streaming)
    │   └── Groq fallback on 503/429
    │
    └── Output guardrails → Stream to user
Tech Stack
Layer	Technology
Backend	FastAPI
Vector DB	ChromaDB
Embeddings	sentence-transformers (all-MiniLM-L6-v2)
Sparse Retrieval	BM25 (rank-bm25)
LLM	Gemini 3.5 Flash
Fallback LLM	Groq (llama / gpt-oss)
Memory	In-memory + Groq summarization
Frontend	Vanilla HTML/CSS/JS
Deployment	Docker + Railway
RAG Evaluation

Custom LLM-as-judge evaluation framework across 31 test queries:

Metric	Score
Retrieval Precision@5	90.32%
Answer Relevancy	98.39%
Faithfulness	87.19%
Answer Accuracy	96.77%
Context Recall	77.96%
Alignment Evaluation

Constitution-based safety evaluation:

Metric	Score
Adversarial Robustness	80.00%
User Trust Score	80.00%
Value Violation Rate	40.00% (lower is better)
Misleading Omission Score	62.50% (lower is better)
Setup
bash
# clone and setup venv
git clone https://github.com/AnshuuSharma/Personal_Assistant
cd Personal_Assistant
python -m venv assistant
assistant\Scripts\activate
pip install -r backend/requirements.txt

# add environment variables
cp .env.example .env
# fill in GEMINI_API_KEY, GROQ_API_KEY, RESUMEIQ_URL

# build vector database
python backend/scripts/prepare_chunks.py
python backend/scripts/ingest.py

# run
uvicorn backend.app:app --host 0.0.0.0 --port 8000

Open frontend/index.html with Live Server.

Docker
bash
docker build -t anshu-assistant .
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  -e RESUMEIQ_URL=your_url \
  anshu-assistant
Project Structure
personal_assistant/
├── backend/
│   ├── app.py                  # FastAPI endpoints
│   ├── core/
│   │   ├── embeddings.py       # sentence transformer
│   │   ├── retriever.py        # hybrid search + RRF
│   │   ├── llm.py              # Gemini streaming + Groq fallback
│   │   ├── memory.py           # hybrid memory system
│   │   └── guardrails.py       # input/output safety
│   ├── agent/
│   │   └── router.py           # RAG pipeline router
│   ├── tools/
│   │   └── resumeiq_tool.py    # ResumeIQ API integration
│   ├── data/
│   │   └── raw/anshu_info.txt  # personal data chunks
│   └── scripts/
│       ├── prepare_chunks.py   # split data into chunks
│       ├── ingest.py           # embed and load to ChromaDB
│       ├── evaluate_rag.py     # RAG evaluation framework
│       └── alignment_eval.py   # alignment evaluation
├── frontend/
│   └── index.html
├── vectorDB/                   # ChromaDB files
├── Dockerfile
└── .env.example
