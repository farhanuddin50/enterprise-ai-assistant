# ⚡ Enterprise AI Business Process Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) assistant designed to automate document QA, business workflows, and information extraction over internal policy files while strictly preventing AI hallucinations.

---

## 🛠️ Architecture & Tech Stack

```
[ Local PDF Documents ] ──► [ Text Splitter ] ──► [ Local HuggingFace Embeddings ]
                                                                │
                                                                ▼
[ User Streamlit UI ] ◄── [ Groq API (Llama 3.1) ] ◄── [ Local FAISS Vector Store ]
```

* **Orchestration Framework:** LangChain
* **LLM Engine:** Groq API (`llama-3.1-8b-instant`) — ultra-fast LPU inference
* **Vector Embeddings:** Local HuggingFace (`all-MiniLM-L6-v2`) — 100% free & privacy-preserving
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Frontend Web UI:** Streamlit
* **Environment & Config:** Python 3.10+, `python-dotenv`

---

## 🔑 Core Features & Design Principles

* **Zero-Hallucination Guardrails:** Enforces strict systemic boundaries using a grounded system prompt. Queries outside document context automatically yield: `"Information not available in policy documents."`
* **Source Attribution & Auditability:** Dynamic Streamlit UI provides collapsible page-level citations mapping every answer back to exact document chunks.
* **Hybrid Privacy/Performance Model:** Embedding generation runs entirely on local hardware, keeping raw text processing private before querying Groq's high-speed LLM endpoint.
* **Automated Evaluation Suite:** Built-in evaluation runner (`src/evaluator.py`) continuously benchmarks grounding accuracy and hallucination resistance against in-context and out-of-context trick queries.

---

## 📂 Project Directory Structure

```text
enterprise-ai-assistant/
├── data/
│   ├── raw/                  # Place source PDFs here (e.g. WorldEnergyOutlook2025.pdf)
│   ├── vectorstore/          # Saved FAISS index files
│   └── eval_reports/         # JSON outputs from automated evaluation tests
├── src/
│   ├── __init__.py           # Package marker
│   ├── ingest.py             # PDF loading, text chunking, and FAISS indexing
│   ├── rag_engine.py         # RAG pipeline logic, retriever setup & Groq LLM integration
│   └── evaluator.py          # Automated hallucination & grounding evaluation test suite
├── app.py                    # Interactive Streamlit Web Application UI
├── README.md                 # Project documentation
└── .env                      # API keys and local configuration
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.10+ installed and a free API Key from [Groq Console](https://console.groq.com).

### 2. Setup Virtual Environment

```powershell
# Clone the repository
git clone <your-repository-url>
cd enterprise-ai-assistant

# Create and activate virtual environment
python -m venv venv
.env\Scripts\Activate.ps1   # On Windows PowerShell
```

### 3. Install Dependencies

```powershell
pip install langchain langchain-community langchain-groq sentence-transformers faiss-cpu pypdf streamlit python-dotenv
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

## 💻 Execution Workflow

### Step 1: Ingest Policy Documents

Place your target PDF (e.g., `WorldEnergyOutlook2025.pdf`) into `data/raw/` and run:

```powershell
python src/ingest.py
```

*Creates semantic text chunks and builds the local FAISS index in `data/vectorstore/faiss_index`.*

### Step 2: Run Automated Evaluation Test

Test the pipeline against in-context questions and out-of-context trick queries:

```powershell
python -m src.evaluator
```

*Outputs pass/fail logs for hallucination tests and saves a summary report to `data/eval_reports/eval_results.json`.*

### Step 3: Launch Interactive Web Application

```powershell
streamlit run app.py
```

*Opens `http://localhost:8501` in your browser for real-time document QA.*

---

## 📊 Evaluation Results Example

| Test Query                                                  | Category       | Expected Output                      | Status    |
| :---------------------------------------------------------- | :------------- | :----------------------------------- | :-------- |
| *What are key policy goals regarding energy reliability?* | In-Context     | Grounded answer with page references | ✅ PASSED |
| *Which countries are gaining influence on market trends?* | In-Context     | List of countries from PDF context   | ✅ PASSED |
| *What was Uniper's net profit margin in Q3 2024?*         | Out-of-Context | "Information not available..."       | ✅ PASSED |
| *What is the recommended recipe for baking sourdough?*    | Out-of-Context | "Information not available..."       | ✅ PASSED |

---

## 🎯 Key Engineering & Architectural Highlights

* **Hybrid Privacy Architecture:** Engineered a split pipeline using local embedding models for private document vectorization alongside high-speed cloud LLMs for generation.
* **Deterministic Grounding:** Authored strict system prompts and low-temperature parameters to guarantee context-bound answers and automated out-of-scope fallback logic.
* **Automated AI Governance:** Built programmatic test suites (`src/evaluator.py`) to systematically audit response accuracy and verify hallucination resistance.
* **Production-Ready Interface:** Designed an interactive Streamlit UI complete with real-time vector status, metadata sidebars, and page-level source citation
* **RAG Architecture:** Engineered offline local embedding + cloud LLM split for privacy and cost efficiency.
* **Grounded Prompts:** Designed structured system prompts for accurate information extraction and automated fallback logic.
* **Testing & Governance:** Built automated test scripts evaluating response correctness and anti-hallucination guardrails.
* **Full-Stack AI Deployment:** Delivered an interactive enterprise web interface complete with metadata sidebars and audit citations.
