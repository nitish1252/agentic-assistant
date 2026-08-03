# College Assistant

An **agentic RAG (Retrieval-Augmented Generation) chatbot** for a college that answers student questions about **academics**, **fees**, and **general topics**. It uses a **LangGraph** state machine to classify each query, route it to the right knowledge base, and generate a personalised answer.

The project ships with **two interfaces** built on the same underlying logic:

| File | Interface |
|------|-----------|
| `app.py` | **Streamlit** web app (chat UI in the browser) |
| `conditional_RAG.py` | **CLI** app (chat in the terminal) |

---

## Features

- **Intelligent query routing** — every question is classified as `academic`, `fee`, or `general`, then sent to the correct source.
- **Retrieval-Augmented Generation (RAG)** — answers are grounded in the official college PDFs (`academics_handbook.pdf` and `fee_structure.pdf`), reducing hallucinations.
- **Personalised answers** — the assistant knows the student's programme (BCA / BBA / B.Com (H)) and highlights programme-specific figures.
- **Chat memory** — the web app keeps full multi-turn conversation history and lets the user clear it anytime.
- **Query type badge** — the Streamlit UI shows a colour-coded badge (`ACADEMIC` / `FEE` / `GENERAL`) on every assistant reply.
- **Cached resources** — embeddings, vector stores, and the compiled graph are cached so PDFs are only loaded once.

---

## How it works (architecture)

The core is a **LangGraph state machine** — a directed graph of processing steps ("nodes") connected by "edges". Each student question flows through it as follows:

```
START
  │
  ▼
classifier ──► route_query (conditional edge)
  │                 │
  │                 ├── "academic" ──► academic_rag ──┐
  │                 ├── "fee"      ──► fee_rag     ──┤
  │                 └── "general"  ──► general      ──┤
  │                                                   ▼
  │                                                response ──► END
```

### The nodes

| Node | Role |
|------|------|
| `classifier` | Sends the latest user message to the LLM and asks it to return exactly one word: `academic`, `fee`, or `general`. Stores the result in `query_type`. |
| `academic_rag` | Runs a similarity search against the **academics handbook** vector store and saves the top-4 matching chunks as `retrieved_context`. |
| `fee_rag` | Same as above, but searches the **fee structure** vector store. |
| `general` | Skips retrieval entirely and sets `retrieved_context = "NO_RETRIEVAL_NEEDED"` so the LLM answers from its own knowledge. |
| `response` | Builds a prompt using the student's programme + retrieved context (or no context), calls the LLM, and returns the final answer. |

### The shared state

Each node reads from and writes to a shared state dictionary (`State`), which persists across all nodes for a given query:

```python
class State(TypedDict):
    programme: str                          # e.g. "BCA", selected by the student
    messages: Annotated[list, add_messages] # chat history (LangChain message format)
    query_type: str                         # "academic" | "fee" | "general"
    retrieved_context: str                  # top-k document chunks or NO_RETRIEVAL_NEEDED
```

The `messages` field uses `add_messages` as a reducer so new AI/user messages are **appended** to the history rather than overwriting it — this is what enables multi-turn conversation.

### The router

`route_query` is a **conditional edge**: it reads `query_type` produced by the classifier and branches to the correct retrieval node:

```python
def route_query(state: State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"
```

---

## Technical stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | [LangGraph](https://www.langchain.com/langgraph) | Graph state machine, nodes, and conditional routing |
| LLM | Groq (`llama-3.3-70b-versatile`) | Query classification + answer generation (via `ChatGroq`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Converts text chunks into vectors (`HuggingFaceEmbeddings`) |
| Vector store | FAISS (`faiss-cpu`) | Stores/retrieves embeddings for similarity search |
| Document loading | `PyPDFLoader` (pypdf) | Reads the PDF knowledge bases |
| Text splitting | `RecursiveCharacterTextSplitter` | Splits PDF text into 800-char chunks with 100-char overlap |
| Web UI | Streamlit | Chat interface, sidebar, session state, caching |
| Secrets | `python-dotenv` | Loads the `GROQ_API_KEY` from `.env` |

---

## Knowledge base

Two PDFs act as the retrieval source (must exist in the project root):

- **`academics_handbook.pdf`** — attendance rules, exams, grading, credits, promotion, course structure, degree requirements, etc.
- **`fee_structure.pdf`** — tuition, payment deadlines, refunds, late charges, scholarships, etc.

At startup, each PDF is:

1. Loaded with `PyPDFLoader`
2. Split into overlapping chunks (`chunk_size=800`, `chunk_overlap=100`)
3. Embedded and indexed in a FAISS vector store
4. Wrapped as a retriever that returns the top-4 most relevant chunks (`k=4`)

---

## Getting started

### 1. Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier available)
- The two knowledge-base PDFs in the project root

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The key is loaded via `load_dotenv()` and picked up automatically.

### 4. Run the app

**Web app (Streamlit):**

```bash
streamlit run app.py
```

**Terminal (CLI):**

```bash
python conditional_RAG.py
```

---

## Usage

**Web app (`app.py`):**
1. Select your programme (BCA / BBA / B.Com (H)) in the sidebar.
2. Type a question in the chat box, e.g. *"How many attendance hours do I need?"* or *"What is the BCA tuition fee?"*
3. Each answer shows a badge for the detected query type.
4. Use **🗑️ Clear Chat** to reset the conversation.

**CLI (`conditional_RAG.py`):**
1. Choose your programme (1, 2, or 3).
2. Ask questions; type `exit` or `quit` to leave.

### Example questions

| Category | Example |
|----------|---------|
| Academic | "What is the attendance requirement to sit for exams?" |
| Fee | "What is the refund policy for tuition fees?" |
| General | "Hello! Can you help me?" |

---

## Project structure

```
agentic-assistant/
├── app.py                 # Streamlit web app (UI + graph)
├── conditional_RAG.py     # CLI version of the same assistant
├── academics_handbook.pdf # Academic knowledge base (RAG source)
├── fee_structure.pdf      # Fee knowledge base (RAG source)
├── requirements.txt       # Python dependencies
├── .env                   # (not committed) GROQ_API_KEY
└── README.md              # This file
```

---

## Notes & limitations

- **`.env` is required** — the app will fail at runtime if `GROQ_API_KEY` is missing.
- **PDFs are required** — `academics_handbook.pdf` and `fee_structure.pdf` must be in the project root, or `PyPDFLoader` will raise an error.
- **First run is slow** — embedding the PDFs on startup can take a minute; the Streamlit app caches this with `@st.cache_resource` so it only happens once.
- **No streaming** — responses are returned all at once rather than token-by-token.
- **Local vector stores** — FAISS indexes are rebuilt in memory on every launch; there is no persistent index or database.
