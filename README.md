## Document RAG Chatbot (Streamlit + Groq)

This is a document chatbot using **RAG**:

- **UI**: Streamlit (upload files, chat, sources/citations)
- **LLM**: Groq (fast hosted inference)
- **Embeddings**: lightweight local embeddings (no Torch/ONNX DLLs)
- **Vector store**: lightweight on-disk index (no Chroma / ONNX)

### Prerequisites

- Python **3.13+**
- A Groq API key in `GROQ_API_KEY`

Set your key (PowerShell):

```powershell
setx GROQ_API_KEY "YOUR_KEY_HERE"
```

Then reopen your terminal (or restart your PC) so the environment variable is available.

### Setup (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### Run

```powershell
.\.venv\Scripts\activate
streamlit run app.py
```

### Notes

- Supported uploads: **PDF, DOCX, CSV**.
- Legacy **.doc** is not supported directly; convert to **.docx** or **.pdf** before upload.

