# META-Powered PDF Question & Answer Assistant  
*Retrieval-Augmented Generation (RAG) system using Meta Llama models*

This project is a hands-on implementation of a Retrieval-Augmented Generation (RAG) pipeline built in Google Colab using Meta Llama models.

The system allows a user to upload any PDF (policy, research paper, report, curriculum, etc.) and ask natural language questions. The model retrieves relevant document sections first, then generates answers grounded strictly in that context.

The goal of this project was to explore practical RAG architecture including semantic chunking, embedding-based retrieval, query rewriting, and hallucination control.

---

## 🚀 Overview

This assistant:

- Loads a PDF from Google Drive  
- Breaks it into semantic chunks  
- Generates embeddings for vector search  
- Uses Query Fusion retrieval for improved recall  
- Answers questions strictly based on retrieved context  

The system is designed to reduce hallucinations by enforcing strict context grounding and low-temperature inference.

---

## 📦 Prerequisites

Before running the notebook, you’ll need:

1. **Google Colab access**  
2. **OpenRouter API key**  
3. **A Google Drive link to a PDF**

---

## 1️⃣ Install Required Libraries

Run once per Colab session:

```python
%pip install -q \
  llama-index \
  llama-index-llms-openrouter \
  llama-index-embeddings-huggingface \
  llama-index-readers-file \
  llama-index-packs-fusion-retriever \
  sentence-transformers \
  nest-asyncio \
  requests

print("✅ Installation complete")
```

Installed components:

- `llama-index` – document indexing & querying  
- `llama-index-llms-openrouter` – Meta Llama integration  
- `llama-index-embeddings-huggingface` – embedding generation  
- `llama-index-readers-file` – PDF ingestion  
- `sentence-transformers` – semantic embeddings  
- `nest-asyncio` – async compatibility  
- `requests` – PDF download handling  

---

## 2️⃣ Configure the AI Model

This step:

- Loads your OpenRouter API key  
- Configures Meta Llama  
- Sets up embeddings  
- Registers everything with LlamaIndex  

Key configuration choices:

- **Low temperature (0.1)** to reduce hallucinations  
- Strict system prompt enforcing grounded answers  
- Token limits for concise responses  

```python
import os
from getpass import getpass
import nest_asyncio

nest_asyncio.apply()

from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

os.environ["OPENROUTER_API_KEY"] = getpass("Enter your OpenRouter API key: ")

llm = OpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="meta-llama/llama-3.3-70b-instruct:free",
    max_tokens=512,
    temperature=0.1,
    timeout=60,
    system_prompt=(
        "You are an expert RAG system that answers ONLY using the provided context. "
        "Never hallucinate. Never guess. If the answer is not in the context, say so. "
        "Provide short, clear, factual responses with 2–4 evidence bullets."
    ),
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = llm
Settings.embed_model = embed_model

print("✅ AI model and settings are ready")
```

---

## 3️⃣ Download the PDF

The system:

1. Accepts a Google Drive link  
2. Extracts the file ID  
3. Downloads the PDF  
4. Saves it as `data/source.pdf`  

```python
import os
import re
import requests

def download_pdf_from_drive(drive_url: str, save_path: str):
    match = re.search(r"/d/([A-Za-z0-9_-]+)", drive_url)
    if match:
        file_id = match.group(1)
    else:
        match = re.search(r"id=([A-Za-z0-9_-]+)", drive_url)
        if match:
            file_id = match.group(1)
        else:
            raise ValueError("❌ Could not extract file ID.")

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = requests.get(download_url)
    resp.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(resp.content)

drive_link = input("Paste your Google Drive PDF link: ").strip()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

pdf_path = os.path.join(DATA_DIR, "source.pdf")

download_pdf_from_drive(drive_link, pdf_path)
print("✅ PDF downloaded")
```

---

## 4️⃣ Semantic Chunking

Instead of naive fixed-size splits, this system uses embedding-based semantic segmentation.

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

semantic_embed = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

parser = SemanticSplitterNodeParser(
    buffer_size=3,
    breakpoint_percentile_threshold=95,
    embed_model=semantic_embed,
)

nodes = parser.get_nodes_from_documents(documents)

for n in nodes:
    n.metadata["source"] = pdf_path
    n.metadata["chunk_type"] = "semantic"

print(f"Created {len(nodes)} semantic nodes")
```

---

## 5️⃣ Query Fusion Retrieval

Uses Query Rewriting Retriever Pack to:

- Rewrite queries multiple times  
- Perform multiple searches  
- Fuse the best results  

```python
from llama_index.core.llama_pack import download_llama_pack

QueryRewritingRetrieverPack = download_llama_pack(
    "QueryRewritingRetrieverPack",
    "./query_rewriting_pack",
)

query_rewriting_pack = QueryRewritingRetrieverPack(
    nodes,
    chunk_size=256,
    vector_similarity_top_k=8,
    fusion_similarity_top_k=8,
    num_queries=6,
)

print("🚀 Query Fusion RAG Engine Ready")
```

---

## 6️⃣ Interactive Q&A

```python
def safe_rag_run(question, retries=3):
    for attempt in range(retries):
        try:
            resp = query_rewriting_pack.run(question)
            if resp is None or str(resp).strip() == "":
                raise ValueError("Empty response.")
            return resp
        except Exception as e:
            print(f"Retrying... ({attempt+1}/{retries})")

    return "❌ Failed to generate response."

while True:
    user_question = input("Enter your question: ").strip()

    if user_question.lower() == "end":
        break

    response = safe_rag_run(user_question)
    print("\nAnswer:\n", response)
```

---

## 🧠 Design Focus

- Grounded responses only  
- Reduced hallucination risk  
- Semantic retrieval  
- Query rewriting for improved recall  
- Structured RAG pipeline  

---

## 🔮 Future Improvements

- FastAPI backend wrapper  
- Persistent vector database (FAISS / Pinecone)  
- Source citation formatting  
- Logging & observability  
- Confidence scoring  

---

## 🎯 Purpose

This project demonstrates:

- Practical RAG architecture  
- Embedding-based document retrieval  
- Prompt engineering for hallucination control  
- Advanced retrieval strategies (Query Fusion)  
- Structured LLM system design  

It reflects hands-on experience building retrieval-backed AI systems rather than simple chatbot wrappers.

