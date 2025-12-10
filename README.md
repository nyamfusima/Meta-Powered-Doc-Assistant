# Welcome to your META-Powered PDF Question & Answer Assistant  
*Build a simple RAG system in Google Colab using Meta Llama models*

This notebook will help you build an AI assistant that can read any PDF you give it and answer questions **only** based on that PDF.

You will:

- Load a PDF (e.g. report, policy, curriculum, research paper)
- Ask natural questions about it
- Get short, accurate answers grounded in the document

The AI uses **Retrieval-Augmented Generation (RAG)**: it searches the PDF first, then answers using only what it finds (no guessing).

***

## 0. Prerequisites

Before you start, make sure you have:

1. **Google Colab access**  
   You will run all code in a Colab notebook.

2. **OpenRouter API key**  
   Follow this video to get a key:  
   https://www.youtube.com/watch?v=-X9DVzzxpAA

3. **A Google Drive link to your PDF**  
   Any PDF stored in Google Drive (set to “Anyone with the link” or at least accessible to your account).

***

## 1. Install all required libraries (Step 1)

In this step, you install all the tools your AI assistant needs.  
Run this cell **once per Colab session**.

**What you install:**

- `llama-index` – framework for reading, chunking, indexing and querying documents
- `llama-index-llms-openrouter` – connects to Meta Llama models via OpenRouter
- `llama-index-embeddings-huggingface` – creates embeddings for semantic search
- `llama-index-readers-file` – reads PDFs and other files
- `llama-index-packs-fusion-retriever` – Meta “Query Fusion” retriever pack
- `sentence-transformers` – semantic understanding and chunking
- `nest-asyncio` – fixes async issues in Colab
- `requests` – downloads the PDF from Google Drive

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

***

## 2. Connect to the AI model (Step 2)

Here you:

- Import core libraries
- Enter your **OpenRouter API key**
- Configure the **Llama model**
- Configure the **embedding model**
- Tell `llama-index` to use them

Run this cell **after** Step 1.

```python
import os
from getpass import getpass
import nest_asyncio

nest_asyncio.apply()

from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Ask for your OpenRouter API key (input is hidden like a password)
os.environ["OPENROUTER_API_KEY"] = getpass("Enter your OpenRouter API key: ")

# Configure the LLM (Meta Llama via OpenRouter)
llm = OpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="meta-llama/llama-3.3-70b-instruct:free",
    max_tokens=512,
    temperature=0.1,  # Low = more precise, less “creative”
    timeout=60,
    system_prompt=(
        "You are an expert RAG system that answers ONLY using the provided context. "
        "Never hallucinate. Never guess. If the answer is not in the context, say so. "
        "Provide short, clear, factual responses with 2–4 evidence bullets."
    ),
)

# Configure the embedding model
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Register both with LlamaIndex settings
Settings.llm = llm
Settings.embed_model = embed_model

print("✅ AI model and settings are ready to use")
```

***

## 3. Download the PDF from Google Drive (Step 3)

This step:

1. Asks you for a **Google Drive link** to your PDF
2. Extracts the **file ID** from the link
3. Downloads the PDF into a local `data/` folder
4. Saves it as `data/source.pdf`

Supported link formats include:

- `https://drive.google.com/file/d/<FILE_ID>/view?...`
- `https://drive.google.com/open?id=<FILE_ID>`

```python
import os
import re
import requests

def download_pdf_from_drive(drive_url: str, save_path: str):
    """
    Download a PDF from a Google Drive sharing link and save it locally.
    """
    # Try pattern: /d/<FILE_ID>/
    match = re.search(r"/d/([A-Za-z0-9_-]+)", drive_url)
    if match:
        file_id = match.group(1)
    else:
        # Try pattern: ?id=<FILE_ID>
        match = re.search(r"id=([A-Za-z0-9_-]+)", drive_url)
        if match:
            file_id = match.group(1)
        else:
            raise ValueError("❌ Could not extract file ID from the link.")

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    print(f"📥 Downloading PDF (file ID {file_id})...")

    resp = requests.get(download_url)
    resp.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(resp.content)

    print(f"✅ PDF downloaded → {save_path}")

# Ask for the Drive link
drive_link = input("📌 Paste your Google Drive PDF link here: ").strip()

# Make sure the data folder exists
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Local path for the PDF
pdf_path = os.path.join(DATA_DIR, "source.pdf")

# Download the PDF
download_pdf_from_drive(drive_link, pdf_path)
```

***

## 4. Break the PDF into semantic chunks (Step 4)

The AI cannot use one giant block of text.  
Here you:

- Load the PDF
- Use a **semantic splitter** to create “smart” chunks (not random splits)
- Label each chunk with simple metadata

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load the PDF as a document
documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
print(f"📄 Loaded {len(documents)} document(s).")

# Embedding model for semantic splitting (can reuse the same model name)
semantic_embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Create a semantic splitter
parser = SemanticSplitterNodeParser(
    buffer_size=3,
    breakpoint_percentile_threshold=95,
    embed_model=semantic_embed,
)

# Generate semantic nodes (chunks)
nodes = parser.get_nodes_from_documents(documents)

# Add simple metadata to each chunk
for n in nodes:
    n.metadata["source"] = pdf_path
    n.metadata["chunk_type"] = "semantic"

print(f"🔍 Created {len(nodes)} high-quality semantic nodes.")
```

***

## 5. Build the Query Fusion retriever (Step 5)

Now you build the **search engine** that powers your RAG system.  
It uses **Query Fusion**:

- Rewrites your question several ways
- Searches multiple times
- Fuses the best results

```python
from llama_index.core.llama_pack import download_llama_pack

# Download or load the Query Fusion pack
QueryRewritingRetrieverPack = download_llama_pack(
    "QueryRewritingRetrieverPack",
    "./query_rewriting_pack",
)

# Create the advanced retriever using your nodes
query_rewriting_pack = QueryRewritingRetrieverPack(
    nodes,                      # semantic chunks from Step 4
    chunk_size=256,
    vector_similarity_top_k=8,
    fusion_similarity_top_k=8,
    num_queries=6,              # number of query rewrites
)

print("🚀 Advanced Query Fusion RAG Engine Ready!")
```

***

## 6. Ask questions in an interactive loop (Step 6)

Finally, you create a simple chat loop:

- Type a question about the PDF
- The system runs the RAG pipeline
- You see a clear answer
- Type `end` to exit

```python
def safe_rag_run(question, retries=3):
    """
    Run the RAG pipeline with basic retry logic.
    """
    for attempt in range(retries):
        try:
            resp = query_rewriting_pack.run(question)

            if resp is None or str(resp).strip() == "":
                raise ValueError("Empty LLM response.")

            return resp

        except Exception as e:
            print(f"⚠️ Error: {e}")
            print(f"🔁 Retrying ({attempt+1}/{retries})...")

    return "❌ Could not generate a valid answer after retries."

print("\nRAG Interactive Mode")
print("Ask any question about your PDF.")
print("Type 'end' to exit.\n")

# Interactive Q&A loop
while True:
    user_question = input("🟦 Enter your question: ").strip()

    if user_question.lower() == "end":
        print("\n👋 Session ended.")
        break

    print("\n🔍 Retrieving answer...\n")

    # Run the question through the RAG pipeline
    response = safe_rag_run(user_question)

    print("\n──────────────────────────────────────────────")
    print("❓ QUESTION:")
    print(user_question)
    print("\n🧠 ANSWER:")
    print(response)
    print("──────────────────────────────────────────────\n")
```

***

## 7. Example questions you can try

Once everything is running, try questions like:

- “What are the main goals in this document?”
- “What does this policy say about attendance?”
- “Summarise the key points in chapter one.”
- “List all the responsibilities of students mentioned in this document.”
- “How is assessment described in this curriculum?”

***