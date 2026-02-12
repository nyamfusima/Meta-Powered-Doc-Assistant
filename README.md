META-Powered PDF Question & Answer Assistant

Retrieval-Augmented Generation (RAG) system using Meta Llama models

This project is a hands-on implementation of a Retrieval-Augmented Generation (RAG) pipeline built in Google Colab using Meta Llama models.

The system allows a user to upload any PDF (policy, research paper, report, curriculum, etc.) and ask natural language questions. The model retrieves relevant document sections first, then generates answers grounded strictly in that context.

The goal of this project was to explore practical RAG architecture including semantic chunking, embedding-based retrieval, query rewriting, and hallucination control.

Overview

This assistant:

Loads a PDF from Google Drive

Breaks it into semantic chunks

Generates embeddings for vector search

Uses Query Fusion retrieval for improved recall

Answers questions strictly based on retrieved context

The system is designed to reduce hallucinations by enforcing strict context grounding and low-temperature inference.

0. Prerequisites

Before running the notebook, you’ll need:

Google Colab access
The notebook is designed to run in Colab.

OpenRouter API key
Used to access Meta Llama models via OpenRouter.

A Google Drive link to a PDF
Any PDF stored in Google Drive (accessible via link or your account).

1. Install Required Libraries

This step installs all necessary dependencies.
Run once per Colab session.

Installed components include:

llama-index – document indexing and querying framework

llama-index-llms-openrouter – integration with Meta Llama models

llama-index-embeddings-huggingface – embedding generation

llama-index-readers-file – PDF reader

llama-index-packs-fusion-retriever – advanced retrieval (Query Fusion)

sentence-transformers – embedding support

nest-asyncio – async compatibility in Colab

requests – PDF download handling

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

2. Configure the AI Model

This step:

Imports required libraries

Loads your OpenRouter API key

Configures the Meta Llama model

Sets up the embedding model

Registers both with LlamaIndex

Key configuration choices:

Low temperature (0.1) to reduce hallucinations

Strict system prompt enforcing grounded responses

Token limits to keep responses concise and factual

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

3. Download the PDF

The system:

Accepts a Google Drive link

Extracts the file ID

Downloads the PDF locally

Stores it as data/source.pdf

This keeps document ingestion simple and reproducible.

# (code unchanged)

4. Semantic Chunking

Instead of naive fixed-length splitting, this project uses embedding-based semantic segmentation.

Process:

Load PDF

Generate embeddings

Identify semantic breakpoints

Create metadata-tagged nodes

This improves retrieval precision and contextual relevance.

# (code unchanged)

5. Query Fusion Retrieval

The system uses the Query Rewriting Retriever Pack, which:

Rewrites the user query multiple times

Performs multiple vector searches

Fuses the best results

This increases recall and reduces the risk of missing relevant context.

# (code unchanged)

6. Interactive Q&A Loop

The final stage provides an interactive loop:

User enters a question

The system retrieves relevant chunks

The LLM generates a grounded response

Retry logic handles transient failures

# (code unchanged)

Example Questions

“What are the main objectives in this document?”

“Summarise the compliance requirements mentioned.”

“What responsibilities are outlined?”

“How is assessment described?”

Design Considerations

This implementation focuses on:

Grounded responses only

Reduced hallucination risk

Semantic retrieval instead of naive search

Structured retrieval pipeline

Reproducible workflow

Possible Extensions

Future improvements could include:

FastAPI backend wrapper

Persistent vector database (FAISS / Pinecone)

Source citation formatting

Query logging & observability

Confidence scoring

Role-based access control

Purpose of This Project

This project was built to explore and demonstrate:

Practical RAG architecture

Embedding-based document retrieval

Prompt control for hallucination mitigation

Advanced retrieval techniques (Query Fusion)

Integration of open LLM infrastructure into structured pipelines

It reflects a hands-on approach to building retrieval-backed AI systems rather than simple chatbot wrappers.
