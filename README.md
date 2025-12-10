# Welcome to Your First META-Powered Document Assistant

Today, we're going to build a simple but powerful AI tool: a **PDF Question-and-Answer system** powered by Meta's Llama AI models.

## What This Tool Will Allow You To Do

📄 **Upload any PDF**
- A company report
- A curriculum  
- A policy document
- A research paper

💬 **Ask natural questions**
- "What are the main goals in this document?"
- "What does this policy say about attendance?"
- "Summarise the key points in chapter one."

🤖 **Get accurate answers** — based only on the PDF itself

The AI does not guess or make things up. It searches the document, finds the right information, and gives you a clean answer.

---

## Before You Begin

Please ensure you have the following:

1. ✅ Access to Google Colab
2. ✅ OpenRouter API Key - [Watch this tutorial](https://www.youtube.com/watch?v=-X9DVzzxpAA)
3. ✅ Google Drive link to the PDF

🤖 *"May the code be with you"*

---

## Step 1 — Installing the Tools We Need

Before we can start building our AI system, we need to install a few essential tools. Think of this like installing apps on your phone before you can use them.

### What You'll Need

1. **LlamaIndex** — helps the AI read and understand documents
2. **OpenRouter connector** — allows us to use Meta's Llama AI model
3. **HuggingFace Embeddings** — helps the AI understand text by turning words into numbers
4. **Document readers** — so the notebook can open and read PDF files
5. **Fusion retriever tools** — advanced search tools that improve answer accuracy

This cell does not build the system yet — it only prepares the environment. Once everything is installed, the rest of the notebook will work smoothly.

> **Note:** After you run this cell once, you don't need to install again unless the runtime resets.

### 📋 Copy and paste this code into Cell 1:

```python
# Step 1 – Install all the tools our AI document assistant needs
#
# In this cell, we install the main libraries that make our RAG (Retrieval-Augmented Generation)
# system work. Think of this as installing the "apps" our notebook will use.
#
# Here is what each tool does:
#
# llama-index
#   - The main framework we use to read documents, break them into pieces, index them,
#     and search through them. It handles most of the heavy lifting for our RAG system.
#
# llama-index-llms-openrouter
#   - Allows us to connect to Meta's Llama models through OpenRouter, so our AI can
#     understand questions and generate answers.
#
# llama-index-embeddings-huggingface
#   - Helps convert text into embeddings (numbers that the AI can understand).
#     This is what allows the system to find the right parts of the PDF when you ask a question.
#
# llama-index-readers-file
#   - Gives the notebook the ability to read PDF files and other documents from your system.
#
# llama-index-packs-fusion-retriever
#   - Provides the "query fusion" tool we use to improve answer accuracy. It takes your question,
#     rewrites it in different ways, searches the document multiple times, and combines the results.
#
# sentence-transformers
#   - A library that helps with semantic understanding. It is used for intelligent document chunking.
#
# nest-asyncio
#   - Allows certain parts of the system to run smoothly inside Google Colab by letting
#     asynchronous code work without errors.
#
# requests
#   - Helps us download the PDF directly from a Google Drive link.
#
# After running this cell, all the required tools will be installed and ready.
# You only need to run this once per session.

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

---

## Step 2 — Setting Up the AI Model

Now that all our tools have been installed, this next cell is where we connect our notebook to the actual AI model that will answer our questions.

### In This Step, We:

✔ **Enter our OpenRouter API key**

✔ **Choose which Llama model we want to use**

✔ **Set rules for how the AI should behave**

✔ **Set up the embedding model** — This is a tool that helps the AI "understand" text by turning it into numerical patterns. It improves how well the system can find information inside the PDF.

### 📋 Copy and paste this code into Cell 2:

```python
# Step 2 – Connect to the AI model and set up how our system will think
#
# In this cell, we prepare the AI model that will answer our questions.
# Everything here is about giving the AI access, choosing the model we want,
# and making sure it behaves correctly.

import os
from getpass import getpass
import nest_asyncio
nest_asyncio.apply()

# Ask the user to enter their OpenRouter API key.
# This works like a password that allows us to use the Llama AI model.
os.environ["OPENROUTER_API_KEY"] = getpass("Enter your OpenRouter API key: ")

from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Here we choose which Llama model we want to use.
# - This model understands the questions we ask
# - It helps generate clear, readable answers
llm = OpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],  # Connect using our key
    model="meta-llama/llama-3.3-70b-instruct:free",  # Choose the AI model
    max_tokens=512,  # Maximum size of the answer
    temperature=0.1, # Lower = more accurate and less creative
    timeout=60,      # Give the model time to respond
    system_prompt=(
        # These are the rules we give the AI.
        # They keep the answers short, accurate, and based ONLY on the PDF.

        "You are an expert RAG system that answers ONLY using the provided context. "
        "Never hallucinate. Never guess. If the answer is not in the context, say so. "
        "Provide short, clear, factual responses with 2–4 evidence bullets."
    ),
)

# This model helps the AI understand text in the PDF by turning words into numbers.
# This step helps the system search the document more effectively.
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Finally, we tell the system to use the AI model and text-understanding model
# we just set up. This makes everything ready for the next steps.
Settings.llm = llm
Settings.embed_model = embed_model

print("✅ AI model and settings are ready to use")
```

---

## Step 3 — Loading the PDF From Google Drive

In this step, we give our AI system the actual PDF document it needs to learn from. Instead of uploading a file manually, we simply paste a Google Drive link, and the notebook automatically downloads the PDF for us.

### This Cell:

✔ **Asks you to paste the Google Drive link** — This is where your PDF is stored.

✔ **Extracts the file ID from the link** — The notebook figures out which file to download.

✔ **Downloads the PDF into the notebook** — Now the AI can read it.

✔ **Saves it in a folder so we can use it later** — This is the document the entire system will work with.

**In simple terms:** Step 3 brings the PDF into our AI workspace so the system can read it, understand it, and answer questions about it.

### 📋 Copy and paste this code into Cell 3:

```python
# Step 3 – Download the PDF from a Google Drive link
#
# This cell takes the Google Drive link you paste in, downloads the PDF,
# and saves it so our AI system can read it later.
# You don't need to upload anything manually — the notebook does it for you.

import requests
import re

def download_pdf_from_drive(drive_url: str, save_path: str):
    # Try to extract the file ID from the Google Drive link.
    # The file ID is the unique part of the link that tells Google which file to download.
    match = re.search(r"/d/([A-Za-z0-9_-]+)", drive_url)
    if match:
        file_id = match.group(1)
    else:
        # If the link is in a different format (?id=...), extract the ID from there instead.
        match = re.search(r"id=([A-Za-z0-9_-]+)", drive_url)
        if match:
            file_id = match.group(1)
        else:
            # If no file ID is found, let the user know the link is invalid.
            raise ValueError("❌ Could not extract file ID from the link.")

    # Build the direct download link for Google Drive using the file ID.
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    print(f"📥 Downloading PDF (file ID {file_id})...")

    # Download the actual PDF file from Google Drive.
    resp = requests.get(download_url)
    resp.raise_for_status()

    # Save the downloaded PDF to the folder we created.
    with open(save_path, "wb") as f:
        f.write(resp.content)

    print(f"✅ PDF downloaded → {save_path}")


# Ask the user to paste the Google Drive link of the PDF they want to use.
drive_link = input("📌 Paste your Google Drive PDF link here: ").strip()

# Create a folder called 'data' if it does not exist already.
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Save the downloaded PDF as 'source.pdf' inside the data folder.
pdf_path = os.path.join(DATA_DIR, "source.pdf")

# Download the PDF using the link provided.
download_pdf_from_drive(drive_link, pdf_path)
```

---

## Step 4 — Preparing the PDF for the AI to Understand

Now that we've downloaded the PDF, the next step is to help the AI make sense of it.

AI models cannot read a full PDF the way humans do, so we need to break the document into small, meaningful sections. This step is extremely important because it determines how accurately the AI will find answers later.

### In This Cell, We:

✔ **Load the PDF into the notebook** — The system opens the file so it can start reading the content.

✔ **Break the PDF into "semantic chunks"** — Instead of cutting the document randomly, we break it into smart, meaningful pieces — almost like dividing a book into paragraphs and sections. This helps the AI understand the document in a more natural way.

✔ **Add simple labels to each chunk** — This makes it easier for the AI to keep track of where each piece came from.

✔ **Prepare these pieces for searching** — These chunks will later be used by the AI to look up the right answer when you ask a question.

**In simple terms:** Step 4 teaches the AI how to read the PDF in a way it can understand and search through accurately.

### 📋 Copy and paste this code into Cell 4:

```python
# Cell 4 – Break the PDF into meaningful pieces the AI can understand
#
# In this step, we take the PDF we downloaded and prepare it so the AI can
# read it properly. The AI cannot understand one big block of text, so we
# break the document into smaller, meaningful sections (called "chunks").
#
# These chunks act like paragraphs or mini-sections that the AI can search
# through when answering your questions.

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load the PDF from the folder where we saved it.
# This reads the entire document into the system.
documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
print(f"📄 Loaded {len(documents)} document(s).")

# We use an embedding model to help the AI understand the meaning of the text.
# This model helps the system decide where natural breaks (sections) should be.
semantic_embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Create the tool that will split the PDF into smart, meaningful pieces.
# Instead of cutting randomly, it looks at the meaning of each sentence
# so the chunks feel more natural (like real sections).
parser = SemanticSplitterNodeParser(
    buffer_size=3,                     # Helps keep small related groups together
    breakpoint_percentile_threshold=95, # Controls how sensitive the splitting is
    embed_model=semantic_embed          # Uses the meaning of the text for splitting
)

# Use the parser to turn the document into these small, meaningful chunks.
nodes = parser.get_nodes_from_documents(documents)

# Add simple labels to each chunk so we know where it came from.
# This is helpful when the AI searches for answers.
for n in nodes:
    n.metadata["source"]     = pdf_path
    n.metadata["chunk_type"] = "semantic"

print(f"🔍 Created {len(nodes)} high-quality semantic nodes.")
```

---

## Step 5 — Building the AI Search Engine

Now that our PDF has been broken into small, meaningful pieces, we are ready to build the "search engine" part of our AI system. This is one of the most important steps.

### In This Cell, We Put Together the Tool That Helps the AI:

✔ Understand your question

✔ Rewrite your question in different ways

✔ Search through the PDF multiple times

✔ Combine the best results

✔ Give you the most accurate answer possible

This technique is called **Query Fusion**, and it comes from **Meta's Llama Cookbook**. It makes the AI much smarter and more reliable — especially when questions are complicated or phrased in different ways.

Think of it like having several researchers all look for the answer separately, then compare findings and give you the clearest final response.

**In simple terms:** Step 5 builds the brain of our search system — the part that helps the AI find the right information in the PDF, even if the question is difficult or worded differently.

### 📋 Copy and paste this code into Cell 5:

```python
# Cell 5 – Build the smart search system (Query Fusion)
#
# Now that we have broken the PDF into small, meaningful pieces (chunks),
# this cell builds the "search engine" that will help the AI find the
# right information when you ask a question.
#
# This uses a method called **Query Fusion**, which comes from Meta's Llama Cookbook.
# Query Fusion improves accuracy by:
#   - Rewriting your question in different ways
#   - Searching the document several times
#   - Combining the best results into one strong answer
#
# Think of it like asking several people the same question and then blending
# their answers to get the most reliable one.

from llama_index.core.llama_pack import download_llama_pack

# Download the special Query Fusion tool.
# This only downloads the first time — after that it loads from your system.
QueryRewritingRetrieverPack = download_llama_pack(
    "QueryRewritingRetrieverPack",
    "./query_rewriting_pack",
)

# Build the advanced search system using our document chunks.
# Here we set how many searches it should perform and how much information to gather.
query_rewriting_pack = QueryRewritingRetrieverPack(
    nodes,                        # The chunks created in Step 4
    chunk_size=256,               # The size of each text chunk to consider
    vector_similarity_top_k=8,    # How many top matches to look at in the first layer of search
    fusion_similarity_top_k=8,    # How many matches to merge for the final answer
    num_queries=6,                # Rewrites your question 6 different ways for a stronger search
)

print("🚀 Advanced Query Fusion RAG Engine Ready!")
```

---

## Step 6 — Asking Questions and Getting Answers

This is the **final** step — and the part you'll interact with the most.

Now that our AI system is fully set up, this cell creates a simple chat-like loop where you can:

✔ Type any question about your PDF

✔ Watch the AI search the document

✔ Receive a clear, accurate answer based only on the PDF

✔ Keep asking as many questions as you want

✔ Type "end" when you're finished

### Behind the Scenes, the AI Is:

1. Reading your question
2. Rewriting it in different ways
3. Searching the PDF for relevant pieces
4. Comparing the results
5. Giving you the best possible answer

This turns your PDF into a smart assistant that can explain, summarise, or find information instantly.

**In simple words:** Step 6 is where you finally get to talk to the AI. You ask questions, it gives answers, and the conversation continues until you decide to stop.

### 📋 Copy and paste this code into Cell 6:

```python
# Cell 6 – Ask questions and get answers from your PDF
#
# This final cell creates a simple question-and-answer loop.
# You can type any question about your PDF, and the AI will search the document
# and give you a clear, accurate answer based only on what is written in the PDF.
#
# The loop continues until you type "end", which stops the system.

def safe_rag_run(question, retries=3):
    # This function safely runs the AI search.
    # If something goes wrong (like a slow internet response),
    # it will try again up to 3 times before giving up.
    for attempt in range(retries):
        try:
            # Ask the AI to answer the question based on the PDF content.
            resp = query_rewriting_pack.run(question)

            # If the AI returns nothing, treat it as an error.
            if resp is None or str(resp).strip() == "":
                raise ValueError("Empty LLM response.")

            return resp

        except Exception as e:
            # If something goes wrong, show the error and retry.
            print(f"⚠️ Error: {e}")
            print(f"🔁 Retrying ({attempt+1}/{retries})...")

    # If all retries fail, return a message saying we couldn't get an answer.
    return "❌ Could not generate a valid answer after retries."

print("\nRAG Interactive Mode")
print("Ask any question about your PDF.")
print("Type 'end' to exit.\n")

# This loop keeps running until the user types "end".
while True:
    # Ask the user to type a question.
    user_question = input("🟦 Enter your question: ").strip()

    # If the user types "end", stop the loop and exit.
    if user_question.lower() == "end":
        print("\n👋 Session ended.")
        break

    print("\n🔍 Retrieving answer...\n")

    # Run the question through our safe search function.
    response = safe_rag_run(user_question)

    # Display the question and the AI's answer.
    print("\n──────────────────────────────────────────────")
    print("❓ QUESTION:")
    print(user_question)
    print("\n🧠 ANSWER:")
    print(response)
    print("──────────────────────────────────────────────\n")
```

---

## 🎉 Congratulations!

You've now built a complete AI-powered PDF Question-and-Answer system using Meta's Llama models.

### What You've Learned:

- How to install and configure AI libraries
- How to connect to OpenRouter and use Llama models
- How to process and chunk PDF documents intelligently
- How to build a query fusion RAG system
- How to create an interactive Q&A interface

### Next Steps:

- Try different PDFs to see how the system performs
- Experiment with different Llama models
- Adjust the chunk sizes and retrieval parameters
- Build upon this foundation to create more advanced AI applications

---

## Troubleshooting

### Common Issues:

**Installation errors:** Make sure you're running the notebook in Google Colab with a stable internet connection.

**API key errors:** Double-check that your OpenRouter API key is valid and has credits available.

**PDF download issues:** Ensure your Google Drive link is set to "Anyone with the link can view" and the file is a valid PDF.

**Empty responses:** The AI might not find relevant information. Try rephrasing your question or check if the information exists in the PDF.

---

## Support

If you encounter any issues or have questions:
1. Review the comments in each code cell
2. Check the error messages carefully
3. Ensure all prerequisites are met
4. Verify your API key and PDF link are correct

Happy coding! 🚀