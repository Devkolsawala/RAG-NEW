📘 Local Document Summarizer + Q&A System

A local Retrieval-Augmented Generation (RAG) system built in Python 3.11, using:

🧠 Ollama (Phi-3 or Mistral models) for local LLM inference

🔍 ChromaDB for document vector storage

💬 Sentence Transformers (all-mpnet-base-v2) for text embeddings

📄 PyPDF2 for PDF document parsing

This project allows users to:

Upload documents (.pdf, .txt, .md)

Ask questions based only on the document content

Get summaries of documents

Run fully offline — no external API calls

🚀 Features

✅ Add local documents to the vector store
✅ Ask context-based questions from those documents
✅ Generate accurate summaries using Phi3
✅ Persist data locally with ChromaDB
✅ Fully private & offline – works entirely on your machine



🧩 Tech Stack


| Component           | Library / Tool                                        |
| ------------------- | ----------------------------------------------------- |
| **LLM Engine**      | [Ollama](https://ollama.ai) (`phi3`, `mistral`, etc.) |
| **Embeddings**      | `sentence-transformers (all-mpnet-base-v2)`           |
| **Vector Database** | `chromadb`                                            |
| **PDF Reader**      | `PyPDF2`                                              |
| **Language**        | Python 3.11                                           |


⚙️ Installation

1️⃣ Clone the Repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


2️⃣ Create and Activate Virtual Environment

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate


3️⃣ Install Requirements

pip install -r requirements.txt

4️⃣ Install and Run Ollama

Download Ollama from https://ollama.ai
 and start the Ollama service:

ollama run phi3

📄 Example Usage
Start the App
python main.py


You’ll see an interactive prompt:

============================================================
Local Document Q&A System (Ollama + Phi3)
============================================================

Commands:
  add <filepath>       - Add a document
  ask <question>       - Ask a question
  summarize <filepath> - Summarize a document
  clear                - Clear database
  quit                 - Exit



  Example Session
> add Resume.pdf
📄 Processing: Resume.pdf
✓ Added 12 chunks to vector store

> ask What skills does this person have?
💡 Answer: The person has skills in Python, machine learning, and data analysis.

> summarize Resume.pdf
📋 Summary:
This resume summarizes a software developer specializing in AI and ML.

📦 Folder Structure
RAG/
│
├── venv/                   # Virtual environment
├── main.py                 # Core RAG code
├── requirements.txt        # Dependencies
├── Resume.pdf              # Example document
└── .gitignore              # Ignore unnecessary files

🧠 Key Functions
Function	Description
add_document(file_path)	Adds a document to ChromaDB
answer_question(question)	Answers based on context from documents
summarize_document(file_path)	Summarizes a full document
clear_database()	Clears the local vector store
🧰 Requirements File

Example requirements.txt:

chromadb
sentence-transformers
PyPDF2
ollama

🧑‍💻 Developer Notes

Default Ollama model: phi3

ChromaDB persistence folder: ./chroma_db

Supports .pdf, .txt, .md, .rst

Recommended embedding model: all-mpnet-base-v2

🛡️ License

MIT License © 2025