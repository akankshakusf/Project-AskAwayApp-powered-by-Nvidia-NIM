# 📄 Nvidia Nim-Powered Document Analyzer: AskAwayApp

## 📌 Overview

**AskAwayApp** is an **AI-driven document analysis tool** that enables users to ask questions about PDF documents and retrieve accurate responses. It leverages **NVIDIA NIM Llama-3.3-70B**, **FAISS for vector storage**, and **NVIDIA Embeddings** for efficient document retrieval.

This application helps users **quickly analyze, search, and extract insights** from large document collections.

---

## 🌟 Features

✅ **AI-Powered Q&A on PDFs** - Ask questions and get relevant answers from your documents.  
✅ **NVIDIA NIM Integration** - Uses **Meta Llama-3.3-70B** for accurate responses.  
✅ **Efficient Embeddings** - Converts PDF documents into vectorized representations for fast retrieval.  
✅ **FAISS-Based Vector Store** - Enables fast and efficient document similarity search.  
✅ **Multi-Document Support** - Analyzes multiple PDFs at once.  
✅ **User-Friendly Interface** - Built with **Streamlit** for ease of use.  

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **LLM**: NVIDIA NIM (Meta Llama-3.3-70B)  
- **Knowledge Source**: PDF documents  
- **Frameworks & Libraries**:
  - LangChain  
  - FAISS (Vector Search)  
  - NVIDIA AI Endpoints  
  - PyPDFLoader (For PDF Processing)  

---

## 🔄 How It Works

1️⃣ **Upload & Process PDFs**:  
   - The app loads PDF files from the `us_census` directory.  
   - Documents are split into smaller chunks using **LangChain's RecursiveCharacterTextSplitter**.  
   - These chunks are converted into **vector embeddings** using **NVIDIA Embeddings**.  
   - FAISS stores the embeddings for quick retrieval.

2️⃣ **User Queries the Document**:  
   - The user enters a question in the text input field.  
   - The **FAISS Vector Store** retrieves relevant document snippets based on similarity search.  

3️⃣ **AI-Generated Response**:  
   - The retrieved documents are passed to **Meta Llama-3.3-70B** via **NVIDIA NIM**.  
   - The model generates **context-aware, accurate answers**.  

4️⃣ **Display Search Results**:  
   - The app shows the AI-generated response.  
   - Users can also view **similar documents** found during the retrieval process.  

---

## 🚀 Business Use Cases

🔹 **Legal Document Analysis**: Quickly search and extract relevant sections from contracts or policies.  
🔹 **Financial & Census Data Extraction**: Analyze census reports and financial statements efficiently.  
🔹 **Academic & Research Assistance**: Summarize and search academic papers or government reports.  
🔹 **Enterprise Knowledge Management**: Retrieve company policies and internal documentation seamlessly.  

