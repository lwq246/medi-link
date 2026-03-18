# 🏥 Medi-Link: AI-Powered Medical Report Summarizer

![Status](https://img.shields.io/badge/Status-Prototype-blue)
![AI Model](https://img.shields.io/badge/AI-Qwen2.5--1.5B-violet)
![Stack](https://img.shields.io/badge/Stack-Next.js_|_FastAPI_|_Qdrant-green)
![Framework](https://img.shields.io/badge/Framework-LangChain_LCEL-orange)

> **Transforming messy, unstructured medical reports into actionable, personalized health insights using LLM-powered retrieval and recursive chunking.**

---

## 📖 Overview

![Project Demo](./assets/medi-link.png)

**Medi-Link** is an end-to-end AI system designed to bridge the gap between complex medical data and patient understanding. It ingests raw medical documents (PDFs/Images), digitizes them using OCR, and utilizes an **LLM-powered retrieval-augmented generation (RAG) pipeline** to provide safe, context-aware health summaries.

Built with **privacy** and **accuracy** in mind, it uses **Qdrant** for vector memory and **Qwen2.5-1.5B-Instruct** for reasoning, ensuring that patients receive intelligible summaries backed by verified medical context, minimizing LLM hallucinations.

---

## 🏗️ Key Engineering Highlights

### 1. 👁️ Intelligent Ingestion & Recursive Chunking

Medical documents are highly sensitive to context. To solve the "Messy Data" problem:

- **OCR Pipeline:** Utilizes **Tesseract OCR** to extract text and tabular data from scanned images.
- **Recursive Character Splitting:** Implemented a chunking strategy with a **1000-character window and 200-character overlap**. This ensures that clinical markers (e.g., "Hemoglobin") stay semantically linked to their values and reference ranges across vector boundaries.

### 2. 🧠 Vector Retrieval-Augmented Generation (RAG)

- **Vector Database:** **Qdrant** (running via Docker) manages high-dimensional embeddings.
- **Semantic Retrieval:** Uses `sentence-transformers/all-mpnet-base-v2` to perform Cosine Similarity searches, grounding the AI's responses in verified medical guidelines.
- **Multi-Collection Architecture:** Separate collections for **Medical Knowledge** (General Guidelines) and **Patient Data** (Specific Reports) ensure structured retrieval.

### 3. 🤖 LLM-Driven Medical Workflows (LCEL)

This project uses **LangChain Expression Language (LCEL)** to build modular, composable LLM pipelines for medical data analysis:

- **Health Report Analyzer:** Summarizes and interprets patient lab results using retrieved medical guidelines.
- **Medical Q&A Assistant:** Answers user questions based on both patient data and trusted medical context.
- **Composable Design:** The LCEL pattern (`Prompt | LLM | StrOutputParser`) enables easy model or prompt updates without changing core logic.

---

## 🛠️ Tech Stack

### **Frontend**

- **Framework:** Next.js 14 (App Router)
- **State Management:** React Hooks (useState/useEffect)
- **Styling:** Tailwind CSS

### **Backend (AI Microservice)**

- **Framework:** FastAPI (Python)
- **Orchestration:** LangChain Core & LangChain-HuggingFace
- **Vector DB:** Qdrant (Docker)
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Conversational Task)
- **OCR:** Pytesseract / PIL

---
