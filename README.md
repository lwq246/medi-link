# 🏥 Medi-Link: Autonomous Multi-Agent Health Intelligence Platform

![Status](https://img.shields.io/badge/Status-Prototype-blue)
![AI Model](https://img.shields.io/badge/AI-Qwen2.5--1.5B-violet)
![Stack](https://img.shields.io/badge/Stack-Next.js_|_FastAPI_|_Qdrant-green)

> **Transforming messy, unstructured medical reports into actionable, personalized health insights.**

---

## 📖 Overview

![Project Demo](./assets/demo-screenshot.png)

**Medi-Link** is an end-to-end AI system designed to bridge the gap between complex medical data and patient understanding. It ingests raw medical documents (PDFs/Images), digitizes them using OCR, and utilizes a **Multi-Agent RAG System** to provide safe, context-aware health summaries.

Built with **privacy** and **accuracy** in mind, it uses **Qdrant** for vector memory and **Qwen2.5-1.5B-Instruct** for reasoning, ensuring that patients receive intelligible summaries backed by medical context, not just generic LLM hallucinations.

---

## 🏗️ Architecture & Features

### 1. 👁️ Intelligent Ingestion Engine (The "Messy Data" Solver)

- **OCR Pipeline:** Utilizes **Tesseract OCR** (wrapped in Python) to extract text and tabular data from scanned images and PDFs.
- **Preprocessing:** Cleans noise and standardizes raw text for analysis.

### 2. 🧠 Retrieval-Augmented Generation (RAG)

- **Vector Database:** **Qdrant** (running via Docker) stores embeddings of patient data and medical guidelines.
- **Hybrid Search:** Retrieves relevant medical definitions dynamically based on the specific biomarkers found in the user's report.
- **Embeddings:** Uses `sentence-transformers/all-mpnet-base-v2` for high-fidelity semantic search.

### 3. 🤖 Agentic Reasoning Core (Multi-Agent System)

Powered by **Qwen2.5-1.5B-Instruct** via Hugging Face Inference API.

- **Agent 1 (The Analyst):** Correlates extracted lab values with RAG-retrieved medical guidelines to identify abnormalities.
- **Agent 2 (The Safety Critic):** A specialized guardrail agent that reviews the draft for hallucinations, ensures empathetic tone, and enforces medical disclaimers.
- **Agent 3 (The Chat Assistant):** Allows users to ask follow-up questions (e.g., _"Is my hemoglobin dangerous?"_) with context-aware answers.

---

## 🛠️ Tech Stack

### **Frontend**

- **Framework:** Next.js 14 (App Router)
- **Styling:** Tailwind CSS
- **Language:** TypeScript

### **Backend (AI Microservice)**

- **Framework:** FastAPI (Python)
- **Orchestration:** LangChain Core
- **Inference:** Hugging Face Serverless API (Router)
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **OCR:** Pytesseract / PIL

### **Infrastructure**

- **Database:** Qdrant (Vector DB) via Docker
- **Environment:** Python Virtual Environment (venv)

---

## 🚀 Getting Started

### Prerequisites

- **Docker Desktop** (for Qdrant)
- **Python 3.10+**
- **Node.js 18+**
- **Tesseract OCR** installed on your machine.

### 1. Clone the Repository

```bash
git clone https://github.com/lwq246/medi-link.git
cd medi-link
```
