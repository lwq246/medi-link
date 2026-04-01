# Medi-Link

AI-powered medical report analysis and chat assistant.

![Status](https://img.shields.io/badge/Status-Prototype-blue)
![Frontend](https://img.shields.io/badge/Frontend-Next.js%2016-black)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688)
![Vector%20DB](https://img.shields.io/badge/Vector%20DB-Qdrant-DC244C)
![LLMs](https://img.shields.io/badge/LLMs-Qwen3--VL%20%2B%20Qwen2.5-5C6BC0)

![Project Demo](./assets/medi-link.png)

## Overview

Medi-Link ingests a medical report (PDF/image), extracts marker data and report metadata, generates a clinical summary, and supports follow-up Q&A over the stored report context.

Current model setup:

- Summary and vision extraction: `Qwen/Qwen3-VL-8B-Instruct`
- Chat assistant: `Qwen/Qwen2.5-7B-Instruct`

The backend stores structured report context in Qdrant and uses Supabase for auth + report/chat persistence.

## Architecture

1. Upload report from frontend dashboard.
2. Backend converts PDF first page to image (if needed).
3. Qwen3-VL extracts JSON metadata + lab markers from image.
4. Backend computes deterministic marker status (`Low`/`High`/`Normal`/`N/A`) from reference ranges.
5. Qwen3-VL generates final markdown summary from verified marker JSON + medical context.
6. Backend stores:

- Summary + file pointer in Supabase (`reports` table)
- Chat logs in Supabase (`chat_messages` table)
- Metadata/marker facts/summary embeddings in Qdrant (`patient_data` collection)

7. Chat endpoint retrieves relevant report vectors from Qdrant and answers with Qwen2.5.

## Repository Layout

- `frontend/`: Next.js app (dashboard, upload UI, chat UI)
- `backend/`: FastAPI service (upload/chat APIs, Qwen calls, vector persistence)
- `docker-compose.yaml`: Qdrant service
- `qdrant_storage/`: local Qdrant data volume

## Prerequisites

- Python 3.11+ (3.12 also works)
- Node.js 18+
- Docker Desktop
- Redis running locally on `localhost:6379`
- Supabase project with required tables/storage bucket
- Hugging Face API token with access to selected models

## Environment Variables

Create `backend/.env`:

```env
HF_TOKEN=your_huggingface_token

SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

## Local Setup

### 1. Start infrastructure

From project root:

```bash
docker compose up -d
```

This starts Qdrant at:

- HTTP API: `http://localhost:6335`
- gRPC: `localhost:6334`

Start Redis separately (must be on `localhost:6379`).

### 2. Run backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn python-dotenv huggingface_hub qdrant-client sentence-transformers supabase redis pymupdf python-multipart
uvicorn main:app --reload
```

Backend runs on `http://localhost:8000`.

### 3. Run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`.

## API Endpoints

- `POST /upload`
  - Auth required (Bearer token from Supabase session)
  - Accepts medical report file (`multipart/form-data`)
  - Returns saved report row (includes `ai_analysis`)

- `POST /chat`
  - Auth required
  - Body:

```json
{
  "report_id": "<report-id>",
  "question": "What is abnormal in this report?"
}
```

## Troubleshooting

### Qdrant connection refused (`WinError 10061`)

If backend startup fails with Qdrant connection errors:

1. Confirm container is running:

```bash
docker ps
```

2. Confirm mapped port `6335` is available:

```bash
curl http://localhost:6335/collections
```

3. Start (or restart) Qdrant:

```bash
docker compose up -d
```

### Uvicorn flag typo

Use:

```bash
uvicorn main:app --reload
```

Not:

```bash
uvicorn main:app --relod
```

## Notes

- Current backend uses direct Hugging Face inference calls.
- LangGraph/Gemini pipeline was removed in favor of single-model Qwen3-VL report analysis.
