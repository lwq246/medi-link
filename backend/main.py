import shutil
import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient, models # Add 'models' here
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import Agents
from backend.medical_service import analyze_patient_report, answer_medical_question

# OCR Tools
import pytesseract
from PIL import Image
import pypdf

# === CONFIG ===
# POINT THIS TO YOUR TESSERACT PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()

# 1. Load AI Model
print("Loading AI Model for API...")
model = SentenceTransformer('all-mpnet-base-v2')

# 2. Connect to Qdrant
qdrant = QdrantClient(host="localhost", port=6335)

# Define Collections
KNOWLEDGE_COLLECTION = "medical_knowledge"
PATIENT_COLLECTION = "patient_data"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""], # Try to split by paragraphs first, then lines
    length_function=len,
)

class ChatRequest(BaseModel):
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_db():
    # Ensure patient collection exists
    if not qdrant.collection_exists(PATIENT_COLLECTION):
        qdrant.create_collection(
            collection_name=PATIENT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 0. Clear patient collection before storing new data
    qdrant.delete(
        collection_name=PATIENT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter()
        )
    )

    # 1. Save and OCR
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = ""
    try:
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(temp_filename)
            extracted_text = pytesseract.image_to_string(image)
        elif file.filename.lower().endswith('.pdf'):
            reader = pypdf.PdfReader(temp_filename)
            for page in reader.pages:
                extracted_text += page.extract_text() + "\n"
    except Exception as e:
        if os.path.exists(temp_filename): os.remove(temp_filename)
        raise HTTPException(status_code=500, detail=str(e))
    
    os.remove(temp_filename)

    # === NEW: CHUNKING STRATEGY ===
    # We split the messy OCR text into overlapping chunks
    chunks = text_splitter.split_text(extracted_text)
    
    # 2 & 3. Vectorize and Store each chunk
    points = []
    for i, chunk in enumerate(chunks):
        # Vectorize the specific chunk
        chunk_vector = model.encode(chunk).tolist()
        
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk_vector,
            payload={
                "filename": file.filename, 
                "text": chunk,
                "chunk_index": i
            }
        ))
    
    # Upsert all chunks at once
    qdrant.upsert(collection_name=PATIENT_COLLECTION, points=points)

    # 4. RAG SEARCH 
    # We use the first chunk to find medical guidelines (usually contains the summary/markers)
    # If chunks are empty, we use the original text vector as fallback
    query_vector = model.encode(chunks[0] if chunks else extracted_text).tolist()
    
    results_object = qdrant.query_points(
        collection_name=KNOWLEDGE_COLLECTION,
        query=query_vector,
        limit=2, # Increased to 2 for better context
        with_payload=True
    )
    search_results = results_object.points
    rag_text_context = "\n".join([hit.payload['text'] for hit in search_results])

    # 5. RUN AGENTS
    print(f"Starting Multi-Agent System with {len(chunks)} chunks...")
    try:
        # We pass the full extracted_text to the agents 
        # (Internal truncation in agents.py will handle the safety)
        final_analysis = analyze_patient_report(extracted_text, rag_text_context)
    except Exception as e:
        print(f"Agent Error: {e}")
        final_analysis = f"AI Analysis Failed: {str(e)}"

    return {
        "filename": file.filename,
        "chunks_processed": len(chunks),
        "raw_text_preview": extracted_text[:100] + "...",
        "ai_analysis": final_analysis 
    }

@app.post("/chat")
async def chat_with_data(request: ChatRequest):
    question_vector = model.encode(request.question).tolist()
    
    # 1. Search Patient Data (UPDATED)
    patient_res = qdrant.query_points(
        collection_name=PATIENT_COLLECTION,
        query=question_vector,
        limit=3,
        with_payload=True
    ).points
    
    # 2. Search Medical Knowledge (UPDATED)
    medical_res = qdrant.query_points(
        collection_name=KNOWLEDGE_COLLECTION,
        query=question_vector,
        limit=2,
        with_payload=True
    ).points
    
    # 3. Combine Context
    context_str = "--- PATIENT REPORT SNIPPETS ---\n"
    context_str += "\n".join([hit.payload['text'] for hit in patient_res])
    context_str += "\n\n--- MEDICAL KNOWLEDGE ---\n"
    context_str += "\n".join([hit.payload['text'] for hit in medical_res])
    
    # 4. Ask Agent
    try:
        answer = answer_medical_question(request.question, context_str)
    except Exception as e:
        answer = f"Error: {str(e)}"
        
    return {"answer": answer}