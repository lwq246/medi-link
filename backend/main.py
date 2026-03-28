import os
import uuid
import shutil
import base64
import traceback
import fitz # PyMuPDF
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Vector DB & AI
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Supabase & Auth
from supabase import create_client, Client
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import your AI Agents
from medical_service import analyze_patient_report, answer_medical_question

# 1. LOAD ENVIRONMENT & CONFIG
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# 2. INITIALIZE CLIENTS
app = FastAPI()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
qdrant = QdrantClient(host="localhost", port=6335) 
security = HTTPBearer()

print("Loading AI Embedding Model...")
embed_model = SentenceTransformer('all-mpnet-base-v2')

# 3. MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. DATA MODELS
class ChatRequest(BaseModel):
    question: str
    report_id: str

# 5. AUTH DEPENDENCY
def get_current_user(authorization: HTTPAuthorizationCredentials = Depends(security)):
    token = authorization.credentials
    try:
        user_res = supabase.auth.get_user(token)
        if not user_res or not user_res.user:
            raise HTTPException(status_code=401, detail="Invalid Session")
        return user_res.user.id
    except Exception as e:
        raise HTTPException(status_code=401, detail="Unauthorized")

# 6. DATABASE INIT
KNOWLEDGE_COLLECTION = "medical_knowledge"
PATIENT_COLLECTION = "patient_data"

def ensure_collection(collection_name: str):
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

@app.on_event("startup")
def startup_db():
    ensure_collection(PATIENT_COLLECTION)
    ensure_collection(KNOWLEDGE_COLLECTION)

# 7. ROUTES

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    img_path = None
    ensure_collection(PATIENT_COLLECTION)

    try:
        # Save file locally
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Convert to Image for Vision
        if file.filename.lower().endswith('.pdf'):
            doc = fitz.open(temp_path)
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)) # Balanced resolution
            img_path = temp_path + ".jpg"
            pix.save(img_path)
            doc.close()
        else:
            img_path = temp_path

        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 2. Upload to Supabase Storage
        storage_path = f"{user_id}/{uuid.uuid4()}_{file.filename}"
        with open(temp_path, "rb") as f:
            supabase.storage.from_("medical-reports").upload(storage_path, f)

        # 3. Pull Knowledge Base context
        search_res = qdrant.query_points(KNOWLEDGE_COLLECTION, query=[0]*768, limit=10).points
        rag_context = "\n".join([hit.payload['text'] for hit in search_res])

        # 4. Run LangGraph (Returns Report Text AND Structured Data)
        print(f"Starting Multi-Agent Analysis for User {user_id}...")
        final_analysis, structured_markers = analyze_patient_report(base64_image, rag_context)

        # 5. Save to Supabase DB
        report_data = {
            "user_id": user_id, "filename": file.filename,
            "ai_analysis": final_analysis, "file_url": storage_path
        }
        db_res = supabase.table("reports").insert(report_data).execute()
        report_id = db_res.data[0]['id']

        # 6. --- NEW: GRANULAR QDRANT PERSISTENCE ---
        qdrant_points = []

        # A. Store the Final Summary text
        qdrant_points.append(PointStruct(
            id=str(uuid.uuid4()), 
            vector=embed_model.encode(final_analysis).tolist(),
            payload={
                "user_id": user_id, 
                "report_id": report_id, 
                "text": f"AI Summary: {final_analysis}",
                "type": "summary"
            }
        ))

        # B. Store every individual marker as a separate searchable point
        for marker in structured_markers:
            # Create a clean string for the vector engine to index
            marker_string = (
                f"Marker: {marker.get('name')}, Result: {marker.get('value')} {marker.get('unit')}, "
                f"Range: {marker.get('min_range')}-{marker.get('max_range')}, Status: {marker.get('status')}"
            )
            
            qdrant_points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_model.encode(marker_string).tolist(),
                payload={
                    "user_id": user_id,
                    "report_id": report_id,
                    "text": marker_string,
                    "type": "marker_data",
                    "marker_name": marker.get('name')
                }
            ))

        # Upload all individual points at once
        qdrant.upsert(collection_name=PATIENT_COLLECTION, points=qdrant_points)

        return db_res.data[0]

    except Exception as e:
        print("--- DATABASE/BACKEND CRASH ---")
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
        if img_path and os.path.exists(img_path) and img_path != temp_path:
            os.remove(img_path)

@app.post("/chat")
async def chat_with_report(request: ChatRequest, user_id: str = Depends(get_current_user)):
    question_vector = embed_model.encode(request.question).tolist()
    
    # Search finds the most relevant individual markers now!
    patient_res = qdrant.query_points(
        collection_name=PATIENT_COLLECTION,
        query=question_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="report_id", match=models.MatchValue(value=request.report_id))
            ]
        ),
        limit=5 # Higher limit to catch multiple related markers
    ).points
    
    medical_res = qdrant.query_points(KNOWLEDGE_COLLECTION, query=question_vector, limit=5).points
    
    context_str = "--- RELEVANT DATA ---\n" + "\n".join([h.payload['text'] for h in patient_res])
    context_str += "\n\n--- CLINICAL GUIDELINES ---\n" + "\n".join([h.payload['text'] for h in medical_res])
    
    try:
        answer = answer_medical_question(request.question, context_str)
        chat_logs = [
            {"report_id": request.report_id, "user_id": user_id, "role": "user", "content": request.question},
            {"report_id": request.report_id, "user_id": user_id, "role": "bot", "content": answer}
        ]
        supabase.table("chat_messages").insert(chat_logs).execute()
        return {"answer": answer}
    except Exception as e:
        return {"answer": "I'm sorry, I'm having trouble connecting."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)