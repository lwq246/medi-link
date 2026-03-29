import os
import json
import re
import base64
from typing import TypedDict, List, Optional, Union
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langgraph.graph import StateGraph, END
import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- INITIALIZE CLIENTS ---
# 1. Hugging Face (For Formatter and Chat)
TOKEN = os.getenv("HF_TOKEN")
FORMATTER_MODEL = "Qwen/Qwen2.5-7B-Instruct" # Upgraded to 72B for better summary logic
hf_client = InferenceClient(api_key=TOKEN)

# 2. Google Gemini (For Vision Extraction)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# --- STATE DEFINITION ---
class LabMarker(TypedDict):
    name: str
    value: Optional[float]
    min_range: Optional[float]
    max_range: Optional[float]
    unit: str
    status: Optional[str]

class AgentState(TypedDict):
    image_base64: str       
    medical_context: str    
    structured_data: List[LabMarker]
    report_metadata: dict      # <--- ADD THIS
    final_report: str

# --- UTILS ---
def clean_to_float(value):
    if value is None: return None
    try:
        # Extract only numbers and decimals (removes units like 'mg/dL')
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(value))
        return float(match.group()) if match else None
    except: return None

# --- NODE 1: THE VISION EXTRACTOR (GEMINI) ---
def vision_extractor_node(state: AgentState):
    print("--- Agent 1: Gemini Vision Extraction (Markers + Metadata) ---")
    
    prompt = (
        "Analyze this medical report image. Extract TWO JSON objects:\n"
        "1. 'metadata': Extract 'name', 'uhid', 'doctor', 'specimen', and 'collected_date'. Use null if missing.\n"
        "2. 'markers': Extract every lab test into a list with 'name', 'value', 'min_range', 'max_range', and 'unit'.\n"
        "Format: {\"metadata\": {...}, \"markers\": [...]}"
    )

    try:
        response = gemini_model.generate_content([
            prompt,
            {'mime_type': 'image/jpeg', 'data': state['image_base64']}
        ])
        
        data = json.loads(re.sub(r"```json\n|\n```", "", response.text).strip())
        
        # We return the new metadata along with the markers
        return {
            "structured_data": data.get("markers", []),
            "report_metadata": data.get("metadata", {}) # <--- CAPTURE METADATA
        }
    except Exception as e:
        print(f"❌ Vision Error: {e}")
        return {"structured_data": [], "report_metadata": {}}
# --- NODE 2: THE AUDITOR (PYTHON CODE) ---
def auditor_node(state: AgentState):
    print("--- Agent 2: Deterministic Math Audit ---")
    markers = state["structured_data"]
    
    for m in markers:
        val = clean_to_float(m.get("value"))
        low = clean_to_float(m.get("min_range"))
        high = clean_to_float(m.get("max_range"))
        
        if val is None:
            m["status"] = "❓ N/A"
        elif low is not None and val < low:
            m["status"] = "🚩 Low"
        elif high is not None and val > high:
            m["status"] = "🚩 High"
        else:
            m["status"] = "✅ Normal"
            
    return {"structured_data": markers}

# --- NODE 3: THE FORMATTER (HUGGING FACE) ---
def formatter_node(state: AgentState):
    print(f"--- Agent 3: Clinical Analysis ({FORMATTER_MODEL}) ---")
    
    json_summary = json.dumps(state["structured_data"], indent=2, ensure_ascii=False)
    
    prompt = (
        "You are a SKEPTICAL Clinical Auditor. Report the data with absolute accuracy.\n\n"
        "LOGIC RULES:\n"
        "1. LOOK at the 'status' provided in the JSON data.\n"
        "2. If status is '✅ Normal', you are FORBIDDEN from explaining it in the Observations.\n"
        "3. Check for Contradictions: If RBC is High but HGB/HCT are Normal, call it a 'Biological Inconsistency'.\n"
        "4. Output format: Table first, then '### 🔎 Observations' for 🚩 items, then '### 🩺 Final Summary'.\n\n"
        f"VERIFIED DATA:\n{json_summary}\n\n"
        f"CLINICAL GUIDELINES:\n{state['medical_context']}"
    )

    try:
        response = hf_client.chat.completions.create(
            model=FORMATTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
            temperature=0.1
        ).choices[0].message.content

        return {"final_report": response}
    except Exception as e:
        print(f"Formatter Error: {e}")
        return {"final_report": "Error generating clinical insights."}

# --- BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

workflow.add_node("vision_extractor", vision_extractor_node)
workflow.add_node("auditor", auditor_node)
workflow.add_node("formatter", formatter_node)

workflow.set_entry_point("vision_extractor")
workflow.add_edge("vision_extractor", "auditor")
workflow.add_edge("auditor", "formatter")
workflow.add_edge("formatter", END)

medical_app = workflow.compile()

# --- PUBLIC FUNCTIONS ---

# In medical_service.py
def analyze_patient_report(image_base64, medical_context):
    result = medical_app.invoke({
        "image_base64": image_base64,
        "medical_context": medical_context,
        "structured_data": [],
        "report_metadata": {}, # <--- Initialize empty
        "final_report": ""
    })
    
    # CRUCIAL: Must return exactly 3 items to match main.py
    return result["final_report"], result["structured_data"], result["report_metadata"]

def answer_medical_question(question, context):
    """Triggered by the FastAPI /chat route"""
    try:
        completion = hf_client.chat.completions.create(
            model=FORMATTER_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Use the context provided."},
                {"role": "user", "content": f"CONTEXT: {context}\n\nQUESTION: {question}"}
            ],
            max_tokens=800
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"I'm sorry, I'm having trouble responding. Error: {str(e)}"