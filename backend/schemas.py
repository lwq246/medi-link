# backend/schemas.py
from typing import TypedDict, List, Optional, Union
from pydantic import BaseModel

# Structure for a single lab result
class LabMarker(TypedDict):
    name: str
    value: Optional[float]
    min_range: Optional[float]
    max_range: Optional[float]
    unit: str
    status: Optional[str]

# The shared state for your LangGraph
class AgentState(TypedDict):
    image_base64: str       
    medical_context: str    
    structured_data: List[LabMarker]
    report_metadata: dict
    final_report: str

# FastAPI Request models (can move here too)
class ChatRequest(BaseModel):
    question: str
    report_id: str