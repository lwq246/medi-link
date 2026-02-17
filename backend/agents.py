import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# 1. Configuration
TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# 2. Initialize the LLM & Chat Wrapper
# We still use the Endpoint, but we wrap it in ChatHuggingFace to fix the "Task" error
raw_llm = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    # We remove task="text-generation" to let it auto-detect or use the Chat wrapper
    huggingfacehub_api_token=TOKEN,
    max_new_tokens=512,
    temperature=0.2,
)

# This wrapper forces LangChain to use the "conversational" (Chat) API
llm = ChatHuggingFace(llm=raw_llm)

# 3. Updated Prompt Templates (Using System/Human message structure for Chat Models)
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional medical AI assistant. Analyze the results against guidelines."),
    ("human", "PATIENT REPORT: {patient_text}\n\nMEDICAL GUIDELINES: {medical_context}\n\nSummarize the findings and highlight abnormal values.")
])

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and knowledgeable medical assistant."),
    ("human", "CONTEXT: {context}\n\nQUESTION: {question}")
])

# 4. Create the Chains
analysis_chain = analysis_prompt | llm | StrOutputParser()
chat_chain = chat_prompt | llm | StrOutputParser()

# === AGENT 1: REPORT ANALYZER ===
def run_multi_agent_system(patient_text, medical_context):
    print(f"Connecting to {MODEL_ID} via Conversational API...")
    
    # Context-level Truncation for Safety
    safe_patient_text = patient_text[:3000] if patient_text else "No data."
    safe_medical_context = medical_context[:1500] if medical_context else "No context."

    try:
        result = analysis_chain.invoke({
            "patient_text": safe_patient_text,
            "medical_context": safe_medical_context
        })
        return result
    except Exception as e:
        print(f"❌ API Error in Analysis: {e}")
        return (
            "⚠️ [OFFLINE MODE]\n"
            "AI unavailable. Please compare the patient values manually against the reference ranges."
        )

# === AGENT 2: CHATBOT ===
def run_chat_agent(question, context):
    print(f"Connecting to {MODEL_ID} for Chat Q&A...")
    
    safe_context = context[:3000] if context else "No context."

    try:
        result = chat_chain.invoke({
            "context": safe_context,
            "question": question
        })
        return result
    except Exception as e:
        print(f"❌ API Error in Chat: {e}")
        return "I am currently offline. Please check your internet connection."

if __name__ == "__main__":
    # Quick Test
    print(run_chat_agent("Hello", "No context"))