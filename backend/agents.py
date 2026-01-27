import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()  # loads .env

TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

client = InferenceClient(token=TOKEN)

def query_huggingface(messages):
    print(f"Connecting to {MODEL_ID}...")
    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

# === AGENT 1: REPORT ANALYZER ===
def run_multi_agent_system(patient_text, medical_context):
    print("Running Analysis Agent...")
    
    messages = [
        {"role": "system", "content": "You are a medical AI assistant. Summarize the text using the provided guidelines."},
        {"role": "user", "content": f"Patient Text: {patient_text}\n\nGuidelines: {medical_context}\n\nSummary:"}
    ]

    result = query_huggingface(messages)
    
    if result:
        return result
            
    # Reliable Fallback
    return (
        "⚠️ [OFFLINE MODE]\n"
        "AI unavailable. Please compare the patient values manually against the reference ranges."
    )

# === AGENT 2: CHATBOT ===
def run_chat_agent(question, context):
    print("Running Chat Agent...")
    
    safe_context = context if context else "No context provided."
    
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": f"Context: {safe_context}\n\nQuestion: {question}"}
    ]

    result = query_huggingface(messages)
    
    if result:
        return result
            
    return "I am currently offline. Please check your internet connection."

# === TEST IT IMMEDIATELY ===
if __name__ == "__main__":
    print("Testing Non-Gated Model...")
    test_response = run_chat_agent("Hello", "")
    print(f"\nFinal Result: {test_response}")