import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

TOKEN = os.getenv("HF_TOKEN")

if not TOKEN:
    print("❌ ERROR: HF_TOKEN not found in .env file!")
    exit()

# 2. Init client
client = InferenceClient(api_key=TOKEN)
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

def get_safe_output(msg):
    """
    Safely extract output from reasoning models
    """
    if msg.content:
        return msg.content.strip()
    
    # fallback to reasoning_content if no final answer
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return f"(Fallback from reasoning)\n{msg.reasoning_content.strip()}"
    
    return "⚠️ No response generated"

try:
    print(f"Testing connection with {MODEL_ID}...")

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant.\n"
                    "IMPORTANT:\n"
                    "- Provide ONLY the final answer\n"
                    "- Do NOT include reasoning steps\n"
                    "- Be concise"
                )
            },
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        max_tokens=200,
        temperature=0.1  # important for reasoning models
    )

    msg = completion.choices[0].message

    print("✅ Success! Clean Response:")
    print(get_safe_output(msg))

except Exception as e:
    print(f"❌ Still failing: {e}")