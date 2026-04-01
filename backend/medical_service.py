import os
import json
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from schemas import LabMarker
# Load environment variables
load_dotenv()

# --- INITIALIZE CLIENTS ---
# 1. Hugging Face (Vision + Chat)
TOKEN = os.getenv("HF_TOKEN")
SUMMARY_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
hf_client = InferenceClient(api_key=TOKEN)

def clean_to_float(value):
    if value is None: return None
    try:
        # Extract only numbers and decimals (removes units like 'mg/dL')
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(value))
        return float(match.group()) if match else None
    except: return None


def parse_range_bounds(min_range, max_range):
    """Parse low/high bounds even when model returns a combined range in one field."""
    low = clean_to_float(min_range)
    high = clean_to_float(max_range)

    # Common OCR/model format: min_range = "1.4 - 4.3", max_range = null
    for raw in (min_range, max_range):
        if raw is None:
            continue
        text = str(raw)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if len(nums) >= 2 and ("-" in text or " to " in text.lower()):
            low = float(nums[0])
            high = float(nums[1])
            break

    if low is not None and high is not None and low > high:
        low, high = high, low

    return low, high

def add_marker_status(markers: list[LabMarker]) -> list[LabMarker]:
    for m in markers:
        val = clean_to_float(m.get("value"))
        low, high = parse_range_bounds(m.get("min_range"), m.get("max_range"))

        if val is None:
            m["status"] = "❓ N/A"
        elif low is None and high is None:
            m["status"] = "❓ N/A"
        elif low is not None and val < low:
            m["status"] = "🚩 Low"
        elif high is not None and val > high:
            m["status"] = "🚩 High"
        else:
            m["status"] = "✅ Normal"
    return markers

def extract_json_payload(text: str):
    cleaned = re.sub(r"```json\s*|```", "", text or "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("Model response did not contain a JSON object.")
        return json.loads(match.group(0))


def analyze_patient_report(image_base64, medical_context):
    print(f"--- Single Model Analysis ({SUMMARY_MODEL}) ---")

    extraction_prompt = (
        "You are an expert medical report extraction assistant. "
        "Read the uploaded lab report image and output STRICT JSON only with no markdown.\n\n"
        "Required JSON schema:\n"
        "{\n"
        "  \"metadata\": {\n"
        "    \"name\": string|null,\n"
        "    \"uhid\": string|null,\n"
        "    \"doctor\": string|null,\n"
        "    \"specimen\": string|null,\n"
        "    \"collected_date\": string|null\n"
        "  },\n"
        "  \"markers\": [\n"
        "    {\n"
        "      \"name\": string,\n"
        "      \"value\": string|null,\n"
        "      \"min_range\": string|null,\n"
        "      \"max_range\": string|null,\n"
        "      \"unit\": string|null\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Include every detectable test marker from the report.\n"
        "- Do not invent values. Use null when unknown.\n"
        "- Return JSON only."
    )

    try:
        extraction_response = hf_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": extraction_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            max_tokens=2500,
            temperature=0.1
        )

        extraction_text = extraction_response.choices[0].message.content
        extracted = extract_json_payload(extraction_text)
        report_metadata = extracted.get("metadata", {}) or {}
        structured_markers = extracted.get("markers", []) or []
        structured_markers = add_marker_status(structured_markers)

        markers_json = json.dumps(structured_markers, indent=2, ensure_ascii=False)
        summary_prompt = (
            "You are a SKEPTICAL Clinical Auditor. Report the data with absolute accuracy.\n\n"
            "LOGIC RULES:\n"
            "1. LOOK at the 'status' provided in the JSON data.\n"
            "2. If status is '✅ Normal', do not explain it in Observations.\n"
            "3. If RBC is High while HGB/HCT are Normal, call it a 'Biological Inconsistency'.\n"
            "4. Output format: table first, then '### 🔎 Observations' for 🚩 items, then '### 🩺 Final Summary'.\n\n"
            f"VERIFIED DATA:\n{markers_json}\n\n"
            f"CLINICAL GUIDELINES:\n{medical_context}"
        )

        summary_response = hf_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=2500,
            temperature=0.1
        )

        final_report = summary_response.choices[0].message.content
        return final_report, structured_markers, report_metadata
    except Exception as e:
        print(f"Qwen summary pipeline error: {e}")
        return "Error generating clinical insights.", [], {}

def answer_medical_question(question, context):
    """Triggered by the FastAPI /chat route"""
    try:
        completion = hf_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Use the context provided."},
                {"role": "user", "content": f"CONTEXT: {context}\n\nQUESTION: {question}"}
            ],
            max_tokens=800
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"I'm sorry, I'm having trouble responding. Error: {str(e)}"