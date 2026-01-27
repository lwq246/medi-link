from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import uuid

# 1. Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "medical_knowledge"

# 2. Load a lightweight AI Model (Hugging Face)
print("Loading AI Model...")
model = SentenceTransformer('all-mpnet-base-v2') # Standard, powerful model

# 3. Define some "Medical Textbook" data
medical_facts = [
    # === COMPLETE BLOOD COUNT (CBC) ===
    "Hemoglobin (Hb): Normal range for men is 13.5-17.5 g/dL, women 12.0-15.5 g/dL. Low levels indicate anemia (iron deficiency, blood loss). High levels may indicate polycythemia or dehydration.",
    "White Blood Cell Count (WBC): Normal range is 4.5 to 11.0 x 10^9/L. High levels (Leukocytosis) suggest infection, inflammation, or leukemia. Low levels (Leukopenia) suggest viral infection, bone marrow issues, or chemotherapy side effects.",
    "Platelets (Plt): Normal range is 150,000 to 450,000 per microliter. Low levels (Thrombocytopenia) increase bleeding risk. High levels (Thrombocytosis) increase clotting risk.",
    "Hematocrit (Hct): Normal range men 41-50%, women 36-44%. Measures the percentage of red blood cells. Low indicates anemia; high indicates dehydration or blood disorders.",
    "Red Blood Cell Count (RBC): Normal range men 4.7-6.1 million/mcL, women 4.2-5.4 million/mcL. Carries oxygen. Low count suggests anemia; high count suggests lung disease or dehydration.",

    # === LIPID PANEL (CHOLESTEROL) ===
    "Total Cholesterol: Normal is less than 200 mg/dL. 200-239 is borderline high. 240+ is high. High levels increase risk of heart disease and stroke.",
    "LDL Cholesterol (Bad): Optimal is less than 100 mg/dL. 100-129 is near optimal. 160+ is high. Causes plaque buildup in arteries.",
    "HDL Cholesterol (Good): Low is less than 40 mg/dL (men) or 50 mg/dL (women). High is 60+ mg/dL. Protects against heart disease.",
    "Triglycerides: Normal is less than 150 mg/dL. High levels are linked to heart disease, obesity, and type 2 diabetes.",

    # === KIDNEY FUNCTION ===
    "Creatinine: Normal range 0.7-1.3 mg/dL (men), 0.6-1.1 mg/dL (women). A waste product filtered by kidneys. High levels indicate kidney dysfunction or blockage.",
    "Blood Urea Nitrogen (BUN): Normal range 7-20 mg/dL. High levels suggest kidney injury, dehydration, or high protein diet. Low levels may indicate liver damage.",
    "eGFR (Estimated Glomerular Filtration Rate): Normal is > 90. 60-89 is mild kidney loss. < 60 indicates kidney disease. Measures how well kidneys filter blood.",

    # === LIVER FUNCTION ===
    "Alanine Aminotransferase (ALT): Normal range 7-56 U/L. An enzyme found in the liver. High levels indicate liver damage (hepatitis, alcohol abuse).",
    "Aspartate Aminotransferase (AST): Normal range 10-40 U/L. High levels indicate damage to liver, heart, or muscle tissues.",
    "Bilirubin: Normal range 0.1 to 1.2 mg/dL. High levels cause jaundice (yellow skin) and indicate liver disease or bile duct blockage.",
    "Albumin: Normal range 3.4 to 5.4 g/dL. A protein made by the liver. Low levels indicate liver disease, kidney disease, or malnutrition.",

    # === DIABETES & METABOLISM ===
    "Fasting Blood Glucose: Normal is 70-99 mg/dL. 100-125 is Prediabetes. 126+ is Diabetes. Measures blood sugar levels.",
    "Hemoglobin A1c (HbA1c): Normal is below 5.7%. 5.7-6.4% is Prediabetes. 6.5% or higher is Diabetes. Measures average blood sugar over past 3 months.",
    "Potassium: Normal range 3.6-5.2 mmol/L. Critical for heart function. Low (Hypokalemia) causes muscle weakness/cramps. High (Hyperkalemia) causes dangerous heart rhythms.",
    "Sodium: Normal range 135-145 mEq/L. High levels indicate dehydration or kidney issues. Low levels indicate water intoxication or heart failure.",

    # === THYROID ===
    "Thyroid Stimulating Hormone (TSH): Normal range 0.4 to 4.0 mIU/L. High TSH often means Hypothyroidism (slow thyroid). Low TSH often means Hyperthyroidism (fast thyroid).",
    "Free T4 (Thyroxine): Normal range 0.8 to 1.8 ng/dL. Works with TSH to diagnose thyroid issues."
]


def seed_db():
    # Re-create collection to start fresh
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    points = []
    print("Vectorizing data...")
    
    # Loop through facts, turn them into numbers, and prepare for DB
    for fact in medical_facts:
        vector = model.encode(fact).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": fact, "type": "medical_guideline"}
        ))

    # Upload to Qdrant
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Success! {len(points)} medical facts inserted into Qdrant.")

if __name__ == "__main__":
    seed_db()