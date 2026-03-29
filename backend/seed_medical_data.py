import uuid
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# 1. Connect to Qdrant (Using your specified port 6335)
client = QdrantClient(host="localhost", port=6335)
COLLECTION_NAME = "medical_knowledge"

# 2. Load AI Model
print("Loading AI Model (NeuML/pubmedbert-base-embeddings)...")
model = SentenceTransformer('NeuML/pubmedbert-base-embeddings') 

# 3. Clinical Interpretations and Expert Logic (Numbers removed)
# This data provides the "Why" and "What it means" for the AI.
medical_facts = [
    # --- BLOOD COUNT INTERPRETATIONS ---
    "CLINICAL NOTE (WBC): High White Blood Cell counts commonly indicate infection or inflammation; they may also be elevated due to corticosteroid use.",
    "CLINICAL NOTE (RBC/HCT): Elevated Red Blood Cell count or Hematocrit is frequently a sign of mild dehydration. Increasing fluid intake is the primary lifestyle recommendation.",
    "CLINICAL NOTE (RBC/HCT/HGB): Low levels of RBC, Hematocrit, or Hemoglobin are the primary clinical indicators of Anemia.",
    "CLINICAL NOTE (MCV): High MCV suggests Macrocytic anemia (often Vit B12/Folate related); Low MCV suggests Microcytic anemia (often Iron related).",
    "CLINICAL NOTE (Platelets): Elevated platelet counts may indicate iron deficiency or a reactive response to recent bleeding, injury, or infection.",
    "CLINICAL NOTE (Absolute Lymphocytes): Low absolute lymphocyte counts can indicate significant illness, or be a side effect of specific medications and immunosuppressants.",
    "CLINICAL NOTE (Absolute Monocytes): High levels may indicate chronic infection, autoimmune disorders, or specific hematological conditions.",

    # --- ELECTROLYTES & RENAL INTERPRETATIONS ---
    "CLINICAL NOTE (Albumin): High albumin is a standard marker of dehydration. Low albumin usually occurs with edema (swelling) or poor nutritional intake.",
    "CLINICAL NOTE (Anion Gap): A high anion gap is a clinical indicator of metabolic acidosis.",
    "CLINICAL NOTE (Calcium): Severely high calcium is a critical heart risk finding as it can cause arrhythmias due to calcium movement from bones to the bloodstream.",
    "CLINICAL NOTE (Potassium): High potassium (Hyperkalemia) is dangerous and requires urgent monitoring for cardiac safety.",
    "CLINICAL NOTE (Creatinine): Creatinine reflects muscle mass. In patients with muscle-wasting conditions, low creatinine is expected and normal for their condition.",

    # --- DUCHENNE & MUSCULAR DYSTROPHY SPECIAL LOGIC ---
    "DUCHENNE SPECIAL NOTE (ALT/AST): In patients with Duchenne/MD, elevated ALT and AST are NORMAL and expected due to constant skeletal muscle breakdown. They should NOT be interpreted as liver disease.",
    "DUCHENNE SPECIAL NOTE (CK): Creatine Kinase (CK) is expected to be chronically and massively elevated in Duchenne patients, reflecting ongoing skeletal muscle injury.",
    "DUCHENNE SPECIAL NOTE (CO2): High CO2 in Duchenne can indicate respiratory acidosis (breathing issues); Low CO2 can indicate respiratory alkalosis or diarrhea.",

    # --- LIVER & ENZYME INTERPRETATIONS ---
    "CLINICAL NOTE (ALP): High Alkaline Phosphatase (ALP) is normal and expected in children and adolescents due to rapid bone growth during puberty.",
    "CLINICAL NOTE (GGT/ALP): If GGT is normal but ALP is high, the elevation is likely originating from bone growth or skeletal issues, not the liver.",
    "CLINICAL NOTE (Bilirubin): High bilirubin in adults may suggest liver obstruction, hepatitis, or the rapid breakdown of red blood cells (hemolysis).",

    # --- LIPID & METABOLIC INTERPRETATIONS ---
    "CLINICAL NOTE (LDL/HDL): High-Density Lipoprotein (HDL) is 'good' cholesterol; higher levels decrease heart disease risk. High LDL increases cardiovascular risk.",
    "CLINICAL NOTE (Triglycerides): High triglycerides, especially with high cholesterol, serve as a marker for increased cardiovascular and metabolic risk.",
    "CLINICAL NOTE (Glucose): A random (non-fasting) glucose result higher than 200 mg/dL is a primary indicator of potential diabetes.",

    # --- IRON & ENDOCRINE INTERPRETATIONS ---
    "CLINICAL NOTE (Transferrin/Iron): Low transferrin levels typically indicate a lack of stored iron in the body (Iron deficiency).",
    "CLINICAL NOTE (Vitamin D): Low Vitamin D (25 OH Vit D) indicates insufficient stores, common in patients with limited sunlight or poor absorption.",
    "CLINICAL NOTE (Testosterone): Low testosterone in young males can be a secondary side effect of long-term corticosteroid use (common in MD treatment).",
    "CLINICAL NOTE (IGF-1): Low IGF-1 levels relative to age and Tanner Stage may suggest a growth hormone deficiency."
]

def seed_db():
    print(f"Clearing and re-initializing collection '{COLLECTION_NAME}'...")
    
    # Re-create collection to ensure clean state
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    points = []
    print(f"Vectorizing {len(medical_facts)} clinical interpretations...")
    
    for fact in medical_facts:
        vector = model.encode(fact).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": fact, 
                "category": "clinical_interpretation",
                "source": "Mayo Clinic / PPMD Logic"
            }
        ))

    # Upload in one batch
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Success! {len(points)} clinical facts inserted into Qdrant.")

if __name__ == "__main__":
    seed_db()