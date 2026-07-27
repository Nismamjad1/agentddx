"""Configuration for AgentDDx.

Runtime settings are read from the environment (see .env.example). The defaults
below are the values used for every experiment reported in the paper.
"""
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

# LLM backbone. One model serves every pipeline stage — entity extraction,
# query construction, relevance scoring and answering — so that comparisons are
# not confounded by auxiliary model choices.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")

# NCBI Entrez requires a contact address with every request.
PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "agentddx@example.org")

# Retrieval and filtering parameters.
MAX_PAPERS_PER_DIAGNOSIS = int(os.getenv("MAX_PAPERS", "3"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.4"))
SELF_CONSISTENCY_PASSES = int(os.getenv("SELF_CONSISTENCY", "1"))  # 1 = single pass

SAMPLE_CASES = [
    {
        "id": "sample_001",
        "title": "Lung mass with smoking history",
        "category": "Oncology",
        "text": (
            "A 67-year-old male with a 40 pack-year smoking history presents with haemoptysis, "
            "progressive dyspnoea, and 10 kg weight loss over 3 months. CT chest reveals a "
            "3.2 cm spiculated mass in the right upper lobe with mediastinal lymphadenopathy. "
            "PFTs show FEV1/FVC 0.61. Serum CEA 18.3 ng/mL."
        ),
        "diagnosis": "Lung adenocarcinoma",
    },
    {
        "id": "sample_002",
        "title": "Acute chest pain — young female",
        "category": "Emergency",
        "text": (
            "A 28-year-old female presents to the ED with sudden onset pleuritic chest pain and "
            "shortness of breath starting 6 hours ago. She is on oral contraceptive pills and "
            "returned from a 12-hour flight 3 days ago. Vitals: HR 112, BP 110/70, SpO2 93%. "
            "D-dimer 2.4 mg/L. CXR shows a small right-sided pleural effusion."
        ),
        "diagnosis": "Pulmonary embolism",
    },
    {
        "id": "sample_003",
        "title": "Progressive weakness — elderly",
        "category": "Neurology",
        "text": (
            "A 74-year-old female presents with progressive bilateral lower limb weakness over "
            "6 weeks, difficulty swallowing, and 10 kg weight loss. Examination reveals proximal "
            "muscle weakness (MRC 3/5), absent deep tendon reflexes, and tongue fasciculations. "
            "EMG shows fibrillation potentials. CK 380 U/L (mildly elevated)."
        ),
        "diagnosis": "Amyotrophic lateral sclerosis",
    },
    {
        "id": "sample_004",
        "title": "Fever and joint pain — young male",
        "category": "Rheumatology",
        "text": (
            "A 19-year-old male presents with a 2-week history of migratory polyarthralgia "
            "affecting the knees, ankles, and wrists, preceded by a sore throat 3 weeks ago. "
            "Examination reveals a pansystolic murmur at the apex, erythema marginatum on the "
            "trunk, and subcutaneous nodules over the elbows. ESR 68 mm/hr, ASO titre 480 IU/mL. "
            "ECG shows prolonged PR interval (240 ms)."
        ),
        "diagnosis": "Acute rheumatic fever",
    },
    {
        "id": "sample_005",
        "title": "Confusion and polyuria — middle-aged",
        "category": "Endocrinology",
        "text": (
            "A 55-year-old male brought to the ED with confusion and lethargy for 2 days. "
            "History of recently diagnosed squamous cell carcinoma of the lung. He reports "
            "polyuria, constipation, and nausea for the past week. Vitals: HR 98, BP 100/60. "
            "Labs show serum calcium 14.2 mg/dL, phosphate 2.1 mg/dL, PTHrP elevated, "
            "creatinine 1.8 mg/dL. ECG shows shortened QT interval."
        ),
        "diagnosis": "Humoral hypercalcaemia of malignancy",
    },
]
