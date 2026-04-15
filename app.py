import pdfplumber
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Get current folder path automatically (IMPORTANT FIX)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Extract text from PDF
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text.lower()

# Load job description (FIXED PATH)
job_file_path = os.path.join(BASE_DIR, "job_description.txt")

with open(job_file_path, "r") as f:
    job_desc = f.read().lower()

# Skills list
skills = [
    "python", "sql", "pandas", "numpy",
    "power bi", "excel", "ms office", "vs code",
    "routing", "subnetting", "ip addressing", "osi model", "tcp/ip",
    "firewall", "security policies", "access control", "nat", "vpn",
    "cisco asa", "palo alto",
    "network security", "risk assessment", "cyber threats", "compliance",
    "analytical thinking", "troubleshooting", "communication", "teamwork"
]

# Extract skills
def extract_skills(text):
    found = []
    for skill in skills:
        if skill in text:
            found.append(skill)
    return found

# Resume folder path (FIXED PATH)
resume_folder = os.path.join(BASE_DIR, "Resumes")

print("\n--- Resume Analysis Result ---\n")

# Loop through resumes
results = []

for file in os.listdir(resume_folder):
    if file.endswith(".pdf"):
        path = os.path.join(resume_folder, file)

        resume_text = extract_text(path)
        resume_skills = extract_skills(resume_text)

        # Similarity score
        cv = CountVectorizer()
        vectors = cv.fit_transform([job_desc, resume_text]).toarray()
        score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

        results.append({
            "Resume Name": file,
            "Skills Found": ", ".join(resume_skills),
            "Match Score (%)": round(score * 100, 2)
        })

# SAVE TO CSV
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("output.csv", index=False)

print("\n✅ Data saved to output.csv")