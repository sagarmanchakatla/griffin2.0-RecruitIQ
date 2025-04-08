import pandas as pd
import os
import re
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF for PDF reading

# Paths
JOB_DESC_FILE = "./job_description.csv"
CV_FOLDER = "./CVs1"

# === Agent 1: Job Description Summarizer ===
class JobDescriptionSummarizerAgent:
    def summarize(self, description):
        responsibilities = re.findall(r"Responsibilities:\s*(.*?)Qualifications:", description, re.DOTALL)
        qualifications = re.findall(r"Qualifications:\s*(.*)", description, re.DOTALL)
        return {
            "responsibilities": responsibilities[0].strip() if responsibilities else "",
            "qualifications": qualifications[0].strip() if qualifications else ""
        }

# === Agent 2: CV Parser (PDF to Text) ===
class CVParserAgent:
    def parse(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = " ".join(page.get_text() for page in doc)
            return text.strip()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""

# === Agent 3: Matching Agent ===
class MatchingAgent:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def compute_score(self, jd_text, cv_text):
        if not jd_text or not cv_text:
            return 0.0
        vectors = self.vectorizer.fit_transform([jd_text, cv_text])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

# === Agent 4: Shortlisting ===
class ShortlistingAgent:
    def shortlist(self, matches, threshold=80):
        return [m for m in matches if m["score"] >= threshold]

# === Agent 5: Interview Scheduler ===
class InterviewSchedulerAgent:
    def send_invite(self, name, score):
        return f"Hi {name}, you’ve been shortlisted with a score of {score:.2f}%. Let’s schedule your interview!"

# === Initialize agents ===
summarizer = JobDescriptionSummarizerAgent()
cv_parser = CVParserAgent()
matcher = MatchingAgent()
shortlister = ShortlistingAgent()
scheduler = InterviewSchedulerAgent()

# === Load Job Descriptions ===
try:
    df = pd.read_csv(JOB_DESC_FILE, encoding="ISO-8859-1")
except Exception as e:
    print(f"Error loading job descriptions: {e}")
    exit()

# === Process ===
results = []

for index, row in df.iterrows():
    job_title = row.get("Job Title", "Unknown Job")
    description = row.get("Job Description", "")
    jd_parts = summarizer.summarize(description)
    jd_combined = jd_parts["responsibilities"] + " " + jd_parts["qualifications"]

    for file in os.listdir(CV_FOLDER):
        if file.endswith(".pdf"):
            file_path = os.path.join(CV_FOLDER, file)
            cv_text = cv_parser.parse(file_path)
            score = matcher.compute_score(jd_combined, cv_text)
            name = file.replace(".pdf", "")
            invite = scheduler.send_invite(name, score) if score >= 80 else "Not shortlisted"

            results.append({
                "job_title": job_title,
                "cv_file": file,
                "candidate": name,
                "score": score,
                "invite": invite
            })

# === Save to SQLite ===
try:
    conn = sqlite3.connect("recruitment.db")
    pd.DataFrame(results).to_sql("shortlisting_results", conn, if_exists="replace", index=False)
    conn.close()
except Exception as e:
    print(f"Error saving to database: {e}")

# === Print Summary ===
for r in results:
    print(f"{r['job_title']} - {r['candidate']} → {r['score']:.2f}% → {r['invite']}")
