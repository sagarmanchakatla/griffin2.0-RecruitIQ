import pandas as pd
import os
import re
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF for PDF reading
import joblib  # For saving model
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (uncomment if needed)
nltk.download('punkt_tab')
nltk.download('stopwords')

# Paths
JOB_DESC_FILE = "./job_description.csv"
CV_FOLDER = "./CVs1"
MODEL_PATH = "./resume_matcher_model.pkl"

# === Agent 1: Job Description Parser ===
class JobDescriptionParserAgent:
    def parse(self, row):
        job_title = row.get("Job Title", "Unknown Job")
        description = row.get("Job Description", "")
        company_name = row.get("Company Name", "Unknown Company")
        hiring_number = row.get("Hiring Number", 0)
        
        responsibilities = re.findall(r"Responsibilities:\s*(.*?)Qualifications:", description, re.DOTALL)
        qualifications = re.findall(r"Qualifications:\s*(.*)", description, re.DOTALL)
        
        return {
            "job_title": job_title,
            "company_name": company_name,
            "hiring_number": hiring_number,
            "responsibilities": responsibilities[0].strip() if responsibilities else "",
            "qualifications": qualifications[0].strip() if qualifications else "",
            "full_description": description
        }

# === Agent 2: CV Parser (PDF to Text with Enhanced Features) ===
class CVParserAgent:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def parse(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = " ".join(page.get_text() for page in doc)
            
            # Extract candidate name (assuming it's near the top of the CV)
            candidate_name = self._extract_name(text[:500])
            
            # Extract skills
            skills = self._extract_skills(text)
            
            # Extract education
            education = self._extract_education(text)
            
            # Extract experience
            experience = self._extract_experience(text)
            
            return {
                "text": text.strip(),
                "candidate_name": candidate_name,
                "skills": skills,
                "education": education,
                "experience": experience
            }
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return {
                "text": "",
                "candidate_name": os.path.basename(pdf_path).replace(".pdf", ""),
                "skills": [],
                "education": "",
                "experience": ""
            }
    
    def _extract_name(self, text_sample):
        # Simple name extraction - could be improved with NER models
        lines = text_sample.strip().split('\n')
        # Assume name is in first 3 non-empty lines
        for line in lines[:3]:
            if line.strip() and len(line.strip()) < 50:  # Names are usually short
                return line.strip()
        return "Unknown"
    
    def _extract_skills(self, text):
        # Look for skills section and extract keywords
        skills_text = re.findall(r"(?:Skills|SKILLS|Technical Skills|TECHNICAL SKILLS)(?::|.{0,20})\s*(.*?)(?:\n\n|\n[A-Z])", text, re.DOTALL)
        if skills_text:
            # Tokenize and clean
            words = word_tokenize(skills_text[0].lower())
            skills = [word for word in words if word.isalnum() and word not in self.stop_words]
            return list(set(skills))  # Remove duplicates
        return []
    
    def _extract_education(self, text):
        education_text = re.findall(r"(?:Education|EDUCATION)(?::|.{0,20})\s*(.*?)(?:\n\n|\n[A-Z])", text, re.DOTALL)
        if education_text:
            return education_text[0].strip()
        return ""
    
    def _extract_experience(self, text):
        experience_text = re.findall(r"(?:Experience|EXPERIENCE|Work Experience|WORK EXPERIENCE)(?::|.{0,20})\s*(.*?)(?:\n\n|\n[A-Z])", text, re.DOTALL)
        if experience_text:
            return experience_text[0].strip()
        return ""

# === Agent 3: Feature Engineering ===
class FeatureEngineeringAgent:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.job_title_vectorizer = TfidfVectorizer(max_features=200)
        self.company_vectorizer = TfidfVectorizer(max_features=100)
    
    def fit_vectorizers(self, job_descriptions, cv_texts):
        all_texts = [jd["full_description"] for jd in job_descriptions] + [cv["text"] for cv in cv_texts]
        self.tfidf_vectorizer.fit(all_texts)
        
        job_titles = [jd["job_title"] for jd in job_descriptions]
        self.job_title_vectorizer.fit(job_titles)
        
        company_names = [jd["company_name"] for jd in job_descriptions]
        self.company_vectorizer.fit(company_names)
    
    def extract_features(self, job_description, cv_data):
        # TF-IDF similarity between job description and CV
        jd_text = job_description["full_description"]
        cv_text = cv_data["text"]
        
        jd_vector = self.tfidf_vectorizer.transform([jd_text])
        cv_vector = self.tfidf_vectorizer.transform([cv_text])
        content_similarity = cosine_similarity(jd_vector, cv_vector)[0][0]
        
        # Job title relevance
        job_title_vector = self.job_title_vectorizer.transform([job_description["job_title"]])
        cv_title_vector = self.job_title_vectorizer.transform([cv_text[:1000]])  # Use beginning of CV for job title match
        title_similarity = cosine_similarity(job_title_vector, cv_title_vector)[0][0]
        
        # Company name relevance
        company_vector = self.company_vectorizer.transform([job_description["company_name"]])
        cv_company_vector = self.company_vectorizer.transform([cv_text])
        company_similarity = cosine_similarity(company_vector, cv_company_vector)[0][0]
        
        # Skills match (count of job skills found in CV skills)
        jd_skills = set(word_tokenize(job_description["qualifications"].lower()))
        cv_skills = set(cv_data["skills"])
        skills_overlap = len(jd_skills.intersection(cv_skills)) / max(len(jd_skills), 1)
        
        # Education relevance
        education_match = 0.0
        if job_description["qualifications"] and cv_data["education"]:
            edu_jd_vector = self.tfidf_vectorizer.transform([job_description["qualifications"]])
            edu_cv_vector = self.tfidf_vectorizer.transform([cv_data["education"]])
            education_match = cosine_similarity(edu_jd_vector, edu_cv_vector)[0][0]
        
        # Experience relevance
        experience_match = 0.0
        if job_description["responsibilities"] and cv_data["experience"]:
            exp_jd_vector = self.tfidf_vectorizer.transform([job_description["responsibilities"]])
            exp_cv_vector = self.tfidf_vectorizer.transform([cv_data["experience"]])
            experience_match = cosine_similarity(exp_jd_vector, exp_cv_vector)[0][0]
        
        # Hiring number consideration (normalize to 0-1 range)
        hiring_urgency = min(job_description["hiring_number"] / 10.0, 1.0)  # Normalize, max at 10
        
        # Calculate any additional features
        features = {
            "content_similarity": content_similarity,
            "title_similarity": title_similarity,
            "company_similarity": company_similarity,
            "skills_match": skills_overlap,
            "education_match": education_match,
            "experience_match": experience_match,
            "hiring_urgency": hiring_urgency
        }
        
        # Convert to array for model
        feature_array = np.array([
            content_similarity, 
            title_similarity,
            company_similarity,
            skills_overlap,
            education_match,
            experience_match,
            hiring_urgency
        ])
        
        return feature_array, features

# === Agent 4: Matching Model ===
class MatchingModelAgent:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_engineer = FeatureEngineeringAgent()
        self.is_fitted = False
    
    def train(self, job_descriptions, cv_data_list):
        # Prepare data for training
        X = []
        y = []
        
        # First, fit feature extractors on all data
        self.feature_engineer.fit_vectorizers(job_descriptions, cv_data_list)
        
        # Generate training data
        print("Generating training data...")
        for jd in job_descriptions:
            for cv_data in cv_data_list:
                features, _ = self.feature_engineer.extract_features(jd, cv_data)
                
                # For training data, use simple weighted average as target
                # In real scenario, you would use human-labeled match scores
                score = features[0] * 0.3 + features[1] * 0.2 + features[2] * 0.05 + \
                       features[3] * 0.2 + features[4] * 0.1 + features[5] * 0.1 + features[6] * 0.05
                
                X.append(features)
                y.append(score)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(X), np.array(y), test_size=0.2, random_state=42
        )
        
        # Train model
        print(f"Training model with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Model R² on training: {train_score:.4f}, test: {test_score:.4f}")
        
        # Save model
        self.save_model()
        
        return train_score, test_score
    
    def predict_match(self, job_description, cv_data):
        if not self.is_fitted:
            self.load_model()
            
        features, feature_dict = self.feature_engineer.extract_features(job_description, cv_data)
        
        # Make prediction
        score = self.model.predict([features])[0]
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score * 100))
        
        return score, feature_dict
    
    def save_model(self):
        model_data = {
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "is_fitted": self.is_fitted
        }
        joblib.dump(model_data, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    def load_model(self):
        try:
            model_data = joblib.load(MODEL_PATH)
            self.model = model_data["model"]
            self.feature_engineer = model_data["feature_engineer"]
            self.is_fitted = model_data["is_fitted"]
            print(f"Model loaded from {MODEL_PATH}")
            return True
        except:
            print(f"No model found at {MODEL_PATH}, using untrained model")
            return False

# === Agent 5: Shortlisting ===
class ShortlistingAgent:
    def shortlist(self, matches, threshold=70):
        return [m for m in matches if m["score"] >= threshold]
    
    def rank_candidates(self, matches):
        return sorted(matches, key=lambda x: x["score"], reverse=True)

# === Agent 6: Interview Scheduler ===
class InterviewSchedulerAgent:
    def send_invite(self, candidate_name, score, job_title, company_name):
        return (
            f"Hi {candidate_name}, you've been shortlisted for the {job_title} position at "
            f"{company_name} with a match score of {score:.2f}%. "
            f"Let's schedule your interview!"
        )

# === Agent 7: Result Analyzer ===
class ResultAnalyzerAgent:
    def analyze_match(self, score, feature_dict):
        strengths = []
        areas_to_improve = []
        
        # Analyze each feature contribution
        for feature, value in feature_dict.items():
            normalized_value = value * 100  # Convert to percentage
            if normalized_value >= 70:
                strengths.append(f"{feature.replace('_', ' ').title()} ({normalized_value:.1f}%)")
            elif normalized_value <= 30:
                areas_to_improve.append(f"{feature.replace('_', ' ').title()} ({normalized_value:.1f}%)")
        
        return {
            "overall_score": score,
            "strengths": strengths[:3],  # Top 3 strengths
            "areas_to_improve": areas_to_improve[:3]  # Top 3 areas to improve
        }

# === Main Application Class ===
class ResumeMatcherApp:
    def __init__(self):
        self.jd_parser = JobDescriptionParserAgent()
        self.cv_parser = CVParserAgent()
        self.matching_model = MatchingModelAgent()
        self.shortlister = ShortlistingAgent()
        self.scheduler = InterviewSchedulerAgent()
        self.analyzer = ResultAnalyzerAgent()
    
    def load_job_descriptions(self, file_path):
        try:
            df = pd.read_csv(file_path, encoding="ISO-8859-1")
            return [self.jd_parser.parse(row) for _, row in df.iterrows()]
        except Exception as e:
            print(f"Error loading job descriptions: {e}")
            return []
    
    def load_cvs(self, folder_path):
        cv_data_list = []
        try:
            for file in os.listdir(folder_path):
                if file.endswith(".pdf"):
                    file_path = os.path.join(folder_path, file)
                    cv_data = self.cv_parser.parse(file_path)
                    cv_data["file_name"] = file
                    cv_data_list.append(cv_data)
            return cv_data_list
        except Exception as e:
            print(f"Error loading CVs: {e}")
            return []
    
    def train_model(self):
        # Load data
        job_descriptions = self.load_job_descriptions(JOB_DESC_FILE)
        cv_data_list = self.load_cvs(CV_FOLDER)
        
        if not job_descriptions or not cv_data_list:
            print("Cannot train model: Missing job descriptions or CVs")
            return False
        
        # Train the model
        return self.matching_model.train(job_descriptions, cv_data_list)
    
    def match_single_resume(self, resume_path):
        # Load job descriptions
        job_descriptions = self.load_job_descriptions(JOB_DESC_FILE)
        
        if not job_descriptions:
            print("No job descriptions available")
            return []
        
        # Parse the resume
        cv_data = self.cv_parser.parse(resume_path)
        cv_data["file_name"] = os.path.basename(resume_path)
        
        results = []
        
        for jd in job_descriptions:
            # Get match score
            score, feature_dict = self.matching_model.predict_match(jd, cv_data)
            
            # Analyze match
            analysis = self.analyzer.analyze_match(score, feature_dict)
            
            # Generate invite if score is high enough
            invite = self.scheduler.send_invite(
                cv_data["candidate_name"], 
                score, 
                jd["job_title"], 
                jd["company_name"]
            ) if score >= 70 else "Not shortlisted"
            
            # Store results
            results.append({
                "job_title": jd["job_title"],
                "company_name": jd["company_name"],
                "cv_file": cv_data["file_name"],
                "candidate": cv_data["candidate_name"],
                "score": score,
                "invite": invite,
                "analysis": analysis
            })
        
        # Rank results
        return self.shortlister.rank_candidates(results)
    
    def match_all_resumes(self):
        # Load data
        job_descriptions = self.load_job_descriptions(JOB_DESC_FILE)
        cv_data_list = self.load_cvs(CV_FOLDER)
        
        if not job_descriptions or not cv_data_list:
            print("Missing job descriptions or CVs")
            return []
        
        results = []
        
        for cv_data in cv_data_list:
            for jd in job_descriptions:
                # Get match score
                score, feature_dict = self.matching_model.predict_match(jd, cv_data)
                
                # Analyze match
                analysis = self.analyzer.analyze_match(score, feature_dict)
                
                # Generate invite if score is high enough
                invite = self.scheduler.send_invite(
                    cv_data["candidate_name"], 
                    score, 
                    jd["job_title"], 
                    jd["company_name"]
                ) if score >= 70 else "Not shortlisted"
                
                # Store results
                results.append({
                    "job_title": jd["job_title"],
                    "company_name": jd["company_name"],
                    "cv_file": cv_data["file_name"],
                    "candidate": cv_data["candidate_name"],
                    "score": score,
                    "invite": invite,
                    "analysis": analysis
                })
        
        # Save to SQLite
        try:
            conn = sqlite3.connect("recruitment.db")
            pd.DataFrame(results).to_sql("shortlisting_results", conn, if_exists="replace", index=False)
            conn.close()
            print(f"Results saved to recruitment.db")
        except Exception as e:
            print(f"Error saving to database: {e}")
        
        return results
    
    def run(self):
        # 1. Train the model
        print("Training the model...")
        self.train_model()
        
        # 2. Match all resumes
        print("Matching all resumes...")
        results = self.match_all_resumes()
        
        # 3. Print summary
        print("\n=== MATCHING RESULTS ===")
        for r in results:
            print(f"{r['job_title']} ({r['company_name']}) - {r['candidate']} → {r['score']:.2f}% → {r['invite'][:30]}...")
        
        return results

# === Usage Example ===
if __name__ == "__main__":
    app = ResumeMatcherApp()
    
    # Run full pipeline
    app.run()
    
    # Example: Match a single resume
    # resume_path = "./resume.pdf"
    resume_path = "./C1061.pdf"
    matches = app.match_single_resume(resume_path)
    
    print(f"\n=== TOP MATCHES FOR {os.path.basename(resume_path)} ===")
    for match in matches[:5]:  # Top 5 matches
        print(f"{match['job_title']} at {match['company_name']}: {match['score']:.2f}%")
        print(f"Strengths: {', '.join(match['analysis']['strengths'])}")
        print(f"Areas to improve: {', '.join(match['analysis']['areas_to_improve'])}")
        print("-" * 50)