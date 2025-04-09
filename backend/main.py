import os
import json
import sqlite3
import smtplib
from email.mime.text import MIMEText
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import PyPDF2  # Added for PDF processing

from langchain_groq import ChatGroq

class JobScreeningSystem:
    def __init__(self, db_name: str = 'recruitment.db'):
        """Initialize the job screening system with database connection"""
        self.db_name = db_name
        self.init_db()
        # self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        # api_key = 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
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", api_key = "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")
        self.keyword_vectorizer = TfidfVectorizer(max_features=100)

    # Database Operations
    def init_db(self) -> None:
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_name TEXT NOT NULL,
            full_description TEXT NOT NULL,
            summary TEXT,
            keywords TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            cv_text TEXT NOT NULL,
            extracted_data TEXT,
            keywords TEXT
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            candidate_id INTEGER NOT NULL,
            match_score REAL,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY(job_id) REFERENCES jobs(job_id),
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
            UNIQUE(job_id, candidate_id)
        )
        """)
        
        # Adding a new table for job recommendations
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            recommendation_score REAL,
            FOREIGN KEY(job_id) REFERENCES jobs(job_id),
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
            UNIQUE(candidate_id, job_id)
        )
        """)
        
        conn.commit()
        conn.close()

    def store_job(self, job_name: str, description: str, summary: str = None, keywords: List[str] = None) -> int:
        """Store a job description in the database"""
        if not summary:
            summary = self.summarize_jd(description)
        
        if not keywords:
            keywords = self.extract_keywords_from_job(description)
            
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO jobs (job_name, full_description, summary, keywords)
        VALUES (?, ?, ?, ?)
        """, (job_name, description, summary, json.dumps(keywords)))
        job_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return job_id

    def store_cv(self, name: str, email: str, cv_text: str, extracted_data: Dict = None, keywords: List[str] = None) -> int:
        """Store a candidate CV in the database"""
        if not extracted_data:
            extracted_data = self.extract_cv_data(cv_text)
            
        if not keywords:
            keywords = self.extract_keywords_from_cv(cv_text)
            
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO candidates (name, email, cv_text, extracted_data, keywords)
        VALUES (?, ?, ?, ?, ?)
        """, (name, email, cv_text, json.dumps(extracted_data), json.dumps(keywords)))
        candidate_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return candidate_id

    def store_match(self, job_id: int, candidate_id: int, score: float) -> None:
        """Store a match between job and candidate"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO matches (job_id, candidate_id, match_score, status)
        VALUES (?, ?, ?, CASE WHEN ? >= 80 THEN 'shortlisted' ELSE 'pending' END)
        """, (job_id, candidate_id, score, score))
        conn.commit()
        conn.close()
        
    def store_recommendation(self, candidate_id: int, job_id: int, score: float) -> None:
        """Store a job recommendation for a candidate"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO recommendations (candidate_id, job_id, recommendation_score)
        VALUES (?, ?, ?)
        """, (candidate_id, job_id, score))
        conn.commit()
        conn.close()

    # PDF Processing
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
        return text

    # Keyword Extraction
    def extract_keywords_from_cv(self, cv_text: str) -> List[str]:
        """Extract keywords from a CV using LLM"""
        prompt = f"""
        Extract the top 20 most relevant professional keywords from this CV. 
        These should include technical skills, soft skills, tools, technologies, 
        qualifications, and domain expertise.
        
        Return only a JSON array of keywords without any additional text.
        
        CV Text:
        {cv_text[:3000]}  # Limiting text length for API constraints
        """
        
        response = self.llm([HumanMessage(content=prompt)])
        try:
            keywords = json.loads(response.content)
            return keywords
        except json.JSONDecodeError:
            # Fallback to simple extraction
            words = cv_text.lower().split()
            return list(set([w for w in words if len(w) > 4]))[:20]
    
    def extract_keywords_from_job(self, job_description: str) -> List[str]:
        """Extract keywords from a job description using LLM"""
        prompt = f"""
        Extract the top 20 most relevant keywords from this job description.
        Focus on required skills, qualifications, technologies, tools, and responsibilities.
        
        Return only a JSON array of keywords without any additional text.
        
        Job Description:
        {job_description[:3000]}  # Limiting text length for API constraints
        """
        
        response = self.llm([HumanMessage(content=prompt)])
        try:
            keywords = json.loads(response.content)
            return keywords
        except json.JSONDecodeError:
            # Fallback to simple extraction
            words = job_description.lower().split()
            return list(set([w for w in words if len(w) > 4]))[:20]

    # Agent 1: Job Description Summarizer
    def summarize_jd(self, job_description: str) -> str:
        """Summarize a job description using LLM"""
        prompt = f"""
        Analyze this job description and extract key information in the following structured JSON format:
        
        {{
            "Job Title": "extracted job title",
            "Required Skills": ["list", "of", "skills"],
            "Required Experience": "years and type",
            "Qualifications": ["education/certifications"],
            "Key Responsibilities": ["bullet", "points"]
        }}
        
        Job Description:
        {job_description}
        """
        
        response = self.llm([HumanMessage(content=prompt)])
        return response.content

    # Agent 2: Recruiting Agent (CV Parser and Matcher)
    def extract_cv_data(self, cv_text: str) -> Dict:
        """Extract structured data from a CV using LLM"""
        prompt = f"""
        Extract the following information from this CV in JSON format:
        {{
            "name": "candidate name",
            "email": "email address",
            "education": [
                {{
                    "degree": "degree name",
                    "institution": "school/university",
                    "year": "graduation year"
                }}
            ],
            "work_experience": [
                {{
                    "job_title": "position",
                    "company": "company name",
                    "duration": "employment period",
                    "responsibilities": ["list", "of", "responsibilities"]
                }}
            ],
            "skills": {{
                "technical": ["list", "of", "technical skills"],
                "soft": ["list", "of", "soft skills"]
            }},
            "certifications": ["list", "of", "certifications"]
        }}
        
        CV Text:
        {cv_text[:4000]}  # Limiting text length for API constraints
        """
        
        response = self.llm([HumanMessage(content=prompt)])
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback with a simple extraction
            return {
                "name": "Unknown",
                "email": "unknown@example.com",
                "skills": {
                    "technical": [],
                    "soft": []
                },
                "work_experience": []
            }

    def estimate_experience(self, work_experience: List[Dict]) -> float:
        """Estimate total years of experience from work history"""
        total_years = 0
        for job in work_experience:
            duration = job.get('duration', '')
            # Simple parsing - could be enhanced
            if 'year' in duration.lower():
                try:
                    years = float(''.join(c for c in duration.split()[0] if c.isdigit() or c == '.'))
                    total_years += years
                except (ValueError, IndexError):
                    total_years += 1
            elif any(word in duration.lower() for word in ['years', 'yr']):
                try:
                    years = float(''.join(c for c in duration.split()[0] if c.isdigit() or c == '.'))
                    total_years += years
                except (ValueError, IndexError):
                    total_years += 1
            else:
                total_years += 0.5  # Default for unspecified durations
        return total_years

    def calculate_match(self, job_summary: str, cv_data: Dict, job_keywords: List[str] = None, cv_keywords: List[str] = None) -> float:
        """Calculate match score between job and candidate"""
        try:
            job_data = json.loads(job_summary)
        except json.JSONDecodeError:
            job_data = {"Required Skills": [], "Required Experience": "0"}

        # Skill matching using TF-IDF
        job_skills = " ".join(job_data.get("Required Skills", []))
        cv_skills = " ".join(cv_data["skills"].get("technical", []) + cv_data["skills"].get("soft", []))
        
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([job_skills, cv_skills])
            skill_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except ValueError:
            skill_similarity = 0

        # Keyword matching if available
        keyword_similarity = 0
        if job_keywords and cv_keywords:
            job_keywords_text = " ".join(job_keywords)
            cv_keywords_text = " ".join(cv_keywords)
            try:
                keyword_matrix = vectorizer.fit_transform([job_keywords_text, cv_keywords_text])
                keyword_similarity = cosine_similarity(keyword_matrix[0:1], keyword_matrix[1:2])[0][0]
            except ValueError:
                keyword_similarity = 0

        # Experience matching
        try:
            job_exp = float(''.join(c for c in job_data.get("Required Experience", "0").split()[0] if c.isdigit() or c == '.'))
        except (ValueError, AttributeError):
            job_exp = 0
            
        cv_exp = self.estimate_experience(cv_data.get("work_experience", []))
        exp_match = min(cv_exp / max(job_exp, 1), 1.5)  # Allow over-qualification up to 150%

        # Qualification matching (simple binary check)
        qual_match = 0
        cv_education = []
        for edu in cv_data.get("education", []):
            if isinstance(edu, dict):
                cv_education.append(edu.get("degree", ""))
            else:
                cv_education.append(str(edu))
        
        for qual in job_data.get("Qualifications", []):
            if any(q.lower() in qual.lower() or qual.lower() in q.lower() 
                  for q in cv_education + cv_data.get("certifications", [])):
                qual_match = 1
                break

        # Weighted average for final score
        final_score = (
            0.35 * skill_similarity + 
            0.25 * keyword_similarity +
            0.25 * exp_match + 
            0.15 * qual_match
        ) * 100  # Convert to percentage

        return round(final_score, 2)

    # Agent 3: Shortlisting Agent
    def shortlist_candidates(self, job_id: int, threshold: float = 80) -> List[Tuple[int, float]]:
        """Shortlist candidates who meet the threshold"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute("""
        UPDATE matches 
        SET status = 'shortlisted' 
        WHERE job_id = ? AND match_score >= ? AND status = 'pending'
        """, (job_id, threshold))
        
        cursor.execute("""
        SELECT candidate_id, match_score 
        FROM matches 
        WHERE job_id = ? AND status = 'shortlisted'
        """, (job_id,))
        
        shortlisted = cursor.fetchall()
        conn.commit()
        conn.close()
        return shortlisted

    # NEW: Job Recommendation System
    def recommend_jobs_for_candidate(self, candidate_id: int, top_n: int = 5) -> List[Tuple[int, str, float]]:
        """Find the best matching jobs for a specific candidate"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Get candidate data
        cursor.execute("""
        SELECT extracted_data, keywords 
        FROM candidates 
        WHERE candidate_id = ?
        """, (candidate_id,))
        candidate_data = cursor.fetchone()
        
        if not candidate_data:
            conn.close()
            return []
            
        extracted_data = json.loads(candidate_data[0])
        cv_keywords = json.loads(candidate_data[1]) if candidate_data[1] else []
        
        # Get all jobs
        cursor.execute("SELECT job_id, job_name, summary, keywords FROM jobs")
        jobs = cursor.fetchall()
        conn.close()
        
        # Calculate match scores for all jobs
        job_scores = []
        for job_id, job_name, summary, keywords_json in jobs:
            job_keywords = json.loads(keywords_json) if keywords_json else []
            match_score = self.calculate_match(summary, extracted_data, job_keywords, cv_keywords)
            job_scores.append((job_id, job_name, match_score))
            
            # Store the recommendation
            self.store_recommendation(candidate_id, job_id, match_score)
            
        # Sort by score and return top N
        job_scores.sort(key=lambda x: x[2], reverse=True)
        return job_scores[:top_n]
        
    def get_candidate_job_recommendations(self, candidate_id: int) -> pd.DataFrame:
        """Get all job recommendations for a candidate as a DataFrame"""
        conn = sqlite3.connect(self.db_name)
        query = """
        SELECT j.job_id, j.job_name, r.recommendation_score
        FROM recommendations r
        JOIN jobs j ON r.job_id = j.job_id
        WHERE r.candidate_id = ?
        ORDER BY r.recommendation_score DESC
        """
        df = pd.read_sql_query(query, conn, params=(candidate_id,))
        conn.close()
        return df

    # Agent 4: Interview Scheduler Agent
    def send_interview_email(self, candidate_email: str, job_title: str, 
                           available_slots: List[str], sender_email: str, 
                           sender_password: str, smtp_server: str = "smtp.gmail.com", 
                           smtp_port: int = 587) -> bool:
        """Send interview invitation email to candidate"""
        subject = f"Interview Invitation for {job_title}"
        
        body = f"""
        Dear Candidate,
        
        Congratulations! You've been shortlisted for the position of {job_title}.
        Please select your preferred interview slot from the following options:
        
        {', '.join(f"{i+1}. {slot}" for i, slot in enumerate(available_slots))}
        
        Reply to this email with your preferred time slot number.
        
        Best regards,
        Recruitment Team
        """
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = candidate_email
        
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Failed to send email: {e}")
            return False

    # Main Workflow Methods
    # def process_job_descriptions(self, job_desc_csv: str) -> None:
    #     """Process a CSV file containing job descriptions"""
    #     jobs_df = pd.read_csv(job_desc_csv)
    #     for _, row in jobs_df.iterrows():
    #         self.store_job(row['job_name'], row['job_description'])
    
    
    def process_job_descriptions(self, job_desc_csv: str) -> None:
        """Process a CSV file containing job descriptions"""
        encodings_to_try = ['utf-8', 'latin1', 'ISO-8859-1', 'windows-1252']
        
        for encoding in encodings_to_try:
            try:
                jobs_df = pd.read_csv(job_desc_csv, encoding=encoding)
                for _, row in jobs_df.iterrows():
                    self.store_job(row['Job Title'], row['Job Description'])
                return  # Success!
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with encoding {encoding}: {e}")
                continue
                
        raise ValueError(f"Could not read {job_desc_csv} with any supported encoding")

    def process_cvs(self, cv_folder: str) -> None:
        """Process all CVs in a folder (both PDF and text files)"""
        for cv_file in os.listdir(cv_folder):
            cv_path = os.path.join(cv_folder, cv_file)
            cv_text = ""
            
            # Process based on file type
            if cv_file.lower().endswith('.pdf'):
                cv_text = self.extract_text_from_pdf(cv_path)
            elif cv_file.endswith('.txt'):
                with open(cv_path, 'r', encoding='utf-8', errors='ignore') as f:
                    cv_text = f.read()
            else:
                continue  # Skip unsupported file types
                
            if not cv_text.strip():
                print(f"Warning: Could not extract text from {cv_file}")
                continue
                
            # Extract name from filename and attempt to extract email from text
            name = os.path.splitext(cv_file)[0].replace('_', ' ')
            
            # Extract email from CV text (simplified)
            email = "candidate@example.com"  # Default
            import re
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', cv_text)
            if email_match:
                email = email_match.group(0)
                
            # Store CV data
            self.store_cv(name, email, cv_text)

    def run_screening(self, job_id: int) -> List[Tuple[int, float]]:
        """Run the complete screening process for a job"""
        # Get job summary and keywords
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT summary, keywords FROM jobs WHERE job_id = ?", (job_id,))
        job_data = cursor.fetchone()
        
        if not job_data:
            conn.close()
            return []
            
        job_summary, job_keywords_json = job_data
        job_keywords = json.loads(job_keywords_json) if job_keywords_json else []
        
        # Get all candidates
        cursor.execute("SELECT candidate_id, extracted_data, keywords FROM candidates")
        candidates = cursor.fetchall()
        conn.close()
        
        # Calculate matches
        for candidate_id, extracted_data_json, cv_keywords_json in candidates:
            extracted_data = json.loads(extracted_data_json)
            cv_keywords = json.loads(cv_keywords_json) if cv_keywords_json else []
            score = self.calculate_match(job_summary, extracted_data, job_keywords, cv_keywords)
            self.store_match(job_id, candidate_id, score)
        
        # Shortlist candidates
        return self.shortlist_candidates(job_id)

    def generate_job_recommendations_for_all_candidates(self) -> None:
        """Generate job recommendations for all candidates in the system"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT candidate_id FROM candidates")
        candidates = cursor.fetchall()
        conn.close()
        
        for (candidate_id,) in candidates:
            self.recommend_jobs_for_candidate(candidate_id)
    
    def get_job_matches(self, job_id: int) -> pd.DataFrame:
        """Get all matches for a job as a DataFrame"""
        conn = sqlite3.connect(self.db_name)
        query = """
        SELECT c.name, c.email, m.match_score, m.status
        FROM matches m
        JOIN candidates c ON m.candidate_id = c.candidate_id
        WHERE m.job_id = ?
        ORDER BY m.match_score DESC
        """
        df = pd.read_sql_query(query, conn, params=(job_id,))
        conn.close()
        return df

# Example Usage
if __name__ == "__main__":
    # Initialize the system
    screening_system = JobScreeningSystem()
    
    # Load data (replace with your actual file paths)
    screening_system.process_job_descriptions("./job_description.csv")
    screening_system.process_cvs("./cv_folder/")  # Will process both PDF and text files
    
    # Generate job recommendations for all candidates
    screening_system.generate_job_recommendations_for_all_candidates()
    
    # Example: Get job recommendations for a specific candidate
    candidate_id = 1  # Adjust as needed
    recommendations = screening_system.recommend_jobs_for_candidate(candidate_id)
    
    print(f"Top job recommendations for candidate {candidate_id}:")
    for job_id, job_name, score in recommendations:
        print(f"Job: {job_name}, Matching Score: {score}%")
    
    # Run screening for a specific job
    job_id = 1  # Adjust as needed
    shortlisted = screening_system.run_screening(job_id)
    
    # Get results
    if shortlisted:
        results = screening_system.get_job_matches(job_id)
        print("\nMatching Results:")
        print(results)