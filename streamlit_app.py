from dotenv import load_dotenv
from matplotlib import pyplot as plt
load_dotenv()

import base64
import streamlit as st
import os
import io
import json
import sqlite3
import pandas as pd
import fitz  # PyMuPDF
import datetime
import google.generativeai as genai
import time
import re
from collections import Counter

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect('job_screening.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS job_descriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        company TEXT,
        summary TEXT,
        required_skills TEXT,
        required_experience TEXT,
        qualifications TEXT,
        responsibilities TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        education TEXT,
        experience TEXT,
        skills TEXT,
        certifications TEXT,
        resume_data BLOB,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER,
        candidate_id INTEGER,
        match_score REAL,
        strengths TEXT,
        weaknesses TEXT,
        interview_requested BOOLEAN DEFAULT FALSE,
        interview_date TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES job_descriptions (id),
        FOREIGN KEY (candidate_id) REFERENCES candidates (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Agent 1: Job Description Summarizer
def job_description_summarizer(job_text):
    prompt = """
    You are a specialized AI agent for summarizing job descriptions. 
    Based on the provided job description, extract and organize the following information:
    
    1. Job Title (if not mentioned, use "Unknown")
    2. Company (if not mentioned, use "Unknown")
    3. Brief Summary (2-3 sentences)
    4. Required Skills (list format, empty list if none found)
    5. Required Experience (years and relevant domains, "Not specified" if none found)
    6. Required Qualifications (education, certifications, empty list if none found)
    7. Job Responsibilities (list format, empty list if none found)
    
    Format the response as a JSON object with these exact field names:
    - title
    - company
    - summary
    - required_skills
    - required_experience
    - qualifications
    - responsibilities
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt, job_text])
    
    try:
        # Extract the JSON data from the response
        response_text = response.text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        
        # Parse JSON and ensure all fields are present
        job_summary = json.loads(json_str)
        
        # Set default values for missing fields
        defaults = {
            "title": "Unknown",
            "company": "Unknown",
            "summary": "No summary available",
            "required_skills": [],
            "required_experience": "Not specified",
            "qualifications": [],
            "responsibilities": []
        }
        
        # Merge with defaults to ensure all keys exist
        return {**defaults, **job_summary}
        
    except Exception as e:
        st.error(f"Error parsing job description summary: {e}")
        return {
            "title": "Unknown",
            "company": "Unknown",
            "summary": "Error in parsing job description",
            "required_skills": [],
            "required_experience": "Not specified",
            "qualifications": [],
            "responsibilities": []
        }
# Agent 2: Resume Parser/Recruiting Agent
def resume_parser(pdf_content, job_summary):
    prompt = """
    You are a specialized AI recruiting agent. Extract all relevant information from the provided resume and match it against the job requirements.
    
    Extract the following (use "Unknown" or empty lists if information is not found):
    1. Candidate Name (field name: "name")
    2. Email Address (field name: "email")
    3. Education Background (field name: "education" - list of degrees, institutions, years)
    4. Work Experience (field name: "experience" - list of companies, roles, durations, achievements)
    5. Skills (field name: "skills" - list of technical, soft, and other relevant skills)
    6. Certifications (field name: "certifications" - list)
    
    Format your response as a JSON object with these exact field names.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([prompt, pdf_content[0], json.dumps(job_summary)])
    
    try:
        # Extract the JSON data from the response
        response_text = response.text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        
        # Parse JSON and ensure all fields are present
        candidate_data = json.loads(json_str)
        
        # Set default values for missing fields
        defaults = {
            "name": "Unknown",
            "email": "unknown@example.com",
            "education": [],
            "experience": [],
            "skills": [],
            "certifications": []
        }
        
        # Merge with defaults to ensure all keys exist
        return {**defaults, **candidate_data}
        
    except Exception as e:
        st.error(f"Error parsing resume data: {e}")
        return {
            "name": "Unknown",
            "email": "unknown@example.com",
            "education": [],
            "experience": [],
            "skills": [],
            "certifications": []
        }
    
# Agent 3: Match Score Calculator
def calculate_match_score(candidate_data, job_summary):
    prompt = f"""
    You are a specialized AI matching agent. Calculate a match score between the candidate and the job description.
    Based on the following data:
    
    Job Requirements:
    {json.dumps(job_summary, indent=2)}
    
    Candidate Profile:
    {json.dumps(candidate_data, indent=2)}
    
    Calculate a match percentage (0-100%) and explain the strengths and weaknesses of this match.
    Format your response as a JSON with these fields:
    1. match_score (a number between 0 and 100)
    2. strengths (array of strings)
    3. weaknesses (array of strings)
    4. recommendation (string: "Interview" if match_score > 80, "Consider" if match_score > 60, "Reject" if match_score <= 60)
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    try:
        # Extract the JSON data from the response
        response_text = response.text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error calculating match score: {e}")
        return {
            "match_score": 0,
            "strengths": [],
            "weaknesses": ["Error in processing"],
            "recommendation": "Reject"
        }

# Agent 4: Interview Scheduler
def create_interview_request(candidate_data, job_summary, match_data):
    prompt = f"""
    You are an AI interview scheduler. Create a personalized interview request email for the candidate.
    The email should include:
    
    1. A greeting addressing the candidate by name
    2. Brief introduction about the company and role
    3. Express interest based on their specific qualifications that matched well
    4. Propose 3 potential interview slots (use next week's dates and business hours)
    5. Mention the interview format (e.g., video call, technical assessment, etc.)
    6. Request confirmation of availability
    7. Professional closing
    
    Candidate: {candidate_data['name']}
    Job: {job_summary['title']} at {job_summary['company']}
    Key Strengths: {', '.join(match_data['strengths'][:3])}
    
    Write a complete, professional email ready to be sent.
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    return response.text

# PDF to Image Converter
def convert_pdf_to_image(uploaded_file):
    if uploaded_file is not None:
        # Read the PDF file
        pdf_bytes = uploaded_file.read()
        
        # Use PyMuPDF (fitz) to convert PDF to image
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        first_page = pdf_document.load_page(0)  # Load the first page
        
        # Render page to an image with a higher resolution
        pix = first_page.get_pixmap(matrix=fitz.Matrix(2, 2))
        
        # Convert to PIL Image
        from PIL import Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts, pdf_bytes
    else:
        raise FileNotFoundError("No file uploaded")

# Save data to database
def save_job_to_db(job_data):
    conn = sqlite3.connect('job_screening.db')
    c = conn.cursor()
    
    c.execute('''
    INSERT INTO job_descriptions 
    (title, company, summary, required_skills, required_experience, qualifications, responsibilities)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        job_data.get('title', 'Unknown'),
        job_data.get('company', 'Unknown'),
        job_data.get('summary', 'No summary available'),
        json.dumps(job_data.get('required_skills', [])),
        job_data.get('required_experience', 'Not specified'),
        json.dumps(job_data.get('qualifications', [])),
        json.dumps(job_data.get('responsibilities', []))
    ))
    
    job_id = c.lastrowid
    conn.commit()
    conn.close()
    return job_id

def save_candidate_to_db(candidate_data, resume_data):
    conn = sqlite3.connect('job_screening.db')
    c = conn.cursor()
    
    c.execute('''
    INSERT INTO candidates 
    (name, email, education, experience, skills, certifications, resume_data)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        candidate_data.get('name', 'Unknown'),
        candidate_data.get('email', 'unknown@example.com'),
        json.dumps(candidate_data.get('education', [])),
        json.dumps(candidate_data.get('experience', [])),
        json.dumps(candidate_data.get('skills', [])),
        json.dumps(candidate_data.get('certifications', [])),
        resume_data
    ))
    
    candidate_id = c.lastrowid
    conn.commit()
    conn.close()
    return candidate_id

def save_match_to_db(job_id, candidate_id, match_data):
    conn = sqlite3.connect('job_screening.db')
    c = conn.cursor()
    
    c.execute('''
    INSERT INTO matches 
    (job_id, candidate_id, match_score, strengths, weaknesses)
    VALUES (?, ?, ?, ?, ?)
    ''', (
        job_id,
        candidate_id,
        match_data['match_score'],
        json.dumps(match_data['strengths']),
        json.dumps(match_data['weaknesses'])
    ))
    
    match_id = c.lastrowid
    conn.commit()
    conn.close()
    return match_id

def update_match_with_interview(match_id, interview_date):
    conn = sqlite3.connect('job_screening.db')
    c = conn.cursor()
    
    c.execute('''
    UPDATE matches 
    SET interview_requested = TRUE, interview_date = ?
    WHERE id = ?
    ''', (interview_date, match_id))
    
    conn.commit()
    conn.close()

def get_all_jobs():
    conn = sqlite3.connect('job_screening.db')
    df = pd.read_sql_query("SELECT id, title, company FROM job_descriptions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_all_candidates():
    conn = sqlite3.connect('job_screening.db')
    df = pd.read_sql_query("SELECT id, name, email FROM candidates ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_job_details(job_id):
    conn = sqlite3.connect('job_screening.db')
    c = conn.cursor()
    c.execute("SELECT * FROM job_descriptions WHERE id = ?", (job_id,))
    job = c.fetchone()
    column_names = [description[0] for description in c.description]
    conn.close()
    
    job_dict = {column_names[i]: job[i] for i in range(len(column_names))}
    
    # Parse JSON strings
    for key in ['required_skills', 'qualifications', 'responsibilities']:
        if key in job_dict and job_dict[key]:
            job_dict[key] = json.loads(job_dict[key])
    
    return job_dict

def get_candidate_details(candidate_id):
    conn = sqlite3.connect('job_screening.db')
    c = conn.cursor()
    c.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
    candidate = c.fetchone()
    column_names = [description[0] for description in c.description]
    conn.close()
    
    candidate_dict = {column_names[i]: candidate[i] for i in range(len(column_names))}
    
    # Parse JSON strings
    for key in ['education', 'experience', 'skills', 'certifications']:
        if key in candidate_dict and candidate_dict[key]:
            candidate_dict[key] = json.loads(candidate_dict[key])
    
    return candidate_dict

def get_matches(job_id=None, candidate_id=None):
    conn = sqlite3.connect('job_screening.db')
    
    if job_id and candidate_id:
        query = """
        SELECT m.*, j.title as job_title, j.company, c.name as candidate_name, c.email
        FROM matches m
        JOIN job_descriptions j ON m.job_id = j.id
        JOIN candidates c ON m.candidate_id = c.id
        WHERE m.job_id = ? AND m.candidate_id = ?
        """
        df = pd.read_sql_query(query, conn, params=(job_id, candidate_id))
    elif job_id:
        query = """
        SELECT m.*, j.title as job_title, j.company, c.name as candidate_name, c.email
        FROM matches m
        JOIN job_descriptions j ON m.job_id = j.id
        JOIN candidates c ON m.candidate_id = c.id
        WHERE m.job_id = ?
        ORDER BY m.match_score DESC
        """
        df = pd.read_sql_query(query, conn, params=(job_id,))
    elif candidate_id:
        query = """
        SELECT m.*, j.title as job_title, j.company, c.name as candidate_name, c.email
        FROM matches m
        JOIN job_descriptions j ON m.job_id = j.id
        JOIN candidates c ON m.candidate_id = c.id
        WHERE m.candidate_id = ?
        ORDER BY m.match_score DESC
        """
        df = pd.read_sql_query(query, conn, params=(candidate_id,))
    else:
        query = """
        SELECT m.*, j.title as job_title, j.company, c.name as candidate_name, c.email
        FROM matches m
        JOIN job_descriptions j ON m.job_id = j.id
        JOIN candidates c ON m.candidate_id = c.id
        ORDER BY m.match_score DESC
        """
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df

# Extract common skills from datasets
def extract_skills_from_texts(texts_list):
    # Define common tech skills (add more as needed)
    skill_patterns = [
        r'\bpython\b', r'\bjava\b', r'\bjavascript\b', r'\bjs\b', r'\bhtml\b', r'\bcss\b',
        r'\bsql\b', r'\bnosql\b', r'\bmongodb\b', r'\baws\b', r'\bazure\b', r'\bgcp\b',
        r'\bcloud\b', r'\bdocker\b', r'\bkubernetes\b', r'\bk8s\b', r'\breact\b', r'\bangular\b',
        r'\bvue\b', r'\bnode\.?js\b', r'\bexpress\b', r'\bdjango\b', r'\bflask\b', r'\brails\b',
        r'\bruby\b', r'\bc\+\+\b', r'\bc#\b', r'\b\.net\b', r'\bphp\b', r'\blaravel\b',
        r'\bwordpress\b', r'\bswift\b', r'\bobjective-c\b', r'\bkotlin\b', r'\bandroid\b',
        r'\bios\b', r'\bmachine learning\b', r'\bml\b', r'\bai\b', r'\bdata science\b',
        r'\bdata analytics\b', r'\bbig data\b', r'\bhadoop\b', r'\bspark\b', r'\btensorflow\b',
        r'\bpytorch\b', r'\bscala\b', r'\bgo\b', r'\bgolang\b', r'\brust\b', r'\bdevops\b',
        r'\bci/cd\b', r'\bagile\b', r'\bscrum\b', r'\bkanban\b', r'\bjira\b', r'\bconfluence\b',
        r'\bgit\b', r'\bgithub\b', r'\bbitbucket\b', r'\blinux\b', r'\bunix\b', r'\bwindows\b',
        r'\bmac\b', r'\brest\b', r'\bsoap\b', r'\bapi\b', r'\bmicroservices\b', r'\bsecurity\b',
        r'\bcryptography\b', r'\bblockchain\b', r'\btesting\b', r'\bunit testing\b', r'\bqa\b',
        r'\bseo\b', r'\banalytics\b', r'\biiot\b', r'\biot\b', r'\bembedded systems\b'
    ]
    
    all_skills = []
    
    for text in texts_list:
        if isinstance(text, str):
            text_lower = text.lower()
            for pattern in skill_patterns:
                matches = re.findall(pattern, text_lower)
                all_skills.extend(matches)
    
    # Count occurrences
    skill_counts = Counter(all_skills)
    return skill_counts.most_common(10)  # Top 10 skills

# Main Streamlit App
def main():
    # Initialize database
    init_db()
    
    st.set_page_config(page_title="AI Recruitment System", layout="wide")
    
    st.title("🤖 AI Recruitment System")
    st.subheader("Enhancing Job Screening with AI and Data Intelligence")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Process Job Description", 
        "📄 Process Resume", 
        "🔍 Match Analysis", 
        "📅 Interview Scheduling",
        "📊 Dashboard"
    ])
    
    # Tab 1: Process Job Description
    with tab1:
        st.header("Job Description Summarizer")
        st.info("This agent reads and summarizes key elements from the job description.")
        
        job_input = st.text_area("Enter Job Description:", height=300)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            process_jd = st.button("Process Job Description", key="process_jd", use_container_width=True)
        with col2:
            if process_jd and job_input:
                with st.spinner("Processing job description..."):
                    job_summary = job_description_summarizer(job_input)
                    job_id = save_job_to_db(job_summary)
                    st.session_state['last_job_id'] = job_id
                    st.session_state['last_job_summary'] = job_summary
        
        if 'last_job_summary' in st.session_state and process_jd:
            job_summary = st.session_state['last_job_summary']
            st.success(f"Job Description Processed! ID: {st.session_state['last_job_id']}")
            
            # Display job summary in an expandable section
            with st.expander("View Job Summary", expanded=True):
                st.subheader(f"{job_summary['title']} at {job_summary['company']}")
                st.write(job_summary['summary'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Required Skills")
                    for skill in job_summary['required_skills']:
                        st.markdown(f"- {skill}")
                    
                    st.subheader("Required Experience")
                    st.write(job_summary['required_experience'])
                
                with col2:
                    st.subheader("Qualifications")
                    for qual in job_summary['qualifications']:
                        st.markdown(f"- {qual}")
                    
                    st.subheader("Job Responsibilities")
                    for resp in job_summary['responsibilities']:
                        st.markdown(f"- {resp}")
    
    # Tab 2: Process Resume
    with tab2:
        st.header("Resume Parser")
        st.info("This agent extracts key data from resumes and compares with job requirements.")
        
        uploaded_file = st.file_uploader("Upload a Resume (PDF)", type=['pdf'])
        
        # Select job to match against
        jobs_df = get_all_jobs()
        if not jobs_df.empty:
            selected_job_id = st.selectbox(
                "Select Job to Match Against", 
                options=jobs_df['id'].tolist(),
                format_func=lambda x: f"{jobs_df.loc[jobs_df['id'] == x, 'title'].iloc[0]} at {jobs_df.loc[jobs_df['id'] == x, 'company'].iloc[0]} (ID: {x})"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                process_resume = st.button("Process Resume", key="process_resume", use_container_width=True)
            
            with col2:
                if process_resume and uploaded_file:
                    with st.spinner("Processing resume..."):
                        # Convert PDF to image format
                        pdf_content, pdf_bytes = convert_pdf_to_image(uploaded_file)
                        
                        # Get job details
                        job_details = get_job_details(selected_job_id)
                        
                        # Parse resume with recruiting agent
                        candidate_data = resume_parser(pdf_content, job_details)
                        
                        # Save candidate to database
                        candidate_id = save_candidate_to_db(candidate_data, pdf_bytes)
                        
                        # Calculate match score
                        match_data = calculate_match_score(candidate_data, job_details)
                        
                        # Save match to database
                        match_id = save_match_to_db(selected_job_id, candidate_id, match_data)
                        
                        st.session_state['last_candidate_id'] = candidate_id
                        st.session_state['last_match_id'] = match_id
                        st.session_state['last_candidate_data'] = candidate_data
                        st.session_state['last_match_data'] = match_data
            
            if 'last_candidate_data' in st.session_state and process_resume:
                candidate_data = st.session_state['last_candidate_data']
                match_data = st.session_state['last_match_data']
                
                st.success(f"Resume Processed! Candidate ID: {st.session_state['last_candidate_id']}")
                
                # Display match results
                with st.expander("View Candidate Summary", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader(f"{candidate_data['name']}")
                        st.write(f"**Email:** {candidate_data['email']}")
                        
                        st.subheader("Education")
                        for edu in candidate_data['education']:
                            st.markdown(f"- {edu}")
                        
                        st.subheader("Certifications")
                        for cert in candidate_data['certifications']:
                            st.markdown(f"- {cert}")
                    
                    with col2:
                        st.subheader("Experience")
                        for exp in candidate_data['experience']:
                            st.markdown(f"- {exp}")
                        
                        st.subheader("Skills")
                        for skill in candidate_data['skills']:
                            st.markdown(f"- {skill}")
                
                with st.expander("View Match Analysis", expanded=True):
                    match_score = match_data['match_score']
                    
                    # Visualize match score
                    st.metric("Match Score", f"{match_score}%")
                    
                    # Color the progress bar based on the match score
                    if match_score >= 80:
                        bar_color = "green"
                    elif match_score >= 60:
                        bar_color = "orange"
                    else:
                        bar_color = "red"
                    
                    st.progress(match_score/100)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Strengths")
                        for strength in match_data['strengths']:
                            st.markdown(f"- {strength}")
                    
                    with col2:
                        st.subheader("Areas for Improvement")
                        for weakness in match_data['weaknesses']:
                            st.markdown(f"- {weakness}")
                    
                    st.subheader("Recommendation")
                    st.info(match_data['recommendation'])
    
    # Tab 3: Match Analysis
        # Tab 3: Match Analysis
    with tab3:
        st.header("Candidate Matching & Shortlisting")
        st.info("This view shows match scores and allows shortlisting of candidates.")
        
        # Initialize filter variables
        filter_job_id = None
        filter_candidate_id = None
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            jobs_df = get_all_jobs()
            if not jobs_df.empty:
                filter_job_id = st.selectbox(
                    "Filter by Job", 
                    options=[None] + jobs_df['id'].tolist(),
                    format_func=lambda x: "All Jobs" if x is None else f"{jobs_df.loc[jobs_df['id'] == x, 'title'].iloc[0]} at {jobs_df.loc[jobs_df['id'] == x, 'company'].iloc[0]}"
                )
        
        with col2:
            candidates_df = get_all_candidates()
            if not candidates_df.empty:
                filter_candidate_id = st.selectbox(
                    "Filter by Candidate", 
                    options=[None] + candidates_df['id'].tolist(),
                    format_func=lambda x: "All Candidates" if x is None else f"{candidates_df.loc[candidates_df['id'] == x, 'name'].iloc[0]} ({candidates_df.loc[candidates_df['id'] == x, 'email'].iloc[0]})"
                )
        
        # Get and display matches - now filter_job_id and filter_candidate_id are always defined
        matches_df = get_matches(filter_job_id, filter_candidate_id)

    # Tab 4: Interview Scheduling
    with tab4:
        st.header("Interview Management")
        st.info("View and manage scheduled interviews.")
        
        # Get all matches with interviews scheduled
        conn = sqlite3.connect('job_screening.db')
        interviews_df = pd.read_sql_query("""
            SELECT m.id, m.interview_date, j.title as job_title, j.company, 
                   c.name as candidate_name, c.email, m.match_score
            FROM matches m
            JOIN job_descriptions j ON m.job_id = j.id
            JOIN candidates c ON m.candidate_id = c.id
            WHERE m.interview_requested = TRUE
            ORDER BY m.interview_date
        """, conn)
        conn.close()
        
        if not interviews_df.empty:
            # Format for display
            interviews_df['Interview Date'] = pd.to_datetime(interviews_df['interview_date']).dt.strftime('%Y-%m-%d')
            interviews_df['Match Score'] = interviews_df['match_score'].apply(lambda x: f"{x}%")
            
            display_df = interviews_df[['id', 'job_title', 'company', 'candidate_name', 'Interview Date', 'Match Score']]
            display_df.columns = ['ID', 'Job Title', 'Company', 'Candidate', 'Interview Date', 'Match Score']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Add filtering options
            col1, col2 = st.columns(2)
            with col1:
                date_filter = st.date_input("Filter by Date", value=None)
            with col2:
                score_threshold = st.slider("Minimum Match Score", 0, 100, 70)
            
            if date_filter:
                interviews_df = interviews_df[pd.to_datetime(interviews_df['interview_date']).dt.date == date_filter]
            interviews_df = interviews_df[interviews_df['match_score'] >= score_threshold]
            
            if not interviews_df.empty:
                st.subheader("Upcoming Interviews")
                for _, row in interviews_df.iterrows():
                    with st.expander(f"{row['candidate_name']} for {row['job_title']} on {row['interview_date']}"):
                        st.write(f"**Company:** {row['company']}")
                        st.write(f"**Candidate Email:** {row['email']}")
                        st.write(f"**Match Score:** {row['match_score']}%")
                        
                        # Add interview notes section
                        notes_key = f"interview_notes_{row['id']}"
                        notes = st.text_area("Interview Notes", key=notes_key)
                        
                        if st.button("Save Notes", key=f"save_notes_{row['id']}"):
                            # Here you would typically save these notes to your database
                            st.success("Notes saved successfully!")
            else:
                st.warning("No interviews match the selected filters.")
        else:
            st.warning("No interviews scheduled yet.")

    # Tab 5: Dashboard
    with tab5:
        st.header("Recruitment Analytics Dashboard")
        st.info("Visualize key metrics and trends in your recruitment pipeline.")
        
        # Get all data
        jobs_df = get_all_jobs()
        candidates_df = get_all_candidates()
        matches_df = get_matches()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Jobs Posted", len(jobs_df))
        with col2:
            st.metric("Total Candidates", len(candidates_df))
        with col3:
            st.metric("Total Matches Made", len(matches_df))
        
        # Visualizations
        if not matches_df.empty:
            st.subheader("Match Score Distribution")
            fig1, ax1 = plt.subplots()
            ax1.hist(matches_df['match_score'], bins=10, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Match Score (%)')
            ax1.set_ylabel('Count')
            st.pyplot(fig1)
            
            # Top skills analysis
            st.subheader("Top Skills in Demand")
            
            # Extract skills from job descriptions
            job_skills = []
            for _, job in jobs_df.iterrows():
                job_details = get_job_details(job['id'])
                if 'required_skills' in job_details:
                    job_skills.extend(job_details['required_skills'])
            
            # Extract skills from candidates
            candidate_skills = []
            for _, candidate in candidates_df.iterrows():
                candidate_details = get_candidate_details(candidate['id'])
                if 'skills' in candidate_details:
                    candidate_skills.extend(candidate_details['skills'])
            
            # Get top skills
            top_job_skills = extract_skills_from_texts(job_skills)
            top_candidate_skills = extract_skills_from_texts(candidate_skills)
            
            # Display in columns
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Most Required Skills in Jobs**")
                for skill, count in top_job_skills:
                    st.markdown(f"- {skill.title()} ({count})")
            
            with col2:
                st.write("**Most Common Skills in Candidates**")
                for skill, count in top_candidate_skills:
                    st.markdown(f"- {skill.title()} ({count})")
            
            # Match success rate by job
            if not jobs_df.empty and not matches_df.empty:
                st.subheader("Match Success Rate by Job")
                job_match_stats = matches_df.groupby('job_title')['match_score'].mean().sort_values(ascending=False)
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                job_match_stats.plot(kind='bar', ax=ax2, color='lightgreen')
                ax2.set_ylabel('Average Match Score (%)')
                ax2.set_xlabel('Job Title')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2)
        else:
            st.warning("No match data available for analysis.")

# Run the app
if __name__ == "__main__":
    main()
