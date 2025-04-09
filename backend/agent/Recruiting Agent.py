import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_cv_data(cv_text):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    prompt = f"""
    Extract the following information from this CV in JSON format:
    - name
    - education (degree, institution, year)
    - work_experience (job_title, company, duration, responsibilities)
    - skills (technical and soft skills)
    - certifications
    
    CV Text:
    {cv_text}
    """
    
    response = llm([HumanMessage(content=prompt)])
    return json.loads(response.content)

def calculate_match(job_summary, cv_data):
    # Convert data to comparable strings
    job_skills = ", ".join(job_summary['Required Skills'])
    cv_skills = ", ".join(cv_data['skills'])
    
    # Experience matching
    job_exp = job_summary['Required Experience']
    cv_exp = estimate_experience(cv_data['work_experience'])
    
    # Vectorize and compare
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_skills, cv_skills])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Calculate overall score (simplified)
    exp_match = 1 if cv_exp >= job_exp else cv_exp/job_exp
    final_score = 0.6*similarity + 0.4*exp_match
    
    return final_score * 100  # as percentage