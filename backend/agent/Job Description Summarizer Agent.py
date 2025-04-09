from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

def summarize_jd(job_description):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    prompt = f"""
    Analyze this job description and extract key information in the following structured format:
    
    Job Title: [extract job title]
    Required Skills: [comma-separated list]
    Required Experience: [years and type]
    Qualifications: [education/certifications]
    Key Responsibilities: [bullet points]
    
    Job Description:
    {job_description}
    """
    
    response = llm([HumanMessage(content=prompt)])
    return response.content