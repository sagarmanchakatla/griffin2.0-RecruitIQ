import smtplib
from email.mime.text import MIMEText

def send_interview_email(candidate_email, job_title, available_slots):
    subject = f"Interview Invitation for {job_title}"
    
    body = f"""
    Dear Candidate,
    
    Congratulations! You've been shortlisted for the position of {job_title}.
    Please select your preferred interview slot from the following options:
    
    {', '.join(available_slots)}
    
    Reply to this email with your preferred time.
    
    Best regards,
    Recruitment Team
    """
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = "recruitment@company.com"
    msg['To'] = candidate_email
    
    # Send email (configure your SMTP server)
    with smtplib.SMTP('smtp.server.com', 587) as server:
        server.starttls()
        server.login("user", "password")
        server.send_message(msg)