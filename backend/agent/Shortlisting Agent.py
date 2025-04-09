def shortlist_candidates(job_id, threshold=80):
    conn = sqlite3.connect('recruitment.db')
    cursor = conn.cursor()
    
    # Get all matches for this job
    cursor.execute("""
    SELECT candidate_id, match_score 
    FROM matches 
    WHERE job_id = ? AND status = 'pending'
    """, (job_id,))
    
    candidates = cursor.fetchall()
    shortlisted = [cand for cand in candidates if cand[1] >= threshold]
    
    # Update status
    for cand in shortlisted:
        cursor.execute("""
        UPDATE matches 
        SET status = 'shortlisted' 
        WHERE job_id = ? AND candidate_id = ?
        """, (job_id, cand[0]))
    
    conn.commit()
    conn.close()
    
    return shortlisted