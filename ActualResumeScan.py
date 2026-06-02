from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile")

from pydantic import BaseModel,Field
from pypdf import PdfReader
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class Hire(BaseModel):
    file_path: str
    application: str
    job_description: str = ""
    experience_level: str = ""
    resume_score: int = 0
    skill_match_reason: str = ""
    candidate_name :str =""
    candidate_email :str=""
    response: str = ""

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
        return full_text.strip()
    except Exception as e:
        print(f"  PDF read error: {e}")
        return ""

from langgraph.graph import StateGraph,START,END
builder = StateGraph(Hire)

class CandidateInfo(BaseModel):
    name: str = Field(description="Full name of the candidate or unknown if not present in the resume")
    email: str = Field(description="Email address of the candidate, or NOT_FOUND if not present")
    github_username: str = Field(description="GitHub username of the candidate, or NOT_FOUND if not present")

class ExperienceLevel(BaseModel):
    level: str = Field(description="Experience level of the candidate: Entry-Level, Intermediate-Level or Senior-Level")

class ResumeScore(BaseModel):
    score: int = Field(description="Integer score from 0 to 10 based on resume quality, clarity and relevance to the job description")
    reason: str = Field(description="One sentence explaining why this score was given")

class JDTechCheck(BaseModel):
    is_technical: bool = Field(description="True if the JD requires technical skills like coding, programming, ML, data etc. False if it is open to all backgrounds like writing, law, linguistics, teaching etc.")

class ProjectClaims(BaseModel):
    projects: list[str] = Field(description="List of project names or titles the candidate claims to have built, extracted from the resume")

class GitHubVerification(BaseModel):
    verification_status: str = Field(description="One of: verified, partial, unverified. verified = projects found on GitHub. partial = some found. unverified = none found.")
    explanation: str = Field(description="One sentence explaining what matched and what did not")

def candidate_info(state: Hire) -> Hire:
    print("Extracting candidate info from resume...")
    structured_llm = llm.with_structured_output(CandidateInfo)
    result = structured_llm.invoke(
        f"Extract the candidate's full name, email and GitHub username from this resume:\n\n{state.application}"
    )
    print(f"Name            : {result.name}")
    print(f"Email           : {result.email}")
    print(f"GitHub Username : {result.github_username}")
    return {
        "candidate_name": result.name, "candidate_email": result.email, "github_username":result.github_username
    }

def candidate_experience(state: Hire) -> Hire:
    print("Evaluating the experience level of the candidates based on their resume uploaded...")
    structured_llm = llm.with_structured_output(ExperienceLevel)
    result = structured_llm.invoke(
        f"Based on this resume, categorize the candidate's experience level for a Machine Learning role.\n\n"
        f"Job Description:\n{state.job_description}\n\n"
        f"Resume:\n{state.application}"
    )
    print(f"Experience Level : {result.level}")
    return {"experience_level": result.level}

def resume_score(state: Hire) -> Hire:
    print("Evaluating the resume and scoring them...")
    structured_llm = llm.with_structured_output(ResumeScore)
    result = structured_llm.invoke(
        f"You are a strict technical recruiter. Score this resume against the job description.\n\n"
        f"Scoring rules:\n"
        f"- If the candidate's core skills do NOT match the JD domain, score must be 0 to 3\n"
        f"- If there is partial match in skills or transferable skills, score 4 to 6\n"
        f"- Only score 7 to 10 if the candidate directly matches the required skills and domain\n"
        f"- Be strict. Do NOT reward general skills like communication or eagerness to learn\n"
        f"Job Description:\n{state.job_description}\n\n"
        f"Resume:\n{state.application}"
    )
    print(f"Resume Score  : {result.score}/10")
    print(f"Reason        : {result.reason}")
    return {"resume_score": result.score, "skill_match_reason": result.reason}

def invitation_for_Assessment(state: Hire) -> Hire:
    print(f"Sending assessment invitation to {state.candidate_email}...")
    sender   = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_APP_PASSWORD")

    subject = "Invitation for Assessment Round"
    body = f"""
Dear {state.candidate_name},

Congratulations! After reviewing your resume, we are pleased to inform you
that you have been shortlisted for the next stage of our recruitment process.

You are invited to appear for our Online Assessment Round.

Assessment Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Date     : To be communicated shortly
Duration : 60 minutes
Format   : Online (link will be shared separately)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Best regards,
HR & Team
    """
    msg = MIMEMultipart()
    msg["From"]    = sender
    msg["To"]      = state.candidate_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, state.candidate_email, msg.as_string())
        print(f" Mail sent to {state.candidate_name} at {state.candidate_email}")
        return {"response": f"Assessment invitation sent to {state.candidate_email}"}
    except Exception as e:
        print(f" Failed to send email: {e}")
        return {"response": f"Failed to send email: {e}"}

def reject_low_score(state: Hire) -> Hire:
    print(f"Rejecting {state.candidate_name} — score below 7.")
    return {"response": f"Rejected: Resume score {state.resume_score}/10 is below threshold."}

def reject_no_email(state: Hire) -> Hire:
    print(f"Rejecting — no email found in resume.")
    return {"response": "Rejected: No email address found in resume."}

def route_email_check(state: Hire) -> str:
    if state.candidate_email in ("NOT_FOUND", ""):
        return "reject_no_email"
    return "candidate_experience"

def route_score(state: Hire) -> str:
    if state.resume_score >= 7:
        return "Assessment"
    return "reject_low_score"


builder.add_node("candidate_info",candidate_info)
builder.add_node("candidate_experience",candidate_experience)
builder.add_node("resume_score",resume_score)
builder.add_node("Assessment",invitation_for_Assessment)
builder.add_node("reject_low_score",reject_low_score)
builder.add_node("reject_no_email",reject_no_email)

builder.add_edge(START,"candidate_info")
builder.add_conditional_edges("candidate_info",route_email_check)
builder.add_edge("candidate_experience","resume_score")
builder.add_conditional_edges("resume_score",route_score)
builder.add_edge("Assessment",END)
builder.add_edge("reject_low_score",END)
builder.add_edge("reject_no_email",END)

app=builder.compile()
graph_image = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)
print("Graph saved as graph.png!")

def run_candidate_screening(pdf_path: str,job_description: str):
    print(f"\nProcessing: {pdf_path}")
    print("=" * 50)
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print("PDF is blank or unreadable. Skipping.")
        return
    
    results = app.invoke({"file_path": pdf_path, "application": pdf_text,"job_description": job_description})
    print("\n Final Results:")
    print(f"  Name             : {results['candidate_name']}")
    print(f"  Email            : {results['candidate_email']}")
    print(f"  Experience Level : {results['experience_level']}")
    print(f"  Resume Score     : {results['resume_score']}/10")
    print(f"  Skill Match Reason : {results.get('skill_match_reason', 'No reason provided')}")
    print(f"  Response         : {results.get('response', 'No response generated')}")


def main():
    job_description = input("Enter the Job Description:\n") 

    
    resume_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resumeSamples")
 
    if not os.path.exists(resume_folder):
        os.makedirs(resume_folder)
        print(f"\nCreated resumes folder at: {resume_folder}")
        print("Drop PDF resumes into this folder and run again.")
        return
 
    resumes = [
        os.path.join(resume_folder, f)
        for f in os.listdir(resume_folder)
        if f.lower().endswith(".pdf")
    ]
 
    if not resumes:
        print("No PDF files found in the resumes folder.")
        return
 
    for resume in resumes:
        run_candidate_screening(resume, job_description)
        print("-" * 55)



main()

