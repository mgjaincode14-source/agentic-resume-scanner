from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile")

from pydantic import BaseModel,Field
import fitz
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class Hire(BaseModel):
    file_path: str
    application: str
    experience_level: str = ""
    resume_score: int = 0
    candidate_name :str =""
    candidate_email :str=""
    response: str = ""

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text.strip()


from langgraph.graph import StateGraph,START,END
builder = StateGraph(Hire)

class CandidateInfo(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address of the candidate, or NOT_FOUND if not present")

class ExperienceLevel(BaseModel):
    level: str = Field(description="Experience level of the candidate: Entry-Level, Intermediate-Level or Senior-Level")

class ResumeScore(BaseModel):
    score: int = Field(description="Integer score from 1 to 10 based on resume quality, clarity and relevance to ML role")

def candidate_info(state: Hire) -> Hire:
    print("Extracting the name and the email of the candidate from the resume uploaded...")
    structured_llm = llm.with_structured_output(CandidateInfo)
    result = structured_llm.invoke(
        f"Extract the candidate's full name and email from this resume:\n\n{state.application}"
    )
    print(f"Name  : {result.name}")
    print(f"Email : {result.email}")
    return {"candidate_name": result.name, "candidate_email": result.email}

def candidate_experience(state: Hire) -> Hire:
    print("Evaluating the experience level of the candidates based on their resume uploaded...")
    structured_llm = llm.with_structured_output(ExperienceLevel)
    result = structured_llm.invoke(
        f"Based on this resume, categorize the candidate's experience level for a Machine Learning role.\n\n"
        f"Resume:\n{state.application}"
    )
    print(f"Experience Level : {result.level}")
    return {"experience_level": result.level}

def resume_score(state: Hire) -> Hire:
    print("Evaluating the resume and scoring them...")
    structured_llm = llm.with_structured_output(ResumeScore)
    result = structured_llm.invoke(
        f"Score this resume from 1 to 10 based on clarity, specificity and relevance to a Prompt engineering role.Be as strict as possible and stick to the job description \n\n"
        f"Resume:\n{state.application}"
    )
    print(f"Resume Score : {result.score}/10")
    return {"resume_score": result.score}

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

    # try:
    #     with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    #         server.login(sender, password)
    #         server.sendmail(sender, state.candidate_email, msg.as_string())
    #     print(f" Mail sent to {state.candidate_name} at {state.candidate_email}")
    #     return {"response": f"Assessment invitation sent to {state.candidate_email}"}
    # except Exception as e:
    #     print(f" Failed to send email: {e}")
    #     return {"response": f"Failed to send email: {e}"}

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
builder.add_node("reject_low_score",     reject_low_score)
builder.add_node("reject_no_email",      reject_no_email)

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

def run_candidate_screening(pdf_path: str):
    print(f"\nProcessing: {pdf_path}")
    print("=" * 50)
    pdf_text = extract_text_from_pdf(pdf_path)
    results = app.invoke({"file_path": pdf_path, "application": pdf_text})
    print("\n Final Results:")
    print(f"  Name             : {results['candidate_name']}")
    print(f"  Email            : {results['candidate_email']}")
    print(f"  Experience Level : {results['experience_level']}")
    print(f"  Resume Score     : {results['resume_score']}/10")
    print(f"  Response         : {results['response']}")

run_candidate_screening(r"C:\Users\moksh\Downloads\Mokshit_Jain_Caterpillar_Resume.pdf")