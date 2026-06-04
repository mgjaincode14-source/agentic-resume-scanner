from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile")

from pydantic import BaseModel, Field
from pypdf import PdfReader
from enum import Enum
import os, re, requests, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langgraph.graph import StateGraph, START, END

# ══════════════════════════════════════════════════════════════════
# ENUMS — no more hardcoded strings
# ══════════════════════════════════════════════════════════════════
class VerificationStatus(str, Enum):
    VERIFIED    = "verified"       # all projects found and confirmed
    PARTIAL     = "partial"        # some projects confirmed
    UNVERIFIED  = "unverified"     # projects claimed but nothing found
    SKIPPED     = "skipped"        # no github / no projects section
    NOT_APPLICABLE = "not_applicable"  # non-tech role

class ExperienceLabel(str, Enum):
    ENTRY        = "Entry-Level"
    INTERMEDIATE = "Intermediate-Level"
    SENIOR       = "Senior-Level"

# ══════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════
class Hire(BaseModel):
    file_path:            str
    application:          str
    job_description:      str = ""
    experience_level:     str = ""
    resume_score:         int = 0
    github_score:         int = 0       # separate GitHub score
    final_score:          int = 0       # weighted combined score
    skill_match_reason:   str = ""
    candidate_name:       str = ""
    candidate_email:      str = ""
    github_username:      str = ""
    github_verification:  str = VerificationStatus.SKIPPED
    github_summary:       str = ""
    github_deep_analysis: str = ""      # detailed repo analysis
    response:             str = ""

# ══════════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT SCHEMAS
# ══════════════════════════════════════════════════════════════════
class CandidateInfo(BaseModel):
    name:            str = Field(description="Full name or Unknown")
    email:           str = Field(description="Email or NOT_FOUND")
    github_username: str = Field(description="GitHub username only e.g. 'johndoe' not full URL, or NOT_FOUND")

class ExperienceLevel(BaseModel):
    level: ExperienceLabel = Field(description="Entry-Level, Intermediate-Level or Senior-Level")

class JDTechCheck(BaseModel):
    is_technical: bool = Field(description="True if JD needs coding/ML/data skills. False if open to all backgrounds.")

class ProjectClaims(BaseModel):
    projects: list[str] = Field(description="Project names/titles from resume Projects section. Empty list if none.")

class DeepRepoAnalysis(BaseModel):
    project_name:        str   = Field(description="Name of the project from the resume")
    is_confirmed:        bool  = Field(description="True if this project genuinely exists and matches in GitHub")
    confidence_score:    int   = Field(description="0 to 10 — how strongly the GitHub repo confirms the resume claim")
    matching_evidence:   str   = Field(description="What specific evidence in the repo confirms or denies the claim")

class GitHubVerificationResult(BaseModel):
    status:          VerificationStatus = Field(description="verified/partial/unverified/skipped/not_applicable")
    summary:         str                = Field(description="2-3 sentence summary of what was found vs claimed")
    resume_score:    int                = Field(description="Resume quality score 0-10 based on resume content alone")
    github_score:    int                = Field(description="GitHub credibility score 0-10 based on actual repos found")
    final_score:     int                = Field(description="Combined weighted score 0-10. Formula: (resume*0.6 + github*0.4)")
    score_reason:    str                = Field(description="One sentence explaining the final score")

# ══════════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ══════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════
# DEEP GITHUB SCANNING
# — fetches repo tree + README, not just name/description
# ══════════════════════════════════════════════════════════════════
def fetch_repo_deep_content(username: str, repo_name: str) -> str:
    """Fetch README + file tree of a specific repo for deep analysis."""
    base    = f"https://api.github.com/repos/{username}/{repo_name}"
    content = f"\n--- Repo: {repo_name} ---\n"

    try:
        # Get repo metadata
        meta = requests.get(base, timeout=10).json()
        content += f"Description : {meta.get('description', 'None')}\n"
        content += f"Language    : {meta.get('language', 'None')}\n"
        content += f"Topics      : {meta.get('topics', [])}\n"
        content += f"Stars       : {meta.get('stargazers_count', 0)}\n"

        # Get README content
        readme_resp = requests.get(f"{base}/readme", timeout=10)
        if readme_resp.status_code == 200:
            import base64
            readme_data    = readme_resp.json()
            readme_content = base64.b64decode(readme_data["content"]).decode("utf-8", errors="ignore")
            # Limit README to first 1500 chars to avoid token overflow
            content += f"\nREADME (first 1500 chars):\n{readme_content[:1500]}\n"

        # Get file tree (top level)
        tree_resp = requests.get(f"{base}/git/trees/HEAD?recursive=0", timeout=10)
        if tree_resp.status_code == 200:
            tree  = tree_resp.json().get("tree", [])
            files = [item["path"] for item in tree[:30]]  # top 30 files
            content += f"\nFile Structure:\n{chr(10).join(files)}\n"

    except Exception as e:
        content += f"Error fetching repo details: {e}\n"

    return content

def fetch_all_github_data(username: str) -> tuple[str, list[str]]:
    """
    Returns:
        summary     : text overview of all repos
        repo_names  : list of repo names for deep scanning
    """
    if not username or username in ("NOT_FOUND", ""):
        return "", []

    # Clean username — strip URL if LLM returned full URL
    username = re.sub(r"https?://(www\.)?github\.com/", "", username).strip("/")

    try:
        url = f"https://api.github.com/users/{username}/repos?sort=updated&per_page=20"
        response = requests.get(url, timeout=10)

        if response.status_code == 403:
            print("  GitHub API rate limit hit.")
            return "RATE_LIMITED", []
        if response.status_code == 404:
            print(f"  GitHub user @{username} not found.")
            return "", []

        repos = response.json()
        if not isinstance(repos, list):
            return "", []

        repo_names = [r["name"] for r in repos]
        summary    = f"GitHub profile @{username} has {len(repos)} public repos:\n"
        for r in repos:
            summary += f"- {r['name']} [{r.get('language','?')}]: {r.get('description','')}\n"

        return summary, repo_names

    except Exception as e:
        print(f"  GitHub fetch error: {e}")
        return "", []

# ══════════════════════════════════════════════════════════════════
# NODE 1 — Extract Candidate Info
# ══════════════════════════════════════════════════════════════════
def candidate_info(state: Hire) -> Hire:
    print("  Extracting name, email, GitHub from resume...")
    try:
        structured_llm = llm.with_structured_output(CandidateInfo)
        result = structured_llm.invoke(
            f"Extract the candidate's full name, email and GitHub username.\n"
            f"For github_username return ONLY the username (e.g. 'johndoe' not 'github.com/johndoe').\n\n"
            f"Resume:\n{state.application}"
        )
        name, email, github = result.name.strip(), result.email.strip(), result.github_username.strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        name, email, github = "Unknown", "NOT_FOUND", "NOT_FOUND"

    print(f"  Name   : {name}")
    print(f"  Email  : {email}")
    print(f"  GitHub : {github}")
    return {"candidate_name": name, "candidate_email": email, "github_username": github}

# ══════════════════════════════════════════════════════════════════
# NODE 2 — Check if JD is technical
# ══════════════════════════════════════════════════════════════════
def check_jd_technical(state: Hire) -> Hire:
    print("  Checking JD type...")
    try:
        structured_llm = llm.with_structured_output(JDTechCheck)
        result = structured_llm.invoke(
            f"Is this JD requiring technical skills or open to all backgrounds?\n\nJD:\n{state.job_description}"
        )
        flag = "technical" if result.is_technical else "non-technical"
    except Exception as e:
        print(f"  LLM error: {e}")
        flag = "technical"

    print(f"  JD Type : {flag}")
    return {"github_verification": flag}

# ══════════════════════════════════════════════════════════════════
# NODE 3 — Deep GitHub Verification
# ══════════════════════════════════════════════════════════════════
def verify_github(state: Hire) -> Hire:
    print("  Running deep GitHub verification...")

    # Step 1: Get all repos
    github_summary, repo_names = fetch_all_github_data(state.github_username)

    if github_summary == "RATE_LIMITED":
        return {
            "github_verification": VerificationStatus.SKIPPED,
            "github_summary":      "GitHub API rate limit reached. Skipped.",
            "github_deep_analysis": ""
        }

    if not github_summary:
        return {
            "github_verification": VerificationStatus.SKIPPED,
            "github_summary":      "GitHub profile not found or empty.",
            "github_deep_analysis": ""
        }

    # Step 2: Extract claimed projects from resume
    try:
        claims_llm = llm.with_structured_output(ProjectClaims)
        claims     = claims_llm.invoke(
            f"Extract all project names from the Projects section of this resume.\n\n{state.application}"
        )
    except Exception as e:
        print(f"  Claims extraction error: {e}")
        claims = ProjectClaims(projects=[])

    if not claims.projects:
        return {
            "github_verification": VerificationStatus.SKIPPED,
            "github_summary":      "No projects section found in resume.",
            "github_deep_analysis": ""
        }

    print(f"  Claimed projects : {claims.projects}")

    # Step 3: Deep scan each repo
    # Only scan repos that could plausibly match claimed projects (max 5 to save tokens)
    deep_content = github_summary + "\n\n=== DEEP REPO ANALYSIS ===\n"

    repos_to_scan = repo_names[:5]  # scan top 5 most recent repos
    for repo_name in repos_to_scan:
        deep_content += fetch_repo_deep_content(state.github_username, repo_name)

    # Step 4: LLM does the actual deep comparison
    try:
        verify_llm = llm.with_structured_output(GitHubVerificationResult)
        result     = verify_llm.invoke(
            f"You are a technical recruiter doing a deep verification.\n\n"
            f"The candidate claims to have built these projects:\n{claims.projects}\n\n"
            f"Here is their complete GitHub data including README contents and file structures:\n"
            f"{deep_content}\n\n"
            f"Also use this Job Description for context:\n{state.job_description}\n\n"
            f"Resume:\n{state.application}\n\n"
            f"Instructions:\n"
            f"1. Do NOT just match project names — read the README and file structure to confirm the tech stack and functionality matches what the candidate claims\n"
            f"2. A project with different repo name but same tech/functionality = still counts as verified\n"
            f"3. Give a resume_score (0-10) based on resume quality alone\n"
            f"4. Give a github_score (0-10) based on what you actually found in GitHub\n"
            f"5. final_score = (resume_score * 0.6) + (github_score * 0.4) — round to nearest int\n"
            f"6. Be strict — inflated claims with no GitHub evidence must lower the score"
        )

        print(f"  Verification     : {result.status}")
        print(f"  Resume Score     : {result.resume_score}/10")
        print(f"  GitHub Score     : {result.github_score}/10")
        print(f"  Final Score      : {result.final_score}/10")
        print(f"  Summary          : {result.summary}")

        return {
            "github_verification":  result.status,
            "github_summary":       result.summary,
            "github_deep_analysis": deep_content[:500],  # store summary for display
            "resume_score":         result.resume_score,
            "github_score":         result.github_score,
            "final_score":          result.final_score,
            "skill_match_reason":   result.score_reason
        }

    except Exception as e:
        print(f"  Verification LLM error: {e}")
        return {
            "github_verification": VerificationStatus.SKIPPED,
            "github_summary":      f"Verification failed: {e}",
            "github_deep_analysis": ""
        }

# ══════════════════════════════════════════════════════════════════
# NODE 4 — Experience Level
# ══════════════════════════════════════════════════════════════════
def candidate_experience(state: Hire) -> Hire:
    print("  Evaluating experience level...")
    try:
        structured_llm = llm.with_structured_output(ExperienceLevel)
        result         = structured_llm.invoke(
            f"Categorize candidate experience level.\n\nJD:\n{state.job_description}\n\nResume:\n{state.application}"
        )
        level = result.level.value
    except Exception as e:
        print(f"  LLM error: {e}")
        level = ExperienceLabel.ENTRY.value

    print(f"  Experience Level : {level}")
    return {"experience_level": level}

# ══════════════════════════════════════════════════════════════════
# NODE 5 — Score Resume (only used if GitHub was skipped/non-tech)
# ══════════════════════════════════════════════════════════════════
def score_resume_only(state: Hire) -> Hire:
    """Used when GitHub verification is skipped — scores resume alone."""
    print("  Scoring resume (no GitHub data)...")

    v = state.github_verification
    if v == VerificationStatus.NOT_APPLICABLE:
        github_note = "Non-technical role. Score based on writing, reasoning, and domain fit."
    else:
        github_note = "No GitHub profile available. Score based on resume content only."

    try:
        structured_llm = llm.with_structured_output(GitHubVerificationResult)
        result         = structured_llm.invoke(
            f"You are a strict recruiter. Score this resume.\n\n"
            f"Rules:\n"
            f"- No domain match → 0 to 3\n"
            f"- Partial match   → 4 to 6\n"
            f"- Direct match    → 7 to 10\n"
            f"- Note: {github_note}\n\n"
            f"Set github_score = 0 (no GitHub data). "
            f"final_score = resume_score (since no GitHub).\n\n"
            f"JD:\n{state.job_description}\n\nResume:\n{state.application}"
        )
        print(f"  Resume Score : {result.resume_score}/10")
        print(f"  Final Score  : {result.final_score}/10")
        print(f"  Reason       : {result.score_reason}")
        return {
            "resume_score":       result.resume_score,
            "github_score":       0,
            "final_score":        result.resume_score,  # no GitHub = resume is final
            "skill_match_reason": result.score_reason
        }
    except Exception as e:
        print(f"  Scoring error: {e}")
        return {"resume_score": 0, "github_score": 0, "final_score": 0, "skill_match_reason": str(e)}

# ══════════════════════════════════════════════════════════════════
# NODE 6 — Send Assessment Email
# ══════════════════════════════════════════════════════════════════
def invitation_for_Assessment(state: Hire) -> Hire:
    print(f"  Sending assessment invitation to {state.candidate_email}...")
    sender   = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_APP_PASSWORD")

    if not sender or not password:
        print("  Email credentials missing in .env")
        return {"response": f"Shortlisted (score {state.final_score}/10) — email credentials missing."}

    subject = "Invitation for Assessment Round"
    body    = f"""
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
    msg            = MIMEMultipart()
    msg["From"]    = sender
    msg["To"]      = state.candidate_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, state.candidate_email, msg.as_string())
        print(f"  Mail sent to {state.candidate_name} at {state.candidate_email}")
        return {"response": f"Assessment invitation sent to {state.candidate_email}"}
    except Exception as e:
        print(f"  Email failed: {e}")
        return {"response": f"Shortlisted but email failed: {e}"}

# ══════════════════════════════════════════════════════════════════
# REJECTION NODES
# ══════════════════════════════════════════════════════════════════
def reject_low_score(state: Hire) -> Hire:
    print(f"  Rejecting {state.candidate_name} — final score {state.final_score}/10.")
    return {"response": f"Rejected: Final score {state.final_score}/10 is below threshold of 7."}

def reject_no_email(state: Hire) -> Hire:
    print("  Rejecting — no email found.")
    return {"response": "Rejected: No email address found in resume."}

# ══════════════════════════════════════════════════════════════════
# ROUTERS
# ══════════════════════════════════════════════════════════════════
def route_email_check(state: Hire) -> str:
    if state.candidate_email in ("NOT_FOUND", ""):
        return "reject_no_email"
    return "check_jd_technical"

def route_github_check(state: Hire) -> str:
    # Non-tech role
    if state.github_verification == "non-technical":
        return "score_resume_only"
    # No GitHub URL in resume
    if state.github_username in ("NOT_FOUND", ""):
        return "score_resume_only"
    # Tech role + GitHub present → deep verify
    return "verify_github"

def route_score(state: Hire) -> str:
    if state.final_score >= 7:
        return "Assessment"
    return "reject_low_score"

# ══════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════════
builder = StateGraph(Hire)

builder.add_node("candidate_info",       candidate_info)
builder.add_node("check_jd_technical",   check_jd_technical)
builder.add_node("verify_github",        verify_github)        # deep scan + scores
builder.add_node("score_resume_only",    score_resume_only)    # resume only scoring
builder.add_node("candidate_experience", candidate_experience)
builder.add_node("Assessment",           invitation_for_Assessment)
builder.add_node("reject_low_score",     reject_low_score)
builder.add_node("reject_no_email",      reject_no_email)

builder.add_edge(START,                   "candidate_info")
builder.add_conditional_edges(            "candidate_info",     route_email_check)
builder.add_conditional_edges(            "check_jd_technical", route_github_check)
builder.add_edge(                         "verify_github",       "candidate_experience")
builder.add_edge(                         "score_resume_only",   "candidate_experience")
builder.add_edge(                         "candidate_experience","resume_score" if False else "Assessment" if False else "reject_low_score" if False else END)
# ↑ replace the above with proper conditional routing below:
builder.add_conditional_edges(            "candidate_experience", route_score)
builder.add_edge(                         "Assessment",           END)
builder.add_edge(                         "reject_low_score",     END)
builder.add_edge(                         "reject_no_email",      END)

app = builder.compile()

graph_image = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)
print("Graph saved as graph.png!")

# ══════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════
def run_candidate_screening(pdf_path: str, job_description: str):
    print(f"\nProcessing: {os.path.basename(pdf_path)}")
    print("=" * 55)

    if not pdf_path.lower().endswith(".pdf"):
        print("  Not a PDF. Skipping.")
        return

    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text or pdf_text.isspace():
        print("  Blank or unreadable PDF. Skipping.")
        return

    try:
        results = app.invoke({
            "file_path":       pdf_path,
            "application":     pdf_text,
            "job_description": job_description
        })
        print(f"\n  Final Results:")
        print(f"  Name                 : {results['candidate_name']}")
        print(f"  Email                : {results['candidate_email']}")
        print(f"  GitHub               : {results['github_username']}")
        print(f"  GitHub Verification  : {results['github_verification']}")
        print(f"  GitHub Summary       : {results['github_summary']}")
        print(f"  Experience Level     : {results['experience_level']}")
        print(f"  Resume Score         : {results['resume_score']}/10")
        print(f"  GitHub Score         : {results['github_score']}/10")
        print(f"  Final Score          : {results['final_score']}/10")
        print(f"  Score Reason         : {results.get('skill_match_reason','N/A')}")
        print(f"  Response             : {results.get('response','No response generated')}")
    except Exception as e:
        print(f"  Unexpected error: {e}. Skipping.")

def main():
    job_description = input("Enter the Job Description:\n")
    resume_folder   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resumeSamples")

    if not os.path.exists(resume_folder):
        os.makedirs(resume_folder)
        print(f"Created folder: {resume_folder}. Add PDFs and run again.")
        return

    resumes = [
        os.path.join(resume_folder, f)
        for f in os.listdir(resume_folder)
        if f.lower().endswith(".pdf")
    ]

    if not resumes:
        print("No PDFs found.")
        return

    print(f"\nFound {len(resumes)} resume(s).")
    for resume in resumes:
        run_candidate_screening(resume, job_description)
        print("-" * 55)

main()