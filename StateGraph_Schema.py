from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile")


from typing_extensions import TypedDict
class Hire(TypedDict):
  application:str
  experience_level:str
  resume_score :int
  response:str

from langgraph.graph import StateGraph, START, END
workflow = StateGraph(Hire)

from langchain_core.prompts import ChatPromptTemplate
def categorize_experience(state: Hire) -> Hire:
  print("\nCategorizing the experience level of candidate : ")
  prompt = ChatPromptTemplate.from_template(
      "Based on the following job application, categorize the candidate. "
      "Reply with ONLY one of these exact words: Entry-level, Mid-level, Senior-level. "
      "No explanation, no extra text, just the label. "
      "Application : {application}"
  )
  chain = prompt | llm
  raw = chain.invoke({"application": state["application"]}).content.strip()

  # Extract the label even if AI adds extra text
  if "Senior" in raw:
      experience_level = "Senior-level"
  elif "Mid" in raw:
      experience_level = "Mid-level"
  else:
      experience_level = "Entry-level"

  print(f"Experience Level : {experience_level}")
  return {"experience_level" : experience_level}
# def categorize_experience(state:Hire) -> Hire:
#   print("\nCategorizing the experience level of candidate : ")
#   prompt = ChatPromptTemplate.from_template(
#       "Based on the following job application, categorize the candidate as 'Entry-level', 'Mid-level' or 'Senior-level'"
#       "Application : {application}"
#   )
#   chain = prompt | llm
#   experience_level = chain.invoke({"application": state["application"]}).content
#   print(f"Experience Level : {experience_level}")
#   return {"experience_level" : experience_level}

# def resume_score(state: Hire) -> Hire:
#   print("\nScoring the candidates based on their resume : ")
#   prompt = ChatPromptTemplate.from_template(
#     "Based on the job application score the candidate on clarity and specificity. "
#     "You MUST reply with ONLY a single integer number from 1 to 10. "
#     "No explanation, no text, just the number. Example: 7"
#     " Application : {application}"
#   )
#   chain = prompt | llm
#   raw_score = chain.invoke({"application": state["application"]}).content.strip()
  
#   # Extract just the number even if AI adds extra text
#   import re
#   match = re.search(r'\b([1-9]|10)\b', raw_score)
#   score = match.group(1) if match else "5"  # default to 5 if no number found
  
#   print(f"Resume Score : {score}")
#   return {"resume_score": score}
def resume_score(state: Hire) -> Hire:
  print("\nScoring the candidates based on their resume : ")
  prompt = ChatPromptTemplate.from_template(
    "Based on the job application score the candidates based on their clarity and specificity on a scale of 10 and just give me the a ssingle number as your response not any explanation further"
    "Application : {application}"
  )
  chain = prompt | llm
  resume_score = chain.invoke({"application": state["application"]}).content
  print(f"resume_score : {resume_score}")    
  return {"resume_score": resume_score}

def schedule_hr_interview(state: Hire) -> Hire:
  print("\nScheduling the interview")
  return {"response" : "Candidate has been shortlisted for an HR interview."}

def escalate_to_recruiter(state: Hire) -> Hire:
  print("Escalating to recruiter")
  return {"response" : "Candidate has senior-level experience but doesn't match job skills."}

def reject_application(state: Hire) -> Hire:
  print("Sending rejecting email")
  return {"response" : "Candidate doesn't meet JD and has been rejected."}

def route_app(state: Hire) -> str:
    score = int(state["resume_score"])
    experience = state["experience_level"]

    if score >= 7:
        return "schedule_hr_interview"
    elif score == 6 and "Senior" in experience:
        return "escalate_to_recruiter"
    else:
        return "reject_application"
  
workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("resume_score", resume_score)
workflow.add_node("schedule_hr_interview", schedule_hr_interview)
workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
workflow.add_node("reject_application", reject_application)

workflow.add_edge("categorize_experience", "resume_score")
workflow.add_conditional_edges("resume_score", route_app)

workflow.add_edge(START, "categorize_experience")
workflow.add_edge("escalate_to_recruiter", END)
workflow.add_edge("reject_application", END)
workflow.add_edge("schedule_hr_interview", END)

app = workflow.compile()

graph_image = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)
print("Graph saved as graph.png — open it in your project folder!")

def run_candidate_screening(application: str):
  results = app.invoke({"application" : application})
  print("\n\nComputed Results :")
#   print(f"Application: {application_text}")
  print(f"Experience Level: {results['experience_level']}")
  print(f"resume_score: {results['resume_score']}")
  print(f"Response: {results['response']}")

application_text = "I have 1 year of experience in Web Development and had deployed an e-commerce website and i have also learned the syntax of python"
results = run_candidate_screening(application_text)