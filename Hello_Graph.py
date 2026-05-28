from dotenv import load_dotenv
from google import genai
import os

# Load .env file
load_dotenv()

# Read API key
API_KEY = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=API_KEY)


def ask_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


# Interactive terminal chat
while True:
    query = input("\nEnter your question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    try:
        answer = ask_gemini(query)

        print("\nGemini Response:\n")
        print(answer)

    except Exception as e:
        print("\nERROR:")
        print(e)

