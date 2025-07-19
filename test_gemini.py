import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-pro")

response = model.generate_content("What stage of baldness is described as a receding hairline at the temples?")
print(response.text)

