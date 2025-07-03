


import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("AIzaSyDpMZL6L4nCMCy4WoFye3EpxXHeCJnL7bk"))

#model = genai.GenerativeModel("gemini-1.5-pro")
#response = model.generate_content("Tell me a joke")

print(list(genai.list_models()))

#print(response.text)
