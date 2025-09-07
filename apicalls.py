

import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from duckduckgo_search import DDGS
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ Gemini API key not found! Please set GEMINI_API_KEY in your .env file.")
else:
    client=genai.Client(api_key=API_KEY)


# Define the grounding tool
grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

# Configure generation settings

# -------------------------
# Gemini call function
# -------------------------
def call_gemini(prompt: str, max_output_tokens: int = 10024):
    try:
        config=types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.2,
            tools=[grounding_tool],
            
        )
        res=client.models.generate_content(
            contents=prompt,
            model="gemini-2.5-flash",
            config=config
        )
        return res.text
    except Exception as e:
        return f"⚠️ Error calling Gemini: {str(e)}"

# -------------------------
# Web search function using DuckDuckGo
# -------------------------
def search_web(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        return f"⚠️ Error searching web: {str(e)}"