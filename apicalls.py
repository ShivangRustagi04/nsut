import os
import google.generativeai as genai
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ Gemini API key not found! Please set GEMINI_API_KEY in your .env file.")
else:
    genai.configure(api_key=API_KEY)

# -------------------------
# Gemini call function
# -------------------------
def call_gemini(prompt: str, max_output_tokens: int = 4096):
    """
    Call Google Gemini API with the given prompt
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=0.7
            )
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error calling Gemini: {str(e)}"

# -------------------------
# Web search function using DuckDuckGo
# -------------------------
def search_web(query, max_results=5):
    """
    Search the web using DuckDuckGo
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        return f"⚠️ Error searching web: {str(e)}"

# -------------------------
# Policy recommendation function with web search
# -------------------------
def recommend_policy(age, income_range, occupation, family_members, existing_insurance, health_conditions, language="English"):
    """
    Get policy recommendations based on user profile
    """
    # Search for popular policies based on user profile
    search_query = f"best insurance policies for {occupation} with income {income_range} India 2025"
    search_results = search_web(search_query)
    
    # Extract relevant information from search results
    search_context = ""
    if isinstance(search_results, list):
        for i, result in enumerate(search_results[:3]):
            search_context += f"Result {i+1}: {result.get('title', '')} - {result.get('body', '')}\n"
    else:
        search_context = "No web search results available."
    
    prompt = f"""
    You are an insurance expert recommending the best insurance policies for users in India.
    
    User Details:
    - Age: {age}
    - Annual Income Range: {income_range}
    - Occupation: {occupation}
    - Family Members: {family_members}
    - Existing Insurance: {existing_insurance}
    - Health Conditions: {health_conditions}
    
    Web Search Context about suitable policies:
    {search_context}
    
    Based on this information, recommend the most suitable insurance policies for this user.
    Consider life insurance, health insurance, and any other relevant insurance types.
    
    Provide your response in {language} language.
    Structure your response with:
    1. Policy recommendations (3-5 policies with company names)
    2. Brief explanation for each recommendation
    3. Estimated premium ranges
    4. Key benefits of each policy
    5. Suitability score for each policy (0-100%)
    
    Keep the response clear, concise, and helpful.
    """
    
    return call_gemini(prompt)

# -------------------------
# Policy analysis function with web search
# -------------------------
def analyze_policy(policy_name, user_details, language="English"):
    """
    Analyze a specific insurance policy
    """
    # Search for policy information
    search_query = f"{policy_name} insurance policy India benefits features 2025"
    search_results = search_web(search_query)
    
    # Extract relevant information from search results
    search_context = ""
    if isinstance(search_results, list):
        for i, result in enumerate(search_results[:3]):
            search_context += f"Result {i+1}: {result.get('title', '')} - {result.get('body', '')}\n"
    else:
        search_context = "No web search results available."
    
    prompt = f"""
    Analyze the insurance policy: {policy_name}
    
    User Details:
    - Age: {user_details.get('age', 'Not provided')}
    - Annual Income Range: {user_details.get('income_range', 'Not provided')}
    - Occupation: {user_details.get('occupation', 'Not provided')}
    - Family Members: {user_details.get('family_members', 'Not provided')}
    - Existing Insurance: {user_details.get('existing_insurance', 'Not provided')}
    - Health Conditions: {user_details.get('health_conditions', 'Not provided')}
    
    Web Search Context:
    {search_context}
    
    Provide a comprehensive analysis of this policy including:
    1. Policy overview and key features
    2. Benefits for this specific user
    3. Potential drawbacks or limitations
    4. Premium estimates (provide specific numbers if possible)
    5. Coverage details
    6. Comparison with similar policies
    7. Final recommendation (should this user consider this policy?)
    
    Provide your response in {language} language.
    Be objective and evidence-based in your analysis.
    """
    
    return call_gemini(prompt, max_output_tokens=4096)

# -------------------------
# Chat function with auto language detection
# -------------------------
def chat_with_user(message, chat_history, language="English"):
    """
    Chat with the insurance assistant
    """
    prompt = f"""
    You are PRAYAAS, a friendly insurance assistant helping users in India.
    Your role is to explain insurance concepts, answer questions, and provide guidance.
    
    Current conversation context:
    {chat_history}
    
    User's message: {message}
    
    Respond helpfully and accurately in {language} language.
    Keep your response concise but informative.
    If the user asks about a specific policy, offer to analyze it for them.
    """
    
    return call_gemini(prompt)