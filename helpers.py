from apicalls import call_gemini, search_web
import plotly.graph_objects as go
import re
from typing import Any
# -------------------------
# Policy recommendation function
# -------------------------
def recommend_policy(age: int, income_range: str|None, occupation: str|None, family_members: int, existing_insurance: list, health_conditions: list, language="English"):
    prompt = f"""
    You are an insurance expert recommending the best insurance policies for users in India.
    
    User Details:
    - Age: {age}
    - Annual Income Range: {income_range}
    - Occupation: {occupation}
    - Family Members: {family_members}
    - Existing Insurance: {existing_insurance}
    - Health Conditions: {health_conditions}
    
    Based on this information, recommend the most suitable insurance policies for this user.
    Consider life insurance, health insurance, and any other relevant insurance types.
    you should provide 3-5 specific policy recommendations available in India as of 2025.
    DO NOT make up policy names; use real ones.
    DO NOT include meta starters like "Here are some recommendations for you". go straight to the recommendations.
    
    Provide your response in {language} language.
    Structure your response with:
    1. Policy recommendations (3-5 policies)
    2. Brief explanation for each recommendation
    3. Estimated premium ranges
    4. Key benefits of each policy
    
    Keep the response clear, concise, and helpful.
    """
    
    return call_gemini(prompt)

# -------------------------
# Policy analysis function with web search
# -------------------------
def analyze_policy(policy_name:str, user_details:dict[str, Any], language="English"):
    # Search for policy information
    search_query = f"{policy_name} insurance policy India benefits features 2024"
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
    4. Premium estimates
    5. Comparison with similar policies
    6. Final recommendation (should this user consider this policy?)
    
    Provide your response in {language} language.
    Be objective and evidence-based in your analysis.
    """
    
    return call_gemini(prompt, max_output_tokens=2048)

# -------------------------
# Create policy visualization
# -------------------------
def create_policy_visualization(policy_name:str, analysis_text:str):
    # Extract key metrics from analysis text using regex (simplified approach)
    premium_match = re.search(r'[₹$](\d+(?:,\d+)*(?:\.\d+)?)\s*(?:to|-|–)\s*[₹$]?(\d+(?:,\d+)*(?:\.\d+)?)', analysis_text)
    coverage_match = re.search(r'coverage.*?(\d+(?:,\d+)*\s*(?:lakhs|L|Lacs|Lakh|Lac|Cr|Crores))', analysis_text, re.IGNORECASE)
    
    # Create sample data for visualization (in a real app, this would come from actual policy data)
    categories = ['Premium Affordability', 'Coverage Adequacy', 'Benefits Match', 'Overall Value']
    scores = [75, 82, 68, 78]  # Default scores
    
    # Adjust based on premium information if available
    if premium_match:
        try:
            min_premium = float(premium_match.group(1).replace(',', ''))
            max_premium = float(premium_match.group(2).replace(',', ''))
            # Simple heuristic: lower premium = higher affordability score
            affordability = max(10, min(100, 100 - (min_premium / 50000 * 100)))
            scores[0] = int(affordability)
        except:
            pass
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name=policy_name
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title=f"{policy_name} Policy Analysis"
    )
    
    return fig

# -------------------------
# Chat function with auto language detection
# -------------------------
def chat_with_user(message:str, chat_history:list[dict[str, str]], language="English"):
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
