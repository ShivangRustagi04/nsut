import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from duckduckgo_search import DDGS
import re

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ Gemini API key not found! Please set GEMINI_API_KEY in your .env file.")
else:
    genai.configure(api_key=API_KEY)

# -------------------------
# Common Indian occupations
# -------------------------
INDIAN_OCCUPATIONS = [
    "Farmer/Agricultural Worker",
    "Daily Wage Laborer",
    "Shopkeeper/Retailer",
    "Driver (Taxi, Truck, Auto)",
    "Household Help/Domestic Worker",
    "Construction Worker",
    "Small Business Owner",
    "Government Employee",
    "Private Sector Employee",
    "Teacher/Educator",
    "Healthcare Worker",
    "IT Professional",
    "Engineer",
    "Student",
    "Homemaker",
    "Retired",
    "Unemployed",
    "Other"
]

# -------------------------
# Income ranges
# -------------------------
INCOME_RANGES = [
    "Up to ₹2.5 Lakh",
    "₹2.5 Lakh - ₹5 Lakh",
    "₹5 Lakh - ₹7.5 Lakh",
    "₹7.5 Lakh - ₹10 Lakh",
    "₹10 Lakh - ₹15 Lakh",
    "₹15 Lakh - ₹20 Lakh",
    "₹20 Lakh - ₹30 Lakh",
    "₹30 Lakh - ₹50 Lakh",
    "₹50 Lakh - ₹1 Crore",
    "₹1 Crore - ₹2 Crore",
    "₹2 Crore - ₹3 Crore",
    "Above ₹3 Crore"
]

# -------------------------
# Gemini call function
# -------------------------
def call_gemini(prompt: str, max_output_tokens: int = 1024):
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
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        return f"⚠️ Error searching web: {str(e)}"

# -------------------------
# Policy recommendation function
# -------------------------
def recommend_policy(age, income_range, occupation, family_members, existing_insurance, health_conditions, language="English"):
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
def analyze_policy(policy_name, user_details, language="English"):
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
def create_policy_visualization(policy_name, analysis_text):
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
def chat_with_user(message, chat_history, language="English"):
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

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PRAYAAS - Insurance Simplifier", layout="wide", initial_sidebar_state="expanded")

# Sidebar for user input
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/477/477103.png", width=100)
    st.title("PRAYAAS")
    st.markdown("**Simplifying Insurance for Everyone**")
    
    st.subheader("🔹 Your Profile")
    age = st.slider("Age", 18, 80, 30)
    income_range = st.selectbox("Annual Income Range", INCOME_RANGES)
    occupation = st.selectbox("Occupation", INDIAN_OCCUPATIONS)
    family_members = st.slider("Family Members", 1, 10, 4)
    existing_insurance = st.multiselect("Existing Insurance", [
        "Term Life", "Health Insurance", "Car Insurance", 
        "Home Insurance", "Investment Plans", "None"
    ])
    health_conditions = st.multiselect("Health Conditions", [
        "None", "Diabetes", "Hypertension", "Heart Condition", 
        "Respiratory Issues", "Other Chronic Condition"
    ])
    
    language = st.selectbox("Preferred Language", [
        "Hindi", "Gujarati", "Tamil", "Telugu", "Bengali", 
        "Marathi", "Kannada", "English"
    ])

# Main content area
st.title("🤝 PRAYAAS: Your Insurance Companion")
st.markdown("Helping you understand and choose the right insurance policies in your preferred language.")

# Tab interface
tab1, tab2, tab3 = st.tabs(["Policy Recommendation", "Policy Analysis", "Chat Assistant"])

with tab1:
    st.header("📋 Personalized Policy Recommendations")
    
    if st.button("Get Policy Recommendations", type="primary"):
        with st.spinner("Analyzing your profile and generating recommendations..."):
            recommendation = recommend_policy(
                age, income_range, occupation, family_members, 
                existing_insurance, health_conditions, language
            )
            
            st.success("Here are insurance policies tailored for you:")
            st.markdown(recommendation)
            
            # Sample visualization
            st.subheader("📊 Recommended Policy Comparison")
            
            # Create sample data for demonstration
            policies = ['Term Life Plus', 'Health Shield', 'Family Care', 'Senior Secure']
            premiums = [12000, 18000, 22000, 15000]
            coverage = [5000000, 1000000, 3000000, 2000000]
            
            fig = go.Figure(data=[
                go.Bar(name='Annual Premium (₹)', x=policies, y=premiums, yaxis='y', offsetgroup=1),
                go.Bar(name='Coverage Amount (₹)', x=policies, y=coverage, yaxis='y2', offsetgroup=2)
            ])
            
            fig.update_layout(
                title='Recommended Policies Comparison',
                xaxis_title='Policy Name',
                yaxis=dict(title='Premium Amount (₹)', titlefont=dict(color='blue')),
                yaxis2=dict(title='Coverage Amount (₹)', titlefont=dict(color='red'), overlaying='y', side='right'),
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🔍 Policy Analysis")
    
    policy_name = st.text_input("Enter policy name to analyze")
    
    if st.button("Analyze Policy", type="primary") and policy_name:
        user_details = {
            'age': age,
            'income_range': income_range,
            'occupation': occupation,
            'family_members': family_members,
            'existing_insurance': existing_insurance,
            'health_conditions': health_conditions
        }
        
        with st.spinner(f"Analyzing {policy_name} and searching for current information..."):
            analysis = analyze_policy(policy_name, user_details, language)
            
            st.success(f"Analysis of {policy_name}:")
            st.markdown(analysis)
            
            # Create visualization
            st.subheader("📈 Policy Visual Analysis")
            viz_fig = create_policy_visualization(policy_name, analysis)
            st.plotly_chart(viz_fig, use_container_width=True)
            
            # Add key metrics card
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recommended for you", "Yes", delta="Good match", delta_color="normal")
            with col2:
                st.metric("Affordability Score", "78/100", delta="Moderate", delta_color="off")
            with col3:
                st.metric("Coverage Adequacy", "82/100", delta="Strong", delta_color="normal")

with tab3:
    st.header("💬 Chat with PRAYAAS")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("Ask me anything about insurance..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("Thinking..."):
            response = chat_with_user(prompt, st.session_state.messages[-5:], language)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>ℹ️ Disclaimer: This is an AI-powered insurance assistant. For official policy details and purchases, please consult with licensed insurance providers.</p>
</div>
""", unsafe_allow_html=True)