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
import json
import numpy as np
from datetime import datetime

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
# Popular insurance policies in India (for search)
# -------------------------
POPULAR_POLICIES = [
    "LIC Jeevan Anand",
    "HDFC Life Click 2 Protect",
    "ICICI Pru iProtect Smart",
    "SBI Life eShield",
    "Max Life Online Term Plan",
    "Bajaj Allianz Life Goal Assure",
    "Tata AIA Life Sampoorna Raksha",
    "Kotak e-Term",
    "Aditya Birla Sun Life Shield",
    "PNB MetLife Mera Term Plan"
]

# -------------------------
# Gemini call function
# -------------------------
def call_gemini(prompt: str, max_output_tokens: int = 1024):
    try:
        model = genai.GenerativeModel("gemini-pro")
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
# Extract structured data from analysis text
# -------------------------
def extract_policy_data(analysis_text):
    data = {
        'premium_range': '₹15,000 to ₹20,000',
        'coverage_amount': '10 Lakhs',
        'policy_term': '20 years',
        'key_features': [],
        'suitability_score': 70,
        'claim_settlement_ratio': 85,
        'flexibility_score': 70,
        'policy_name': 'Current Policy',
        'avg_premium': 18000,
        'coverage_value': 1000000
    }
    
    # Extract premium information
    premium_match = re.search(r'[₹$](\d+(?:,\d+)*(?:\.\d+)?)\s*(?:to|-|–)\s*[₹$]?(\d+(?:,\d+)*(?:\.\d+)?)', analysis_text)
    if premium_match:
        data['premium_range'] = f"₹{premium_match.group(1)} to ₹{premium_match.group(2)}"
        data['avg_premium'] = (float(premium_match.group(1).replace(',', '')) + float(premium_match.group(2).replace(',', ''))) / 2
    
    # Extract coverage amount
    coverage_match = re.search(r'coverage.*?(\d+(?:,\d+)*\s*(?:lakhs|L|Lacs|Lakh|Lac|Cr|Crores|million))', analysis_text, re.IGNORECASE)
    if coverage_match:
        data['coverage_amount'] = coverage_match.group(1)
        # Extract numeric value for calculations
        num_match = re.search(r'(\d+(?:,\d+)*)', coverage_match.group(1))
        if num_match:
            data['coverage_value'] = float(num_match.group(1).replace(',', '')) * 100000
    
    # Extract policy term
    term_match = re.search(r'term.*?(\d+)\s*(?:years|yrs|year)', analysis_text, re.IGNORECASE)
    if term_match:
        data['policy_term'] = f"{term_match.group(1)} years"
    
    # Extract key features using a more sophisticated approach
    lines = analysis_text.split('\n')
    feature_keywords = ['covers', 'provides', 'includes', 'benefit', 'feature', 'offers', 'protection']
    
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in feature_keywords):
            if len(line) > 15 and len(line) < 150:  # Reasonable length for a feature
                data['key_features'].append(line.strip())
    
    # Limit to 5 key features
    data['key_features'] = data['key_features'][:5]
    
    # Extract policy name if possible
    name_match = re.search(r'([A-Za-z0-9\s]+)(?:policy|plan|insurance)', analysis_text, re.IGNORECASE)
    if name_match:
        data['policy_name'] = name_match.group(1).strip()
    
    # Calculate suitability score based on text sentiment
    positive_words = ['good', 'excellent', 'great', 'beneficial', 'recommended', 'suitable', 'ideal', 'comprehensive']
    negative_words = ['expensive', 'limited', 'restrictive', 'not suitable', 'not recommended', 'limitation', 'drawback']
    
    positive_count = sum(1 for word in positive_words if word in analysis_text.lower())
    negative_count = sum(1 for word in negative_words if word in analysis_text.lower())
    
    if positive_count + negative_count > 0:
        data['suitability_score'] = min(100, max(30, 50 + (positive_count - negative_count) * 10))
    
    return data

# -------------------------
# Score calculation helper functions
# -------------------------
def calculate_affordability_score(policy_data, user_details):
    income_text = user_details.get('income_range', '₹5 Lakh - ₹7.5 Lakh')
    
    # Extract numeric value from income range
    income_match = re.search(r'₹(\d+(?:,\d+)*).*?₹(\d+(?:,\d+)*)', income_text)
    if income_match:
        min_income = float(income_match.group(1).replace(',', '')) * 1000
        avg_income = min_income * 1.5  # Approximate average
    else:
        avg_income = 500000  # Default
    
    # Calculate affordability score (lower premium = better)
    premium_match = re.search(r'₹(\d+(?:,\d+)*).*?₹(\d+(?:,\d+)*)', policy_data.get('premium_range', '₹15000 to ₹20000'))
    if premium_match:
        min_premium = float(premium_match.group(1).replace(',', ''))
        affordability = max(10, min(100, 100 - (min_premium / (avg_income/10)) * 100))
    else:
        affordability = 75  # Default
        
    policy_data['affordability_score'] = affordability
    return affordability

def calculate_coverage_score(policy_data, user_details):
    family_members = user_details.get('family_members', 4)
    income_text = user_details.get('income_range', '₹5 Lakh - ₹7.5 Lakh')
    
    # Extract numeric value from income range
    income_match = re.search(r'₹(\d+(?:,\d+)*).*?₹(\d+(?:,\d+)*)', income_text)
    if income_match:
        min_income = float(income_match.group(1).replace(',', '')) * 1000
        avg_income = min_income * 1.5  # Approximate average
    else:
        avg_income = 500000  # Default
    
    # Calculate coverage adequacy based on income and family size
    coverage_match = re.search(r'(\d+(?:,\d+)*)', str(policy_data.get('coverage_amount', '10 Lakhs')))
    if coverage_match:
        coverage = float(coverage_match.group(1).replace(',', '')) * 100000  # Convert lakhs to rupees
        # Heuristic: Good coverage is 10-15x annual income for a family
        adequate_coverage = avg_income * 10 * (1 + (family_members-1)*0.2)
        coverage_adequacy = min(100, max(20, (coverage / adequate_coverage) * 100))
    else:
        coverage_adequacy = 70  # Default
        
    policy_data['coverage_score'] = coverage_adequacy
    return coverage_adequacy

# -------------------------
# Create multiple policy visualizations
# -------------------------
def create_policy_visualizations(policy_name, analysis_text, user_details):
    # Extract structured data from analysis
    policy_data = extract_policy_data(analysis_text)
    
    # 1. Radar Chart
    radar_fig = create_radar_chart(policy_name, policy_data, user_details)
    
    # 2. Metrics Bar Chart
    metrics_fig = create_metrics_chart(policy_data)
    
    # 3. Premium vs Coverage Scatter Plot
    scatter_fig = create_premium_coverage_chart(policy_data, user_details)
    
    # 4. Feature Importance Chart
    feature_fig = create_feature_importance_chart(policy_data)
    
    # 5. Timeline of Benefits
    timeline_fig = create_benefit_timeline_chart(policy_data, user_details)
    
    # 6. Comparison with Ideal Policy
    comparison_fig = create_policy_comparison_chart(policy_data, user_details)
    
    # 7. Premium Breakdown Pie Chart
    premium_fig = create_premium_breakdown_chart(policy_data)
    
    return {
        'radar': radar_fig,
        'metrics': metrics_fig,
        'scatter': scatter_fig,
        'features': feature_fig,
        'timeline': timeline_fig,
        'comparison': comparison_fig,
        'premium_breakdown': premium_fig
    }, policy_data

# -------------------------
# Radar Chart
# -------------------------
def create_radar_chart(policy_name, policy_data, user_details):
    categories = ['Premium Affordability', 'Coverage Adequacy', 'Benefits Match', 'Claim Settlement', 'Flexibility', 'Overall Value']
    
    # Calculate scores based on user profile and policy data
    affordability = calculate_affordability_score(policy_data, user_details)
    coverage_adequacy = calculate_coverage_score(policy_data, user_details)
    benefits_match = min(100, 60 + len(policy_data['key_features']) * 8)
    claim_settlement = policy_data.get('claim_settlement_ratio', 85)  # Default if not found
    flexibility = policy_data.get('flexibility_score', 70)  # Default if not found
    
    # Overall value (weighted average)
    overall_value = (affordability * 0.25 + coverage_adequacy * 0.25 + 
                    benefits_match * 0.2 + claim_settlement * 0.2 + flexibility * 0.1)
    
    scores = [affordability, coverage_adequacy, benefits_match, claim_settlement, flexibility, overall_value]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name=policy_name,
        line=dict(color='blue', width=2),
        fillcolor='rgba(65, 105, 225, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=10)
            )
        ),
        showlegend=False,
        title=f"{policy_name} - Comprehensive Analysis",
        height=450,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# -------------------------
# Metrics Bar Chart
# -------------------------
def create_metrics_chart(policy_data):
    metrics = ['Affordability', 'Coverage', 'Benefits', 'Claims', 'Flexibility']
    scores = [
        policy_data.get('affordability_score', 70),
        policy_data.get('coverage_score', 75),
        policy_data.get('benefits_score', 80),
        policy_data.get('claim_settlement_ratio', 85),
        policy_data.get('flexibility_score', 70)
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metrics,
        y=scores,
        marker_color=colors,
        text=[f'{score}%' for score in scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Policy Metrics Score',
        yaxis=dict(range=[0, 100], title='Score (%)'),
        height=350,
        showlegend=False
    )
    
    return fig

# -------------------------
# Premium vs Coverage Scatter Plot
# -------------------------
def create_premium_coverage_chart(policy_data, user_details):
    # Sample data for comparison (in real app, this would come from database)
    policies = [
        {'name': 'Basic Plan', 'premium': 8000, 'coverage': 500000},
        {'name': 'Standard Plan', 'premium': 15000, 'coverage': 1000000},
        {'name': 'Premium Plan', 'premium': 25000, 'coverage': 2000000},
        {'name': policy_data.get('policy_name', 'Current Policy'), 
         'premium': policy_data.get('avg_premium', 18000), 
         'coverage': policy_data.get('coverage_value', 1500000)}
    ]
    
    df = pd.DataFrame(policies)
    
    fig = px.scatter(
        df, x='premium', y='coverage', 
        text='name', size='coverage',
        title='Premium vs Coverage Comparison',
        labels={'premium': 'Annual Premium (₹)', 'coverage': 'Coverage Amount (₹)'}
    )
    
    fig.update_traces(
        textposition='top center',
        marker=dict(line=dict(width=2, color='DarkSlateGrey'))
    )
    
    fig.update_layout(height=400)
    
    return fig

# -------------------------
# Feature Importance Chart
# -------------------------
def create_feature_importance_chart(policy_data):
    features = policy_data.get('key_features', [])
    if not features:
        # Default features if none extracted
        features = [
            "Death Benefit",
            "Critical Illness Cover",
            "Tax Benefits",
            "Premium Waiver",
            "Accidental Death Benefit"
        ]
    
    importance_scores = [90, 85, 75, 70, 65][:len(features)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features,
        x=importance_scores,
        orientation='h',
        marker_color='lightseagreen'
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis=dict(title='Importance Score', range=[0, 100]),
        height=300,
        margin=dict(l=150)  # Extra margin for long feature names
    )
    
    return fig

# -------------------------
# Benefit Timeline Chart
# -------------------------
def create_benefit_timeline_chart(policy_data, user_details):
    age = user_details.get('age', 30)
    years = list(range(age, age + 31, 5))
    
    # Sample benefit accumulation (would be based on actual policy details)
    benefits = [100000 * (1.05 ** (i - age)) for i in years]
    premiums_paid = [15000 * (i - age) for i in years]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years, y=benefits,
        mode='lines+markers',
        name='Accumulated Benefits',
        line=dict(color='green', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=years, y=premiums_paid,
        mode='lines+markers',
        name='Premiums Paid',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title='Benefit Projection Timeline',
        xaxis=dict(title='Age'),
        yaxis=dict(title='Amount (₹)'),
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

# -------------------------
# Policy Comparison Chart
# -------------------------
def create_policy_comparison_chart(policy_data, user_details):
    # Sample data for comparison
    policies = ['Policy A', 'Policy B', 'Policy C', policy_data.get('policy_name', 'Current Policy')]
    
    categories = ['Premium', 'Coverage', 'Benefits', 'Flexibility']
    
    data = [
        [70, 65, 80, 60],  # Policy A
        [80, 75, 70, 85],  # Policy B
        [65, 85, 75, 70],  # Policy C
        [policy_data.get('affordability_score', 75), 
         policy_data.get('coverage_score', 80),
         policy_data.get('benefits_score', 85),
         policy_data.get('flexibility_score', 75)]  # Current Policy
    ]
    
    fig = go.Figure()
    
    for i, policy in enumerate(policies):
        fig.add_trace(go.Scatterpolar(
            r=data[i],
            theta=categories,
            fill='toself',
            name=policy
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title='Policy Comparison',
        height=500
    )
    
    return fig

# -------------------------
# Premium Breakdown Pie Chart
# -------------------------
def create_premium_breakdown_chart(policy_data):
    labels = ['Base Premium', 'Administrative Fees', 'Risk Charge', 'Taxes', 'Investment Component']
    values = [60, 10, 15, 10, 5]  # Default percentages
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3)
    ))
    
    fig.update_layout(
        title='Premium Breakdown',
        height=400,
        showlegend=True
    )
    
    return fig

# -------------------------
# Policy recommendation function with web search
# -------------------------
def recommend_policy(age, income_range, occupation, family_members, existing_insurance, health_conditions, language="English"):
    # Search for popular policies based on user profile
    search_query = f"best insurance policies for {occupation} with income {income_range} India 2024"
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
    4. Premium estimates (provide specific numbers if possible)
    5. Coverage details
    6. Comparison with similar policies
    7. Final recommendation (should this user consider this policy?)
    
    Provide your response in {language} language.
    Be objective and evidence-based in your analysis.
    """
    
    return call_gemini(prompt, max_output_tokens=2048)

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
        with st.spinner("Analyzing your profile and searching for the best policies..."):
            recommendation = recommend_policy(
                age, income_range, occupation, family_members, 
                existing_insurance, health_conditions, language
            )
            
            st.success("Here are insurance policies tailored for you:")
            st.markdown(recommendation)
            
            # Sample visualization based on user profile
            st.subheader("📊 Recommended Policy Types Based on Your Profile")
            
            # Create sample data based on user profile
            user_profile = {
                'age': age,
                'income': income_range,
                'occupation': occupation,
                'family_members': family_members
            }
            
            # Determine policy focus based on user profile
            policy_focus = []
            policy_weights = []
            
            if age > 50:
                policy_focus.extend(['Health Insurance', 'Critical Illness', 'Senior Citizen'])
                policy_weights.extend([40, 35, 25])
            elif family_members > 3:
                policy_focus.extend(['Family Health', 'Term Life', 'Child Education'])
                policy_weights.extend([45, 35, 20])
            else:
                policy_focus.extend(['Term Life', 'Health Insurance', 'Investment'])
                policy_weights.extend([40, 35, 25])
            
            # Create pie chart
            fig = px.pie(
                values=policy_weights, 
                names=policy_focus, 
                title='Recommended Insurance Focus'
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🔍 Policy Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        policy_name = st.text_input("Enter policy name to analyze")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Suggest Popular Policies"):
            st.write("Popular policies: " + ", ".join(POPULAR_POLICIES[:3]))
    
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
            
            # Create all visualizations
            st.subheader("📈 Comprehensive Policy Analysis")
            visualizations, policy_data = create_policy_visualizations(policy_name, analysis, user_details)
            
            # Display visualizations in a grid
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(visualizations['radar'], use_container_width=True)
            with col2:
                st.plotly_chart(visualizations['metrics'], use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(visualizations['scatter'], use_container_width=True)
            with col4:
                st.plotly_chart(visualizations['features'], use_container_width=True)

            col5, col6 = st.columns(2)
            with col5:
                st.plotly_chart(visualizations['timeline'], use_container_width=True)
            with col6:
                st.plotly_chart(visualizations['premium_breakdown'], use_container_width=True)

            st.plotly_chart(visualizations['comparison'], use_container_width=True)
            
            # Display extracted policy data
            st.subheader("📋 Policy Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Premium Range", policy_data['premium_range'])
            with col2:
                st.metric("Coverage Amount", policy_data['coverage_amount'])
            with col3:
                st.metric("Policy Term", policy_data['policy_term'])
            
            # Display key features
            if policy_data['key_features']:
                st.subheader("✨ Key Features")
                for feature in policy_data['key_features']:
                    st.markdown(f"✓ {feature}")
            
            # Final recommendation card
            st.subheader("🎯 Recommendation")
            if policy_data['suitability_score'] >= 70:
                st.success(f"**Recommended** (Suitability: {policy_data['suitability_score']}%)")
            elif policy_data['suitability_score'] >= 50:
                st.warning(f"**Moderately Recommended** (Suitability: {policy_data['suitability_score']}%)")
            else:
                st.error(f"**Not Recommended** (Suitability: {policy_data['suitability_score']}%)")

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