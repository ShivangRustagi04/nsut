import streamlit as st
import plotly.express as px
from apicalls import recommend_policy, analyze_policy, chat_with_user
from helpers import (
    INDIAN_OCCUPATIONS, INCOME_RANGES,
    create_policy_visualizations
)

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
            
            # # Sample visualization based on user profile
            # st.subheader("📊 Recommended Policy Types Based on Your Profile")
            
            # # Determine policy focus based on user profile
            # policy_focus = []
            # policy_weights = []
            
            # if age > 50:
            #     policy_focus.extend(['Health Insurance', 'Critical Illness', 'Senior Citizen'])
            #     policy_weights.extend([40, 35, 25])
            # elif family_members > 3:
            #     policy_focus.extend(['Family Health', 'Term Life', 'Child Education'])
            #     policy_weights.extend([45, 35, 20])
            # else:
            #     policy_focus.extend(['Term Life', 'Health Insurance', 'Investment'])
            #     policy_weights.extend([40, 35, 25])
            
            # # Create pie chart
            # fig = px.pie(
            #     values=policy_weights, 
            #     names=policy_focus, 
            #     title='Recommended Insurance Focus'
            # )
            # st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🔍 Policy Analysis")
    
    col1 = st.columns([3,1])[0]
    with col1:
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