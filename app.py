import streamlit as st
import plotly.graph_objects as go
from helpers import recommend_policy, analyze_policy, create_policy_visualization, chat_with_user

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
    # Health conditions
    # -------------------------
HEALTH_CONDITIONS = [
        "None",
        "Diabetes",
        "Hypertension", 
        "Heart Condition",
        "Respiratory Issues",
        "Asthma",
        "Thyroid Disorder",
        "Kidney Disease",
        "Liver Disease",
        "Cancer (Past/Present)",
        "Mental Health Condition",
        "Arthritis",
        "Back/Spine Issues",
        "Eye/Vision Problems",
        "Hearing Problems",
        "Other Chronic Condition"
    ]


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
    health_conditions = st.multiselect("Health Conditions", HEALTH_CONDITIONS)

    language = st.selectbox("Preferred Language", [
        "Hindi", "Gujarati", "Tamil", "Telugu", "Bengali", 
        "Marathi", "Kannada", "English"
    ])
        # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>ℹ️ Disclaimer: This is an AI-powered insurance assistant. For official policy details and purchases, please consult with licensed insurance providers.</p>
    </div>
    """, unsafe_allow_html=True)

# Main content area
st.title("PRAYAAS: Your Insurance Companion")
st.markdown("Helping you understand and choose the right insurance policies in your preferred language.")

# Tab interface
tab1, tab2, tab3 = st.tabs(["Policy Recommendation", "Policy Analysis", "Chat Assistant"])

with tab1:
    st.header("Personalized Policy Recommendations")
    
    if st.button("Get Policy Recommendations", type="primary"):
        with st.spinner("Analyzing your profile and generating recommendations..."):
            recommendation = recommend_policy(
                age, income_range, occupation, family_members, 
                existing_insurance, health_conditions, language or "English"
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
            analysis = analyze_policy(policy_name, user_details, language or "English")
            
            st.success(f"Analysis of {policy_name}:")
            st.markdown(analysis)
            
            # Create visualization
            st.subheader("📈 Policy Visual Analysis")
            viz_fig = create_policy_visualization(policy_name, analysis or "")
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
            response = chat_with_user(prompt, st.session_state.messages[-5:], language or "English")

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
