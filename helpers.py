import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
# Extract structured data from analysis text
# -------------------------
def extract_policy_data(analysis_text):
    """
    Extract structured data from policy analysis text
    """
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
    """
    Calculate affordability score based on user income and policy premium
    """
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
    """
    Calculate coverage adequacy score based on user profile and policy coverage
    """
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
    """
    Create all policy visualizations
    """
    # Extract structured data from analysis
    policy_data = extract_policy_data(analysis_text)
    
    # Create all visualizations
    radar_fig = create_radar_chart(policy_name, policy_data, user_details)
    metrics_fig = create_metrics_chart(policy_data)
    scatter_fig = create_premium_coverage_chart(policy_data, user_details)
    feature_fig = create_feature_importance_chart(policy_data)
    timeline_fig = create_benefit_timeline_chart(policy_data, user_details)
    comparison_fig = create_policy_comparison_chart(policy_data, user_details)
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
    """
    Create radar chart for policy analysis
    """
    categories = ['Premium Affordability', 'Coverage Adequacy', 'Benefits Match', 'Claim Settlement', 'Flexibility', 'Overall Value']
    
    # Calculate scores based on user profile and policy data
    affordability = calculate_affordability_score(policy_data, user_details)
    coverage_adequacy = calculate_coverage_score(policy_data, user_details)
    benefits_match = min(100, 60 + len(policy_data['key_features']) * 8)
    claim_settlement = policy_data.get('claim_settlement_ratio', 85)
    flexibility = policy_data.get('flexibility_score', 70)
    
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
    """
    Create metrics bar chart
    """
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
    """
    Create premium vs coverage scatter plot
    """
    # Sample data for comparison
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
    """
    Create feature importance chart
    """
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
        margin=dict(l=150)
    )
    
    return fig

# -------------------------
# Benefit Timeline Chart
# -------------------------
def create_benefit_timeline_chart(policy_data, user_details):
    """
    Create benefit timeline chart
    """
    age = user_details.get('age', 30)
    years = list(range(age, age + 31, 5))
    
    # Sample benefit accumulation
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
    """
    Create policy comparison chart
    """
    # Sample data for comparison
    policies = ['Policy A', 'Policy B', 'Policy C', policy_data.get('policy_name', 'Current Policy')]
    
    categories = ['Premium', 'Coverage', 'Benefits', 'Flexibility']
    
    data = [
        [70, 65, 80, 60],
        [80, 75, 70, 85],
        [65, 85, 75, 70],
        [policy_data.get('affordability_score', 75), 
         policy_data.get('coverage_score', 80),
         policy_data.get('benefits_score', 85),
         policy_data.get('flexibility_score', 75)]
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
    """
    Create premium breakdown pie chart
    """
    labels = ['Base Premium', 'Administrative Fees', 'Risk Charge', 'Taxes', 'Investment Component']
    values = [60, 10, 15, 10, 5]
    
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