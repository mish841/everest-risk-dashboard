import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta

fake = Faker()
np.random.seed(42)
random.seed(42)

# 1. Companies Table
def generate_companies(n=100):
    industries = ['Tech', 'Healthcare', 'Finance', 'Retail', 'Manufacturing']
    sizes = ['Small', 'Medium', 'Large']
    companies = []

    for i in range(n):
        companies.append({
            'company_id': i,
            'company_name': fake.company(),
            'industry': random.choice(industries),
            'company_size': random.choice(sizes),
            'revenue': round(np.random.uniform(1e6, 5e8), 2),  # $1M to $500M
            'location': fake.city(),
            'founded_year': random.randint(1950, 2020)
        })
    return pd.DataFrame(companies)

# 2. Policies Table
def generate_policies(companies_df, max_policies_per_company=3):
    policy_types = ['Auto', 'Home', 'Liability', 'Cyber', 'Property']
    policies = []
    policy_id = 0

    for _, row in companies_df.iterrows():
        n_policies = random.randint(1, max_policies_per_company)
        for _ in range(n_policies):
            start_date = fake.date_between(start_date='-5y', end_date='-1y')
            end_date = start_date + timedelta(days=365*random.randint(1, 3))
            policies.append({
                'policy_id': policy_id,
                'company_id': row['company_id'],
                'policy_type': random.choice(policy_types),
                'coverage_amount': round(np.random.uniform(1e5, 5e6), 2),
                'premium_cost': round(np.random.uniform(500, 20000), 2),
                'start_date': start_date,
                'end_date': end_date
            })
            policy_id += 1
    return pd.DataFrame(policies)

# 3. Claims Table
def generate_claims(policies_df, claim_probability=0.3):
    statuses = ['Approved', 'Denied', 'Pending']
    reasons = ['Accident', 'Fire', 'Theft', 'Natural Disaster', 'Malfunction']
    claims = []
    claim_id = 0

    for _, row in policies_df.iterrows():
        if random.random() < claim_probability:
            n_claims = random.randint(1, 3)
            for _ in range(n_claims):
                claim_date = fake.date_between(start_date=row['start_date'], end_date=row['end_date'])
                claims.append({
                    'claim_id': claim_id,
                    'policy_id': row['policy_id'],
                    'claim_date': claim_date,
                    'claim_amount': round(np.random.uniform(1000, 500000), 2),
                    'claim_status': random.choice(statuses),
                    'claim_reason': random.choice(reasons)
                })
                claim_id += 1
    return pd.DataFrame(claims)

# 4. Risk Scores Table
def generate_risk_scores(companies_df, policies_df, claims_df):
    # Industry risk multipliers (just an example assumption)
    industry_risk = {
        'Tech': 0.8,
        'Healthcare': 1.0,
        'Finance': 1.2,
        'Retail': 1.3,
        'Manufacturing': 1.5
    }
    
    scores = []
    for _, company in companies_df.iterrows():
        company_id = company['company_id']
        company_policies = policies_df[policies_df['company_id'] == company_id]
        company_claims = claims_df[claims_df['policy_id'].isin(company_policies['policy_id'])]

        num_claims = len(company_claims)
        avg_claim_amount = company_claims['claim_amount'].mean() if num_claims > 0 else 0
        approval_rate = (
            (company_claims['claim_status'] == 'Approved').sum() / num_claims
            if num_claims > 0 else 1
        )
        avg_premium = company_policies['premium_cost'].mean() if not company_policies.empty else 0
        avg_coverage = company_policies['coverage_amount'].mean() if not company_policies.empty else 0
        avg_duration = (
            (company_policies['end_date'] - company_policies['start_date']).dt.days.mean()
            if not company_policies.empty else 0
        )

        # Base score calculated using a weighted formula
        risk_score = (
            num_claims * 5
            + avg_claim_amount * 0.0001
            - approval_rate * 10
            + avg_premium * 0.001
            + avg_coverage * 0.00005
            + avg_duration * 0.01
        )

        # Adjust based on industry
        risk_score *= industry_risk[company['industry']]

        # Clip and scale to 0-100
        risk_score = min(max(risk_score, 0), 100)

        scores.append({
            'company_id': company_id,
            'risk_score': round(risk_score, 2),
            'model_version': 'v2.0',
            'score_date': fake.date_between(start_date='-30d', end_date='today')
        })

    return pd.DataFrame(scores)


# Generate policies and companies 
companies_df = generate_companies()
policies_df = generate_policies(companies_df)

# Convert policy date columns to datetime
policies_df['start_date'] = pd.to_datetime(policies_df['start_date'])
policies_df['end_date'] = pd.to_datetime(policies_df['end_date'])

claims_df = generate_claims(policies_df)
risk_scores_df = generate_risk_scores(companies_df, policies_df, claims_df)

# Save to CSV
companies_df.to_csv('companies.csv', index=False)
policies_df.to_csv('policies.csv', index=False)
claims_df.to_csv('claims.csv', index=False)
risk_scores_df.to_csv('risk_scores.csv', index=False)
