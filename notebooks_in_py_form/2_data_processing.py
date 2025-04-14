from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

# Load generated CSVs
companies_df = pd.read_csv('companies.csv')
policies_df = pd.read_csv('policies.csv')
claims_df = pd.read_csv('claims.csv')
risk_scores_df = pd.read_csv('risk_scores.csv')

# Merge tables
merged_df = policies_df.merge(companies_df, on='company_id')
merged_df = merged_df.merge(risk_scores_df[['company_id', 'risk_score']], on='company_id')

# Aggregate claims data
claims_agg = claims_df.groupby('policy_id').agg(
    num_claims=('claim_id', 'count'),
    avg_claim_amount=('claim_amount', 'mean'),
    last_claim_date=('claim_date', 'max'),
    claim_approval_rate=('claim_status', lambda x: (x == 'Approved').mean())
).reset_index()

# Merge claims into merged_df
merged_df = merged_df.merge(claims_agg, on='policy_id', how='left')

# Fill NaN for policies with no claims
merged_df.fillna({
    'num_claims': 0,
    'avg_claim_amount': 0,
    'claim_approval_rate': 0,
    'last_claim_date': '2000-01-01'  # placeholder old date
}, inplace=True)

# Compute derived features
merged_df['policy_duration_days'] = (
    pd.to_datetime(merged_df['end_date']) - pd.to_datetime(merged_df['start_date'])
).dt.days

merged_df['time_since_last_claim_days'] = (
    pd.to_datetime('2025-04-12') - pd.to_datetime(merged_df['last_claim_date'])
).dt.days

# Label Encode categorical variables
le = LabelEncoder()
merged_df['policy_type_encoded'] = le.fit_transform(merged_df['policy_type'])
merged_df['industry_encoded'] = le.fit_transform(merged_df['industry'])
merged_df['company_size_encoded'] = le.fit_transform(merged_df['company_size'])

# Select features and target
features = [
    'policy_type_encoded', 'coverage_amount', 'premium_cost', 'policy_duration_days',
    'num_claims', 'avg_claim_amount', 'claim_approval_rate', 'time_since_last_claim_days',
    'industry_encoded', 'company_size_encoded', 'revenue'
]

target = 'risk_score'

# Train-Test Split
X = merged_df[features]
y = merged_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize numerical features (fit on training, apply to test)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features)

# Save processed datasets (if you want)
X_train_scaled.to_csv('X_train_processed.csv', index=False)
X_test_scaled.to_csv('X_test_processed.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
