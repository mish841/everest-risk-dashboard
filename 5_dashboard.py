# Everest Insurance Risk Dashboard (Overhauled)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from dash import dash_table
import base64
import os
import shap
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('merged_df_with_risk.csv')
X_test = pd.read_csv('X_test_processed.csv')

# Load your trained model and explainer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the saved model and explainer
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
X_train = pd.read_csv('X_train_processed.csv')
y_train = pd.read_csv('y_train.csv')
y_train_binned = y_train['risk_score'].apply(lambda score: 1 if score >= 60 else 0)
rf_model.fit(X_train, y_train_binned)

explainer = shap.Explainer(rf_model.predict, X_train)
shap_values = explainer(X_test)

# Classify risk level
def classify_risk(score):
    if score <= 19:
        return 'Very Low'
    elif score <= 39:
        return 'Low'
    elif score <= 59:
        return 'Moderate'
    elif score <= 79:
        return 'High'
    else:
        return 'Very High'

df['risk_level'] = df['risk_score'].apply(classify_risk)

# Assign colors by risk level
risk_colors = {
    'Very Low': '#00FF00',   # Green
    'Low': '#FFFF00',        # Yellow
    'Moderate': '#FFA500',   # Orange
    'High': '#FF0000',       # Red
    'Very High': '#800000'   # Dark Red
}
df['color'] = df['risk_level'].map(risk_colors)

# THEME
DARK_BG = "#0F0F1A"
ACCENT_BLUE = "#2E6BFF"
TEXT_LIGHT = "#FFFFFF"
CARD_BG = "#1C1C2E"
CARD_WHITE = "#FFFFFF"
FONT = "Poppins, sans-serif"

# DASH APP
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Everest Risk Dashboard"

app.validation_layout = html.Div([
    dcc.Tabs(id='tabs'),
    html.Div(id='tab-content'),
    dcc.Dropdown(id='industry-filter'),
    dcc.Dropdown(id='policy-filter'),
    dcc.Dropdown(id='shap-company-selector')
])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Insurance Risk Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
        {%metas%}
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# GLOBAL FILTERS
def get_filters():
    return html.Div([
        html.Div([
            html.Label("Filter by Industry", style={'color': TEXT_LIGHT}),
            dcc.Dropdown(
                options=[{"label": i, "value": i} for i in df['industry'].unique()],
                multi=True,
                id='industry-filter',
                placeholder="Select industry..."
            )
        ], style={'width': '45%'}),

        html.Div([
            html.Label("Filter by Policy Type", style={'color': TEXT_LIGHT}),
            dcc.Dropdown(
                options=[{"label": i, "value": i} for i in df['policy_type'].unique()],
                multi=True,
                id='policy-filter',
                placeholder="Select policy type..."
            )
        ], style={'width': '45%'})
    ], style={'display': 'flex', 'gap': '30px', 'marginBottom': '20px'})

# KPI CARDS
kpi_style = {
    'padding': '15px',
    'backgroundColor': CARD_BG,
    'borderRadius': '8px',
    'textAlign': 'center',
    'color': TEXT_LIGHT,
    'fontWeight': 'bold',
    'flex': 1
}

def get_kpis(filtered_df):
    return html.Div([
        html.Div([
            html.H3("Avg Risk Score", style={'color': ACCENT_BLUE}),
            html.H1(f"{filtered_df['risk_score'].mean():.1f}")
        ], style=kpi_style),
        html.Div([
            html.H3("High Risk %", style={'color': ACCENT_BLUE}),
            html.H1(f"{(filtered_df['risk_level'] == 'Very High').mean() * 100:.1f}%")
        ], style=kpi_style),
        html.Div([
            html.H3("Total Companies", style={'color': ACCENT_BLUE}),
            html.H1(filtered_df['company_name'].nunique())
        ], style=kpi_style)
    ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '30px'})

# TABS
app.layout = html.Div([
    html.Div([
        html.H1("Insurance Risk Assessment Dashboard", style={'color': ACCENT_BLUE}),
        html.Img(src="assets/everest-logo.jpeg", style={'height': '60px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '20px'}),
    get_filters(),
    dcc.Tabs(id='tabs', value='summary', children=[
        dcc.Tab(label='Summary', value='summary', style={'fontFamily': FONT}),
        dcc.Tab(label='Top Performers', value='top', style={'fontFamily': FONT}),
        dcc.Tab(label='Insights (SHAP)', value='shap', style={'fontFamily': FONT})
    ]),

    html.Div(id='tab-content')
], style={'padding': '25px', 'backgroundColor': DARK_BG, 'fontFamily': FONT})

# TAB CALLBACK
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('industry-filter', 'value'),
    Input('policy-filter', 'value')
)
def render_tab(tab, industry, policy):
    filtered = df.copy()
    if industry:
        filtered = filtered[filtered['industry'].isin(industry)]
    if policy:
        filtered = filtered[filtered['policy_type'].isin(policy)]

    if tab == 'summary':
        risk_summary = (
            filtered.groupby('industry')['risk_score']
            .mean()
            .reset_index()
        )
        risk_summary['risk_level'] = risk_summary['risk_score'].apply(classify_risk)
        return html.Div([
            get_kpis(filtered),
            html.Div([
                html.Div([dcc.Graph(figure=px.pie(filtered, names='policy_type', title='Policy Distribution'))],
                         style={'flex': 1, 'backgroundColor': CARD_BG, 'padding': '10px', 'borderRadius': '8px'}),
                html.Div([dcc.Graph(figure=px.bar(
                    risk_summary,
                    x='industry',
                    y='risk_score',
                    color='risk_level',
                    color_discrete_map=risk_colors,
                    title='Avg Risk by Industry'
                ))], style={'flex': 1, 'backgroundColor': CARD_BG, 'padding': '10px', 'borderRadius': '8px'}),
                html.Div([
                    html.H4("Risk Level Legend", style={'color': ACCENT_BLUE, 'fontWeight': 'bold'}),
                    html.P("🟩 Very Low (0–19)"),
                    html.P("🟨 Low (20–39)"),
                    html.P("🟧 Moderate (40–59)"),
                    html.P("🟥 High (60–79)"),
                    html.P("🔳 Very High (80–100)")
                ], style={'flex': 0.5, 'backgroundColor': CARD_WHITE, 'padding': '15px', 'borderRadius': '8px'})
            ], style={'display': 'flex', 'gap': '20px'})
        ])

    elif tab == 'top':
        top_companies = (
            filtered.groupby(['company_id', 'company_name', 'industry'])
            .agg({'risk_score': 'mean'})
            .reset_index()
            .sort_values('risk_score', ascending=False)
            .head(10)
        )
        return html.Div([
            html.H3("Top 10 Companies by Risk Score", style={'color': ACCENT_BLUE}),
            dash_table.DataTable(
                columns=[
                    {"name": "Company Name", "id": "company_name"},
                    {"name": "Industry", "id": "industry"},
                    {"name": "Risk Score", "id": "risk_score"},
                ],
                data=top_companies.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '5px',
                    'color': TEXT_LIGHT,
                    'backgroundColor': CARD_BG,
                    'fontFamily': FONT
                },
                style_header={
                    'backgroundColor': ACCENT_BLUE,
                    'color': TEXT_LIGHT,
                    'fontWeight': 'bold'
                },
            )
        ])

    elif tab == 'shap':
        return html.Div([
            html.H3("Model Explainability (SHAP)", style={'color': ACCENT_BLUE}),
            html.Label("Select a company to view SHAP explanation:", style={'color': TEXT_LIGHT}),
            dcc.Dropdown(
                id='shap-company-selector',
                options=[{'label': df.iloc[i]['company_name'], 'value': i} for i in range(len(X_test))],
                placeholder="Choose a company",
                style={'width': '50%', 'color': DARK_BG, 'marginBottom': '20px'}
            ),
            html.Div(id='shap-plot-output'),
            html.Div([
                html.H4("🔎 How We Built the Model", style={'color': ACCENT_BLUE}),
                html.Ul([
                    html.Li("Used Random Forest Classifier (100 trees, balanced class weights)."),
                    html.Li("Input features included policy, claim, and company attributes."),
                    html.Li("Target = whether a company is High Risk (risk score ≥ 60)."),
                    html.Li("Model trained on processed training data and tested for accuracy."),
                    html.Li("SHAP used for explainability to interpret individual predictions.")
                ], style={'color': TEXT_LIGHT})
            ], style={'marginTop': '30px', 'backgroundColor': CARD_BG, 'padding': '20px', 'borderRadius': '8px'})
        ])

# CALLBACK: SHAP waterfall plot
@app.callback(
    Output('shap-plot-output', 'children'),
    Input('shap-company-selector', 'value')
)
def update_shap_plot(company_idx):
    if company_idx is None:
        return html.P("Please select a company to view SHAP explanation.", style={'color': TEXT_LIGHT})

    shap_values_row = shap_values[company_idx].values
    feature_names = X_test.columns

    fig = go.Figure(go.Bar(
        x=shap_values_row,
        y=feature_names,
        orientation='h',
        marker_color=shap_values_row,
        text=[f"{val:.3f}" for val in shap_values_row],
        textposition="auto"
    ))

    fig.update_layout(
        title=f"SHAP Explanation for {df['company_name'].iloc[company_idx]}",
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        font=dict(color=TEXT_LIGHT),
        xaxis_title="SHAP Value (Impact on Risk Prediction)"
    )

    return html.Div([
        dcc.Graph(figure=fig),
        html.Div([
            html.H4("Explanation", style={'color': ACCENT_BLUE}),
            html.P("📊 Each bar shows how much that feature contributed to the model's final risk score prediction for this company.", style={'color': TEXT_LIGHT}),
            html.P("➡️ Positive values (bars to the right) mean the feature 🔺 increased the predicted risk.", style={'color': TEXT_LIGHT}),
            html.P("⬅️ Negative values (bars to the left) mean the feature 🔻 reduced the predicted risk.", style={'color': TEXT_LIGHT}),
            html.P("📏 The longer the bar, the more influential that feature was in shaping the outcome.", style={'color': TEXT_LIGHT})
        ], style={'marginTop': '15px', 'backgroundColor': CARD_BG, 'padding': '15px', 'borderRadius': '8px'})
    ])

if __name__ == '__main__':
   app.run(debug=True)