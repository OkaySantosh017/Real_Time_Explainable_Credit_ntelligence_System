from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import shap


def make_synthetic_data(n=2500):
    rng = np.random.default_rng(42)
    income = rng.normal(62000, 22000, n).clip(10000, 200000)
    credit_history = rng.integers(0, 11, n)
    loan_amount = rng.normal(19000, 12000, n).clip(1000, 100000)
    spending_score = rng.integers(1, 101, n)
    transaction_frequency = rng.integers(2, 60, n)
    age = rng.integers(20, 71, n)
    tenure_months = rng.integers(1, 300, n)
    risk = (
        (loan_amount / (income + 1)) * 0.5
        + (10 - credit_history) * 2
        + (100 - spending_score) * 0.05
        + (50 / (transaction_frequency + 1))
        + (60 - age) * 0.02
    )
    risk = np.clip(risk + rng.normal(0, 1.3, n), 0, 20)
    target = (risk < 8).astype(int)
    data = pd.DataFrame({
        'income': income,
        'credit_history': credit_history,
        'loan_amount': loan_amount,
        'spending_score': spending_score,
        'transaction_frequency': transaction_frequency,
        'age': age,
        'tenure_months': tenure_months,
        'target': target
    })
    return data


class CreditInput(BaseModel):
    income: float
    credit_history: float
    loan_amount: float
    spending_score: float
    transaction_frequency: float
    age: float
    tenure_months: float

app = FastAPI(title='Credit Intelligence ML API')

# train model at startup
raw = make_synthetic_data(2500)
features = [
    'income',
    'credit_history',
    'loan_amount',
    'spending_score',
    'transaction_frequency',
    'age',
    'tenure_months'
]

X = raw[features]
y = raw['target']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=180, random_state=42, max_depth=8)
model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model)


def map_risk(score_num: float):
    if score_num >= 70:
        return 'Low'
    if score_num >= 50:
        return 'Medium'
    return 'High'


def phrase_explanation(shap_values, input_dict):
    contributions = dict(zip(features, shap_values))
    sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top = sorted_feats[:3]
    reasons = []
    for name, val in top:
        direction = 'increasing' if val > 0 else 'decreasing'
        reasons.append(f"{name.replace('_', ' ')} is {direction} your score")
    return ' and '.join(reasons)


@app.get('/')
async def root():
    return {'status': 'ready', 'model': 'random-forest', 'api': '/predict'}


@app.post('/predict')
async def predict(payload: CreditInput):
    row = np.array([[
        payload.income,
        payload.credit_history,
        payload.loan_amount,
        payload.spending_score,
        payload.transaction_frequency,
        payload.age,
        payload.tenure_months
    ]])
    row_scaled = scaler.transform(row)
    proba = model.predict_proba(row_scaled)[0][1]
    score = int(np.round(proba * 100))
    risk = map_risk(score)

    try:
        shap_vals_raw = explainer.shap_values(row_scaled)
        if isinstance(shap_vals_raw, list) and len(shap_vals_raw) > 1:
            shap_vals = np.array(shap_vals_raw[1])[0]
        else:
            shap_vals = np.array(shap_vals_raw)[0]
        explanation_text = phrase_explanation(shap_vals, payload.dict())
        feature_importance = {features[i]: float(np.round(abs(shap_vals[i]), 4)) for i in range(len(features))}
    except Exception:
        shap_vals = model.feature_importances_
        explanation_text = "Your profile is impacted by high loan amount and lower credit history."
        feature_importance = {features[i]: float(np.round(abs(model.feature_importances_[i]), 4)) for i in range(len(features))}

    return {
        'score': int(score),
        'risk': risk,
        'explanation': explanation_text,
        'feature_importance': feature_importance
    }
