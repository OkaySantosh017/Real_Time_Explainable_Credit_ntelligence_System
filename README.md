# Real-Time Explainable Credit Intelligence System

This repository contains a full-stack system with:
- **frontend/**: React + Tailwind UI
- **backend/**: Node.js + Express + MongoDB
- **ml-model/**: Python + FastAPI + RandomForest + SHAP

## Architecture
- User signs up / logs in via backend with JWT
- Credit analysis form sends data to backend
- Backend calls ML service `/predict`, stores history
- Frontend displays score, risk, explanation, and history

## Setup

### 1) ML model service
```bash
cd ml-model
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2) Backend
```bash
cd backend
npm install
cp .env.example .env
# edit .env if needed
npm run dev
```

### 3) Frontend
```bash
cd frontend
npm install
npm run dev
```

Open frontend at `http://localhost:5173`, backend at `http://localhost:5000`, ML at `http://localhost:8000`.

## Notes
- Frontend uses `VITE_BACKEND_URL` (defaults to `http://localhost:5000/api`)
- The ML endpoint returns `score`, `risk`, `explanation`, and `feature_importance`

## Sample Input
```
{
  "income": 60000,
  "credit_history": 7,
  "loan_amount": 18000,
  "spending_score": 65,
  "transaction_frequency": 22,
  "age": 34,
  "tenure_months": 48
}
```
