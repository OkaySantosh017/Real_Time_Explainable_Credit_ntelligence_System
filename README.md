Real-Time Explainable Credit Intelligence System
This project is an end-to-end Azure-based data engineering and machine learning solution designed to evaluate credit risk in real time while ensuring model transparency using Explainable AI.
🔹 Key Features
Real-time credit risk prediction & loan approval system
Integrated Explainable AI (SHAP, LIME) for model interpretability
Scalable cloud-based data pipeline on Azure
🔹 Architecture
ADF → Azure Data Lake Gen2 → Databricks (PySpark) → Synapse Analytics → ML Models → Power BI
🔹 Tech Stack
Data Engineering: Azure Data Factory (ADF), Azure Data Lake Gen2, Azure Synapse
Processing: Azure Databricks, PySpark, SQL
Machine Learning: Logistic Regression, Random Forest, XGBoost
Explainability: SHAP, LIME
Visualization: Power BI
🔹 Workflow
Data Ingestion: Built ETL pipelines using ADF to ingest financial & customer data
Data Processing: Cleaned and transformed data using PySpark in Databricks
Data Modeling: Designed fact & dimension tables in Synapse
Model Development: Trained ML models for credit risk prediction
Explainability: Applied SHAP & LIME for decision transparency
Visualization: Created dashboards in Power BI for insights
Deployment: Automated end-to-end pipeline for real-time predictions
🔹 Results
Achieved 94% accuracy and 0.97 AUC
Improved decision transparency and trust
Reduced manual analysis through automated pipelines
