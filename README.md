# no--show-iq-

NoShowIQ: Patient Appointment Risk Predictor

📌 Project OverviewNoShowIQ is a production-grade MLOps solution designed to help medical clinics predict the likelihood of patient "no-shows". By identifying high-risk appointments in advance, clinics can optimize scheduling and reduce lost revenue.

Live Deployment URL: [INSERT YOUR HUGGING FACE URL HERE]

 🛠️ Features & Technical Stack
 
 Machine Learning: Binary classification handling 80/20 class imbalance.
 
 API: FastAPI/Flask with endpoints for health, prediction, and history.
 
 Database: MongoDB Atlas for logging predictions and training runs.
 
 Containerization: Multi-stage Docker build under 300MB.
 
 CI/CD: GitHub Actions for automated linting (Flake8) and testing.
 
 📂 Project StructurePlaintextnoshow-iq-<sap-id>/
├── .github/workflows/  # CI/CD pipelines (Lint, Test, Deploy)
├── src/
│   └── noshow_iq/      # Core package modules (Q2)
│       ├── preprocess.py   # Data cleaning & feature engineering
│       ├── model.py        # ML training & evaluation
│       └── api.py          # API endpoints (FastAPI/Flask)
├── tests/              # Pytest suite (6+ tests required)
├── Dockerfile          # Multi-stage production image
├── docker-compose.yml  # Local stack (App, Mongo, Mongo-Express)
├── requirements.txt    # Project dependencies
└── pyproject.toml      # Package configuration for TestPyPI



