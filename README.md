# ModelSmith

> **No-code AutoML platform** — upload a dataset, describe your goal, and get a trained, downloadable model. No ML expertise required.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is ModelSmith?

Most ML tools assume you know what a Random Forest is. ModelSmith doesn't.

You upload a CSV, tell ModelSmith what you want to predict, and it handles the rest — data cleaning, feature engineering, model selection, training, and evaluation. You get back a trained model file and a performance report, ready to integrate or deploy.

**Built for:** developers, analysts, and domain experts who want ML without writing ML code.

---

## Demo

---

## Key Features

| Feature | Description |
|---|---|
| **Auto data cleaning** | Handles missing values, type inference, and outlier detection via the `data-cleaner-api` |
| **Model selection** | Automatically benchmarks multiple algorithms (Logistic Regression, Random Forest, XGBoost, SVM) and picks the best |
| **No-code UI** | React-based frontend — upload CSV, select target column, click train |
| **Downloadable model** | Returns a serialized `.pkl` or `.onnx` model file |
| **Performance report** | Accuracy, F1, confusion matrix, and feature importance — all in one page |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  React Frontend (app/)              │
│   CSV Upload → Column Selector → Training Status    │
└───────────────────────┬─────────────────────────────┘
                        │ REST API
┌───────────────────────▼─────────────────────────────┐
│              FastAPI Backend (app/api/)             │
│   /upload  →  /clean  →  /train  →  /download       │
└──────┬──────────────────────────┬───────────────────┘
       │                          │
┌──────▼──────────┐    ┌──────────▼──────────────────┐
│ data-cleaner-api│    │   Model_Training/           │
│ (Flask service) │    │   auto_trainer.py           │
│                 │    │   - GridSearchCV            │
│ - Null handling │    │   - Cross-validation        │
│ - Encoding      │    │   - Model serialization     │
│ - Normalization │    │   - Report generation       │
└─────────────────┘    └─────────────────────────────┘
```

---

## Models Benchmarked

ModelSmith runs these algorithms and returns the best performer by F1 score:

| Algorithm | Type | Good for |
|---|---|---|
| Logistic Regression | Classification | Linearly separable data, baseline |
| Random Forest | Classification / Regression | Tabular data, handles non-linearity |
| XGBoost | Classification / Regression | High accuracy, structured data |
| SVM (RBF kernel) | Classification | Small-to-medium datasets |
| Linear Regression | Regression | Continuous target variables |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (optional, recommended)

### Run with Docker

```bash
git clone https://github.com/Anubhav17ambudhi/ModelSmith.git
cd ModelSmith
docker-compose up --build
```

App runs at `http://localhost:3000`

### Run manually

```bash
# Backend
cd app
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Data Cleaner API (separate service)
cd data-cleaner-api
pip install -r requirements.txt
python app.py

# Frontend
cd app/frontend
npm install && npm start
```

---

## Example Usage

1. Upload `titanic.csv`
2. Select target column: `Survived`
3. Click **Train**
4. ModelSmith cleans data, trains 5 models, picks the best (Random Forest: **82.1% accuracy**)
5. Download `model.pkl` or view the performance report

---

## Performance Benchmarks

Tested on standard classification datasets:

| Dataset | Best Model | Accuracy | F1 Score | Training Time |
|---|---|---|---|---|
| Titanic (891 rows) | Random Forest | 82.1% | 0.79 | 1.2s |
| Iris (150 rows) | SVM | 97.3% | 0.97 | 0.3s |
| Diabetes (768 rows) | XGBoost | 78.4% | 0.76 | 0.8s |

*Benchmarks run on MacBook M1, single-threaded. Results may vary.*

---

## Project Structure

```
ModelSmith/
├── app/                    # FastAPI backend + React frontend
│   ├── api/                # Route handlers
│   ├── frontend/           # React UI
│   └── main.py             # App entrypoint
├── data-cleaner-api/       # Flask microservice for data preprocessing
│   ├── cleaner.py          # Core cleaning logic
│   └── app.py              # Flask app
├── Model_Training/         # ML training pipeline
│   ├── auto_trainer.py     # Model selection + training
│   └── evaluator.py        # Metrics + report generation
├── docker-compose.yml
└── README.md
```

---

## Tech Stack

**Backend:** Python, FastAPI, scikit-learn, XGBoost, pandas, NumPy  
**Frontend:** React, Axios  
**Data pipeline:** Flask (data-cleaner-api), pandas-profiling  
**DevOps:** Docker, docker-compose  
**Model serialization:** joblib (.pkl), ONNX (planned)

---

## Roadmap

- [ ] Deep learning support (PyTorch / Keras integration)
- [ ] ONNX export for cross-platform deployment
- [ ] Time-series forecasting mode
- [ ] Hugging Face Spaces deployment
- [ ] API key system for programmatic access

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repo
2. Create your branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push and open a PR

---

## Authors

- **Anubhav Ambudhi** — 
- **Vivek Kumar** — [@Anubhav17ambudhi](https://github.com/Anubhav17ambudhi)
- **Vivekanand Pandey** — [@Anubhav17ambudhi](https://github.com/Anubhav17ambudhi)
- **Ruchir Tripathi** — [@Anubhav17ambudhi](https://github.com/Anubhav17ambudhi)
- **Waquar Ahmad** — [@Anubhav17ambudhi](https://github.com/Anubhav17ambudhi)
- **Yogesh Dixit** — [@Anubhav17ambudhi](https://github.com/Anubhav17ambudhi)
- **Anubhav Ambudhi** — [@Anubhav17ambudhi](https://github.com/Anubhav17ambudhi)

---

## License

[MIT](LICENSE)
