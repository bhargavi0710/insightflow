# InsightFlow AI

> Upload any dataset. Get instant analytics, ML predictions, and AI-powered insights — no code required.

**Live Demo → [insightflow-virid.vercel.app](https://insightflow-virid.vercel.app)**

---

## What it does

InsightFlow AI turns raw CSV data into actionable insights in seconds. Upload a dataset and immediately get:

- **Statistical analysis** — mean, median, std dev, null counts across every column
- **Interactive charts** — bar and line charts for both numeric and categorical data
- **ML forecasting** — train an XGBoost model to predict any column, with SHAP explainability
- **Text classification** — classify free text (e.g. sentiment analysis) using TF-IDF + Logistic Regression
- **Live inference** — use the trained model to predict on new inputs, with confidence intervals
- **AI chat** — ask natural language questions about your data, powered by Llama 3.3 via Groq

---

## Demo

| Feature | Screenshot |
|---|---|
| Dataset Overview | Stats table with null detection |
| Chart Builder | Categorical and numeric charts |
| ML Forecast | XGBoost + SHAP feature importance |
| Prediction | Live inference with confidence range |
| Text Classification | Word-level explainability |
| AI Chat | Full-dataset context, markdown responses |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User's Browser                       │
│              Next.js 15 + Framer Motion                  │
└──────────────────────┬──────────────────────────────────┘
                       │ REST API calls
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Backend                          │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Analytics  │  │  ML Forecast │  │   AI Chat     │  │
│  │  (Pandas)   │  │  (XGBoost +  │  │   (Groq /     │  │
│  │             │  │   SHAP)      │  │  Llama 3.3)   │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐                      │
│  │   SQLite    │  │  Joblib      │                      │
│  │  Database   │  │ Model Store  │                      │
│  └─────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

**Frontend** → Vercel  
**Backend** → Render  
**Database** → SQLite (persisted on Render disk)  
**AI** → Groq API (free tier, Llama 3.3 70B)

---

## Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| FastAPI | REST API framework |
| XGBoost | Gradient boosting for regression + classification |
| SHAP | Model explainability (feature importance) |
| Scikit-learn | TF-IDF text classification, metrics, cross-validation |
| Pandas | Data loading, transformation, statistical analysis |
| Joblib | Model persistence (save/reload trained models) |
| Groq + Llama 3.3 | AI chat with full dataset context |
| SQLAlchemy + SQLite | Dataset metadata storage |

### Frontend
| Technology | Purpose |
|---|---|
| Next.js 15 | React framework with App Router |
| Framer Motion | Animations (tab transitions, staggered cards, progress bars) |
| Recharts | Charts (bar, line, scatter with reference lines) |
| Tailwind CSS | Styling |
| React Markdown | Renders formatted AI responses |
| Axios | API communication |

---

## ML Pipeline

InsightFlow automatically detects the right ML approach for your data:

### Tabular ML (numeric + categorical CSV data)
1. **Auto-detects** regression vs classification based on target column type
2. **Feature engineering** — one-hot encodes low-cardinality categorical columns automatically
3. **Trains XGBoost** with tuned hyperparameters (n_estimators=200, subsample=0.8)
4. **5-fold cross-validation** for robust accuracy estimation
5. **SHAP values** explain which features drove each prediction
6. **Saves model to disk** — inference works instantly without retraining
7. **Confidence intervals** on regression predictions based on test MAE

### Text Classification (free text → category)
1. **TF-IDF vectorization** (3000 features, bigrams, English stop words removed)
2. **Logistic Regression** — fast, accurate, explainable baseline for text
3. **Word-level explainability** — shows top words driving each class prediction
4. **Sample predictions** — shows real examples with correct/incorrect labels

---

## Key Design Decisions

**Why XGBoost over Linear Regression?**
Housing price relationships are non-linear — crime rate affects price differently depending on neighborhood wealth. XGBoost captures these interactions; linear models can't. Validated with cross-validation rather than assuming.

**Why TF-IDF + Logistic Regression for text (not a transformer)?**
For a zero-latency demo on free infrastructure, TF-IDF + LR achieves 85%+ accuracy on sentiment classification in under 20 seconds of training. A transformer (BERT) would need GPU infrastructure and 10x the training time. The explainability (word weights) is also cleaner to show users.

**Why SHAP over raw feature importance?**
Raw feature importance from XGBoost tells you which features the model used most. SHAP tells you the magnitude and direction of each feature's effect on individual predictions — a more honest and useful measure.

**Why MAE-based confidence intervals?**
Statistically rigorous intervals require quantile regression or bootstrapping. MAE-based intervals are a practical approximation that gives users actionable uncertainty information without adding model complexity.

---

## Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Create `backend/.env`:
```
DATABASE_URL=sqlite:///./insightflow.db
GROQ_API_KEY=your_groq_key_here
SECRET_KEY=your_secret_key_here
```

```bash
uvicorn main:app --reload
# API runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# App runs at http://localhost:3000
```

---

## Supported Dataset Types

| Dataset Type | Features | Target | Supported |
|---|---|---|---|
| Numeric tabular | Numeric columns | Numeric column | Regression |
| Mixed tabular | Numeric + categorical | Numeric column | Regression |
| Classification tabular | Numeric columns | Low-cardinality numeric/text | Classification |
| Text + label | Free text column | Category column | Text classification |
| Time series | Date + numeric | — | Charts only |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/datasets/upload` | POST | Upload a CSV file |
| `/api/datasets/` | GET | List all datasets |
| `/api/datasets/{id}/preview` | GET | Preview first 10 rows |
| `/api/analytics/{id}/summary` | GET | Full statistical summary |
| `/api/analytics/{id}/chart-data` | GET | Chart data for two columns |
| `/api/forecast/{id}/forecast` | GET | Train model + get results |
| `/api/forecast/{id}/predict` | POST | Predict on new input values |
| `/api/forecast/{id}/text-classify` | GET | Run text classification |
| `/api/chat/{id}/chat` | POST | AI chat about dataset |

---

## What I learned building this

- How to structure a full-stack ML application with clean separation between analytics, ML, and AI layers
- Why cross-validation matters more than a single train/test split for honest model evaluation
- How SHAP values differ from raw feature importance and why that matters for user trust
- The tradeoffs between XGBoost and linear models in non-linear prediction tasks
- How to make ML models usable (inference UI, confidence intervals) rather than just reportable (metrics)
- Production deployment with environment-specific configuration, CORS, and free-tier constraints

---

## Roadmap

- [ ] PostgreSQL for production database
- [ ] User authentication and dataset history per user
- [ ] Anomaly detection tab
- [ ] PDF report export
- [ ] Model comparison (XGBoost vs Random Forest vs Linear)
- [ ] Time series forecasting with Prophet

---

## License

MIT