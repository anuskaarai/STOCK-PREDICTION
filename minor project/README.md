# Quant-Edge — AI Stock Price Prediction

A production-ready stock price prediction dashboard for **Indian Stock Exchange (NSE/BSE)** stocks, powered by a hybrid **BiLSTM + Multi-Head Attention + XGBoost** ensemble model.

![Dashboard Preview](frontend/screenshot.png)

## Features

- **AI-Powered Predictions** — 21-hour hourly price forecasting using deep learning
- **On-Demand Training** — Enter any NSE/BSE ticker (e.g., `TCS.NS`, `RELIANCE.NS`) and the model trains automatically
- **37 Technical Indicators** — RSI, MACD, Bollinger Bands, VWAP, ATR, EMA, SMA and more
- **Model Caching** — Trained models are cached for 24 hours to speed up repeat queries
- **Interactive Dashboard** — Premium dark-mode UI with Chart.js visualizations
- **Historical + Predicted** — Unified chart showing past prices leading into future predictions
- **REST API** — FastAPI backend with Swagger documentation at `/docs`

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML, CSS, JavaScript, Chart.js |
| **Backend** | Python, FastAPI, Uvicorn |
| **ML Model** | TensorFlow/Keras (BiLSTM + Attention), XGBoost |
| **Data** | yfinance (Yahoo Finance API) |
| **Preprocessing** | scikit-learn (RobustScaler), pandas, numpy |

## Architecture

```
┌─────────────┐    POST /predict/    ┌──────────────────┐
│   Frontend   │ ──────────────────► │   FastAPI Server  │
│  (Chart.js)  │ ◄────────────────── │                  │
└─────────────┘   JSON Response      │  ┌────────────┐  │
                                     │  │ yfinance    │  │
                                     │  │ (data fetch)│  │
                                     │  └─────┬──────┘  │
                                     │        ▼         │
                                     │  ┌────────────┐  │
                                     │  │ Feature     │  │
                                     │  │ Engineering │  │
                                     │  │ (37 cols)   │  │
                                     │  └─────┬──────┘  │
                                     │        ▼         │
                                     │  ┌────────────┐  │
                                     │  │ BiLSTM +    │  │
                                     │  │ Attention + │  │
                                     │  │ XGBoost     │  │
                                     │  └────────────┘  │
                                     └──────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd "minor project"
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the Frontend

Open `frontend/index.html` in your browser.

### 4. Make a Prediction

Enter a ticker like `TCS.NS` and click **Predict**. The first prediction will take ~2 minutes (training), subsequent ones will be instant (cached).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict/` | Generate predictions (JSON body: `{ticker, forecast_hours, force_retrain}`) |
| `GET` | `/predict/{ticker}` | Quick predict with defaults |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger API documentation |

## Project Structure

```
minor project/
├── app/
│   ├── main.py              # FastAPI entry point + lifespan
│   ├── config.py            # Hyperparameters & feature columns
│   ├── routers/
│   │   └── predictions.py   # REST endpoints (async + semaphore)
│   ├── schemas/
│   │   └── prediction.py    # Pydantic request/response models
│   ├── services/
│   │   ├── data_fetcher.py      # yfinance data retrieval
│   │   ├── feature_engineer.py  # 37-feature pipeline
│   │   ├── model_builder.py     # BiLSTM + Attention architecture
│   │   ├── model_manager.py     # Model caching & persistence
│   │   └── predictor.py         # Orchestrator (train/predict)
│   └── models/              # Cached trained models (auto-created)
├── frontend/
│   ├── index.html           # Dashboard UI
│   ├── style.css            # Premium dark theme
│   └── app.js               # Chart.js + API integration
└── requirements.txt
```

## License

This project is for educational/academic purposes.
