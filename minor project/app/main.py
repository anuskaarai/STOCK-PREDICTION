from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
import sys

from app.routers import predictions

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Concurrency guard: only 1 training job at a time
training_semaphore = asyncio.Semaphore(1)

@asynccontextmanager
async def lifespan(app):
    logger.info("Stock Prediction API starting up...")
    yield
    logger.info("Stock Prediction API shutting down...")

app = FastAPI(
    title="Stock Prediction API",
    description="Multi-task BiLSTM + Attention + XGBoost Stock Price Prediction",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware (allow frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose semaphore via app state for use in routers
app.state.training_semaphore = training_semaphore

# Include Routers
app.include_router(predictions.router)

@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint."""
    return {"status": "ok", "service": "Stock Prediction API"}

