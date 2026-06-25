import logging
import time
import random

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("dummy-model-server")

app = FastAPI()
Instrumentator().instrument(app).expose(app)

_leak_store = []

# Memory thresholds (in number of 10MB chunks) for log severity escalation
WARN_THRESHOLD = 6   # ~60MB
CRITICAL_THRESHOLD = 11  # ~110MB, approaching the 150Mi pod limit


@app.get("/")
def root():
    return {"status": "ok", "service": "dummy-model-server"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/predict")
def predict():
    latency = random.uniform(0.05, 0.15)
    time.sleep(latency)
    logger.info(f"predict served in {latency*1000:.1f}ms")
    return {"prediction": random.random(), "latency_ms": "normal"}


@app.get("/predict-slow")
def predict_slow():
    latency = random.uniform(1.0, 2.0)
    time.sleep(latency)
    logger.warning(
        f"predict-slow latency degraded: {latency*1000:.1f}ms (expected <150ms). "
        f"Possible causes: resource contention, model version regression, or upstream dependency slowdown."
    )
    return {"prediction": random.random(), "latency_ms": "degraded"}


@app.get("/leak")
def leak():
    _leak_store.append(bytearray(10 * 1024 * 1024))
    chunks = len(_leak_store)
    approx_mb = chunks * 10

    if chunks >= CRITICAL_THRESHOLD:
        logger.error(
            f"CRITICAL memory usage: ~{approx_mb}MB allocated ({chunks} chunks). "
            f"Approaching pod memory limit. OOMKill imminent if this trend continues."
        )
    elif chunks >= WARN_THRESHOLD:
        logger.warning(
            f"Elevated memory usage: ~{approx_mb}MB allocated ({chunks} chunks). "
            f"Memory growth detected, investigate for potential leak."
        )
    else:
        logger.info(f"Memory allocation: ~{approx_mb}MB ({chunks} chunks)")

    return {"leaked_chunks": chunks, "approx_mb": approx_mb}


@app.get("/reset-leak")
def reset_leak():
    _leak_store.clear()
    logger.info("Memory leak store reset to baseline")
    return {"status": "leak reset"}