from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import pickle
import logging
from datetime import datetime
import os

# Logging setup
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting TextFlare API...")

# Load vectorizer
try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("TF-IDF vectorizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading vectorizer: {e}")
    raise

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TFLite model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading TFLite model: {e}")
    raise

# FastAPI setup
app = FastAPI(title="TextFlare API", description="TFLite + TFIDF Text Classifier", version="1.0")

# CORS (allow all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://yourdomain.com"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    try:
        logger.info(f"Received prediction request: {data.text}")

        # Vectorize input
        vector = vectorizer.transform([data.text]).toarray().astype(np.float32)
        logger.debug(f"Vectorized input shape: {vector.shape}")

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], vector)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        logger.info(f"Prediction result: {output.tolist()}")
        return {"prediction": output.tolist()}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}
