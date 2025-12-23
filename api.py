
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment API")

model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextIn):
    result = model(data.text)[0]
    return {
        "sentiment": result["label"],
        "confidence": result["score"]
    }
