import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI  
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

class TextItem(BaseModel):
    text: str

@app.get("/")
@app.get("/Home/")
def read_root():
    return {"message": "Welcome to the Arabic News Credibility Analyzer API!"}

@app.post("/predict/")
async def get_prediction(request: TextItem):
    result = predict(request.text)
    return result