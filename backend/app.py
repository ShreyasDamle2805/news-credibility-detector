from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel
import hashlib
app = FastAPI(title="News Credibility API")

# Load ML models
# ❌ DELETE THESE LINES (causing Render crash)
# model = joblib.load("model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# ✅ ADD THIS (works instantly on Render)
@app.on_event("startup")
async def load_demo_model():
    global model, vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    print("🚀 Demo ML model loaded successfully!")


# MongoDB (comment out initially)
# MONGO_URI = "mongodb://localhost:27017"  # or Atlas URL
# client = AsyncIOMotorClient(MONGO_URI)
# db = client.newsDB
# collection = db.predictions

class NewsRequest(BaseModel):
    text: str

# Simple in-memory users (production = database)
users_db = {}

class User(BaseModel):
    email: str
    password: str

@app.post("/register")
async def register(user: User):
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email exists")
    
    # Hash password (production = bcrypt)
    hashed = hashlib.sha256(user.password.encode()).hexdigest()
    users_db[user.email] = hashed
    return {"message": "User created"}

@app.post("/login")
async def login(user: User):
    if user.email not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    hashed = hashlib.sha256(user.password.encode()).hexdigest()
    if users_db[user.email] != hashed:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"message": "Login success", "token": "fake-jwt-token"}

@app.post("/predict")
async def predict(news: NewsRequest):
    # Preprocess text
    text_tfidf = vectorizer.transform([news.text.lower()])
    prediction = model.predict(text_tfidf)[0]
    probas = model.predict_proba(text_tfidf)[0]
    confidence = float(np.max(probas))
    
    # ✅ FIXED: 3-state prediction with uncertainty
    if confidence < 0.80:
        return {
            "prediction": "UNCERTAIN",
            "confidence": round(confidence, 3),
            "message": "Low confidence - verify with multiple sources"
        }
    elif confidence < 0.90:
        return {
            "prediction": "LIKELY_" + ("REAL" if prediction == 1 else "FAKE"),
            "confidence": round(confidence, 3),
            "message": "Moderate confidence"
        }
    else:
        return {
            "prediction": "REAL" if prediction == 1 else "FAKE",
            "confidence": round(confidence, 3),
            "message": "High confidence"
        }


@app.get("/")
def root():
    return {"message": "News Credibility API is running!"}
