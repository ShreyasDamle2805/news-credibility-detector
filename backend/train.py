import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load Excel files (move your Excel files here first)
try:
    # For CSV files (most common)
    fake_df = pd.read_csv("Fake.csv")  
    true_df = pd.read_csv("True.csv")
    fake_df['label'] = 'FAKE'
    true_df['label'] = 'REAL'
    df = pd.concat([fake_df, true_df], ignore_index=True)

    print("✅ Loaded Excel files")
    
except:
    print("❌ Excel files not found. Create data.csv instead.")
    df = pd.read_csv("data.csv")

# Clean + prepare data
df["content"] = df.get("title", pd.Series()).fillna("") + " " + df.get("text", pd.Series()).fillna("")
df = df.dropna(subset=["content"])
df["label"] = df["label"].map({"FAKE": 0, "REAL": 1, "fake": 0, "real": 1}).dropna()

X = df["content"]
y = df["label"]
print(f"Dataset: {len(df)} samples")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("\n📊 MODEL PERFORMANCE:")
print(classification_report(y_test, y_pred))

# Save models
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n✅ Model and vectorizer saved successfully!")
