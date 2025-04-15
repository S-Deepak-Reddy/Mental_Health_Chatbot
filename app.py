from flask import Flask, request, render_template, jsonify
import pandas as pd
import os
from datetime import datetime, timedelta

# Open-source transformer model for sentiment
from transformers import pipeline

# Initialize sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

app = Flask(__name__)

DATA_FILE = "student_responses.csv"

# Initialize storage if not present
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["timestamp", "student", "message", "sentiment"]).to_csv(DATA_FILE, index=False)

# 🔍 Sentiment Analyzer using local transformer
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label']
    return "Positive" if label == "POSITIVE" else "Negative"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    student = data.get("student")
    message = data.get("message")
    timestamp = datetime.now()

    # Analyze sentiment
    sentiment = analyze_sentiment(message)

    # Save data
    new_entry = pd.DataFrame([[timestamp, student, message, sentiment]],
                             columns=["timestamp", "student", "message", "sentiment"])
    new_entry.to_csv(DATA_FILE, mode='a', header=False, index=False)

    # Simple canned bot response (no LLM)
    bot_response = "I'm here for you. Let's talk more about it." if sentiment == "Negative" else "That’s great to hear! How else can I support you?"

    return jsonify({
        "bot_response": bot_response,
        "sentiment": sentiment
    })

@app.route("/at-risk", methods=["GET"])
def at_risk():
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    recent = df[df["timestamp"] > datetime.now() - timedelta(days=2)]
    negatives = recent[recent["sentiment"].str.lower() == "negative"]
    return jsonify({"students_at_risk": negatives["student"].unique().tolist()})

@app.route("/weekly-report", methods=["GET"])
def weekly_report():
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df["week"] = df["timestamp"].dt.to_period("W").apply(lambda r: r.start_time)
    report = df.groupby(["week", "sentiment"]).size().unstack().fillna(0).reset_index()
    return report.to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True)
