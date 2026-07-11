import gradio as gr
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def analyze_sentiment(text):
    result = sentiment_model(text[:512])[0]
    label = result['label'].lower()
    score = round(result['score'] * 100, 2)
    if 'positive' in label:
        return 'Positive', score, '😊'
    elif 'negative' in label:
        return 'Negative', score, '😞'
    else:
        return 'Neutral', score, '😐'

def sentibot(user_message, history):
    sentiment, score, emoji = analyze_sentiment(user_message)
    response = "Sentiment Analysis Result:\n"
    response += "Sentiment: " + sentiment + " " + emoji + "\n"
    response += "Confidence: " + str(score) + "%"
    return response

demo = gr.ChatInterface(
    fn=sentibot,
    title="SentiBot - Social Media Sentiment Analyzer",
    description="Type any social media post to analyze its sentiment!"
)

demo.launch()
