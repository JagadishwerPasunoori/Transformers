import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = BertTokenizer.from_pretrained("sentiment_model")

# Streamlit app
st.title("Sentiment Analysis with BERT")
st.write("Enter a movie review to analyze its sentiment.")

# Text input
user_input = st.text_area("Movie Review")

if st.button("Analyze"):
    # Tokenize the input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Display the results
    sentiment = "Positive" if predictions[0][1] > predictions[0][0] else "Negative"
    st.write(f"Sentiment: {sentiment}")