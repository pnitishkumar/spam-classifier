import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Title
st.title("📩 Spam Email Classifier")

st.write("This app uses Machine Learning to detect spam messages.")

@st.cache_resource
def load_model():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = load_model()

# User input
user_input = st.text_input("Enter a message")

# Predict button
if st.button("Predict"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        result = model.predict(input_vec)

        if result[0] == 1:
            st.error("🚫 This is SPAM")
        else:
            st.success("✅ This is NOT Spam")
    else:
        st.warning("Please enter a message")