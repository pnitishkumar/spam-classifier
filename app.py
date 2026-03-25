import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 🔧 Clean text function (VERY IMPORTANT)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# 🚀 Load and train model
def load_model():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Convert labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # ✅ Balance dataset (Fix 2)
    spam_df = df[df['label'] == 1]
    ham_df = df[df['label'] == 0]
    ham_df = ham_df.sample(len(spam_df))
    df = pd.concat([spam_df, ham_df])

    # ✅ Add extra spam examples (Fix 1)
    extra_spam = [
        "win money now",
        "free iphone offer",
        "claim your prize now",
        "click here to win",
        "you have won lottery",
        "urgent claim reward",
        "free cash now",
        "limited offer click now"
    ]

    extra_df = pd.DataFrame({
        'label': [1]*len(extra_spam),
        'message': extra_spam
    })

    df = pd.concat([df, extra_df])

    # ✅ Clean text
    df['message'] = df['message'].apply(clean_text)

    # ✅ Vectorizer improvement
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=7000)
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    # ✅ Better model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X, y)

    return vectorizer, model


# 🔥 Load model
vectorizer, model = load_model()

# 🎨 UI
st.title("📩 Spam Email Classifier")
st.write("This app uses Machine Learning to detect spam messages.")

# User input
user_input = st.text_input("Enter a message")

# Predict button
if st.button("Predict"):
    if user_input:
        # ✅ Clean input (VERY IMPORTANT)
        cleaned_input = clean_text(user_input)

        input_vec = vectorizer.transform([cleaned_input])
        result = model.predict(input_vec)[0]
        prob = model.predict_proba(input_vec)[0]

        # ✅ Show result
        if result == 1:
            st.error(f"🚫 This is SPAM (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"✅ This is NOT Spam (Confidence: {prob[0]*100:.2f}%)")
    else:
        st.warning("Please enter a message")