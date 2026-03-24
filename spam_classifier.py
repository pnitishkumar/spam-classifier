import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(stop_words='english')

df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())

df['label'] = df['label'].map({'ham': 0, 'spam': 1})


X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predictions
nb_pred = nb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Accuracy
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

from sklearn.metrics import confusion_matrix

print("Confusion Matrix (Naive Bayes):")
print(confusion_matrix(y_test, nb_pred))

msg = ["Congratulations! You won a free ticket"]

msg_vec = vectorizer.transform(msg)
prediction = nb_model.predict(msg_vec)

print("Spam" if prediction[0] == 1 else "Not Spam")