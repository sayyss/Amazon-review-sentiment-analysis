#install imports

import pandas as pd
import numpy as np
import nltk
import time
import re
import bz2
import os
import math

# scikitlearn libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# for plotting results
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
def load_fasttext_data(file_path, max_samples=None, encoding='utf-8'):
    """
    Load FastText-format data into a DataFrame of raw texts + 'Positive'/'Negative' labels.
    """
    reviews, labels = [], []
    count = 0
    start_time = time.time()
    print(f"Loading up to {max_samples or 'all'} samples from {file_path} ...")

    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            if max_samples and count >= max_samples:
                break
            m = re.match(r'__label__(\d+)\s+(.*)', line)
            if not m:
                continue
            lbl = int(m.group(1))
            text = m.group(2)
            sentiment = 'Positive' if lbl == 2 else 'Negative'
            reviews.append(text)
            labels.append(sentiment)
            count += 1
            if count % 100000 == 0:
                print(f"  ⋮ loaded {count} samples")

    df = pd.DataFrame({'Review': reviews, 'Sentiment': labels})
    print(f"Loaded {len(df)} samples in {time.time() - start_time:.1f}s")
    return df

# remove stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalpha() and t not in stop_words
    ]
    return " ".join(tokens)
train_file = 'new_train.ft.txt'    
MAX_SAMPLES = 100000          

df = load_fasttext_data(train_file, max_samples=MAX_SAMPLES)
df['Processed'] = df['Review'].apply(preprocess_text)
df = df[df['Processed'].str.strip().astype(bool)]  # drop empty results

print(df['Sentiment'].value_counts(), "\n")
print("Example processed review:\n", df['Processed'].iloc[0])

X = df['Processed']
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("\n Running TF-IDF + Logistic Regression…")
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('clf',  LogisticRegression(max_iter=1000))
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

print("LogReg Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test,  y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
# Create figures, Figure 1- Confusion matrix HeatMap
y_pred = lr_pipeline.predict(X_test) 
cm = confusion_matrix(y_test, y_pred, labels=['Negative','Positive'])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=['Neg','Pos'], 
            yticklabels=['Neg','Pos'],
            cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (70/30 split)')
plt.tight_layout()
plt.show()
# Figure 2- Top Words for positive and negative reviews
vectorizer = lr_pipeline.named_steps['tfidf']
clf        = lr_pipeline.named_steps['clf']

feat = vectorizer.get_feature_names_out()
coef = clf.coef_[0]

N = 15
# 1) most-negative first → less-negative
neg_idx = np.argsort(coef)[:N]
# 2) least-positive first → most-positive last
pos_idx = np.argsort(coef)[-N:]

idxs   = np.concatenate([neg_idx, pos_idx])
labels = feat[idxs]
values = coef[idxs]
colors = ['red']*N + ['blue']*N

plt.figure(figsize=(12,5))
plt.bar(range(2*N), values, color=colors)
plt.axhline(0, color='black', linewidth=0.8)

# draw the x-ticks
plt.xticks(range(2*N), labels, rotation=45, ha='right')
plt.ylabel('Coefficient (log-reg)')
plt.title('Top Words for Positive and Negative Reviews')
plt.tight_layout()
plt.show()