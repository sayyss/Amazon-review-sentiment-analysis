#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
import pandas as pd
import re
import time
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('sentiwordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def load_fasttext_data(file_path, test_size=0.2, max_samples=None, random_state=42):
    
    """Loads fastText-format sentiment data, converts it to a DataFrame, and splits into training and test sets."""
    reviews = []
    labels = []
    count = 0
    print(f"Loading data from {file_path}...")
    start_time = time.time()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if max_samples and count >= max_samples:
                break
            match = re.match(r'__label__(\d+)\s+(.*)', line)
            if match:
                label = int(match.group(1))
                text = match.group(2)
                sentiment = 'positive' if label == 2 else 'negative'
                reviews.append(text)
                labels.append(sentiment)
                count += 1
                if count % 100000 == 0:
                    print(f"Loaded {count} reviews...")
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': labels
    })
    elapsed_time = time.time() - start_time
    print(f"Loaded {len(df)} reviews in {elapsed_time:.2f} seconds")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['sentiment']
    )
    print(f"Split data into {len(train_df)} training samples and {len(test_df)} test samples")
    return train_df, test_df
    
def preprocess_text(text):
    
    """Preprocesses text: lowercases, removes special characters, tokenizes, removes stopwords, and lemmatizes."""
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)
    
def penn_to_wn(tag):
    """Converts Penn Treebank POS tags to WordNet-compatible POS tags."""
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    elif tag.startswith('V'):
        return 'v'
    return None
def sentiwordnet_analysis(text):
    """Analyzes sentiment of text using SentiWordNet to return positivity, negativity, objectivity, and compound score."""
    from nltk.corpus import sentiwordnet as swn
    if pd.isna(text) or text == '':
        return {"pos": 0, "neg": 0, "obj": 0, "compound": 0}
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_score = 0
    neg_score = 0
    token_count = 0
    for word, tag in pos_tags:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in ('a', 'n', 'r', 'v'):
            continue
        synsets = list(swn.senti_synsets(word, pos=wn_tag))
        if not synsets:
            continue
        synset = synsets[0]
        pos_score += synset.pos_score()
        neg_score += synset.neg_score()
        token_count += 1
    if token_count > 0:
        pos_score = pos_score / token_count
        neg_score = neg_score / token_count
    obj_score = 1 - (pos_score + neg_score)
    compound_score = pos_score - neg_score
    return {
        "pos": pos_score,
        "neg": neg_score,
        "obj": obj_score,
        "compound": compound_score
    }
class SentiWordNetTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that extracts SentiWordNet sentiment scores for each review."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features = []
        total = len(X)
        print(f"Extracting SentiWordNet features for {total} reviews...")
        start_time = time.time()
        for i, text in enumerate(X):
            scores = sentiwordnet_analysis(text)
            features.append([
                scores['pos'],
                scores['neg'],
                scores['obj'],
                scores['compound']
            ])
            if (i+1) % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i+1}/{total} reviews... ({elapsed:.2f}s)")
        print(f"SentiWordNet feature extraction completed in {time.time() - start_time:.2f} seconds")
        return np.array(features)
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer that extracts basic text features like length, word count, and punctuation usage."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features = []
        total = len(X)
        print(f"Extracting text features for {total} reviews...")
        start_time = time.time()
        for i, text in enumerate(X):
            if pd.isna(text) or text == '':
                features.append([0, 0, 0, 0, 0])
                continue
            tokens = word_tokenize(text.lower())
            text_length = len(text)
            token_count = len(tokens)
            avg_token_length = sum(len(token) for token in tokens) / max(token_count, 1)
            exclamation_count = text.count('!')
            question_count = text.count('?')
            features.append([
                text_length,
                token_count,
                avg_token_length,
                exclamation_count,
                question_count
            ])
            if (i+1) % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i+1}/{total} reviews... ({elapsed:.2f}s)")
        print(f"Text feature extraction completed in {time.time() - start_time:.2f} seconds")
        return np.array(features)
        
def create_hybrid_model():
    """Builds a hybrid sentiment classifier combining TF-IDF, SentiWordNet, and text features using Logistic Regression."""
    tfidf = Pipeline([
        ('tfidf', TfidfVectorizer(
            min_df=5, max_df=0.95,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=10000))
    ])
    senti_features = Pipeline([
        ('sentiwordnet', SentiWordNetTransformer())
    ])
    text_features = Pipeline([
        ('textfeatures', TextFeatureExtractor())
    ])
    combined_features = FeatureUnion([
        ('tfidf_features', tfidf),
        ('sentiwordnet_features', senti_features),
        ('text_features', text_features)
    ])
    model = Pipeline([
        ('features', combined_features),
        ('classifier', LogisticRegression(C=10, max_iter=1000))
    ])
    return model
    
def train_and_evaluate_full_sets(train_path, test_size, max_train=None):
    """Trains the hybrid model on training data and evaluates its performance on the test set."""
    
    train_df, test_df = load_fasttext_data(train_path, test_size, max_samples=max_train)
    
    print(f"\nTraining set shape: {train_df.shape}")
    print(f"Training sentiment distribution: {train_df['sentiment'].value_counts().to_dict()}")
    print(f"\nTest set shape: {test_df.shape}")
    print(f"Test sentiment distribution: {test_df['sentiment'].value_counts().to_dict()}")
    print("\nPreprocessing training data...")
    
    start_time = time.time()
    train_df['processed_review'] = train_df['review'].apply(preprocess_text)
    
    print(f"Training preprocessing completed in {time.time() - start_time:.2f} seconds")
    print("\nPreprocessing test data...")
    
    start_time = time.time()
    test_df['processed_review'] = test_df['review'].apply(preprocess_text)
    
    print(f"Test preprocessing completed in {time.time() - start_time:.2f} seconds")
    print("\nCreating and training the model...")
    
    start_time = time.time()
    model = create_hybrid_model()
    model.fit(train_df['processed_review'], train_df['sentiment'])
    
    print(f"Model training completed in {time.time() - start_time:.2f} seconds")
    print("\nEvaluating on test set...")
    start_time = time.time()
    test_predictions = model.predict(test_df['processed_review'])
    elapsed = time.time() - start_time
    
    print(f"Evaluation completed in {elapsed:.2f} seconds ({len(test_df) / elapsed:.2f} reviews/second)")
    
    accuracy = accuracy_score(test_df['sentiment'], test_predictions)
    print(f"\nFull Test Set Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_df['sentiment'], test_predictions))
    conf_matrix = confusion_matrix(test_df['sentiment'], test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Full Test Set')
    plt.tight_layout()
    plt.show()
    
    test_df['predicted_sentiment'] = test_predictions
    test_df['correct'] = test_df['sentiment'] == test_df['predicted_sentiment']
    print("\nSample of Misclassified Positive Reviews:")
    
    misclassified_pos = test_df[(test_df['sentiment'] == 'positive') & (test_df['predicted_sentiment'] == 'negative')]
    if not misclassified_pos.empty:
        for i, row in misclassified_pos.head(3).iterrows():
            print(f"Review: {row['review']}")
            scores = sentiwordnet_analysis(row['processed_review'])
            print(f"SentiWordNet Scores - Pos: {scores['pos']:.4f}, Neg: {scores['neg']:.4f}, Compound: {scores['compound']:.4f}")
            print("-" * 80)
            
    print("\nSample of Misclassified Negative Reviews:")
    misclassified_neg = test_df[(test_df['sentiment'] == 'negative') & (test_df['predicted_sentiment'] == 'positive')]
    
    if not misclassified_neg.empty:
        for i, row in misclassified_neg.head(3).iterrows():
            
            print(f"Review: {row['review']}")
            scores = sentiwordnet_analysis(row['processed_review'])
            print(f"SentiWordNet Scores - Pos: {scores['pos']:.4f}, Neg: {scores['neg']:.4f}, Compound: {scores['compound']:.4f}")
            print("-" * 80)
            
    return model, test_df
    
def sentiwordnet_only_analysis(df):
    """Performs sentiment classification using only SentiWordNet compound scores, without a machine learning model."""
    
    print("\nRunning SentiWordNet-only analysis...")
    if 'processed_review' not in df.columns:
        print("Preprocessing reviews...")
        df['processed_review'] = df['review'].apply(preprocess_text)
    print("Calculating SentiWordNet scores...")
    scores = []
    for i, text in enumerate(df['processed_review']):
        score = sentiwordnet_analysis(text)
        scores.append(score)
        if (i+1) % 5000 == 0:
            print(f"Processed {i+1}/{len(df)} reviews...")
    df['pos_score'] = [score['pos'] for score in scores]
    df['neg_score'] = [score['neg'] for score in scores]
    df['obj_score'] = [score['obj'] for score in scores]
    df['compound_score'] = [score['compound'] for score in scores]
    df['sentiwordnet_prediction'] = df['compound_score'].apply(
        lambda score: 'positive' if score >= 0 else 'negative'
    )
    accuracy = (df['sentiment'] == df['sentiwordnet_prediction']).mean()
    print(f"\nSentiWordNet-only Accuracy: {accuracy:.4f}")
    print("\nSentiWordNet-only Classification Report:")
    print(classification_report(df['sentiment'], df['sentiwordnet_prediction']))
    conf_matrix = confusion_matrix(df['sentiment'], df['sentiwordnet_prediction'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Positive'],
               yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - SentiWordNet Only')
    plt.tight_layout()
    plt.show()
    return df
    
def main():
    
    """Main function to run the hybrid model and compare model vs. lexicon-only performance."""
    train_path = 'new_train.ft.txt'
    max_train = None
    model, test_df = train_and_evaluate_full_sets(
        train_path,
        0.3,
        max_train
    )
    
    sentiwordnet_results = sentiwordnet_only_analysis(test_df)
    hybrid_accuracy = (test_df['sentiment'] == test_df['predicted_sentiment']).mean()
    sentiwordnet_accuracy = (test_df['sentiment'] == test_df['sentiwordnet_prediction']).mean()
    
    print("\nPerformance Comparison:")
    print(f"Hybrid Model Accuracy: {hybrid_accuracy:.4f}")
    print(f"SentiWordNet-only Accuracy: {sentiwordnet_accuracy:.4f}")
    print(f"Improvement: {(hybrid_accuracy - sentiwordnet_accuracy) * 100:.2f}%")
    
if __name__ == "__main__":
    main()


# In[ ]:




