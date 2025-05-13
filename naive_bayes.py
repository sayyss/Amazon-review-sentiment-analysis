import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# define the list of negative words
negations = {'not', 'no', 'none'}

def handle_negation(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    result_tokens = []

    i = 0
    while i < len(tagged_tokens):
        word, tag = tagged_tokens[i]
        lower_word = word.lower()

        if lower_word in negations:
            negation = lower_word
            result_tokens.append(negation)  

            for j in range(1, 3): 
                if i + j >= len(tagged_tokens):
                    break
                next_word, next_tag = tagged_tokens[i + j]
                if next_tag.startswith(('JJ', 'VB')): 
                    result_tokens.append(next_word + "_NEG")  
                    i += j
                    break
            else:
                i += 1
                continue
        else:
            result_tokens.append(word)
        i += 1

    return ' '.join(result_tokens)

def load_fasttext_data(filename):
    texts = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                label, text = parts
                if '__label__1' in label:
                    sentiment = 'negative'
                elif '__label__2' in label:
                    sentiment = 'positive'
                else:
                    continue
                texts.append(text)
                labels.append(sentiment)
    return pd.DataFrame({'text': texts, 'sentiment': labels})

df = load_fasttext_data("new_train.ft.txt")

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()                     
    text = re.sub(r'[^a-z\s]', '', text)    
    
    text = handle_negation(text)
    
    words = text.split()
    filtered_words = [word for word in words 
                     if word not in stop_words or word.endswith('_NEG')]
    
    return ' '.join(filtered_words)

df['cleaned_text'] = df['text'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['sentiment'], test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print("Classification Result:")
print(classification_report(y_test, y_pred))


results_df = pd.DataFrame({
    'text': X_test,  
    'true_label': y_test,
    'predicted_label': y_pred
})

# find out the rows that are predicted wrongly
mistakes = results_df[results_df['true_label'] != results_df['predicted_label']]

# printout wrong result
print("wrong result：")
print(mistakes[['text', 'true_label', 'predicted_label']])

import numpy as np

# obtain the word list
feature_names = vectorizer.get_feature_names_out()

# obtain the log probability of each word for each category
# model.feature_log_prob_  shape (n_classes, n_features)
log_probs = model.feature_log_prob_

#for each incorrect comment, list the keywordsd tha most affect the model's prediction
def find_top_words(text):
    vec = vectorizer.transform([text])
    vec_array = vec.toarray()[0]

    word_indices = np.where(vec_array > 0)[0]

    words = []
    for idx in word_indices:
        word = feature_names[idx]
        # take the log probility difference of this word
        contribution = log_probs[:, idx]
        words.append((word, contribution))

    # order it 
    words.sort(key=lambda x: np.abs(x[1][1] - x[1][0]), reverse=True)
    return words

example_text = mistakes.iloc[0]['text']  # the first incorrect example
top_words = find_top_words(example_text)

print("The word that the model cares about most in this comment is：")
for word, contrib in top_words[:10]:  
    print(f"word: {word}, contributions：{contrib}")

print(mistakes.iloc[0]['text'])