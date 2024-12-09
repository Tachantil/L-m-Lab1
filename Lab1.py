import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

positive_texts = open("positive-words.txt").read().splitlines()
negative_texts = open("negative-words.txt").read().splitlines()

def preprocess_text(texts):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    preprocessed_texts = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        preprocessed_texts.append(" ".join(stemmed_tokens))
    return preprocessed_texts

positive_texts = preprocess_text(positive_texts)
negative_texts = preprocess_text(negative_texts)

positive_freq = Counter(word_tokenize(" ".join(positive_texts)))
negative_freq = Counter(word_tokenize(" ".join(negative_texts)))

print("ТОП-5 позитивних слів:", positive_freq.most_common(5))
print("ТОП-5 негативних слів:", negative_freq.most_common(5))

texts = positive_texts + negative_texts
labels = [1]*len(positive_texts) + [0]*len(negative_texts)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=300, solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Точність моделі:", accuracy_score(y_test, y_pred))
print("Звіт про класифікацію:\n", classification_report(y_test, y_pred))

#гіперпараметри
def evaluate_hyperparameters():
    print("\n=== Дослідження гіперпараметрів ===\n")

    for C in np.logspace(-3, 3, 7):
        model = LogisticRegression(C=C, max_iter=300, solver='liblinear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\nПараметр C={C}")
        print("Точність:", accuracy_score(y_test, y_pred))
        print("Звіт про класифікацію:\n", classification_report(y_test, y_pred))

    for max_iter in [100, 200, 300, 500]:
        model = LogisticRegression(C=1.0, max_iter=max_iter, solver='liblinear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\nКількість ітерацій: {max_iter}")
        print("Точність:", accuracy_score(y_test, y_pred))
        print("Звіт про класифікацію:\n", classification_report(y_test, y_pred))

    #перевірка оптимізації
    for solver in ['liblinear', 'saga', 'lbfgs']:
        model = LogisticRegression(C=1.0, max_iter=300, solver=solver)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\nМетод solver: {solver}")
        print("Точність:", accuracy_score(y_test, y_pred))
        print("Звіт про класифікацію:\n", classification_report(y_test, y_pred))

    #балансування 
    model = LogisticRegression(C=1.0, max_iter=300, solver='liblinear', class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nТочність з балансуванням класів:", accuracy_score(y_test, y_pred))
    print("Звіт про класифікацію:\n", classification_report(y_test, y_pred))

evaluate_hyperparameters()
