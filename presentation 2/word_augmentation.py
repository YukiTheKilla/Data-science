import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import movie_reviews, stopwords, wordnet
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('movie_reviews')
nltk.download('wordnet')
nltk.download('omw-1.4')

texts = [" ".join(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()]
labels = [fileid.split('/')[0] for fileid in movie_reviews.fileids()]  # 'pos' или 'neg'

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

def add_synonyms(text):
    tokens = word_tokenize(text)
    new_tokens = []
    for word in tokens:
        synsets = wordnet.synsets(word)
        if synsets:
            synonyms = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
            if synonyms:
                new_tokens.append(random.choice(list(synonyms)))
            else:
                new_tokens.append(word)
        else:
            new_tokens.append(word)
    return " ".join(new_tokens)

def augment_data(texts, labels):
    augmented_texts = []
    augmented_labels = []
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        text_no_stopwords = remove_stopwords(text)
        augmented_texts.append(text_no_stopwords)
        augmented_labels.append(label)
        
        text_with_synonyms = add_synonyms(text)
        augmented_texts.append(text_with_synonyms)
        augmented_labels.append(label)
        
    return augmented_texts, augmented_labels

X_train_augmented, y_train_augmented = augment_data(X_train, y_train)

vectorizer = CountVectorizer()
vectorizer.fit(X_train_augmented + X_test)

X_train_vectorized = vectorizer.transform(X_train_augmented)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train_augmented)
y_pred = model.predict(X_test_vectorized)
accuracy_with_augmentation = accuracy_score(y_test, y_pred)

X_train_vectorized_no_aug = vectorizer.transform(X_train)
X_test_vectorized_no_aug = vectorizer.transform(X_test)

model_no_aug = MultinomialNB()
model_no_aug.fit(X_train_vectorized_no_aug, y_train)
y_pred_no_aug = model_no_aug.predict(X_test_vectorized_no_aug)
accuracy_no_augmentation = accuracy_score(y_test, y_pred_no_aug)

print(f"Точность с аугментацией: {accuracy_with_augmentation:.4f}")
print(f"Точность без аугментации: {accuracy_no_augmentation:.4f}")

labels = ['Без аугментации', 'С аугментацией']
accuracies = [accuracy_no_augmentation, accuracy_with_augmentation]

plt.figure(figsize=(8, 6))
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylabel('Точность')
plt.title('Влияние аугментации текстовых данных на точность модели')
plt.ylim(0.5, 1)
plt.show()