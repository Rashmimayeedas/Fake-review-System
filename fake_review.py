# All-in-One Fake Review Detection System

# Step 1: Import libraries
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 2: Define mini dataset (embedded)
data = {
    'text': [
        'Great product! Fast delivery and works well.',
        'Worst thing I ever bought. Total scam.',
        'I loved it. Highly recommended!',
        'Scam alert! This is a fake product.',
        'Super quality and perfect service.',
        'Totally disappointed. This is fraud.',
        'Very satisfied with the product.',
        'Received a broken item. Fake seller!',
        'Five stars. Will order again!',
        'This is not what I ordered. Fake!',
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = genuine, 1 = fake
}

df = pd.DataFrame(data)

# Step 3: Preprocess text
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess)

# Step 4: Split dataset
X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Evaluate model
y_pred = model.predict(X_test_vec)
print("\n Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Predict new review
def predict_review(review):
    cleaned = preprocess(review)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)
    return " Genuine" if result[0] == 0 else " Fake"

# Step 9: Try with custom inputs
print("\n Sample Predictions:")
samples = [
    "Excellent quality and on-time delivery!",
    "Fake item received. Don't buy from here.",
    "Product as described. Happy with purchase.",
    "Total fraud. Waste of money!"
]

for review in samples:
    print(f"Review: '{review}' → {predict_review(review)}")