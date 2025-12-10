 Fake Review Detection System (NLP + Machine Learning)

This project is an AI-powered Fake Review Detection System that uses Natural Language Processing (NLP) and Machine Learning to identify whether a product review is Genuine or Fake.
It uses TF-IDF Vectorization and a Multinomial Naive Bayes classifier to perform accurate text classification.

🚀 Features

Detects Fake vs Genuine product reviews

NLP preprocessing (lowercasing, stopword removal, punctuation removal)

TF-IDF based feature extraction

Naive Bayes classification

Built-in mini dataset

Allows custom review prediction

Model evaluation with accuracy and classification report

📂 Project Structure
📁 Fake-Review-Detection
│
├── fake_review_detection.py   # Main project code
├── README.md                  # Project documentation
└── requirements.txt           # Required libraries (optional)

🛠 Technologies Used

Python

Pandas

NLTK

Scikit-Learn

TF-IDF Vectorizer

Multinomial Naive Bayes

📘 How It Works

Load dataset (embedded manually)

Preprocess review text

Split into training and testing sets

Convert text to TF-IDF vectors

Train Naive Bayes Model

Evaluate model

Predict new reviews as Genuine or Fake

🧪 Sample Prediction
Review: "Fake item received. Don't buy."
Output: Fake

Review: "Excellent quality and fast delivery!"
Output: Genuine

📊 Model Evaluation (Example)

Accuracy Score

Precision, Recall, F1-score

Classification Report

🔧 Setup Instructions
pip install pandas nltk scikit-learn


Run the script:

python fake_review_detection.py

📌 Future Improvements

Use a larger dataset

Add logistic regression/SVM models

Deploy using Flask/Streamlit

Integrate deep learning (LSTM/BERT)
