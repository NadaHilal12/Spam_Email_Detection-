# 📧 AI Spam Email Detection System

An advanced AI-powered Spam Email Detection web application built using:

- Streamlit
- NLP (Natural Language Processing)
- TF-IDF Vectorization
- SGDClassifier
- Scikit-learn

The system analyzes English emails and predicts whether the email is:

✅ HAM / SAFE  
🚨 SPAM

---

# 🚀 Live Demo

🔗 Streamlit Cloud App:

https://ai-spam-email-detector-engymohamedhanafy.streamlit.app/

---

# 📌 Features

## ✅ Current Features

- Real-time Spam Detection
- NLP Text Preprocessing
- URL & Email Cleaning
- Stopword Removal
- Lemmatization
- TF-IDF Vectorization
- Bigram Feature Extraction
- Confidence Score Visualization
- Suspicious Keyword Detection
- Modern Professional UI
- Fast Prediction Performance

---

# 🌍 Future Improvements

Planned future enhancements include:

- Arabic Spam Detection
- Multi-language Support
- Deep Learning Models (BERT / DistilBERT)
- Phishing URL Detection
- Email Attachment Scanning
- Real-time Inbox Monitoring
- User Feedback Retraining
- AI Explainability
- Cloud API Deployment

---

# 🧠 Machine Learning Pipeline

## NLP Preprocessing

The application performs:

- Lowercasing
- URL Removal
- Email Removal
- Punctuation Cleaning
- Tokenization
- Stopword Removal
- Lemmatization

---

## Feature Engineering

TF-IDF Vectorization with:

- Max Features = 8000
- Bigram Support `(1,2)`
- Sparse Representation

---

## Model

The classifier used:

```python
SGDClassifier(
    loss='hinge',
    alpha=1e-4,
    random_state=42
)
```

Optimized for:

- High-dimensional text data
- Fast inference
- Scalability
- Sparse text classification

---

# 📊 Dataset

Dataset used:

`spam_ham_dataset.csv`

Contains labeled:

- Spam Emails
- Ham (Safe) Emails

---

# ⚠️ Current Limitation

This version currently supports:

✅ English Emails Only

Future versions will support:

- Arabic
- French
- German
- Additional languages

---

# 📂 Project Structure

```bash
project/
│
├── app.py
├── spam_model.pkl
├── tfidf.pkl
├── spam_ham_dataset.csv
├── requirements.txt
└── README.md
```

---

# ▶️ Run Locally

## 1️⃣ Clone Repository

```bash
git clone YOUR_REPOSITORY_LINK
```

---

## 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run Application

```bash
streamlit run app.py
```

---

# ☁️ Streamlit Cloud Deployment

Deploy easily on:

- Streamlit Community Cloud

Steps:

1. Upload project to GitHub
2. Open Streamlit Cloud
3. Connect GitHub repository
4. Select `app.py`
5. Deploy

---

# 📸 Application Preview

You can add screenshots here later.

---

# ⭐ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Matplotlib
- Seaborn

---

# 👩‍💻 Author

Developed by Engy Mohamed
