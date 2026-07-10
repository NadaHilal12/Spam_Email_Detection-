# Import Libraries

import streamlit as st
import pandas as pd
import nltk
import joblib
import re
import time

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Page Config

st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="wide"
)

# Custom Styling

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1, h2, h3 {
    color: white;
}

.stTextArea textarea {
    font-size: 16px;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# Download NLTK Resources

nltk.download('stopwords')
nltk.download('wordnet')

# Load Trained Model

@st.cache_resource
def load_model():

    model = joblib.load("spam_model.pkl")

    vectorizer = joblib.load("tfidf.pkl")

    return model, vectorizer

model, vectorizer = load_model()

# NLP Setup

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# Text Preprocessing

def preprocess_text(text):

    if pd.isna(text):
        return ""

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r'http\\S+|www\\S+', ' ', text)

    # remove emails
    text = re.sub(r'\\S+@\\S+', ' ', text)

    # remove punctuation & numbers
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)

    # tokenize
    tokens = text.split()

    # stopwords + lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return ' '.join(tokens)

# Sidebar

with st.sidebar:

    st.title("📌 About Project")

    st.markdown("---")

    st.info("""
    This application uses:

    - NLP Preprocessing
    - TF-IDF Vectorization
    - Bigram Feature Extraction
    - SGDClassifier

    Built for fast and accurate spam detection.
    """)

    st.markdown("---")

    st.subheader("🌍 Future Expansion")

    st.warning("""
    Planned Features:

    - Arabic Spam Detection
    - Multi-language Support
    - Deep Learning Models
    - Real-Time Email Monitoring
    - Phishing Detection
    - Attachment Scanning
    """)

    st.markdown("---")

    st.info("""
            ⚠️ Current Version Notice

            This application currently supports English emails only.

            Multilingual support including Arabic, French, and German
            will be added in future versions.
            """)

# Main Header

st.title("📧 AI Spam Email Detection System")

st.markdown("""
### Advanced NLP-Powered Email Classification

Analyze emails instantly using Machine Learning and Natural Language Processing.
""")

# Tabs

tab1, tab2 = st.tabs([
    "📨 Email Analyzer",
    "🧠 About AI Model"
])

# TAB 1 — Email Analyzer

with tab1:

    email_input = st.text_area(
        "✉️ Paste Email Content",
        height=300,
        placeholder="Enter email content here..."
    )

    predict_btn = st.button("🔍 Analyze Email")

    if predict_btn:

        if email_input.strip() == "":

            st.warning("Please enter email text.")

        else:

            with st.spinner("Analyzing Email..."):

                time.sleep(1)

                processed_email = preprocess_text(email_input)

                email_vector = vectorizer.transform([processed_email])

                prediction = model.predict(email_vector)[0]

                score = model.decision_function(email_vector)[0]

                confidence = abs(score)

            st.markdown("---")

            # Prediction Result

            if prediction == 1:

                st.error("🚨 SPAM EMAIL DETECTED")

            else:

                st.success("✅ SAFE / HAM EMAIL")

            # Confidence Score

            confidence_percent = min(
                confidence * 10,
                99
            )

            st.progress(confidence_percent / 100)

            st.write(
                f"### Confidence Score: {confidence_percent:.1f}%"
            )

            # Email Statistics

            st.markdown("---")

            st.subheader("📊 Email Statistics")

            words = len(email_input.split())

            characters = len(email_input)

            suspicious_keywords = [
                "free",
                "win",
                "offer",
                "money",
                "urgent",
                "click",
                "bonus",
                "limited",
                "prize"
            ]

            detected = [
                word for word in suspicious_keywords
                if word in email_input.lower()
            ]

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Words", words)

            with c2:
                st.metric("Characters", characters)

            with c3:
                st.metric(
                    "Suspicious Keywords",
                    len(detected)
                )

            if detected:

                st.warning(
                    f"Detected suspicious words: {', '.join(detected)}"
                )

# TAB 2 — About Model

with tab2:

    st.subheader("🧠 AI Architecture")

    st.write("""
    This system was built using:

    ### NLP Pipeline
    - Text Cleaning
    - Stopword Removal
    - Lemmatization
    - TF-IDF Vectorization
    - Bigram Feature Extraction

    ### Machine Learning
    - SGDClassifier
    - Linear Classification
    - Sparse Feature Optimization

    ### Why This Model?
    SGDClassifier performs exceptionally well for:
    - Text Classification
    - Spam Detection
    - High-dimensional text data
    """)

    st.markdown("---")

    st.subheader("⚡ System Highlights")

    st.success("""
    - Fast Real-Time Prediction
    - Optimized NLP Pipeline
    - Scalable Architecture
    - Future Multilingual Support
    - Production-Ready Design
    """)

# Footer

st.markdown("---")

st.caption("""
AI Spam Detection System — Built with Streamlit, NLP, TF-IDF, and SGDClassifier.

Designed for scalability and future multilingual support.
""")
