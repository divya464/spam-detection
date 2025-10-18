import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -----------------------------
# Setup
# -----------------------------
nltk.download('stopwords')
stop_words = stopwords.words('english')

st.set_page_config(page_title="Spam Mail Detector", page_icon="📧")
st.title("📧 Spam Mail Detection App")
st.write("Upload your dataset and test messages to detect spam emails.")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV dataset with columns 'text' and 'label'", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("Preview of your dataset:")
        st.dataframe(df.head())

        # Check required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            st.error("CSV must have 'text' and 'label' columns!")
        else:
            # -----------------------------
            # Text preprocessing
            # -----------------------------
            def clean_text(text):
                text = str(text).lower()
                text = re.sub(r"http\S+", " ", text)           # remove urls
                text = re.sub(r"\S+@\S+\.\S+", " ", text)      # remove emails
                text = re.sub(r"[^a-z0-9\s]", " ", text)       # keep alphanumeric
                text = re.sub(r"\s+", " ", text).strip()
                return text

            df['text_clean'] = df['text'].apply(clean_text)
            st.write("✅ Text cleaned.")

            # -----------------------------
            # Train/Test split
            # -----------------------------
            X = df['text_clean']
            y = df['label']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(df['label'].unique())>1 else None)

            # -----------------------------
            # Train pipeline
            # -----------------------------
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words, max_df=0.9, ngram_range=(1,2))),
                ('clf', MultinomialNB())
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"✅ Model trained! Accuracy on test set: **{acc*100:.2f}%**")

            # -----------------------------
            # Interactive prediction
            # -----------------------------
            st.subheader("Test a new email/message")
            user_input = st.text_area("Enter email text here:")

            if st.button("Predict"):
                if user_input.strip() == "":
                    st.warning("Please enter a message!")
                else:
                    text_cleaned = clean_text(user_input)
                    pred = pipeline.predict([text_cleaned])[0]
                    prob = pipeline.predict_proba([text_cleaned])[0]
                    st.write(f"**Prediction:** {pred.upper()}")
                    st.write(f"**Probability:** Spam: {prob[pipeline.classes_ == 'spam'][0]:.2f}, Ham: {prob[pipeline.classes_ == 'ham'][0]:.2f}")

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("📌 Upload a CSV file to start training the spam detector.")
    st.write("Your CSV must have two columns:")
    st.write("- `text` → the email content")
    st.write("- `label` → 'spam' or 'ham'")
