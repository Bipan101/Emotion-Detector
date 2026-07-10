import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Download NLTK data (only runs once, then cached)
nltk.download('stopwords')
nltk.download('punkt')

# ---- Load saved artifacts ----
@st.cache_resource
def load_artifacts():
    with open('./models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('./models/model_LR.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('./models/emotion_numbers.pkl', 'rb') as f:
        emotion_numbers = pickle.load(f)
    return vectorizer, model, emotion_numbers

tfidf_vectorizer, model_LR, emotion_numbers = load_artifacts()

# Reverse mapping: number -> emotion label
number_to_emotion = {v: k for k, v in emotion_numbers.items()}

stop_words = set(stopwords.words('english'))

# ---- Preprocessing (must match training exactly) ----
def remove_punc(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])

def remove_emojis(text):
    return ''.join([i for i in text if i.isascii()])

def remove_stopwords(text):
    words = text.split()
    cleaned = [w for w in words if w not in stop_words]
    return ' '.join(cleaned)

def preprocess(text):
    text = text.lower()
    text = remove_punc(text)
    text = remove_numbers(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    return text

# ---- Streamlit UI ----
st.set_page_config(page_title="Emotion Detector", page_icon="🎭")

st.title("🎭 Emotion Detection from Text")
st.write("Enter a sentence and the model will predict the emotion behind it.")

user_input = st.text_area("Type your text here:", height=120, placeholder="e.g. I am so happy today!")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned_text = preprocess(user_input)
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        prediction = model_LR.predict(vectorized_text)[0]
        predicted_emotion = number_to_emotion[prediction]

        st.success(f"Predicted Emotion: **{predicted_emotion.upper()}**")

        # Optional: show prediction probabilities
        if hasattr(model_LR, "predict_proba"):
            probs = model_LR.predict_proba(vectorized_text)[0]
            st.subheader("Confidence Scores")
            prob_dict = {number_to_emotion[i]: float(p) for i, p in enumerate(probs)}
            st.bar_chart(prob_dict)

st.markdown("---")
st.caption("Model: Logistic Regression + TF-IDF")