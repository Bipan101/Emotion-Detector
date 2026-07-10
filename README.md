# Emotion Detection from Text using NLP & Machine Learning

A machine learning project that classifies the emotion expressed in a piece of text (e.g., *joy, sadness, anger, fear, love, surprise*) using classical NLP preprocessing and ML classifiers. The final model is deployed as an interactive **Streamlit web app**.

---

## 📌 Project Overview

This project takes raw text data, cleans and preprocesses it using standard NLP techniques, converts it into numerical features using **Bag of Words** and **TF-IDF**, and trains multiple machine learning models to predict the emotion behind the text. The best-performing model is then saved and used to power a simple web UI where users can type a sentence and get an instant emotion prediction.

---

## 🗂️ Dataset

- **File:** `train.txt`
- **Format:** Semicolon-separated (`;`) text file with two columns — `text` and `emotion`
- Each row contains a sentence and its corresponding emotion label.

---

## 🧹 Text Preprocessing Pipeline

The raw text was cleaned using the following steps, in order:

1. **Lowercasing** – Converts all text to lowercase for consistency.
2. **Removing Punctuation** – Strips punctuation marks using Python's `string` module.
3. **Removing Numbers** – Removes digit characters from the text.
4. **Removing Emojis & Special Characters** – Keeps only ASCII characters.
5. **Removing Stopwords** – Removes common English stopwords (e.g., "the", "is", "and") using NLTK's stopword list.

After cleaning, the text was split into **training (80%)** and **testing (20%)** sets using `train_test_split` with `random_state=42`.

---

## 🔢 Feature Extraction

Two vectorization techniques were used to convert text into numerical form:

| Technique | Description |
|---|---|
| **Bag of Words (CountVectorizer)** | Represents text as raw word frequency counts. |
| **TF-IDF (TfidfVectorizer)** | Weighs words by importance, reducing the influence of frequently occurring but less informative words. |

---

## 🤖 Models Trained & Results

Three model + feature combinations were trained and evaluated on accuracy:

| Model | Feature Extraction | Accuracy |
|---|---|---|
| Multinomial Naive Bayes | Bag of Words (CountVectorizer) | **76.81%** |
| Multinomial Naive Bayes | TF-IDF | **66.09%** |
| **Logistic Regression** | **TF-IDF** | **🏆 86.28%** |

### 🏆 Best Model: Logistic Regression + TF-IDF
Logistic Regression combined with TF-IDF features achieved the highest accuracy of **86.28%**, outperforming both Naive Bayes variants. This is the model used in the deployed Streamlit application.

**Why did Logistic Regression + TF-IDF perform best?**
- TF-IDF down-weights common, less-informative words while emphasizing distinctive ones.
- Logistic Regression can learn more nuanced decision boundaries between classes compared to Naive Bayes' independence assumption, which tends to oversimplify relationships between words.

---

## 📁 Project Structure

```
├── app.py                    # Streamlit web app for live predictions
├── train.txt                 # Dataset (text;emotion pairs)
├── requirements.txt           # Python dependencies
├── README.md                   # Project documentation
├── models/
│   ├── model_LR.pkl            # Saved trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer (fitted on training data)
│   └── emotion_numbers.pkl      # Saved label mapping (emotion name <-> number)
└── notebooks/
    └── NLP-with-ML.ipynb        # Main notebook: preprocessing, training, evaluation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Bipan101/Emotion-Detector.git
   cd Emotion-Detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in your terminal (usually `http://localhost:8501`) in your browser.

---

## 🖥️ How the App Works

1. User types a sentence into the text box.
2. The input goes through the **same preprocessing pipeline** used during training (lowercasing, punctuation/number/emoji removal, stopword removal).
3. The cleaned text is vectorized using the saved **TF-IDF vectorizer**.
4. The saved **Logistic Regression model** predicts the emotion class.
5. The predicted label is decoded back to its original emotion name and displayed, along with a confidence score bar chart.

---

## 🛠️ Tech Stack

- **Python 3**
- **Pandas / NumPy** – Data handling
- **NLTK** – Stopword removal, tokenization
- **Scikit-learn** – Vectorization (CountVectorizer, TfidfVectorizer), models (MultinomialNB, LogisticRegression), evaluation
- **Streamlit** – Web app interface
- **Matplotlib / Seaborn** – Visualization (EDA)

---

## 📈 Future Improvements

- Experiment with more advanced models (SVM, Random Forest, XGBoost).
- Try word embeddings (Word2Vec, GloVe) or transformer-based models (BERT) for potentially higher accuracy.(But i did NLP with ML here so i didn't use those as they are of DL thingy)
- Add cross-validation and hyperparameter tuning (e.g., `GridSearchCV`) for more robust model selection.
- Handle class imbalance if present in the emotion labels.
- Deploy the app publicly via Streamlit Community Cloud or Hugging Face Spaces.

---

## 👤 Author

**Bipan Neupane**
- GitHub: [@Bipan101](https://github.com/Bipan101)
- Portfolio: [bipanneupane.com.np](https://bipanneupane.com.np)

---

## 📄 License

This project is open source and available under the [MIT License](https://github.com/Bipan101/Emotion-Detector/blob/Main/LICENSE).