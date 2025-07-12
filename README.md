# Sentiment Analysis using Logistic Regression

This project performs sentiment analysis on product/tweet reviews using a Logistic Regression model. It uses text preprocessing, TF-IDF vectorization, and classification techniques to predict whether a review is **positive** or **negative**.

---

## 📌 Project Overview

- ✅ Preprocessed and cleaned textual data using regular expressions
- ✅ Applied **TF-IDF Vectorizer** to convert text to numerical features
- ✅ Handled **class imbalance** using upsampling (resampling the minority class)
- ✅ Trained a **Logistic Regression** classifier using scikit-learn
- ✅ Achieved high accuracy and well-balanced F1-score
- ✅ Visualized performance using a **confusion matrix**

---

## 🛠️ Technologies Used

- Python 🐍  
- Jupyter Notebook 📓  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `re`

---

## 🧠 ML Model Used

- **Logistic Regression**

---

## 📊 Final Results

- **Accuracy**: `95.52%`

| Metric     | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| Negative   | 0.98      | 0.93   | 0.95     |
| Positive   | 0.94      | 0.98   | 0.96     |
| **Average**| **0.96**  | **0.96**| **0.96** |

---

## 🧼 Data Preprocessing Steps

- Lowercased all reviews
- Removed:
  - URLs, mentions (@), hashtags (#)
  - Special characters and punctuation
  - Numbers
- Applied `.strip()` to clean whitespace

---

## ⚖️ Class Balancing (Upsampling)

To solve class imbalance:
- Separated majority (negative) and minority (positive) classes
- Upsampled minority class using `resample()` from `sklearn.utils`
- Combined and shuffled the dataset for balanced training

---

## 📁 Dataset Source

Dataset used from open-source Twitter sentiment classification:
- [Twitter Sentiment Analysis GitHub Dataset](https://github.com/dD2405/Twitter_Sentiment_Analysis)

Columns used:
- `tweet`: the review text
- `label`: sentiment (0 = Negative, 1 = Positive)

---

## 🧪 How to Run This Project

1. Open the notebook in Jupyter:
2. Run each cell one by one
3. Output will show:
- Accuracy
- Classification Report
- Confusion Matrix
4. Try replacing the classifier or playing with TF-IDF features for experimentation

---

## 🤖 Future Improvements

- Try other models like Naive Bayes, SVM, or XGBoost
- Add stopword removal
- Use advanced preprocessing with `nltk` or `spaCy`
- Deploy the model using Flask or Streamlit

---

## 📬 Author

**Arqammohammed Manur**  
📧 Email: anmanur145@gmail.com  
[GitHub](https://github.com/Arqam-M) | [LinkedIn](https://linkedin.com/in/arqammohammed-manur-10a068211)

---
