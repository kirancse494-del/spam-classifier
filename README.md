# 📧 Spam Email Classifier

A beginner-friendly Machine Learning project that detects whether a message is **SPAM** or **NOT SPAM** using Python and scikit-learn.

---

## 🚀 What This Project Does

This classifier is trained on 5,574 real SMS messages and can predict with **~97% accuracy** whether any new message is spam or legitimate.

**Example predictions:**

| Message | Prediction |
|---|---|
| "Congratulations! You've won a FREE iPhone!" | 🚨 SPAM |
| "Hey, are we still meeting tomorrow?" | ✅ NOT SPAM |
| "URGENT: Your bank account is suspended." | 🚨 SPAM |
| "Can you send me today's class notes?" | ✅ NOT SPAM |

---

## 🧠 How It Works

1. **Load Data** — Uses the public SMS Spam Collection dataset (5,574 messages)
2. **Preprocess** — Converts labels to numbers (spam=1, ham=0)
3. **TF-IDF Vectorization** — Converts text into numeric features
4. **Train Model** — Uses Naive Bayes algorithm
5. **Evaluate** — Tests accuracy on unseen data
6. **Predict** — Classify any new message instantly

---

## 🛠️ Tech Stack

- Python 3.x
- pandas
- scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/spam-classifier.git
cd spam-classifier
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn
```

**3. Run the classifier**
```bash
python spam_classifier.py
```

---

## 📊 Results

```
Model Accuracy: ~97%

              precision    recall  f1-score
Ham (Not Spam)   0.98       0.99      0.99
Spam             0.96       0.93      0.94
```

---

## 📁 Project Structure

```
spam-classifier/
│
├── spam_classifier.py    # Main Python script
└── README.md             # Project documentation
```

---

## 👤 Author

**Kiran Indurthi**  
B.Tech Computer Science | Aspiring AI Developer  
[LinkedIn](https://www.linkedin.com/in/kiranflutterdev)

---

## 📌 About This Project

This project was built as part of my journey transitioning into AI/ML development. It demonstrates:
- Text preprocessing and feature engineering
- Supervised Machine Learning classification
- Model evaluation and performance metrics
