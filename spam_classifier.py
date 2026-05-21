# ============================================================
#  SPAM EMAIL CLASSIFIER
#  Built with Python + scikit-learn
#  Author: Kiran Indurthi
#  Description: A Machine Learning model that detects whether
#               a message is SPAM or NOT SPAM (ham)
# ============================================================

# --- STEP 1: Import Libraries ---
import pandas as pd                          # For handling data
from sklearn.model_selection import train_test_split   # To split data
from sklearn.feature_extraction.text import TfidfVectorizer  # Text to numbers
from sklearn.naive_bayes import MultinomialNB          # ML Algorithm
from sklearn.metrics import accuracy_score, classification_report  # Measure performance
import urllib.request                        # To download dataset


# --- STEP 2: Load the Dataset ---
# We use the SMS Spam Collection dataset (public dataset from UCI)
# It contains 5,574 messages labeled as 'spam' or 'ham' (not spam)

print("📥 Loading dataset...")

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
urllib.request.urlretrieve(url, "sms_spam.tsv")

df = pd.read_csv("sms_spam.tsv", sep="\t", header=None, names=["label", "message"])

print(f"✅ Dataset loaded! Total messages: {len(df)}")
print(f"   Spam messages : {len(df[df['label'] == 'spam'])}")
print(f"   Ham messages  : {len(df[df['label'] == 'ham'])}")
print()


# --- STEP 3: Prepare the Data ---
# Convert labels: 'spam' → 1, 'ham' → 0
df["label_num"] = df["label"].map({"spam": 1, "ham": 0})

X = df["message"]       # Input  → the message text
y = df["label_num"]     # Output → 1 (spam) or 0 (not spam)


# --- STEP 4: Split into Training and Testing Sets ---
# 80% of data used to TRAIN the model
# 20% of data used to TEST how accurate it is
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")
print()


# --- STEP 5: Convert Text to Numbers (TF-IDF) ---
# Machine Learning models can't read text directly.
# TF-IDF converts each message into a numeric representation
# based on how important each word is.

vectorizer = TfidfVectorizer(stop_words="english")  # Ignore common words like 'the', 'is'
X_train_tfidf = vectorizer.fit_transform(X_train)   # Learn vocabulary + transform training data
X_test_tfidf  = vectorizer.transform(X_test)        # Transform test data using same vocabulary


# --- STEP 6: Train the Machine Learning Model ---
# We use Naive Bayes — a simple but powerful algorithm for text classification
print("🤖 Training the model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("✅ Model trained!\n")


# --- STEP 7: Test the Model ---
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print()
print("📋 Detailed Report:")
print(classification_report(y_test, y_pred, target_names=["Ham (Not Spam)", "Spam"]))


# --- STEP 8: Try It Yourself! ---
# Test with your own messages

def predict_message(message):
    """
    Takes a message string and predicts whether it's SPAM or HAM.
    """
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf)[0]

    label = "🚨 SPAM" if prediction == 1 else "✅ NOT SPAM (Ham)"
    confidence = probability[prediction] * 100

    print(f"\nMessage   : \"{message}\"")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1f}%")


# --- Test Messages ---
print("\n" + "="*55)
print("       TESTING WITH SAMPLE MESSAGES")
print("="*55)

predict_message("Congratulations! You've won a FREE iPhone. Click now to claim your prize!")
predict_message("Hey, are we still meeting tomorrow at 3pm?")
predict_message("URGENT: Your bank account has been suspended. Verify now!")
predict_message("Can you please send me the notes from today's class?")
predict_message("Win $1000 cash prize! Text WIN to 98765 now. Limited offer!")
