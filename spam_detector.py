import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# Load SMS dataset
sms_data = pd.read_csv("spam.csv", encoding="latin-1")
sms_data = sms_data[['v1','v2']].rename(columns={'v1': 'label', 'v2': 'message'})
sms_data['label'] = sms_data['label'].map({'ham': 0, 'spam': 1})

# Preprocess Email dataset (folders 1 = ham, 2 = spam)
email_data = []
email_base_path = "email_dataset"  # folder containing '1/' and '2/'

# Load ham emails
for filename in os.listdir(os.path.join(email_base_path, "1")):
    filepath = os.path.join(email_base_path, "1", filename)
    with open(filepath, "r", encoding="latin-1") as f:
        text = f.read()
        email_data.append(["ham", text])

# Load spam emails
for filename in os.listdir(os.path.join(email_base_path, "2")):
    filepath = os.path.join(email_base_path, "2", filename)
    with open(filepath, "r", encoding="latin-1") as f:
        text = f.read()
        email_data.append(["spam", text])

# Convert to DataFrame
email_data = pd.DataFrame(email_data, columns=["label", "message"])
email_data['label'] = email_data['label'].map({'ham': 0, 'spam': 1})

# Combine SMS and Email datasets
combined_data = pd.concat([sms_data, email_data], ignore_index=True)
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Balance dataset by upsampling spam
ham = combined_data[combined_data['label'] == 0]
spam = combined_data[combined_data['label'] == 1]

spam_upsampled = resample(spam, replace=True, n_samples=len(ham), random_state=42)
balanced_data = pd.concat([ham, spam_upsampled])
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split dataset into train/test
messages = balanced_data['message']
labels = balanced_data['label']
messages_train, messages_test, labels_train, labels_test = train_test_split(
    messages, labels, test_size=0.2, random_state=42
)

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    stop_words="english",
    min_df=2,
    max_features=15000
)
messages_train_tfidf = vectorizer.fit_transform(messages_train)
messages_test_tfidf = vectorizer.transform(messages_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(messages_train_tfidf, labels_train)

# Evaluate model
predictions_test = model.predict(messages_test_tfidf)
accuracy = accuracy_score(labels_test, predictions_test)
print("\nModel accuracy:", accuracy)

#  Spam detector for user input
print("\nSpam Detector Ready!")
print("Type a message to check if it is spam.")
print("Type 'quit' to stop.\n")

while True:
    user_message = input("Enter a message: ")
    if user_message.lower() == "quit":
        print("Program ended.")
        break

    message_tfidf = vectorizer.transform([user_message])
    prediction = model.predict(message_tfidf)
    probability = model.predict_proba(message_tfidf)[0][1]
    print("Spam probability:", round(probability, 2))

    if prediction[0] == 1:
        print("This message is SPAM\n")
    else:
        print("This message is NOT spam\n")