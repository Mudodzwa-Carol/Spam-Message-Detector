# 📩 Spam Message Detector (Machine Learning)

A beginner-friendly machine learning project that classifies messages as **spam** or **ham (not spam)** using Python and Scikit-learn.

---

## 🚀 Project Overview

This project builds a text classification model that can:

- Read a message (SMS or email)
- Predict whether it is spam or not
- Output a probability score for how likely it is to be spam

The model was improved by combining **SMS data** with **email data**, allowing it to better detect more realistic and modern spam patterns.

---

## 🧠 How It Works

1. **Load datasets** (SMS + Email)
2. **Clean and format data**
3. **Convert labels**
   - `ham → 0`
   - `spam → 1`
4. **Vectorize text using TF-IDF**
5. **Split into training and testing sets**
6. **Train a Logistic Regression model**
7. **Evaluate accuracy**
8. **Predict on new user input**

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn

---

📂 Spam-Message-Detector/
    ├── spam_detector.py        # Main script
    ├── spam.csv                # SMS dataset
    ├── email_dataset/          # Email dataset (folders 1 = ham, 2 = spam)
    ├── requirements.txt
    └── README.md

## ⚙️ Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/Mudodzwa-Carol/spam-message-detector.git
cd spam-message-detector

Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate   # Windows

Install dependencies:

pip install -r requirements.txt
▶️ Running the Project
python spam_detector.py

Then enter any message to test if it is spam.

📊 Example Output
Enter a message: Get cheap meds online
Spam probability: 0.89
This message is SPAM
📈 Key Insight

The biggest improvement in this project came from better data, not a more complex model.

Combining SMS and email datasets helped the model learn more realistic spam patterns.

A model is only as good as the data it is trained on.

⚠️ Limitations

The model still misclassifies some messages

More advanced preprocessing could improve performance

Could be upgraded with deep learning (e.g. LSTM, transformers)

💡 Future Improvements

Add more diverse datasets

Deploy as a web app (Flask/Streamlit)

Improve preprocessing (stemming, lemmatization)

Try more advanced models

👩🏽‍💻 Author

Mudodzwa Carol Mashau