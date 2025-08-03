📚 Book Review Sentiment Analysis


Overview
This project aims to analyze customer feedback by predicting the sentiment of book reviews as either positive or negative. By understanding user sentiment, businesses can make informed decisions regarding marketing, product improvement, and customer engagement strategies.

🧠 Objective
To build and evaluate machine learning and deep learning models that classify book reviews based on sentiment. The target variable is derived from the book rating:

Ratings 4 or 5 → Positive (Label = 1)

Ratings 1 or 2 → Negative (Label = 0)

🗃️ Dataset
Source: A dataset of user-submitted book reviews and corresponding ratings.

Features: review (text), rating (integer 1–5)

Preprocessing:

Removed neutral ratings (e.g., 3)

Cleaned and tokenized text

Used TF-IDF and n-grams for vectorization

🧪 Models Used
Model	Accuracy	AUC Score
Logistic Regression	82.0%	0.90
Naive Bayes	81.0%	0.89
Neural Network (with Dropout & n-grams)	82.7%	0.91

Improvements:
Added dropout layers to reduce overfitting.

Introduced bi-grams to capture multi-word sentiment patterns.

Increased epochs to improve training performance.

Performed cross-validation for more reliable model selection.

📉 Error Analysis
Common misclassifications include:

Ironic or sarcastic reviews misread as positive due to keyword bias.

Mixed or neutral reviews often skewed toward positive predictions.

Future improvement may include incorporating contextual embeddings (e.g., BERT).

📌 Key Insights
Machine learning and neural models can effectively predict customer sentiment from review text.

Business teams can use this pipeline to automate sentiment tracking and respond proactively to user feedback.

🛠️ Tech Stack
Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras)

Natural Language Processing (TF-IDF, n-gram features)

Matplotlib / Seaborn (for visualization)

🚀 How to Run
bash
Copy
Edit
# Clone the repo
git clone https://github.com/yourusername/bookreview-sentiment-analysis.git
cd bookreview-sentiment-analysis

# Install dependencies
pip install -r requirements.txt

# Run the training script
python sentiment_model.py
📈 Results Visualization
Includes:

Confusion matrix

ROC curve

Accuracy/loss plots over epochs (for deep learning models)
