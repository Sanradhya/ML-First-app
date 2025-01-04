import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('NetflixOriginals.csv', encoding='ISO-8859-1')

# Create a sentiment column based on IMDB Score
df['sentiment'] = df['IMDB Score'].apply(lambda x: 'positive' if x >= 6 else 'negative')

# Extract features and labels
X = df['Title']  # Use movie titles as the review text
y = df['sentiment'].map({'positive': 1, 'negative': 0})  # Map sentiments to binary

# Text preprocessing
stop_words = set(stopwords.words('english'))
X = X.apply(lambda title: ' '.join([word for word in word_tokenize(title.lower()) if word.isalpha() and word not in stop_words]))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
