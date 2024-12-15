import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset 
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]  # Keep relevant columns
df.columns = ['label', 'message']

# Encode labels: spam = 1, ham = 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Step 1: Preprocess data
X = df['message']
y = df['label']

# Step 2: Vectorize the text data (Bag of Words)
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))  # Unigrams + bigrams
X = vectorizer.fit_transform(X)

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Naive Bayes model (with fit_prior=True to handle class imbalance)
model = MultinomialNB(alpha=1.0, fit_prior=True)
model.fit(X_train, y_train)

# Step 5: Evaluate the model using cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with a new email
sample_email = ["Browse the Framer Marketplace for tons of free and premium templates, then make it your own with a few quick tweaks."]
sample_vectorized = vectorizer.transform(sample_email)
print("Spam Prediction:", model.predict(sample_vectorized))  # Output: 1 (spam) or 0 (ham)
