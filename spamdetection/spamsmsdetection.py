# Spam SMS Detection Project

# Importing necessary libraries
import pandas as pd
import seaborn as sns
import pickle
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

#Loading the CSV file using the Pandas

data =pd.read_csv(r"C:\Users\manasa\Downloads\spam (1).csv",encoding='latin-1')
print("*********************SPAM SMS DETECTION***************************")
# Keep only the useful columns
data = data[['v1', 'v2']]  
data.columns = ['label', 'text']
#print the first 5 rows in a dataset
print(data.head())
#print the last 5 rows in a dataset
print(" ")
print(data.tail())

# Checking for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Visualizing label distribution
sns.countplot(x='label', data=data)
plt.title("Distribution of Ham and Spam messages")
plt.show()

# Encoding Labels
le = LabelEncoder()
data['label_num'] = le.fit_transform(data['label'])

print("\nData after encoding:")
print(data.head())

# Text Preprocessing
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

data['clean_text'] = data['text'].apply(clean_text)

print("\nData after cleaning text:")
print(data.head())

# Converting text to numbers (Feature Extraction using TF-IDF)
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data['clean_text']).toarray()
y = data['label_num']

# Splitting the data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the models

# Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

# Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

# Model Evaluation

# Logistic Regression Evaluation
y_pred_lr = model_lr.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Naive Bayes Evaluation
y_pred_nb = model_nb.predict(X_test)
print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# Random Forest Evaluation
y_pred_rf = model_rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Saving the trained model and vectorizer
pickle.dump(model_rf, open('spam_classifier.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))

# Testing with New Messages
# Load the saved model and vectorizer
model = pickle.load(open('spam_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Sample test
sample = ["Congratulations! You have won a $1000 Walmart gift card. Click here to claim now."]
sample_transformed = vectorizer.transform(sample).toarray()

# Predict
prediction = model.predict(sample_transformed)
print("\nPrediction for sample message:", prediction)

# 0 → Ham (not spam), 1 → Spam
if prediction[0] == 1:
    print("The message is: SPAM ")
else:
    print("The message is: NOT SPAM ")
