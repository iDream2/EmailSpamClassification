import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle as pk


df = pd.read_csv("./spam.csv",  encoding='latin1' )
print(df.head())
print(df.info())
print(df.shape)

print( df.columns )
print("Unique values of the Unnamed : 2 column  ~")
print(df["Unnamed: 2"].unique())

print("\n")
print("\n")
print("\n")
print("\n")


print("Unique values of the Unnamed : 3 column  ~")
print(df["Unnamed: 3"].unique())

print("\n")
print("\n")
print("\n")


print("Unique values of the Unnamed : 4 column  ~")
print(df["Unnamed: 4"].unique())

df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

# Rename columns
df = df.rename(columns={"v1": "label", "v2": "message"})

# Encode the target variable
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Prepare features and target
X = df['message']
y = df['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.53, random_state=42)

# TF-IDF vectorization for the text data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)


# Save the model and vectorizer
with open('clf.pkl', 'wb') as model_file:
    pk.dump(clf, model_file)
with open('vectorizer.pkl', 'wb') as vec_file:
    pk.dump(vectorizer, vec_file)

# Make predictions
pred = clf.predict(X_test_tfidf)
print("\nAll the predictions made based on the X_test\n")
print(pred)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, pred)
print("Accuracy Score: ", accuracy)

# Cross-validation to get a more robust accuracy measure
cross_val_scores = cross_val_score(clf, vectorizer.transform(X), y, cv=5)
print("Cross-Validation Accuracy Scores: ", cross_val_scores)
print("Mean Cross-Validation Accuracy: ", cross_val_scores.mean())


# Load the trained model and vectorizer
with open('clf.pkl', 'rb') as model_file:
    clf = pk.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pk.load(vec_file)

# Example data point to predict
new_message = ["Free entry in 2 a weekly competition to win FA Cup final tickets. Text FA to 87121 to receive entry question(std txt rate)T&C's apply"]

# Transform the new data point using the loaded vectorizer
new_message_tfidf = vectorizer.transform(new_message)

# Make the prediction
prediction = clf.predict(new_message_tfidf)

# Convert prediction to label (optional, for interpretation)
label = le.inverse_transform(prediction)

print("Prediction: ", prediction[0])
print("Label: ", label[0])
