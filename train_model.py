import pickle
# Alternatively, you can use joblib
# from joblib import dump

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Fetch the MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_jobs=-1, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Evaluate the classifier
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model using pickle
with open('mnist_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Alternatively, using joblib
# dump(clf, 'mnist_model.joblib')
