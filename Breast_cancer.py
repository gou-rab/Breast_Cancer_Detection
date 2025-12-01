# KNN Breast Cancer Classifier

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data          # features (30 columns)
y = data.target        # labels (0 = malignant, 1 = benign)

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train KNN model
k = 5  # number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 4. Predict on test data
y_pred = knn.predict(X_test)

# 5. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Predict for a new sample (example)
# Use one sample from test set just to show prediction
new_sample = X_test[0].reshape(1, -1)
new_pred = knn.predict(new_sample)
print("\nPrediction for one test sample:")
print("Predicted class:", data.target_names[new_pred[0]])
print("Actual class   :", data.target_names[y_test[0]])
