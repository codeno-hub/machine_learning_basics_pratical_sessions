pip install scikit-learn


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load a built-in dataset
data = load_iris()
X = data.data       # Features (petal/sepal length and width)
y = data.target     # Target labels (species)

# 2. Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions on unseen test data
predictions = model.predict(X_test)

# 5. Evaluate the model's accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Predict on a single new sample
sample = [[5.1, 3.5, 1.4, 0.2]]
predicted_class = model.predict(sample)
print(f"Predicted Species: {data.target_names[predicted_class[0]]}")
