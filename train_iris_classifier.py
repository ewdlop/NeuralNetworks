import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from huggingface_hub import HfApi, create_repo, whoami
import os

# Load the data
df = pd.read_csv('datasets/IRIS.csv')

# Separate features and target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model locally
joblib.dump(model, 'models/iris_classifier.joblib')

# Initialize Hugging Face API
api = HfApi()

try:
    # Get the current user's username
    username = whoami()['name']
    repo_name = f"{username}/iris-classifier"
    
    print(f"\nCreating repository: {repo_name}")
    # Create a new repository
    create_repo(repo_name, repo_type="model", exist_ok=True)
    
    print("Uploading model to Hugging Face Hub...")
    # Upload the model
    api.upload_file(
        path_or_fileobj="iris_classifier.joblib",
        path_in_repo="iris_classifier.joblib",
        repo_id=repo_name,
        repo_type="model"
    )
    
    print(f"\nModel successfully uploaded to Hugging Face Hub!")
    print(f"Repository: https://huggingface.co/{repo_name}")
    
except Exception as e:
    print(f"\nError uploading to Hugging Face: {str(e)}")
    print("\nPlease make sure you're logged in to Hugging Face using:")
    print("huggingface-cli login")
    print("\nIf you're already logged in, please check your internet connection and try again.") 