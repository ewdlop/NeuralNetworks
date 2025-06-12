import joblib
import numpy as np

# Load the model
model = joblib.load('models/iris_classifier.joblib')

# Example measurements (sepal_length, sepal_width, petal_length, petal_width)
# These are example measurements for each species
examples = [
    [5.1, 3.5, 1.4, 0.2],  # Iris-setosa
    [6.0, 2.2, 4.0, 1.0],  # Iris-versicolor
    [6.3, 3.3, 6.0, 2.5]   # Iris-virginica
]

# Make predictions
predictions = model.predict(examples)

# Print predictions
print("\nPredictions for example measurements:")
for i, (measurements, prediction) in enumerate(zip(examples, predictions)):
    print(f"\nExample {i+1}:")
    print(f"Measurements: sepal_length={measurements[0]}, sepal_width={measurements[1]}, "
          f"petal_length={measurements[2]}, petal_width={measurements[3]}")
    print(f"Predicted species: {prediction}")

# If you want to make a prediction for a single flower
def predict_single_flower(sepal_length, sepal_width, petal_length, petal_width):
    measurements = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(measurements)[0]
    return prediction

# Example of using the function
print("\nPredicting a single flower:")
result = predict_single_flower(5.1, 3.5, 1.4, 0.2)
print(f"Predicted species: {result}") 