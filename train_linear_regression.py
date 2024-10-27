import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('flights.csv')
df['date_oftravel'] = pd.to_datetime(df['date_oftravel'])

# Prepare data for the regression model
X = df[["source", "destination", "flightType", "time", "distance", "agency", "date_oftravel"]]
y = df["price"]

# Define categorical and numerical features
categorical_features = ["source", "destination", "flightType", "agency"]
numeric_features = ["time", "distance"]

# Preprocessor for transforming features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(sparse_output=False), categorical_features)
    ]
)

# Create the pipeline with Linear Regression
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model_pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(model_pipeline, 'linear_regression_model.pkl')
print("Linear Regression model trained and saved as 'linear_regression_model.pkl'")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Calculate and display accuracy metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression Mean Squared Error: {mse}")
print(f"Linear Regression R² Score: {r2}")

# Save accuracy metrics to a file
with open('linear_regression_metrics.txt', 'w') as f:
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"R² Score: {r2}\n")

# Plot the actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual Prices vs. Predicted Prices (Linear Regression)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.grid()
plt.savefig('linear_regression_plot.png')
plt.close()
print("Linear Regression plot saved as 'linear_regression_plot.png'")
