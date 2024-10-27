import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.tree import DecisionTreeRegressor, plot_tree
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

# Create the pipeline with Decision Tree Regressor
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor())
])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model_pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(model_pipeline, 'decision_tree_model.pkl')
print("Decision Tree model trained and saved as 'decision_tree_model.pkl'")

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Calculate and display accuracy metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Decision Tree Mean Squared Error: {mse}")
print(f"Decision Tree R² Score: {r2}")

# Save accuracy metrics to a file
with open('decision_tree_metrics.txt', 'w') as f:
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"R² Score: {r2}\n")

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model_pipeline.named_steps['regressor'], filled=True, feature_names=numeric_features + list(
    preprocessor.transformers_[1][1].get_feature_names_out(categorical_features)))
plt.title('Decision Tree Visualization')
plt.savefig('decision_tree_plot.png')
plt.close()
print("Decision Tree plot saved as 'decision_tree_plot.png'")
