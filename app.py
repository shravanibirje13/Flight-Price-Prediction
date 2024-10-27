from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import io
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv('flights.csv')
df['date_oftravel'] = pd.to_datetime(df['date_oftravel'])

# Prepare data for regression model
X_reg = df[["source", "destination", "flightType", "time", "distance", "agency", "date_oftravel"]]
y_reg = df["price"]

# Categorical and numerical feature transformation
categorical_features = ["source", "destination", "flightType", "agency"]
numeric_features = ["time", "distance"]

preprocessor_reg = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(sparse_output=False), categorical_features)
    ]
)

# Create pipeline for regression models
linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor_reg),
    ("regressor", LinearRegression())
])
tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor_reg),
    ("regressor", DecisionTreeRegressor())
])

# Fit models
linear_model.fit(X_reg, y_reg)
tree_model.fit(X_reg, y_reg)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    source = request.form['source']
    destination = request.form['destination']
    flight_type = request.form['flight_type']
    time = float(request.form['time'])
    distance = float(request.form['distance'])
    agency = request.form['agency']
    date_of_travel = pd.to_datetime(request.form['date_of_travel'])

    # Create input data for prediction
    input_data = {
        "source": source,
        "destination": destination,
        "flightType": flight_type,
        "time": time,
        "distance": distance,
        "agency": agency,
        "date_oftravel": date_of_travel,
    }

    input_df = pd.DataFrame([input_data])

    # Select model type
    model_choice = request.form.get('model_choice')

    if model_choice == 'linear':
        price_prediction = linear_model.predict(input_df)[0]
        predicted_prices = linear_model.predict(X_reg)

        # Generate scatter plot for Linear Regression
        plt.figure(figsize=(10, 6))
        plt.scatter(y_reg, predicted_prices, alpha=0.6)
        plt.plot([y_reg.min(), y_reg.max()], [y_reg.min(), y_reg.max()], color='red', linestyle='--')  # Diagonal line
        plt.title('Actual Prices vs Predicted Prices')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.grid()
        plt.savefig('static/prediction_plot.png')  # Save the plot
        plt.close()

        # Send the linear regression plot
        plot_filename = 'prediction_plot.png'

    else:
        price_prediction = tree_model.predict(input_df)[0]
        predicted_prices = tree_model.predict(X_reg)

        # Generate decision tree plot
        plt.figure(figsize=(20, 10))
        plot_tree(tree_model.named_steps['regressor'], filled=True, feature_names=numeric_features + list(
            preprocessor_reg.transformers_[1][1].get_feature_names_out(categorical_features)))
        plt.title('Decision Tree Visualization')
        plt.savefig('static/tree_plot.png')  # Save the tree plot
        plt.close()

        # Send the decision tree plot
        plot_filename = 'tree_plot.png'

    # Calculate metrics
    mse = mean_squared_error(y_reg, predicted_prices)
    avg_actual_price = np.mean(y_reg)
    r2 = r2_score(y_reg, predicted_prices)

    # Return the prediction and metrics
    return render_template(
        'result.html',
        price=price_prediction,
        plot=plot_filename,
        mse=mse,
        avg_price=avg_actual_price,
        r2=r2,
        model_choice=model_choice  # Pass the model choice for graph display
    )

# Route to show the graph
@app.route('/show_graph/<model_choice>')
def show_graph(model_choice):
    # Determine the graph image path based on the selected model
    if model_choice == 'linear':
        graph_path = 'prediction_plot.png'  # Linear regression graph
    else:
        graph_path = 'tree_plot.png'  # Decision tree graph

    # Fetch metrics to display on the graph page
    predicted_prices = linear_model.predict(X_reg) if model_choice == 'linear' else tree_model.predict(X_reg)
    mse = mean_squared_error(y_reg, predicted_prices)
    avg_actual_price = np.mean(y_reg)
    r2 = r2_score(y_reg, predicted_prices)

    return render_template('graph.html', graph_path=graph_path, model_choice=model_choice, mse=mse, avg_price=avg_actual_price, r2=r2)

if __name__ == '__main__':
    app.run(debug=True)
