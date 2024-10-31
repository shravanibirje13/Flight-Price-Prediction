**Flight Price Prediction Project**
**Project Overview**

This project aims to predict flight ticket prices based on multiple factors such as the flight's source, destination, travel date, flight type, time, distance, and agency. Using machine learning models, the application provides estimates of flight prices and enables users to select between a Linear Regression and a Decision Tree model for prediction. It also offers visual insights through scatter plots and decision tree diagrams, along with accuracy metrics to help users evaluate model performance.

**Project Structure** - The files included in this project are organized as follows:
1. app.py - The main Flask application for running the web interface.
2. train_decision_tree.py - Script to train and save the Decision Tree model.
3. train_linear_regression.py - Script to train and save the Linear Regression model.
4. flight_final - flights.csv.csv - Dataset used for model training and predictions.
5. decision_tree_metrics.txt - File containing accuracy metrics for the Decision Tree model.
6. linear_regression_metrics.txt - File containing accuracy metrics for the Linear Regression model.
7. decision_tree_model.pkl - Saved Decision Tree model file.
8. linear_regression_model.pkl - Saved Linear Regression model file.
9. model.pkl - Another saved model file (if a backup or additional model).
10. decision_tree_plot.png - Visualization of the Decision Tree structure.
11. linear_regression_plot.png - Scatter plot for Linear Regression predictions.
12. templates/ - Folder containing HTML templates for the frontend.
13. static/ - Folder for (images) static assets

**How It Works**
1. Data Preparation: The data from flight_final - flights.csv.csv is loaded, and preprocessing is done to prepare it for the models. Categorical variables are one-hot encoded, and numeric variables are used directly in model training.

2. Model Training and Prediction: The project includes scripts to train two models:
 Linear Regression (train_linear_regression.py)
 Decision Tree Regression (train_decision_tree.py)
Both scripts create and save the trained models for use in predictions.

3. User Interaction: The user enters flight details, and the application predicts the price using the selected model. Results, accuracy metrics, and graphs are displayed to help users understand the model’s predictions.

**Usage**
1. Run app.py: Launch the app locally by running the app.py file.
2. Enter Flight Details: Input details such as source, destination, flight type, etc., on the homepage.
3. Select Prediction Model: Choose either Linear Regression or Decision Tree for the prediction.
4. View Results: The app will display the predicted price along with relevant graphs and metrics.

**Models Used** - 
1. Linear Regression - Suitable for relationships with linear trends.
2. Decision Tree Regression - Captures complex, non-linear relationships.

**Evaluation Metrics**
1. Mean Squared Error (MSE) - Measures the average squared error between predictions and actual values.
2. R² Score - Indicates the model's ability to explain data variability.
Metrics are saved in decision_tree_metrics.txt and linear_regression_metrics.txt for the Decision Tree and Linear Regression models, respectively.

**Results Visualization**
1. Scatter Plot (linear_regression_plot.png): Displays actual vs. predicted prices for Linear Regression.
2. Decision Tree Diagram (decision_tree_plot.png): Visualizes the Decision Tree structure for interpretability.

**Future Improvements** 
1. Additional Models: Add models like Random Forest and Gradient Boosting for better predictions.
2. More Feature Engineering: Add additional date-based features, seasonality, etc.
3. Improved UI: Make the application more user-friendly and visually appealing.
4. API Integration: Integrate live data sources to enhance prediction accuracy.

**Output**
![image](https://github.com/user-attachments/assets/f614c1a1-34e2-45b0-93be-967c52510738)
LINEAR REGRESSION
![image](https://github.com/user-attachments/assets/5ebdc595-042e-4503-82cb-9decccd1c65c)
![image](https://github.com/user-attachments/assets/92bf030e-2b71-48ca-89c9-d87a119e7b4d)
![image](https://github.com/user-attachments/assets/780f71f1-048b-4edd-b8be-0a75756a8651)
DECISION TREE
![image](https://github.com/user-attachments/assets/19ab3e4d-ea1a-4b95-971e-5bdc05036558)
![image](https://github.com/user-attachments/assets/37f938df-8d57-4bad-940e-d4ecd1dac261)
![image](https://github.com/user-attachments/assets/43c23f6b-501e-482c-a790-9c2c67b3e0e2)
