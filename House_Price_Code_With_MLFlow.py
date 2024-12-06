

# Create pipeline also trains a Linear Regression model. Also incorporates **MLFlow** tracking to log parameters, metrics, and model.


import mlflow

# Set the tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Define the experiment name
experiment_name = "HousingPricePrediction"

# Try to get the experiment by name
experiment = mlflow.get_experiment_by_name(experiment_name)

# If the experiment doesn't exist, create it
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
else:
    experiment_id = experiment.experiment_id
    print(f"Experiment '{experiment_name}' found with ID: {experiment_id}")

# Try to access the experiment using the ID
experiment = mlflow.get_experiment(experiment_id)

# Print experiment details
print(f"Experiment details: {experiment}")

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error

# Ensure that mlflow is correctly pointing to the right server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Constants for dataset download and file paths
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Download the dataset if it doesn't exist
def fetch_housing_data():
    if not os.path.isdir(HOUSING_PATH):
        os.makedirs(HOUSING_PATH)
    housing_csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    if not os.path.exists(housing_csv_path):
        # Download and extract the dataset (you can skip this if the data is already available)
        import tarfile
        from urllib import request
        tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
        request.urlretrieve(HOUSING_URL, tgz_path)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=HOUSING_PATH)
    return pd.read_csv(housing_csv_path)

# Load data
housing = fetch_housing_data()

# Define a custom transformer to select the columns for processing
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]

# Function to prepare the full pipeline with numerical and categorical columns
def prepare_data_pipeline(housing, cat_attributes=["ocean_proximity"]):
    # Numerical attributes pipeline
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(housing.select_dtypes(include=['number']).columns)),
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ])

    # Categorical attributes pipeline (Ordinal encoding for example)
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attributes)),
        ('ordinal_encoder', OrdinalEncoder())
    ])

    # Full preprocessing pipeline for both numeric and categorical attributes
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, housing.select_dtypes(include=['number']).columns),
        ("cat", cat_pipeline, cat_attributes)
    ])

    # Fit and transform the data using the full pipeline
    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared, full_pipeline

# Prepare the data using the pipeline
housing_prepared, full_pipeline = prepare_data_pipeline(housing)

# Train a Linear Regression model and evaluate it using MLFlow
def train_and_log_model(housing_prepared):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        housing_prepared, housing['median_house_value'], test_size=0.2, random_state=42
    )

    # Ensure the experiment exists, or create it
    experiment_name = "HousingPricePrediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Create the experiment if it doesn't exist
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # Start an MLFlow experiment
    with mlflow.start_run(experiment_id=experiment_id):
        # Log the parameters
        mlflow.log_param("model_type", "Linear Regression")
        mlflow.log_param("test_size", 0.2)

        # Initialize the model
        model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate the mean absolute error
        mae = mean_absolute_error(y_test, predictions)
        mlflow.log_metric("mae", mae)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Print the results
        print(f"Mean Absolute Error (MAE): {mae}")

# Call the train and log function
train_and_log_model(housing_prepared)