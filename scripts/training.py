import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import shap
import numpy as np


def initialize_mlflow(uri, experiment_name):
    """Initializes MLflow and returns the tracking id."""
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        tracking_id = run.info.run_uuid
    return tracking_id

def train_and_log_model(model, model_name, tracking_id, x_train, y_train, x_test, y_test):
  """Trains and logs the model with MLflow."""
  with mlflow.start_run(run_id = tracking_id) as run:
    mlflow.set_tag("model_name", model_name)

    # Ensure y_train is 1D array
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Create an input_example
    input_example = x_train.iloc[[0]]

    #Infer a model signature
    signature = mlflow.models.infer_signature(input_example, model.predict(input_example))

    mlflow.sklearn.log_model(model, model_name, signature=signature, input_example = input_example)
    return mse, r2, model


def explain_model_with_shap(model, x_train, x_test):
    """ Explains the model with shap"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test,show = False)
    return shap_values