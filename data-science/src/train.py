# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import shutil
from mlflow.tracking.artifact_utils import _download_artifact_from_uri

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("train")

    parser.add_argument("--train_data", type=str, required=True, help="Path to train data")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data")
    parser.add_argument("--model_output", type=str, required=True, help="Path to save output model")

    # Model hyperparameters
    parser.add_argument("--model_type", type=str, default="random_forest", choices=["random_forest", "decision_tree"],
                        help="Model type to use for regression")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators (for RandomForest)")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree")
    parser.add_argument("--criterion", type=str, default="squared_error",
                        help="Function to measure the quality of a split (for DecisionTree)")

    return parser.parse_args()

def select_first_file(path):
    """Selects the first file in a folder, assuming there's only one file."""
    files = os.listdir(path)
    if not files:
        raise FileNotFoundError(f"No files found in: {path}")
    return os.path.join(path, files[0])

def main(args):
    '''Train and evaluate the model'''
    mlflow.start_run()

    # Load datasets
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Strip column names of whitespace
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    print("Columns in train_df:", train_df.columns.tolist())

    target_col = "price"
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found in dataset.")

    X_train = train_df.drop(target_col, axis=1).values
    y_train = train_df[target_col].values

    X_test = test_df.drop(target_col, axis=1).values
    y_test = test_df[target_col].values

    # Initialize and train the model
    if args.model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        mlflow.log_param("n_estimators", args.n_estimators)
    else:
        model = DecisionTreeRegressor(criterion=args.criterion, max_depth=args.max_depth, random_state=42)
        mlflow.log_param("criterion", args.criterion)

    mlflow.log_param("model_type", args.model_type)
    mlflow.log_param("max_depth", args.max_depth)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"📊 Mean Squared Error: {mse:.2f}")
    print(f"📊 R^2 Score: {r2:.2f}")

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    # Log model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # Copy model from MLflow run directory to args.model_output
    model_uri = "runs:/" + mlflow.active_run().info.run_id + "/model"
    local_path = _download_artifact_from_uri(model_uri)

    final_model_path = Path(args.model_output)
    final_model_path.mkdir(parents=True, exist_ok=True)

    shutil.copytree(local_path, final_model_path, dirs_exist_ok=True)

    print(f"✅ MLflow model directory copied to output path: {final_model_path}")

    # Debug: Print contents of model output folder
    print("📁 Output model directory contents:")
    for root, dirs, files in os.walk(final_model_path):
        level = root.replace(str(final_model_path), '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

    mlflow.end_run()

if __name__ == "__main__":
    args = parse_args()

    print(f"Train dataset path: {args.train_data}")
    print(f"Test dataset path: {args.test_data}")
    print(f"Model output path: {args.model_output}")
    print(f"Model type: {args.model_type}")
    print(f"n_estimators: {args.n_estimators}")
    print(f"max_depth: {args.max_depth}")
    print(f"criterion: {args.criterion}")

    main(args)
