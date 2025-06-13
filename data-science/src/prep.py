# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Prepares raw data and provides training and test datasets.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description="Data preparation script for regression.")
    parser.add_argument("--raw_data", type=str, help="Path to raw data", required=True)
    parser.add_argument("--train_data", type=str, help="Path to save processed training data", required=True)
    parser.add_argument("--test_data", type=str, help="Path to save processed testing data", required=True)
    parser.add_argument("--test_train_ratio", type=float, default=0.2, help="Test-train ratio")
    return parser.parse_args()


def main(args):
    '''Read, preprocess, split, and save datasets'''

    print(f"Reading data from: {args.raw_data}")
    df = pd.read_csv(args.raw_data, delimiter='\t')  # Change delimiter if needed

    # === Data Preprocessing ===
    # Fill missing numeric values
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # One-hot encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

    # Assume 'price' is the target column for regression
    if 'price' not in df.columns:
        raise ValueError("Target column 'price' not found in dataset.")

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_train_ratio, random_state=42
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Ensure output directories exist
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    # Write to CSV
    train_output_path = os.path.join(args.train_data, 'train.csv')
    test_output_path = os.path.join(args.test_data, 'test.csv')
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    # === MLflow Logging ===
    mlflow.log_metric("train_rows", train_df.shape[0])
    mlflow.log_metric("test_rows", test_df.shape[0])

    print(f"Training data saved to: {train_output_path}")
    print(f"Testing data saved to: {test_output_path}")


if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Test dataset path: {args.test_data}",
        f"Test-train ratio: {args.test_train_ratio}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
