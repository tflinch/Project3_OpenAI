# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os
import json
import sys

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description="Model registration script for sweep output.")
    parser.add_argument('--model_name', type=str, required=True, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, required=True, help='Model directory')
    parser.add_argument('--model_info_output_path', type=str, required=True, help='Path to write model info JSON')
    args = parser.parse_args()
    print(f"[INFO] Arguments: {args}")
    return args

def main(args):
    '''Loads the best-trained model and registers it'''
    print(f"[INFO] Registering model '{args.model_name}' from path '{args.model_path}'")

    try:
        received_model_path = Path(args.model_path)
        cleaned_model_path = received_model_path.resolve()
        print(f"[DEBUG] Resolved model path: {cleaned_model_path}")

        # Early check for unresolved AzureML placeholders
        if "${{" in str(cleaned_model_path)}:
            raise ValueError(f"[ERROR] Model path still contains unexpanded AzureML placeholder: {cleaned_model_path}")

        # Path existence check
        if not cleaned_model_path.exists():
            raise FileNotFoundError(f"[ERROR] Model path does not exist: {cleaned_model_path}")
        if not cleaned_model_path.is_dir():
            raise ValueError(f"[ERROR] Model path is not a directory: {cleaned_model_path}")

        print(f"[DEBUG] Directory contents: {os.listdir(cleaned_model_path)}")

        if not (cleaned_model_path / "MLmodel").exists():
            raise FileNotFoundError(f"[ERROR] 'MLmodel' file not found in: {cleaned_model_path}")

        # Register with MLflow
        mlflow.set_tracking_uri("azureml://tracking")
        model_uri = f"file://{cleaned_model_path}"
        print(f"[DEBUG] Model URI: {model_uri}")

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=args.model_name,
            tags={"registered_by_aml_pipeline": True}
        )

        print(f"[SUCCESS] Model registered: name='{registered_model.name}', version='{registered_model.version}'")

        # Save model info
        output_path = Path(args.model_info_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {"model_name": registered_model.name, "version": registered_model.version},
                f,
                indent=4
            )
        print(f"[INFO] Model info saved to: {output_path}")

    except Exception as e:
        print(f"[ERROR] Exception occurred during model registration:\n{e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()
