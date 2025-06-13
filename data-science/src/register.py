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

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(description="Model registration script for sweep output.")
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered', required=True)
    parser.add_argument('--model_path', type=str, help='Model directory', required=True)
    parser.add_argument('--model_info_output_path', type=str, help='Path to write model info JSON', required=True)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')
    return args

def main(args):
    '''Loads the best-trained model and registers it'''
    print(f"Registering model '{args.model_name}' from path '{args.model_path}'")

    # --- Path validation ---
    received_model_path = Path(args.model_path)
    cleaned_model_path = received_model_path.resolve()

    if "${{name}}" in str(cleaned_model_path):
        raise ValueError(
            f"Model path still contains unexpanded Azure ML placeholder: '{cleaned_model_path}'."
        )

    if not cleaned_model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: '{cleaned_model_path}'")
    if not cleaned_model_path.is_dir():
        raise ValueError(f"Model path is not a directory: '{cleaned_model_path}'")
    if not (cleaned_model_path / "MLmodel").exists():
        raise ValueError(f"MLmodel file not found in directory: '{cleaned_model_path}'")

    # --- Register model with MLflow ---
    try:
        model_uri = f"file://{cleaned_model_path}"
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=args.model_name,
            tags={"registered_by_aml_pipeline": True}
        )

        print(f"✅ Model registered: name='{registered_model.name}', version='{registered_model.version}'")

        # Save model info to output path
        os.makedirs(os.path.dirname(args.model_info_output_path), exist_ok=True)
        with open(args.model_info_output_path, "w") as f:
            json.dump(
                {"model_name": registered_model.name, "version": registered_model.version},
                f,
                indent=4
            )
        print(f"ℹ️ Model info saved to: {args.model_info_output_path}")

    except Exception as e:
        print(f"❌ Error registering model: {e}")
        raise

if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Model info output path: {args.model_info_output_path}")

    main(args)

    mlflow.end_run()
