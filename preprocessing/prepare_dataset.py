# preprocessing/prepare_dataset.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_dataset(manifest_csv_path, output_dir):

    # Manifest must be exactly the path passed from train.py
    if not os.path.exists(manifest_csv_path):
        raise FileNotFoundError(
            f"{manifest_csv_path} does not exist. "
            "Create a CSV listing all images and metadata."
        )

    print("Using manifest:", manifest_csv_path)

    df = pd.read_csv(manifest_csv_path)

    # Basic validations
    if "image_path" not in df.columns:
        raise ValueError("Manifest must contain 'image_path' column")

    if len(df) == 0:
        raise ValueError("Manifest.csv is empty!")

    # Split dataset
    train_df, temp = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42)

    # Output CSVs
    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Splits created:")
    print(" Train:", train_path)
    print(" Val:", val_path)
    print(" Test:", test_path)

    return train_path, val_path, test_path
