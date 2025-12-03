import pandas as pd
import os

# Path to your manifest.csv
manifest_path = r"C:\Users\user\Desktop\Deeptrack\Model_benchmark\dataset\manifest.csv"

# Read CSV
df = pd.read_csv(manifest_path)

# Inspect
print(df.head())
print("Total samples:", len(df))
