import pandas as pd
import os

dataset_root = r"C:\Users\user\Desktop\Deeptrack\Model_benchmark\dataset"
csv_file = os.path.join(dataset_root, "manifest.csv")

df = pd.read_csv(csv_file)

def fix_path(p):
    return os.path.join(dataset_root, p.replace('/', '\\'))

df['image_path'] = df['image_path'].apply(fix_path)

df.to_csv(csv_file, index=False)

print("CSV paths updated successfully!")
