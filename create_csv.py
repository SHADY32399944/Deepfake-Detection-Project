import os
import pandas as pd

dataset_dir = r"C:\Users\user\Desktop\Deeptrack\Model_benchmark\dataset"
rows = []

for root, _, files in os.walk(os.path.join(dataset_dir, "images")):
    for f in files:
        if f.lower().endswith((".jpg", ".png")):
            label = 0 if "real" in root.lower() else 1
            identity = os.path.basename(root)  # or modify if you have subfolders per person
            gen_type = "real" if label == 0 else "faceswap"
            rel_path = os.path.relpath(os.path.join(root, f), dataset_dir)  # relative to dataset folder
            rows.append([rel_path, label, identity, gen_type])

df = pd.DataFrame(rows, columns=["image_path", "label", "identity_id", "generator_type"])
manifest_path = os.path.join(dataset_dir, "manifest.csv")
df.to_csv(manifest_path, index=False)
print(f"manifest.csv created at {manifest_path} with {len(df)} entries")
