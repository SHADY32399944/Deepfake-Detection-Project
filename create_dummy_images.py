from PIL import Image
import os

# base path for images
base_path = r"C:\Users\user\Desktop\Deeptrack\Model_benchmark\dataset\images"

# create real and fake subfolders
for folder in ["real", "fake"]:
    folder_path = os.path.join(base_path, folder)
    os.makedirs(folder_path, exist_ok=True)
    for i in range(5):  # create 5 dummy images per folder
        img = Image.new("RGB", (224, 224), color=(i*40, i*40, i*40))
        img.save(os.path.join(folder_path, f"img{i}.jpg"))

print("Dummy images created at:", base_path)
