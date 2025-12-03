import argparse
import os
import sys

# Add subfolders to path
sys.path.extend([os.path.join(os.path.dirname(__file__), f)
                 for f in ["train_xception", "train_transformer", "preprocessing", "evaluation"]])

from train_xception.train_xception import train_xception
from train_transformer.train_transformer import train_transformer
from preprocessing.prepare_dataset import preprocess_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['xception', 'vit'], required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=2)  # reduced workers
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--pretrained', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Normalize data_dir path and construct manifest path
    args.data_dir = os.path.normpath(args.data_dir)
    manifest_path = os.path.join(args.data_dir, "manifest.csv")
    print("Looking for manifest at:", manifest_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"{manifest_path} does not exist! Create a CSV listing all images and metadata.")
    print("Using manifest:", manifest_path)

    # Preprocess dataset and generate CSV splits
    train_csv, val_csv, test_csv = preprocess_dataset(manifest_path, args.output_dir)
    print("Splits created:\n Train:", train_csv, "\n Val:", val_csv, "\n Test:", test_csv)

    # Train selected model
    if args.model == 'xception':
        train_xception(train_csv, val_csv, test_csv, args)
    elif args.model == 'vit':
        train_transformer(train_csv, val_csv, test_csv, args)
