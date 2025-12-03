import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from timm import create_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Custom dataset to load images from CSV
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        label = int(row['label'])
        if self.transform:
            img = self.transform(img)
        return img, label

def train_transformer(train_csv, val_csv, test_csv, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training Transformer model...")
    print("Train CSV:", train_csv)
    print("Val CSV:", val_csv)
    print("Test CSV:", test_csv)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Num workers:", args.num_workers)

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # Datasets and loaders
    train_dataset = ImageDataset(train_csv, transform)
    val_dataset = ImageDataset(val_csv, transform)
    test_dataset = ImageDataset(test_csv, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Vision Transformer model
    model = create_model('vit_base_patch16_224', pretrained=args.pretrained, num_classes=len(train_dataset.data['label'].unique()))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Store metrics
    metrics_summary = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} - Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} - Validation"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss = criterion(outputs, labels)
                val_loss_total += val_loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss_avg = val_loss_total / len(val_dataset)
        val_acc = accuracy_score(val_labels, val_preds)

        metrics_summary['epoch'].append(epoch)
        metrics_summary['train_loss'].append(train_loss)
        metrics_summary['val_loss'].append(val_loss_avg)
        metrics_summary['val_acc'].append(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss_avg:.4f}, val_acc={val_acc:.4f}")

    # Final evaluation on test set
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, average='weighted')
    test_rec = recall_score(test_labels, test_preds, average='weighted')
    test_f1 = f1_score(test_labels, test_preds, average='weighted')

    print("\nTransformer Model Evaluation on Test Set:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

    # Summary table
    print("\nModel Output Summary")
    print("Metric       Transformer")
    print(f"Accuracy     {test_acc:.4f}")
    print(f"Precision    {test_prec:.4f}")
    print(f"Recall       {test_rec:.4f}")
    print(f"F1           {test_f1:.4f}")
