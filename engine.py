import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(loader, desc='train'):
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device, compute_roc=False):
    model.eval()
    ys, ys_pred = [], []
    running_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='eval'):
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy().ravel()
            ys_pred.extend(probs.tolist())
            ys.extend(labels.cpu().numpy().ravel().tolist())

    avg_loss = running_loss / len(loader.dataset)
    preds = [1 if p>=0.5 else 0 for p in ys_pred]
    precision, recall, f1, _ = precision_recall_fscore_support(ys, preds, average='binary', zero_division=0)
    acc = accuracy_score(ys, preds)
    roc_auc = roc_auc_score(ys, ys_pred) if compute_roc else None
    return avg_loss, {'precision':precision,'recall':recall,'f1':f1,'accuracy':acc,'roc_auc':roc_auc}
