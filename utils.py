import random, numpy as np, torch, json
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, epoch, outdir, name='checkpoint.pt'):
    path = Path(outdir)/name
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch}, path)

def write_metrics(metrics, path):
    with open(path,'w') as f:
        json.dump(metrics,f,indent=2)
