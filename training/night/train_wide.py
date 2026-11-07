import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train
from wide_model import WideActorCritic

def load_wide(device):
    return WideActorCritic().to(device)

train.load_default = load_wide
train.main()
