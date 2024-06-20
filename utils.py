import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def loss_fn(pred, target, dice, focal):
    return dice(pred, target, use_sigmoid=True) + focal(pred, target)
