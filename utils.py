import torch
import numpy as np
import random
from model import CNNCifar

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def start_attack(selfish_loss, max_gap, each_epoch, attack_epoch, epoch_interval=50, epsilon=0.1):
    if attack_epoch == float("inf") and each_epoch > epoch_interval:
        gap = np.min(selfish_loss[1: each_epoch-epoch_interval+1]) - np.min(selfish_loss[1:])
        max_gap = max(max_gap, gap)
        if each_epoch % 10 == 0:
            print("=" * 100)
            # print(selfish_loss)
            print("max loss gap: ", max_gap, ", loss gap: ", gap)
        if 0 < gap < epsilon * max_gap:
            attack_epoch = each_epoch
            print("Start Attack!!!")
    return attack_epoch, max_gap

def create_model(model_name):
    if model_name == "cnnc":
        return CNNCifar(3, 10)
    else:
        raise NotImplementedError