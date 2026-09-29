import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    optimizer.zero_grad()    
    ypred = model(x)
    mse = F.mse_loss(ypred, y)
    mse.backward()
    optimizer.step()
    return mse.item()