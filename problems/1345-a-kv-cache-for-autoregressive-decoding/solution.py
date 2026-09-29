import torch
import torch.nn as nn
import math

class KVCache:
    def __init__(self):
        self.reset()

    def append(self, k, v):
        if self.k is None:
            self.k, self.v = k, v 
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return (self.k, self.v)
    
    def __len__(self):
        return self.k.size(2) if self.k is not None else 0

    def reset(self):
        self.k = None 
        self.v = None 

def softmax(z, eps=1e-05):
    z = z - z.max(dim=-1)
    exponentials = torch.exp(z)
    return exponentials / torch.sum(exponentials, dim=-1)

def attend_with_cache(q, k_new, v_new, cache):
    # TODO: append the new keys/values, then attend q over the full cache
    # TODO: softmax(q @ k^T / sqrt(head_dim)) @ v
    k, v = cache.append(k_new, v_new)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    return torch.softmax(scores, dim=-1) @ v
