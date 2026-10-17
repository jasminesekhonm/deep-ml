import numpy as np

def top_p_sampling(logits, p):
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")

    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    order = np.argsort(-probs, kind="stable")            
    cumulative = np.cumsum(probs[order])
    cutoff = int(np.searchsorted(cumulative, p)) + 1  
    keep = order[:cutoff]

    filtered = np.zeros_like(probs)                    
    filtered[keep] = probs[keep]
    return (filtered / filtered.sum()).tolist()