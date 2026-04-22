"""Stub DCNv2 for CPU/non-CUDA environments (hourglass model won't use this)."""
import torch.nn as nn

class DCN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("DCNv2 requires CUDA. Use hourglass architecture instead.")
