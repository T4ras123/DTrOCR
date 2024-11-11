import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import adamw
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
  def __init__(self, in_channels=3, patch_size=8, emb_siz=128):
    super().__init__()
    self.patch_size = patch_size
    self.projection = nn.Sequential(
      Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), 
      nn.Linear(patch_size * patch_size * in_channels, emb_siz)
    )
  
  def forward(self, x):
    return self.projection(x)
  

class Attention(nn.Module):
  