import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
import os
import sys
import tiktoken
from PIL import Image

@dataclass
class OCRConfig:
    n_layers:int = 6
    vocab_size:int = 100
    n_heads:int = 8
    n_embed:int = 512
    batch_size:int = 32
        
class Self_Attention(nn.Module):
    def __init__(self, config:OCRConfig):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        self.c_attn = nn.Linear(config.n_embed, config.n_embed * 4)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_heads = config.n_heads
        self.n_embed = config.n_embed
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C//self.n_heads).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
    
    
class Block(nn.Module):
    def __init__(self, config:OCRConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = Self_Attention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
        
        
class MLP(nn.Module):
    def __init__(self, config:OCRConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, config.n_embed * 4)
        self.c_proj = nn.Linear(config.n_embed * 4, config.n_embed)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
    
    
    
class OCR(nn.Module):
    def __init__(self, config:OCRConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, idx, targets=None):
        _, T = idx.size()
        assert T <= self.config.block_size 
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer['wpe'](pos)
        tok_emb = self.transformer['wte'](idx)
        x = pos_emb + tok_emb
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else: 
            loss = None
        return logits, loss
    
class TextLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        data_path = "data/ocr.txt"

        with open(data_path, 'r') as f:
            data = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(data, allowed_special={"<|endoftext|>"})
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        print(f"Total tokens: {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current = 0

    def next_batch(self):
        B, T = self.B, self.T
        buff = self.tokens[self.current:self.current + B * T + 1]
        x = buff[:-1].view(B, T)
        y = buff[1:].view(B, T)
        self.current += B * T

        if self.current + B * T + 1 >= len(self.tokens):
            self.current = 0

        return x, y 
    
    
class ImageLaTeXDataset(Dataset):
    def __init__(self, image_paths, latex_texts, tokenizer, transform=None, max_length=512):
        self.image_paths = image_paths
        self.latex_texts = latex_texts
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        text = self.latex_texts[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
        return image, encoding['input_ids'].squeeze()