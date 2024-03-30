import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

from tinytorchutil import *
import numpy as np

import fastcore.all as fc

from tokenizer import *

# Hyperparameters
block_size = 8
train_split = .9
batch_size = 32
dropout = 0.1
n_hidden = 4 # size of hidden layers in FeedForward layer after multihead attention block
n_head = 6
embed_size = 384
n_atn_block = 6
lr = 3e-4
n_epoch = 2000
tokenizer = 'BPE' # BPE or CHAR
bpe_merges_file = 'bpe_tiny_shakespear.bpe' if tokenizer=='BPE' else None

# Load and preprocess data
with open('dataset.txt', 'r', encoding='utf-8') as file:
    data = file.read()

if tokenizer=='CHAR':
    tokenizer = CharacterTokenizer(data)
elif tokenizer=='BPE':
    tokenizer = BPETokenizer()
    tokenizer.load(bpe_merges_file)

class CharSeqDataset():
    def __init__(self, x, block_size): self.x,self.block_size = x, block_size
    def __len__(self): return len(self.x)-block_size
    def __getitem__(self, i): 
        seq = torch.tensor(self.x[i:i+self.block_size], dtype=torch.long)
        target = torch.tensor(self.x[i+1:i+self.block_size+1], dtype=torch.long)
        return seq, target

tokens = tokenizer.encode(data)
n = len(tokens)*train_split
tds = CharSeqDataset(tokens[:n], block_size)
vds = CharSeqDataset(tokens[n:], block_size)

dls = DataLoaders(tds, vds, bs=batch_size)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        bs, ts, ch = x.shape # batch_size, time_step, channel_size/embedding_size
        affinity = queries @ keys.transpose(-2, -1) * keys.shape[-1]**-0.5 # scaling the values by inverse root of the number of values
        affinity.masked_fill_(self.tril[:ts][:ts]==0, float('-inf'))
        
        affinity = F.softmax(affinity)
        affinity = self.dropout(affinity)
        attention = affinity @ values # weighted attention
        return attention

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size)] * n_head)
        self.ln = nn.Linear(n_head * head_size, embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        multi_atn = torch.cat([head(x) for head in self.heads], dim=-1)
        multi_atn = self.ln(multi_atn)
        multi_atn = self.dropout(multi_atn)
        return multi_atn
    
class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(embed_size, embed_size * n_hidden),
            nn.ReLU(),
            nn.Linear(embed_size * n_hidden, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.seq(x)
    
class Block(nn.Module):
    def __init__(self, embed_size, n_head):
        super().__init__()
        self.head_size = embed_size // n_head
        self.multihead = MultiHeadedAttention(n_head, self.head_size)
        self.fdfwd = FeedForward(embed_size)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        out = self.multihead(x)
        out = self.layernorm1(out + x)
        out = self.fdfwd(out)
        out = self.layernorm2(out + x)

class TransformerGPTModel(nn.Module):
    def __init__(self, embed_size, n_block, n_head, block_size, vocab_size):
        fc.store_attr()
        self.inp_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(block_size, embed_size)
        self.atn_blks = nn.Sequential(*[Block(embed_size, n_head) for _ in range(n_block)])
        self.layernorm = nn.LayerNorm(embed_size)
        self.lin = nn.Linear(embed_size, vocab_size)

        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0, std = 0.02)
        elif isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0, std = 0.02)
            if module.bias is not None:
                init.zeros_(module.bias)

    def forward(self, x):
        bs, ts = x.shape
        x_emb = self.inp_emb(x)
        p_emb = self.pos_emb(torch.arange(ts, device=def_device))
        out = self.atn_blks(x_emb + p_emb)
        out = self.layernorm(out)
        logits = self.lin(out)
        return logits

    def generate(self, x, max_token_new):
        for _ in range(max_token_new):
            block = x[-self.block_size:]
            logits = self(block.view(1,-1))
            logits = logits[-1, -1, :]
            pb = F.softmax(logits, dim=-1)
            token_pred = torch.multinomial(pb, num_samples=1)
            x = torch.cat((x, token_pred))
        return x

class GPTLearner(TrainLearner):
    def get_loss(self):
        bs, ts, ch = self.preds.shape
        self.loss = self.loss_func(self.preds.view(bs*ts, ch), self.batch[1].view(bs*ts))

model = TransformerGPTModel(embed_size, n_atn_block, n_head, block_size, tokenizer.vocab_size)
opt = torch.optim.AdamW
loss_func = F.cross_entropy
cbs = [ProgressCB(), MetricsCB(), DeviceCB()]

learner = GPTLearner(model, dls, loss_func, lr, cbs, opt)
learner.fit(n_epoch)

x = torch.zeros((1), dtype=torch.long, device=def_device)
preds = model.generate(x, max_token_new=200).tolist()
print(tokenizer.decode(preds))