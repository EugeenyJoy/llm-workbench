# ==========================================
# БАЗОВАЯ АРХИТЕКТУРА GPT (Transformer decoder)
# ==========================================
# Это ЯДРО модели.
# Здесь нет ускорений или внешних библиотек.
#
# Именно эта архитектура масштабируется
# до больших LLM.
# ==========================================

import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self,n_embd,n_head):

        super().__init__()

        # ==============================
        # SELF ATTENTION
        # ==============================

        self.attn = nn.MultiheadAttention(
            n_embd,
            n_head,
            batch_first=True
        )

        self.ln1 = nn.LayerNorm(n_embd)

        # ==============================
        # FEEDFORWARD СЕТЬ
        # ==============================

        self.ff = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd,n_embd)
        )

        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self,x):

        # attention + residual connection
        attn,_ = self.attn(x,x,x)

        x = x + attn

        # feedforward + residual
        x = x + self.ff(self.ln2(x))

        return x



class GPT(nn.Module):

    def __init__(self,config):

        super().__init__()

        # ==================================
        # TOKEN EMBEDDINGS
        # ==================================

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # ==================================
        # POSITION EMBEDDINGS
        # ==================================

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        # ==================================
        # STACK OF TRANSFORMER BLOCKS
        # ==================================

        self.blocks = nn.ModuleList([
            Block(config.n_embd,config.n_head)
            for _ in range(config.n_layer)
        ])

        self.ln = nn.LayerNorm(config.n_embd)

        # ==================================
        # OUTPUT LAYER
        # ==================================

        self.head = nn.Linear(
            config.n_embd,
            config.vocab_size
        )

        self.block_size = config.block_size


    def forward(self,idx):

        B,T = idx.shape

        tok = self.token_embedding(idx)

        pos = self.position_embedding(
            torch.arange(T,device=idx.device)
        )

        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)

        logits = self.head(x)

        return logits