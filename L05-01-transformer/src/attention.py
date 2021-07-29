import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads. Receive/output {\R \in 256}
        self.key = nn.Linear(config.n_embd, config.n_embd)  # bias True by default
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        # B: Batch size (one sentence per batch?)
        # T: Input lenght (words on current sentence)
        # C: Embed. size
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) ... hs = C/nh
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # k.size(-1) is hs
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

"""
Write your SynthesizerAttention below.
Hint: paste over the CausalSelfAttention above and modify it minimally.
"""

class SynthesizerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # NEW learnable weights. w_1 is a linear transformation, w2 is not!
        self.w1 = nn.Linear(config.n_embd, config.n_embd)  # This is A_i
        self.w2 = nn.Parameter(  # This is B_i, matrix of shape (d//h, l)
            torch.zeros(
                config.n_embd // config.n_head,
                config.block_size-1
                ))
        self.b2 = nn.Parameter(torch.zeros(config.block_size-1))
        # value projection
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in
        #     the input sequence
        self.register_buffer("mask", torch.tril(
            torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.block_size = config.block_size

        nn.init.uniform_(self.w2,-0.001,0.001)

    def forward(self, x, layer_past=None):
        # [part g]: Write your SynthesizerAttention below.
        #   Do not modify __init__().

        # b: Batch size (one sentence per batch?)
        # l: Input lenght (words on current sentence)
        # d: Embed. size
        b, l, d = x.size()
        h = self.n_head
        
        # Linear layer w1 will transform x and output y = xA+b. Since x is a
        # multi. dim tensor, the transformation will be applied only to last 
        # dim of x. This is why when defining nn.Linear we say the input is
        # a vector of features, and the output is also a transformed feature vector
        # Notice, it is the same (transform x with w1 -> reshape -> ReLU) and
        # (transform -> ReLU -> reshape)
        relu_input = self.w1(x)  # (b, l, d)
        relu = F.relu(relu_input)  # (b, l, d)

        # We need to @ relu output with B of shape (d/h, l), then we need to reshape
        # relu output in a way we need to keep the number of features in the tensor
        # i.e. b*l*h*d/h = b*l*d. Notice x.view(b, l, h, d//h).transpose(1,2) != x.view(b, h, l, d//h)
        # Note: For evaluation time it is needed to make w2.unsqueeze(0).unsqueeze(0)[..., :l] to 
        # shape it as (1, 1, d//h, l) before @ with relu, same happens with the
        # sum with b2 (with b2[..., :l]) at training time PyTorch can handle those changes,
        # not sure why, perhaps because at eval time l has different size
        w2 = self.w2.unsqueeze(0).unsqueeze(0)[..., :l]  # (1, 1, d//h, l)
        b2 = self.b2[..., :l]

        softmax_input = relu.view(b, l, h, d//h).transpose(1,2) @ w2 + b2  # (b, h, l, l)
        softmax_input = softmax_input.masked_fill(self.mask[:,:,:l,:l] == 0, -1e10)
        softmax = F.softmax(softmax_input, dim=-1)  # (b, h, l, l)
        softmax = self.attn_drop(softmax)  # (b, h, l, l)

        # Linear value outputs a shape of (b, l, d), we need to reshape and transpose
        value = self.value(x).view(b, l, h, d//h).transpose(1,2)  # (b, h, l, d//h)
        y_i = softmax @ value  # (b, h, l, d//h)

        # Concatenate all y_i: re-assemble all head outputs side by side
        y = y_i.transpose(1,2).contiguous().view(b, l, d)

        # The final output projection
        self.resid_drop(self.proj(y))

        return y