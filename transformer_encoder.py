import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


# layer normalization
class Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # return x
        z = (x - x.mean(dim=-1, keepdim=True)) / \
            (x.std(dim=-1, keepdim=True) + self.eps)
        x = self.alpha*z + self.bias
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.d_k = dim // num_head
        self.num_head = num_head
        self.dropout = dropout

        self.q = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def attention(self, q, k, v, mask):
        # torch.Size([8, 4, 10, 10]) = batch_size, num_head, LqxLk
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        score = torch.clamp_min(score, -6e3)
        if mask is not None:
            mask = mask.unsqueeze(1)
            # print(score.min())
            # score = score.masked_fill(mask == 0, -6e6) #-65504
            score = score.masked_fill(mask == 0, -6e4)  # -65504
            #score = score.masked_fill(mask == 0, -half('inf'))
            # https://github.com/NVIDIA/apex/issues/93
            # How to use fp16 training with masked operations

        score = F.softmax(score, dim=-1)

        if self.dropout > 0:
            score = F.dropout(score, self.dropout, training=self.training)

        value = torch.matmul(score, v)
        return value, score

    def forward(self, q, k, v, mask=None):
        batch_size, T, dim = q.shape

        # perform linear operation and split into h heads
        k = self.k(k).reshape(batch_size, -1, self.num_head, self.d_k)
        q = self.q(q).reshape(batch_size, -1, self.num_head, self.d_k)
        v = self.v(v).reshape(batch_size, -1, self.num_head, self.d_k)

        # transpose to get dimensions batch_size * num_head * T * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        value, score = self.attention(q, k, v, mask)

        # concatenate heads and put through final linear layer
        value = value.transpose(1, 2).contiguous().reshape(
            batch_size, -1, self.dim)
        value = self.out(value)
        return value, score


# ---

class TransformerEncodeLayer(nn.Module):
    def __init__(self, dim, ff_dim, num_head, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(dim).cuda()
        self.norm2 = Norm(dim).cuda()

        self.attn = MultiHeadAttention(dim, num_head, dropout=0.1).cuda()
        self.ff = FeedForward(dim, ff_dim).cuda()

        self.dropout1 = nn.Dropout(dropout).cuda()
        self.dropout2 = nn.Dropout(dropout).cuda()

    def forward(self, x, x_mask):
        x1, scores = self.attn(x, x, x)  # self-attention
        x1 = x + self.dropout1(x1)
        x = self.norm1(x1)

        x2 = self.ff(x)
        x2 = x + self.dropout2(x2)
        x = self.norm2(x2)
        return x, scores
