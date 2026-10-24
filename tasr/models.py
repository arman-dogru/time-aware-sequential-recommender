"""Tiny causal sequential encoders: SASRec-style (softmax attention) and HSTU-style (pointwise SiLU
attention with gating), with switchable time features:

  pos   : 'abs' learned absolute positions | 'rel' relative-position attention bias | 'none'
  trel  : relative *time* attention bias  b(i,j) = w_h[bucket(q_i - t_j)], where q_i is the query
          time of position i (timestamp of the event being predicted) -- as in HSTU's
          RelativeBucketedTimeAndPositionBasedBias; buckets are log_{sqrt 2} of the gap in seconds.
  trope : time rotary encoding -- q_i rotated by omega * q_i, k_j by omega * t_j (seconds), periods
          geometric from 1 minute to ~4 years, applied to half of each head's dims.
  tabs  : absolute calendar embeddings (hour-of-day, day-of-week, month-index) of each event
          added to the input, and of the query time added to the output state.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TBUCKETS = 64


def time_bucket(dt):
    dt = dt.clamp(min=0).float()
    return (2.0 * torch.log2(1.0 + dt)).long().clamp(0, N_TBUCKETS - 1)


class TimeRoPE(nn.Module):
    def __init__(self, rot_dim, min_period=60.0, max_period=4 * 365 * 86400.0):
        super().__init__()
        n = rot_dim // 2
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), n)
        self.register_buffer("omega", 2 * math.pi / periods)
        self.rot_dim = rot_dim

    def forward(self, x, t):  # x (B,H,L,dh), t (B,L) seconds relative to a reference
        ang = t.double()[:, None, :, None] * self.omega.double()  # double: t ~ 1e8 s
        cos, sin = torch.cos(ang).float(), torch.sin(ang).float()
        r, p = x[..., : self.rot_dim], x[..., self.rot_dim:]
        r1, r2 = r[..., ::2], r[..., 1::2]
        out = torch.stack([r1 * cos - r2 * sin, r1 * sin + r2 * cos], -1).flatten(-2)
        return torch.cat([out, p], -1)


class Block(nn.Module):
    def __init__(self, arch, d, heads, L, dropout, pos, trel, trope):
        super().__init__()
        self.arch, self.h, self.dh, self.L = arch, heads, d // heads, L
        self.ln1 = nn.LayerNorm(d)
        if arch == "hstu":
            self.uvqk = nn.Linear(d, 4 * d)
            self.ln_attn = nn.LayerNorm(d)
            self.out = nn.Linear(d, d)
        else:
            self.qkv = nn.Linear(d, 3 * d)
            self.out = nn.Linear(d, d)
            self.ln2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.drop = nn.Dropout(dropout)
        self.pos_bias = nn.Parameter(torch.zeros(heads, L)) if pos == "rel" else None
        self.time_bias = nn.Parameter(torch.zeros(heads, N_TBUCKETS)) if trel else None
        self.rope = TimeRoPE(self.dh // 2 if self.dh // 2 % 2 == 0 else self.dh // 2 - 1) if trope else None

    def bias(self, B, L, tb, device):
        b = torch.zeros(1, self.h, L, L, device=device)
        if self.pos_bias is not None:
            i = torch.arange(L, device=device)
            rel = (i[:, None] - i[None, :]).clamp(0, self.L - 1)
            b = b + self.pos_bias[:, rel][None]
        if self.time_bias is not None:
            b = b + self.time_bias[:, tb].permute(1, 0, 2, 3)  # tb (B,L,L) -> (B,H,L,L)
        return b

    def forward(self, x, mask, tb, qt, kt):
        B, L, d = x.shape
        y = self.ln1(x)
        if self.arch == "hstu":
            u, v, q, k = F.silu(self.uvqk(y)).chunk(4, -1)
        else:
            q, k, v = self.qkv(y).chunk(3, -1)
        sh = lambda z: z.view(B, L, self.h, self.dh).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        if self.rope is not None:
            q, k = self.rope(q, qt), self.rope(k, kt)
        logits = q @ k.transpose(-1, -2)
        bias = self.bias(B, L, tb, x.device)
        if self.arch == "hstu":
            a = F.silu(logits + bias) / self.L * mask
            o = (a @ v).transpose(1, 2).reshape(B, L, d)
            o = self.out(self.drop(self.ln_attn(o) * u))
            return x + self.drop(o)
        logits = logits / math.sqrt(self.dh) + bias
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
        a = self.drop(torch.softmax(logits, -1).nan_to_num(0.0))
        o = (a @ v).transpose(1, 2).reshape(B, L, d)
        x = x + self.drop(self.out(o))
        return x + self.drop(self.ffn(self.ln2(x)))


class SeqRec(nn.Module):
    def __init__(self, n_items, arch="hstu", d=64, layers=2, heads=2, L=200, dropout=0.2,
                 pos="rel", trel=True, trope=False, tabs=False, t_ref=956_700_000):
        super().__init__()
        self.cfg = dict(arch=arch, d=d, layers=layers, heads=heads, L=L, pos=pos, trel=trel, trope=trope, tabs=tabs)
        self.emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.abs_pos = nn.Embedding(L, d) if pos == "abs" else None
        self.tabs = tabs
        if tabs:
            self.cal_in = nn.ModuleList([nn.Embedding(24, d), nn.Embedding(7, d), nn.Embedding(48, d)])
            self.cal_q = nn.ModuleList([nn.Embedding(24, d), nn.Embedding(7, d), nn.Embedding(48, d)])
        self.blocks = nn.ModuleList(Block(arch, d, heads, L, dropout, pos, trel, trope) for _ in range(layers))
        for m in self.modules():  # small init for every embedding (default N(0,1) would swamp item embeddings)
            if isinstance(m, nn.Embedding) and m is not self.emb:
                nn.init.normal_(m.weight, std=0.02)
        self.ln_f = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.trel = trel
        self.t_ref = t_ref  # ~ 2000-04-26, first ML-1M rating

    @staticmethod
    def _cal(mods, t):
        t = t.clamp(min=0)
        hour = (t // 3600) % 24
        dow = (t // 86400 + 3) % 7  # 1970-01-01 was a Thursday
        month = ((t - 946_684_800).clamp(min=0) // (30 * 86400)).clamp(max=47)  # months since 2000-01
        return mods[0](hour) + mods[1](dow) + mods[2](month)

    def encode(self, x, t, q):
        """x,t,q: (B,L) item ids, event ts, query ts (time of the event predicted at that position)."""
        B, L = x.shape
        valid = (x > 0)
        h = self.emb(x) * math.sqrt(self.emb.embedding_dim)
        if self.abs_pos is not None:
            h = h + self.abs_pos(torch.arange(L, device=x.device))[None]
        if self.tabs:
            h = h + self._cal(self.cal_in, t) * valid[..., None]
        h = self.drop(h) * valid[..., None]
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        mask = (causal[None] & valid[:, None, :])[:, None].float()  # (B,1,L,L): query i attends key j<=i
        tb = time_bucket(q[:, :, None] - t[:, None, :]) if self.trel else None
        qt, kt = (q - self.t_ref).clamp(min=0), (t - self.t_ref).clamp(min=0)
        for blk in self.blocks:
            h = blk(h, mask, tb, qt, kt) * valid[..., None]
        h = self.ln_f(h)
        if self.tabs:
            h = h + self._cal(self.cal_q, q)
        return F.normalize(h, dim=-1)

    def item_table(self):
        return F.normalize(self.emb.weight, dim=-1)
