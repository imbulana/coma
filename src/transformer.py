# adapted from https://github.com/lucidrains/conformer

import torch
from torch import nn, einsum, arange, cat
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from src.local_attention import LocalAttention

# helper functions

def is_masked(x, mask):
    return (x.masked_select(~mask.unsqueeze(-1)).abs() < 1e-5).all()

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

def l2norm(tensor):
    dtype = tensor.dtype
    normed = F.normalize(tensor, dim = -1)
    return normed.type(dtype)

# helper classes

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(
        self,
        x,
        pos = None,
        seq_start_pos = None,
        offset = 0,
        mask = None
    ):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = arange(seq_len, device = device) + offset

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = cat((emb.sin(), emb.cos()), dim = -1)
        emb = emb * self.scale
        
        out = emb + x

        if exists(mask):
            out = out.masked_fill(~mask.unsqueeze(-1), 0.)
        
        return out

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

# attention, feedforward, and conv module

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x, mask = None):
        if exists(mask):
            mask = rearrange(mask, 'b n -> b 1 n')
            x = x.masked_fill(~mask, 0.)

        x = F.pad(x, self.padding)
        out = self.conv(x)

        if exists(mask):
            out = out.masked_fill(~mask, 0.)
        return out

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class LocalMHA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        window_size,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        prenorm = True,
        qk_rmsnorm = False,
        qk_scale = 8,
        use_xpos = False,
        xpos_scale_base = None,
        exact_windowsize = None,
        gate_values_per_head = False,
        **kwargs
    ):
        super().__init__()        
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.causal = causal
        self.window_size = window_size
        self.exact_windowsize = default(exact_windowsize, True)

        self.attn_fn = LocalAttention(
            dim = dim_head,
            window_size = window_size,
            causal = causal,
            autopad = True,
            scale = (qk_scale if qk_rmsnorm else None),
            exact_windowsize = self.exact_windowsize,
            use_xpos = use_xpos,
            xpos_scale_base = xpos_scale_base,
            **kwargs
        )

        self.to_v_gate = None

        if gate_values_per_head:
            self.to_v_gate = nn.Sequential(
                nn.Linear(dim, heads)
            )

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None,
        attn_bias = None,
        cache = None,
        return_cache = False
    ):
        seq_len = x.shape[-2]

        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) 

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        if exists(cache):
            assert seq_len == 1

            assert self.causal and not exists(mask), 'only allow caching for specific configuration'

            ck, cv = cache

            q = q * (q.shape[-1] ** -0.5)

            k = cat((ck, k), dim = -2)
            v = cat((cv, v), dim = -2)

            effective_window_size = self.attn_fn.look_backward * self.window_size

            if self.exact_windowsize:
                kv_start_index = -(effective_window_size + 1)
            else:
                seq_len = k.shape[-2]
                kv_start_index = -(effective_window_size + (seq_len % self.window_size))

            k, v = tuple(t[..., kv_start_index:, :] for t in (k, v))

            if exists(self.attn_fn.rel_pos):
                rel_pos = self.attn_fn.rel_pos
                pos_emb, xpos_scale = rel_pos(k)
                q, k = apply_rotary_pos_emb(q, k, pos_emb, scale = xpos_scale)

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            if exists(attn_bias):
                k_len = k.shape[-2]
                attn_bias = attn_bias[..., -1:, -k_len:]
                assert attn_bias.shape[-1] == sim.shape[-1]
                sim = sim + attn_bias

            attn = sim.softmax(dim = -1)
            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        else:
            out = self.attn_fn(q, k, v, mask = mask, attn_bias = attn_bias)

        if return_cache:
            kv = torch.stack((k, v))

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            gates = rearrange(gates, 'b n h -> b h n 1')
            out = out * gates.sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_cache:
            return out

        return out, kv

class MultiscaleLocalMHA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        window_sizes=[8, 16, 32],
        dim_head=64,
        heads=8,
        dropout=0.,
        causal=False,
        prenorm=True,
        qk_rmsnorm=False,
        qk_scale=8,
        **kwargs
    ):
        super().__init__()
        self.scales = nn.ModuleList([
            LocalMHA(
                dim=dim,
                window_size=window_size,
                dim_head=dim_head,
                heads=heads,
                dropout=dropout,
                causal=causal,
                prenorm=prenorm,
                qk_rmsnorm=qk_rmsnorm,
                qk_scale=qk_scale,
                **kwargs
            ) for window_size in window_sizes
        ])
        self.scale_weights = (
            nn.Parameter(torch.ones(len(window_sizes)) / len(window_sizes)) 
            if len(window_sizes) > 1 else None
        )

    def forward(self, x, mask=None):
        outs = []
        for attn in self.scales:
            x = attn(x, mask=mask)
            if exists(mask):
                x = x.masked_fill(~mask.unsqueeze(-1), 0.)
            outs.append(x)
        
        if exists(self.scale_weights):
            outs = torch.stack(outs, dim=0)
            weights = torch.softmax(self.scale_weights, dim=0)
            out = einsum('s b n d, s -> b n d', outs, weights)
        else:
            out = outs[0]
        
        return out

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        out = self.net(x)
        if exists(mask):
            out = out.masked_fill(~mask.unsqueeze(-1), 0.)

        return out

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net1 = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
        )

        self.conv_dw = DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding)

        self.net2 = nn.Sequential(
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        x = self.net1(x)
        if exists(mask):
            x = x.masked_fill(~rearrange(mask, 'b n -> b 1 n'), 0.)

        x = self.conv_dw(x, mask=mask)

        out = self.net2(x)
        if exists(mask):
            out = out.masked_fill(~mask.unsqueeze(-1), 0.)

        return out

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        attn_causal = False,
        attn_window_sizes = [8, 16, 64],
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        prenorm = True,
        qk_scale = 8,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = MultiscaleLocalMHA(dim=dim, window_sizes=attn_window_sizes, dim_head=dim_head, heads=heads, dropout=attn_dropout, causal=attn_causal, prenorm=prenorm, qk_scale=qk_scale)
        self.conv = ConformerConvModule(dim = dim, causal = conv_causal, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x, mask=mask) + x

        x = self.attn(x, mask=mask) + x
        if exists(mask):
            x = x.masked_fill(~mask.unsqueeze(-1), 0.)

        x = self.conv(x, mask=mask) + x
        x = self.ff2(x, mask=mask) + x
        x = self.post_norm(x)
        if exists(mask):
            x = x.masked_fill(~mask.unsqueeze(-1), 0.)

        return x

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        vocab_size,
        max_seq_len=1024,
        *,
        depth,
        num_classes = 30, # num composers
        dim_head = 64,
        heads = 8,
        attn_window_sizes = [8, 16, 64],
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False,
        prenorm = True,
        qk_scale = 8,
        padding_idx = 0,
        pooling_strategy = "sequence_attention",
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.pos_emb = ScaledSinusoidalEmbedding(dim)
        self.pooling_strategy = pooling_strategy
        
        if self.pooling_strategy == "first":
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(ConformerBlock(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                attn_window_sizes = attn_window_sizes,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                conv_dropout = conv_dropout,
            ))

        self.sequence_attention = nn.Linear(dim, 1, bias=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2, bias=False),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 2, num_classes, bias=True)
        )

    def forward(self, x, mask=None, pad=True, return_encoding=False):
        batch_size = x.shape[0]

        x = self.token_emb(x)
        
        # add CLS token if using first-token pooling
        if self.pooling_strategy == "first":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            if exists(mask):
                cls_mask = torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        x = self.pos_emb(x, mask=mask)

        # encoder layers

        for block in self.layers:
            x = block(x, mask=mask)

        # pooling and classification
        
        if self.pooling_strategy == "sequence_attention":
            attn_logits = self.sequence_attention(x)
            if exists(mask):
                attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            attn_weights = torch.softmax(attn_logits, dim=1)
            x = torch.sum(x * attn_weights, dim=1)

        elif self.pooling_strategy == "mean":
            if exists(mask):
                x = x.masked_fill(~mask.unsqueeze(-1), 0.)
                x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                x = x.mean(dim=1)

        elif self.pooling_strategy == "max":
            if exists(mask):
                x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            x = x.max(dim=1)[0]

        elif self.pooling_strategy == "first":
            x = x[:, 0]

        else:
            raise ValueError(f"unknown pooling strategy: {self.pooling_strategy}")
            
        logits = self.classifier(x)

        if return_encoding:
            return logits, x
        else:
            return logits


if __name__ == "__main__":
    vocab_size = 512
    batch_size = 2
    num_classes = 10

    model = Transformer(
        dim = 128,
        vocab_size = vocab_size,
        max_seq_len = 1024,
        depth = 2,
        dim_head = 4,
        heads = 4,
        ff_mult = 4,
        attn_window_sizes = [16, 64],
        conv_expansion_factor = 2,
        conv_kernel_size = 17,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        num_classes=num_classes,
    )

    pad_len = 10
    x = torch.randint(0, vocab_size, (batch_size, 1024 - pad_len))
    labels = torch.randint(0, num_classes, (batch_size, 1))

    mask = torch.ones(batch_size, 1024, dtype=torch.bool)

    mask[:,-pad_len:] = False
    assert mask[:,-pad_len:].sum() == 0

    pad = torch.zeros(x.shape[0], pad_len, dtype=x.dtype, device=x.device)
    x = cat((x, pad), dim=-1)
    assert x[:, -pad_len:].sum() == 0

    # print('x:', x.shape)
    # print('labels:', labels.shape)
    # print('mask:', mask.shape)

    print('num params:', sum(p.numel() for p in model.parameters()))
    # print('output shape:', model(x, mask=mask, return_encoding=True))
    print('output shape:', model(x, mask=mask, return_encoding=False).shape)