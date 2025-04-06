from functools import wraps

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from torch_cluster import fps

# from timm.models.layers import DropPath

import math

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_kvpacked_func

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)


    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None, window_size=-1):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        kv = self.to_kv(context)

        q = rearrange(q, 'b n (h d) -> b n h d', h = h) # flash_attn
        kv = rearrange(kv, 'b n (p h d) -> b n p h d', h = h, p=2) # flash_attn

        # print(q.shape, kv.shape)
        out = flash_attn_kvpacked_func(q.bfloat16(), kv.bfloat16(), window_size=(window_size, window_size))
        out = out.to(x.dtype)

        return self.to_out(rearrange(out, 'b n h d -> b n (h d)')) # flash_attn
        
class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed

    
class CrossAttnNetwork(nn.Module):
    def __init__(
        self, depth=24, dim=512, heads=8, dim_head=64, dims=8,
    ):
        super().__init__()

        self.dims = dims

        weight_tie_layers = False

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = dim // dim_head, dim_head = dim_head), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])


        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))
        

        self.post_bottleneck_proj = nn.Linear(dims, dim)

        self.pre_bottleneck_proj = nn.Linear(dim, dims)
        self.pre_bottleneck_norm = nn.LayerNorm(dims, elementwise_affine=False, eps=1e-6)

        self.gamma = nn.Parameter(torch.ones(dims))
        self.beta = nn.Parameter(torch.zeros(dims))

    def encode(self, inputs, x):
    
        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context = inputs, mask = None) + x
        x = cross_ff(x) + x

        ###
        sampled = self.pre_bottleneck_norm(self.pre_bottleneck_proj(x))

        return x, sampled


    def learn(self, x, res=None):        

        if res is not None:
            x = x + res

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x

    def post_bottleneck(self, sampled):
        sampled = sampled * self.gamma + self.beta

        sampled = self.post_bottleneck_proj(sampled)
        return sampled
    
    def forward(self, inputs, latents, res=None):
        x, sampled = self.encode(inputs, latents)
        
        sampled = self.post_bottleneck(sampled)

        decoded = self.learn(sampled, res=res)
        return x, decoded

def subsample(pc, N, M):
    # pc: B x N x 3
    B, N0, D = pc.shape
    assert N == N0
    
    ###### fps
    flattened = pc.view(B*N, D)

    batch = torch.arange(B).to(pc.device)
    batch = torch.repeat_interleave(batch, N)

    pos = flattened

    ratio = 1.0 * M / N

    idx = fps(pos, batch, ratio=ratio)

    sampled_pc = pos[idx]
    sampled_pc = sampled_pc.view(B, -1, 3)
    ######

    return sampled_pc

class AutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim=1024,
        queries_dim=1024,
    ):
        super().__init__()
        
        self.networks = nn.ModuleList([
            CrossAttnNetwork(8, dim, dim // 64, 64, dims=16),
            CrossAttnNetwork(8, dim, dim // 64, 64, dims=32),
            CrossAttnNetwork(8, dim, dim // 64, 64, dims=64),
        ])
        
        self.upsample_blocks = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, dim, heads = dim // 64, dim_head = 64), context_dim = dim),
                PreNorm(dim, FeedForward(dim))
            ]) if i != 2 else None for i in range(3) 
        ])
        
        self.num_points = [2048, 512, 128]

        self.decoder_cross_attn = nn.ModuleList([
            PreNorm(queries_dim, Attention(queries_dim, dim, heads = dim // 64, dim_head = 64), context_dim = dim),
            PreNorm(queries_dim, Attention(queries_dim, dim, heads = dim // 64, dim_head = 64), context_dim = dim),
            PreNorm(queries_dim, Attention(queries_dim, dim, heads = dim // 64, dim_head = 64), context_dim = dim),
        ])
        

        self.point_embed = PointEmbed(dim=dim)

        self.to_outputs = nn.Linear(dim * 3, 1)

    def encode(self, pc):

        B, N, _ = pc.shape
        assert N == 8192

        levels = []

        for M in self.num_points:

            sampled_pc = subsample(pc, N, M)
                
            pc_embeddings = self.point_embed(pc)
            sampled_pc_embeddings = self.point_embed(sampled_pc)
            levels.append((pc_embeddings, sampled_pc_embeddings))
            
            pc = sampled_pc
            N = pc.shape[1]

        return levels

    def encode_level(self, pc, sampled_pc, cross_attn_network):
        x, sampled = cross_attn_network.encode(pc, sampled_pc)
        return x, sampled
    
    def decode_level(self, pc, x, cross_attn_network, upsample_block, res=None):
        x = cross_attn_network.learn(x, res=res)

        if upsample_block is not None:
            cross_attn, cross_ff = upsample_block
            upsampled = cross_attn(pc, context = x, mask = None) + pc
            upsampled = cross_ff(upsampled) + upsampled
            return x, upsampled

        return x, None

    def get_level_latents(self, inputs):
        _, N, _ = inputs.shape
        assert N == 8192
        
        pc = inputs

        levels = self.encode(pc)
        
        samples = []

        x = None
        for (pc_embeddings, sampled_pc_embeddings), cross_attn_network in zip(levels, self.networks):
            if x is not None:
                pc_embeddings = x
            x, sampled = self.encode_level(pc_embeddings, sampled_pc_embeddings, cross_attn_network)# - 0.5
            samples.append(sampled.clone())
        samples.reverse()
        return samples

    def decode_level_latents(self, xs, queries):
        xs = [cross_attn_network.post_bottleneck(x) for x, cross_attn_network in zip(xs, self.networks[::-1])]

        prev_x = None

        latents = []

        for x, high_res, cross_attn_network, upsample_block in zip(xs, xs[1:] + [None], self.networks[::-1], self.upsample_blocks):
            x, upsampled = self.decode_level(high_res, x, cross_attn_network, upsample_block, res=prev_x)

            prev_x = upsampled
            latents.append(x.clone())


        if queries.shape[1] > 100000:
            N = 100000
            os = []
            for block_idx in range(math.ceil(queries.shape[1] / N)):
                queries_embeddings = self.point_embed(queries[:, block_idx*N:(block_idx+1)*N, :])

                l = torch.cat([self.decoder_cross_attn[0](queries_embeddings[:, :, :], context = latents[0]), 
                    self.decoder_cross_attn[1](queries_embeddings[:, :, :], context = latents[1]),
                    self.decoder_cross_attn[2](queries_embeddings[:, :, :], context = latents[2]),
                ], dim=2)
                o = self.to_outputs(l).squeeze(-1)
                os.append(o)
            o = torch.cat(os, dim=1)
            return o
        
        
        queries_embeddings = self.point_embed(queries)

        latents = torch.cat([self.decoder_cross_attn[0](queries_embeddings, context = latents[0]), 
            self.decoder_cross_attn[1](queries_embeddings, context = latents[1]),
            self.decoder_cross_attn[2](queries_embeddings, context = latents[2]),
        ], dim=2)

        o = self.to_outputs(latents).squeeze(-1)
        
        return o
    

    def forward(self, inputs, queries):
        B, N, _ = inputs.shape
        assert N == 8192
        
        pc = inputs

        xs = self.get_level_latents(pc)
        o = self.decode_level_latents(xs, queries)
        return {'logits': o}