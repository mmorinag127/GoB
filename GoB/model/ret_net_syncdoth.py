import haiku as hk
import jax
from jax import numpy as jnp
import math

from .token_mixing import make_object_tokenizer
from .weight_init import make_w_init

def make_ret_net(model, depth, dim, expansion, dim_qk, dim_v, n_heads, msr_bias, ffn_bias, w_init, msr_winit, ffn_winit, gamma_type='default', **kwargs):
    dim_inner = dim*expansion
    
    
    
    def ret_net(feature, *args, training=True, one_hot=None, mask=None, glob=None, **kwargs):
        tokens, mask = make_object_tokenizer(dim, w_init, name='tokenizer')(feature, one_hot = one_hot, glob = glob, mask = mask)
        
        
        hidden_state = tokens
        for idx in range(depth):
            msr_out, curr_kv, rtn_weight = make_msr_layer(dim, dim_qk, dim_v, msr_bias, n_heads, msr_winit, gamma_type, name=f'msr_{idx}')(x=hidden_state, rtn_mask=rtn_mask, past_kv=past_kv, chunk_size=chunk_size, training=training, mode=mode, offset=offset)
            hidden_state = hidden_state + msr_out
            ffn_out = make_ffn_layer(dim, dim_inner, ffn_bias, ffn_winit, name=f'ffn{idx}')(hidden_state, training=training)
            hidden_state = hidden_state + ffn_out
        return out
    return ret_net



def make_xpos(head_dim, scale_base=512):
    
    _scale = (jnp.arange(0, head_dim, 2) + 0.4*head_dim)/(1.4 + head_dim)
    def fixed_pos_embedding(x):
        _, N, D = x.shape
        inv_freq = 1.0/(10000**(jnp.arange(0, D)/D))
        sinusoid_inp = jnp.einsum('i,j -> ij', jnp.arange(0, N), inv_freq )
        return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)
    
    def duplicate_interleave(m):
        D = m.shape[0]
        m = jnp.reshape(m, [-1, 1])
        m = jnp.tile(m, [1,2])
        m = jnp.reshape(m, [D, -1])
        return m
    
    def rotate_every_two(x):
        n,m,_ = x.shape
        x1 = x[:, :,  ::2]
        x2 = x[:, :, 1::2]
        x = jnp.stack([x1, x2], axis=-1)
        return jnp.reshape(x, [n,m,-1])
    
    def apply_rotary_pos_emb(x, sin, cos, scale=1):
        sin, cos = map(lambda t: duplicate_interleave(t*scale), (sin, cos))
        return (x*cos) + (rotate_every_two(x)*sin)
    
    
    def apply(x, offset=0, downscale=False):
        B, N, D = x.shape
        min_pos = 0
        max_pos = min_pos+offset+N
        scale = _scale*jnp.arange(min_pos, max_pos, 1)[:,None]/scale_base
        sin, cos = fixed_pos_embedding(scale)
        if scale.shape[0] > N:
            scale = scale[-N:]
            sin = sin[-N:]
            cos = cos[-N:]
        if downscale:
            scale = 1./scale
        return apply_rotary_pos_emb(x, sin, cos, scale)
    
    def xpos(q, k, offset=0):
        q = apply(q, offset=offset, downscale=False)
        k = apply(k, offset=offset, downscale=True)
        return q,k
    return xpos

def make_msr_layer(dim, dim_qk, dim_v, with_bias, n_heads, w_init, gamma_type='default', **kwargs):
    w_init = make_w_init(w_init)
    
    if gamma_type=='default':
        gamma = 1 - 2**(-5 - jnp.arange(0, n_heads, ))
    else:
        gamma = 1 - jnp.exp(jnp.linspace(math.log(1./32), math.log(1./512), n_heads))
    
    def make_parallel_decay_mask(length, rtn_mask=None):
        a = jnp.arange(length)
        a = jnp.tile(a[None, :, None], [n_heads, length, 1])
        exp = a - jnp.transpose(a, [0,2,1])
        decay_mask = jnp.reshape(gamma, [-1,1,1])**exp
        decay_mask = jnp.tril(decay_mask) 
        decay_mask = jnp.expand_dims(decay_mask, axis=0)
        
        if rtn_mask is not None:
            rtn_mask = jnp.reshape(rtn_mask, [-1,1,1,length])
            decay_mask = decay_mask * rtn_mask
        return decay_mask
    
    def make_recurrent_decay():
        decay = jnp.reshape(gamma, [1, n_heads, 1, 1])
        return decay
    
    def make_chunkwise_decay(chunk_size, rtn_mask=None):
        # within chunk decay
        decay_mask = make_parallel_decay_mask(chunk_size, rtn_mask=rtn_mask)
        # decay of the chunk
        chunk_decay = jnp.reshape(gamma, [1, n_heads, 1, 1])**chunk_size
        # cross-chunk decay
        exp = jnp.expand_dims(jnp.arange(chunk_size), axis=0) + 1
        inner_decay = jnp.reshape(jnp.expand_dims(gamma, axis=-1)**exp, [1, n_heads, chunk_size, 1])
        return decay_mask, chunk_decay, inner_decay
    
    def parallel_retention(q, k, v, decay_mask):
        # q,k  : (B, H, N, D_QK)
        # v    : (B, H, N, D_V)
        # mask : (B, H, N, N)
        # retention
        scale = math.sqrt(k.shape[-1])
        retention = (q @ jnp.transpose(k, [0,1,3,2]))/scale
        retention = retention * decay_mask
        out = retention @ v
        
        # kv cache
        curr_kv = jnp.expand_dims(k, axis=-1) * jnp.expand_dims(v, axis=-2)
        # (B,H, D_QK, D_V)
        intra_decay = decay_mask[:, :, -1, :, None, None]
        curr_kv = jnp.sum(curr_kv * intra_decay, axis=2)
        return out, curr_kv, retention
    
    def recurrent_retention(q, k, v, past_kv=None, decay=None, rtn_mask=None):
        # q, k, v : (B, H, 1, D_QKV)
        # past_kv : (B, H, D_QK, D_V)
        # decay   : (H, 1, 1)
        # rtn_mask: (B, 1)
        past_kv = past_kv if past_kv is not None else 0
        decay = decay if decay is not None else 0
        rtn_mask = jnp.reshape(rtn_mask, [-1,1,1,1]) if rtn_mask is not None else 1
        # (B, H, D_QK, D_V)
        scale = math.sqrt(k.shape[-1])
        curr_kv = decay * past_kv + rtn_mask * (jnp.transpose(k, [0,1,3,2]) @ v)
        out = q @ curr_kv / scale
        return out, curr_kv
    
    def chunkwise_retention(q, k, v, decay_mask, past_kv = None, chunk_decay=None, inner_decay=None):
        # q, k, v : (B, H, C, D_QKV), C=chunk size
        # past_kv : (B, H, D_QK, D_V)
        # chunk decay : (1, H, 1, 1)
        # inner decay : (1, H, C, 1)
        scale = math.sqrt(k.shape[-1])
        retention = q @ jnp.transpose(k, [0,1,3,2])/scale
        retention = retention * decay_mask
        inner_retention = retention * v 
        cross_retention = 0
        past_chunk = 0
        if past_kv is not None:
            cross_retention = (q @ past_kv) * inner_decay / scale
            past_kv = chunk_decay * past_kv
        
        retention = inner_retention + cross_retention
        curr_kv = jnp.expand_dims(k, axis=-1) * jnp.expand_dims(v, axis=-2)
        # NOTE: intra decay is omitted in the paper, but this details is important
        intra_decay = decay_mask[:, :, -1, :, None, None]
        curr_kv = jnp.sun(curr_kv * intra_decay, axis=2)
        curr_kv = past_chunk + curr_kv
        return retention, curr_kv
    
    def msr_layer(x, *args, rtn_mask=None, past_kv=None, chunk_size=None, training=True, mode='parallel', offset=0, **kwargs):
        B, N, D = x.shape
        
        q, k, v = jnp.split(hk.Linear(dim_qk+dim_qk+dim_v, with_bias=with_bias, name='qkv')(x), [dim_qk, dim_qk+dim_qk], axis=-1)
        q, k = make_xpos(dim_qk)(q, k, offset=offset)
        q = jnp.transpose(jnp.reshape(q, [B, N, n_heads, -1]), [0,2,1,3])
        k = jnp.transpose(jnp.reshape(k, [B, N, n_heads, -1]), [0,2,1,3])
        v = jnp.transpose(jnp.reshape(v, [B, N, n_heads, -1]), [0,2,1,3])
        
        if mode == 'parallel':
            decay_mask = make_parallel_decay_mask(N, rtn_mask=rtn_mask)
            rtn_out, curr_kv, rtn_weight = parallel_retention(q, k, v, decay_mask)
        elif mode == 'recurrent':
            decay = make_recurrent_decay()
            rtn_out, curr_kv = recurrent_retention(q, k, v, past_kv=past_kv, decay=decay, rtn_mask=rtn_mask)
        elif mode == 'chunkwise':
            chunk_idx = jnp.linspace(0, D, chunk_size+1, dtype=jnp.int64)[1:-1]
            q_chunks = jnp.split(q, chunk_idx, axis=-1)
            k_chunks = jnp.split(k, chunk_idx, axis=-1)
            v_chunks = jnp.split(v, chunk_idx, axis=-1)
            if rtn_mask is not None:
                rtn_mask_chunks = jnp.split(rtn_mask, chunk_idx, axis=-1)
            else:
                rtn_mask_chunks = [None]*chunk_size
            ret_chunks = []
            for i, (_q, _k, _v, _rtn_mask) in enumerate(zip(q_chunks,k_chunks,v_chunks,rtn_mask_chunks)):
                csz = q.shape[-1]
                decay_mask, chunk_decay, innder_decay = make_chunkwise_decay(csz, _rtn_mask)
                out_chunk, past_kv = chunkwise_retention(_q, _k, _v, decay_mask, past_kv=past_kv, chunk_decay=chunk_decay, innder_decay=innder_decay)
                ret_chunks.append(out_chunk)
            rtn_out = jnp.concatenate(ret_chunks, axis=2)
            curr_kv = past_kv
        else:
            raise ValueError(f'mode: {mode} is not supoorted')
        rtn_out = jnp.transpose(jnp.transpose(rtn_out, [0,2,1]), B, N, dim_v)
        normed = hk.GroupNorm(groups=n_heads, create_scale=True, create_offset=True, data_format='NC')(jnp.reshape(rtn_out, [B*N, dim_v]))
        normed = jnp.reshape(normed, [B,N,dim_v])
        gated = jax.nn.silu(hk.Linear(dim_v, with_bias=False, name='gate')(x))*normed
        out = hk.Linear(dim, with_bias=with_bias, name='proj')
        if mode == 'parallel':
            return out, curr_kv, rtn_weight
        return out, curr_kv, None
    return msr_layer

def make_ffn_layer(dim, dim_inner, with_bias, w_init=None, name='ffn', **kwargs):
    _w_init = make_w_init(w_init)
    def ffn_layer(x, *args, training=True, **kwargs):
        out = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False, name='ln', w_init=_w_init)(x)
        out = hk.Linear(dim_inner, with_bias=with_bias, name='linear0')(out)
        out = jax.nn.gelu(out)
        out = hk.Linear(dim, with_bias=with_bias, name='linear1')(out)
        return out
    return ffn_layer
