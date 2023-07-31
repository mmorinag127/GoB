import haiku as hk
import jax
from jax import numpy as jnp
import math

from .token_mixing import make_object_tokenizer, make_drop_path, make_layer_scale, make_drop_path_rate
from .weight_init import make_w_init

def make_ret_net(model, depth, dim, expansion, n_heads, dropout, drop_path, layer_scale, msr_bias, ffn_bias, w_init, msr_winit, ffn_winit, chunk_size, **kwargs):
    dim_inner = dim*expansion
    
    
    drop_path_rate = make_drop_path_rate(depth, drop_path)
    
    def ret_net(feature, *args, mode='parallel', training=True, one_hot=None, mask=None, glob=None, **kwargs):
        tokens, mask = make_object_tokenizer(dim, w_init, name='tokenizer')(feature, one_hot = one_hot, glob = glob, mask = mask)
        rel_pos = make_ret_rel_pos(dim, n_heads, chunk_size)(tokens.shape[1], tokens.dtype, mode=mode)
        
        hidden_state = tokens
        incremental_state = {}
        for idx in range(depth):
            
            msr_out, incremental_state = make_msr_layer(dim, expansion, msr_bias, n_heads, dropout, msr_winit)(hidden_state, rel_pos, incremental_state, training=training, mode=mode)
            msr_out = hk.dropout(hk.next_rng_key() if hk.running_init() else None, dropout if training else 0.0, msr_out)
            
            hidden_state = make_residual(drop_path=drop_path_rate[idx], layer_scale=layer_scale, name=f'msr_res')(hidden_state, msr_out, training=training)
            
            ffn_out = make_ffn_layer(dim, dim_inner, ffn_bias, ffn_winit, name=f'ffn{idx}')(hidden_state, training=training)
            hidden_state = make_residual(drop_path=drop_path_rate[idx], layer_scale=layer_scale, name=f'ffn_res')(hidden_state, ffn_out, training=training)
        
        output = hk.LayerNorm(axis = -1, create_scale = True, create_offset = True, name='ln')(hidden_state)
        return output
    return ret_net

def make_residual(drop_path=None, layer_scale=None, name='residual'):
    def residual_connection(x, residual, training=True):
        if drop_path is not None:
            residual = make_drop_path(drop_path, name='droppath')(residual, training=training)
        if layer_scale is not None:
            residual = make_layer_scale(x.shape[-1], layer_scale, name='layerscale')(x)
        return x + residual
    return hk.to_module(residual_connection)(name=name)

def make_ret_rel_pos(dim, n_heads, chunk_size, name=f'rel_pos', **kwargs):
    angle = 1.0 / (10000**jnp.linspace(0,1, dim//n_heads//2))
    angle = jnp.reshape(jnp.tile(jnp.expand_dims(angle, axis=-1), [1,2]), [-1])
    decay = jnp.log(1 - 2**(-5 - jnp.arange(n_heads)) )
    def ret_rel_pos(slen, dtype, mode='parallel'):
        if mode == 'recurrent': 
            sin = jnp.sin(angle *(slen-1))
            cos = jnp.cos(angle *(slen-1))
            return (sin, cos), jnp.exp(decay)
        elif mode == 'chunkwise':
            index = jnp.arange(slen)
            sin = jnp.sin(index[:, None] * angle[None, :])
            cos = jnp.cos(index[:, None] * angle[None, :])
            
            big_pos = jnp.finfo(dtype).max
            block_index = jnp.arange(chunk_size)
            mask = jnp.tril(jnp.ones([chunk_size, chunk_size], dtype=jnp.bool_))
            mask = jnp.where(~mask, block_index[:,None] - block_index[None,:], big_pos)
            mask = jnp.exp(mask * decay[:, None, None])
            mask = jnp.nan_to_num(mask)
            scale = jnp.sqrt(jnp.sum(mask, axis=-1, keepdims=True))
            mask = mask / scale
            
            cross_decay = jnp.exp(decay * chunk_size)
            inner_decay = jnp.exp(decay[:, None] * (block_index + 1))
            cross_decay = cross_decay[:, None, None]
            inner_decay = inner_decay[:, :, None] / (scale / scale[:, -1, None])
            return (sin, cos), (mask, cross_decay, inner_decay)
        elif mode == 'parallel':
            index = jnp.arange(slen)
            big_pos = jnp.finfo(dtype).max
            sin = jnp.sin(index[:, None] * angle[None, :])
            cos = jnp.cos(index[:, None] * angle[None, :])
            mask = jnp.tril(jnp.ones([slen, slen], dtype=jnp.bool_))
            mask = jnp.where(~mask, index[:, None] - index[None, :], big_pos)
            mask = jnp.exp(mask * decay[:, None, None])
            mask = jnp.nan_to_num(mask)
            mask = mask / jnp.sqrt(jnp.sum(mask, axis=-1, keepdims=True))
            return (sin, cos), mask
        raise ValueError(f'mode:{mode} is not supported in ret_rel_pos')
    return ret_rel_pos

def make_msr_layer(dim, v_factor, with_bias, n_heads, w_init, name='msr_layer', **kwargs):
    w_init = make_w_init(w_init)
    dim_head = dim*v_factor // n_heads
    dim_key = dim // n_heads
    dscale = dim_key**-0.5
    
    def rotate_every_two(x):
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        return jnp.reshape(jnp.stack([-x2, x1], axis=-1), x.shape)
    
    def theta_shift(x, sin, cos):
        return (x*cos) + (rotate_every_two(x)*sin)
    
    def parallel_retention(qr, kr, v, mask):
        B,L,_ = v.shape
        vr = jnp.swapaxes(jnp.reshape(v, [B,L,n_heads,dim_head]), 1,2)
        
        qk = qr @ jnp.swapaxes(kr, 2,3)
        qk = qk * mask
        qk = qk / jnp.clip(jnp.sum(jnp.abs(qk),axis=-1, keepdims=True), a_min=1)
        output = jnp.matmul(qk, vr)
        output = jnp.swapaxes(output, 1,2)
        return output
    
    def recurrent_retention(qr, kr, v, decay, incremental_state):
        B = v.shape[0]
        v = jnp.reshape(v, [B, n_heads, dim_head, 1])
        kv = kr * v
        
        if 'prev_kv' in incremental_state:
            prev_kv = incremental_state['prev_kv']
            prev_scale = incremental_state['scale']
            scale = prev_scale * decay + 1
            kv = prev_kv * jnp.reshape((1 - 1/scale), [n_heads,1,1]) + kv / jnp.reshape(scale, [n_heads,1,1])
        else:
            scale = jnp.ones_like(decay)
        incremental_state['prev_kv'] = kv
        incremental_state['scale'] = scale
        output = jnp.sum(qr * kv, axis=3)
        return output, incremental_state
    
    def chunkwise_retention(qr, kr, v, inner_mask):
        mask, cross_decay, inner_decay = inner_mask
        B,L,_ = v.shape
        chunk_len = mask.shape[1]
        n_chunks = L // chunk_len
        assert L % chunk_len == 0
        
        qr = jnp.swapaxes(jnp.reshape(qr, [B,n_heads,n_chunks,chunk_len,dim_key]),  1,2)
        kr = jnp.swapaxes(jnp.reshape(kr, [B,n_heads,n_chunks,chunk_len,dim_key]),  1,2)
        v  = jnp.swapaxes(jnp.reshape(v,  [B,n_chunks,chunk_len,n_heads,dim_head]), 2,3)
        kr_t = jnp.swapaxes(kr, -1, -2)
        
        qk = qr @ kr_t # B,H,CL,CL CL:chunk_len
        qk = qk * mask
        inner_scale = qk / jnp.clip(jnp.sum(jnp.abs(qk),axis=-1, keepdims=True), a_min=1)
        qk = qk / inner_scale
        inner_output = jnp.matmul(qk, v) # B,H,VH,CL,HD VH:n_value_heads, HD:dim_head
        
        # reduce kv in one chunk
        kv = kr_t @ (v * mask[:, -1, :, None])
        kv = jnp.reshape(kv, [B,n_chunks,n_heads,dim_key,dim_head])
        
        kv_recurrent = []
        cross_scale = []
        kv_state = jnp.zeros([B,n_heads,dim_key,dim_head])
        kv_scale = jnp.ones([B,n_heads,1,dim_head])
        
        # accumulate kv by loop
        for idx in range(n_chunks):
            kv_recurrent.append(kv_state/kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, idx]
            kv_scale = jnp.clip(jnp.sum(jnp.abs(kv_state), axis=-2, keepdims=True), a_min=1)
        
        kv_recurrent = jnp.stack(kv_recurrent, axis=1)
        cross_scale  = jnp.stack(cross_scale,  axis=1)
        cross_output = (qr * inner_decay) @ kv_recurrent
        
        print(f'inner_output : {inner_output.shape}')
        print(f'cross_scale  : {cross_scale.shape}')
        print(f'inner_scale  : {inner_scale.shape}')
        print(f'cross_output : {cross_output.shape}')
        print(f'qk           : {qk.shape}')
        
        
        output = inner_output / cross_scale + cross_output / inner_scale
        output = jnp.swapaxes(output, 2,3)
        return output
    
    def msr_layer(x, rel_pos, incremental_state=None, training=True, mode='parallel', **kwargs):
        B,L,_ = x.shape
        (sin, cos), inner_mask = rel_pos
        
        q = hk.Linear(dim,          with_bias=with_bias, w_init=w_init, name=f'linear_q')(x)
        k = hk.Linear(dim,          with_bias=with_bias, w_init=w_init, name=f'linear_k')(x)
        v = hk.Linear(dim*v_factor, with_bias=with_bias, w_init=w_init, name=f'linear_v')(x)
        g = hk.Linear(dim*v_factor, with_bias=with_bias, w_init=w_init, name=f'linear_g')(x)
        k *= dscale
        q = jnp.swapaxes(jnp.reshape(q, [B,L,n_heads,dim_key]), 1,2) # B,H,L,D
        k = jnp.swapaxes(jnp.reshape(k, [B,L,n_heads,dim_key]), 1,2) # B,H,L,D
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)
        
        if mode == 'recurrent':
            output, incremental_state = recurrent_retention(qr, kr, v, inner_mask, incremental_state)
        elif mode == 'chunkwise':
            output = chunkwise_retention(qr, kr, v, inner_mask)
        elif mode == 'parallel':
            output = parallel_retention(qr, kr, v, inner_mask)
        else:
            raise ValueError(f'mode:{mode} is not supported in msr layer')
        
        output = hk.GroupNorm(groups=n_heads, create_scale=True, create_offset=True, data_format='N...C')(output)
        output = jnp.reshape(output, [B,L,dim_head*n_heads])
        output = jax.nn.silu(g) * output
        output = hk.Linear(dim, with_bias= with_bias, name=f'linear_out')(output)
        return output, incremental_state
    return hk.to_module(msr_layer)(name=name)

def make_ffn_layer(dim, dim_inner, with_bias, w_init=None, name='ffn', **kwargs):
    _w_init = make_w_init(w_init)
    def ffn_layer(x, *args, training=True, **kwargs):
        out = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False, name='ln')(x)
        out = hk.Linear(dim_inner, with_bias=with_bias, name='linear0', w_init=_w_init)(out)
        out = jax.nn.gelu(out)
        out = hk.Linear(dim, with_bias=with_bias, name='linear1', w_init=_w_init)(out)
        return out
    return hk.to_module(ffn_layer)(name=name)





