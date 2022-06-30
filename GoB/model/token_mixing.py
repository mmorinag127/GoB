
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import einops

from .layers import drop_path

def make_patch_layer(patch_size, dim):
    def patch_layer(image):
        patch =  einops.rearrange(image, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        patch = hk.Linear(output_size = dim)(patch)
        patch = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(patch)
        return patch
    return patch_layer

def make_meta_layer(dim, mixing_layer, drop_path_rate, lscale, film):
    def meta_layer(x, training, gamma = None, beta = None):
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(x)
        out = mixing_layer(out, training)
        if film is not None:
            gamma = hk.Linear(dim, with_bias = False)(gamma)
            gamma = jnp.reshape(gamma, (gamma.shape[0], -1, gamma.shape[-1]))
            beta  = hk.Linear(dim, with_bias = False)(beta)
            beta = jnp.reshape(beta, (beta.shape[0], -1, beta.shape[-1]))
            out = gamma * out
            out = out + beta
        
        if lscale is not None:
            param = hk.get_parameter('layer_scale', [dim], init = lambda shape, dtype: jnp.ones(shape, dtype)*lscale)
            out = param*out
        out = drop_path(hk.next_rng_key(), drop_path_rate if training else 0.0, out)
        return x + out
    return meta_layer

def make_mixer_layer(dim, dim_inner, dropout_rate):
    def mixer_layer(x, training):
        out = jnp.swapaxes(x, 1, 2)
        out = hk.Linear(dim_inner)(out)
        out = jax.nn.gelu(out)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0, out)
        out = hk.Linear(x.shape[1])(out)
        out = jnp.swapaxes(out, 1, 2)
        out = hk.dropout(hk.next_rng_key(), dropout_rate, out)
        return out
    return mixer_layer

def make_attention_layer(dim, dim_inner, dropout_rate, dim_head, n_heads, qkv_bias):
    def attention_layer(x, training):
        B, N, _ = x.shape
        q = jnp.reshape(hk.Linear(output_size = dim_inner, with_bias = qkv_bias)(x), [B, N, n_heads, dim_head])
        k = jnp.reshape(hk.Linear(output_size = dim_inner, with_bias = qkv_bias)(x), [B, N, n_heads, dim_head])
        v = jnp.reshape(hk.Linear(output_size = dim_inner, with_bias = qkv_bias)(x), [B, N, n_heads, dim_head])
        
        atn = jnp.einsum('b s n h, b t n h-> b n s t', q, k)
        atn /= np.sqrt(q.shape[-1])
        atn = jax.nn.softmax(atn, axis=-1)
        atn = hk.dropout(hk.next_rng_key(), dropout_rate, atn)
        
        out = jnp.einsum('b t n h, b n s t -> b s n h', v, atn)
        out = einops.rearrange(out, 'b n h c -> b n (h c)')
        out = hk.Linear(output_size = dim)(out)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0, out)
        return out
    return attention_layer

def make_mixing_layer(dim, dim_inner, dropout_rate):
    def mixing_layer(x, training):
        out = hk.Linear(dim_inner)(x)
        out = jax.nn.gelu(out)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0, out)
        out = hk.Linear(dim)(out)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0, out)
        return out
    return mixing_layer

def make_gmlp_layer(dim, dim_inner, n_heads, dropout_rate, init_eps = 1e-1):
    def gmlp_layer(x, training):
        out = hk.Linear(dim_inner)(x)
        out = jax.nn.gelu(out)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0, out)
        
        # Spatial Gating Unit(SGU)
        res, gate = jnp.split(out, 2, axis = -1)
        gate = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = False, create_offset = False)(gate)
        n_patches = x.shape[1]
        weight = hk.get_parameter('sgu_weight', [n_heads, n_patches, n_patches], init = hk.initializers.UniformScaling(scale = init_eps))
        bias   = hk.get_parameter('sgu_bias',   [n_heads, n_patches], init = jnp.ones)
        gate = einops.rearrange(gate, 'b n (h d) -> b h n d', h = n_heads)
        gate = jnp.einsum('b h n d, h m n -> b h m d', gate, weight)
        gate = gate + einops.rearrange(bias, 'h n -> () h n ()')
        gate = einops.rearrange(gate, 'b h n d -> b n (h d)')
        out = jax.nn.gelu(gate)
        out = out * res
        
        out = hk.Linear(dim)(out)
        out = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0, out)
        
        return out
    return gmlp_layer


def make_film_generator(depth, dim, dim_inner, norm = None, activation = None, dropout = None):
    def film_generator(x, training):
        out = x
        for _ in range(depth - 1):
            out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = False, create_offset = False)(out) if norm is not None else out
            out = hk.Linear(dim_inner)(out)
            out = jnp.nn.gelu(out) if activation is not None else out
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = False, create_offset = False)(out) if norm is not None else out
        out = hk.Linear(dim*2)(out)
        out = jnp.nn.gelu(out) if activation is not None else out
        
        return jnp.split(out, 2, -1)
    return film_generator


def make_pos_cls_layer(B, N, C, dropout):
    def pos_cls_layer(x, training):
        cls_token = hk.get_parameter('cls_embedder', [1,   1, C], init = hk.initializers.RandomNormal() )
        pos_info  = hk.get_parameter('pos_embedder', [1, N+1, C], init = hk.initializers.RandomNormal(stddev=0.02) )
        
        cls_token = einops.repeat(cls_token, '() n d -> b n d', b = B)
        out = jnp.concatenate([cls_token, x], axis = 1)
        out += pos_info[:, :(N+1)]
        out = hk.dropout(hk.next_rng_key(), dropout if training else 0, out)
        
        return out
    return pos_cls_layer






















