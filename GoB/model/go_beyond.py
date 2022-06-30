from cmath import exp
import haiku as hk
import einops
import jax
import jax.numpy as jnp
import numpy as np

from .layers import make_drop_path_rate, drop_path
from .token_mixing import  make_patch_layer


class GoB(hk.Module):
    def __init__(self, patch_size, depth, dim, expansion, n_heads, dropout, drop_path, layer_scale, qkv_bias, n_classes):
        super().__init__()
        self.n_classes = n_classes
        
        self.patch_size = patch_size
        self.depth = depth
        self.dim = dim
        self.expansion = expansion
        self.n_heads = n_heads
        self.dropout = dropout if dropout is not None else 0
        self.drop_path = make_drop_path_rate(depth, drop_path)
        self.layer_scale = layer_scale
        self.qkv_bias = qkv_bias
        self.dim_inner = dim * expansion
        self.dim_head = self.dim_inner // self.n_heads
        
    def __call__(self, image, training):
        dropout_rate = self.dropout if training else 0
        
        # making patch
        node = make_patch_layer(self.patch_size, self.dim)(image)
        
        # position embedding and cls token
        B, N, _ = node.shape
        cls_token = hk.get_parameter('cls_embedder', [1, 1, self.dim], init = hk.initializers.RandomNormal() )
        glob = einops.repeat(cls_token, '() n d -> b n d', b = B)
        
        # position embedding and cls token
        pos_info  = hk.get_parameter('pos_embedder', [1, N, N], init = hk.initializers.RandomNormal(stddev=0.02) )
        edge = einops.repeat(pos_info, '() n m -> b h n m', b = B, h = self.n_heads)
        
        
        # MSA block
        for i in range(self.depth):
            drop_path_rate = self.drop_path[i] if training else 0.0
            
            edge_, node_ = make_gob_node_layer(self.dim, self.dim_inner, dropout_rate, self.n_heads, self.dim_head, self.qkv_bias)(edge, node, training)
            edge = make_gob_meta_layer(edge.shape[-1], drop_path_rate, self.layer_scale)(edge, edge_, training)
            node = make_gob_meta_layer(self.dim, drop_path_rate, self.layer_scale)(node, node_, training)
            
            glob_ = make_gob_glob_layer(self.dim, self.dim_inner, dropout_rate, self.n_heads, self.dim_head, self.qkv_bias)(node, glob, training)
            glob = make_gob_meta_layer(glob.shape[-1], drop_path_rate, self.layer_scale)(glob, glob_, training)
        
        
        out = jnp.squeeze(glob, axis = 1)
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(out)
        out = hk.Linear(output_size = self.n_classes)(out)
        return out
        
        
        
        
def make_gob_meta_layer(dim, drop_path_rate, lscale):
    def meta_layer(x, out, training):
        if lscale is not None:
            param = hk.get_parameter('layer_scale', dim, init = lambda shape, dtype: jnp.ones(shape, dtype)*lscale)
            out = param*out
        out = drop_path(hk.next_rng_key(), drop_path_rate if training else 0.0, out)
        return x + out
    return meta_layer


def make_gob_node_layer(dim, dim_inner, dropout_rate, n_heads, dim_head, qkv_bias):
    def gob_node_layer(edge, node, training):
        
        B, N, _ = node.shape
        qk = jnp.reshape(hk.Linear(N,         with_bias = qkv_bias)(edge), [B, n_heads, N, N])
        v  = jnp.reshape(hk.Linear(dim_inner, with_bias = qkv_bias)(node), [B, N, n_heads, dim_head])
        
        
        edge = qk/np.sqrt(v.shape[-1])
        edge = jax.nn.softmax(edge, axis = -1)
        edge = hk.dropout(hk.next_rng_key(), dropout_rate, edge)
        
        node = jnp.einsum('b m h c, b h n m -> b n h c', v, edge)
        node = einops.rearrange(node, 'b n h c -> b n (h c)')
        
        node = hk.Linear(dim)(node)
        node = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0, node)
        
        #edge = jnp.mean(edge, axis = 1)
        return edge, node
    return gob_node_layer

def make_gob_glob_layer(dim, dim_inner, dropout_rate, n_heads, dim_head, qkv_bias):
    def gob_glob_layer(node, glob, training):
        
        B, N, _ = node.shape
        q = jnp.reshape(hk.Linear(dim_inner, with_bias = qkv_bias)(node), [B, N, n_heads, dim_head])
        k = jnp.reshape(hk.Linear(dim_inner, with_bias = qkv_bias)(glob), [B, 1, n_heads, dim_head])
        v = jnp.reshape(hk.Linear(dim_inner, with_bias = qkv_bias)(glob), [B, 1, n_heads, dim_head])
        
        qk = jnp.einsum('b n h c, b k h c -> b h k n', q, k)
        qk /= np.sqrt(q.shape[-1])
        qk = jax.nn.softmax(qk, axis = -1)
        qk = hk.dropout(hk.next_rng_key(), dropout_rate, qk)
        
        glob = jnp.einsum('b h k n, b k h c -> b k n', qk, v)
        glob = hk.Linear(dim)(glob)
        glob = hk.dropout(hk.next_rng_key(), dropout_rate if training else 0, glob)
        return glob
    return gob_glob_layer






















