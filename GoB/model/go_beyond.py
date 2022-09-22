from cmath import exp
import haiku as hk
import einops
import jax
import jax.numpy as jnp
import numpy as np

from .token_mixing import make_mixing_layer, make_patch_layer, make_drop_path_rate, drop_path

def make_go_beyond(patch_size, depth, dim, expansion, dropout, drop_path, layer_scale, **kwargs):
    dim_inner = dim * expansion
    def go_beyond(image, prop, training, check=None):
        
        node = make_patch_layer(patch_size, dim)(image)
        glob = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(prop)
        glob = hk.Linear(dim)(glob)
        
        B, N, D = node.shape
        edge = hk.get_parameter('edge_embedder', [1, N, N, D], init = hk.initializers.RandomNormal(stddev=0.02))
        edge = jnp.repeat(edge, B, axis=0)
        
        droppath_rates = make_drop_path_rate(depth, drop_path)
        aux = None
        
        for idx in range(depth):
            droppath_rate = droppath_rates[idx]
            edge, node, glob = make_gob_layer(dim, dim_inner, dropout, droppath_rate, layer_scale)(edge, node, glob, training)
        
        
        # edge = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(edge)
        # node = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(node)
        # glob = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(glob)
        
        return (edge, node, glob), aux
    return go_beyond

def make_gob_head(depth, dim, expansion, dropout, n_classes, **kwargs):
    dim_inner = dim*expansion
    def head(x, training):
        edge, node, glob = x
        dropout_rate = dropout if training else 0
        
        out = glob
        for idx in range(depth):
            out = hk.LayerNorm(axis=-1,create_scale=True,create_offset=True)(out)
            out = out + make_mixing_layer(dim, dim_inner, dropout_rate)(out, training)
        
        out = hk.LayerNorm(axis=-1,create_scale=True,create_offset=True)(out)
        out = hk.Linear(n_classes, b_init = hk.initializers.Constant(-2.0))(out)
        return out
    return head


def make_gob_meta_layer(dim, dropout, droppath, lscale):
    def gob_meta_layer(x, res, training):
        out = hk.Linear(dim)(x)
        out = hk.dropout(hk.next_rng_key(), dropout if training else 0.0, out)
        if lscale is not None:
            param = hk.get_parameter('layer_scale', [dim], init = lambda shape, dtype: jnp.ones(shape, dtype)*lscale)
            out = param*out
        out = drop_path(hk.next_rng_key(), droppath if training else 0.0, out)
        return res + out
    return gob_meta_layer


def make_gob_layer(dim, dim_inner, dropout, droppath, lscale):
    def gob_layer(edge, node, glob, training):
        
        e = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(edge)
        n = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(node)
        g = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(glob)
        e = hk.Linear(dim_inner)(e)
        n = hk.Linear(dim_inner)(n)
        g = hk.Linear(dim_inner)(g)
        
        e = jnp.einsum('B N D, B N M D -> B N M D', jnp.einsum('B N D, B D -> B N D',   n, g), e)
        e = hk.dropout(hk.next_rng_key(), dropout if training else 0.0, jax.nn.gelu(e))
        
        n = jnp.einsum('B N M, B M D -> B N D',     jnp.einsum('B N M D, B D -> B N M', e, g), n)
        n = hk.dropout(hk.next_rng_key(), dropout if training else 0.0, jax.nn.gelu(n))
        
        g = jnp.einsum('B D, B D -> B D',           jnp.einsum('B N M D, B N D -> B D', e, n), g)
        g = hk.dropout(hk.next_rng_key(), dropout if training else 0.0, jax.nn.gelu(g))
        
        edge = make_gob_meta_layer(dim, dropout, droppath, lscale)(e, edge, training)
        node = make_gob_meta_layer(dim, dropout, droppath, lscale)(n, node, training)
        glob = make_gob_meta_layer(dim, dropout, droppath, lscale)(g, glob, training)
        
        return edge, node, glob
    return gob_layer

def make_edge_layer(dim, dim_inner, dropout):
    def edge_layer(edge, node, glob, training):
        
        edge = hk.Linear(dim_inner)(edge)
        node = hk.Linear(dim_inner)(node)
        glob = hk.Linear(dim_inner)(glob)
        
        edge1 = jnp.einsum('B N D, B D -> B N D', node, glob)
        edge = jnp.einsum('B N D, B N M D -> B N M D', edge1, edge)
        
        edge = hk.Linear(dim)(edge)
        edge = hk.dropout(hk.next_rng_key(), dropout if training else 0.0, edge)
        
        return edge
    return edge_layer

def make_node_layer(dim, dim_inner, dropout):
    def node_layer(edge, node, glob, training):
        
        edge = hk.Linear(dim_inner)(edge)
        node = hk.Linear(dim_inner)(node)
        glob = hk.Linear(dim_inner)(glob)
        
        node1 = jnp.einsum('B N M D, B D -> B N M', edge, glob)
        node1 = jax.nn.softmax(node1, axis = -1)
        node  = jnp.einsum('B N M, B M D -> B N D', node1, node)
        
        node = hk.Linear(dim)(node)
        node = hk.dropout(hk.next_rng_key(), dropout if training else 0.0, node)
        return node
    return node_layer

def make_glob_layer(dim, dim_inner, dropout):
    def glob_layer(edge, node, glob, training):
        
        edge = hk.Linear(dim_inner)(edge)
        node = hk.Linear(dim_inner)(node)
        glob = hk.Linear(dim_inner)(glob)
        
        glob1 = jnp.einsum('B N M D, B N D -> B D', edge, node)
        glob  = jnp.einsum('B D, B D -> B D', glob1, glob)
        
        glob = hk.Linear(dim)(glob)
        glob = hk.dropout(hk.next_rng_key(), dropout if training else 0.0, glob)
        return glob
    return glob_layer

























