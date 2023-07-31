import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from token_mixing import naive_top_k

def make_particle_embedding_layer(dim, dim_inner, dropout_rate, topK, name = 'particle_emb_layer'):
    def particle_embedding_layer(g, node, pos, training):
        q = hk.Linear(dim_inner, name = f'{name}_linear0')(node)
        k = hk.Linear(dim_inner, name = f'{name}_linear1')(pos)
        v = hk.Linear(dim_inner, name = f'{name}_linear2')(node)
        
        atn = jnp.einsum('bqd, bkd -> bqk', q, k)
        atn /= np.sqrt(q.shape[-1])
        atn = jax.nn.softmax(atn, axis = -1)
        atn = hk.dropout(hk.next_rng_key(), dropout_rate, atn)
        
        
        index = naive_top_k(atn, k = topK, axis = -1)
        
        qkv = jnp.einsum('bqd, bkd -> bqd', atn, v)
        
        
        
        
        
        return x
    return particle_embedding_layer