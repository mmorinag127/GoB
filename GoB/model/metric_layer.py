import haiku as hk
import jax
import jax.numpy as jnp



def make_arc_face_layer(dim, s, m):
    def arc_face_layer(x, training):
        
        _, _, Din = x.shape
        W = hk.get_parameter('arc_face_W', [Din, dim], init = lambda shape, dtype: jnp.ones(shape, dtype))
        
        W /= jnp.linalg.norm(W, ord=2)
        x /= jnp.linalg.norm(x, ord=2)
        
        logits = W * x
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        