#from cmath import exp
import haiku as hk
import einops
import jax.numpy as jnp


from .layers import make_drop_path_rate
from .token_mixing import make_meta_layer, make_attention_layer, make_mixing_layer, make_mixer_layer, make_patch_layer, make_meta_film_layer


class ViT(hk.Module):
    def __init__(self, patch_size, depth, dim, expansion, n_heads, dropout, drop_path, layer_scale, qkv_bias, n_classes, film = None):
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
        self.film = film
    
    def __call__(self, image, prop, training):
        dropout_rate = self.dropout if training else 0
        
        # making patch
        out = make_patch_layer(self.patch_size, self.dim)(image)
        
        # position embedding and cls token
        B, N, C = out.shape
        cls_token = hk.get_parameter('cls_embedder', [1,   1, C], init = hk.initializers.RandomNormal() )
        pos_info  = hk.get_parameter('pos_embedder', [1, N+1, C], init = hk.initializers.RandomNormal(stddev=0.02) )
        cls_token = einops.repeat(cls_token, '() n d -> b n d', b = B)
        out = jnp.concatenate([cls_token, out], axis = 1)
        out += pos_info[:, :(N+1)]
        out = hk.dropout(hk.next_rng_key(), dropout_rate, out)
        
        gamma = hk.Linear()
        
        
        # MSA block
        for i in range(self.depth):
            drop_path_rate = self.drop_path[i] if training else 0.0
            out = make_meta_film_layer(self.dim, make_attention_layer(self.dim, self.dim_inner, dropout_rate, self.dim_head, self.n_heads, self.qkv_bias), drop_path_rate, self.layer_scale)(out, gamma, beta, training)
            out = make_meta_film_layer(self.dim, make_mixing_layer(self.dim, self.dim_inner, dropout_rate), drop_path_rate, self.layer_scale)(out, gamma, beta, training)
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(out)
        out = jnp.mean(out, axis=1)
        out = hk.Linear(output_size = self.n_classes)(out)
        return out

class Mixer(hk.Module):
    def __init__(self, patch_size, depth, dim, expansion, dropout, drop_path, layer_scale, n_classes):
        super().__init__()
        self.n_classes = n_classes
        
        self.patch_size = patch_size
        self.depth = depth
        self.dim = dim
        self.expansion = expansion
        self.dropout = dropout if dropout is not None else 0
        self.drop_path = make_drop_path_rate(depth, drop_path)
        self.layer_scale = layer_scale
        self.dim_inner = dim * expansion
        
    def __call__(self, image, training):
        dropout_rate = self.dropout if training else 0
        
        
        # making patch
        out = make_patch_layer(self.patch_size, self.dim)(image)
        
        # Mixer Block
        for i in range(self.depth):
            drop_path_rate = self.drop_path[i] if training else 0.0
            
            out = make_meta_layer(self.dim, make_mixer_layer( self.dim, self.dim_inner, dropout_rate), drop_path_rate, self.layer_scale)(out, training)
            out = make_meta_layer(self.dim, make_mixing_layer(self.dim, self.dim_inner, dropout_rate), drop_path_rate, self.layer_scale)(out, training)
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(out)
        out = jnp.mean(out, axis=1)
        out = hk.Linear(output_size = self.n_classes)(out)
        return out

class gMLP(hk.Module):
    def __init__(self, patch_size, depth, dim, expansion, n_heads, dropout, drop_path, layer_scale, n_classes):
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
        self.dim_inner = dim * expansion
        self.dim_head = self.dim_inner // self.n_heads
        
    def __call__(self, image, training):
        dropout_rate = self.dropout if training else 0
        
        # making patch
        out = make_patch_layer(self.patch_size, self.dim)(image)
        
        # Mixer Block
        for i in range(self.depth):
            drop_path_rate = self.drop_path[i] if training else 0.0
            
            out = make_meta_layer(self.dim, make_mixer_layer( self.dim, self.dim_inner, dropout_rate), drop_path_rate, self.layer_scale)(out, training)
            out = make_meta_layer(self.dim, make_mixing_layer(self.dim, self.dim_inner, dropout_rate), drop_path_rate, self.layer_scale)(out, training)
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(out)
        out = jnp.mean(out, axis=1)
        out = hk.Linear(output_size = self.n_classes)(out)
        return out


class CNN(hk.Module):
    def __init__(self, dim, patch_size, n_classes):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.n_classes = n_classes
        
    def __call__(self, image, training):
        
        out = hk.Conv2D(self.dim, kernel_shape = self.patch_size, stride = self.patch_size)(image)
        out = hk.Conv2D(self.dim, kernel_shape = 2, stride = 2)(out)
        out = hk.Conv2D(self.dim, kernel_shape = 2, stride = 2)(out)
        
        out = hk.Flatten()(out)
        out = hk.Linear(self.n_classes)(out)
        #out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(out)
        return out
        

class Test(hk.Module):
    def __init__(self, dim, patch_size, n_classes):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.n_classes = n_classes
        
    def __call__(self, image, training):
        
        out = hk.Flatten()(image)
        out = hk.Linear(self.n_classes)(out)
        #out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(out)
        return out
        




























