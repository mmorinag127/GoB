#from cmath import exp
import haiku as hk
import einops
import jax.numpy as jnp


from .layers import make_drop_path_rate
from .token_mixing import make_film_generator, make_meta_layer, make_attention_layer, make_mixing_layer, make_mixer_layer, make_patch_layer, make_pos_cls_layer, make_gmlp_layer






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
        
    def __call__(self, image, training, prop = None):
        dropout_rate = self.dropout if training else 0
        
        # making patch
        out = make_patch_layer(self.patch_size, self.dim)(image)
        
        # position embedding and cls token
        B, N, C = out.shape
        out = make_pos_cls_layer(B, N, C, dropout_rate)(out, training)
        
        # film
        gamma, beta = None, None
        if self.film is not None:
            gamma, beta = make_film_generator(**self.film)(prop, training)
        
        # MSA block
        for i in range(self.depth):
            drop_path_rate = self.drop_path[i] if training else 0.0
            out = make_meta_layer(self.dim, make_attention_layer(self.dim, self.dim_inner, dropout_rate, self.dim_head, self.n_heads, self.qkv_bias), drop_path_rate, self.layer_scale, self.film)(out, training, gamma, beta)
            out = make_meta_layer(self.dim, make_mixing_layer(self.dim, self.dim_inner, dropout_rate), drop_path_rate, self.layer_scale, self.film)(out, training, gamma, beta)
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(out)
        out = jnp.mean(out, axis=1)
        out = hk.Linear(output_size = self.n_classes)(out)
        return out

class Mixer(hk.Module):
    def __init__(self, patch_size, depth, dim, expansion, dropout, drop_path, layer_scale, n_classes, film = None):
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
        self.film = film
        
    def __call__(self, image, training, prop = None):
        dropout_rate = self.dropout if training else 0
        
        # making patch
        out = make_patch_layer(self.patch_size, self.dim)(image)
        
        # film 
        gamma, beta = None, None
        if self.film is not None:
            gamma, beta = make_film_generator(**self.film)(prop, training)
        
        # Mixer Block
        for i in range(self.depth):
            drop_path_rate = self.drop_path[i] if training else 0.0
            
            out = make_meta_layer(self.dim, make_mixer_layer( self.dim, self.dim_inner, dropout_rate), drop_path_rate, self.layer_scale, self.film)(out, training, gamma, beta)
            out = make_meta_layer(self.dim, make_mixing_layer(self.dim, self.dim_inner, dropout_rate), drop_path_rate, self.layer_scale, self.film)(out, training, gamma, beta)
        
        out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(out)
        out = jnp.mean(out, axis=1)
        out = hk.Linear(output_size = self.n_classes)(out)
        return out

class gMLP(hk.Module):
    def __init__(self, patch_size, depth, dim, expansion, n_heads, dropout, drop_path, layer_scale, n_classes, film = None):
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
        self.film = film
        
    def __call__(self, image, training, prop = None):
        dropout_rate = self.dropout if training else 0
        
        # making patch
        out = make_patch_layer(self.patch_size, self.dim)(image)
        
        # film 
        gamma, beta = None, None
        if self.film is not None:
            gamma, beta = make_film_generator(**self.film)(prop, training)
        
        # Mixer Block
        for i in range(self.depth):
            drop_path_rate = self.drop_path[i] if training else 0.0
            out = make_meta_layer(self.dim, make_gmlp_layer( self.dim, self.dim_inner, self.n_heads, dropout_rate), drop_path_rate, self.layer_scale, self.film)(out, training, gamma, beta)
            
        
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
        
    def __call__(self, image, training, prop = None):
        
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
        
    def __call__(self, image, training, prop = None):
        
        out = hk.Flatten()(image)
        out = hk.Linear(self.n_classes)(out)
        #out = hk.LayerNorm(axis = -1, param_axis = -1, create_scale = True, create_offset = True)(out)
        return out

































