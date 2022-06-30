import jax.numpy as jnp
import numpy as np
import haiku as hk

def make_model(config, mp_policy):
    model_type = config.setup.model
    
    if 'ViT' in model_type:
        from .metaformer import ViT as Model
    elif 'CNN' in model_type:
        from .metaformer import CNN as Model
    elif 'Mixer' in model_type:
        from .metaformer import Mixer as Model
    elif 'gMLP' in model_type:
        from .metaformer import gMLP as Model
    elif 'GoB' in model_type:
        from .go_beyond import GoB as Model
    elif 'Test' in model_type:
        from .metaformer import Test as Model
    else:
        raise ValueError(f'{model_type} is not supported!!')
    
    hk.mixed_precision.set_policy(Model, mp_policy)
    
    def _forward(batch, training):
        image = batch['image']
        prop  = batch['prop']
        
        model = Model(**config.model[model_type])
        return model(image, training = training, prop = prop)
    
    forward = hk.transform_with_state(_forward)
    
    return forward


def make_model_info(params):
    total_params = 0
    total_bytes = 0
    
    def div_A(N, A, k):
        tmp = N / A
        if tmp >= A:
            return div_A(tmp, A, k+1)
        return tmp, k
    
    def str_k(k):
        if k not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return '?'
        return ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'][k]
    
    for i, key in enumerate(params):
        param = params[key]
        for k in param.keys():
            d = str(param[k].dtype)
            total_params += np.prod(param[k].shape)
            total_bytes += param[k].nbytes
    
    B, k = div_A(total_params, 1024, 0)
    total_size = f'{B:.1f}{str_k(k)}iB'
    return {'params': total_params, 'size': total_size, 'byte': total_bytes}
    
    
    
    
