import jax.numpy as jnp
import numpy as np
import haiku as hk

def make_model(config, mp_policy):
    model_type = config.setup.model
    
    from .metaformer import make_metaformer
    
    def _forward(image, training, check = None):
        #image = batch['image']
        #prop  = batch['prop']
        model = make_metaformer(**config.model[model_type])
        return model(image, training = training, check = check)#, prop = prop)
    
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
    
    
    
    
