from copy import deepcopy
import jax.numpy as jnp
import numpy as np
import haiku as hk

def make_model(config, mp_policy):
    model_type = config.setup.model
    head_type = config.setup.head
    moe_type = config.setup.MoE
    film_type = config.setup.FiLM

    if 'Test' in model_type:
        def _forward(batch, training = True, check = None):
            image = batch['image']
            out = hk.Linear(config.setup.n_class)(jnp.reshape(image, (image.shape[0], -1)))
            return out, {}
        
        forward = hk.transform_with_state(_forward)
        return forward
    
    model_args = deepcopy(config.model[model_type])
    
    
    if moe_type is not None and moe_type != 'None':
        model_args['MoE'] = config.MoE[moe_type]
    
    film_args = None
    if film_type is not None and film_type != 'None':
        film_args = config.FiLM[film_type]
        from .token_mixing import make_film_generator
    
    
    
    
    if 'GoB' in model_type:
        from .go_beyond import make_go_beyond as make_model
        from .go_beyond import make_gob_head as make_head
    else:
        from .metaformer import make_metaformer as make_model
        from .metaformer import make_head as make_head
    
    def _forward(batch, training = True, check = None):
        image = batch['image']
        
        aux = None
        
        gamma, beta = None, None
        if film_args is not None:
            gamma, beta = make_film_generator(**film_args)(batch['prop'], training = training)
        
        model = make_model(**model_args)
        out, aux = model(image, gamma = gamma, beta = beta, training = training, check = check)
        out = make_head(**config.head[head_type])(out, gamma = gamma, beta = beta, training = training)
        return out, aux
    
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
    
    
    
    
