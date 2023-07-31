from copy import deepcopy
import jax.numpy as jnp
import numpy as np
import haiku as hk

def model_config(model_setup, config, split_str = '-'):
    if model_setup is None or model_setup == 'None' or model_setup == 'Test':
        return None
    # ViT-D:32-DEP:4-DR:0.1
    params = model_setup.split(split_str)
    name, params = params[0], params[1:]
    model_def = deepcopy(config[name])
    
    model_dtypes = {k:v[0] for k,v in model_def.items()}
    model_params = {k:v[1] for k,v in model_def.items()}
    model_ptable = {v[2]:k for k,v in model_def.items()}
    
    for param in params:
        key, val = param.split(':')
        param = model_ptable[key]
        t = model_dtypes[param]
        if t == 'int':
            val = int(val)
        elif t == 'float':
            val = float(val)
        elif t == 'str':
            val = str(val)
        elif t == 'bool':
            val = bool(val)
        
        model_params[param] = val
    model_params['model'] = name
    return model_params

class ModelConfig:
    def __init__(self, config):
        self.n_class = config.setup.n_class
        self.xargs = config.setup.xargs
        self.model_type = config.setup.layers.model
        
        self.args = {}
        for key, val in config.setup.layers.items():
            self.args[key] = model_config(val, config[key])
    def model_info(self):
        return deepcopy(self.args)

class Model(hk.Module):
    def __init__(self, cfg):
        super().__init__(name='model')
        self.args = deepcopy(cfg.args)
        self.xargs = deepcopy(cfg.xargs)
        self.n_class = cfg.n_class
        self.model_type = cfg.model_type
        
    def _make_model(self):
        if 'GoB' in self.model_type:
            from .metaformer import make_go_beyond as make_model
        elif 'PN' in self.model_type:
            from .particle_net import make_particle_net as make_model
        else:
            raise ValueError(f'{self.model_type} is not supported')
        from .head import make_head, make_pooling
        
        args = deepcopy(self.args)
        self.model = make_model(**args['model'], norm_args = self.args['norm'], moe_args = args['MoE'])
        self.pooling = make_pooling(**args['pooling'])
        self.head  = make_head(**args['head'], n_classes=self.n_class)
        if self.args['FiLM'] is not None:
            from .token_mixing import make_film_gen
            self.film_layer  = lambda : make_film_gen(**args['FiLM'])
        else:
            self.film_layer = lambda *args, **kwargs: (None, None)
        
    def __call__(self, batch, training = True):
        data = {key:batch[val] for key,val in self.xargs.items()}
        # below is just testing
        counter = hk.get_state('counter', (), jnp.int32, init=hk.initializers.Constant(0))
        counter = counter + 1
        gamma, beta = self.film_layer()(batch, training=training)
        
        tokens, aux = self.model(**data, training = training, gamma=gamma, beta=beta)
        embedding = self.pooling(tokens, training=training)
        out = self.head(embedding, training = training)
        hk.set_state('counter', counter)
        return out, aux

def make_model(config, mp_policy, without_state=False):
    hk.mixed_precision.set_policy(Model, mp_policy)
    # import jmp
    # mp_bn_policy = jmp.Policy(param_dtype = jnp.float32, compute_dtype = jnp.float32, output_dtype = jnp.float32)
    #hk.mixed_precision.set_policy(hk.BatchNorm, mp_bn_policy)
    
    if 'Test' in config.setup.layers.model:
        from metaformer import Test
        hk.mixed_precision.set_policy(Test, mp_policy) 
        def _forward(batch, training = True):
            t = Test()
            return t(batch, training = training)
        forward = hk.transform_with_state(_forward)
        return forward
    
    model_cfg = ModelConfig(config)
    def _forward(batch, training = True):
        model = Model(model_cfg)
        model._make_model()
        return model(batch, training = training)
    if without_state:
        forward = hk.transform(_forward)
    else:
        forward = hk.transform_with_state(_forward)
    return forward

def make_model_info(params, config):
    total_params = 0
    total_bytes = 0
    
    def div_A(N, A, k):
        tmp = N / A
        if tmp >= A:
            return div_A(tmp, A, k+1)
        return tmp, k+1
    
    def str_k(k):
        if k not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return '?'
        return ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'][k]
    
    for i, key in enumerate(params):
        param = params[key]
        for k in param.keys():
            # print(k, param[k].shape)
            d = str(param[k].dtype)
            total_params += np.prod(param[k].shape)
            total_bytes += param[k].nbytes
    
    B, k = div_A(total_params, 1024, 0)
    total_size = f'{B:.1f}{str_k(k)}iB'
    model_info = ModelConfig(config).model_info()
    return {'params': total_params, 'size': total_size, 'byte': total_bytes}, model_info









