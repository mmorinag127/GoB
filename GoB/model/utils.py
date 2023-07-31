from copy import deepcopy
import jax.numpy as jnp
import numpy as np
import haiku as hk

def model_config(model_setup, config, split_str = '-', is_model = True):
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
    if is_model:
        model_params['model'] = name
    return model_params

class ModelConfig:
    def __init__(self, config):
        self.model_args = model_config(config.setup.model, config.model)
        self.moe_args = model_config(config.setup.MoE, config.MoE)
        self.head_args = model_config(config.setup.head, config.head)
        self.norm_args = model_config(config.setup.norm, config.norm, is_model=False)
        self.film_args = model_config(config.setup.FiLM, config.FiLM)
        self.n_class = config.setup.n_class
        if self.head_args is not None:
            self.head_args['n_classes'] = self.n_class
    
    def model_info(self):
        ret = {'model':{}}
        
        if self.model_args is not None:
            for key, val in self.model_args.items():
                ret[f'model'][f'{key}'] = val
        
        if self.moe_args is not None:
            ret['MoE'] = {}
            for key, val in self.moe_args.items():
                ret[f'MoE'][f'{key}'] = val
        
        if self.head_args is not None:
            ret['head'] = {}
            for key, val in self.head_args.items():
                ret[f'head'][f'{key}'] = val
        
        if self.norm_args is not None:
            ret['norm'] = {}
            for key, val in self.norm_args.items():
                ret[f'norm'][f'{key}'] = val
        
        if self.film_args is not None:
            ret['film'] = {}
            for key, val in self.film_args.items():
                ret[f'film'][f'{key}'] = val
        return ret

class Model(hk.Module):
    def __init__(self, cfg):
        super().__init__(name='model')
        self.model_name = cfg.model_args['model']
        self.model_args = cfg.model_args
        self.moe_args = cfg.moe_args
        self.head_args = cfg.head_args
        self.norm_args = cfg.norm_args
        self.film_args = cfg.film_args
        self.n_class = cfg.n_class
        
    def _make_model(self):
        from .token_mixing import make_film
        if 'GoB' in self.model_name:
            from .go_beyond import make_go_beyond as make_model_
            from .go_beyond import make_gob_head as make_head_
        elif 'ViT' in self.model_name or 'Mixer' in self.model_name:
            from .metaformer import make_metaformer as make_model_
            from .metaformer import make_head as make_head_
            self.xargs = ['image']
        elif 'FvT' in self.model_name:
            from .metaformer import make_transformer as make_model_
            from .metaformer import make_head as make_head_
            self.xargs = ['node', 'mask']
        
        self.moe_aux_loss_w_label = 0.0
        if self.moe_args is not None:
            if self.moe_args['loss_w_label'] > 0.0:
                self.moe_aux_loss_w_label = self.moe_args['loss_w_label']
        
        self.model = make_model_(**deepcopy(self.model_args), moe_args = deepcopy(self.moe_args), norm_args = deepcopy(self.norm_args))
        self.head  = make_head_(**deepcopy(self.head_args))
        self.film  = make_film(**deepcopy(self.film_args)) if self.film_args is not None else lambda *args, **kwargs: (None, None)
        
    def __call__(self, batch, training = True):
        data = [batch[arg] for arg in self.xargs]
        # below is just testing
        counter = hk.get_state('counter', (), jnp.int32, init=hk.initializers.Constant(0))
        counter = counter + 1
        gamma, beta = self.film(batch, training = training)
        out, aux = self.model(*data, gamma = gamma, beta = beta, training = training)
        out = self.head(out, gamma = gamma, beta = beta, training = training)
        hk.set_state('counter', counter)
        # if self.moe_aux_loss_w_label > 0.0:
        #     from .moe_layer import expert_label_importance_loss
        #     expert_label_loss = []
        #     for i, gates_softmax in enumerate(aux['moe_score']):
        #         tmp_aux_loss = expert_label_importance_loss(self.n_class, gates_softmax, batch['label'], batch['mask'])
        #         aux['aux_loss'][i] += self.moe_aux_loss_w_label * tmp_aux_loss
        #         expert_label_loss.append(tmp_aux_loss)
        #     aux['expert_label_loss'] = expert_label_loss
        # else:
        #     aux['expert_label_loss'] = [0]
        return out, aux

class Data2VecModel(hk.Module):
    def __init__(self, cfg):
        super().__init__()
        self.isGoB = cfg.isGoB
        self.model_args = cfg.model_args
        self.moe_args = cfg.moe_args
        # self.head_args = cfg.head_args
        self.film_args = cfg.film_args
        
        
        
        
    def _make_model(self):
        from .token_mixing import make_film
        if self.isGoB:
            from .go_beyond import make_go_beyond as make_model_
            from .go_beyond import make_gob_head as make_head_
        else:
            from .metaformer import make_metaformer as make_model_
            from .metaformer import make_head as make_head_
        
        self.model = make_model_(**self.model_args, moe_args = self.moe_args)
        self.head  = make_head_(**self.head_args)
        self.film  = make_film(**self.film_args) if self.film_args is not None else lambda *args, **kwargs: (None, None)
        # self.film  = make_film(**self.film_args) if self.film_args is not None else None
        
    def __call__(self, batch, training = True, check = None):
        image = batch['image']
        aux =  None
        # below is just testing
        counter = hk.get_state('counter', (), jnp.int32, init=hk.initializers.Constant(0))
        counter = counter + 1
        gamma, beta = None, None
        # if self.film is not None:
        #     gamma, beta = self.film(batch, traninng = training)
        
        #gamma, beta = self.film(batch, training = training)
        out, aux = self.model(image, gamma = gamma, beta = beta, training = training, check = check)
        out = self.head(out, gamma = gamma, beta = beta, training = training)
        hk.set_state('counter', counter)
        
        return out, aux

def make_model(config, mp_policy, without_state=False):
    hk.mixed_precision.set_policy(Model, mp_policy)
    
    if 'Test' in config.setup.model:
        print('!!!!!!!!!!!!!!!!!!!!!!')
        from .metaformer import Test
        hk.mixed_precision.set_policy(Test, mp_policy) 
        def _forward(batch, training = True, check = None):
            t = Test()
            return t(batch, training = training, check=check)
        forward = hk.transform_with_state(_forward)
        return forward
    
    model_cfg = ModelConfig(config)
    def _forward(batch, training = True, check = None):
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
            d = str(param[k].dtype)
            total_params += np.prod(param[k].shape)
            total_bytes += param[k].nbytes
    
    B, k = div_A(total_params, 1024, 0)
    total_size = f'{B:.1f}{str_k(k)}iB'
    model_info = ModelConfig(config).model_info()
    return {'params': total_params, 'size': total_size, 'byte': total_bytes}, model_info
    
    
    
    
