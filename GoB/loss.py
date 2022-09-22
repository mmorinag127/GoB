import jax
import jax.numpy as jnp
import optax
import tree





def l2_loss(params):
    l2_params = [p for p in tree.flatten(params) if p.ndim > 1 ]
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in l2_params)

def make_softmax_ce_loss(config):
    smooth_label = config.setup.smooth_label
    weight_decay = config.setup.weight_decay
    n_class = config.setup.n_class
    
    def loss_f(logits, labels, params, aux_loss = None):
        #one_hot = jax.nn.one_hot(labels, n_class)
        if smooth_label is not None:
            labels = optax.setup.smooth_label(labels, smooth_label)
        
        one_hot = jax.nn.one_hot(labels, n_class)
        ce_loss = optax.softmax_cross_entropy(logits = logits, labels = one_hot)#.mean()
        ce_loss = jnp.nan_to_num(ce_loss, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        ce_loss = jnp.mean(ce_loss)
        loss = ce_loss + weight_decay * l2_loss(params)
        if aux_loss is not None:
            loss += aux_loss
        
        return loss
    return loss_f

def make_cb_ce_loss(config):
    n_class = config.setup.n_class
    beta = config.loss.cb_ce_loss.beta
    gamma = config.loss.cb_ce_loss.gamma
    weight_decay = config.setup.weight_decay
    #class_weights = config.cb_ce_loss.class_weight
    
    def loss_f(logits, labels, params, aux_loss = None):
        
        one_hot = jax.nn.one_hot(labels, n_class)
        
        counts = jnp.sum(one_hot, axis = 0)
        
        weight = (1.0 - beta) / (1.0 - jnp.power(beta, counts))
        #weight = weight.at[weight == float('inf')].set(0.0)
        weight = weight / jnp.sum(weight, axis = -1)
        weight = jnp.nan_to_num(weight, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        sm = jax.nn.softmax(logits, axis = 1)
        
        if gamma is not None:
            focal = jnp.power(1 - sm, gamma)
            if False:
                pt = jnp.sum(labels*sm + (1.0 - labels)*(1.0-sm), axis = -1)
                focal = jnp.power(1 - pt, gamma)
            
            
            focal = jnp.nan_to_num(focal, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            focal = 1.0
        
        
        
        weight = weight * focal * one_hot
        #weight = jnp.nan_to_num(weight, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        #sm = jnp.nan_to_num(sm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        ce_loss = -jnp.sum(weight * jnp.log(sm), axis = -1)
        
        ce_loss = jnp.nan_to_num(ce_loss, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if aux_loss is not None:
            ce_loss += aux_loss
        
        ce_loss = jnp.mean(ce_loss)
        
        L2_loss = l2_loss(params)
        L2_loss = jnp.nan_to_num(L2_loss, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        loss = ce_loss + weight_decay * L2_loss
        
        return loss
    return loss_f

def make_cb_ce_cmean_loss(config):
    n_class = config.setup.n_class
    beta = config.loss.cb_ce_loss.beta
    gamma = config.loss.cb_ce_loss.gamma
    weight_decay = config.setup.weight_decay
    #class_weights = config.cb_ce_loss.class_weight
    
    def loss_f(logits, labels, params, aux_loss = None):
        
        one_hot = jax.nn.one_hot(labels, n_class)
        
        counts = jnp.sum(one_hot, axis = 0)
        
        weight = (1.0 - beta) / (1.0 - jnp.power(beta, counts))
        weight = weight / jnp.sum(weight, axis = -1)
        weight = jnp.nan_to_num(weight, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        sm = jax.nn.softmax(logits, axis = 1)
        
        if gamma is not None:
            focal = jnp.power(1 - sm, gamma)
            if False:
                pt = jnp.sum(labels*sm + (1.0 - labels)*(1.0-sm), axis = -1)
                focal = jnp.power(1 - pt, gamma)
            
            
            focal = jnp.nan_to_num(focal, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            focal = 1.0
        
        weight = weight * focal * one_hot
        #weight = jnp.nan_to_num(weight, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        #sm = jnp.nan_to_num(sm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        ce_loss = -jnp.sum(weight * jnp.log(sm), axis = -1)
        
        ce_loss = jnp.nan_to_num(ce_loss, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if aux_loss is not None:
            ce_loss += aux_loss
        
        ce_loss = jnp.mean(ce_loss)
        
        L2_loss = l2_loss(params)
        L2_loss = jnp.nan_to_num(L2_loss, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        loss = ce_loss + weight_decay * L2_loss
        
        return loss
    return loss_f

def make_cb_bce_loss(config):
    n_class = config.setup.n_class
    beta = config.loss.cb_ce_loss.beta
    gamma = config.loss.cb_ce_loss.gamma
    weight_decay = config.setup.weight_decay
    #class_weights = config.cb_ce_loss.class_weight
    
    def loss_f(logits, labels, params, aux_loss = None):
        
        one_hot = jax.nn.one_hot(labels, n_class)
        
        counts = jnp.sum(one_hot, axis = 0)
        
        weight = (1.0 - beta) / (1.0 - jnp.power(beta, counts))
        #weight = weight.at[weight == float('inf')].set(0.0)
        weight = weight / jnp.sum(weight, axis = -1)
        weight = jnp.nan_to_num(weight, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        sm = jax.nn.softmax(logits, axis = 1)
        
        if gamma is not None:
            pt = one_hot*sm + (1 - one_hot)*(1-sm)
            focal = jnp.power(1 - pt, gamma)
            focal = jnp.nan_to_num(focal, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            focal = 1.0
        
        weight = weight * focal * one_hot
        
        
        prob = one_hot*jnp.log(sm) + (1-one_hot)*jnp.log(1-sm)
        ce_loss = -jnp.sum(weight * prob, axis = -1)
        
        ce_loss = jnp.nan_to_num(ce_loss, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if aux_loss is not None:
            ce_loss += aux_loss
        
        ce_loss = jnp.mean(ce_loss)
        
        L2_loss = l2_loss(params)
        L2_loss = jnp.nan_to_num(L2_loss, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        loss = ce_loss + weight_decay * L2_loss
        
        return loss
    return loss_f



# class AffinityLoss(nn.Module):
#     def __init__(self, lam):
#         super(AffinityLoss,self).__init__()
#         self.lam = lam

#     def forward(self, y_pred, y_true):
#         onehot   = y_true[:,:-1]
#         distance = y_pred[:,:-1]
#         rw       = torch.mean(y_pred[:,-1])
#         d_fi_wyi = torch.sum(onehot * distance, -1).unsqueeze(1)
#         losses   = torch.clamp(self.lam + distance - d_fi_wyi, min = 0)
#         L_mm     = torch.sum(losses*(1.0 - onehot), -1)/y_true.size(0)
#         loss     = torch.sum(L_mm + rw, -1)
#         return loss


