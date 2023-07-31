from functools import reduce
import operator
import os, datetime

from enum import Enum

import yaml 
import hashlib

from logger import make_logger
logger = make_logger(__name__)

class TrainingMode(Enum):
    nominal = 'nominal'
    train = 'train'
    eval_test = 'eval-test'
    eval_loss = 'eval-loss'
    eval_weight = 'eval-weight'
    data2vec = 'data2vec'
    transfer = 'transfer'
    fine_tuning = 'fine-tuning'
    def training(self):
        if self.value in ['train', 'nominal', 'data2vec', 'finetuning', 'transfer']:
            return True
        return False
    def restore(self):
        if self.value in ['eval-test', 'eval-loss', 'eval-weight', 'trainsfer', 'fine-tuning']:
            return True
        return False
    def eval(self):
        if self.value in ['eval-test', 'eval-loss', 'eval-weight', 'trainsfer', 'fine-tuning', 
                        'nominal', 'data2vec', 'finetuning', 'transfer']:
            return True


class Config:
    def __init__(self, opts, process_idx, **kwargs):
        self.mode = TrainingMode(opts.mode)
        self.process_idx = process_idx
        
        self.experiment_name = opts.experiment_name
        self.run_name        = opts.run_name
        if self.run_name is None:
            self.run_name = datetime.datetime.now().strftime("%Y-%m%d-%H:%M")
        
        if self.mode.training():
            self.make_config(path = opts.config, **kwargs)
            self.exp_dir = f'results/{self.experiment_name}'
            _run_name = ''
            for key, val in self.config.setup.layers.items():
                if val is not None and val != 'None':
                    _run_name += f'/{val}'
            self._run_name = _run_name
            
            self.workdir = f'{self.exp_dir}/{self._run_name}/{self.run_name}-{self.config.setup.seed}'
            self.make_workdir()
            self.save_config(opts.model_name)
        else:
            self.exp_dir = f'results/{self.experiment_name}'
            self._run_name = f'{opts.model}/{opts.norm}/{opts.pool}/{opts.head}'
            if opts.MoE not in ['None', None]:
                self._run_name = f'{self._run_name}/{opts.MoE}'
            if opts.FiLM not in ['None', None]:
                self._run_name = f'{self._run_name}/{opts.FiLM}'
                
            self.workdir = f'{self.exp_dir}/{self._run_name}/{self.run_name}-{opts.seed}'
            
            self.make_config(f'{self.workdir}/config-{opts.model_name}.yaml', seed = opts.seed)
            
            for key, val in kwargs.items():
                self.update(key, val)
            
            self.update('phases', ['test'] )
    
    def apply_hash(self, text):
        return hashlib.sha512(text.encode()).hexdigest()
    
    def __getitem__(self, key):
        return self.config[key]
    
    
    def __getattr__(self, key):
        if key in self.config.keys():
            return getattr(self.config, key)
    
    def make_config(self, path, **kwargs):
        from attrdict import AttrDict
        
        with open(path, 'r', encoding = 'utf-8') as fi:
            self.config = AttrDict(yaml.load(fi, Loader = yaml.SafeLoader))
        
        for key, val in kwargs.items():
            self.update(key, val)
        
    
    def update(self, key, val):
        if val is not None:
            key = key.split(':')
            reduce(operator.getitem, key[:-1], self.config)[key[-1]] = val
    
    def make_workdir(self):
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
    
    def save_config(self, model_name):
        with open(f'{self.workdir}/config-{model_name}.yaml', 'w') as f : 
            dconfig = dict(self.config)
            f.write(yaml.dump(dconfig, default_flow_style = False))
    
    def make_table(self, table, max_length=80):
        total_len = 0
        ss = []
        s1, s2 = '', ''
        for key, val in table.items():
            l = max(len(f'{key}'), len(f'{val}'))
            if total_len + l + 3 >= max_length:
                ss.append(s1)
                ss.append(s2)
                total_len = 0
                s1, s2 = '', ''
            v = f'{val}'
            s1 += ' {key: >{l}s} |'.format(key=key, l=l)
            s2 += ' {val: >{l}s} |'.format(val=v,   l=l)
            total_len += l+3
        if len(s1) > 0: 
            ss.append(s1)
            ss.append(s2)
        return ss
    
    def print_table(self, key, val, max_length=80):
        ss = self.make_table(val)
        k = key
        for s in ss:
            logger.info(f'{k: >20s} : {s}')
            k = ''
    
    def print_summary(self, process_idx, run_info, model_info):
        if process_idx != 0:
            return
        def print_t(s, n=20):
            d = '-'*n
            logger.info(f'{d:<20s} :')
            if len(s) > 0:
                logger.info(f'{s:<20s} :')
        
        def print_table2(key, val, max_length=2):
            if type(val) is dict:
                for k, v in val.items():
                    kkey = f'{key}/{k}'
                    print_table2(kkey, v)
            else:
                self.print_table(key, val)
        
        print_t('run setup')
        other = {}
        for key, val in self.config.setup.items():
            if type(val) is dict:
                self.print_table(key, val)
            else:
                v = f'{val}'
                other[key] = v
        self.print_table('', other)
        
        if run_info is not None:
            print_t(f'experiment info')
            for key, val in run_info.items():
                v = f'{val}'
                logger.info(f'{key: >20s} : {v:<20s}')
        
        if model_info is not None:
            print_t('model info')
            for key, val in model_info.items():
                v = f'{val}'
                logger.info(f'{key: >20s} : {v:<20s}')
        
        if self.hparams is not None:
            self._hparams = {}
            print_t('hyper parameters')
            for key, val in self.hparams.items():
                if key == 'setup':
                    for _key in self.hparams['setup']:
                        kkey = self.config['setup'][_key]
                        self._hparams[f'{_key}:{kkey}'] = self.config[_key][kkey]
                else:
                    self._hparams[key] = val
            
            for key, val in self._hparams.items():
                if type(val) is dict:
                    #print_table2(key, val)
                    v = f' '.join(f'{k}:{v}' for k,v in val.items())
                    logger.info(f'{key: <20s} : {v:<20s}')
                else:
                    v = f'{val}'
                    logger.info(f'{key: <20s} : {v:<20s}')
                
        d = '-'*20
        logger.info(f'{d:<20s} :')
        logger.info(f'{self.experiment_name:20s} : {self.run_name:20s}')
        k = 'workdir'
        logger.info(f'{k:20s} : {self.workdir}')
        
