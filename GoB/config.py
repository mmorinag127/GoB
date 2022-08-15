from functools import reduce
import operator
import os, datetime

from enum import Enum

import yaml 

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
            self.run_name = f'{self.config.setup.model}/{self.run_name}'
            self.workdir = f'{self.exp_dir}/{self.run_name}-{self.config.seed}'
            self.make_workdir()
            self.save_config(opts.model_name)
        else:
            self.exp_dir = f'results/{self.experiment_name}'
            self.run_name = f'{opts.model}/{self.run_name}'
            self.workdir = f'{self.exp_dir}/{self.run_name}-{opts.seed}'
            
            self.make_config(f'{self.workdir}/config-{opts.model_name}.yaml', seed = opts.seed)
            
            for key, val in kwargs.items():
                self.update(key, val)
            
            #self.update('phases', ['test'] )
    
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
    
    def print_summary(self, process_idx, run_info, model_summary, datasets, model_info, phases, hparams):
        if process_idx != 0:
            return
        
        if model_summary is not None:
            d = '-'*20
            logger.info(f'{d:<20s} :')
            for s in model_summary.split('\n'):
                #logger.info(s)
                print(s)
        
        for key, val in self.config.setup.items():
            v = f'{val}'
            logger.info(f'{key: >20s} : {v:<20s}')
        
        if run_info is not None:
            d = '-'*20
            logger.info(f'{d:<20s} :')
            d = f'experiment info'
            logger.info(f'{d:<20s} :')
            for key, val in run_info.items():
                v = f'{val}'
                logger.info(f'{key: >20s} : {v:<20s}')
        
        if model_info is not None:
            d = '-'*20
            logger.info(f'{d:<20s} :')
            d = f'model info'
            logger.info(f'{d:<20s} :')
            for key, val in model_info.items():
                v = f'{val}'
                logger.info(f'{key: >20s} : {v:<20s}')
        
        if hparams is not None:
            d = '-'*20
            logger.info(f'{d:<20s} :')
            d = f'hyper parameters'
            logger.info(f'{d:<20s} :')
            for key, val in hparams.items():
                v = f'{val}'
                logger.info(f'{key: >20s} : {v:<20s}')
                
        if datasets is not None:
            d = '-'*20
            logger.info(f'{d:<20s} :')
            for phase in phases:
                d = f'data_info({phase})'
                logger.info(f'{d:<20s} :')
                for key, val in datasets[phase].info().items():
                    k = f'{key}'
                    v = f'{val}'
                    logger.info(f'{k: >20s} : {v:<20s}')
        d = '-'*20
        logger.info(f'{d:<20s} :')
        logger.info(f'{self.experiment_name:20s} : {self.run_name:20s}')
        k = 'workdir'
        logger.info(f'{k:20s} : {self.workdir}')
        
        
        
        
        
        
        