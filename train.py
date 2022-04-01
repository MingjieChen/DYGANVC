import os
import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
import warnings
warnings.simplefilter('ignore')

from functools import reduce
from munch import Munch

from torch.utils.tensorboard import SummaryWriter
from data_loader import build_data_loader
from model import build_model
from optimizers import build_optimizer
from vqmel_spkemb_ls_trainer import VQMelSpkEmbLSTrainer
import argparse
import random
torch.backends.cudnn.benchmark = True #


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main(config_path):
    # load config yaml
    config = yaml.safe_load(open(config_path))
    print(config)

    # create exp dir
    log_dir = config['log_dir']
    exp_dir = osp.join(log_dir,config['model_name'],config['exp_name'])
    if not osp.exists(exp_dir): os.makedirs(exp_dir, exist_ok=True)
    # back up config yaml to exp dir
    if not osp.exists(osp.join(exp_dir, osp.basename(config_path))):
        shutil.copy(config_path, osp.join(exp_dir, osp.basename(config_path)))
    writer = SummaryWriter(exp_dir + "/tb")
    
    # build data_lodaer
    train_data_loader, dev_data_loader = build_data_loader(config['data_loader'])

    # build model

    model, model_ema = build_model(config['model'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ = [model[key].to(device) for key in model]
    _ = [model_ema[key].to(device) for key in model_ema]
    # build optimizer   
    optimizer = build_optimizer({key: model[key].parameters() for key in model}, config['optimizer'])
    trainer = eval(config['trainer'])(args=Munch(config['loss']), model=model,
                            model_ema=model_ema,
                            optimizer=optimizer,
                            device=device,
                            train_dataloader=train_data_loader,
                            config=config,
                            dev_dataloader=dev_data_loader,
                            fp16_run=config['fp16_run'])
    
    if config.get('pretrained_model', '') != '':
        trainer.load_checkpoint(config['pretrained_model'],
                                load_only_params=config.get('load_only_params', True))

    epochs = config['epochs']
    start_epoch = 1
    for _ in range(trainer.epochs+1, epochs+1):
        epoch = trainer.epochs + 1
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()
        results = train_results.copy()
        results.update(eval_results)
        for key, value in results.items():
            if isinstance(value, float):
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure('eval_spec', v, epoch)
        if (epoch % config['save_freq']) == 0:
            trainer.save_checkpoint(osp.join(exp_dir, f'epoch_{epoch}.pth'))




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--config_path', type = str, default = 'Config/config.yml')
    args = parser.parse_args()
    set_seed(1234)
    main(args.config_path)
