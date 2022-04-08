# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from munch import Munch
import torch.nn.functional as F

    

# for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def compute_g_loss(nets, args, batch):
    args = Munch(args)
    
    x_real, y_src, y_src_emb, x_trg, y_trg, y_trg_emb, x_src_ref, x_vq = batch
    
    
    # adversarial loss
    x_fake = nets.generator(y_trg_emb, x_vq)
    out = nets.discriminator(x_fake, y_trg) 
    loss_adv = torch.mean( (1.0 - out)**2)
    
    # identity mapping loss
    x_id = nets.generator(y_src_emb, x_vq)
    loss_id = torch.mean(torch.abs(x_id - x_real))
    
    
    
    loss = args.lambda_adv * loss_adv  \
           + args.lambda_id * loss_id \

    return loss, Munch(adv=loss_adv.item(),
                       id=loss_id.item(),
                       )
def compute_d_loss(nets, args, batch):
    args = Munch(args)
    
    x_real, y_src, y_src_emb, x_trg, y_trg, y_trg_emb, x_src_ref, x_vq = batch

    # with real audios
    x_real.requires_grad_()
    real_out = nets.discriminator(x_real, y_src)
    loss_real = torch.mean((1.0 - real_out)**2)
    
    if nets.discriminator.training:
        loss_reg = r1_reg(real_out, x_real)
    else:
        loss_reg = torch.FloatTensor([0]).to(x_real.device)
    
    loss_con_reg = torch.FloatTensor([0]).to(x_real.device)
    
    # with fake audios
    with torch.no_grad():
        x_fake = nets.generator(y_trg_emb, x_vq)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = torch.mean(out**2)
    
        

    loss = loss_real + loss_fake  + args.lambda_reg * loss_reg

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item(),
                       )


class VQMelSpkEmbLSTrainer(object):
    def __init__(self,
                 args,
                 model=None,
                 model_ema=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 train_dataloader=None,
                 dev_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 fp16_run=False
    ):
        self.args = args
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.fp16_run = fp16_run


    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
            "model": {key: self.model[key].state_dict() for key in self.model}
        }
        if self.model_ema is not None:
            state_dict['model_ema'] = {key: self.model_ema[key].state_dict() for key in self.model_ema}

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for key in self.model:
            self._load(state_dict["model"][key], self.model[key])

        if self.model_ema is not None:
            for key in self.model_ema:
                self._load(state_dict["model_ema"][key], self.model_ema[key])
        
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])


    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    @staticmethod
    def moving_average(model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def _train_epoch(self):
        self.epochs += 1
        use_con_reg = (self.epochs >= self.args.con_reg_epoch)
        
        train_losses = defaultdict(list)
        _ = [self.model[k].train() for k in self.model]
        scaler = torch.cuda.amp.GradScaler() if (('cuda' in str(self.device)) and self.fp16_run) else None

        
        for train_steps_per_epoch, batch in enumerate(self.train_dataloader, 1):

            ### load data
            batch = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    d_loss, d_losses = compute_d_loss(self.model, self.args.d_loss, batch)
                scaler.scale(d_loss).backward()
            else:
                d_loss, d_losses = compute_d_loss(self.model, self.args.d_loss, batch)
                d_loss.backward()
            self.optimizer.step('discriminator', scaler=scaler)
            


            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    g_loss, g_losses = compute_g_loss(
                        self.model, self.args.g_loss, batch)
                scaler.scale(g_loss).backward()
            else:
                g_loss, g_losses = compute_g_loss(
                    self.model, self.args.g_loss, batch)
                g_loss.backward()

            self.optimizer.step('generator', scaler=scaler)

            # compute moving average of network parameters
            self.moving_average(self.model.generator, self.model_ema.generator, beta=0.999)
            
            d_loss_string = f" epoch {self.epochs}, iters {train_steps_per_epoch}"
            g_loss_string = f" epoch {self.epochs}, iters {train_steps_per_epoch}"
            for key in d_losses:
                train_losses["train/%s" % key].append(d_losses[key])
                d_loss_string += f" {key}:{d_losses[key]:.5f} "
            print(d_loss_string)    
            for key in g_losses:
                train_losses["train/%s" % key].append(g_losses[key])
                g_loss_string += f" {key}:{g_losses[key]:.5f} "
            print(g_loss_string)    
            print()
            self.steps += 1


        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        
        eval_losses = defaultdict(list)
        _ = [self.model[k].eval() for k in self.model]
        for eval_steps_per_epoch, batch in enumerate(self.dev_dataloader, 1):

            ### load data
            batch = [b.to(self.device) for b in batch]

            #  discriminator
            d_loss, d_losses = compute_d_loss(
                self.model, self.args.d_loss, batch)

            # train the generator
            g_loss, g_losses = compute_g_loss(
                self.model, self.args.g_loss, batch)

            for key in d_losses:
                eval_losses["eval/%s" % key].append(d_losses[key])
            for key in g_losses:
                eval_losses["eval/%s" % key].append(g_losses[key])

            

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_string = f"epoch {self.epochs}, eval: "
        for key, value in eval_losses.items():
            eval_string += f"{key}: {value:.6f} "
        print(eval_string)    
        print()
        return eval_losses


        
