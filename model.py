
import importlib
from generator import *
from discriminator import *
from munch import Munch
import copy



def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model, flush=True)
    print(name,flush=True)
    print("The number of parameters: {}".format(num_params), flush=True)


def build_model(config):
    
    
    generator = eval(config['generator']['model_name'])(config['generator'])
    discriminator = eval(config['discriminator']['model_name'])(config['discriminator'])
    generator_ema = copy.deepcopy(generator)
    nets = Munch(generator=generator,
                 discriminator=discriminator,
                 )
    
    nets_ema = Munch(generator=generator_ema,
                     )

    for net_name in nets.keys():
        print_network(nets[net_name], net_name)
    return nets, nets_ema
    

