
import importlib
#from .old_generator import Generator as OldGenerator
#from .old_discriminator import Discriminator as OldDiscriminator
#from .old_speaker_encoder import SPEncoder as OldSPEncoder
from .generator import *
#from .generator_v2 import GeneratorV2
from .discriminator import *
#from .mapping import MappingNetwork
#from .speaker_encoder import *
from munch import Munch
import copy
from .content_discriminator import ContentDiscriminator



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
    if 'mapping_network' in config:
        mapping_network = eval(config['mapping_network']['model_name'])(config['mapping_network'])
        mapping_network_ema = copy.deepcopy(mapping_network)
        nets['mapping_network'] = mapping_network
        nets_ema['mapping_network'] = mapping_network_ema
    if 'speaker_encoder' in config:    
        speaker_encoder = eval(config['speaker_encoder']['model_name'])(config['speaker_encoder'])
        speaker_encoder_ema = copy.deepcopy(speaker_encoder)
        nets['speaker_encoder'] = speaker_encoder
        nets_ema['speaker_encoder'] = speaker_encoder_ema
    
    if 'content_discriminator' in config:
        content_discriminator = ContentDiscriminator(config['content_discriminator'])
        nets['content_discriminator'] = content_discriminator    

    for net_name in nets.keys():
        print_network(nets[net_name], net_name)
    return nets, nets_ema
    

