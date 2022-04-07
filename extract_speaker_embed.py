from speaker_encoder.voice_encoder import SpeakerEncoder
import sys
import json
import os
import numpy as np
from speaker_encoder.audio import preprocess_wav
data_dir=sys.argv[1]
dump_dir=sys.argv[2]
ckpt_path=sys.argv[3]
speaker_path=sys.argv[4]
os.makedirs(dump_dir,exist_ok = True)
# create encoder
encoder = SpeakerEncoder(ckpt_path, 'cpu')
def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model, flush=True)
    print(name,flush=True)
    print("The number of parameters: {}".format(num_params), flush=True)
print_network(encoder, 'encoder')
# load speakers
speakers = json.load(open(speaker_path))

# loop through speakers
import glob
for spk in speakers:
    audio_files = list(glob.glob(os.path.join(data_dir,f'{spk}/E1*.wav'))) + list(glob.glob(os.path.join(data_dir,f'{spk}/E2*.wav')))
    audios = [preprocess_wav(audio,24000) for audio in audio_files]
    spk_emb = encoder.embed_speaker(audios)
    print(f'spk_emb {spk} {spk_emb.shape}')
    np.save(os.path.join(dump_dir,f'{spk}.npy'), spk_emb)





