import torch
import fairseq
import torchaudio
import sys
import numpy as np
import librosa
from scipy.io import wavfile
audio_dir=sys.argv[1]
out_dir=sys.argv[2]
ckpt=sys.argv[3]

import glob
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import subprocess
from tqdm import tqdm
audio_paths = list(glob.glob(os.path.join(audio_dir,'*/*.wav')))
def load_wav(path):
    sr, x = wavfile.read(path)
    signed_int16_max = 2**15
    if x.dtype == np.int16:
        x = x.astype(np.float32) / signed_int16_max
    print(f'24khz wav {x.shape}')
    if sr != 16000:
        x = librosa.resample(x, sr, 16000)
    print(f'resample {x.shape}')
    x = np.clip(x, -1.0, 1.0)

    return x
def process(cp, _audio_paths, out_dir):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp])
    model = model[0]
    model.eval()
    for audio_path in _audio_paths:
        wav = load_wav(audio_path)
        
        wav_input_16khz = wav
        
        wav_input_16khz = torch.FloatTensor(wav_input_16khz).unsqueeze(0)
        z = model.feature_extractor(wav_input_16khz)
        print(f"z {z.size()}")
        dense, idxs = model.vector_quantizer.forward_idx(z)

        dense = dense[0].data.numpy()
        idxs = idxs[0].data.numpy()
        print(f" dense {dense.shape} idxs {idxs.shape}")
        file_id = os.path.basename(audio_path).split('.')[0]
        spk=os.path.basename(os.path.dirname(audio_path))
        os.makedirs(os.path.join(out_dir,spk), exist_ok = True)
        np.save(os.path.join(out_dir, spk, file_id+'_dense'), dense)

process(ckpt, audio_paths, out_dir )

