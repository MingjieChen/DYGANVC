import sys
sys.path.insert(0,'/share/mini1/res/t/vc/studio/timap-en/vcc2020/model')
import torch
from new_dygan_vc.model import build_model
import librosa
from speaker_encoder.voice_encoder import SpeakerEncoder
from speaker_encoder.audio import preprocess_wav
import fairseq
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm
test_list = '/share/mini1/res/t/vc/studio/timap-en/vcc2020/test_meta.txt'
model_ckpt = '/share/mini1/res/t/vc/studio/timap-en/vcc2020/exp/new_dygan_vc/0223_mel_G0_vqmel_spkemb_3/epoch_100.pth'
model_config = '/share/mini1/res/t/vc/studio/timap-en/vcc2020/exp/new_dygan_vc/0223_mel_G0_vqmel_spkemb_3/mel_G0_vqmel_spkemb_ls.yml'
speaker_encoder_ckpt = 'speaker_encoder/ckpt/pretrained_bak_5805000.pt'
vqwav2vec_ckpt = '/share/mini1/res/t/vc/studio/timap-en/vcc2020/fairseq/examples/wav2vec/ckpt/vq-wav2vec_kmeans.pt'

# test_list

with open(test_list) as f:
    test_meta = f.readlines()
    f.close()
print('load meta')
# build dygan model
import yaml
with open(model_config) as config_f:
    model_config = yaml.safe_load(config_f)
    config_f.close()
_, model = build_model(model_config['model'])

params = torch.load(model_ckpt, map_location=torch.device('cpu'))    
params = params['model_ema']

model.generator.load_state_dict(params['generator'])
_ = [ model[module].eval() for module in model]
print('load dygan')
# speaker encoder model
encoder = SpeakerEncoder(speaker_encoder_ckpt, 'cpu')
print('load speaker encoder')
# vqwav2vec model
vqwav2vec_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([vqwav2vec_ckpt])
vqwav2vec_model = vqwav2vec_model[0]
vqwav2vec_model.eval()
print('load vqwav2vec')

# vocoder
from parallel_wavegan.utils import load_model

vocoder_path = '/share/mini1/res/t/vc/studio/tiresyn-en/vcc2020/ParallelWaveGAN/egs/vcc2020/voc1/exp/train_task1_parallel_wavegan.task1/checkpoint-400000steps.pkl'
vocoder = load_model(vocoder_path)
vocoder.remove_weight_norm()
vocoder.eval()
print('load vocoder')

import time
start = time.time()
for meta in tqdm(test_meta):
    src_wav_path, trg_wav_path = meta.split(' ')

    # extract vqwav2vec features
    
    sr, src_wav = wavfile.read(src_wav_path.strip()) 

    signed_int16_max = 2**15
    if src_wav.dtype == np.int16:
        src_wav = src_wav.astype(np.float32) / signed_int16_max
    if sr != 16000:
        src_wav = librosa.resample(src_wav, sr, 16000)
    src_wav = np.clip(src_wav, -1.0, 1.0)
    src_wav_input_16khz = torch.FloatTensor(src_wav).unsqueeze(0)
    z = vqwav2vec_model.feature_extractor(src_wav_input_16khz)
    vqwav2vec_feature, _ = vqwav2vec_model.vector_quantizer.forward_idx(z)
    vqwav2vec_feature = vqwav2vec_feature.unsqueeze(0)
    
    # extract speaker embs
    trg_wav = preprocess_wav(trg_wav_path.strip(), 24000)
    spk_emb = encoder.embed_utterance(trg_wav)
    trg_spk_emb = torch.FloatTensor([spk_emb])


    # inference dygan
    converted_feat = model.generator(None, trg_spk_emb, vqwav2vec_feature)
    
    cvt_wav = vocoder.inference(converted_feat.transpose(-1,-2).squeeze())

print(f'total time {time.time() - start}')





