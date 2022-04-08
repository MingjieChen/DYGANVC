from torch.utils import data
from sklearn.preprocessing import StandardScaler
import torch
import glob
import os
from os.path import join, basename, dirname, split, exists
import numpy as np
import h5py
import torchaudio
from utils import to_categorical, load_wav, logmelspectrogram
import json
import librosa

class VQMelSpkEmbDataset(data.Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.min_length = config['min_length']
        # load scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.load(config['stats'])[0]
        self.scaler.scale_ = np.load(config['stats'])[1]
        self.scaler.n_features_in_ = self.scaler.mean_.shape[0]

        # get speakers
        self.speakers = json.load(open(config['speakers']))
        self.spk_emb_dir = config['spk_emb_dir']
        self.spk2files = {}
        self.wav_files = []
        #parse data_dir
        data_dir = config['data_dir']
        for spk in self.speakers:
            if spk not in self.spk2files:
                self.spk2files[spk] = []    
            if not exists(f'{data_dir}/{spk}'):
                raise Exception
            _spk_files = sorted(glob.glob(f'{data_dir}/{spk}/E1*.wav')) + sorted(glob.glob(f'{data_dir}/{spk}/E2*.wav'))       
            if split == 'train':
                _spk_files = _spk_files[:-10]
            elif split == 'dev':
                _spk_files = _spk_files[-10:]    
            else:
                raise Exception    
            print(spk)
            print(len(_spk_files))
            self.spk2files[spk].extend(_spk_files)
            for f in _spk_files:
                self.wav_files.append((spk, f))    
                
        self.spk2idx = { spk : ind for ind, spk in enumerate(self.speakers)}
        print(f"loading files {len(self.wav_files)}")
        
        self.vq_dir = config['vq_dir']

                
    def __len__(self):
        return len(self.wav_files)
    
    def wav_to_mel(self, x):
        # load wav
        wav = load_wav(x)
        mel = logmelspectrogram(wav).T.astype(np.float32)
        
        mel_norm = self.scaler.transform(mel)
        return mel_norm

    def sample_seg(self, mel, vq = None):
        # zero padding
        if mel.shape[0] < self.min_length:
            mel = np.pad(mel, [[0,self.min_length - mel.shape[0]],[0,0]])
            if vq is not None:
                vq = np.pad(vq, [[0, self.min_length - vq.shape[0]],[0,0]])
        s = np.random.randint(0, mel.shape[0] - self.min_length + 1)
        if vq is not None:
            return mel[s:s + self.min_length, :], vq[s:s+self.min_length, :]
        else:
            return mel[s:s+self.min_length, :]    
    def __getitem__(self, index):
        src_spk, src_wav_path = self.wav_files[index]
        basename = os.path.basename(src_wav_path).split('.')[0]
        src_spk_idx = self.spk2idx[src_spk]
        src_spk_cat = np.squeeze(to_categorical([src_spk_idx], num_classes=len(self.speakers)))
        src_spk_emb = np.load(os.path.join(self.spk_emb_dir,src_spk+'.npy'))
        # sample another source wav
        src_ref_wav_idx = np.random.randint(0, len(self.spk2files[src_spk]))
        src_ref_wav_path = self.spk2files[src_spk][src_ref_wav_idx]
        # sample a target speaker
        speakers = self.speakers[:]
        # remove source speaker
        #speakers.remove(src_spk)
        sampled_trg_spk_idx = np.random.randint(0, len(speakers))
        trg_spk = speakers[sampled_trg_spk_idx]
        trg_spk_emb = np.load(os.path.join(self.spk_emb_dir,trg_spk+'.npy')) 
        trg_spk_idx = self.spk2idx[trg_spk]
        trg_spk_cat = np.squeeze(to_categorical([trg_spk_idx], num_classes=len(self.speakers)))
        # sample a target speaker wav
        trg_wav_path_idx = np.random.randint(0, len(self.spk2files[trg_spk]))
        trg_wav_path = self.spk2files[trg_spk][trg_wav_path_idx]
        
        # extract mels
        src_mel = self.wav_to_mel(src_wav_path)
        trg_mel = self.wav_to_mel(trg_wav_path)
        src_ref_mel = self.wav_to_mel(src_ref_wav_path)
        
        # load vqw2v feat
        vqw2v_path=os.path.join(self.vq_dir,src_spk,f'{basename}_dense.npy')
        vqw2v_dense = np.load(vqw2v_path).T    
                
        if not os.path.exists(vqw2v_path):
            raise Exception
        
        # solve mismatch between vq feats and mels
        # https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/phone_linear/expert.py
        mel_length = src_mel.shape[0]
        vq_length = vqw2v_dense.shape[0]
        if mel_length > vq_length:
            pad_vec = vqw2v_dense[-1,:]
            repeated_pad_vec = np.tile(pad_vec, mel_length - vq_length).reshape(mel_length-vq_length,512)
            vqw2v_dense = np.concatenate((vqw2v_dense, repeated_pad_vec),0)
        elif mel_length < vq_length:
            vqw2v_dense = vqw2v_dense[:mel_length,:]    
        assert src_mel.shape[0] == vqw2v_dense.shape[0], f"mel {src_mel.shape} vq {vqw2v_dense.shape}"
        src_mel, vqw2v_dense = self.sample_seg(src_mel, vqw2v_dense)
        trg_mel = self.sample_seg(trg_mel)
        src_ref_mel = self.sample_seg(src_ref_mel)
        
        # convert to tensor
        src_mel_tensor = torch.FloatTensor(src_mel.T).unsqueeze(0)
        src_cat_tensor = torch.LongTensor([src_spk_idx]).squeeze_()
        src_1hot_tensor = torch.FloatTensor(src_spk_cat)
        src_emb_tensor = torch.FloatTensor(src_spk_emb)

        trg_mel_tensor = torch.FloatTensor(trg_mel.T).unsqueeze(0)
        trg_cat_tensor = torch.LongTensor([trg_spk_idx]).squeeze_()
        trg_1hot_tensor = torch.FloatTensor(trg_spk_cat)
        trg_emb_tensor = torch.FloatTensor(trg_spk_emb)

        src_ref_mel_tensor = torch.FloatTensor(src_ref_mel.T).unsqueeze(0)
        vq_tensor = torch.FloatTensor(vqw2v_dense.T).unsqueeze(0)
        
        return src_mel_tensor, src_cat_tensor, src_emb_tensor, trg_mel_tensor, trg_cat_tensor, trg_emb_tensor, src_ref_mel_tensor, vq_tensor


class TestDataset(object):

    def __init__(self, speakers, data_dir, split):
        self.speakers = json.load(open(speakers))
        self.data_dir = data_dir
        self.spk2idx = { spk : ind for ind, spk in enumerate(self.speakers)}
        self.spk2files = {}
        
        for spk in self.speakers:
            if spk not in self.spk2files:
                self.spk2files[spk] = []
            if exists(f"{data_dir}/{spk}_raw/{spk}_{split}"):
                files = glob.glob(f"{data_dir}/{spk}_raw/{spk}_{split}/*.h5")
                print(spk)
                print(len(files))
                self.spk2files[spk].extend(files)







def build_data_loader(config):
    train_dataset = eval(config['dataset'])(config, split = 'train')
    dev_dataset = eval(config['dataset'])(config, split = 'dev')
    train_data_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=config['shuffle'],
                                  num_workers=config['num_workers'],
                                  drop_last=config['drop_last'])
    dev_data_loader = data.DataLoader(dataset=dev_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=False,
                                  num_workers=config['num_workers'],
                                  drop_last=False)
    
    return train_data_loader, dev_data_loader

