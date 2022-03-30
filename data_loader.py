from torch.utils import data
from sklearn.preprocessing import StandardScaler
import torch
import glob
import os
from os.path import join, basename, dirname, split, exists
import numpy as np
import h5py
import torchaudio
from utils import read_hdf5, to_categorical, load_wav, logmelspectrogram
import json
import librosa
#import joblib

class MelDataset(data.Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.min_length = config['min_length']
        # get speakers
        self.speakers = json.load(open(config['speakers']))
        self.spk2files = {}
        self.wav_files = []
        # load scaler
        #self.scaler = joblib.load(config['mvn_joblib'])
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.load(config['stats'])[0]
        self.scaler.scale_ = np.load(config['stats'])[1]
        self.scaler.n_features_in_ = self.scaler.mean_.shape[0]
        
        
        #parse data_dir
        data_dir = config['data_dir']
        for spk in self.speakers:
            if spk not in self.spk2files:
                self.spk2files[spk] = []    
            if not exists(f'{data_dir}/{spk}'):
                print(f"{data_dir}/{spk}")
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
        

                
    def __len__(self):
        return len(self.wav_files)

    def sample_seg(self, mel):
        # zero padding
        if mel.shape[0] < self.min_length:
            mel = np.pad(mel, [[0,self.min_length - mel.shape[0]],[0,0]])
        s = np.random.randint(0, mel.shape[0] - self.min_length + 1)
        
        return mel[s:s + self.min_length, :]
    
    def wav_to_mel(self, x):
        # load wav
        wav = load_wav(x)
        # trim
        #wav,_ = librosa.effects.trim(wav,top_db=60,frame_length=2048,hop_length=512)
        # low_cut_filter
        # skip low_cut_filter
        # mel
        mel = logmelspectrogram(wav).T.astype(np.float32)
        
        # normalization
        #mean = -4.0
        #std = 4.0
        #mel_norm = (mel - mean) / std
        mel_norm = self.scaler.transform(mel)
        return mel_norm
            
    def __getitem__(self, index):
        '''
            output: src_mel, src_cat, src_1hot, trg_mel, trg_cat, trg_1hot, src_mel_ref
        '''
        # sample one source wav
        src_spk, src_wav_path = self.wav_files[index]
        basename = os.path.basename(src_wav_path).split('.')[0]
        src_spk_idx = self.spk2idx[src_spk]
        src_spk_cat = np.squeeze(to_categorical([src_spk_idx], num_classes=len(self.speakers)))
        src_mel = self.wav_to_mel(src_wav_path)

        # segmentation to fixed length
        src_mel = self.sample_seg(src_mel)
        
        # convert to tensor
        src_mel_tensor = torch.FloatTensor(src_mel.T).unsqueeze(0)
        src_cat_tensor = torch.LongTensor([src_spk_idx]).squeeze_()
        src_1hot_tensor = torch.FloatTensor(src_spk_cat)

        return src_mel_tensor, src_cat_tensor, src_1hot_tensor
class MelSrcTrgDataset(data.Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.min_length = config['min_length']
        # get speakers
        self.speakers = json.load(open(config['speakers']))
        self.spk2files = {}
        self.wav_files = []
        
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.load(config['stats'])[0]
        self.scaler.scale_ = np.load(config['stats'])[1]
        self.scaler.n_features_in_ = self.scaler.mean_.shape[0]
        #parse data_dir
        data_dir = config['data_dir']
        for spk in self.speakers:
            if spk not in self.spk2files:
                self.spk2files[spk] = []    
            if not exists(f'{data_dir}/{spk}'):
                print(f"{data_dir}/{spk}")
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
        

                
    def __len__(self):
        return len(self.wav_files)

    def sample_seg(self, mel):
        # zero padding
        if mel.shape[0] < self.min_length:
            mel = np.pad(mel, [[0,self.min_length - mel.shape[0]],[0,0]])
        s = np.random.randint(0, mel.shape[0] - self.min_length + 1)
        
        return mel[s:s + self.min_length, :]
    
    def wav_to_mel(self, x):
        # load wav
        wav = load_wav(x)
        # trim
        wav,_ = librosa.effects.trim(wav,top_db=60,frame_length=2048,hop_length=512)
        # low_cut_filter
        # skip low_cut_filter
        # mel
        mel = logmelspectrogram(wav).T.astype(np.float32)
        
        # normalization
        #mean = -4.0
        #std = 4.0
        #mel_norm = (mel - mean) / std
        mel_norm = self.scaler.transform(mel)
        return mel_norm
            
    def __getitem__(self, index):
        '''
            output: src_mel, src_cat, src_1hot, trg_mel, trg_cat, trg_1hot, src_mel_ref
        '''
        # sample one source wav
        src_spk, src_wav_path = self.wav_files[index]
        basename = os.path.basename(src_wav_path).split('.')[0]
        src_spk_idx = self.spk2idx[src_spk]
        src_spk_cat = np.squeeze(to_categorical([src_spk_idx], num_classes=len(self.speakers)))
        # sample another source wav
        src_ref_wav_idx = np.random.randint(0, len(self.spk2files[src_spk]))
        src_ref_wav_path = self.spk2files[src_spk][src_ref_wav_idx]
        
        # sample a target speaker
        speakers = self.speakers[:]
        # remove source speaker
        speakers.remove(src_spk)
        sampled_trg_spk_idx = np.random.randint(0, len(speakers))
        trg_spk = speakers[sampled_trg_spk_idx] 
        trg_spk_idx = self.spk2idx[trg_spk]
        trg_spk_cat = np.squeeze(to_categorical([trg_spk_idx], num_classes=len(self.speakers)))
        # sample a target speaker wav
        trg_wav_path_idx = np.random.randint(0, len(self.spk2files[trg_spk]))
        trg_wav_path = self.spk2files[trg_spk][trg_wav_path_idx]

        # extract mels
        src_mel = self.wav_to_mel(src_wav_path)
        trg_mel = self.wav_to_mel(trg_wav_path)
        src_ref_mel = self.wav_to_mel(src_ref_wav_path)

        # segmentation to fixed length
        src_mel = self.sample_seg(src_mel)
        trg_mel = self.sample_seg(trg_mel)
        src_ref_mel = self.sample_seg(src_ref_mel)
        
        # convert to tensor
        src_mel_tensor = torch.FloatTensor(src_mel.T).unsqueeze(0)
        src_cat_tensor = torch.LongTensor([src_spk_idx]).squeeze_()
        src_1hot_tensor = torch.FloatTensor(src_spk_cat)

        trg_mel_tensor = torch.FloatTensor(trg_mel.T).unsqueeze(0)
        trg_cat_tensor = torch.LongTensor([trg_spk_idx]).squeeze_()
        trg_1hot_tensor = torch.FloatTensor(trg_spk_cat)

        src_ref_mel_tensor = torch.FloatTensor(src_ref_mel.T).unsqueeze(0)
        return src_mel_tensor, src_cat_tensor, src_1hot_tensor, trg_mel_tensor, trg_cat_tensor, trg_1hot_tensor, src_ref_mel_tensor
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
        self.vq_type = config['vq_type']

                
    def __len__(self):
        return len(self.wav_files)
    
    def wav_to_mel(self, x):
        # load wav
        wav = load_wav(x)
        # trim
        #wav,_ = librosa.effects.trim(wav,top_db=60,frame_length=2048,hop_length=512)
        # low_cut_filter
        # skip low_cut_filter
        # mel
        mel = logmelspectrogram(wav).T.astype(np.float32)
        
        # normalization
        #mean = -4.0
        #std = 4.0
        #mel_norm = (mel - mean) / std
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
        if self.vq_type == 'dense':
            vqw2v_path=os.path.join(self.vq_dir,src_spk,f'{basename}_dense.npy')
            vqw2v_dense = np.load(vqw2v_path).T    
        elif self.vq_type == 'idx':
            vqw2v_path=os.path.join(self.vq_dir,src_spk,f'{basename}_idxs.npy')
            vqw2v_dense = np.load(vqw2v_path)
        else:
            raise Exception    
                
        if not os.path.exists(vqw2v_path):
            raise Exception
        
        # solve mismatch between vq feats and mels
        # https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/phone_linear/expert.py
        mel_length = src_mel.shape[0]
        vq_length = vqw2v_dense.shape[0]
        if mel_length > vq_length:
            pad_vec = vqw2v_dense[-1,:]
            vqw2v_dense = np.concatenate((vqw2v_dense, np.repeat(pad_vec, mel_length - vq_length, 0)),1)
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
class VQMelSrcTrgDataset(data.Dataset):
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
        self.vq_type = config['vq_type']

                
    def __len__(self):
        return len(self.wav_files)
    
    def wav_to_mel(self, x):
        # load wav
        wav = load_wav(x)
        # trim
        #wav,_ = librosa.effects.trim(wav,top_db=60,frame_length=2048,hop_length=512)
        # low_cut_filter
        # skip low_cut_filter
        # mel
        mel = logmelspectrogram(wav).T.astype(np.float32)
        
        # normalization
        #mean = -4.0
        #std = 4.0
        #mel_norm = (mel - mean) / std
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
        # sample another source wav
        src_ref_wav_idx = np.random.randint(0, len(self.spk2files[src_spk]))
        src_ref_wav_path = self.spk2files[src_spk][src_ref_wav_idx]
        # sample a target speaker
        speakers = self.speakers[:]
        # remove source speaker
        #speakers.remove(src_spk)
        sampled_trg_spk_idx = np.random.randint(0, len(speakers))
        trg_spk = speakers[sampled_trg_spk_idx] 
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
        if self.vq_type == 'dense':
            vqw2v_path=os.path.join(self.vq_dir,src_spk,f'{basename}_dense.npy')
            vqw2v_dense = np.load(vqw2v_path).T    
        elif self.vq_type == 'idx':
            vqw2v_path=os.path.join(self.vq_dir,src_spk,f'{basename}_idxs.npy')
            vqw2v_dense = np.load(vqw2v_path)
        else:
            raise Exception    
                
        if not os.path.exists(vqw2v_path):
            raise Exception
        
        # solve mismatch between vq feats and mels
        # https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/phone_linear/expert.py
        mel_length = src_mel.shape[0]
        vq_length = vqw2v_dense.shape[0]
        if mel_length > vq_length:
            pad_vec = vqw2v_dense[-1,:]
            vqw2v_dense = np.concatenate((vqw2v_dense, np.repeat(pad_vec, mel_length - vq_length, 0)),1)
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

        trg_mel_tensor = torch.FloatTensor(trg_mel.T).unsqueeze(0)
        trg_cat_tensor = torch.LongTensor([trg_spk_idx]).squeeze_()
        trg_1hot_tensor = torch.FloatTensor(trg_spk_cat)

        src_ref_mel_tensor = torch.FloatTensor(src_ref_mel.T).unsqueeze(0)
        vq_tensor = torch.FloatTensor(vqw2v_dense.T).unsqueeze(0)
        
        return src_mel_tensor, src_cat_tensor, src_1hot_tensor, trg_mel_tensor, trg_cat_tensor, trg_1hot_tensor, src_ref_mel_tensor, vq_tensor
class VQMelDataset(data.Dataset):
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
        self.vq_type = config['vq_type']

                
    def __len__(self):
        return len(self.wav_files)

    def sample_seg(self, mel, vq):
        # zero padding
        if mel.shape[0] < self.min_length:
            mel = np.pad(mel, [[0,self.min_length - mel.shape[0]],[0,0]])
            vq = np.pad(vq, [[0, self.min_length - vq.shape[0]],[0,0]])
        s = np.random.randint(0, mel.shape[0] - self.min_length + 1)
        
        return mel[s:s + self.min_length, :], vq[s:s+self.min_length, :]
    def __getitem__(self, index):
        src_spk, src_wav_path = self.wav_files[index]
        basename = os.path.basename(src_wav_path).split('.')[0]
        src_spk_idx = self.spk2idx[src_spk]
        src_spk_cat = np.squeeze(to_categorical([src_spk_idx], num_classes=len(self.speakers)))
        # load wav
        wav = load_wav(src_wav_path)
        # trim
        wav,_ = librosa.effects.trim(wav,top_db=60,frame_length=2048,hop_length=512)
        # low_cut_filter
        # skip low_cut_filter
        # mel
        mel = logmelspectrogram(wav).T
        
        # normalization
        mel_norm = self.scaler.transform(mel)
        # load vqw2v feat
        if self.vq_type == 'dense':
            vqw2v_path=os.path.join(self.vq_dir,src_spk,f'{basename}_dense.npy')
            vqw2v_dense = np.load(vqw2v_path).T    
        elif self.vq_type == 'idx':
            vqw2v_path=os.path.join(self.vq_dir,src_spk,f'{basename}_idxs.npy')
            vqw2v_dense = np.load(vqw2v_path)
        else:
            raise Exception    
                
        if not os.path.exists(vqw2v_path):
            raise Exception
        
        # solve mismatch between vq feats and mels
        # https://github.com/s3prl/s3prl/blob/master/s3prl/downstream/phone_linear/expert.py
        mel_length = mel_norm.shape[0]
        vq_length = vqw2v_dense.shape[0]
        if mel_length > vq_length:
            pad_vec = vqw2v_dense[-1,:]
            vqw2v_dense = np.concatenate((vqw2v_dense, np.repeat(pad_vec, mel_length - vq_length, 0)),1)
        elif mel_length < vq_length:
            vqw2v_dense = vqw2v_dense[:mel_length,:]    
        assert mel_norm.shape[0] == vqw2v_dense.shape[0], f"mel {mel.shape} vq {vqw2v_dense.shape}"
        mel_norm, vqw2v_dense = self.sample_seg(mel_norm, vqw2v_dense)
        
        return torch.FloatTensor(mel_norm.T).unsqueeze(0), torch.FloatTensor(vqw2v_dense.T).unsqueeze(0), torch.LongTensor([src_spk_idx]).squeeze_(), torch.FloatTensor(src_spk_cat) 

class HDF5Dataset(data.Dataset):

    def __init__(self, config, split):
        super().__init__()
        self.speakers = json.load(open(config['speakers']))
        self.min_length = config['min_length']
        data_dir = config['data_dir']
        self.mel_files = []
        self.spk2files = {}
        for spk in self.speakers:
            if spk not in self.spk2files:
                self.spk2files[spk] = []
            if exists(f"{data_dir}/{spk}_raw/{spk}_{split}"):
                files = glob.glob(f"{data_dir}/{spk}_raw/{spk}_{split}/*.h5")
                print(spk)
                print(len(files))
                self.spk2files[spk].extend(files)
                for f in files:
                    self.mel_files.append((spk, f))
        
        self.spk2idx = { spk : ind for ind, spk in enumerate(self.speakers)}

        print(f"loading files {len(self.mel_files)}")

        self.num_files = len(self.mel_files)

    def sample_seg(self, feat):
        # zero padding
        if feat.shape[0] < self.min_length:
            feat = np.pad(feat, [[0,self.min_length - feat.shape[0]],[0,0]])
        s = np.random.randint(0, feat.shape[0] - self.min_length + 1)
        
        return feat[s:s + self.min_length, :]
    
    
    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        src_spk, src_filename = self.mel_files[index]
        src_spk_idx = self.spk2idx[src_spk]
        src_spk_cat = np.squeeze(to_categorical([src_spk_idx], num_classes=len(self.speakers)))


        src_mel = read_hdf5(src_filename)
        src_mel = self.sample_seg(src_mel)   
        src_mel = np.transpose(src_mel, (1, 0))  
        # to one-hot
        
        # mel, index, one-hot
        return torch.FloatTensor(src_mel).unsqueeze(0), torch.LongTensor([src_spk_idx]).squeeze_(), torch.FloatTensor(src_spk_cat)


class HDF5SrcTrgDataset(data.Dataset):

    def __init__(self, config, split):
        super().__init__()
        self.speakers = json.load(open(config['speakers']))
        self.min_length = config['min_length']
        data_dir = config['data_dir']
        self.mel_files = []
        self.spk2files = {}
        for spk in self.speakers:
            if spk not in self.spk2files:
                self.spk2files[spk] = []
            if exists(f"{data_dir}/{spk}_raw/{spk}_{split}"):
                files = glob.glob(f"{data_dir}/{spk}_raw/{spk}_{split}/*.h5")
                print(spk)
                print(len(files))
                self.spk2files[spk].extend(files)
                for f in files:
                    self.mel_files.append((spk, f))
        
        self.spk2idx = { spk : ind for ind, spk in enumerate(self.speakers)}
        print(self.spk2idx)
        print(f"loading files {len(self.mel_files)}")

        self.num_files = len(self.mel_files)

    def sample_seg(self, feat):
        # zero padding
        if feat.shape[0] < self.min_length:
            feat = np.pad(feat, [[0,self.min_length - feat.shape[0]],[0,0]])
        s = np.random.randint(0, feat.shape[0] - self.min_length + 1)
        
        return feat[s:s + self.min_length, :]
    
    
    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        src_spk, src_filename = self.mel_files[index]
        src_spk_idx = self.spk2idx[src_spk]
        src_mel = read_hdf5(src_filename)
        src_mel = self.sample_seg(src_mel)   
        src_mel = np.transpose(src_mel, (1, 0))  
        src_spk_cat = np.squeeze(to_categorical([src_spk_idx], num_classes=len(self.speakers)))
        
        # sample another source wav
        src_ref_mel_idx = np.random.randint(0, len(self.spk2files[src_spk]))
        src_ref_mel_path = self.spk2files[src_spk][src_ref_mel_idx]
        src_ref_mel = read_hdf5(src_ref_mel_path)
        src_ref_mel = self.sample_seg(src_ref_mel)
        src_ref_mel = np.transpose(src_ref_mel, (1,0))

        # sample a target speaker
        speakers = self.speakers[:]
        # remove source speaker
        #speakers.remove(src_spk)
        sampled_trg_spk_idx = np.random.randint(0, len(speakers))
        trg_spk = speakers[sampled_trg_spk_idx] 
        trg_spk_idx = self.spk2idx[trg_spk]
        trg_spk_cat = np.squeeze(to_categorical([trg_spk_idx], num_classes=len(self.speakers)))
        # sample a target speaker wav
        trg_mel_path_idx = np.random.randint(0, len(self.spk2files[trg_spk]))
        trg_mel_path = self.spk2files[trg_spk][trg_mel_path_idx]
        trg_mel = read_hdf5(trg_mel_path)
        trg_mel = self.sample_seg(trg_mel)
        trg_mel = np.transpose(trg_mel,(1,0))

        src_mel_tensor = torch.FloatTensor(src_mel).unsqueeze(0)
        src_cat_tensor = torch.LongTensor([src_spk_idx]).squeeze_()
        src_1hot_tensor = torch.FloatTensor(src_spk_cat)

        trg_mel_tensor = torch.FloatTensor(trg_mel).unsqueeze(0)
        trg_cat_tensor = torch.LongTensor([trg_spk_idx]).squeeze_()
        trg_1hot_tensor = torch.FloatTensor(trg_spk_cat)

        src_ref_mel_tensor = torch.FloatTensor(src_ref_mel).unsqueeze(0)


        
        return src_mel_tensor, src_cat_tensor, src_1hot_tensor, trg_mel_tensor, trg_cat_tensor, trg_1hot_tensor, src_ref_mel_tensor


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

