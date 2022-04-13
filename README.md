# Efficient Non-Autoregressive GAN Voice Conversion using VQWav2vec Features and Dynamic Convolution

### Mingjie Chen, Yanghao Zhou, Heyan Huang, Thomas Hain

***

> It was shown recently that a combination of ASR and TTS models  yield highly 
competitive performance on standard voice conversion tasks such as the Voice
Conversion Challenge 2020 (VCC2020). To obtain good performance
both models require pretraining on large amounts of data, thereby obtaining
large models that are potentially inefficient in use. In this work we present a model that
is significantly smaller and thereby faster in processing while obtaining equivalent performance. 
To achieve this the proposed model, Dynamic-GAN-VC (DYGAN-VC), uses a non-autoregressive structure
and makes use of vector quantised embeddings obtained from a VQWav2vec model. Furthermore 
dynamic convolution is introduced to improve speech content modeling while requiring a small
number of parameters. Objective and subjective evaluation was performed using the VCC2020 task, 
yielding MOS scores of up to 3.86, and character error rates as low as 4.3\%. This was achieved with approximately half the number of model parameters, and up to 8 times faster decoding speed. 

[[paper](https://arxiv.org/abs/2203.17172)] [[demo](https://mingjiechen.github.io/dygan-vc)]

## Dataset
[VCC2020 track1](https://github.com/nii-yamagishilab/VCC2020-database)

## Dependencies
1. [fairseq](https://github.com/pytorch/fairseq)
2. [Parallel WaveGAN vocoder](https://github.com/kan-bayashi/ParallelWaveGAN)

## How to run
1. clone repository
```bash
git clone https://github.com/MingjieChen/DYGANVC.git
cd DYGANVC
```

2. Create conda env and install pytorch 1.7 through conda
```bash
conda create --name torch_1.7 python==3.7
conda activate torch_1.7
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install librosa -c conda-forge
```
Choose your own cudatoolkit version according to your own GPU.

3. install packages
```bash
pip install fairseq parallel_wavegan munch pyyaml SoundFile tqdm scikit-learn tensorboardX webrtcvad
```

4. download dataset and unzip dataset to vcc20/
```bash
./data_download.sh vcc20
```

5. extract speaker embeddings
```bash
./extract_speaker_embed.sh
```

6. download vqwav2vec ckpt
```bash
mkdir vqw2v
cd vqw2v
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt
```

7. extract vqwav2vec features
```bash
./vqw2v_feat_extract.sh
```

8. start training
```bash
./run_train.sh
```

9. inference
```bash
python inference.py
```
