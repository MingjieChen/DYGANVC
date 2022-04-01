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

## Dataset
[VCC2020 track1](https://github.com/nii-yamagishilab/VCC2020-database)

## Dependencies
1. [fairseq](https://github.com/pytorch/fairseq)
2. [speaker embeddings](https://github.com/kan-bayashi/ParallelWaveGAN)

## How to run
1. clone repository
```bash
git clone https://github.com/MingjieChen/DYGANVC.git
cd DYGANVC
```

2. Create conda env and install pytorch 1.7 through conda
```bash
conda create --name env_name python==3.7
conda activate env_name
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
Choose your own configs for cudatoolkit version according to your own GPU.

3. install packages
```bash
pip install fairseq librosa parallel_wavegan munch pyyaml SoundFile tqdm scikit-learn tensorboardX
```

4. download dataset and unzip dataset to vcc20/



