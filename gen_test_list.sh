#!/bin/bash

src_wavs=$(ls vcc20/SE*/E3*.wav)

trg_wavs=$(ls vcc20/TE*/E30001.wav)

touch test_meta.txt
for src_wav in $src_wavs ; do
    for trg_wav in $trg_wavs ; do
        echo "$src_wav $trg_wav" >> test_meta.txt
    done
done        
