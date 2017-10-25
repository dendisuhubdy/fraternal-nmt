#!/bin/bash
source activate lisa
python train.py -data /data/lisatmp3/suhubdyd/multi30k.atok.low -save_model /data/lisatmp3/suhubdyd/models/${1}_model -gpuid 0 -kappa ${2} -dropout ${3} -seed ${4}
