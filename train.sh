#!/bin/bash
source activate lisa
#python train.py -data /data/lisatmp3/suhubdyd/multi30k.atok.low -save_model /data/lisatmp3/suhubdyd/models/${1}_model -gpuid 0 -kappa ${2} -dropout ${3} -seed ${4}
python train.py -data /data/milatmp1/suhubdyd/datasets/multi30k/multi30k.atok.low -save_model /data/milatmp1/suhubdyd/models/seeds/encoder${1}decoder${2}dropout${3}wdrop${4}seed${5} -gpuid 0 -kappa_enc ${1} -kappa_dec ${2} -dropout ${3} -weightdropout ${4} -seed ${5}
