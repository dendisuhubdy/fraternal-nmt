#!/bin/bash
source activate lisa
#python train.py -data /data/lisatmp3/suhubdyd/multi30k.atok.low -save_model /data/lisatmp3/suhubdyd/models/${1}_model -gpuid 0 -kappa ${2} -dropout ${3} -seed ${4}
python train.py -data /data/lisatmp3/suhubdyd/multi30k.atok.low -save_model /data/lisatmp3/suhubdyd/models/encoder${1}decoder${2}dropout${3}wdropFalse -gpuid 0 -kappa_enc ${1} -kappa_dec ${2} -dropout ${3} -weightdropout False
python train.py -data /data/lisatmp3/suhubdyd/multi30k.atok.low -save_model /data/lisatmp3/suhubdyd/models/encoder${1}decoder${2}dropout${3}wdropTrue -gpuid 0 -kappa_enc ${1} -kappa_dec ${2} -dropout ${3} -weightdropout True
