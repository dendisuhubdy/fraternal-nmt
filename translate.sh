#!/bin/bash
python translate.py -gpu 0 -model /data/milatmp1/suhubdyd/models/seeds/${1} -src /data/milatmp1/suhubdyd/datasets/multi30k/test.en.atok -tgt /data/milatmp1/suhubdyd/datasets/multi30k/test.de.atok -replace_unk -verbose -output ./translations/${2}.test.pred.atok
perl tools/multi-bleu.perl /data/lisatmp3/suhubdyd/datasets/multi30k/test.de.atok < ./translations/${2}.test.pred.atok
