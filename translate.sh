#!/bin/bash
python translate.py -gpu 0 -model /data/lisatmp3/suhubdyd/models/${1} -src /data/lisatmp3/suhubdyd/multi30k/test.en.atok -tgt /data/lisatmp3/suhubdyd/multi30k/test.de.atok -replace_unk -verbose -output ./translations/${2}.test.pred.atok

perl tools/multi-bleu.perl /data/lisatmp3/suhubdyd/multi30k/test.de.atok < ./translations/${2}.test.pred.atok
