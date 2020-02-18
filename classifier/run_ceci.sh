#!/bin/sh

for i in `ls /data/user/teodoro/uniprot/dataset/NAR/test/tag/`; 
  do mkdir /data/user/teodoro/uniprot/results/NAR/$i; 
  python3 -u main_classifier.py -c cnn -s /data/user/teodoro/uniprot/dataset/NAR/test/tag/$i -o /data/user/teodoro/uniprot/results/NAR/$i -m /data/user/teodoro/uniprot/results/no_large/tag/cnn_1e-05.pkl ; 
done
