#!/bin/bash

dest_dir="/data/user/teodoro/uniprot/dataset/no_large/train/xml"
id_file="/data/user/teodoro/uniprot/annotation/train_files"
pmid_down=()
# iterate through array using a counter
for i in `ls $dest_dir | grep -v PMC | grep xml`; do
    #do something to each element of array
    fid=`echo $i | sed 's/.xml//'`
    pmid_down[$fid]=1
done

for i in `cat $id_file | grep -v PMC`; do
    pmid=$i
    if [ ${pmid_down[$pmid]+isset} ]; then
        echo "skipping "$pmid
    else 
        efetch -db pubmed -id $pmid -format xml > $dest_dir/$pmid.xml
    fi
done
