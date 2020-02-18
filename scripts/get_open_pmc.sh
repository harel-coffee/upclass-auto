#!/bin/bash

dest_dir="/data/user/teodoro/uniprot/dataset/no_large/train/xml"
id_file="/data/user/teodoro/uniprot/annotation/train_files"
pmc_down=()
# iterate through array using a counter
for i in `ls $dest_dir | grep PMC | grep xml`; do
    #do something to each element of array
    fid=`echo $i | sed 's/.*PMC//' | sed 's/.xml//'`
    pmc_down[$fid]=1
done

for i in `cat $id_file | grep PMC`; do
    pmc=`echo $i | sed 's/PMC//'`
    if [ ${pmc_down[$pmc]+isset} ]; then
        echo "skipping "$pmc
    else 
        efetch -db pmc -id $pmc -format xml > $dest_dir/PMC$pmc.xml
    fi
done
