#!/bin/bash



echo "Downloading and building the LIDC/LIDC-IDRI dataset: Started."

rm -rf '/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms'
/opt/nbia-data-retriever/nbia-data-retriever -c '/nas-ctm01/datasets/public/MEDICAL/lidc-db/TCIA_LIDC-IDRI_20200921.tcia' -d '/nas-ctm01/datasets/public/MEDICAL/lidc-db/data' -v -f
cp -r '/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/TCIA_LIDC-IDRI_20200921/LIDC-IDRI' '/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms'
mv '/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/TCIA_LIDC-IDRI_20200921/metadata.csv' '/nas-ctm01/datasets/public/MEDICAL/lidc-db/data/dicoms/metadata.csv'
cd '/nas-ctm01/datasets/public/MEDICAL/lidc-db/data'
rm -rf *.log
rm -rf 'TCIA_LIDC-IDRI_20200921'

echo "Downloading and building the LIDC/LIDC-IDRI dataset: Finished."
