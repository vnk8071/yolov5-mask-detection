#!/bin/bash
echo 'Downloading and setting up data'
DEST_DIR='data' 
FILENAME='mask-fpt-ai.zip'

mkdir $DEST_DIR
gdown https://drive.google.com/uc?id=1D1_lUIucMWqSDqZokohl8Aqd2G5K-LIz
mv $FILENAME $DEST_DIR
unzip "${DEST_DIR}/${FILENAME}" -d $DEST_DIR
rm "${DEST_DIR}/${FILENAME}"
echo 'Done'
