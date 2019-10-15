#!/bin/bash
##commandline
## bash pipeline.sh 1a3a
##
cd ./rdb2c2/ridge
`python extract_ridge.py \$1`
cd ./rdb2c2/TF-feature
`python build-train-file.py \$1`
`python prediction.py \$1`
##### final npy file and image file ./rdb2c2/TF-feature/resul and image file #####
