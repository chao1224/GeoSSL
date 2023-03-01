#!/usr/bin/env bash

source $HOME/.bashrc
conda activate Geom3D
conda deactivate
conda activate Geom3D

echo $@
date

echo "start"
python -u pretrain_EBM.py $@
echo "end"
date
