#!/usr/bin/env bash

source $HOME/.bashrc
conda activate Geom3D_old
conda deactivate
conda activate Geom3D_old

echo $@
date

echo "start"
python -u finetune_lep.py $@
echo "end"
date
