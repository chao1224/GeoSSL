
echo "$SLURM_TMPDIR"
cd "$SLURM_TMPDIR"


mkdir GEOM
cd GEOM
cp ~/scratch/3D_SSL/data/GEOM/rdkit_folder.tar.gz ./
tar -xvf rdkit_folder.tar.gz


cd ~/scratch/3D_SSL/src/
conda activate Geom3d
python generate_GEOM.py --n_mol=1000 --n_conf=1 --n_upper=1000
python generate_GEOM.py --n_mol=50000 --n_conf=5 --n_upper=1000


mv $SLURM_TMPDIR/GEOM/processed/* ../data/GEOM/
