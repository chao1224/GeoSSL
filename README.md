# Molecular Geometry Pretraining with SE(3)-Invariant Denoising Distance Matching

**ICLR 2023**

Authors: Shengchao Liu, Hongyu Guo, Jian Tang

[[Project Page](https://chao1224.github.io/GeoSSL)]
[[OpenReview](https://openreview.net/forum?id=CjTHVo1dvR)]
[[ArXiv](https://arxiv.org/abs/2206.13602)]

This repository provides the source code for the ICLR'23 paper **Molecular Geometry Pretraining with SE(3)-Invariant Denoising Distance Matching**, with the following task:
- This project explores the geometric representation learning on molecules.
- We consider pure geometric information, *i.e.*, the molecule conformation.
- For pretraining, we consider [Molecule3D](https://arxiv.org/abs/2110.01717).
- For fine-tuning (downstream), we consider QM9, MD17, LBA & LEP.

## Environments

## Pretraining

For pretraining, we provide implementations on eight pretraining baselines and our proposed GeoSSL-DDM under the `examples` folder:
- Supervised pretraining in `pretrain_Supervised.py`.
- Type prediction pretraining in `pretrain_ChargePrediction.py`.
- Distance prediction pretraining in `pretrain_DistancePrediction.py`.
- Angle prediction pretraining in `pretrain_TorsionAnglePreddiction.py`.
- 3D InfoGraph pretraining in `pretrain_3DInfoGraph.py`.
- GeoSSL pretraining framework in `pretrain_GeoSSL.py`.
  - GeoSSL-RR pretraining with argument `--GeoSSL_option=RR`.
  - GeoSSL-InfoNCE pretraining with argument `--GeoSSL_option=InfoNCE`.
  - GeoSSL-EBM-NCE pretraining with argument `--GeoSSL_option=EBM-NCE`.
  - GeoSSL-DDM pretraining (ours) with argument `--GeoSSL_option=DDM`.

<p align="center">
  <img src="fig/pipeline.png" /> 
</p>

## Downstream

The downstream scripts can be found under the `examples` folder:
- `finetune_qm9.py`
- `finetune_md17.py`
- `finetune_lba.py`
- `finetune_lep.py`

The running scripts and corresponding hyper-parameters can be found in `scripts/xxx`. Note that as a fair comparison, we keep a fixed hyper-parameter set for each downstream task, and the only difference is the pretrained checkpoints.

## Checkpoints

We provide both the log files and checkpoints for GeoSSL-DDM [here](). For other baselines, due to the space limitation, we only provide the log files.

## Cite us

Feel free to cite this work if you find it useful to you!

```
@inproceedings{
    liu2023molecular,
    title={Molecular Geometry Pretraining with {SE}(3)-Invariant Denoising Distance Matching},
    author={Shengchao Liu and Hongyu Guo and Jian Tang},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=CjTHVo1dvR}
}
```
