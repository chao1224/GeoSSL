import argparse
from email.policy import default

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=int, default=0)

parser.add_argument(
    "--model_3d",
    type=str,
    default="schnet",
    choices=[
        "schnet",
        "dimenet",
        "dimenetPP",
        "tfn",
        "se3_transformer",
        "egnn",
        "spherenet",
        "segnn",
        "painn",
        "gemnet",
        "nequip",
        "allegro",
    ],
)
parser.add_argument(
    "--model_2d",
    type=str,
    default="gin",
    choices=[
        "gin",
        "schnet",
        "dimenet",
        "dimenetPP",
        "tfn",
        "se3_transformer",
        "egnn",
        "spherenet",
        "segnn",
        "painn",
        "gemnet",
        "nequip",
        "allegro",
    ],
)

# about dataset and dataloader
parser.add_argument("--dataset", type=str, default="qm9")
parser.add_argument("--task", type=str, default="alpha")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--only_one_atom_type", dest="only_one_atom_type", action="store_true")
parser.set_defaults(only_one_atom_type=False)

# for MD17
# The default hyper from here: https://github.com/divelab/DIG_storage/tree/main/3dgraph/md17
parser.add_argument("--md17_energy_coeff", type=float, default=0.05)
parser.add_argument("--md17_force_coeff", type=float, default=0.95)

# for COLL
# The default hyper from here: https://github.com/divelab/DIG_storage/tree/main/3dgraph/md17
parser.add_argument("--coll_energy_coeff", type=float, default=0.05)
parser.add_argument("--coll_force_coeff", type=float, default=0.95)

# for LBA
# The default hyper from here: https://github.com/drorlab/atom3d/blob/master/examples/lep/enn/utils.py#L37-L43
parser.add_argument("--LBA_year", type=int, default=2020)
parser.add_argument("--LBA_dist", type=float, default=6)
parser.add_argument("--LBA_maxnum", type=int, default=500)
parser.add_argument("--LBA_use_complex", dest="LBA_use_complex", action="store_true")
parser.add_argument("--LBA_no_complex", dest="LBA_use_complex", action="store_false")
parser.set_defaults(LBA_use_complex=False)

# for LEP
# The default hyper from here: https://github.com/drorlab/atom3d/blob/master/examples/lep/enn/utils.py#L48-L55
parser.add_argument("--LEP_dist", type=float, default=6)
parser.add_argument("--LEP_maxnum", type=float, default=400)
parser.add_argument("--LEP_droph", dest="LEP_droph", action="store_true")
parser.add_argument("--LEP_useh", dest="LEP_droph", action="store_false")
parser.set_defaults(LEP_droph=False)

# for MoleculeNet
parser.add_argument("--moleculenet_num_conformers", type=int, default=10)

# about training strategies
parser.add_argument("--split", type=str, default="customized_01",
                    choices=["customized_01", "customized_02", "random", "atom3d_lba_split30"])
parser.add_argument("--MD17_train_batch_size", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr_scale", type=float, default=1)
parser.add_argument("--decay", type=float, default=0)
parser.add_argument("--print_every_epoch", type=int, default=1)
parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
parser.add_argument("--lr_scheduler", type=str, default="CosineAnnealingLR")
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--lr_decay_step_size", type=int, default=100)
parser.add_argument("--lr_decay_patience", type=int, default=50)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--verbose", dest="verbose", action="store_true")
parser.add_argument("--no_verbose", dest="verbose", action="store_false")
parser.set_defaults(verbose=False)
parser.add_argument("--use_rotation_transform", dest="use_rotation_transform", action="store_true")
parser.add_argument("--no_rotation_transform", dest="use_rotation_transform", action="store_false")
parser.set_defaults(use_rotation_transform=False)

# for SchNet
parser.add_argument("--num_filters", type=int, default=128)
parser.add_argument("--num_interactions", type=int, default=6)
parser.add_argument("--num_gaussians", type=int, default=51)
parser.add_argument("--cutoff", type=float, default=10)
parser.add_argument("--readout", type=str, default="mean", choices=["mean", "add"])

# for PaiNN
parser.add_argument("--painn_radius_cutoff", type=float, default=5.0)
parser.add_argument("--painn_n_interactions", type=int, default=3)
parser.add_argument("--painn_n_rbf", type=int, default=20)
parser.add_argument("--painn_readout", type=str, default="add", choices=["mean", "add"])

######################### for Charge Prediction SSL #########################
parser.add_argument("--charge_masking_ratio", type=float, default=0.3)

######################### for Distance Perturbation SSL #########################
parser.add_argument("--distance_sample_ratio", type=float, default=1)

######################### for Torsion Angle Perturbation SSL #########################
parser.add_argument("--torsion_angle_sample_ratio", type=float, default=0.001)

######################### for Position Perturbation SSL #########################
parser.add_argument("--PP_mu", type=float, default=0)
parser.add_argument("--PP_sigma", type=float, default=0.3)


######################### for GraphMVP SSL #########################
### for 2D GNN
parser.add_argument("--gnn_type", type=str, default="gin")
parser.add_argument("--num_layer", type=int, default=5)
parser.add_argument("--emb_dim", type=int, default=128)
parser.add_argument("--dropout_ratio", type=float, default=0.5)
parser.add_argument("--graph_pooling", type=str, default="mean")
parser.add_argument("--JK", type=str, default="last")
parser.add_argument("--gnn_2d_lr_scale", type=float, default=1)

######################### for GeoSSL #########################
parser.add_argument("--EBMGeoSSL_mu", type=float, default=0)
parser.add_argument("--EBMGeoSSL_sigma", type=float, default=0.3)
parser.add_argument("--EBMGeoSSL_atom_masking_ratio", type=float, default=0.3)
parser.add_argument("--GeoSSL_option", type=str, default="EBM_NCE", choices=["EBM_NCE", "EBM_SM_01", "EBM_SM_02", "EBM_SM_03", "EBM_NCE_SM_01", "InfoNCE", "RR"])
parser.add_argument("--EBM_NCE_SM_coefficient", type=float, default=10.)

parser.add_argument("--SM_sigma_begin", type=float, default=10)
parser.add_argument("--SM_sigma_end", type=float, default=0.01)
parser.add_argument("--SM_num_noise_level", type=int, default=50)
parser.add_argument("--SM_noise_type", type=str, default="symmetry", choices=["symmetry", "random"])
parser.add_argument("--SM_anneal_power", type=float, default=2)

######################### for GraphMVP SSL #########################
### for 3D GNN
parser.add_argument("--gnn_3d_lr_scale", type=float, default=1)

### for masking
parser.add_argument("--SSL_masking_ratio", type=float, default=0.15)

### for 2D-3D Contrastive SSL
parser.add_argument("--CL_neg_samples", type=int, default=1)
parser.add_argument("--CL_similarity_metric", type=str, default="InfoNCE_dot_prod",
                    choices=["InfoNCE_dot_prod", "EBM_dot_prod"])
parser.add_argument("--T", type=float, default=0.1)
parser.add_argument("--normalize", dest="normalize", action="store_true")
parser.add_argument("--no_normalize", dest="normalize", action="store_false")
parser.add_argument("--alpha_1", type=float, default=1)

### for 2D-3D Generative SSL
parser.add_argument("--GraphMVP_AE_model", type=str, default="VAE")
parser.add_argument("--detach_target", dest="detach_target", action="store_true")
parser.add_argument("--no_detach_target", dest="detach_target", action="store_false")
parser.set_defaults(detach_target=True)
parser.add_argument("--AE_loss", type=str, default="l2", choices=["l1", "l2", "cosine"])
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--alpha_2", type=float, default=1)

### for 2D SSL
parser.add_argument("--GraphMVP_2D_mode", type=str, default="AM", choices=["AM", "CP"])
parser.add_argument("--alpha_3", type=float, default=1)
### for AttributeMask
parser.add_argument("--mask_rate", type=float, default=0.15)
parser.add_argument("--mask_edge", type=int, default=0)
### for ContextPred
parser.add_argument("--csize", type=int, default=3)
parser.add_argument("--contextpred_neg_samples", type=int, default=1)
#######################################################################



##### about if we would print out eval metric for training data
parser.add_argument("--eval_train", dest="eval_train", action="store_true")
parser.add_argument("--no_eval_train", dest="eval_train", action="store_false")
parser.set_defaults(eval_train=False)
##### about if we would print out eval metric for training data
##### this is only for COLL
parser.add_argument("--eval_test", dest="eval_test", action="store_true")
parser.add_argument("--no_eval_test", dest="eval_test", action="store_false")
parser.set_defaults(eval_test=True)

parser.add_argument("--input_data_dir", type=str, default="")

# about loading and saving
parser.add_argument("--input_model_file", type=str, default="")
parser.add_argument("--output_model_dir", type=str, default="")

args = parser.parse_args()
print("arguments\t", args)
