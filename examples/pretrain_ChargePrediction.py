import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import args
from Geom3D.models import SchNet, DimeNetPlusPlus, PaiNN
from Geom3D.datasets import Molecule3DDataset, MoleculeDataset3DRadius
from torch_geometric.loader import DataLoader


class ChargePredictor(nn.Module):
    def __init__(self, emb_dim):
        super(ChargePredictor, self).__init__()
        self.predictor = nn.Linear(emb_dim, node_class)
        self.criterion = nn.CrossEntropyLoss()
        return

    def forward(self, node_repr, charge_actual):
        charge_pred = self.predictor(node_repr)
        loss = self.criterion(charge_pred, charge_actual)
        return loss


def save_model(save_best, epoch=None):
    if not args.output_model_dir == "":
        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))
            threeD_file = "model.pth"

        elif epoch is None:
            threeD_file = "model_final.pth"

        else:
            threeD_file = "model_{}.pth".format(epoch)

        saver_dict = {
            "model": molecule_model_3D.state_dict(),
        }
        saved_file_path = os.path.join(args.output_model_dir, threeD_file)
        torch.save(saver_dict, saved_file_path)
    return


def train(args, device, loader, optimizer):
    start_time = time.time()

    molecule_model_3D.train()
    charge_predictor.train()

    charge_loss_accum = 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for batch in l:
        batch = batch.to(device)

        charge = batch.x[:, 0]
        charge_actual = torch.clone(charge)
        M = charge.shape[0]
        sampled_M = int(M * args.charge_masking_ratio)
        masked_index = np.random.choice(M, sampled_M, replace=False)
        charge[masked_index] = node_class - 1

        charge = charge.to(device)
        charge_actual = charge_actual.to(device)

        if args.model_3d == "schnet":
            _, node_repr = molecule_model_3D(charge, batch.positions, batch.batch, return_latent=True)
        elif args.model_3d == "dimenetPP":
            _, node_repr = molecule_model_3D(charge, batch.positions, batch.batch, extract_representation=True, return_latent=True)
        elif args.model_3d == "painn":
            _, node_repr = molecule_model_3D(charge, batch.positions, batch.radius_edge_index, batch.batch, return_latent=True)

        charge_loss = charge_predictor(node_repr[masked_index], charge_actual[masked_index])
        charge_loss_accum += charge_loss.item()

        optimizer.zero_grad()
        charge_loss.backward()
        optimizer.step()
    
    global optimal_loss
    charge_loss_accum /= len(loader)
    if charge_loss_accum < optimal_loss:
        optimal_loss = charge_loss_accum
        save_model(save_best=True)
    print("L1 Loss: {:.5f}\tTime: {:.5f}".format(charge_loss_accum, time.time() - start_time))

    return


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)
    node_class, edge_class = 9, 4

    if args.input_data_dir == "":
        data_root = "../data/GEOM/{}".format(args.dataset)
    else:
        data_root = "{}/{}".format(args.input_data_dir, args.dataset)

    dataset = Molecule3DDataset(data_root, dataset=args.dataset)
    if args.model_3d == "painn":
        data_root = "{}_{}".format(data_root, args.painn_radius_cutoff)
        dataset = MoleculeDataset3DRadius(data_root, preprcessed_dataset=dataset, radius=args.painn_radius_cutoff)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.model_3d == "schnet":
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.num_filters,
            num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians,
            cutoff=args.cutoff,
            readout=args.readout,
            node_class=node_class,
        ).to(device)
    elif args.model_3d == "dimenetPP":
        assert args.dimenetPP_out_emb_channels == args.emb_dim
        molecule_model_3D = DimeNetPlusPlus(
            node_class=node_class,
            hidden_channels=args.emb_dim,
            out_channels=1,
            num_blocks=args.dimenetPP_num_blocks,
            int_emb_size=args.dimenetPP_int_emb_size,
            basis_emb_size=args.dimenetPP_basis_emb_size,
            out_emb_channels=args.dimenetPP_out_emb_channels,
            num_spherical=args.dimenetPP_num_spherical,
            num_radial=args.dimenetPP_num_radial,
            cutoff=args.dimenetPP_cutoff,
            envelope_exponent=args.dimenetPP_envelope_exponent,
            num_before_skip=args.dimenetPP_num_before_skip,
            num_after_skip=args.dimenetPP_num_after_skip,
            num_output_layers=args.dimenetPP_num_output_layers,
        ).to(device)
    elif args.model_3d == "painn":
        molecule_model_3D = PaiNN(
            n_atom_basis=args.emb_dim,  # default is 64
            n_interactions=args.painn_n_interactions,
            n_rbf=args.painn_n_rbf,
            cutoff=args.painn_radius_cutoff,
            max_z=node_class,
            n_out=1,
            readout=args.painn_readout,
        ).to(device)
    else:
        raise Exception("3D model {} not included.".format(args.model_3d))
    
    charge_predictor = ChargePredictor(args.emb_dim).to(device)

    # set up parameters
    model_param_group = [
        {"params": molecule_model_3D.parameters(), "lr": args.lr * args.gnn_3d_lr_scale},
        {"params": charge_predictor.parameters(), "lr": args.lr}
    ]

    # set up optimizers
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    # start training
    for epoch in range(1, args.epochs + 1):
        print("epoch: {}".format(epoch))
        train(args, device, loader, optimizer)

    # save final model weight
    save_model(save_best=False)
