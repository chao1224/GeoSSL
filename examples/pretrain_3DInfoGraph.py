import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import uniform
from config import args
from util import cycle_index
from Geom3D.models import GNN, SchNet, DimeNetPlusPlus, PaiNN
from Geom3D.datasets import Molecule3DDataset, MoleculeDataset3DRadius


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim=1)

        
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


def do_InfoGraph(node_repr, molecule_repr, batch,
                 criterion, infograph_discriminator_SSL_model):

    summary_repr = torch.sigmoid(molecule_repr)
    positive_expanded_summary_repr = summary_repr[batch.batch]
    shifted_summary_repr = summary_repr[cycle_index(len(summary_repr), 1)]
    negative_expanded_summary_repr = shifted_summary_repr[batch.batch]

    positive_score = infograph_discriminator_SSL_model(
        node_repr, positive_expanded_summary_repr)
    negative_score = infograph_discriminator_SSL_model(
        node_repr, negative_expanded_summary_repr)
    infograph_loss = criterion(positive_score, torch.ones_like(positive_score)) + \
                     criterion(negative_score, torch.zeros_like(negative_score))

    num_sample = float(2 * len(positive_score))
    infograph_acc = (torch.sum(positive_score > 0) +
                     torch.sum(negative_score < 0)).to(torch.float32) / num_sample
    infograph_acc = infograph_acc.detach().cpu().item()

    return infograph_loss, infograph_acc


def train(args, device, loader, optimizer):
    start_time = time.time()

    molecule_model_3D.train()
    infograph_discriminator_SSL_model.train()

    CL_loss_accum, CL_acc_accum = 0, 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for batch in l:
        batch = batch.to(device)

        if args.model_3d == "schnet":
            molecule_repr, node_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.batch, return_latent=True)
        elif args.model_3d == "dimenetPP":
            molecule_repr, node_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.batch, extract_representation=True, return_latent=True)
        elif args.model_3d == "painn":
            molecule_repr, node_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.radius_edge_index, batch.batch, return_latent=True)

        CL_loss, CL_acc = do_InfoGraph(
            node_repr, molecule_repr, batch,
            criterion, infograph_discriminator_SSL_model)

        CL_loss_accum += CL_loss.detach().cpu().item()
        CL_acc_accum += CL_acc

        loss = CL_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    temp_loss = CL_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print("CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tTime: {:.5f}".format(CL_loss_accum, CL_acc_accum, time.time() - start_time))
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

    # set up 3D base model
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
    infograph_discriminator_SSL_model = Discriminator(args.emb_dim).to(device)

    # set up parameters
    model_param_group = [
        {"params": molecule_model_3D.parameters(), "lr": args.lr * args.gnn_3d_lr_scale},
        {"params": infograph_discriminator_SSL_model.parameters(), "lr": args.lr * args.gnn_3d_lr_scale},
    ]
    criterion = nn.BCEWithLogitsLoss()

    # set up optimizers
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    # start training
    for epoch in range(1, args.epochs + 1):
        print("epoch: {}".format(epoch))
        train(args, device, loader, optimizer)

    # save final model weight
    save_model(save_best=False)
