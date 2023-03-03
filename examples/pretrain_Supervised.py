import os
from re import X
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader

from config import args
from util import cycle_index
from Geom3D.datasets import Molecule3DDataset, MoleculeDataset3DRadius
from Geom3D.models import DimeNetPlusPlus, SchNet, PaiNN, AutoEncoder


def model_setup():
    if args.model_3d == "schnet":
        model = SchNet(
            hidden_channels=args.emb_dim,
            num_filters=args.num_filters,
            num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians,
            cutoff=args.cutoff,
            readout=args.readout,
            node_class=node_class,
        )
        graph_pred_linear = torch.nn.Linear(intermediate_dim, num_tasks)

    elif args.model_3d == "dimenetPP":
        model = DimeNetPlusPlus(
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
        )
        graph_pred_linear = torch.nn.Linear(intermediate_dim, num_tasks)

    elif args.model_3d == "painn":
        model = PaiNN(
            n_atom_basis=args.emb_dim,  # default is 64
            n_interactions=args.painn_n_interactions,
            n_rbf=args.painn_n_rbf,
            cutoff=args.painn_radius_cutoff,
            max_z=node_class,
            n_out=1,
            readout=args.painn_readout,
        )
        graph_pred_linear = model.create_output_layers()

    else:
        raise Exception("3D model {} not included.".format(args.model_3d))
    return model, graph_pred_linear


def train(device, loader, optimizer):
    model.train()

    accum_loss, accum_acc = 0, 0
    num_iters = len(loader)
    start_time = time.time()

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = batch.to(device)

        if args.model_3d == "schnet":
            molecule_3D_repr = model(batch.x[:, 0], batch.positions, batch.batch)

        elif args.model_3d == "painn":
            molecule_3D_repr = model(batch.x, batch.positions, batch.radius_edge_index, batch.batch)
            
        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_3D_repr).squeeze()
        else:
            pred = molecule_3D_repr.squeeze()

        B = pred.size()[0]
        y = batch.y.view(B, -1)[:, task_id]
        y = (y - TRAIN_mean) / TRAIN_std

        loss = criterion(pred, y)
        
        accum_loss += loss.detach().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    accum_loss /= num_iters
    accum_acc /= num_iters
    if accum_loss < optimal_loss:
        optimal_loss = accum_loss
        save_model(save_best=True)

    print("Supervised Loss: {:.5f}\tSupervised Acc: {:.5f}\tTime: {:.3f}".format(accum_loss, accum_acc, time.time() - start_time))
    return


def load_model(model, graph_pred_linear, model_weight_file):
    print("Loading from {}".format(model_weight_file))
    model_weight = torch.load(model_weight_file)
    model.load_state_dict(model_weight["model"])
    if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
        graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])
    return


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            global optimal_loss
            print("save model with optimal loss: {:.5f}".format(optimal_loss))

            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    num_tasks = 1
    # Credit to https://github.com/divelab/MoleculeX/blob/molx/Molecule3D/preprocess/PropertyCSV.py#L24
    # Credit to https://github.com/divelab/MoleculeX/blob/molx/Molecule3D/pred_prop_3d.py#L17
    task_id = 6
    assert "Molecule3D" in args.dataset
    if args.input_data_dir == "":
        data_root = "../data/GEOM/{}".format(args.dataset)
    else:
        data_root = "{}/{}".format(args.input_data_dir, args.dataset)
    dataset = Molecule3DDataset(data_root, dataset=args.dataset)
    if args.model_3d == "painn":
        data_root = "{}_{}".format(data_root, args.painn_radius_cutoff)
        dataset = MoleculeDataset3DRadius(data_root, preprcessed_dataset=dataset, radius=args.painn_radius_cutoff)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    y_value = torch.stack([dataset.get(i).y for i in range(len(dataset))], dim=0)
    y_value = y_value[:, task_id]
    print(y_value.shape)
    TRAIN_mean, TRAIN_std = y_value.mean().item(), y_value.std().item()
    print("Train mean: {}\tTrain std: {}".format(TRAIN_mean, TRAIN_std))

    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "mae":
        criterion = nn.L1Loss()
    else:
        raise ValueError("Loss {} not included.".format(args.loss))

    # set up model
    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    node_class = 9
    model, graph_pred_linear = model_setup()

    if args.input_model_file is not "":
        load_model(model, graph_pred_linear, args.input_model_file)
    model.to(device)
    print(model)
    if graph_pred_linear is not None:
        graph_pred_linear.to(device)
        print(graph_pred_linear)

    # set up optimizer
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    if graph_pred_linear is not None:
        model_param_group.append(
            {"params": graph_pred_linear.parameters(), "lr": args.lr}
        )

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        print("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        print("Apply lr scheduler StepLR")

    for epoch in range(1, args.epochs + 1):
        print("Epoch: {}".format(epoch))
        start_time = time.time()
        train(device, loader, optimizer)
        
        if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
            lr_scheduler.step()
        print()

    save_model(save_best=False)
