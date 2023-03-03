from multiprocessing.dummy import active_children
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
from Geom3D.models import SchNet, PaiNN, AutoEncoder
from Geom3D.dataloaders import AtomTupleExtractor, DataLoaderAtomTuple
from NCSN import NCSN_version_03


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
    else:
        raise Exception("3D model {} not included.".format(args.model_3d))
    return model


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))

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


def perturb(x, positions, mu, sigma):
    x_perturb = x

    device = positions.device
    positions_perturb = positions + torch.normal(mu, sigma, size=positions.size()).to(device)

    return x_perturb, positions_perturb
    

def do_RR(args, batch, model, mu, sigma, **kwargs):
    positions = batch.positions

    x_01 = batch.x[:, 0]
    positions_01 = positions
    x_02, positions_02 = perturb(x_01, positions, mu, sigma)

    if args.model_3d == "schnet":
        molecule_3D_repr_01 = model(x_01, positions_01, batch.batch)
        molecule_3D_repr_02 = model(x_02, positions_02, batch.batch)
    elif args.model_3d == "painn":
        molecule_3D_repr_01 = model(x_01, positions_01, batch.radius_edge_index, batch.batch)
        molecule_3D_repr_02 = model(x_02, positions_02, batch.radius_edge_index, batch.batch)

    if args.normalize:
        molecule_3D_repr_01 = F.normalize(molecule_3D_repr_01, dim=-1)
        molecule_3D_repr_02 = F.normalize(molecule_3D_repr_02, dim=-1)

    AE_loss_1 = AE_model_01(molecule_3D_repr_01, molecule_3D_repr_02)
    AE_loss_2 = AE_model_02(molecule_3D_repr_02, molecule_3D_repr_01)
    AE_loss = (AE_loss_1 + AE_loss_2) / 2

    AE_acc = 0
    return AE_loss, AE_acc


def do_EBM_NCE(args, batch, model, criterion, mu, sigma, num_neg=1):
    positions = batch.positions

    x_01 = batch.x[:, 0]
    positions_01 = positions
    x_02, positions_02 = perturb(x_01, positions, mu, sigma)

    if args.model_3d == "schnet":
        molecule_3D_repr_01 = model(x_01, positions_01, batch.batch)
        molecule_3D_repr_02 = model(x_02, positions_02, batch.batch)
    elif args.model_3d == "painn":
        molecule_3D_repr_01 = model(x_01, positions_01, batch.radius_edge_index, batch.batch)
        molecule_3D_repr_02 = model(x_02, positions_02, batch.radius_edge_index, batch.batch)

    if args.normalize:
        molecule_3D_repr_01 = F.normalize(molecule_3D_repr_01, dim=-1)
        molecule_3D_repr_02 = F.normalize(molecule_3D_repr_02, dim=-1)

    B = len(molecule_3D_repr_01)

    molecule_3D_repr_01_pos = molecule_3D_repr_01
    molecule_3D_repr_02_pos = molecule_3D_repr_02
    molecule_3D_repr_01_neg = molecule_3D_repr_01_pos.repeat((num_neg, 1))
    molecule_3D_repr_02_neg = torch.cat([molecule_3D_repr_02[cycle_index(B, i + 1)] for i in range(num_neg)], dim=0)

    pred_pos = torch.sum(molecule_3D_repr_01_pos * molecule_3D_repr_02_pos, dim=1)
    pred_neg = torch.sum(molecule_3D_repr_01_neg * molecule_3D_repr_02_neg, dim=1)

    loss_pos = criterion(pred_pos.double(), torch.ones(B).to(device).double())
    loss_neg = criterion(pred_neg.double(), torch.zeros(B * num_neg).to(device).double())

    SSL_loss = (loss_pos + num_neg * loss_neg) / (1 + num_neg)

    num_pred = len(pred_pos) + len(pred_neg)
    SSL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / num_pred
    return SSL_loss, SSL_acc.detach().item()


def do_InfoNCE(args, batch, model, criterion, mu, sigma, num_neg=1):
    positions = batch.positions

    x_01 = batch.x[:, 0]
    positions_01 = positions
    x_02, positions_02 = perturb(x_01, positions, mu, sigma)

    if args.model_3d == "schnet":
        molecule_3D_repr_01 = model(x_01, positions_01, batch.batch)
        molecule_3D_repr_02 = model(x_02, positions_02, batch.batch)
    elif args.model_3d == "painn":
        molecule_3D_repr_01 = model(x_01, positions_01, batch.radius_edge_index, batch.batch)
        molecule_3D_repr_02 = model(x_02, positions_02, batch.radius_edge_index, batch.batch)

    if args.normalize:
        molecule_3D_repr_01 = F.normalize(molecule_3D_repr_01, dim=-1)
        molecule_3D_repr_02 = F.normalize(molecule_3D_repr_02, dim=-1)

    def cal_loss(X, Y):
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device) # B*1

        CL_loss = CE_criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B
        return CL_loss, CL_acc
    
    SSL_loss_01, SSL_acc_01 = cal_loss(molecule_3D_repr_01, molecule_3D_repr_02)
    SSL_loss_02, SSL_acc_02 = cal_loss(molecule_3D_repr_02, molecule_3D_repr_01)

    SSL_loss = (SSL_loss_01 + SSL_loss_02) / 2

    SSL_acc = (SSL_acc_01 + SSL_acc_02) / 2
    return SSL_loss, SSL_acc


def do_DDM(args, batch, model, criterion, mu, sigma, num_neg=1):
    positions = batch.positions

    x_01 = batch.x[:, 0]
    positions_01 = positions
    x_02, positions_02 = perturb(x_01, positions, mu, sigma)

    if args.model_3d == "schnet":
        _, molecule_3D_repr_01 = model(x_01, positions_01, batch.batch, return_latent=True)
        _, molecule_3D_repr_02 = model(x_02, positions_02, batch.batch, return_latent=True)
    elif args.model_3d == "painn":
        _, molecule_3D_repr_01 = model(x_01, positions_01, batch.radius_edge_index, batch.batch, return_latent=True)
        _, molecule_3D_repr_02 = model(x_02, positions_02, batch.radius_edge_index, batch.batch, return_latent=True)

    if args.normalize:
        molecule_3D_repr_01 = F.normalize(molecule_3D_repr_01, dim=-1)
        molecule_3D_repr_02 = F.normalize(molecule_3D_repr_02, dim=-1)

    super_edge_index = batch.super_edge_index

    u_pos_01 = torch.index_select(positions_01, dim=0, index=super_edge_index[0])
    v_pos_01 = torch.index_select(positions_01, dim=0, index=super_edge_index[1])
    distance_01 = torch.sqrt(torch.sum((u_pos_01-v_pos_01)**2, dim=1)).unsqueeze(1) # (num_edge, 1)
    
    u_pos_02 = torch.index_select(positions_02, dim=0, index=super_edge_index[0])
    v_pos_02 = torch.index_select(positions_02, dim=0, index=super_edge_index[1])
    distance_02 = torch.sqrt(torch.sum((u_pos_02-v_pos_02)**2, dim=1)).unsqueeze(1) # (num_edge, 1)

    loss_01 = NCSN_model_01(batch, molecule_3D_repr_01, distance_02)
    loss_02 = NCSN_model_02(batch, molecule_3D_repr_02, distance_01)

    loss = (loss_01 + loss_02) / 2

    return loss, 0


def train(device, loader, optimizer):
    model.train()
    if AE_model_01 is not None:
        AE_model_01.train()
    if AE_model_02 is not None:
        AE_model_02.train()
    if NCSN_model_01 is not None:
        NCSN_model_01.train()
    if NCSN_model_02 is not None:
        NCSN_model_02.train()

    accum_loss, accum_acc = 0, 0
    num_iters = len(loader)
    start_time = time.time()

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = batch.to(device)

        if args.GeoSSL_option == "EBM_NCE":
            loss, acc = do_EBM_NCE(
                args, batch, model, criterion=criterion,
                mu=args.GeoSSL_mu, sigma=args.GeoSSL_sigma)
        elif args.GeoSSL_option == "InfoNCE":
            loss, acc = do_InfoNCE(
                args, batch, model, criterion=criterion,
                mu=args.GeoSSL_mu, sigma=args.GeoSSL_sigma)
        elif args.GeoSSL_option == "RR":
            loss, acc = do_RR(
                args, batch, model, criterion=criterion,
                mu=args.GeoSSL_mu, sigma=args.GeoSSL_sigma)
        elif args.GeoSSL_option == "DDM":
            loss, acc = do_DDM(
                args, batch, model, criterion=criterion,
                mu=args.GeoSSL_mu, sigma=args.GeoSSL_sigma)
        else:
            raise ValueError("{} not included.".format(args.GeoSSL_option))
        accum_loss += loss.detach().item()
        accum_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    accum_loss /= num_iters
    accum_acc /= num_iters
    if accum_loss < optimal_loss:
        optimal_loss = accum_loss
        save_model(save_best=True)

    print("SSL Loss: {:.5f}\tSSL Acc: {:.5f}\tTime: {:.3f}".format(accum_loss, accum_acc, time.time() - start_time))
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

    if args.input_data_dir == "":
        data_root = "../data/GEOM/{}".format(args.dataset)
    else:
        data_root = "{}/{}".format(args.input_data_dir, args.dataset)

    if args.SM_noise_type == "random":
        option = "permutation"
    elif args.SM_noise_type == "symmetry":
        option = "combination"
    else:
        raise Exception
    transform = AtomTupleExtractor(ratio=args.distance_sample_ratio, option=option)
    dataset = Molecule3DDataset(data_root, dataset=args.dataset, mask_ratio=args.GeoSSL_atom_masking_ratio, transform=transform)
    if args.model_3d == "painn":
        data_root = "{}_{}".format(data_root, args.painn_radius_cutoff)
        dataset = MoleculeDataset3DRadius(data_root, preprcessed_dataset=dataset, mask_ratio=args.GeoSSL_atom_masking_ratio, radius=args.painn_radius_cutoff)

    loader = DataLoaderAtomTuple(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up model
    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    node_class = 9
    model = model_setup()

    if args.input_model_file is not "":
        model.from_pretrained(args.input_model_file)
    model.to(device)
    print(model)

    AE_model_01 = AE_model_02 = None
    NCSN_model_01 = NCSN_model_02 = None
    if args.GeoSSL_option == "RR":
        AE_model_01 = AutoEncoder(emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target,beta=args.beta).to(device)
        AE_model_02 = AutoEncoder(emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target,beta=args.beta).to(device)
    elif args.GeoSSL_option == "DDM":
        NCSN_model_01 = NCSN_version_03(
            args.emb_dim,
            sigma_begin=args.SM_sigma_begin, sigma_end=args.SM_sigma_end, num_noise_level=args.SM_num_noise_level,
            noise_type=args.SM_noise_type, anneal_power=args.SM_anneal_power).to(device)
        NCSN_model_02 = NCSN_version_03(
            args.emb_dim,
            sigma_begin=args.SM_sigma_begin, sigma_end=args.SM_sigma_end, num_noise_level=args.SM_num_noise_level,
            noise_type=args.SM_noise_type, anneal_power=args.SM_anneal_power).to(device)
        
    # set up optimizer
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    if AE_model_01 is not None:
        model_param_group += [{"params": AE_model_01.parameters(), "lr": args.gnn_2d_lr_scale}]
    if AE_model_02 is not None:
        model_param_group += [{"params": AE_model_02.parameters(), "lr": args.gnn_2d_lr_scale}]
    if NCSN_model_01 is not None:
        model_param_group += [{"params": NCSN_model_01.parameters()}]
    if NCSN_model_02 is not None:
        model_param_group += [{"params": NCSN_model_02.parameters()}]

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss()
    CE_criterion = nn.CrossEntropyLoss()
    optimal_loss = 1e10

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        print("Apply lr scheduler CosineAnnealingLR")

    best_val_mae, best_val_idx = 1e10, 0
    for epoch in range(1, args.epochs + 1):
        print("Epoch: {}".format(epoch))
        start_time = time.time()
        train(device, loader, optimizer)
        lr_scheduler.step()
        print()

    save_model(save_best=False)
