import os
import time
import numpy as np
from tqdm import tqdm
from config import args
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader

from splitters import random_split, atom3d_lba_split

from Geom3D.datasets import TransformLBA, DatasetLBA, DatasetLBARadius, MoleculeDatasetOneAtom
from Geom3D.models import SchNet, PaiNN


def train(epoch, device, loader):
    model.train()
    if graph_pred_linear is not None:
        graph_pred_linear.train()
    
    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    
    loss_acc = 0
    num_iters = len(loader)

    for step, batch in enumerate(L):
        batch = batch.to(device)

        if args.model_3d == "schnet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)
        elif args.model_3d == "painn":
            molecule_3D_repr = model(batch.x, batch.positions, batch.radius_edge_index, batch.batch)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_3D_repr).squeeze()
        else:
            pred = molecule_3D_repr.squeeze()
        actual = batch.y

        loss = criterion(pred, actual)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_acc += loss.cpu().detach().item()

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    loss_acc /= len(loader)
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    if args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc)
        
    return loss_acc


@torch.no_grad()
def eval(device, loader):
    model.eval()
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    
    loss_all, total = 0, 0
    y_true, y_pred = [], []

    for batch in loader:
        batch = batch.to(device)
        
        if args.model_3d == "schnet":
            molecule_3D_repr = model(batch.x, batch.positions, batch.batch)
        elif args.model_3d == "painn":
            molecule_3D_repr = model(batch.x, batch.positions, batch.radius_edge_index, batch.batch)
        
        if graph_pred_linear is not None:
            output = graph_pred_linear(molecule_3D_repr).squeeze()
        else:
            output = molecule_3D_repr.squeeze()
        y = batch.y

        B = y.size()[0]

        loss = F.mse_loss(output, y)
        loss_all += loss.item() * B
        total += B
        y_true.extend(y.tolist())
        y_pred.extend(output.tolist())

    pearson_corr = np.corrcoef(y_true, y_pred)[0,1]
    spearman_corr = spearmanr(y_true, y_pred)[0]

    return np.sqrt(loss_all / total), pearson_corr, spearman_corr, y_true, y_pred


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
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
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

    assert args.dataset == "lba"
    if args.input_data_dir == "":
        data_root = "../data/{}/".format(args.dataset)
    else:
        data_root = "{}/{}/".format(args.input_data_dir, args.dataset)

    dataframe_transformer = TransformLBA(dist=args.LBA_dist, maxnum=args.LBA_maxnum, move_lig=True)
    print("dataframe_transformer\t", dataframe_transformer)
    dataset = DatasetLBA(
        root=data_root, dist=args.LBA_dist, year=args.LBA_year,
        dataframe_transformer=dataframe_transformer, use_complex=args.LBA_use_complex)

    if args.model_3d == "painn":
        data_root = "{}_{}".format(dataset.root, args.painn_radius_cutoff)
        dataset = DatasetLBARadius(data_root, preprcessed_dataset=dataset, radius=args.painn_radius_cutoff)

    if args.only_one_atom_type:
        data_root = "{}_one_atom".format(dataset.root)
        dataset = MoleculeDatasetOneAtom(data_root, preprcessed_dataset=dataset)

    if args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset)
    elif args.split == "atom3d_lba_split30":
        train_dataset, valid_dataset, test_dataset = atom3d_lba_split(dataset, data_root, args.LBA_year)
    else:
        raise Exception("split {} not supported in LBA.".format(args.split))

    print("Train: {}\tValid: {}\tTest: {}".format(len(train_dataset), len(valid_dataset), len(test_dataset)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    node_class, edge_class = 9, 4
    num_tasks = 1
    
    # set up model
    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

    graph_pred_linear = None
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
    elif args.model_3d == "painn":
        model = PaiNN(
            n_atom_basis=args.emb_dim,  # default is 64
            n_interactions=args.painn_n_interactions,
            n_rbf=args.painn_n_rbf,
            cutoff=args.painn_radius_cutoff,
            max_z=node_class,
            n_out=num_tasks,
            readout=args.painn_readout,
        )
        graph_pred_linear = model.create_output_layers()
    else:
        raise Exception("Model {} no included.".format(args.model_3d))

    if args.input_model_file is not "":
        load_model(model, graph_pred_linear, args.input_model_file)
    model.to(device)
    print(model)
    if graph_pred_linear is not None:
        graph_pred_linear.to(device)
    print(graph_pred_linear)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    if graph_pred_linear is not None:
        model_param_group.append(
            {"params": graph_pred_linear.parameters(), "lr": args.lr}
        )
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    criterion = nn.MSELoss()

    # set up learning scheduler
    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        print("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.epochs, eta_min=1e-4
        )
        print("Apply lr scheduler CosineAnnealingWarmRestarts")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        print("Apply lr scheduler StepLR")
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_lr=args.min_lr
        )
        print("Apply lr scheduler ReduceLROnPlateau")

    train_metrics_list, valid_metrics_list, test_metrics_list = [], [], []
    best_valid_mse, best_valid_idx = 1e10, 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(epoch, device, train_loader)
        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if args.eval_train:
            train_mse, train_perason, train_spearman, train_target, train_pred = eval(device, train_loader)
        else:
            train_mse = train_perason = train_spearman = 0

        valid_mse, valid_pearson, valid_spearman, val_target, val_pred = eval(device, valid_loader)
        test_mse, test_pearson, test_spearman, test_target, test_pred  = eval(device, test_loader)

        train_metrics_list.append([train_mse, train_perason, train_spearman])
        valid_metrics_list.append([valid_mse, valid_pearson, valid_spearman])
        test_metrics_list.append([test_mse, test_pearson, test_spearman])

        print("MSE\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_mse, valid_mse, test_mse))
        print("Pearson\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_perason, valid_pearson, test_pearson))
        print("Spearman\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_spearman, valid_spearman, test_spearman))

        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_valid_idx = epoch - 1
            if not args.output_model_dir == "":
                save_model(save_best=True)

                filename = os.path.join(
                    args.output_model_dir, "evaluation_best.pth"
                )
                np.savez(
                    filename,
                    val_target=val_target,
                    val_pred=val_pred,
                    test_target=test_target,
                    test_pred=test_pred,
                )

        print("Took\t{}\n".format(time.time() - start_time))

    print("Best train\tMSE: {:.5f}\tPearson Corrleation: {:.5f}\tSpearman Correlation: {:.5f}".format(
        train_metrics_list[best_valid_idx][0], train_metrics_list[best_valid_idx][1], train_metrics_list[best_valid_idx][2]
    ))
    print("Best valid\tMSE: {:.5f}\tPearson Corrleation: {:.5f}\tSpearman Correlation: {:.5f}".format(
        valid_metrics_list[best_valid_idx][0], valid_metrics_list[best_valid_idx][1], valid_metrics_list[best_valid_idx][2]
    ))
    print("Best test\tMSE: {:.5f}\tPearson Corrleation: {:.5f}\tSpearman Correlation: {:.5f}".format(
        test_metrics_list[best_valid_idx][0], test_metrics_list[best_valid_idx][1], test_metrics_list[best_valid_idx][2]
    ))
    save_model(save_best=False)
