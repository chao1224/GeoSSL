import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from config import args
from Geom3D.datasets import DatasetMD17, DatasetMD17Radius
from Geom3D.models import SchNet, PaiNN

from torch.autograd import grad


def train(device, loader):
    model.train()
    if graph_pred_linear is not None:
        graph_pred_linear.train()

    loss_acc = 0
    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    for step, batch_data in enumerate(L):
        batch_data = batch_data.to(device)
        positions = batch_data.positions
        positions.requires_grad_()
        x = batch_data.x

        if args.model_3d == "schnet":
            molecule_3D_repr = model(x, positions, batch_data.batch)
        elif args.model_3d == "painn":
            molecule_3D_repr = model(x, positions, batch_data.radius_edge_index, batch_data.batch)

        if graph_pred_linear is not None:
            pred_energy = graph_pred_linear(molecule_3D_repr).squeeze(1)
        else:
            pred_energy = molecule_3D_repr.squeeze(1)

        pred_force = -grad(outputs=pred_energy, inputs=positions, grad_outputs=torch.ones_like(pred_energy), create_graph=True, retain_graph=True)[0]

        actual_energy = batch_data.y
        actual_force = batch_data.force

        loss = args.md17_energy_coeff * criterion(pred_energy, actual_energy) + args.md17_force_coeff * criterion(pred_force, actual_force)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_acc += loss.cpu().detach().item()

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    loss_acc /= len(loader)
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc)
    return loss_acc


def eval(device, loader):
    model.eval()
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    pred_energy_list, actual_energy_list = [], []
    pred_force_list = torch.Tensor([]).to(device)
    actual_force_list = torch.Tensor([]).to(device)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    for batch_data in L:
        batch_data = batch_data.to(device)
        positions = batch_data.positions
        positions.requires_grad_()
        x = batch_data.x

        if args.model_3d == "schnet":
            molecule_3D_repr = model(x, positions, batch_data.batch)
        elif args.model_3d == "painn":
            molecule_3D_repr = model(x, positions, batch_data.radius_edge_index, batch_data.batch)

        if graph_pred_linear is not None:
            pred_energy = graph_pred_linear(molecule_3D_repr).squeeze(1)
        else:
            pred_energy = molecule_3D_repr.squeeze(1)

        force = -grad(outputs=pred_energy, inputs=positions, grad_outputs=torch.ones_like(pred_energy), create_graph=True, retain_graph=True)[0].detach_()

        # for one input in benzene2017, the computed force is nan. 
        # (might be some issue when computing gradient, but only occur for one input graph). 
        # So we skipped this input.
        if torch.sum(torch.isnan(force)) != 0:
            mask = torch.isnan(force)
            force = force[~mask].reshape((-1, 3))
            batch_data.force = batch_data.force[~mask].reshape((-1, 3))

        pred_energy_list.append(pred_energy.cpu().detach())
        actual_energy_list.append(batch_data.y.cpu())
        pred_force_list = torch.cat([pred_force_list, force], dim=0)
        actual_force_list = torch.cat([actual_force_list, batch_data.force], dim=0)

    pred_energy_list = torch.cat(pred_energy_list, dim=0)
    actual_energy_list = torch.cat(actual_energy_list, dim=0)
    energy_mae = torch.mean(torch.abs(pred_energy_list - actual_energy_list)).cpu().item()
    force_mae = torch.mean(torch.abs(pred_force_list - actual_force_list)).cpu().item()

    # pred_energy_list = pred_energy_list.cpu().numpy()
    # actual_energy_list = actual_energy_list.cpu().numpy()
    # pred_force_list = pred_force_list.cpu().numpy()
    # actual_force_list = actual_force_list.cpu().numpy()
    return energy_mae, force_mae


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

    assert args.dataset == "md17"
    data_root = "../data/md17"
    dataset = DatasetMD17(data_root, task=args.task)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=args.seed)
    print("train:", len(split_idx["train"]), split_idx["train"][:5])
    print("valid:", len(split_idx["valid"]), split_idx["valid"][:5])
    print("test:", len(split_idx["test"]), split_idx["test"][:5])

    if args.model_3d == "painn":
        data_root = "../data/md17_{}".format(args.painn_radius_cutoff)
        dataset = DatasetMD17Radius(
            data_root,
            preprcessed_dataset=dataset,
            radius=args.painn_radius_cutoff
        )
    train_dataset, val_dataset, test_dataset = \
        dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    DataLoaderClass = DataLoader

    train_loader = DataLoaderClass(train_dataset, args.MD17_train_batch_size, shuffle=True, num_workers=args.num_workers, **dataloader_kwargs)
    val_loader = DataLoaderClass(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, **dataloader_kwargs)
    test_loader = DataLoaderClass(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, **dataloader_kwargs)

    node_class = 9
    num_tasks = 1

    # set up model
    if args.JK == "concat":
        intermediate_dim = (args.num_layer + 1) * args.emb_dim
    else:
        intermediate_dim = args.emb_dim

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
        raise Exception("3D model {} not included.".format(args.model_3d))

    if args.input_model_file is not "":
        load_model(model, graph_pred_linear, args.input_model_file)
    model.to(device)
    print(model)
    if graph_pred_linear is not None:
        graph_pred_linear.to(device)
        print(graph_pred_linear)

    criterion = torch.nn.L1Loss()
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    if graph_pred_linear is not None:
        model_param_group.append(
            {"params": graph_pred_linear.parameters(), "lr": args.lr}
        )
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

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
    else:
        print("lr scheduler {} is not included.".format(args.lr_scheduler))

    train_energy_mae_list, train_force_mae_list = [], []
    val_energy_mae_list, val_force_mae_list = [], []
    test_energy_mae_list, test_force_mae_list = [], []
    best_val_force_mae = 1e10
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(device, train_loader)
        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_energy_mae, train_force_mae = eval(device, train_loader)
            else:
                train_energy_mae = train_force_mae = 0
            val_energy_mae, val_force_mae = eval(device, val_loader)
            if args.eval_test:
                test_energy_mae, test_force_mae = eval(device, test_loader)
            else:
                test_energy_mae = test_force_mae = 0

            train_energy_mae_list.append(train_energy_mae)
            train_force_mae_list.append(train_force_mae)
            val_energy_mae_list.append(val_energy_mae)
            val_force_mae_list.append(val_force_mae)
            test_energy_mae_list.append(test_energy_mae)
            test_force_mae_list.append(test_force_mae)
            print("Energy\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_energy_mae, val_energy_mae, test_energy_mae))
            print("Force\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_force_mae, val_force_mae, test_force_mae))

            if val_force_mae < best_val_force_mae:
                best_val_force_mae = val_force_mae
                best_val_idx = len(train_energy_mae_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True)
        print("Took\t{}\n".format(time.time() - start_time))
    
    save_model(save_best=False)

    if args.eval_test:
        optimal_test_energy, optimal_test_force = test_energy_mae_list[best_val_idx], test_force_mae_list[best_val_idx]
    else:
        optimal_model_weight = os.path.join(args.output_model_dir, "model.pth")
        load_model(model, graph_pred_linear, optimal_model_weight)
        optimal_test_energy, optimal_test_force = eval(device, test_loader)

    print("best Energy\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_energy_mae_list[best_val_idx], val_energy_mae_list[best_val_idx], optimal_test_energy))
    print("best Force\ttrain: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_force_mae_list[best_val_idx], val_force_mae_list[best_val_idx], optimal_test_force))
