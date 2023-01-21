
from collections import defaultdict

import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
from torch_geometric.data import Data


# note this is different from the 2D case
allowable_features = {
    # atom maps in {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9; 'P': 15, 'S': 16, 'CL': 17}
    "possible_atomic_num_list": [1, 6, 7, 8, 9, 15, 16, 17, "unknown"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE,
    ],
}


# shorten the sentence
feats = allowable_features


def mol_to_graph_data_obj_simple_2D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric. Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr"""
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if atomic_number not in feats["possible_atomic_num_list"]:
            atomic_number = "unknown"

        atom_feature = \
            [feats["possible_atomic_num_list"].index(atomic_number)] + \
            [feats["possible_chirality_list"].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    N = len(mol.GetAtoms())

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_feats_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_dir = bond.GetBondDir()
            if bond_dir not in feats["possible_bond_dirs"]:
                bond_dir = Chem.rdchem.BondDir.NONE
            edge_feature = \
                [feats["possible_bonds"].index(bond_type)] + \
                [feats["possible_bond_dirs"].index(bond_dir)]
            edges_list.append((i, j))
            edge_feats_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feats_list.append(edge_feature)

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feats_list), dtype=torch.long)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return data


def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric. Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr"""
    # atoms
    atom_features_list = []
    atom_count = defaultdict(int)
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atom_count[atomic_number] += 1
        if atomic_number not in feats["possible_atomic_num_list"]:
            atomic_number = "unknown"

        atom_feature = \
            [feats["possible_atomic_num_list"].index(atomic_number)] + \
            [feats["possible_chirality_list"].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    N = len(mol.GetAtoms())

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_feats_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_dir = bond.GetBondDir()
            if bond_dir not in feats["possible_bond_dirs"]:
                bond_dir = Chem.rdchem.BondDir.NONE
            edge_feature = \
                [feats["possible_bonds"].index(bond_type)] + \
                [feats["possible_bond_dirs"].index(bond_dir)]
            edges_list.append((i, j))
            edge_feats_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feats_list.append(edge_feature)

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feats_list), dtype=torch.long)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        positions=positions,
    )
    return data, atom_count


def mol_to_graph_data_obj_MMFF_3D(rdkit_mol, num_conformers):
    try:
        N = len(rdkit_mol.GetAtoms())
        if N > 100: # for sider
            raise Exception
        rdkit_mol = Chem.AddHs(rdkit_mol)
        mol = rdkit_mol
        result_list = AllChem.EmbedMultipleConfs(mol, num_conformers)
        result_list = AllChem.MMFFOptimizeMoleculeConfs(mol)
        mol = Chem.RemoveHs(mol)
        energy_list = [x[1] for x in result_list]
        index = np.argmin(energy_list)
        energy = energy_list[index]
        conformer = mol.GetConformer(id=int(index))
    except:
        print("======bad")
        mol = rdkit_mol
        AllChem.Compute2DCoords(mol)
        energy = 0
        conformer = mol.GetConformer()

    atom_features_list = []
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if atomic_number not in feats["possible_atomic_num_list"]:
            atomic_number = "unknown"

        atom_feature = \
            [feats["possible_atomic_num_list"].index(atomic_number)] + \
            [feats["possible_chirality_list"].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    N = len(mol.GetAtoms())

    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_feats_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_dir = bond.GetBondDir()
            if bond_dir not in feats["possible_bond_dirs"]:
                bond_dir = Chem.rdchem.BondDir.NONE
            edge_feature = \
                [feats["possible_bonds"].index(bond_type)] + \
                [feats["possible_bond_dirs"].index(bond_dir)]
            edges_list.append((i, j))
            edge_feats_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feats_list.append(edge_feature)

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feats_list), dtype=torch.long)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        positions=positions,
    )
    return data


def graph_data_obj_to_nx_simple(data):
    """ torch geometric -> networkx
    NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: networkx object """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        G.add_node(i, atom_num_idx=atomic_num_idx,
                   chirality_tag_idx=chirality_tag_idx)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx,
                       bond_type_idx=bond_type_idx,
                       bond_dir_idx=bond_dir_idx)

    return G


def nx_to_graph_data_obj_simple(G):
    """ vice versa of graph_data_obj_to_nx_simple()
    Assume node indices are numbered from 0 to num_nodes - 1.
    NB: Uses simplified atom and bond features, and represent as indices.
    NB: possible issues with recapitulating relative stereochemistry
        since the edges in the nx object are unordered. """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['atom_num_idx'], node['chirality_tag_idx']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data