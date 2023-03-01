
import os
import numpy as np
from itertools import repeat
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph, to_networkx


class Molecule3DMaskingDataset(InMemoryDataset):
    def __init__(self, root, dataset, mask_ratio, transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.root = root
        self.dataset = dataset
        self.mask_ratio = mask_ratio

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super(Molecule3DMaskingDataset, self).__init__(root, self.transform, self.pre_transform, self.pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("Dataset: {}\nData: {}".format(self.dataset, self.data))
        return

    def subgraph(self, data):
        G = to_networkx(data)
        node_num, _ = data.x.size()
        sub_num = int(node_num * (1 - self.mask_ratio))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        # BFS
        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(
                set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

        idx_nondrop = idx_sub
        idx_nondrop.sort()

        # prev_edge_idx = data.edge_index
        edge_idx, edge_attr = subgraph(
            subset=idx_nondrop,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            relabel_nodes=True,
            num_nodes=node_num
        )
        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.positions = data.positions[idx_nondrop]
        data.__num_nodes__, _ = data.x.shape

        if "radius_edge_index" in data:
            # prev_radius_edge_idx = data.radius_edge_index
            radius_edge_index, _ = subgraph(
                subset=idx_nondrop,
                edge_index=data.radius_edge_index,
                relabel_nodes=True, 
                num_nodes=node_num)
            data.radius_edge_index = radius_edge_index
            # assert torch.max(prev_edge_idx) == torch.max(prev_radius_edge_idx)
            # assert torch.max(edge_idx) == torch.max(radius_edge_index)
        
        # TODO: will also need to do this for other edge_index

        return data

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        if self.mask_ratio > 0:
            data = self.subgraph(data)
        return data
    
    def _download(self):
        return

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        return