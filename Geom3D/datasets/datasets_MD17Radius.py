from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import radius_graph


class DatasetMD17Radius(InMemoryDataset):
    def __init__(self, root, preprcessed_dataset, radius):
        self.root = root
        self.dataset = preprcessed_dataset.dataset
        self.task = preprcessed_dataset.task
        self.preprcessed_dataset = preprcessed_dataset
        self.radius = radius

        self.transform = preprcessed_dataset.transform
        self.pre_transform = preprcessed_dataset.pre_transform
        self.pre_filter = preprcessed_dataset.pre_filter

        super(DatasetMD17Radius, self).__init__(root, self.transform, self.pre_transform, self.pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print("Dataset: {}\nTask: {}\nData: {}".format(self.dataset, self.task, self.data))
        return

    @property
    def processed_file_names(self):
        return self.task + '_pyg.pt'

    def process(self):
        print("Preprocessing on MD17 Radius ...")

        data_list = []
        for i in tqdm(range(len(self.preprcessed_dataset))):
            data = self.preprcessed_dataset.get(i)
            radius_edge_index = radius_graph(data.positions, r=self.radius, loop=False)
            data.radius_edge_index = radius_edge_index
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return
