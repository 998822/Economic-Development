import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class GPSDataset(Dataset):
    """
    Dataset for GPS data.
    """
    def __init__(self, metadata_file, root_dir, transform1=None, transform2=None):
        """
        Initialize the GPSDataset.

        Args:
            metadata_file (str): Path to the CSV file containing metadata.
            root_dir (str): Root directory of the dataset.
            transform1 (callable, optional): First data transformation to apply. Default is None.
            transform2 (callable, optional): Second data transformation to apply. Default is None.
        """
        self.metadata = pd.read_csv(metadata_file).values
        self.root_dir = root_dir
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata[idx][0])
        image = Image.open(img_name).convert('RGB')

        if self.transform1:
            img1 = self.transform1(image)
        
        if self.transform2:
            img2 = self.transform2(image)
            return img1, img2, idx
                
        return img1, idx

class ClusterDataset(Dataset):
    """
    Dataset for clusters.
    """
    def __init__(self, cluster_list, transform=None):
        """
        Initialize the ClusterDataset.

        Args:
            cluster_list (list): List of cluster numbers.
            transform (callable, optional): Data transformation to apply. Default is None.
        """
        self.file_list = []
        self.transform = transform

        for cluster_num in cluster_list:
            file_list = list(pd.read_csv(os.path.join("./data", str(cluster_num), "cluster.csv"))['y_x'].values)
            self.file_list.extend(file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join("./data12/train/data_zl12", self.file_list[idx])).convert('RGB')

        if self.transform:
            image = self.transform(image).squeeze()

        return image

def make_data_loader(cluster_list, batch_sz, num_workers=0):
    """
    Create a DataLoader for the given cluster list.

    Args:
        cluster_list (list): List of cluster numbers.
        batch_sz (int): Batch size for the DataLoader.
        num_workers (int, optional): Number of workers for data loading. Default is 0 (no multi-threading).

    Returns:
        DataLoader: The DataLoader for the cluster list.
    """
    cluster_dataset = ClusterDataset(cluster_list, transform=transforms.Compose([                    
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]))
    cluster_loader = DataLoader(cluster_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers, drop_last=True)
    return cluster_loader

def generate_loader_dict(total_list, batch_sz, num_workers=0):
    """
    Generate a dictionary of DataLoaders for the given cluster list.

    Args:
        total_list (list): List of cluster numbers.
        batch_sz (int): Batch size for the DataLoaders.
        num_workers (int, optional): Number of workers for data loading. Default is 0 (no multi-threading).

    Returns:
        dict: Dictionary with cluster IDs as keys and corresponding DataLoaders as values.
    """
    loader_dict = {}
    for cluster_id in total_list:
        cluster_loader = make_data_loader([cluster_id], batch_sz, num_workers)
        loader_dict[cluster_id] = cluster_loader

    return loader_dict