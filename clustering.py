import time
import csv
import numpy as np
import torch
import faiss
from torchvision import transforms
from tqdm import tqdm
from datasets import GPSDataset

class KMeans(object):
    def __init__(self, k):
        self.k = k
    
    def cluster(self, data, pca=20):
        """
        Perform K-means clustering on the input data.

        Args:
            data (numpy.array): Input data with shape (num_samples, feature_dim).
            pca (int): Number of PCA components to reduce the feature dimension.

        Returns:
            float: Final K-means loss value.
            torch.Tensor: Tensor containing cluster assignments for each data point.
        """
        end = time.time()
        # PCA-reducing, whitening, and L2-normalization
        xb = preprocess_features(data, pca)
        # Cluster the data
        I, loss = run_kmeans(xb, self.k)
        images_lists = [[] for i in range(self.k)]
        labels = []
        for i in range(len(data)):
            labels.append(I[i])
            images_lists[I[i]].append(i)
        labels = torch.tensor(labels).cuda()
        print(labels)
        print('K-means time: {0:.0f} s'.format(time.time() - end))
        return loss, labels

def preprocess_features(npdata, pca=20):
    """
    Preprocess input features by applying PCA-whitening and L2 normalization.

    Args:
        npdata (numpy.array): Input data with shape (num_samples, feature_dim).
        pca (int): Number of PCA components to reduce the feature dimension.

    Returns:
        numpy.array: Processed data after PCA-whitening and L2 normalization.
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')
    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]
    return npdata

def run_kmeans(x, nmb_clusters):
    """
    Run K-means clustering on the input data using Faiss library.

    Args:
        x (numpy.array): Input data with shape (num_samples, feature_dim).
        nmb_clusters (int): Number of clusters to form.

    Returns:
        list: List containing cluster assignments for each data point.
        float: Final K-means loss value.
    """
    n_data, d = x.shape
    # Faiss implementation of K-means
    clus = faiss.Clustering(d, nmb_clusters)
    # Change Faiss seed at each K-means to get different initialization centroids
    clus.seed = 31
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    # Perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    stats = clus.iteration_stats
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    print('K-means loss evolution: {0}'.format(losses))
    return [int(n[0]) for n in I], losses[-1]

@torch.no_grad()
def compute_features(dataloader, model, N, batch_size, hidden):
    """
    Extract features from the input data using the given model.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the input data.
        model (torch.nn.Module): Trained model to extract features.
        N (int): Total number of data samples.
        batch_size (int): Batch size for feature extraction.
        hidden (int): Size of the extracted feature vector.

    Returns:
        numpy.array: Extracted features with shape (num_samples, hidden).
    """
    model.eval()
    # Discard the label information in the dataloader
    for i, (inputs, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = inputs.cuda()
        aux = model(inputs).data.cpu().numpy()
        aux = aux.reshape(-1, hidden)
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * batch_size: (i + 1) * batch_size] = aux
        else:
            features[i * batch_size:] = aux
    return features

def extract_cluster(ckpt_path, csv_path, data_path, batch_size, hidden, classes_num=5, offset=0):
    """
    Extract clusters from the input data using the trained model.

    Args:
        ckpt_path (str): Path to the checkpoint file of the trained model.
        csv_path (str): Path to the CSV file containing image paths.
        data_path (str): Path to the data directory.
        batch_size (int): Batch size for feature extraction.
        hidden (int): Size of the extracted feature vector.
        classes_num (int): Number of clusters to form.
        offset (int): Offset value for cluster labels.

    Returns:
        list: List containing image paths and their corresponding cluster labels.
    """
    convnet = torch.load(ckpt_path)
    convnet = torch.nn.DataParallel(convnet)
    convnet.cuda()
    cluster_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    clusterset = GPSDataset(csv_path, data_path, cluster_transform)
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=batch_size, shuffle=False, num_workers=0)

    deepcluster = KMeans(classes_num)
    features = compute_features(clusterloader, convnet, len(clusterset), batch_size, hidden)
    clustering_loss, p_label = deepcluster.cluster(features, pca=3)
    labels = p_label.tolist()
    f = open(csv_path, 'r', encoding='utf-8')
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    images.pop(0)
    cluster = []
    for i in range(0, len(images)):
        cluster.append([images[i], labels[i] + offset])

    return cluster
