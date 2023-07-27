import torch
import torch.nn as nn
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
import itertools
from torchvision import transforms
from utils import AUGLoss
from datasets import GPSDataset
from clustering import Kmeans, compute_features

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ClusterTransforms:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class TrainTransforms:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def deactivate_batchnorm(model):
    """Deactivate batch normalization layers in the given model."""
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for m in layer:
            if isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
                m.eval()
                with torch.no_grad():
                    m.weight.fill_(1.0)
                    m.bias.zero_()

def train_epoch(epoch, model, optimizer, loader_list, cluster_path_list, device, batch_sz):
    """
    Perform one training epoch for the given model.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loader_list (list): List of data loaders for each cluster.
        cluster_path_list (list): List of paths for clustering.
        device (torch.device): The device to run the computations on.
        batch_sz (int): Batch size used in training.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    # Deactivate the batch normalization before training
    # deactivate_batchnorm(model)
    
    train_loss = AverageMeter()
    reg_loss = AverageMeter()
    
    # For each cluster route
    path_idx = 0
    avg_loss = 0
    count = 0
    for cluster_path in cluster_path_list:
        path_idx += 1
        dataloaders = []
        for cluster_id in cluster_path:
            dataloaders.append(loader_list[cluster_id])
    
        pbar = tqdm(enumerate(zip(*dataloaders)), total=14*len(dataloaders))
        for batch_idx, data in pbar:
            cluster_num = len(data)
            data_zip = torch.cat(data, 0).to(device)

            # Generating Score
            scores = model(data_zip).squeeze()
            score_list = torch.split(scores, batch_sz, dim=0)
            
            # Standard deviation as a loss
            loss_var = torch.zeros(1).to(device)
            for score in score_list:
                loss_var += score.var()
            loss_var /= len(score_list)
            
            # Differentiable Ranking with sigmoid function
            rank_matrix = torch.zeros((batch_sz, cluster_num, cluster_num)).to(device)
            for itertuple in list(itertools.permutations(range(cluster_num), 2)):
                score1 = score_list[itertuple[0]]
                score2 = score_list[itertuple[1]]
                diff = 30 * (score2 - score1)
                results = torch.sigmoid(diff)
                rank_matrix[:, itertuple[0], itertuple[1]] = results
                rank_matrix[:, itertuple[1], itertuple[0]] = 1 - results

            rank_predicts = rank_matrix.sum(1)
            temp = torch.Tensor(range(cluster_num))
            target_rank = temp.unsqueeze(0).repeat(batch_sz, 1).to(device)

            # Equivalent to spearman rank correlation loss
            loss_train = ((rank_predicts - target_rank)**2).mean()
            loss = loss_train + loss_var * 6
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss_train.item(), batch_sz)
            reg_loss.update(loss_var.item(), batch_sz)
            avg_loss += loss.item()
            count += 1

            # Print status
            if batch_idx % 10 == 0:
                pbar.set_description(
                    f'Epoch: [{epoch}][{path_idx}][{batch_idx}] Train loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) Reg loss: {reg_loss.val:.4f} ({reg_loss.avg:.4f})'
                )
                
    return avg_loss / count

def finetune(csv_path, zone, data_path, model, hidden, k=8, pca=3, num_epoch=10, bz=16):
    """
    Fine-tune the given model using clustering.

    Args:
        csv_path (str): Path to the CSV file containing data information.
        zone (str): Zone information.
        data_path (str): Path to the dataset.
        model (nn.Module): The model to be fine-tuned.
        hidden (int): Number of hidden units in the model.
        k (int, optional): Number of clusters for K-means. Defaults to 8.
        pca (int, optional): Number of PCA components. Defaults to 3.
        num_epoch (int, optional): Number of fine-tuning epochs. Defaults to 10.
        bz (int, optional): Batch size. Defaults to 16.

    Returns:
        str: Path to the saved fine-tuned model.
    """
    criterion = nn.CrossEntropyLoss().cuda()
    criterion2 = AUGLoss().cuda()

    cluster_transforms = ClusterTransforms()
    train_transforms1 = TrainTransforms()
    train_transforms2 = TrainTransforms()

    clusterset = GPSDataset(csv_path, data_path, cluster_transforms.transform)
    trainset = GPSDataset(csv_path, data_path, train_transforms1.transform, train_transforms2.transform)

    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=bz, shuffle=False, num_workers=0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bz, shuffle=True, num_workers=0, drop_last=True)
    deepcluster = Kmeans(k)

    features = compute_features(clusterloader, model, len(clusterset), bz, hidden) 
    clustering_loss, p_label = deepcluster.cluster(features, pca=3)
    model.train()

    fc = nn.Linear(hidden, k)
    fc.weight.data.normal_(0, 0.01)
    fc.bias.data.zero_()
    fc.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer1 = torch.optim.SGD(fc.parameters(), lr=0.001)

    X_ = features
    pca = PCA(n_components=0.9) 
    pca.fit(X_)
    reduced_X = pca.transform(X_)
    print(X_.shape, reduced_X.shape)

    for epoch in range(0, num_epoch):
        print("Epoch : %d" % (epoch))
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for batch_idx, (inputs1, inputs2, indexes) in pbar:
            inputs1, inputs2, indexes = inputs1.cuda(), inputs2.cuda(), indexes.cuda()           
            batch_size = inputs1.shape[0]
            labels = p_label[indexes].cuda()
            inputs = torch.cat([inputs1, inputs2])
            outputs = model(inputs)
            outputs = outputs.reshape(-1, hidden)
            outputs1 = outputs[:batch_size]
            outputs2 = outputs[batch_size:]
            outputs3 = fc(outputs1)
            ce_loss = criterion(outputs3, labels)
            aug_loss = criterion2(outputs1, outputs2) / 20
            loss = ce_loss + aug_loss
            optimizer.zero_grad()
            optimizer1.zero_grad()
            ce_loss.backward()
            optimizer.step()
            optimizer1.step()

            if batch_idx % 20 == 0:
                pbar.set_description(f"[BATCH_IDX : {batch_idx} LOSS : {loss.item()} CE_LOSS {ce_loss.item()} AUG_LOSS : {aug_loss.item()}" )

    os.makedirs('finetune', exist_ok=True)
    torch.save(model, os.path.join('finetune', f'{zone}.pt'))
    return os.path.join('finetune', f'{zone}.pt')
