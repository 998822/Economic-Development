import pandas as pd
import numpy as np
from PIL import Image
from tqdm import trange, tqdm
from utils import create_space
import convert as conv
import glob, shutil, copy, csv, os, faiss, time, random

import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict 
from functools import cmp_to_key
from scipy.optimize import nnls

import glob
import torch
import numpy as np
from skimage import io, transform

import torchvision.transforms.functional as F 
from torch.utils.data import Dataset
from PIL import Image
import random

import itertools 
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import pandas as pd
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

class GPSDataset(Dataset):
    def __init__(self, metadata, root_dir,transform1=None, transform2=None):
        self.metadata = pd.read_csv(metadata).values
        self.root_dir = root_dir
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata[idx][0])
        image =  Image.open(img_name).convert('RGB')
        if self.transform1:
            img1 = self.transform1(image)
        if self.transform2:
            img2 = self.transform2(image)
            return img1, img2, idx
                
        return img1, idx

class AUGLoss(nn.Module):
    def __init__(self):
        super(AUGLoss, self).__init__()

    def forward(self, x1, x2):
        b = (x1 - x2)
        b = b*b
        b = b.sum(1)
        b = torch.sqrt(b)
        return b.sum()

# Below codes are from Deep Clustering for Unsupervised Learning of Visual Features github code        
def preprocess_features(npdata, pca=20):
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')
    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)
    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]
    return npdata

def cluster_assign(images_lists, dataset):
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)

def run_kmeans(x, nmb_clusters):
    n_data, d = x.shape
    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = 31
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    stats = clus.iteration_stats
    losses = np.array([stats.at(i).obj for i in range(stats.size())])
    print('k-means loss evolution: {0}'.format(losses))
    return [int(n[0]) for n in I], losses[-1]

@torch.no_grad()
def compute_features(dataloader, model, N, batch_size, hidden):
    model.eval()
    # discard the label information in the dataloader
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

class Kmeans(object):
    def __init__(self, k):
        self.k = k
    def cluster(self, data,pca):
        end = time.time()
        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data,pca)
        # cluster the data
        I, loss = run_kmeans(xb, self.k)
        self.images_lists = [[] for i in range(self.k)]
        label = []
        for i in range(len(data)):
            label.append(I[i])
            self.images_lists[I[i]].append(i)
        label = torch.tensor(label).cuda()
        print(label)
        print('k-means time: {0:.0f} s'.format(time.time() - end))
        return loss, label
    
def extract_cluster(ckpt_path, csv_path, data_path, batch_size, hidden, classes_num=5, offset=0):
    convnet = torch.load(ckpt_path, )
    convnet = torch.nn.DataParallel(convnet)    
    convnet.cuda()
    cluster_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    
    
    clusterset = GPSDataset(csv_path, data_path, cluster_transform)
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    deepcluster = Kmeans(classes_num)
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
        cluster.append([images[i], labels[i]+offset]) 
        
    return cluster

class Graph: 
    def __init__(self,vertices): 
        self.V= vertices            
        self.graph = defaultdict(list)  
 
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
   
    def printPathsFunc(self, u, d, visited, path, current_path_list): 
        visited[u]= True
        path.append(u) 

        if u == d: 
            path_copy = path[:]
            current_path_list.append(path_copy)
        else: 
            for i in self.graph[u]: 
                if visited[i]==False: 
                    self.printPathsFunc(i, d, visited, path, current_path_list) 
                      
        path.pop() 
        visited[u]= False
        return current_path_list

        
    def printPaths(self, s, d): 
        total_results = []
        for start in s:
            for dest in d:
                path = []
                visited =[False]*(self.V) 
                current_path_list = []
                current_path_results = self.printPathsFunc(start, dest, visited, path, current_path_list) 
                if len(current_path_results) != 0:
                    total_results.extend(current_path_results)
        return total_results
    
    
def graph_process(config_path):
    cluster_unify = []
    partial_order = []
    start_candidates = []
    end_candidates = []
    
    f = open(config_path, 'r')
    while True:
        line = f.readline()
        if '=' in line:
            unify = list(map(int, line.split('=')))
            cluster_unify.append(unify)
        elif '<' in line:
            order = list(map(int, line.split('<')))
            partial_order.append(order)
            start_candidates.append(order[0])
            end_candidates.append(order[1])
            
        if not line: break
    f.close()
        
    start = []
    end = []
    for element in start_candidates:
        if element in end_candidates:
            continue
        start.append(element)
    
    for element in end_candidates:
        if element in start_candidates:
            continue
        end.append(element)
    
    start = list(set(start))
    end = list(set(end))
    return start, end, partial_order, cluster_unify



def generate_graph(partial_order_list, vertex_num):
    cluster_graph = Graph(vertex_num)
    for pair in partial_order_list:
        cluster_graph.addEdge(pair[0], pair[1])
    return cluster_graph 


def save_graph_config(ordered_list, name):
    f = open(name, 'w')
        
    for i in range(len(ordered_list) - 1):
        f.write('{}<{}\n'.format(ordered_list[i+1][0], ordered_list[i][0]))
    
    for orders in ordered_list:
        if len(orders) >= 2:
            f.write(str(orders[0]))
            for element in orders[1:]:
                f.write('={}'.format(element))
            f.write('\n')        
    f.close()        
    
            
def graph_inference_nightlight(grid_df, nightlight_df, cluster_num, file_path):
    def numeric_compare(x, y):
        pop_list1 = df_merge_group.get_group(x)['nightlights'].tolist()
        pop_list2 = df_merge_group.get_group(y)['nightlights'].tolist()
        tTestResult = stats.ttest_ind(pop_list1, pop_list2)
        if (tTestResult.pvalue < 0.01) and (np.mean(pop_list1) < np.mean(pop_list2)):
            return 1
        elif (tTestResult.pvalue < 0.01) and (np.mean(pop_list1) >= np.mean(pop_list2)):
            return -1
        else:
            return 0
        
    df_merge = pd.merge(nightlight_df, grid_df, how='left', on='y_x')
    df_merge = df_merge.dropna()
    df_merge_group = df_merge.groupby('cluster_id')
    
    sorted_list = sorted(range(cluster_num - 1), key=cmp_to_key(numeric_compare))
    ordered_list = []
    ordered_list.append([sorted_list[0]])
    curr = 0
    for i in range(len(sorted_list) - 1):
        if numeric_compare(sorted_list[i], sorted_list[i+1]) == 0:
            ordered_list[curr].append(sorted_list[i+1])
        else:
            curr += 1
            ordered_list.append([sorted_list[i+1]])
            
    ordered_list.append([cluster_num - 1])        
    save_graph_config(ordered_list, file_path)
    return ordered_list

class ClusterDataset(Dataset):
    def __init__(self, cluster_list, transform=None):
        self.file_list = []
        self.transform = transform      
        for cluster_num in cluster_list:
            file=list(pd.read_csv("./data/{}/cluster.csv".format(str(cluster_num)))['y_x'].values)
            self.file_list.extend(file)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = Image.open("./data12/train/data_zl12/{}".format(self.file_list[idx])).convert('RGB')
        if self.transform:
            image = self.transform(image).squeeze()
        return image

    
class RandomRotate(object):
    def __call__(self, images):
        rotated = np.stack([self.random_rotate(x) for x in images])
        return rotated
    
    def random_rotate(self, image):
        rand_num = np.random.randint(0, 4)
        if rand_num == 0:
            return np.rot90(image, k=1, axes=(0, 1))
        elif rand_num == 1:
            return np.rot90(image, k=2, axes=(0, 1))
        elif rand_num == 2:
            return np.rot90(image, k=3, axes=(0, 1))   
        else:
            return image
    
    
class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, images):
        normalized = np.stack([F.normalize(x, self.mean, self.std, self.inplace) for x in images]) 
        return normalized
 

    
class Grayscale(object):
    def __init__(self, prob = 1):
        self.prob = prob

    def __call__(self, images):     
        random_num = np.random.randint(100, size=1)[0]
        if random_num <= self.prob * 100:
            gray_images = (images[:, 0, :, :] + images[:, 1, :, :] + images[:, 2, :, :]) / 3
            gray_scaled = gray_images.unsqueeze(1).repeat(1, 3, 1, 1)
            return gray_scaled
        else:
            return images

    

class ToTensor(object):
    def __call__(self, images):
        images = images.transpose((0, 3, 1, 2))
        return torch.from_numpy(images).float()

class AverageMeter(object):
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
        
def make_data_loader(cluster_list, batch_sz):
    cluster_dataset = ClusterDataset(cluster_list, transform = transforms.Compose([                    
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]))
    cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=batch_sz, shuffle=True, num_workers=0, drop_last=True)
    return cluster_loader



def generate_loader_dict(total_list, batch_sz):
    loader_dict = {}
    for cluster_id in total_list:
        cluster_loader = make_data_loader([cluster_id], batch_sz)
        loader_dict[cluster_id] = cluster_loader        
    
    #for cluster_tuple in unified_cluster_list:
        #cluster_loader = make_data_loader(cluster_tuple, batch_sz)
        #for cluster_num in cluster_tuple:
            #loader_dict[cluster_num] = cluster_loader
    return loader_dict


def deactivate_batchnorm(model):
    for layer in [model.layer1,model.layer2,model.layer3,model.layer4]:
        for m in model.layer1:
            if isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
                m.eval()
                with torch.no_grad():
                    m.weight.fill_(1.0)
                    m.bias.zero_()
#  rain(epoch, model, optimizer, loader_list, cluster_path_list, device)       
        
def train(epoch, model, optimizer, loader_list, cluster_path_list, device,batch_sz):
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
#         for batch_idx, data in pbar:
#             cluster_num = len(data)
#             out = []
# #             data = [im.cuda() for im in data]
# #             for i in data:
# #                 #print(i.shape)
# #                 i = i.cuda()
# #                 out.append(model(i))
# #                 del i
#             data_zip = torch.cat(data, 0).to(device)
#             # print(data_zip.shape)
#             # Generating Score
#           #  print(data_zip.shape)
#            # return 
#             scores = model(data_zip).squeeze()
#             # scores = torch.cat(out, 0)
#             score_list = torch.split(scores, batch_sz, dim = 0)
            
#             # Standard deviation as a loss
#             loss_var = torch.zeros(1).to(device)
#             for score in score_list:
#                 #print(loss_var)
#                 loss_var += score.var()
#             loss_var /= len(score_list)
            
#             # Differentiable Ranking with sigmoid function
#             rank_matrix = torch.zeros((batch_sz, cluster_num, cluster_num)).to(device)
#             for itertuple in list(itertools.permutations(range(cluster_num), 2)):
#                 score1 = score_list[itertuple[0]]
#                 score2 = score_list[itertuple[1]]
#                 diff = 30 * (score2 - score1)
#                 results = torch.sigmoid(diff)
#                 #print(results.shape)
#                 rank_matrix[:, itertuple[0], itertuple[1]] = results.squeeze()
#                 rank_matrix[:, itertuple[1], itertuple[0]] = (1 - results).squeeze()

#             rank_predicts = rank_matrix.sum(1)
#             temp = torch.Tensor(range(cluster_num))
#             target_rank = temp.unsqueeze(0).repeat(batch_sz, 1).to(device)

#             # Equivalent to spearman rank correlation loss
#             loss_train = ((rank_predicts - target_rank)**2).mean()
#             loss = loss_train + loss_var * 4
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss.update(loss_train.item(), batch_sz)
#             reg_loss.update(loss_var.item(), batch_sz)
#             avg_loss += loss.item()
#             count += 1

        for batch_idx, data in pbar:
            cluster_num = len(data)
            data_zip = torch.cat(data, 0).to(device)

            # Generating Score
            scores = model(data_zip).squeeze()
            score_list = torch.split(scores, batch_sz, dim = 0)
            
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
                pbar.set_description('Epoch: [{epoch}][{path_idx}][{elps_iters}] '
                      'Train loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      'Reg loss: {reg_loss.val:.4f} ({reg_loss.avg:.4f})'.format(
                          epoch=epoch, path_idx=path_idx, elps_iters=batch_idx, train_loss=train_loss, reg_loss=reg_loss))
                
    return avg_loss / count


def graph_inference_nightlight(grid_df, nightlight_df, cluster_num, file_path):
    def numeric_compare(x, y):
        pop_list1 = df_merge_group.get_group(x)['nightlights'].tolist()
        pop_list2 = df_merge_group.get_group(y)['nightlights'].tolist()
        tTestResult = stats.ttest_ind(pop_list1, pop_list2)
        if (tTestResult.pvalue < 0.01) and (np.mean(pop_list1) < np.mean(pop_list2)):
            return 1
        elif (tTestResult.pvalue < 0.01) and (np.mean(pop_list1) >= np.mean(pop_list2)):
            return -1
        else:
            return 0
        
    df_merge = pd.merge(nightlight_df, grid_df, how='left', on='y_x')
    df_merge = df_merge.dropna()
    df_merge_group = df_merge.groupby('cluster_id')
    
    sorted_list = sorted(range(cluster_num - 1), key=cmp_to_key(numeric_compare))
    ordered_list = []
    ordered_list.append([sorted_list[0]])
    curr = 0
    for i in range(len(sorted_list) - 1):
        if numeric_compare(sorted_list[i], sorted_list[i+1]) == 0:
            ordered_list[curr].append(sorted_list[i+1])
        else:
            curr += 1
            ordered_list.append([sorted_list[i+1]])
            
    ordered_list.append([cluster_num - 1])        
    save_graph_config(ordered_list, file_path)
    return ordered_list
cluster_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
train_transform1 =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
train_transform2 =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def finetune(csv_path, zone, data_path, model, hidden, k=8, pca=3, num_epoch=10, bz=16):
    criterion = nn.CrossEntropyLoss().cuda()
    criterion2 = AUGLoss().cuda()
    
    clusterset = GPSDataset(csv_path, data_path, cluster_transform)
    trainset = GPSDataset(csv_path, data_path, train_transform1, train_transform2)

    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=bz, shuffle=False, num_workers=0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bz, shuffle=True, num_workers=0, drop_last = True)
    deepcluster = Kmeans(k)

    features = compute_features(clusterloader, model, len(clusterset), bz, hidden) 
    clustering_loss, p_label = deepcluster.cluster(features,pca=3)
    model.train()

    fc = nn.Linear(hidden, k)
    fc.weight.data.normal_(0, 0.01)
    fc.bias.data.zero_()
    fc.cuda()

    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    optimizer1 = torch.optim.SGD(fc.parameters(),lr=0.001)

    from sklearn.decomposition import PCA

    X_ = features
    pca = PCA(n_components = 0.9) 
    pca.fit(X_)
    reduced_X = pca.transform(X_)
    print(X_.shape,reduced_X.shape)

    for epoch in range(0, num_epoch):
        print("Epoch : %d"% (epoch))
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for batch_idx, (inputs1, inputs2, indexes) in pbar:
            inputs1, inputs2, indexes = inputs1.cuda(), inputs2.cuda(), indexes.cuda()           
            batch_size = inputs1.shape[0]
            labels = p_label[indexes].cuda()
            inputs = torch.cat([inputs1, inputs2])
            outputs = model(inputs)
            outputs=outputs.reshape(-1,hidden)
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
