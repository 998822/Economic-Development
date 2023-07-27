import argparse
import itertools
import os
import pandas as pd
import timm
import random
import numpy as np
import torch
import torch.nn as nn
from clustering import extract_cluster
from datasets import generate_loader_dict  
import geoio
from graph import graph_inference_nightlight
from trainer import finetune, train_epoch
from utils import AverageMeter, create_space, deg2num, num2deg
from sklearn.mixture import GaussianMixture as GMM

def parse_args():
    parser = argparse.ArgumentParser(description="Train clustering model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")   
    parser.add_argument("--proxy_path", type=str, required=True, help="Path to proxy data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden size of model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--cluster_num", type=int, default=15, help="Number of clusters")
    
    return parser.parse_args()

def add_nightlights(df, tif, tif_array):
    '''
    Match by lat/lon and compute avg nightlights 
    using 10km x 10km box around point
    '''
    
    nightlights = []
    
    for _, row in df.iterrows():
        min_lat, min_lon, max_lat, max_lon = create_space(row.cluster_lat, row.cluster_lon, s=8)
        
        xmin, ymax = tif.proj_to_raster(min_lon, min_lat)
        xmax, ymin = tif.proj_to_raster(max_lon, max_lat)
        
        if xmin < 0 or xmax >= tif_array.shape[1]:
            raise ValueError(f"No match for {row.cluster_lat}, {row.cluster_lon}")
            
        if ymin < 0 or ymax >= tif_array.shape[0]:
            raise ValueError(f"No match for {row.cluster_lat}, {row.cluster_lon}")
            
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        nightlights.append(tif_array[ymin:ymax,xmin:xmax].mean())
        
    df['nightlights'] = nightlights

def downsample(df, lower, upper, fraction):
    boolean_idx = ((lower <= df['nightlights']) & (df['nightlights'] <= upper))
    subset = df[boolean_idx]
    
    if len(subset) / len(df) <= fraction:
        return df
    
    num_to_drop = int((len(subset) - fraction * len(df)) / (1 - fraction))
    drop_indices = random.sample(subset.index.tolist(), num_to_drop)
    
    return df.drop(drop_indices).reset_index(drop=True)

def create_nightlight_bins(df, cutoffs):
    cutoffs = sorted(cutoffs, reverse=True) 
    labels = list(range(len(cutoffs)))[::-1]
    
    df['nightlights_bin'] = len(cutoffs)
    
    for cutoff, label in zip(cutoffs, labels):
        df.loc[df['nightlights'] <= cutoff, 'nightlights_bin'] = label
        
def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    np.random.seed(31)
    random.seed(31)

    # Process data
    images = [x for x in os.listdir(args.data_path) if 'png' in x]
    data = pd.DataFrame()
    data['name'] = images
    
    data['cluster_lat'] = [num2deg(deg2num(float(x.strip('.png').split('_')[1]),float(x.strip('.png').split('_')[0]),12)[1],deg2num(float(x.strip('.png').split('_')[1]),float(x.strip('.png').split('_')[0]),12)[0],12)[0] for x in images]

    data['cluster_lon'] = [num2deg(deg2num(float(x.strip('.png').split('_')[1]),float(x.strip('.png').split('_')[0]),12)[1],deg2num(float(x.strip('.png').split('_')[1]),float(x.strip('.png').split('_')[0]),12)[0],12)[1] for x in images]

    # Add nightlights
    tif = geoio.GeoImage(args.proxy_path)
    tif_array = np.squeeze(tif.get_data())  
    add_nightlights(data, tif, tif_array)

    data = data[['cluster_lat','cluster_lon','nightlights']] 
    data_drop = downsample(data, lower=0, upper=2, fraction=0.6) 
    
    # Cluster nightlights 
    X = data_drop['nightlights'].values.reshape(-1,1)
    gmm = GMM(n_components=3).fit(X)
    labels = gmm.predict(data_drop['nightlights'].reshape(-1,1))

    label0_max = data_drop[labels==0]['nightlights'].max()
    label1_max = data_drop[labels==1]['nightlights'].max() 
    label2_max = data_drop[labels==2]['nightlights'].max()

    create_nightlight_bins(data_drop, [label0_max, label1_max, label2_max])

    # Save binned data
    for i in range(3):
        bin_data = data_drop[data_drop['nightlights_bin'] == i][['cluster_lat','cluster_lon']]
        bin_data.to_csv(f'./nightlights_labeled{i}.csv', index=False)

    # Fine-tune model on nightlight bins
    model = timm.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 1) 
    model = model.cuda()

    nature_csv = './nightlights_labeled0.csv'
    rural_csv = './nightlights_labeled1.csv'
    city_csv = './nightlights_labeled2.csv'

    ckpt_nature = finetune(nature_csv, 'nature', args.data_path, model, args.hidden, num_epoch=args.num_epochs, bz=args.batch_size)
    ckpt_rural = finetune(rural_csv, 'rural', args.data_path, model, args.hidden, num_epoch=args.num_epochs) 
    ckpt_city = finetune(city_csv, 'city', args.data_path, model, args.hidden, num_epoch=args.num_epochs)

    # Clustering
    ckpt_rural = 'finetune/rural.pt'
    ckpt_city = 'finetune/city.pt'
    ckpt_nature = 'finetune/nature.pt'

    city_cluster = extract_cluster(ckpt_city, city_csv, args.data_path, batch_size=10, hidden=args.hidden, classes_num=5, offset=10)
    rural_cluster = extract_cluster(ckpt_rural, rural_csv, args.data_path, batch_size=10, hidden=args.hidden, classes_num=5, offset=5)
    nature_cluster = extract_cluster(ckpt_nature, nature_csv, args.data_path, batch_size=10, hidden=args.hidden, classes_num=5, offset=0)

    total_cluster = city_cluster + rural_cluster + nature_cluster
    cluster_df = pd.DataFrame(total_cluster, columns=['image','cluster_id']) 

    # Save clusters
    cluster_dir = './data/'
    os.makedirs(cluster_dir, exist_ok=True)

    for i in range(args.cluster_num + 1):
        os.makedirs(os.path.join(cluster_dir, str(i)), exist_ok=True)
        cluster_data = cluster_df[cluster_df['cluster_id'] == i][['image']]
        cluster_data.to_csv(os.path.join(cluster_dir, str(i), 'cluster.csv'), index=False)

    cluster_df.to_csv('./unified.csv', index=False)

    # Train clustering model
    grid_df = pd.read_csv('./unified.csv')
    nightlight_df = pd.read_csv('./nightlights_labeled.csv')

    graph_config = graph_inference_nightlight(grid_df, nightlight_df, args.cluster_num, './CNconfig')

    loader_list = generate_loader_dict(range(args.cluster_num), args.batch_size)
    cluster_paths = [list(item)[::-1] for item in itertools.product(*graph_config)]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        loss = train_epoch(epoch, model, optimizer, loader_list, cluster_paths, device='cuda:0', batch_sz=args.batch_size)
        
        if epoch % 5 == 0 and epoch != 0:
            if loss < best_loss:
                best_loss = loss
                torch.save(model, os.path.join(args.output_dir, 'model_best.pt'))
                print(f"New best loss: {best_loss:.4f}")

    torch.save(model, os.path.join(args.output_dir, "model.pth"))

if __name__ == '__main__':
    main()