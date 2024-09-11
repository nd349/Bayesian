import glob
import warnings
warnings.filterwarnings("ignore")

import torch, time
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import random, os
import netCDF4 as nc
from torch.utils.tensorboard import SummaryWriter
from os import listdir
from scipy.stats import pearsonr
# from torchmetrics import R2Score, F1Score

from unet_model import UNet

num_lats, num_lons = 481, 601
full_xLim = [ -125.0, -120.0 ]
full_yLim = [   36.0,   40.0 ]
orig_lats = np.linspace(full_yLim[0], full_yLim[1], num_lats)
orig_lons = np.linspace(full_xLim[0], full_xLim[1], num_lons)
clon_index = int(orig_lons.shape[0]/2)
clat_index = int(orig_lats.shape[0]/2)
clon = orig_lats[clon_index]
clat = orig_lons[clat_index]
lats = orig_lats[clat_index-200:clat_index+200]
lons = orig_lons[clon_index-200:clon_index+200]


RES = 16
NUNIT = 1024
DIM1 = 400   # 
DIM2 = 400
SHAPE1 = int((DIM1 + 15)//RES*RES  )
SHAPE2 = int((DIM2 + 15)//RES*RES  )
LATNTSHAPE = int(SHAPE2//16*SHAPE1//16)
HRRR_DIR = PATH
# WRF_DIR = '/home/disk/hermes/data/met_data/BarnettShale_2013/wrf/MYJ_LSM/' # no longer in use
predlist = ['GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'SHGT', 'T02M', 'ADS',
           'UWND', 'VWND', 'WWND', 'PRES', 'TEMP', 'AD'] # 14
Rd = 2.87053  # hPa·K-1·m3·kg–1
ggg = 9.80665 # m/s*s

def split_df(_df, limit=None):
    # Split df into train_df and val_df
    if limit:
        totalsize = min(_df.shape[0], limit)
    else:
        totalsize = _df.shape[0]
    trainsize, validsize = int(totalsize*0.85**2), int(totalsize*0.85*0.15)
    testsize = totalsize - trainsize - validsize
    train_list  = _df[:trainsize]['path'].tolist()
    valid_list = _df[trainsize:(trainsize+validsize)]['path'].tolist()
    test_list = _df[(trainsize+validsize):(trainsize+validsize+testsize)]['path'].tolist()

    print(f"Train: {len(train_list)} \nVal: {len(valid_list)} \nTest: {len(test_list)}")
    return train_list, valid_list, test_list

def zstandard(arr):
    _mu = np.nanmean(arr)
    _std = np.nanstd(arr)
    return (arr - _mu)/_std

def cropx(xx, yy):
    # crop input and output to 400 by 400
    dim1 = xx.shape[0]
    dim2 = xx.shape[1]
    dim1res = int((dim1 - 400)/2)
    dim2res = int((dim2 - 400)/2)
    return xx[dim1res:dim1res+400, dim2res:dim2res+400, :], yy[dim1res:dim1res+400, dim2res:dim2res+400]

def get_distance(file):
    # print(file)
    # print(file.split("/")[-1].split("_")[3:6])
    timestamp, rlon, rlat = file.split("/")[-1].split("_")[3:6]
    rlon = float(rlon)
    rlat = float(rlat)
    # print(timestamp, rlon, rlat)
    data = nc.Dataset(file)
    lat = np.array(data['lat'])
    lon = np.array(data['lon'])
    
    
    rlat_index = np.unravel_index((np.abs(lat- rlat)).argmin(), lat.shape)
    rlon_index = np.unravel_index((np.abs(lon- rlon)).argmin(), lon.shape)
    # print(rlat_index, rlon_index, rlon, rlat)
    data.close()
    
    lon, lat = np.meshgrid(lon, lat)
    
    lon = lon*np.pi/180
    lat = lat*np.pi/180
    rlon = rlon*np.pi/180
    rlat = rlat*np.pi/180

    a = np.sin((lat-rlat)/2)**2 + np.cos(lat)*np.cos(rlat)*(np.sin((lon-rlon)/2)**2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = 6371e3*c/1000
    return d

def transform_func_12h(xx, _6xx, _12xx, yy, comb_plume, transform=''):
    '''
    xx: (400, 400, 14)
    yy: (400, 400)
    predlist: 'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'SHGT', 'T02M', 'ADS',
              'UWND', 'VWND', 'WWND', 'PRES', 'TEMP', 'AD'
    '''
    
    ###
    # typical mean values:
    # [ 3.77901167e-02 -7.10877708e-02  1.24484683e+00  2.56569862e+02
    #   9.80964342e+02  2.82531180e+02  2.88608260e+02  1.18414726e+00
    #   7.51264114e+00  9.86283611e-01  1.14933095e-02  8.46494663e+02
    #   2.86330624e+02  1.02993263e+02]
    #          'GPR', 'U10M', 'V10M', 'PBLH', 'PRSS', 'PRES', 'TEMP'
    SCALERS = [1,     1e1,     1e1,    1e-3,   1e-3,    1,      1]
    BIAS =    [  0,     0,       0,       0,      0,     0,     0]
    
    _xx, _yy = cropx(xx, yy)
    for i in range(7):
        _xx[:, :, i] = _xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    
    original_yy = _yy.copy()
    if transform=='log_square':
        _yy[np.where(_yy == 0)] = np.nan
        _yy = (np.log(_yy) + 30)**2
        _yy[np.where(np.isnan(_yy))] = 0
    elif transform=='log':
        _yy[np.where(_yy <= 1e-8)] = np.nan
        _yy = np.log(_yy) + 20
        _yy[np.where(np.isnan(_yy))] = 0
    elif transform=='multiply':
        _yy = _yy*1e5
    elif transform=="log-epsilon":
        epsilon = 1e-3
        _yy = np.log(_yy+epsilon)-np.log(epsilon)
        _yy = _yy*1000
    elif transform=="log-epsilon-threshold":
        epsilon = 1e-3
        _yy[np.where(_yy <= 1e-6)] = 0
        _yy = np.log(_yy+epsilon)-np.log(epsilon)
        _yy = _yy*1000
        
    _yy = _yy[:, :, np.newaxis]  # 400, 400, 1
    
    _xx[:, :, 0] = zstandard(_xx[:, :, 0])
    _xx[:, :, 5] = _xx[:, :, 5]/Rd/_xx[:, :, 6]*1e-1
    _xx = np.delete(_xx, [5, 6], axis=-1) # 400, 400, X

    
    _6xx, _ = cropx(_6xx, yy)
    for i in range(7):
        _6xx[:, :, i] = _6xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    
    
    _6xx[:, :, 0] = zstandard(_6xx[:, :, 0])
    _6xx[:, :, 5] = _6xx[:, :, 5]/Rd/_6xx[:, :, 6]*1e-1
    
    _6xx = np.delete(_6xx, [5, 6], axis=-1) # 400, 400, X
    # _6xx = np.delete(_6xx, [0], axis=-1) # 400, 400, X
    
    
    # _12xx, _ = cropx(_12xx, yy)
    # for i in range(7):
    #     _12xx[:, :, i] = _12xx[:, :, i]*SCALERS[i] #+ BIAS[i]
    
    # _12xx[:, :, 0] = zstandard(_12xx[:, :, 0])
    # _12xx[:, :, 5] = _12xx[:, :, 5]/Rd/_12xx[:, :, 6]*1e-1
    # _12xx = np.delete(_12xx, [6], axis=-1) # 400, 400, X
    # _12xx = np.delete(_12xx, [0], axis=-1) # 400, 400, X
    # # import pdb; pdb.set_trace()
    # comb_plume = np.array(comb_plume)[:, :, np.newaxis]
    # comb_plume[np.where(comb_plume>=0.08)] = 1
    # comb_plume[np.where(comb_plume<0.08)] = 0
    
    # return np.concatenate([_xx, _6xx, _12xx, comb_plume, _yy], axis=-1), original_yy
    return np.concatenate([_xx, _6xx, _yy], axis=-1), original_yy

class FootDataset(Dataset):
    def __init__(self, files, transform=None, limit=None, extension='.npz'):
        self.data_files = files
        self.limit = limit
        self.transform = transform
        self.extension = extension
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file = self.data_files[idx]
        # print(file)
        data = self.load_data(file)
        # print(data['pred'].shape)
        # dist = get_distance(file)
        # print(dist)
        tempxy, original_yy = transform_func_12h(data['_pred'], data['_6hpred'], data['_12hpred'], data['obs'], data['combined_gaussian_plume'], transform=self.transform)
        # import pdb; pdb.set_trace()
        # print(tempxy.shape)
        
        tempx = tempxy[:, :, :-1]
        tempxx = np.zeros((tempx.shape[2], tempx.shape[0], tempx.shape[1]))
        # print(tempxx.shape, dist.shape)
        # tempxx = np.concatenate([tempxx, dist[np.newaxis, :, :], np.exp(0.01*dist)[np.newaxis, :, :]], axis=0)
        # print("Updated:,", tempxx.shape)
        for idx in range(tempx.shape[2]):
            tempxx[idx, :, :] = tempx[:, :, idx]
        tempy = tempxy[:, :, -1]
        tempy = np.array(tempy[np.newaxis, :, :])
        original_yy = np.array(original_yy[np.newaxis, :, :])
        # tempy = tempy[:, :, np.newaxis]
        # print(type(tempxx), type(tempy), type(original_yy))
        if self.extension == ".nc" or self.extension == ".ncdf":
            data.close()
        return tempxx, tempy, original_yy, file
    
    def load_data(self, file):
        if self.extension == '.npz':
            return np.load(file)
        elif self.extension == ".nc" or self.extension == ".ncdf":
            return nc.Dataset(file)

    
def getR2(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def checkpoint(model, optimizer, filename):
    torch.save({
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
    }, filename)

def resume(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def inference(a, type='log-epsilon'):
    # a = -np.sqrt(a)
    if type=='log-epsilon':
        epsilon = 1e-3
        a = a/1000
        a = a + np.log(epsilon)
        a = np.exp(a)-epsilon
        a[np.where(a<0)] = 0
    elif type=='log':
        a[np.where(a==0)]=np.nan
        a = np.exp(a-20)
        a[np.isnan(a)] = 0
    else:
        raise Exception(f"inference type is not given.....(type={type})")
    return a

def write_emulated_footprint(foot, timestamp, receptor_lon, receptor_lat, path):
    file = f"{path}emulator_{timestamp}_{receptor_lon}_{receptor_lat}.nc"
    out_nc = nc.Dataset(file, "w", format="NETCDF4")
    out_nc.createDimension("lat", lats.shape[0])
    out_nc.createDimension("lon", lons.shape[0])
    out_nc.createDimension("info", 1)
    
    lat = out_nc.createVariable("lat", np.float32, ("lat",))
    lon = out_nc.createVariable("lon", np.float32, ("lon",))
    val = out_nc.createVariable("foot", np.float32, ("lat", "lon"))
    rlat = out_nc.createVariable("receptor_lat", np.float32, ("info"))
    rlon = out_nc.createVariable("receptor_lon", np.float32, ("info"))
    clon_y = out_nc.createVariable("clon", np.float32, ("info"))
    clat_y = out_nc.createVariable("clat", np.float32, ("info"))
    lat[:] = lats
    lon[:] = lons
    val[:, :] = foot
    rlat[:] = receptor_lat
    rlon[:] = receptor_lon
    clon_y[:] = clon
    clat_y[:] = clat
    out_nc.close()

def generate_footprints(test_DG):
    unet.eval()
    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_DG)):
            inputs, labels, _, files = data
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
            prediction = unet(inputs)
            # if idx == 0:
            #     labels_valid = labels.cpu()
            #     prediction_valid = prediction.cpu()
            #     # _valid = _.cpu()
            # else:
            #     labels_valid = torch.cat([labels_valid, labels.cpu()], axis=0)
            #     prediction_valid = torch.cat([prediction_valid, prediction.cpu()], axis=0)
                # _valid = torch.cat([_valid, _.cpu()])
            
            # a = transform(labels.cpu().detach().numpy())
            b = inference(prediction.cpu().detach().numpy(), type=transform)
            c = files
            for i in range(inputs.shape[0]):
                # file = f"{stilt_path}{files[i].split('/')[-1][4:]}"
                timestamp, receptor_lon, receptor_lat = files[i].split('/')[-1].split("_")[3:6]
                foot = b[i, 0]
                write_emulated_footprint(foot, timestamp, receptor_lon, receptor_lat, output_path)
                
            
experiment = EXP_NAME
location = f"PATH/{experiment}/"

output_path = PATH

transform = 'log'
batch_size = 8

learning_rate = 1e-4
weight_decay = 1e-4

print("Experiment:", experiment)
print("Save location:", location)
print("Transformation (Y):", transform)
print("Batchsize:", batch_size)
print("learning_rate:", learning_rate)
print("Weight decay:", weight_decay)


# Data loading
train_list = list(pd.read_csv("train_list_beacon_generalizable.csv")['path'])
valid_list = list(pd.read_csv("valid_list_beacon_generalizable.csv")['path'])
test_list = list(pd.read_csv("test_list_beacon_generalizable.csv")['path'])
test2020_list = list(pd.read_csv("Covid2020_beacon_generalizable.csv")['path'])


train_DG = DataLoader(FootDataset(train_list, transform=transform, extension='.nc'),  batch_size=batch_size,
                                              shuffle=True, num_workers=4, pin_memory=True)
valid_DG = DataLoader(FootDataset(valid_list, transform=transform, extension='.nc'),  batch_size=batch_size,
                                              shuffle=True, num_workers=4)
test_DG = DataLoader(FootDataset(test_list, transform=transform, extension='.nc'),  batch_size=batch_size,
                                              shuffle=True, num_workers=4)

test2020_DG = DataLoader(FootDataset(test2020_list, transform=transform, extension='.nc'),  batch_size=batch_size,
                                              shuffle=True, num_workers=8, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet = UNet(n_channels=10, n_classes=1)
# writer.add_graph(unet, data[0].type(torch.float32), verbose=True)
# unet = AttU_Net(img_ch=16)
# print("Model:", unet)
# unet = nn.DataParallel(unet, device_ids=[0,1])
unet = unet.to(device)
optimizer = optim.Adam(unet.parameters(), lr=learning_rate, weight_decay=weight_decay)
resume(unet, optimizer, PATH)
print(optimizer)
print(unet)
generate_footprints(test2020_DG)
