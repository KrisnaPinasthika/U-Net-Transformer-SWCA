import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 
import pandas as pd
from Loader.NYULoader import NYUDepth
from torch.utils.data import DataLoader
from TrainTest.NYUTrainTest import train

from model.VanillaUNetSWCA import VanillaUNetSWCA
from model.PreTrainedUNetSWCA import PreTrainedUNetSWCA
from model.UNetMGA import UNetMGA
# from model.UNet import UNet
# from model.AttentionUNet import AttentionUNet


if __name__ == '__main__': 
    # Todo: read the dataset
    BASE_PATH = "../datasets/nyu_data/data/"
    TRAIN_PATH = os.path.join(BASE_PATH, "nyu2_train")
    TEST_PATH = os.path.join(BASE_PATH, "nyu2_test")

    df_train = pd.read_csv(os.path.join(BASE_PATH, "nyu2_train.csv"), header=None, names=["image", "depth"])
    df_train["scene"] = df_train["image"].apply(lambda x: "_".join(x.split("/")[2].split("_")[:-2]))
    df_test = pd.read_csv(os.path.join(BASE_PATH, "nyu2_test.csv"), header=None, names=["image", "depth"])
    
    img_size = (256, 256)
    epochs = 25
    batch_size = 32
    
    data_train = NYUDepth(df=df_train, img_size=img_size, data_path=r'../datasets/nyu_data/', is_test=False)
    data_test = NYUDepth(df=df_test, img_size=img_size, data_path=r'../datasets/nyu_data/', is_test=True)
    
    trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    testloader = DataLoader(data_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    
    # Todo: model creation
    device = 'cuda'
    backbone_name = 'eff_b5'
    attention_type = 'normal'
    model_name = f"UNetMGA_{backbone_name}_{attention_type}"
    
    model = UNetMGA(
        backbone=backbone_name,
        attention_type=attention_type, 
        device=device
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=.5)
    
    ssim_configuration = {
        'max_val' : 1000. / 10., 
        'kernel_size' : 7, 
        'k1' : 0.01, 
        'k2' : 0.03        
    }
    
    train(
        model=model,
        model_name=model_name, 
        min_depth=10,
        max_depth=1000, # centimeter 
        l1loss_weight=0.1, 
        loader=trainloader,
        testloader=testloader,
        ssim_configuration=ssim_configuration, 
        epochs=epochs, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        device=device, 
        save_model=True, 
        save_train_state=False
    )