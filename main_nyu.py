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
from model.UNet import UNet
from model.AttentionUNet import AttentionUNet
from model.UNetTransformer import UNetTransformer

"""
    TRAIN: 
    dense_169 | 32 | 2 2 2 2
    res_50 | 32 | 2 2 2 2
    
"""

if __name__ == '__main__': 
    # Todo: read the dataset
    BASE_PATH = "../datasets/nyu_data/data/"
    TRAIN_PATH = os.path.join(BASE_PATH, "nyu2_train")
    TEST_PATH = os.path.join(BASE_PATH, "nyu2_test")

    df_train = pd.read_csv(os.path.join(BASE_PATH, "nyu2_train.csv"), header=None, names=["image", "depth"])
    df_train["scene"] = df_train["image"].apply(lambda x: "_".join(x.split("/")[2].split("_")[:-2]))
    df_test = pd.read_csv(os.path.join(BASE_PATH, "nyu2_test.csv"), header=None, names=["image", "depth"])
    
    img_size = (480, 640)
    # img_size = (256, 256)
    epochs = 25
    batch_size = 16
    
    data_train = NYUDepth(df=df_train, img_size=img_size, data_path=r'../datasets/nyu_data/', is_test=False)
    data_test = NYUDepth(df=df_test, img_size=img_size, data_path=r'../datasets/nyu_data/', is_test=True)
    
    trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    testloader = DataLoader(data_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    
    # Todo: model creation
    device = 'cuda'
    
    bottleneck_head = 64
    backbone_name = 'eff_b5'
    layers = [2, 2, 4, 4]
    model_name = f"PreTrained_{backbone_name}_{bottleneck_head}_window5_5_5_5_layers{layers[0]}{layers[2]}_NYU"
    # model_name = f"PreTrained_{backbone_name}_{bottleneck_head}_UNetTransformer_NYU"
    # model_name = f"PreTrained_{backbone_name}_AttentionUNet_NYU"
    # model = AttentionUNet(backbone=backbone_name).to('cuda')
    
    # attention_type = 'weighted'
    
    model = PreTrainedUNetSWCA(
        device='cuda', 
        backbone_name=backbone_name, 
        bottleneck_head=bottleneck_head,
        # window_sizes=[8, 8, 8, 8], 
        window_sizes=[5, 5, 5, 5], 
        # window_sizes=[20, 10, 10, 5], 
        layers=layers, 
        qkv_bias=False, 
        attn_drop_prob=0.2, 
        lin_drop_prob=0.1, 
    ).to('cuda')
    
#     model = VanillaUNetSWCA(
#         device='cuda', 
#         bottleneck_head=bottleneck_head,
#         window_sizes=[8, 8, 8, 8], 
#         layers=layers, 
#         qkv_bias=False, 
#         attn_drop_prob=0.2, 
#         lin_drop_prob=0.1).to('cuda')


    # backbone_name = 'eff_b5'
    # for model, model_name in ( [UNet(backbone=backbone_name), f"UNet_{backbone_name}_NYU" ], 
    #                           [AttentionUNet(backbone=backbone_name), f"AttentionUNet_{backbone_name}_NYU"]):
    # model = UNetTransformer(backbone_name)
    # model.to(device)
    # model_name = "UNetTransformer_25epoch_NYU"

    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=.5, verbose=False)

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