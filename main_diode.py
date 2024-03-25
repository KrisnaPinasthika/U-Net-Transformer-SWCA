import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Loader.DiodeLoader import DiodeDataLoader
from TrainTest.DiodeTrainTest import train
from model.VanillaUNetSWCA import VanillaUNetSWCA
from model.PreTrainedUNetSWCA import PreTrainedUNetSWCA
from model.UNet import UNet
from model.AttentionUNet import AttentionUNet
from model.UNetTransformer import UNetTransformer

torch.random.manual_seed(64)
np.random.seed(64)

def getData(path):
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    filelist.sort()
    data = {
        "image": [x for x in filelist if x.endswith(".png")],
        "depth": [x for x in filelist if x.endswith("_depth.npy")],
        "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
    }

    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    tipe_dataset = 'I'
    
    if tipe_dataset.upper() == 'O':
        TRAIN_PATH = r"../datasets/diode/train/outdoor/"
        TEST_PATH = r"../datasets/diode/val/outdoor/"
        max_depth = 75.
        backbone_name = 'res_50'
    elif tipe_dataset.upper() == 'I':
        TRAIN_PATH = r"../datasets/diode/train/indoors/"
        TEST_PATH = r"../datasets/diode/val/indoors/"
        max_depth = 50.
        backbone_name = 'eff_b1'

    df_train = getData(TRAIN_PATH)
    df_test = getData(TEST_PATH)

    dim = (256, 256)
    batch_size = 16

    data_train = DiodeDataLoader(data_frame=df_train, max_depth=max_depth, img_depth_dim=dim,is_test=False)
    data_test = DiodeDataLoader(data_frame=df_test, max_depth=max_depth, img_depth_dim=dim, is_test=True)

    trainloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda")
    epochs = 50

    model = UNetTransformer(backbone=backbone_name)
    model.to(device)
    
    model_name = f"UNetTransformer_{backbone_name}_DIODE_{tipe_dataset}"

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=.5)

    ssim_configuration = {
        'max_val' : max_depth / 0.1, 
        'kernel_size' : 7, 
        'k1' : 0.01, 
        'k2' : 0.03        
    }

    train(
        model=model,
        model_name=model_name,
        l1loss_weight=1., 
        max_depth=max_depth,
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

    torch.cuda.empty_cache()

# Pelatihan Vanilla Model
    # layers = 
#     for layers in [[2, 2, 2, 2], [2, 2, 4, 4], [4, 4, 4, 4]]:
#         layer_str = f"{layers[0]}{layers[2]}"
#         for bottleneck_head in [8, 16, 32]:
#             model = VanillaUNetSWCA(
#                 device=device, 
#                 bottleneck_head=bottleneck_head,
#                 window_sizes=[8, 8, 8, 8], 
#                 layers=layers, 
#                 qkv_bias=False, 
#                 attn_drop_prob=0.4, 
#                 lin_drop_prob=0.2,
#                 use_sigmoid=False
#             ).to(device)


#             model.load_state_dict(
#                 torch.load(f'SavedModel/1.2. PengaruhHead/VanillaUNet_{bottleneck_head}_window8_layers{layer_str}_NYU_25_epoch.pt')
#             )

#             model_name = f"VanillaUNet_{bottleneck_head}_window8_layers{layer_str}_{tipe_dataset}"

#             optimizer = optim.Adam(model.parameters(), lr=1e-4)
#             scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=.5)

#             ssim_configuration = {
#                 'max_val' : max_depth / 0.1, 
#                 'kernel_size' : 7, 
#                 'k1' : 0.01, 
#                 'k2' : 0.03        
#             }

#             train(
#                 model=model,
#                 model_name=model_name,
#                 l1loss_weight=1., 
#                 max_depth=max_depth,
#                 loader=trainloader,
#                 testloader=testloader,
#                 ssim_configuration=ssim_configuration, 
#                 epochs=epochs, 
#                 optimizer=optimizer, 
#                 scheduler=scheduler, 
#                 device=device, 
#                 save_model=False, 
#                 save_train_state=False
#             )

#             torch.cuda.empty_cache()

    
# Pelatihan Pre-trained model
#     for backbone_name in ['res_34', 'res_50', 'res_101', 'res_18']:
#     for backbone_name in ['dense_201', 'dense_169', 'dense_121']:
#     for backbone_name in ['eff_b6', 'eff_b5', 'eff_b3', 'eff_b1']:
 
#         model = PreTrainedUNetSWCA(
#             device='cuda', 
#             backbone_name=backbone_name, 
#             bottleneck_head=bottleneck_head,
#             window_sizes=[8, 8, 8, 8], 
#             layers=[2, 2, 4, 4], 
#             qkv_bias=False, 
#             attn_drop_prob=0.6, 
#             lin_drop_prob=0.2, 
#         ).to('cuda')

#         model.load_state_dict(
#             torch.load(f'SavedModel/2.PengaruhPretrained/PreTrained_{backbone_name}_8_window8_layers24_NYU_25_epoch.pt')
#         )

#         model_name = f"PreTrained_{backbone_name}_8_window8_layers24_{tipe_dataset}_lambda1"

#         optimizer = optim.Adam(model.parameters(), lr=1e-4)
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=.5)

#         ssim_configuration = {
#             'max_val' : max_depth / 0.1, 
#             'kernel_size' : 7, 
#             'k1' : 0.01, 
#             'k2' : 0.03        
#         }


#         train(
#             model=model,
#             model_name=model_name,
#             l1loss_weight=1., 
#             max_depth=max_depth,
#             loader=trainloader,
#             testloader=testloader,
#             ssim_configuration=ssim_configuration, 
#             epochs=epochs, 
#             optimizer=optimizer, 
#             scheduler=scheduler, 
#             device=device, 
#             save_model=True, 
#             save_train_state=False
#         )

#         torch.cuda.empty_cache()