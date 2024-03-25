import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from Loader.DiodeLoader import DiodeDataLoader
from TrainTest.DiodeTrainTest import train, test
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
    print(f"tipe_dataset : {tipe_dataset}")
    if tipe_dataset.upper() == 'O':
        TEST_PATH = r"../datasets/diode/val/outdoor/"
        max_depth = 75.
        backbone_name = 'res_50'
    elif tipe_dataset.upper() == 'I':
        TEST_PATH = r"../datasets/diode/val/indoors/"
        max_depth = 50.
        backbone_name = 'eff_b1'
        
    device = torch.device("cuda")
    dim = (256, 256)
    batch_size = 1
    
    df_test = getData(TEST_PATH)
    data_test = DiodeDataLoader(data_frame=df_test, max_depth=max_depth, img_depth_dim=dim, is_test=True)
    testloader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    ## --------------------------- ##
    ## Pengujian Varian U-Net      ##
    ## --------------------------- ##
    unet = UNet(backbone_name)
    attn_unet = AttentionUNet(backbone_name)
    unet_trans = UNetTransformer(backbone_name)
    models = [unet_trans, unet, attn_unet, ]
    model_names = ['UNetTransformer', 'UNet', 'AttentionUNet', ]
    
    for model, model_name in zip(models, model_names):
        model.to(device)

        model.load_state_dict(
            torch.load(
                os.path.join("SavedModel/4.PengujianUNetVarian/",
                             f"{model_name}_{backbone_name}_DIODE_{tipe_dataset}_50_epoch.pt")
            )
        )

        print(f"{model_name}_{backbone_name}")

        ssim_configuration = {
            'max_val' : max_depth / 0.1, 
            'kernel_size' : 7, 
            'k1' : 0.01, 
            'k2' : 0.03        
        }
        test(
            model, 
            loader=testloader, 
            max_depth=max_depth, 
            l1loss_weight=1.0, 
            ssim_configuration=ssim_configuration, 
            device=device
        )

        torch.cuda.empty_cache()

        print()
    
    ## --------------------------- ##
    ## Pengujian Vanilla model ##
    ## --------------------------- ##
    
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
#                 torch.load(os.path.join(SAVEDMODEL_PATH, f"VanillaUNet_{bottleneck_head}_window8_layers{layer_str}_{tipe_dataset}_50_epoch.pt"))
#             )

#             print(f"VanillaU-Net_{layer_str}_{bottleneck_head}")

#             ssim_configuration = {
#                 'max_val' : max_depth / 0.1, 
#                 'kernel_size' : 7, 
#                 'k1' : 0.01, 
#                 'k2' : 0.03        
#             }
#             test(
#                 model, 
#                 loader=testloader, 
#                 max_depth=max_depth, 
#                 l1loss_weight=1.0, 
#                 ssim_configuration=ssim_configuration, 
#                 device=device
#             )

#             torch.cuda.empty_cache()

#             print()
        
    
    ## --------------------------- ##
    ## Pengujian Pre-trained model ##
    ## --------------------------- ##
    
#     for backbone_name in ['res_18', 'res_34', 'res_50', 'res_101',
#                              'dense_121', 'dense_201', 'dense_169', 
#                              'eff_b1', 'eff_b3', 'eff_b5', 'eff_b6']:
#         bottleneck_head = 8
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
#             torch.load(os.path.join(SAVEDMODEL_PATH, f"PreTrained_{backbone_name}_8_window8_layers24_{tipe_dataset}_50_epoch.pt"))
#         )

#         print(f"PreTrained_{backbone_name}")

#         ssim_configuration = {
#             'max_val' : max_depth / 0.1, 
#             'kernel_size' : 7, 
#             'k1' : 0.01, 
#             'k2' : 0.03        
#         }
#         test(
#             model, 
#             loader=testloader, 
#             max_depth=max_depth, 
#             l1loss_weight=1.0, 
#             ssim_configuration=ssim_configuration, 
#             device=device
#         )

#         torch.cuda.empty_cache()
        
#         print()