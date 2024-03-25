import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os 
import pandas as pd
from Loader.NYULoader import NYUDepth
from torch.utils.data import DataLoader
from TrainTest.NYUTrainTest import test, testScaledMedian
from model.VanillaUNetSWCA import VanillaUNetSWCA
from model.PreTrainedUNetSWCA import PreTrainedUNetSWCA

from model.UNet import UNet
from model.AttentionUNet import AttentionUNet
from model.UNetTransformer import UNetTransformer


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
    batch_size = 28
    
    data_test = NYUDepth(df=df_test, img_size=img_size, data_path=r'../datasets/nyu_data/', is_test=True)
    testloader = DataLoader(data_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    
    
    # Todo: model creation
    device = 'cuda'
    ssim_configuration = {
        'max_val' : 1000. / 10., 
        'kernel_size' : 7, 
        'k1' : 0.01, 
        'k2' : 0.03        
    }
    
    backbone = 'eff_b5'

    # model = UNetTransformer(backbone).to(device)
    # model = UNet(backbone).to(device)
    # model = AttentionUNet(backbone).to(device)
#     model.load_state_dict(
#         torch.load(r'./SavedModel/PreTrained_eff_b5_UNet_NYU_25_epoch.pt')
#     )
    
#     test(model, 
#          loader=testloader, 
#          l1loss_weight=0.1, 
#          min_depth=10.,
#          max_depth=1000., 
#          ssim_configuration=ssim_configuration, 
#          device=device
#     )
    
#     testScaledMedian(model, 
#         loader=testloader, 
#         l1loss_weight=0.1, 
#         min_depth=10.,
#         max_depth=1000., 
#         ssim_configuration=ssim_configuration, 
#         device=device
#     )

    count = 0
    for window_sizes, bottleneck_head in zip([[5, 5, 5, 5],  [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], 
                                                 [10, 10, 10, 5], [10, 10, 10, 5], [10, 10, 10, 5], [10, 10, 10, 5], 
                                                 [20, 10, 10, 5], [20, 10, 10, 5], [20, 10, 10, 5], [20, 10, 10, 5],], 
                                             
                                                [8, 32, 64, 128, 8, 32, 64, 128, 8, 32, 64, 128, 8, 32, 64, 128] 
                                            ):
        print(f"Windows : {window_sizes}, Bottleneck Heads : {bottleneck_head}")
        model = PreTrainedUNetSWCA(
            device='cuda', 
            backbone_name=backbone, 
            bottleneck_head=bottleneck_head,
            window_sizes=window_sizes, 
            layers=[2, 2, 4, 4], 
            qkv_bias=False, 
            attn_drop_prob=0.2, 
            lin_drop_prob=0.1, 
        ).to('cuda')
        
        window_configs = f"window{window_sizes[0]}_{window_sizes[1]}_{window_sizes[2]}_{window_sizes[3]}"
        model.load_state_dict(
            torch.load(f"./SavedModel/PreTrained_{backbone}_{bottleneck_head}_{window_configs}_layers24_NYU_25_epoch.pt")
        )


        test(model, 
             loader=testloader, 
             l1loss_weight=0.1, 
             min_depth=10.,
             max_depth=1000., 
             ssim_configuration=ssim_configuration, 
             device=device
        )

        testScaledMedian(model, 
            loader=testloader, 
            l1loss_weight=0.1, 
            min_depth=10.,
            max_depth=1000., 
            ssim_configuration=ssim_configuration, 
            device=device
        )
        
        count += 1
        if count % 4 == 0:
            print()