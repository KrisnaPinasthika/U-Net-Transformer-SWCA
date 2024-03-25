import torch 
import time as t
from model.UNet import UNet
from model.AttentionUNet import AttentionUNet 
from model.UNetTransformer import UNetTransformer
from model.PreTrainedUNetSWCA import PreTrainedUNetSWCA

device = torch.device('cuda')

def getModel(model):
    model = model.lower()
    
    if model == 'unet':
        model_dict = {
            'eff_b1': UNet(backbone='eff_b1'),
            'eff_b5': UNet(backbone='eff_b5'),
            'res_50': UNet(backbone='res_50'),
        }
    elif model == 'attention_unet': 
        model_dict = {
            'eff_b1': AttentionUNet(backbone='eff_b1'),
            'eff_b5': AttentionUNet(backbone='eff_b5'),
            'res_50': AttentionUNet(backbone='res_50'),
        }
    elif model == 'unet_transformer':
        model_dict = {
            'eff_b1': UNetTransformer(backbone='eff_b1'),
            'eff_b5': UNetTransformer(backbone='eff_b5'),
            'res_50': UNetTransformer(backbone='res_50'),
        }
    elif model == 'unet_transformer_swca':
        model_dict = {
            'eff_b1': PreTrainedUNetSWCA(
                device='cuda', 
                backbone_name='eff_b1', 
                bottleneck_head=8,
                window_sizes=[8, 8, 8, 8], 
                layers=[2, 2, 4, 4], 
                qkv_bias=False, 
                attn_drop_prob=0.2, 
                lin_drop_prob=0.1),
            'eff_b5': PreTrainedUNetSWCA(
                device='cuda', 
                backbone_name='eff_b5', 
                bottleneck_head=8,
                window_sizes=[8, 8, 8, 8], 
                layers=[2, 2, 4, 4], 
                qkv_bias=False, 
                attn_drop_prob=0.2, 
                lin_drop_prob=0.1),
            'res_50': PreTrainedUNetSWCA(
                device='cuda', 
                backbone_name='res_50', 
                bottleneck_head=8,
                window_sizes=[8, 8, 8, 8], 
                layers=[2, 2, 4, 4], 
                qkv_bias=False, 
                attn_drop_prob=0.2, 
                lin_drop_prob=0.1),
        }
        
    else:
        print('model tidak tersedia')
        return 0 
    
    return model_dict

models = getModel('unet_transformer_swca')

model_effb1, model_effb5, model_effres50 = models['eff_b1'], models['eff_b5'], models['res_50']

fps_counter = {}
inference_counter = {}

for model, name in zip([model_effb1, model_effb5, model_effres50], ['eff_b1', 'eff_b5', 'res_50']):
    print(name)
    model.to(device)
    random_sample = torch.randn((1, 3, 256, 256)).to(device)
    
    # Todo: Warm up for 10 sec
    print('WARMUP')
    warmup_start = t.time()
    while True: 
        pred = model(random_sample)
        
        if (t.time() - warmup_start) >= 10:
            break
    
    print('Inference time start')
    prev_frame_time = 0
    new_frame_time = 0

    fps_history, inference_history = [], []
    fps_inference_time_counter = t.time()
    while True: 
        inference_start = t.time()
        pred = model(random_sample)
        inference_end = t.time() - inference_start
        
        new_frame_time = t.time() 
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time 
        
        fps_history.append(fps)
        inference_history.append(inference_end)
        
        if (t.time() - fps_inference_time_counter) >= 60.:
            break
    
    fps_counter[name] = fps_history
    inference_counter[name] = inference_history

print('DONE')

print('FPS')
print('==='*20)
print(fps_counter)
print('==='*20)

print('Inference')
print('==='*20)
print(inference_counter)
print('==='*20)