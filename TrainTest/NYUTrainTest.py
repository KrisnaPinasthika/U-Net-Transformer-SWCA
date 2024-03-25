import pandas as pd
from datetime import datetime
import time
import numpy as np
import torch
from .Metrics import rmse, rel, accuracy_th
from .Loss import loss_l1, loss_depthsmoothness, loss_ssim, loss_si

def depthNorm(x, max_depth): 
    return max_depth / x 

def getMetrics(y_true, y_pred):
    rmse_val = rmse(y_true, y_pred)
    rel_val = rel(y_true, y_pred)
    acc_1 = accuracy_th(y_true, y_pred, 1)
    acc_2 = accuracy_th(y_true, y_pred, 2)
    acc_3 = accuracy_th(y_true, y_pred, 3)
    return rmse_val, rel_val, acc_1, acc_2, acc_3

def save_state(save_state_path, model, optimizer, epoch, loss):
    torch.save(
    {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, 
    save_state_path)
    print(f"<-- Model state saved [{epoch} epoch] -->")

def load_state(save_state_path, model, optimizer):
    checkpoint = torch.load(save_state_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"<-- Model state loaded succesfully [Last epoch = {checkpoint['epoch']}] -->")
    return model, optimizer

def train(model, model_name, min_depth, max_depth, l1loss_weight, loader, testloader, ssim_configuration, epochs,
    optimizer, scheduler, device, save_model=False, save_train_state=False):
    current_date_name = datetime.now().strftime(r"%d-%m-%y")
    model.train()
    total_batch = loader.__len__()
    print(f"Training [Total batch : {total_batch}] [Model: {model_name}]")

    # Todo: create pandas dataframe
    hist_rmse, hist_rel, hist_acc_1, hist_acc_2, hist_acc_3 = [], [], [], [], []
    best_loss, best_acc1 = 99999, 0.

    start = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        running_loss = 0.0
        running_rmse, running_rel, running_acc_1, running_acc_2, running_acc_3 = 0.0, 0.0, 0.0, 0.0, 0.0
        

        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Todo: forward + backward + optimize
            outputs = model(inputs)

            # Todo: calculate loss
            loss_1 = loss_l1(labels, outputs)
            loss_2 = loss_depthsmoothness(labels, outputs)
            loss_3 = torch.clamp(
                loss_ssim(
                    labels, 
                    outputs, 
                    max_val=ssim_configuration['max_val'], 
                    kernel_size=ssim_configuration['kernel_size'], 
                    k1=ssim_configuration['k1'], 
                    k2=ssim_configuration['k2']), 0., 1.)
            loss = l1loss_weight * loss_1 + loss_2 + loss_3
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
                running_loss += loss.item()
                metrics = getMetrics(labels, outputs)
                running_rmse += metrics[0]
                running_rel += metrics[1]
                running_acc_1 += metrics[2]
                running_acc_2 += metrics[3]
                running_acc_3 += metrics[4]

                if ((i + 1) % int(total_batch // 3) == 0) or ((i + 1) == total_batch) or ((i+1)%50 == 0):
                    print(f"  Batch[{i+1}/{total_batch}] \t Loss : {(running_loss / (i + 1)):.4f} RMSE : {running_rmse  / (i + 1):.4f} REL : {running_rel  / (i + 1):.4f} ACC^1 : {running_acc_1  / (i + 1):.4f} ACC^2 : {running_acc_2  / (i + 1):.4f} ACC^3 : {running_acc_3  / (i + 1):.4f}")

        print(f"  --> Epoch {epoch + 1} Total training time : {(time.time() - start):.2f} Second")
        test_loss, test_acc1 = test(model=model, loader=testloader, l1loss_weight=l1loss_weight, min_depth=min_depth, max_depth=max_depth, ssim_configuration=ssim_configuration, device=device)
        condition = (test_loss < best_loss) and (test_acc1 > best_acc1)
        
        if (save_model and condition):
            best_loss = test_loss
            best_acc1 = test_acc1
            print(f"  --> Saved Best Loss : {best_loss:.4f}, Acc1 : {best_acc1:.4f}")
            torch.save(model.state_dict(), f'./SavedModel/{model_name}_{epochs}_epoch_{current_date_name}.pt')
        
        # if save_model and ((epoch+1) % 5 == 0):
        #     print(f"  --> Saved Kelipatan {epoch+1} Loss : {best_loss:.4f}, Acc1 : {best_acc1:.4f}")
        #     torch.save(model.state_dict(), f'./SavedModel/{model_name}_{epoch+1}_epoch_kelipatan_{current_date_name}.pt')
        
        # Todo: save artefact history and model
        hist_rmse.append(running_rmse.cpu().numpy() / total_batch)
        hist_rel.append(running_rel.cpu().numpy() / total_batch)
        hist_acc_1.append(running_acc_1.cpu().numpy() / total_batch)
        hist_acc_2.append(running_acc_2.cpu().numpy() / total_batch)
        hist_acc_3.append(running_acc_3.cpu().numpy() / total_batch)

        scheduler.step()
        print()
        
    print(f"Total training time : {(time.time() - start):.2f} Second")

    # Todo: save loss and metrics during training 
    train_df = pd.DataFrame()
    train_df["rmse"] = hist_rmse
    train_df["rel"] = hist_rel
    train_df["acc_1"] = hist_acc_1
    train_df["acc_2"] = hist_acc_2
    train_df["acc_3"] = hist_acc_3
    train_df.to_csv(f'./TrainingHistory/train_{model_name}_{epochs}_epoch_{current_date_name}.csv')
    
    if save_train_state:
        save_state(
            save_state_path=f'./SavedTrainingState/{model_name}_{current_date_name}', 
            model=model, 
            optimizer=optimizer, 
            epoch=epochs, 
            loss=(running_loss/total_batch)
        )

def test(model, loader, l1loss_weight, min_depth, max_depth, ssim_configuration, device):
    model.eval()
    total_batch = loader.__len__()
    print(f"Testing Phase [Total batch : {total_batch}]")

    running_loss = 0.0
    running_rmse, running_rel, running_acc_1, running_acc_2, running_acc_3 = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Todo: forward
            outputs = model(inputs) # output model dalam cm dan terbalik
            outputs = torch.clamp(depthNorm(outputs, max_depth=max_depth), min_depth, max_depth) / max_depth # dijadikan rentang [0.01 - 1]
            outputs = outputs * 10 # dijadikan dalam rentang [0.1, 10] meter

            # Todo: calculate loss
            loss_1 = loss_l1(labels, outputs)
            loss_2 = loss_depthsmoothness(labels, outputs)
            loss_3 = torch.clamp(
                loss_ssim(
                    labels, 
                    outputs, 
                    max_val=ssim_configuration['max_val'], 
                    kernel_size=ssim_configuration['kernel_size'], 
                    k1=ssim_configuration['k1'], 
                    k2=ssim_configuration['k2']), 0., 1.)
            loss = l1loss_weight * loss_1 + loss_2 + loss_3
            

            running_loss += loss.item()
            metrics = getMetrics(labels, outputs)
            running_rmse += metrics[0]
            running_rel += metrics[1]
            running_acc_1 += metrics[2]
            running_acc_2 += metrics[3]
            running_acc_3 += metrics[4]

    out_loss = running_loss/total_batch
    out_rmse = running_rmse/total_batch
    out_rel = running_rel/total_batch
    out_acc1 = running_acc_1 /total_batch
    out_acc2 = running_acc_2 /total_batch
    out_acc3 = running_acc_3 /total_batch
    
    # print(f"  Batch[{i+1}/{i+1}] \t Loss : {(out_loss):.4f} RMSE : {out_rmse:.4f} REL : {out_rel:.4f} ACC^1 : {out_acc1:.4f} ACC^2 : {out_acc2:.4f} ACC^3 : {out_acc3:.4f}")
    print(f"{(out_loss):.4f}\t{out_rmse:.4f}\t{out_rel:.4f}\t{out_acc1:.4f}\t{out_acc2:.4f}\t{out_acc3:.4f}")
    
    return out_loss, out_acc1


def testScaledMedian(model, loader, l1loss_weight, min_depth, max_depth, ssim_configuration, device):
    model.eval()
    total_batch = loader.__len__()
    # print(f"Scaled Testing Phase [Total batch : {total_batch}]")

    running_loss = 0.0
    running_rmse, running_rel, running_acc_1, running_acc_2, running_acc_3 = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Todo: forward
            outputs = model(inputs) # output model dalam cm dan terbalik
            outputs = torch.clamp(depthNorm(outputs, max_depth=max_depth), min_depth, max_depth) / max_depth # dijadikan rentang [0.01 - 1]
            outputs = outputs * 10 # dijadikan dalam rentang [0.1, 10] meter
                
            median_gt = labels.flatten(start_dim=1).median(dim=-1, keepdim=True).values.unsqueeze(2).unsqueeze(3)
            median_pred  = outputs.flatten(start_dim=1).median(dim=-1, keepdim=True).values.unsqueeze(2).unsqueeze(3)
            s = median_gt / median_pred
            outputs = outputs * s

            # Todo: calculate loss
            loss_1 = loss_l1(labels, outputs)
            loss_2 = loss_depthsmoothness(labels, outputs)
            max_val, kernel_size, k1, k2 = ssim_configuration['max_val'], ssim_configuration['kernel_size'], ssim_configuration['k1'], ssim_configuration['k2']
            loss_3 = torch.clamp(loss_ssim(labels, outputs, max_val=max_val, kernel_size=kernel_size, k1=k1, k2=k2), 0., 1.)
            loss = l1loss_weight * loss_1 + loss_2 + loss_3

            running_loss += loss.item()
            metrics = getMetrics(labels, outputs)
            running_rmse += metrics[0]
            running_rel += metrics[1]
            running_acc_1 += metrics[2]
            running_acc_2 += metrics[3]
            running_acc_3 += metrics[4]

    out_loss = running_loss/total_batch
    out_rmse = running_rmse/total_batch
    out_rel = running_rel/total_batch
    out_acc1 = running_acc_1 /total_batch
    out_acc2 = running_acc_2 /total_batch
    out_acc3 = running_acc_3 /total_batch
    
    # print(f"  Batch[{i+1}/{i+1}] \t Loss : {(out_loss):.4f} RMSE : {out_rmse:.4f} REL : {out_rel:.4f} ACC^1 : {out_acc1:.4f} ACC^2 : {out_acc2:.4f} ACC^3 : {out_acc3:.4f}")
    
    print(f"{(out_loss):.4f}\t{out_rmse:.4f}\t{out_rel:.4f}\t{out_acc1:.4f}\t{out_acc2:.4f}\t{out_acc3:.4f}")
    
    return out_loss, out_acc1