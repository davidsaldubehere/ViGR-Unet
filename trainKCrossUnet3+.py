import torch
import torch.nn as nn

from torchvision.transforms import ToTensor,Compose,Resize,ToPILImage,PILToTensor,RandomRotation
from torch.utils.data import DataLoader
import torch.optim as optim

from sklearn.model_selection import KFold
from CamusEDImageDataset import CamusEDImageDataset
from Unet3Plus import UNet3Plus

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from FocalLoss import FocalLoss


from tqdm import tqdm

from torchmetrics.functional import dice
import glob
import config
import os
from datasets import load_metric


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NB_EPOCHS = 40
    NUM_FOLDS = 5

    #Use a "light" version if True (3.7M params) or the paper version if False (31M params)
    lightUnet = False
    
    imagePaths = sorted(glob.glob(os.path.join(config.IMAGE_DATASET_PATH, "*.nii")))
    maskPaths = sorted(glob.glob(os.path.join(config.MASK_DATASET_PATH, "*.nii")))
    NBSAMPLES = len(imagePaths)

    # Create dataset
    dataset = CamusEDImageDataset(
        imagePaths=imagePaths,
        maskPaths=maskPaths,
        transform=Compose([ToPILImage(),Resize((256,256)),ToTensor()]),
    )
    
    # Define the K-fold cross-validator
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # List to store metrics for each fold
    fold_train_losses = []
    fold_val_losses = []
    fold_dice_scores = []
    fold_iou_scores = []
    
    # Initialize IoU metric
    
    
    # K-fold Cross Validation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(np.arange(NBSAMPLES))):
        print(f'FOLD {fold+1}/{NUM_FOLDS}')
        print('-' * 50)
        
        # Create training and validation datasets for this fold
        train_data = torch.utils.data.Subset(dataset, train_ids)
        valid_data = torch.utils.data.Subset(dataset, val_ids)
        
        # Create data loaders
        train_dataloader = DataLoader(train_data, batch_size=4)
        valid_dataloader = DataLoader(valid_data, batch_size=4)
        
        # Initialize the model, optimizer, and loss function
        net = UNet3Plus().to(device)
        
        if fold == 0:
            total_params = sum(p.numel() for p in net.parameters())
            print(f'Total parameters: {total_params}')
        
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Lists to store metrics for this fold
        lossEvolve = []
        valEvolve = []
        diceEvolve = []
        
        # Variables to track best model
        best_val_loss = float('inf')
        best_dice_score = 0.0
        best_iou_score = 0.0
        
        # Dictionary to store per-class dice scores
        class_dices = {i:[] for i in range(4)}
        
        # Training loop
        for epoch in tqdm(range(NB_EPOCHS)):
            net.train()
            print(f'FOLD {fold+1}/{NUM_FOLDS} - EPOCH: {epoch+1}/{NB_EPOCHS}')
            
            # Train
            train_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                final = net(inputs)
                loss = criterion(final, labels.type(torch.LongTensor).to(device))
                                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            net.eval()
            val_loss = 0.0
            dice_curr = 0.0
            metric = load_metric("mean_iou", cache_dir = "/scratch/das6859/cache")
            epoch_class_dices = {i:[] for i in range(4)}  # Store per-class dice for this epoch
            
            with torch.no_grad():
                for j, data in enumerate(valid_dataloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.int()
                    
                    final=  net(inputs)
                    loss = criterion(final, labels.type(torch.LongTensor).to(device))
                    
                    val_loss += loss.item()
                    
                    # Calculate overall dice score
                    dice_curr += dice(final, labels, average="micro", ignore_index=0)
                    
                    # Calculate per-class dice scores
                    dice_per_class = dice(final.detach().cpu(), labels.detach().cpu(), average=None, num_classes=4)
                    for i in range(len(dice_per_class)):
                        epoch_class_dices[i].append(dice_per_class[i].item())
                    
                    # Add batch to IoU metric
                    metric.add_batch(
                        predictions=final.argmax(dim=1).detach().cpu().numpy(), 
                        references=labels.type(torch.LongTensor).detach().cpu().numpy()
                    )
            
            # Compute IoU metrics
            iou_metrics = metric.compute(
                num_labels=4,
                ignore_index=0,
                reduce_labels=False,
            )
            
            # Calculate mean IoU (excluding background class)
            mean_iou = np.mean(iou_metrics["per_category_iou"][1:])
            
            # Calculate average metrics for this epoch
            avg_train_loss = train_loss / (i+1)
            avg_val_loss = val_loss / (j+1)
            avg_dice_score = dice_curr / (j+1)
            
            # Calculate average per-class dice scores for this epoch
            avg_class_dices = {i: np.mean(epoch_class_dices[i]) for i in range(4)}
            
            # Record metrics for this epoch
            lossEvolve.append(avg_train_loss)
            valEvolve.append(avg_val_loss)
            diceEvolve.append(avg_dice_score.cpu())
            
            # Update class dices dictionary for tracking
            for i in range(4):
                if i not in class_dices:
                    class_dices[i] = []
                class_dices[i].append(avg_class_dices[i])
            
            print(f"Training Loss: {avg_train_loss:.4f} \tValid Loss: {avg_val_loss:.4f} \tDice: {avg_dice_score:.4f} \tMean IoU: {mean_iou:.4f}")
            print("Per-class IoU:", iou_metrics["per_category_iou"])
            print("Per-class dice:", [avg_class_dices[i] for i in range(4)])
            
            # Check if this is the best model so far (based on validation loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_dice_score = avg_dice_score
                best_iou_score = mean_iou
                # Save the best model
                print(f"Saving best model with validation loss: {best_val_loss:.4f}")
                torch.save(net.state_dict(), f'/scratch/das6859/UNetVGNNCAMUS/weights/unet3+ES_fold{fold+1}_best.pt')
        # Store the best metrics for this fold
        fold_train_losses.append(min(lossEvolve))
        fold_val_losses.append(best_val_loss)
        fold_dice_scores.append(best_dice_score.cpu())
        fold_iou_scores.append(best_iou_score)
        
        # Print summary of best metrics for this fold
        print(f"Fold {fold+1} best validation loss: {best_val_loss:.4f}")
        print(f"Fold {fold+1} best dice score: {best_dice_score:.4f}")
        print(f"Fold {fold+1} best mean IoU: {best_iou_score:.4f}")
        
        # Print per-class dice scores for best model in this fold
        print("Per-class dice scores:")
        for i in range(4):
            best_idx = valEvolve.index(best_val_loss)
            print(f"Class {i}: {class_dices[i][best_idx]:.4f}")
    
    # Print final performance across all folds
    print('\nK-FOLD CROSS VALIDATION RESULTS')
    print('--------------------------------')
    print(f'Best Average Training Loss: {np.mean(fold_train_losses):.4f} ± {np.std(fold_train_losses):.4f}')
    print(f'Best Average Validation Loss: {np.mean(fold_val_losses):.4f} ± {np.std(fold_val_losses):.4f}')
    print(f'Best Average Dice Score: {np.mean(fold_dice_scores):.4f} ± {np.std(fold_dice_scores):.4f}')
    print(f'Best Average Mean IoU: {np.mean(fold_iou_scores):.4f} ± {np.std(fold_iou_scores):.4f}')
    
    print('Finished Training')
