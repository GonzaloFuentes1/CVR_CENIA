# Imports
import os
import csv
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset

from GAMR_model import MAREO
from torchvision import datasets, transforms
import multiprocessing as mp

# Training function
def train_MAREO(
    run,
    model,
    model_name,
    device,
    optimizer,
    scheduler,
    epochs,
    train_loader,
    val_loader,
    current_dir
    ):
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    # Create filename for saving training progress
    train_file = f"{current_dir}/model_{model_name}_run_{run}_train.csv"
    header = [
        'Model', 
        'Run', 
        'Epoch',
        'Loss',
        'Accuracy',
        'Val loss',
        'Val accuracy'
        ]
    # Open the file in write mode
    with open(train_file, 'w') as f:
        # Create the csv writer
        writer = csv.writer(f)
        # Write header to the csv file
        writer.writerow(header)
        # Iterate over epoches and batches
        best_val_loss = 1_000_000.0
        for epoch in range(1, epochs + 1):
            # Make sure gradient tracking is on
            model.train(True)
            running_train_acc = 0.0
            running_train_loss = 0.0
            for i, (x, target) in enumerate(train_loader):
                # Load data to device
                x = x.to(device)
                target = target.to(device)
                # Zero out gradients for optimizer 
                optimizer.zero_grad()
                # Run model 
                y_pred_linear, y_pred = model(x, device)

                # Loss
                loss = loss_fn(y_pred_linear, target)
                running_train_loss += loss.item()
                # Update model
                loss.backward()
                optimizer.step()
                # Accuracy
                train_acc = torch.eq(y_pred, target).float().mean().item() * 100.0
                running_train_acc += train_acc
            
            scheduler.step()

            avg_train_loss = running_train_loss / (i + 1)
            avg_train_acc = running_train_acc / (i + 1)

            # Validation
            model.train(False)
            running_val_acc = 0.0
            running_val_loss = 0.0
            for j, (x, target) in enumerate(val_loader):
                # Load data to device
                x = x.to(device)
                target = target.to(device)
                # Run model
                y_pred_linear, y_pred = model(x, device)
                # Loss
                loss = loss_fn(y_pred_linear, target)
                running_val_loss += loss.item()
                # Accuracy
                train_acc = torch.eq(y_pred, target).float().mean().item() * 100.0
                running_val_acc += train_acc

            avg_val_loss = running_val_loss / (j + 1)
            avg_val_acc = running_val_acc / (j + 1)

            # Save info to file
            row = [
                model_name, # 'Model'
                run, # 'Run'
                epoch, # 'Epoch'
                avg_train_loss, # 'Loss'
                avg_train_acc, # 'Accuracy'
                avg_val_loss, # 'Val loss'
                avg_val_acc # 'Val accuracy'
                ]
            writer.writerow(row)

            # Report
            if epoch % 1 == 0:
                print('[Epoch: ' + str(epoch) + '] ' + \
                        '[Train loss = ' + '{:.4f}'.format(avg_train_loss) + '] ' + \
                        '[Val loss = ' + '{:.4f}'.format(avg_val_loss) + '] ' + \
                        '[Train acc = ' + '{:.2f}'.format(avg_train_acc) + '] ' + \
                        '[Val acc = ' + '{:.2f}'.format(avg_val_acc) + '] ')
            
            # Track best performance, and save the model's state
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path =  f"{current_dir}/model_{model_name}_run_{run}_best_epoch_weights.pt"
                torch.save(model.state_dict(), model_path)
            
            if avg_val_acc > 99.0:
                break

# Test function
def test_MAREO(
    run,
    model,
    model_name,
    device,
    ds_names_and_loaders,
    current_dir
    ):
    loss_fn = nn.CrossEntropyLoss()
    model.train(False)

    # Create filename for saving training progress
    test_file = f"{current_dir}/model_{model_name}_run_{run}_test.csv"
    header = ['Model', 'Run']
    for name, loader in ds_names_and_loaders:
        loss_name = f"Loss {name}"
        acc_name = f"Accuracy {name}"
        header.append(loss_name)
        header.append(acc_name)
    # Open the file in write mode
    with open(test_file, 'w') as f:
        # Create the csv writer
        writer = csv.writer(f)
        # Write header to the csv file
        writer.writerow(header)
        # Initialize row
        row = [model_name, run]
        # Iterate over datasets
        for name, data_loader in ds_names_and_loaders:
            running_test_acc = 0.0
            running_test_loss = 0.0
            for j, (x, target) in enumerate(data_loader):
                # Load data to device
                x = x.to(device)
                target = target.to(device)
                # Run model
                y_pred_linear, y_pred = model(x, device)
                # Loss
                loss = loss_fn(y_pred_linear, target)
                running_test_loss += loss.item()
                # Accuracy
                train_acc = torch.eq(y_pred, target).float().mean().item() * 100.0
                running_test_acc += train_acc

            avg_test_loss = running_test_loss / (j + 1)
            avg_test_acc = running_test_acc / (j + 1)

            # Save info to row
            row.append(avg_test_loss)
            row.append(avg_test_acc)
        # Write all data to file
        writer.writerow(row)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # Limit number of threads 
    torch.set_num_threads(10)
    # SVRT-1 parameters
    max_epochs = 20
    LR = 0.0001
    BATCH_SIZE = 64
    BATCH_SIZE_TEST = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RUNS = 10
    MAREO_TIME_STEPS = 4

    # Train on SVRT-1
    current_dir = 'results/RTE_SD_MAREO'
    # check_path(current_dir)

    transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    DATA_ROOT = "data/SD_jpg/task_SD"  # Cambia por tu ruta real

    train_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform_train)
    val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=transform_eval)
    test_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=transform_eval)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    test_names_loaders = [
        ('MainTest', test_loader)
]

    # Train
    for i in range(1, RUNS+1):
        # Insantiate model
        model = MAREO(
            encoder='custom',
            norm_type='contextnorm', 
            steps=MAREO_TIME_STEPS
            )
        # Set to training mode 
        model.train()
        model.to(device)
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=0.1)
        os.makedirs(current_dir, exist_ok=True)
        # Train
        train_MAREO(
            run=i,
            model=model,
            model_name='MAREO',
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=max_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            current_dir=current_dir
            )
        # Set to evaluation mode
        model.eval()
        # Test
        test_MAREO(
            run=i,
            model=model,
            model_name='MAREO',
            device=device,
            ds_names_and_loaders=test_names_loaders,
            current_dir=current_dir
            )
    # Merge files into datasets
    test_df = []
    train_df = []
    for root, dirs, files in os.walk(current_dir, topdown=False):
        for name in files:
            path_to_file = os.path.join(root, name)
            if name.endswith('test.csv'):
                df = pd.read_csv(path_to_file)
                test_df.append(df)
            elif name.endswith('train.csv'):
                df = pd.read_csv(path_to_file)
                train_df.append(df)

    test_df = pd.concat(test_df)
    test_df_name = f'{current_dir}/MAREO_test_RTE_SD.csv'
    test_df.to_csv(test_df_name)

    train_df = pd.concat(train_df)
    train_df_name = f'{current_dir}/MAREO_train_RTE_SD.csv'
    train_df.to_csv(train_df_name)
