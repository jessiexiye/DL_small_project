# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
# Torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import random
from argparse import ArgumentParser

from dataloader import Multiclass_Dataset

from model_3class import VGG 
import os
import time
from tqdm import tqdm
import csv
from csv import DictWriter
import copy

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def evaluate(dataloader, model):
    """
    return the overall accuracy of the model on the dataset
    """
    with torch.no_grad():
        model.eval()
        count = 0
        correct = 0
        total_loss = 0.0
        reg_loss = 0.0
        l2_lambda = 0.00001
        criterion = nn.BCEWithLogitsLoss()
        for images_data, target_labels in tqdm(dataloader):
            if config.use_gpu:
                images_data = images_data.cuda()
                target_labels = target_labels.cuda()
            predicted_labels = model(images_data)
            total_loss += criterion(predicted_labels, target_labels)
            count += predicted_labels.shape[0]
            preds = predicted_labels.argmax(dim=1)
            targets = target_labels.argmax(dim=1)
            correct += (torch.eq(preds, targets)).sum().item()
        
        l2_reg = torch.tensor(0.)
        if config.use_gpu:
            l2_reg = l2_reg.cuda()
        for param in model.parameters():
            l2_reg += torch.norm(param)
            reg_loss += l2_lambda * l2_reg

        total_loss += reg_loss
        accuracy = correct * 1.0 / count
    return accuracy, total_loss.item()

def train(config):
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    device = torch.device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.use_gpu:
        torch.cuda.manual_seed_all(config.seed)

    # Dataset
    multiclass_train = Multiclass_Dataset('train')
    multiclass_val = Multiclass_Dataset('val')
    multiclass_test = Multiclass_Dataset('test')

    # Dataloader
    multiclass_train_loader = DataLoader(multiclass_train, batch_size = config.batch_size, shuffle = True)
    multiclass_test_loader = DataLoader(multiclass_test, batch_size = config.batch_size, shuffle = True)
    multiclass_val_loader = DataLoader(multiclass_val, batch_size = config.batch_size, shuffle = True)

    # L2 regularization parameter
    l2_lambda = 0.00001

    # Instantiate model, criterion and optimizer
    model = VGG(num_classes=3)
    if config.use_gpu:
        model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    lr_sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    best_acc = -0.1
    best_epoch = -1
    start_time = time.time()

    # Train the model that classifies normal and multiclass images
    print("***** Training the multiclass classifier *****")
    for epoch in range(config.epochs):
        total_loss = 0.0
        model.train()
        for images_data, target_labels in tqdm(multiclass_train_loader):
            # images_data: [batch_size, 1, 150, 150]
            # target_labels: [batch_size, 3]
            if config.use_gpu:
                images_data = images_data.cuda()
                target_labels = target_labels.cuda()
            total_loss = 0.0
            model.train()
            optimizer.zero_grad()
            predicted_labels = model(images_data)
            loss = criterion(predicted_labels, target_labels)

            # L2 regularization
            l2_reg = torch.tensor(0.)
            if config.use_gpu:
                l2_reg = l2_reg.cuda()
            for param in model.parameters():
                l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()  

        # Evaluate the performance and save the model parameters each epoch   
        train_acc, train_loss = evaluate(multiclass_train_loader, model)
        val_acc, val_loss = evaluate(multiclass_test_loader, model)
        test_acc, test_loss = evaluate(multiclass_val_loader, model)
        torch.save(model.state_dict(), './checkpoints2/' + str(epoch) + '_params_multiclass.pth')
        
        # Save the best performing model parameters based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), './checkpoints2/'+ 'best_params_multiclass.pth')
        print(f"{now()} Epoch{epoch}: loss: {total_loss}, train_acc: {train_acc}, val_acc: {val_acc}")
        lr_sheduler.step()

        # Record loss and accuracies for learning curve plots
        fieldnames = ['epoch', 'train_loss', 'valid_loss', 'train_acc', 'val_acc']
        out_dict = {'epoch': epoch, 'train_loss': train_loss, 'valid_loss':valid_loss, 'train_acc': train_acc, 'val_acc': val_acc}
        with open('./outputs/multiclass.csv', 'a') as out_f:
            writer = DictWriter(out_f, fieldnames=fieldnames)
            writer.writerow(out_dict)     

    end_time = time.time()
    print("*"*20)
    print(f"{now()} finished; epoch {best_epoch} best_acc: {best_acc}, time/epoch: {(end_time-start_time)/config.epochs}")
    print()

def test_whole(config):
    # Dataset
    multiclass_train = Multiclass_Dataset('train')
    multiclass_val = Multiclass_Dataset('val')
    multiclass_test = Multiclass_Dataset('test')

    # Dataloader
    multiclass_train_loader = DataLoader(multiclass_train, batch_size = 2*config.batch_size, shuffle = True)
    multiclass_test_loader = DataLoader(multiclass_test, batch_size = 2*config.batch_size, shuffle = True)
    multiclass_val_loader = DataLoader(multiclass_val, batch_size = 2*config.batch_size, shuffle = True)

    # Load the trained parameters from the best performing model
    model = VGG(num_classes=3)
    best_model_path = './checkpoints2/' + 'best_params_multiclass.pth'
    model.load_state_dict(copy.deepcopy(torch.load(best_model_path, config.device)))
    model.to(config.device)

    # Evaluate the performances on training, validation and test sets
    train_acc, train_loss = evaluate(multiclass_train_loader, model)
    val_acc, val_loss = evaluate(multiclass_test_loader, model)
    test_acc, test_loss = evaluate(multiclass_val_loader, model)
    print("Test results: ")
    print(f"train_acc: {train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--use_gpu', default=False, action='store_true', help='whether to use GPU')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='number of training epochs')
    parser.add_argument('--train', default=False, action='store_true', help='whether the train from scratch')
    parser.add_argument('--test', default=False, action='store_true', help='whether to test only')

    config = parser.parse_args()
    print(config)

    if config.train:
        train(config)
        print('Testing best trained parameters ... ')
        test_whole(config)
    elif config.test:
        test_whole(config)