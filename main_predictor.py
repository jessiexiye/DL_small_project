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

from dataloader import Infection_Dataset, Covid_Dataset, Multiclass_Dataset

from model import VGG 
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
    return the binary accuracy of the model on the dataset
    """
    with torch.no_grad():
        model.eval()
        count = 0
        correct = 0
        total_loss = 0.0
        reg_loss = 0.0
        l2_lambda = 0.001
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

def test(dataloader, model_infection, model_covid):
    """
    return the overall accuracy using the two classifiers on the dataset
    """
    model_infection.eval()
    model_covid.eval()
    count = 0
    correct = 0
    for images_data, target_labels in tqdm(dataloader):
        if config.use_gpu:
            images_data = images_data.cuda()
            target_labels = target_labels.cuda()
        predicted_infection = model_infection(images_data)
        count += predicted_infection.shape[0]
        preds_infection = predicted_infection.argmax(dim=1)
        targets = target_labels.argmax(dim=1)
        for i in range(len(preds_infection)):
            if preds_infection[i] == 0:
                if preds_infection[i] == targets[i]:
                    correct += 1
                continue
            else:
                predicted_covid = model_covid(images_data[i].unsqueeze(0))
                preds_covid = predicted_covid.argmax(dim=1) + 1
                correct += (preds_covid == targets[i])
    accuracy = correct * 1.0 / count
    return accuracy.item()
            
def train(config):
    
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    device = torch.device(config.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.use_gpu:
        torch.cuda.manual_seed_all(config.seed)

    # Dataset
    infection_train = Infection_Dataset('train')
    infection_test = Infection_Dataset('test')
    infection_val = Infection_Dataset('val')

    covid_train = Covid_Dataset('train')
    covid_test = Covid_Dataset('test')
    covid_val = Covid_Dataset('val')

    # Dataloader from dataset 
    infection_train_loader = DataLoader(infection_train, batch_size = config.batch_size, shuffle = True)
    infection_test_loader = DataLoader(infection_test, batch_size = config.batch_size, shuffle = True)
    infection_val_loader = DataLoader(infection_val, batch_size = config.batch_size, shuffle = True)

    covid_train_loader = DataLoader(covid_train, batch_size = config.batch_size, shuffle = True)
    covid_test_loader = DataLoader(covid_test, batch_size = config.batch_size, shuffle = True)
    covid_val_loader = DataLoader(covid_val, batch_size = config.batch_size, shuffle = True)

    # L2 regularization parameter
    l2_lambda = 0.001

    # Instantiate model, criterion and oprimizer
    model = VGG()
    if config.use_gpu:
        model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    lr_sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    best_acc = -0.1
    best_epoch = -1
    start_time = time.time()
    
    # Train the model that classifies normal and infection images
    print("***** Training the first classifier *****")
    for epoch in range(config.epochs):
        total_loss = 0.0
        model.train()
        for images_data, target_labels in tqdm(infection_train_loader):
            # images_data: [batch_size, 1, 150, 150]
            # target_labels: [batch_size, 2]
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

        # Evaluate the performance and save the model parameters each epoch   
        train_acc, train_loss = evaluate(infection_train_loader, model)
        val_acc, val_loss = evaluate(infection_test_loader, model)
        torch.save(model.state_dict(), './checkpoints/' + str(epoch) + '_params_infection.pth')

        # Save the best performing model parameters based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), './checkpoints/'+ 'best_params_infection.pth')
        print(f"{now()} Epoch{epoch}: train_loss: {train_loss}, val_loss: {val_loss}, train_acc: {train_acc}, val_acc: {val_acc}")
        lr_sheduler.step()

        # Record loss and accuracies for learning curve plots
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
        out_dict = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc}
        with open('./outputs/infection.csv', 'a') as out_f:
            writer = DictWriter(out_f, fieldnames=fieldnames)
            writer.writerow(out_dict)     

    end_time = time.time()
    print("*"*20)
    print(f"{now()} finished; epoch {best_epoch} best_acc: {best_acc}, time/epoch: {(end_time-start_time)/config.epochs}")
    print()
    
    # Instantiate another model
    model_covid = VGG()
    if config.use_gpu:
        model_covid.to(device)
    optimizer_covid = optim.AdamW(model_covid.parameters(), lr=config.lr)
    lr_sheduler_covid = optim.lr_scheduler.StepLR(optimizer_covid, step_size=3, gamma=0.7)
    best_acc_covid = -0.1
    best_epoch_covid = -1
    start_time_covid = time.time()

    # Train another model that classifies covid and non-covid images
    print("***** Training the second classifier *****")
    for epoch_covid in range(config.epochs):
        total_loss_covid = 0.0
        model_covid.train()
        for images_data, target_labels in tqdm(covid_train_loader):
            # images_data: [batch_size, 1, 150, 150]
            # target_labels: [batch_size, 2]
            if config.use_gpu:
                images_data = images_data.cuda()
                target_labels = target_labels.cuda()
            total_loss_covid = 0.0
            model_covid.train()
            optimizer_covid.zero_grad()
            predicted_labels = model_covid(images_data)
            loss = criterion(predicted_labels, target_labels)

            # L2 regularization
            l2_reg = torch.tensor(0.)
            if config.use_gpu:
                l2_reg = l2_reg.cuda()
            for param in model_covid.parameters():
                l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
            loss.backward()
            optimizer_covid.step()
            total_loss_covid += loss.item() 

        # Evaluate the performance and save the model parameters each epoch       
        train_acc_covid, train_loss_covid = evaluate(covid_train_loader, model_covid)
        val_acc_covid, val_loss_covid = evaluate(covid_test_loader, model_covid)
        torch.save(model_covid.state_dict(), './checkpoints/' + str(epoch_covid) + '_params_covid.pth')
        
        # Save the best performing model parameters based on validation accuracy
        if val_acc_covid > best_acc_covid:
            best_acc_covid = val_acc_covid
            best_epoch_covid = epoch_covid
            torch.save(model_covid.state_dict(), './checkpoints/'+ 'best_params_covid.pth')
        print(f"{now()} epoch {epoch_covid}: train_loss: {train_loss_covid}, val_loss: {val_loss_covid}, train_acc: {train_acc_covid}, val_acc_covid: {val_acc_covid}")  
        lr_sheduler_covid.step() 

        # Record loss and accuracies for learning curve plots
        fieldnames = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
        out_dict = {'epoch': epoch_covid, 'train_loss': train_loss_covid, 'val_loss': val_loss_covid,'train_acc': train_acc_covid, 'val_acc': val_acc_covid}
        with open('./outputs/infection.csv', 'a') as out_f:
            writer = DictWriter(out_f, fieldnames=fieldnames)
            writer.writerow(out_dict)   
    end_time = time.time()
    print("*"*20)
    print(f"{now()} finished; epoch {best_epoch_covid} best_acc_covid: {best_acc_covid}, time/epoch: {(end_time-start_time)/config.epochs}")

def test_whole(config):

    # Dataset
    multiclass_train = Multiclass_Dataset('train')
    multiclass_val = Multiclass_Dataset('val')
    multiclass_test = Multiclass_Dataset('test')

    # Dataloader
    multiclass_train_loader = DataLoader(multiclass_train, batch_size = 2*config.batch_size, shuffle = True)
    multiclass_test_loader = DataLoader(multiclass_test, batch_size = 2*config.batch_size, shuffle = True)
    multiclass_val_loader = DataLoader(multiclass_val, batch_size = 2*config.batch_size, shuffle = True)

    # for epoch in range(config.epochs):
    #     # Load trained model parameters
    #     model_infection = VGG()
    #     model_infection_path = './checkpoints/' + str(epoch) + '_params_infection.pth'
    #     model_infection.load_state_dict(copy.deepcopy(torch.load(model_infection_path, config.device)))
    #     model_infection.to(config.device)
    #     model_covid = VGG()
    #     model_covid_path = './checkpoints/' + str(epoch) + '_params_covid.pth'
    #     model_covid.load_state_dict(copy.deepcopy(torch.load(model_covid_path, config.device)))
    #     model_covid.to(config.device)

    #     train_accuracy_whole = test(multiclass_train_loader, model_infection, model_covid)
    #     test_accuracy_whole = test(multiclass_test_loader, model_infection, model_covid)
    #     val_accuracy_whole = test(multiclass_val_loader, model_infection, model_covid)
    #     print(f"epoch {epoch}: train_acc_overall: {train_accuracy_whole}, val_acc_overall: {val_accuracy_whole}, test_acc_overall: {test_accuracy_whole}")  
        
    # Evaluate the performance using the combination of the best models 
    model_infection = VGG()
    model_infection_path = './checkpoints/' + 'best_params_infection.pth'
    model_infection.load_state_dict(copy.deepcopy(torch.load(model_infection_path, config.device)))
    model_infection.to(config.device)
    model_covid = VGG()
    model_covid_path = './checkpoints/' + 'best_params_covid.pth'
    model_covid.load_state_dict(copy.deepcopy(torch.load(model_covid_path, config.device)))
    model_covid.to(config.device)

    train_accuracy_whole = test(multiclass_train_loader, model_infection, model_covid)
    test_accuracy_whole = test(multiclass_test_loader, model_infection, model_covid)
    val_accuracy_whole = test(multiclass_val_loader, model_infection, model_covid)
    print()
    print(f"Combination of best models: train_acc_overall: {train_accuracy_whole}, val_acc_overall: {val_accuracy_whole}, test_acc_overall: {test_accuracy_whole}")  
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--use_gpu', default=False, action='store_true', help='whether to use GPU')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--train', default=False, action='store_true', help='whether the train from scratch')
    parser.add_argument('--test', default=False, action='store_true', help='whether to test only')

    config = parser.parse_args()
    print(config)

    if config.train:
        train(config)
        print()
        test_whole(config)
    elif config.test:
        test_whole(config)