import time
from utils import *
from train import *
import torch.optim as optim
import torch.nn as nn
import os

from tqdm import tqdm

from nn_edm import NNEDMModel

def l1_regularization(model, lambda_l1):
    l1_norm = 0
    for param in model.parameters():
        l1_norm += param.abs().sum()
    return lambda_l1 * l1_norm

def train(model, device, train_dataloader, val_dataloader, epochs=1000, patience=50):
    optimizer = optim.Adam(model.parameters(),lr=0.00001)
    criterion = torch.nn.MSELoss()
    saved_model_path = 'model.pth'
    scheduler = None
    train_loss_arr = []
    val_loss_arr = []
    best_val_loss = float('inf')
    lambda_l1 = 0.9


    for epoch in range(epochs):
            ts = time.time()
            train_losses = []
            for iter, (inputs, labels) in enumerate(train_dataloader):
                # TODO  reset optimizer gradients
                optimizer.zero_grad()


                # both inputs and labels have to reside in the same device as the model's
                inputs =  inputs.to(device)
                labels =   labels.to(device)

                outputs= model(inputs)

                loss =   criterion(outputs, labels)
                train_losses.append(loss.item())

                # backpropagate

                l1_loss = l1_regularization(model, lambda_l1)
                total_loss = loss + l1_loss

                # Backward pass and optimization
                total_loss.backward()
                

                # update the weights
                optimizer.step()

            if epoch%100 == 0:
                print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
            if scheduler is not None:
                scheduler.step()
            curr_val_loss = eval(model, device, val_dataloader, epoch)
            val_loss_arr.append(curr_val_loss)
            train_loss_arr.append(np.mean(train_losses))
            if curr_val_loss < best_val_loss:
                best_val_loss = curr_val_loss
                # save the best model
                torch.save(model.state_dict(), saved_model_path)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    #early = epoch-patience
                    break
    plot_losses(train_loss_arr, val_loss_arr, 'loss')
        
    ## TODO

def eval(model, device, val_dataloader, epoch=None):

    model.eval() 
    criterion =  torch.nn.MSELoss() #IF CHANGING MAKE SURE TO CHANGE IN TRAIN
    
    losses = []


    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_dataloader):
            inputs =  inputs.to(device)
            labels =   labels.to(device)
            outputs = model(inputs)

            #print(outputs.size())
            #print(labels.size())
            loss = criterion(outputs, labels)
            losses.append(loss.item())



    if epoch is None:
        print(f"Loss is {np.mean(losses)}")
    elif epoch%100 == 0:
        print(f"Loss at epoch: {epoch} is {np.mean(losses)}")

    model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(losses)
