"""Train model"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pneumonia_detector.model import PneumoniaClassifier
from pneumonia_detector.preprocess import XrayDataset, create_weighted_sampler
from pneumonia_detector.train import TRAIN_TRANSFORMS, EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
torch.manual_seed(88)


def initialize_model(image_size=256, learning_rate=0.001):
    """
    Instantiate model and optimizer with parameters

    Parameters
    ----------
    image_size : int
        image size to use for model, assumes square image. Default value 256.
    learning_rate : float
        Set the learning rate for the model, default value 0.001.

    Returns
    ---------
    pneumonia_model : pneumonia_detector.model.PneumoniaClassifier
    optimizer : torch.optim.Optimizer
    """

    pneumonia_model = PneumoniaClassifier(image_size=image_size)

    # Send model to device (GPU/CPU)
    pneumonia_model.to(device)

    optimizer = torch.optim.Adam(pneumonia_model.parameters(), lr=learning_rate)

    return pneumonia_model, optimizer


def train_model(
    model_dir,
    model_filename,
    training_dir,
    validation_dir,
    batch_size,
    patience,
    n_epochs,
    image_size=256,
    learning_rate=0.001,
):
    """
    Function to carry out model training.

    Parameters
    ----------
    model_dir : str
        path to model directory.
    model_filename : str
        model filename as appears in model directory
    training_dir : str
        path to training set directory
    validation_dir : str
        path to validation set directory
    batch_size : int
        batch size for model training.
    patience : int
        number of epochs to wait without validation accuracy improvement before early stopping.
    n_epochs : int
        maximum number of training epochs to run.
    image_size : int
        image size for training images. Images are square so result is (image_size, image_size).
        Default (256, 256).
    learning_rate : float
        learning rate to use during training. Default value is 0.001

     Returns
    ---------
    model : pneumonia_detector.model.PneumoniaClassifier
        image classification model.
    avg_train_losses : list
        List of the mean training losses for each epoch
    avg_valid_losses : list
        List of the mean validation losses for each epoch
    """

    # create train Dataset and Dataloader objects
    Path(model_dir).mkdir(exist_ok=True)
    train_dataset = XrayDataset(root_dir=training_dir, transform=TRAIN_TRANSFORMS)
    sampler = create_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=0,
        sampler=sampler,
    )

    # create validation Dataset and Dataloader objects
    val_dataset = XrayDataset(root_dir=validation_dir, transform=TRAIN_TRANSFORMS)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )

    model, optimizer = initialize_model(
        image_size=image_size, learning_rate=learning_rate
    )

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, path=os.path.join(model_dir, model_filename)
    )
    print("Starting training...")
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, (data, target) in enumerate(train_loader, 1):
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in val_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (
            f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
            + f"valid_loss: {valid_loss:.5f}"
        )

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(os.path.join(model_dir, model_filename)))
    params = {
        "model_dir": model_dir,
        "model_filename": model_filename,
        "training_dir": training_dir,
        "validation_dir": validation_dir,
        "batch_size": batch_size,
        "patience": patience,
        "n_epochs": n_epochs,
        "image_size": image_size,
        "learning_rate": learning_rate,
        "device": device.type,
    }
    with open(os.path.join(model_dir, f"{model_filename}_params.json"), "w+") as f:
        json.dump(params, f, indent=4)

    return model, avg_train_losses, avg_valid_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="train_model", description="Trains a PneumoniaClassifier model"
    )
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--model_filename", type=str)
    parser.add_argument("--training_dir", type=str)
    parser.add_argument("--validation_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    model, _, _ = train_model(**vars(args))
