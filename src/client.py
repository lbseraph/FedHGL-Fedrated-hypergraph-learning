from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models import HGNN

class Client:
    def __init__(
        self,
        rank: int,
        G: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        idx: torch.Tensor,
        device: torch.device,
        args: Any,                  
    ):
        torch.manual_seed(rank)
        np.random.seed(rank)
        
        self.model = HGNN(
            in_ch = args.num_features,
            n_class = args.num_classes,
            n_hid = args.hiddens_num,
            dropout=0.5, 
            layer_num=args.layers_num,
        )
        self.model = self.model.to(device)
        self.rank = rank  # rank = client ID
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.train_losses: list = []
        self.train_accs: list = []
        self.val_losses: list = []
        self.val_accs: list = []
        self.test_losses: list = []
        self.test_accs: list = []

        self.G = G.to(device)
        self.labels = labels.to(device)
        self.features = features.to(device)
        self.idx_train = idx["train"].to(device)
        self.idx_test = idx["test"].to(device)
        self.idx_val = idx["val"].to(device)

        self.local_step = args.local_step

    def zero_params(self):
        with torch.no_grad():
            for p in self.model.parameters():
                p.zero_()

    def update_params(self, params, current_global_epoch: int):
        # load global parameter from global server
        self.zero_params()
        for p, mp in zip(params, self.model.parameters()):
            mp.data = mp.data + p.data

    def train(self, current_global_round: int) -> None:
        # clean cache
        torch.cuda.empty_cache()
        for iteration in range(self.local_step):
            self.model.train()

            loss_train, acc_train = train(
                iteration,
                self.model,
                self.optimizer,
                self.criterion,
                self.features,
                self.G,
                self.labels,
                self.idx_train,
            )
            self.train_losses.append(loss_train)
            self.train_accs.append(acc_train)

            loss_test, acc_test = self.local_test()
            self.test_losses.append(loss_test)
            self.test_accs.append(acc_test)

            loss_val, acc_val = self.local_val()
            self.val_losses.append(loss_val)
            self.val_accs.append(acc_val)

    def local_val(self):
        local_val_loss, local_val_acc = test(
            self.model, self.criterion, self.features, self.G, self.labels, self.idx_val
        )
        return [local_val_loss, local_val_acc]

    def local_test(self):
        local_test_loss, local_test_acc = test(
            self.model, self.criterion, self.features, self.G, self.labels, self.idx_test
        )
        return [local_test_loss, local_test_acc]

    def get_params(self):
        self.optimizer.zero_grad(set_to_none=True)
        return self.model.parameters()

    def get_all_loss_accuray(self):
        return [
            np.array(self.train_losses),
            np.array(self.train_accs),
            np.array(self.test_losses),
            np.array(self.test_accs),
            np.array(self.val_losses),
            np.array(self.val_accs),
        ]

    def get_rank(self) -> int:
        return self.rank

def accuracy(output: torch.Tensor, labels: torch.Tensor):
    """
    This function returns the accuracy of the output with respect to the ground truth given

    Arguments:
    output: (torch.Tensor) - the output labels predicted by the model

    labels: (torch.Tensor) - ground truth labels

    Returns:
    The accuracy of the model (float)
    """

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def test(
    model: torch.nn.Module,
    criterion: torch.nn.CrossEntropyLoss,
    features: torch.Tensor,
    G: torch.Tensor,
    labels: torch.Tensor,
    idx_test: torch.Tensor,
):
    """
    This function tests the model and calculates the loss and accuracy

    Arguments:
    model: (torch.nn.Module) - Specific model passed
    features: (torch.Tensor) - Tensor representing the input features
    G: (torch.Tensor) - Laplacian matrix
    labels: (torch.Tensor) - Contains the ground truth labels for the data.
    idx_test: (torch.Tensor) - Indices specifying the test data points

    Returns:
    The loss and accuracy of the model

    """
    model.eval()
    output = model(features, G)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    return loss_test.item(), acc_test.item()  # , f1_test, auc_test


def train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.CrossEntropyLoss,
    features: torch.Tensor,
    G: torch.Tensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
):  # Centralized or new FL
    """
    This function trains the model and returns the loss and accuracy

    Arguments:
    model: (torch.nn.Module) - Specific model passed
    features: (torch.FloatTensor) - Tensor representing the input features
    G: (torch.Tensor) - Laplacian matrix
    labels: (torch.LongTensor) - Contains the ground truth labels for the data.
    idx_train: (torch.LongTensor) - Indices specifying the test data points
    epoch: (int) - specifies the number of epoch on
    optimizer: (optimizer) - type of the optimizer used

    Returns:
    The loss and accuracy of the model

    """

    model.train()
    optimizer.zero_grad()

    output = model(features, G)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss_train.item(), acc_train.item()