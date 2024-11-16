from typing import Any

import numpy as np
import torch

from models import GCN, HGNN, SAGE, SGC, HNHN, HyperGCN

# from models import HGNN, GCN
# test

class Client:
    def __init__(
        self,
        rank: int,
        structure: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        device: torch.device,
        args: Any,                  
    ):
        if args.method == "FedHGN":
            self.model = HGNN(
                in_ch = args.num_features,
                n_class = args.num_classes,
                n_hid = args.hiddens_num,
                dropout=0.5, 
                layer_num=args.num_layers,
            )
        elif args.method == "HNHN":
            self.model = HNHN(
                in_ch = args.num_features,
                n_class = args.num_classes,
                n_hid = args.hiddens_num,
                dropout=0.5, 
                layer_num=args.num_layers,
            )
        elif args.method == "HyperGCN":
            self.model = HyperGCN(
                in_ch = args.num_features,
                n_class = args.num_classes,
                n_hid = args.hiddens_num,
                dropout=0.5, 
                layer_num=args.num_layers,
            ) 
        elif args.method == "FedGCN":
            # if args.dname == "news":
            self.model = GCN(
                nfeat=args.num_features,
                nhid=args.hiddens_num,
                nclass=args.num_classes,
                dropout=0.5,
                NumLayers=args.num_layers,
                cached=True,
            )
        elif args.method == "FedCog":
            # if args.dname == "news":
            self.model = SGC(
                in_ch = args.num_features,
                n_class = args.num_classes,
                n_hid = args.hiddens_num,
                dropout=0.5, 
                layer_num=args.num_layers,
            )
        elif args.method == "FedSage":
            self.model = SAGE(
                nfeat=args.num_features,
                nhid=args.hiddens_num,
                nclass=args.num_classes,
                dropout=0.5,
                NumLayers=args.num_layers,
            )
        self.n_client = args.n_client
        self.safety=args.safety
        self.epsilon = args.epsilon
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
        self.best_val_acc = 0
        
        # if args.dname == "news" and args.method == "FedGCN":
        #     self.structure = structure
        # else:
        self.structure = structure.to(device)
        self.labels = labels.to(device)
        self.features = features.to(device)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.idx_train = (train_mask == True).nonzero().view(-1)
        self.idx_val = (val_mask == True).nonzero().view(-1)
        self.idx_test = (test_mask == True).nonzero().view(-1)

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

        for iteration in range(self.local_step):
            
            # print("client", self.rank, current_global_round, torch.cuda.memory_allocated(), torch.cuda.memory_cached())
            # clean cache
            torch.cuda.empty_cache()
            loss_train, acc_train = train(
                iteration,
                self.model,
                self.optimizer,
                self.criterion,
                self.features,
                self.structure,
                self.labels,
                self.idx_train
            )
            self.train_accs.append(acc_train)

            loss_val, acc_val = self.local_val()
            self.val_losses.append(loss_val)
            self.val_accs.append(acc_val)
            if (acc_val > self.best_val_acc):
                self.best_val_acc = acc_val
                if self.safety:
                     torch.save(self.model.state_dict(), f"model/{type(self.model).__name__}_client_{self.n_client}_{self.rank}_{self.epsilon}.pt")
                else:
                    torch.save(self.model.state_dict(), f"model/{type(self.model).__name__}_client_{self.n_client}_{self.rank}.pt")
        self.train_losses.append(loss_train)
        loss_test, acc_test = self.local_test()
        self.test_losses.append(loss_test)
        self.test_accs.append(acc_test)



    def local_val(self):
        local_val_loss, local_val_acc = test(
            self.model, self.criterion, self.features, self.structure, self.labels, self.idx_val
        )
        return [local_val_loss, local_val_acc]

    def local_test(self):
        local_test_loss, local_test_acc = test(
            self.model, self.criterion, self.features, self.structure, self.labels, self.idx_test
        )
        # print(local_test_loss, local_test_acc)
        return local_test_loss, local_test_acc

    def get_params(self):
        self.optimizer.zero_grad(set_to_none=True)
        return self.model.parameters()

    def get_all_loss_accuracy(self):
        return [
            np.array(self.train_losses),
            np.array(self.test_accs),
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

def train(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.CrossEntropyLoss,
    features: torch.Tensor,
    structure: torch.Tensor,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
):  
    model.train()
    optimizer.zero_grad()
    output = model(features, structure)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss_train.item(), acc_train.item()

def test(
    model: torch.nn.Module,
    criterion: torch.nn.CrossEntropyLoss,
    features: torch.Tensor,
    structure: torch.Tensor,
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
    output = model(features, structure)
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return loss_test.item(), acc_test.item()  # , f1_test, auc_test
