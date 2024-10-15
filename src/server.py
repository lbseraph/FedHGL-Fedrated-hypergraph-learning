from typing import Any

import torch
from models import GCN, HGNN, SAGE, SGC, HNHN, HyperGCN
from client import Client

class Server:
    def __init__(
        self,
        clients: list[Client],
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
            self.model = GCN(
                nfeat=args.num_features,
                nhid=args.hiddens_num,
                nclass=args.num_classes,
                dropout=0.5,
                NumLayers=args.num_layers,
                cached=False,
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
        self.model = self.model.to(device)
        self.clients = clients
        self.local = args.local
        self.num_of_clients = len(clients)
        self.broadcast_params(-1)

    def zero_params(self):
        with torch.no_grad():
            for p in self.model.parameters():
                p.zero_()

    def train(self, current_global_epoch: int):

        for client in self.clients:
            client.train(current_global_epoch)
            torch.cuda.empty_cache()
        
        if len(self.clients) > 1 and not self.local:
            self.zero_params()

            # FedAvg algorithm
            for client in self.clients:
                params = client.get_params()
                for p, mp in zip(params, self.model.parameters()):
                    mp.data = mp.data + p.data

            for p in self.model.parameters():
                p.data = p.data / self.num_of_clients

            self.broadcast_params(current_global_epoch)

    def broadcast_params(self, current_global_epoch: int):
        for client in self.clients:
            client.update_params(
                self.model.parameters(), current_global_epoch
            )  # run in submit order
