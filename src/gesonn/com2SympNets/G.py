"""
Author:
    A BELIERES FRENDO (IRMA)
Date:
    05/05/2023

Machine learning code for symplectic maps
Inspired from a code given by V MICHEL DANSAC (INRIA)
"""

# %%


# ----------------------------------------------------------------------
#   IMPORTS - MACHINE CONFIGURATION
# ----------------------------------------------------------------------

# imports
import copy
import os
import time

import torch
import torch.nn as nn
from gesonn.com1PINNs import metricTensors
from gesonn.out1Plot import makePlots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is G.py")


# ----------------------------------------------------------------------
#   CLASSE SYMP_NET_FORWARD - HERITIERE DE NN.DATAPARALLEL
# ----------------------------------------------------------------------


class Symp_Net_Forward(nn.DataParallel):
    # constructeur
    def __init__(self, n):
        super(Symp_Net_Forward, self).__init__(nn.Module())
        min_value = -1
        max_value = 1

        n = (n, 1)

        self.K = torch.nn.parameter.Parameter(
            min_value
            + (max_value - min_value) * torch.rand(n, dtype=torch.double, device=device)
        )
        self.a = torch.nn.parameter.Parameter(
            min_value
            + (max_value - min_value) * torch.rand(n, dtype=torch.double, device=device)
        )
        self.b = torch.nn.parameter.Parameter(
            min_value
            + (max_value - min_value) * torch.rand(n, dtype=torch.double, device=device)
        )

    # forward function -> defines the network structure
    def forward(self, x_or_y):
        Kx_or_y = torch.einsum("ik,jk->ijk", self.K, x_or_y)
        shape_x_or_y = x_or_y.size()
        ones = torch.ones(shape_x_or_y, device=device)
        B = torch.einsum("ik,jk->ijk", self.b, ones)
        A = torch.einsum("ik,jk->ijk", self.a, ones)
        Asigma = A * torch.tanh(Kx_or_y + B)
        return torch.einsum("ik,ijk->jk", self.K, Asigma)


# ----------------------------------------------------------------------
#   CLASSE SYMP_NET_FORWARD_NO_BIAS - HERITIERE DE NN.DATAPARALLEL
# ----------------------------------------------------------------------


class Symp_Net_Forward_No_Bias(nn.DataParallel):
    # constructeur
    def __init__(self, n):
        super(Symp_Net_Forward_No_Bias, self).__init__(nn.Module())
        min_value = -1
        max_value = 1

        n = (n, 1)

        self.K = torch.nn.parameter.Parameter(
            min_value
            + (max_value - min_value) * torch.rand(n, dtype=torch.double, device=device)
        )
        self.a = torch.nn.parameter.Parameter(
            min_value
            + (max_value - min_value) * torch.rand(n, dtype=torch.double, device=device)
        )

    # forward function -> defines the network structure
    def forward(self, x_or_y):
        Kx_or_y = torch.einsum("ik,jk->ijk", self.K, x_or_y)
        shape_x_or_y = x_or_y.size()
        ones = torch.ones(shape_x_or_y, device=device)
        A = torch.einsum("ik,jk->ijk", self.a, ones)
        Asigma = A * torch.tanh(Kx_or_y)
        return torch.einsum("ik,ijk->jk", self.K, Asigma)


# ----------------------------------------------------------------------
#   CLASSE SYMP_NET - RESEAU DE NEURONES
# ----------------------------------------------------------------------


class Symp_Net:
    DEFAULT_SYMPNETS_DICT = {
        "learning_rate": 1e-2,
        "nb_of_networks": 2,
        "networks_size": 5,
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "default",
        "symplecto_name": "bizaroid",
        "to_be_trained": True,
    }

    # constructeur
    def __init__(self, **kwargs):
        SympNetsDict = kwargs.get("SympNetsDict", self.DEFAULT_SYMPNETS_DICT)

        if SympNetsDict.get("learning_rate") == None:
            SympNetsDict["learning_rate"] = self.DEFAULT_SYMPNETS_DICT["learning_rate"]
        if SympNetsDict.get("nb_of_networks") == None:
            SympNetsDict["nb_of_networks"] = self.DEFAULT_SYMPNETS_DICT[
                "nb_of_networks"
            ]
        if SympNetsDict.get("networks_size") == None:
            SympNetsDict["networks_size"] = self.DEFAULT_SYMPNETS_DICT["networks_size"]
        if SympNetsDict.get("rho_min") == None:
            SympNetsDict["rho_min"] = self.DEFAULT_SYMPNETS_DICT["rho_min"]
        if SympNetsDict.get("rho_max") == None:
            SympNetsDict["rho_max"] = self.DEFAULT_SYMPNETS_DICT["rho_max"]
        if SympNetsDict.get("file_name") == None:
            SympNetsDict["file_name"] = self.DEFAULT_SYMPNETS_DICT["file_name"]
        if SympNetsDict.get("symplecto_name") == None:
            SympNetsDict["symplecto_name"] = self.DEFAULT_SYMPNETS_DICT[
                "symplecto_name"
            ]
        if SympNetsDict.get("to_be_trained") == None:
            SympNetsDict["to_be_trained"] = self.DEFAULT_SYMPNETS_DICT["to_be_trained"]

        # Storage file
        self.file_name = (
            "./../../../outputs/SympNets/net/" + SympNetsDict["file_name"] + ".pth"
        )
        self.fig_storage = "./../outputs/SympNets/img/" + SympNetsDict["file_name"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = os.path.join(script_dir, self.file_name)
        # Learning rate
        self.learning_rate = SympNetsDict["learning_rate"]
        # Layers parameters
        self.nb_of_networks = SympNetsDict["nb_of_networks"]
        self.networks_size = SympNetsDict["networks_size"]
        # Geometry of the shape
        self.rho_min, self.rho_max = SympNetsDict["rho_min"], SympNetsDict["rho_max"]
        self.theta_min, self.theta_max = 0, 2 * torch.pi
        self.Vol = torch.pi * self.rho_max**2
        self.name_symplecto = SympNetsDict["symplecto_name"]

        self.create_networks()
        self.load(self.file_name)

        self.to_be_trained = SympNetsDict["to_be_trained"]

    def sympnet_layer_append(self, nets, optims, i):
        nets.append(nn.DataParallel(Symp_Net_Forward(self.networks_size)).to(device))
        optims.append(torch.optim.Adam(nets[i].parameters(), lr=self.learning_rate))

    def create_networks(self):
        # réseau relatif au symplecto
        self.up_nets = []
        self.down_nets = []
        self.up_optimizers = []
        self.down_optimizers = []
        for i in range(self.nb_of_networks):
            self.sympnet_layer_append(self.up_nets, self.up_optimizers, i)
            self.sympnet_layer_append(self.down_nets, self.down_optimizers, i)

    def load_sympnet_layer(
        self, nets, optimizers, checkpoint_nets, checkpoint_optimizers, checkpoint
    ):
        nets_state_dicts = checkpoint[checkpoint_nets]
        i = 0
        for _, state_dict in zip(nets, nets_state_dicts):
            nets[i].load_state_dict(state_dict)
            i += 1

        optimizers_state_dicts = checkpoint[checkpoint_optimizers]
        i = 0
        for _, state_dict in zip(optimizers, optimizers_state_dicts):
            optimizers[i].load_state_dict(state_dict)
            i += 1

    def load(self, file_name):
        self.loss_history = []

        try:
            try:
                checkpoint = torch.load(file_name)
            except RuntimeError:
                checkpoint = torch.load(file_name, map_location=torch.device("cpu"))

            self.load_sympnet_layer(
                self.up_nets,
                self.up_optimizers,
                "up_models_state_dict",
                "up_optimizers_state_dict",
                checkpoint,
            )
            self.load_sympnet_layer(
                self.down_nets,
                self.down_optimizers,
                "down_models_state_dict",
                "down_optimizers_state_dict",
                checkpoint,
            )

            self.loss = checkpoint["loss"]

            try:
                self.loss_history = checkpoint["loss_history"]
            except KeyError:
                pass

            self.to_be_trained = False

        except FileNotFoundError:
            self.to_be_trained = True
            print("G_SYMPNET : network was not loaded from file: training needed")

    def get_physical_parameters(self):
        return {
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
        }

    @staticmethod
    def save_sympnet_layer(state_dicts):
        return [state_dict for state_dict in state_dicts]

    def save(
        self,
        file_name,
        epoch,
        up_nets_state_dict,
        up_optimizers_state_dict,
        down_nets_state_dict,
        down_optimizers_state_dict,
        loss,
        loss_history,
        nb_of_networks,
        networks_size,
    ):
        torch.save(
            {
                epoch: epoch,
                "up_models_state_dict": self.save_sympnet_layer(up_nets_state_dict),
                "up_optimizers_state_dict": self.save_sympnet_layer(
                    up_optimizers_state_dict
                ),
                "down_models_state_dict": self.save_sympnet_layer(down_nets_state_dict),
                "down_optimizers_state_dict": self.save_sympnet_layer(
                    down_optimizers_state_dict
                ),
                "loss": loss,
                "loss_history": loss_history,
                "nb_of_networks": nb_of_networks,
                "networks_size": networks_size,
            },
            file_name,
        )

    def apply_symplecto(self, x, y):
        x_net, y_net = x, y
        for i in range(self.nb_of_networks):
            x_net, y_net = x_net + self.up_nets[i](y_net), y_net
            x_net, y_net = x_net, y_net + self.down_nets[i](x_net)
        return x_net, y_net

    @staticmethod
    def random(min_value, max_value, shape, requires_grad=False, device=device):
        random_numbers = torch.rand(
            shape, device=device, dtype=torch.double, requires_grad=requires_grad
        )
        return min_value + (max_value - min_value) * random_numbers

    def make_collocation(self, n_collocation):
        shape = (n_collocation, 1)

        theta_collocation = self.random(
            self.theta_min, self.theta_max, shape, requires_grad=True
        )

        self.x_collocation = torch.cos(theta_collocation)
        self.y_collocation = torch.sin(theta_collocation)

    def train(self, **kwargs):
        # nombre de pas de descente
        epochs = kwargs.get("epochs", 500)
        # nombre de pts tirés pour monte-carlo
        n_collocation = kwargs.get("n_collocation", 10000)

        plot_history = kwargs.get("plot_history", False)
        save_plots = kwargs.get("save_plots", False)

        # trucs de sauvegarde ?
        try:
            best_loss_value = self.loss.item()
        except AttributeError:
            best_loss_value = 1e10

        # boucle principale de la descnet ede gradient
        tps1 = time.time()
        for epoch in range(epochs):
            # mise à 0 du gradient
            for i in range(self.nb_of_networks):
                self.up_optimizers[i].zero_grad()
                self.down_optimizers[i].zero_grad()

            # mise à 0 de la loss
            self.loss = torch.tensor([0.0], device=device)

            # Loss based on PDE
            if n_collocation > 0:
                self.make_collocation(n_collocation)

                x_ex, y_ex = metricTensors.apply_symplecto(
                    self.x_collocation, self.y_collocation, name=self.name_symplecto
                )
                x_net, y_net = self.apply_symplecto(
                    self.x_collocation, self.y_collocation
                )
                self.loss = (
                    ((x_ex - x_net) ** 2 + (y_ex - y_net) ** 2).sum()
                    / n_collocation
                    * torch.math.pi
                )

            self.loss.backward()
            for i in range(self.nb_of_networks):
                self.up_optimizers[i].step()
                self.down_optimizers[i].step()

            self.loss_history.append(self.loss.item())

            if epoch % 500 == 0:
                print(f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}")
                try:
                    self.save(
                        self.file_name,
                        epoch,
                        best_up_nets,
                        best_up_optimizers,
                        best_down_nets,
                        best_down_optimizers,
                        best_loss,
                        self.loss_history,
                        self.get_physical_parameters(),
                        self.nb_of_networks,
                    )
                except NameError:
                    pass

            if self.loss.item() < best_loss_value:
                print(f"epoch {epoch: 5d}:    best loss = {self.loss.item():5.2e}")
                best_loss = self.loss.clone()
                best_loss_value = best_loss.item()
                best_up_nets = self.copy_sympnet(self.up_nets)
                best_up_optimizers = self.copy_sympnet(self.up_optimizers)
                best_down_nets = self.copy_sympnet(self.down_nets)
                best_down_optimizers = self.copy_sympnet(self.down_optimizers)
        tps2 = time.time()

        print(f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}")

        try:
            self.save(
                self.file_name,
                epoch,
                best_up_nets,
                best_up_optimizers,
                best_down_nets,
                best_down_optimizers,
                best_loss,
                self.loss_history,
                self.get_physical_parameters(),
                self.nb_of_networks,
            )
            self.load(self.file_name)
        except UnboundLocalError:
            pass

        if plot_history:
            self.plot_result(save_plots)
            hausdorff_error = self.get_hausdorff_error()
            print("Hausdorff error: ", hausdorff_error)

        return tps2 - tps1

    @staticmethod
    def copy_sympnet(to_be_copied):
        return [copy.deepcopy(copie.state_dict()) for copie in to_be_copied]

    def get_hausdorff_error(self, n_pts=10_000):
        import numpy as np
        import scipy.spatial.distance as dist

        self.make_collocation(n_pts)

        x_ex, y_ex = metricTensors.apply_symplecto(
            self.x_collocation, self.y_collocation, name=self.name_symplecto
        )
        x_net, y_net = self.apply_symplecto(self.x_collocation, self.y_collocation)

        X_net = []
        X_ex = []
        for x, y in zip(x_net.flatten().tolist(), y_net.flatten().tolist()):
            X_net.append((x, y))
        for x, y in zip(x_ex.flatten().tolist(), y_ex.flatten().tolist()):
            X_ex.append((x, y))

        X_net = np.array(X_net)
        X_ex = np.array(X_ex)

        return max(
            dist.directed_hausdorff(X_net, X_ex)[0],
            dist.directed_hausdorff(X_ex, X_net)[0],
        )

    def plot_result(self, save_plots):
        makePlots.loss(self.loss_history, save_plots, self.fig_storage)

        n_shape = 10_000
        self.make_collocation(n_shape)

        x_ex, y_ex = metricTensors.apply_symplecto(
            self.x_collocation, self.y_collocation, name=self.name_symplecto
        )
        x_net, y_net = self.apply_symplecto(self.x_collocation, self.y_collocation)

        makePlots.shape(
            x_net.detach().cpu(), y_net.detach().cpu(), save_plots, self.fig_storage
        )
        makePlots.shape_error(
            x_net.detach().cpu(),
            y_net.detach().cpu(),
            x_ex.detach().cpu(),
            y_ex.detach().cpu(),
            save_plots,
            self.fig_storage,
            title=f"Hausdorff error: {self.get_hausdorff_error():5.2e}",
        )
