"""
Author:
    A BELIERES FRENDO (ENSTA Paris)
Date:
    05/05/2023

To run:
    python3 pdeSymplec.py

ML for symplectomorphism
Inspired from a code given by V MICHEL DANSAC (INRIA)
"""


# %%


# ----------------------------------------------------------------------
#   IMPORTS - MACHINE CONFIGURATION
# ----------------------------------------------------------------------

# imports
import os
import copy
import torch
import torch.nn as nn

# local imports
from soviets.com1PINNs import metricTensors

try:
    import torchinfo

    no_torchinfo = False
except ModuleNotFoundError:
    no_torchinfo = True

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

        self.k1 = torch.nn.parameter.Parameter(
            min_value
            + (max_value - min_value) * torch.rand(n, dtype=torch.double, device=device)
        )
        self.a1 = torch.nn.parameter.Parameter(
            min_value
            + (max_value - min_value) * torch.rand(n, dtype=torch.double, device=device)
        )
        self.b1 = torch.nn.parameter.Parameter(
            min_value
            + (max_value - min_value) * torch.rand(n, dtype=torch.double, device=device)
        )
        # self.k2 = torch.nn.parameter.Parameter(
        #     min_value
        #     + (max_value - min_value) * torch.rand(n, dtype=torch.double, device=device)
        # )
        # self.b2 = torch.nn.parameter.Parameter(
        #     min_value
        #     + (max_value - min_value) * torch.rand(n, dtype=torch.double, device=device)
        # )

    # forward function -> defines the network structure
    def forward(self, x_or_y, mu):
        Kx_or_y = torch.einsum("ik,jk->ijk", self.k1, x_or_y)
        shape_x_or_y = x_or_y.size()
        ones = torch.ones(shape_x_or_y, device=device)
        B1 = torch.einsum("ik,jk->ijk", self.b1, ones)
        A1 = torch.einsum("ik,jk->ijk", self.a1, ones)
        # A2mu = torch.einsum("ik,jk->ijk", self.k2, mu)
        Asigma1 = A1 * torch.tanh(Kx_or_y + B1 + mu)
        return torch.einsum("ik,ijk->jk", self.k1, Asigma1)


# ----------------------------------------------------------------------
#   CLASSE SYMP_NET - RESEAU DE NEURONES
# ----------------------------------------------------------------------


class Symp_Net:
    DEFAULT_SYMPNETS_DICT = {
        "learning_rate": 1e-3,
        "nb_of_networks": 6,
        "networks_size": 10,
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "bizaroid",
        "symplecto_name": "bizaroid",
        "to_be_trained": True,
    }

    # constructeur
    def __init__(self, **kwargs):
        SympNetsDict = kwargs.get("SympNetsDict", self.DEFAULT_SYMPNETS_DICT)

        self.rho_min, self.rho_max = SympNetsDict["rho_min"], SympNetsDict["rho_max"]
        self.theta_min, self.theta_max = 0, 2 * torch.pi
        self.mu_min, self.mu_max = kwargs.get("mu_min", 0.5), kwargs.get("mu_max", 2)

        self.Vol = torch.pi * self.rho_max**2

        self.file_name = (
            "./../data/SympNets/net/GSympNet_" + SympNetsDict["file_name"] + ".pth"
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = os.path.join(script_dir, self.file_name)

        self.learning_rate = SympNetsDict["learning_rate"]

        # taille des différentes couches du réseau de neurones
        self.nb_of_networks = SympNetsDict["nb_of_networks"]
        self.networks_size = SympNetsDict["networks_size"]

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
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
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

    def apply_symplecto(self, x, y, mu):
        for i in range(self.nb_of_networks):
            x = x + self.up_nets[i](y, mu)
            y = y + self.down_nets[i](x, mu)
        return x, y

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
        self.mu_collocation = self.random(
            self.mu_min, self.mu_max, shape, requires_grad=True
        )

        self.zeros = torch.zeros(shape, dtype=torch.double, device=device)

    def train(self, **kwargs):
        # nombre de pas de descente
        epochs = kwargs.get("epochs", 500)
        # nombre de pts tirés pour monte-carlo
        n_collocation = kwargs.get("n_collocation", 10000)

        plot_history = kwargs.get("plot_history", False)

        # trucs de sauvegarde ?
        try:
            best_loss_value = self.loss.item()
        except AttributeError:
            best_loss_value = 1e10

        # boucle principale de la descnet ede gradient
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
                    self.x_collocation,
                    self.y_collocation,
                    self.mu_collocation,
                    name=self.name_symplecto,
                )
                x_net, y_net = self.apply_symplecto(
                    self.x_collocation, self.y_collocation, self.mu_collocation
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
            self.plot_result()

    @staticmethod
    def copy_sympnet(to_be_copied):
        return [copy.deepcopy(copie.state_dict()) for copie in to_be_copied]

    def plot_result(self, derivative=False, random=False):
        import matplotlib.pyplot as plt
        from matplotlib import rc

        rc("font", **{"family": "serif", "serif": ["fontenc"], "size": 15})
        rc("text", usetex=True)

        _, ax = plt.subplots(2, 2)
        ax[0, 0].semilogy(self.loss_history)
        ax[0, 0].set_title("loss history")

        n_shape = 10000
        self.make_collocation(n_shape)
        mu_visu1 = torch.ones((n_shape, 1), requires_grad=True, device=device) * (
            0.25 * self.mu_max + 0.75 * self.mu_min
        )
        mu_visu2 = torch.ones((n_shape, 1), requires_grad=True, device=device) * (
            0.5 * self.mu_max + 0.5 * self.mu_min
        )
        mu_visu3 = torch.ones((n_shape, 1), requires_grad=True, device=device) * (
            0.75 * self.mu_max + 0.25 * self.mu_min
        )

        x_ex1, y_ex1 = metricTensors.apply_symplecto(
            self.x_collocation, self.y_collocation, mu_visu1, name=self.name_symplecto
        )
        x_net1, y_net1 = self.apply_symplecto(
            self.x_collocation, self.y_collocation, mu_visu1
        )
        x_net2, y_net2 = self.apply_symplecto(
            self.x_collocation, self.y_collocation, mu_visu2
        )
        x_net3, y_net3 = self.apply_symplecto(
            self.x_collocation, self.y_collocation, mu_visu3
        )

        ax[0, 1].scatter(
            self.x_collocation.detach().cpu(),
            self.y_collocation.detach().cpu(),
            s=1,
            label="cercle",
        )
        ax[0, 1].set_aspect("equal")
        ax[0, 1].set_title("condition initiale")
        ax[0, 1].legend()

        ax[1, 0].scatter(
            x_net1.detach().cpu(), y_net1.detach().cpu(), s=1, label="prediction"
        )
        ax[1, 0].scatter(
            x_net2.detach().cpu(), y_net2.detach().cpu(), s=1, label="prediction"
        )
        ax[1, 0].scatter(
            x_net3.detach().cpu(), y_net3.detach().cpu(), s=1, label="prediction"
        )
        ax[1, 0].set_aspect("equal")
        ax[1, 0].set_title("transformation approchée")
        ax[1, 0].legend()

        ax[1, 1].scatter(
            x_net1.detach().cpu(),
            y_net1.detach().cpu(),
            s=1,
            c="yellow",
            label="prediction",
        )
        ax[1, 1].scatter(x_ex1.detach().cpu(), y_ex1.detach().cpu(), s=1, label="exact")
        ax[1, 1].set_aspect("equal")
        ax[1, 1].set_title("transformation exacte")
        ax[1, 1].legend()

        plt.show()
