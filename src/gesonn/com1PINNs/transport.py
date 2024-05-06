"""
Author:
    A BELIERES FRENDO (IRMA)
Date:
    05/05/2023

Solve poisson EDP in a shape genereted by a symplectomorphism
-Delta u = f
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
from gesonn.com1PINNs import boundary_conditions as bc
from gesonn.com1PINNs import metricTensors, sourceTerms

# local imports
from gesonn.out1Plot import makePlots

try:
    import torchinfo

    no_torchinfo = False
except ModuleNotFoundError:
    no_torchinfo = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is poisson.py")

# ----------------------------------------------------------------------
#   CLASSE NET - HERITIERE DE NN.DATAPARALLEL
# ----------------------------------------------------------------------


class PDE_Forward(nn.DataParallel):
    # constructeur
    def __init__(self, layer_sizes):
        super(PDE_Forward, self).__init__(nn.Module())

        self.hidden_layers = []
        for l1, l2 in zip(layer_sizes[:-1], layer_sizes[+1:]):
            self.hidden_layers.append(nn.Linear(l1, l2).double())
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.output_layer = nn.Linear(layer_sizes[-1], 1).double()

    # forward function -> defines the network structure
    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)

        layer_output = torch.tanh(self.hidden_layers[0](inputs))

        for hidden_layer in self.hidden_layers[1:]:
            layer_output = torch.tanh(hidden_layer(layer_output))

        # return torch.sigmoid(self.output_layer(layer_output))
        return self.output_layer(layer_output)


# ----------------------------------------------------------------------
#   CLASSE NETWORK - RESEAU DE NEURONES
#   approximation d'un symplectomorphisme
# ----------------------------------------------------------------------


class PINNs:
    DEFAULT_PINNS_DICT = {
        "learning_rate": 5e-3,
        "layer_sizes": [2, 10, 20, 20, 10, 1],
        "x_min": 0,
        "x_max": 1,
        "t_min": 0,
        "t_max": 1,
        "file_name": "default",
        "symplecto_name": None,
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "dirichlet_homgene",
    }

    # constructeur
    def __init__(self, **kwargs):
        PINNsDict = kwargs.get("PINNsDict", self.DEFAULT_PINNS_DICT)

        if PINNsDict.get("learning_rate") == None:
            PINNsDict["learning_rate"] = self.DEFAULT_PINNS_DICT["learning_rate"]
        if PINNsDict.get("layer_sizes") == None:
            PINNsDict["layer_sizes"] = self.DEFAULT_PINNS_DICT["layer_sizes"]
        if PINNsDict.get("x_min") == None:
            PINNsDict["x_min"] = self.DEFAULT_PINNS_DICT["x_min"]
        if PINNsDict.get("x_max") == None:
            PINNsDict["x_max"] = self.DEFAULT_PINNS_DICT["x_max"]
        if PINNsDict.get("t_min") == None:
            PINNsDict["t_min"] = self.DEFAULT_PINNS_DICT["t_min"]
        if PINNsDict.get("t_max") == None:
            PINNsDict["t_max"] = self.DEFAULT_PINNS_DICT["t_max"]
        if PINNsDict.get("file_name") == None:
            PINNsDict["file_name"] = self.DEFAULT_PINNS_DICT["file_name"]
        if PINNsDict.get("symplecto_name") == None:
            PINNsDict["symplecto_name"] = self.DEFAULT_PINNS_DICT["symplecto_name"]
        if PINNsDict.get("source_term") == None:
            PINNsDict["source_term"] = self.DEFAULT_PINNS_DICT["source_term"]
        if PINNsDict.get("boundary_condition") == None:
            PINNsDict["boundary_condition"] = self.DEFAULT_PINNS_DICT[
                "boundary_condition"
            ]
        if PINNsDict.get("to_be_trained") == None:
            PINNsDict["to_be_trained"] = self.DEFAULT_PINNS_DICT["to_be_trained"]

        # Storage file
        self.file_name = (
            "./../../../outputs/PINNs/net/" + PINNsDict["file_name"] + ".pth"
        )
        self.fig_storage = "./../outputs/PINNs/img/" + PINNsDict["file_name"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = os.path.join(script_dir, self.file_name)
        # Learning rate
        self.learning_rate = PINNsDict["learning_rate"]
        # Layer sizes
        self.layer_sizes = PINNsDict["layer_sizes"]
        # Geometry of the shape
        self.x_min, self.x_max = PINNsDict["x_min"], PINNsDict["x_max"]
        self.t_min, self.t_max = PINNsDict["t_min"], PINNsDict["t_max"]
        self.name_symplecto = PINNsDict["symplecto_name"]
        # Source term of the Poisson problem
        self.source_term = PINNsDict["source_term"]
        # Boundary condition of the Poisson problem
        self.boundary_condition = PINNsDict["boundary_condition"]

        self.create_network()
        self.load(self.file_name)

        self.to_be_trained = PINNsDict["to_be_trained"]

        # flag for saving results
        self.save_results = kwargs.get("save_results", False)

    def create_network(self):
        # on utilise l'optimizer Adam

        # réseau associé à la solution de l'EDP
        self.rho_net = nn.DataParallel(PDE_Forward(self.layer_sizes)).to(device)
        self.rho_optimizer = torch.optim.Adam(
            self.rho_net.parameters(), lr=self.learning_rate
        )

    def load(self, file_name):
        self.loss_history = []

        try:
            try:
                checkpoint = torch.load(file_name)
            except RuntimeError:
                checkpoint = torch.load(file_name, map_location=torch.device("cpu"))

            self.rho_net.load_state_dict(checkpoint["u_model_state_dict"])
            self.rho_optimizer.load_state_dict(checkpoint["rho_optimizer_state_dict"])

            self.loss = checkpoint["loss"]

            try:
                self.loss_history = checkpoint["loss_history"]
            except KeyError:
                pass

            self.to_be_trained = False

        except FileNotFoundError:
            self.to_be_trained = True
            print("POISSON : network was not loaded from file: training needed")

    def get_physical_parameters(self):
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "t_min": self.t_min,
            "t_max": self.t_max,
        }

    @staticmethod
    def save(
        file_name,
        epoch,
        rho_net_state,
        rho_optimizer_state,
        loss,
        loss_history,
    ):
        torch.save(
            {
                epoch: epoch,
                "u_model_state_dict": rho_net_state,
                "rho_optimizer_state_dict": rho_optimizer_state,
                "loss": loss,
                "loss_history": loss_history,
            },
            file_name,
        )

    def get_res(self, x, t):
        rho = self.get_rho(x, t)

        c = self.get_c(x)

        dx_rho = torch.autograd.grad(rho.sum(), x, create_graph=True)[0]
        dt_rho = torch.autograd.grad(rho.sum(), t, create_graph=True)[0]

        return (dt_rho + c * dx_rho)**2
        
        # fv à revoir ...
        # rho_2 = self.get_rho(x, t)**2
        # dx_rho_2 = torch.autograd.grad(rho_2.sum(), x, create_graph=True)[0]
        # dt_rho_2 = torch.autograd.grad(rho_2.sum(), t, create_graph=True)[0]
        # return dt_rho_2 + dx_rho_2
    
    def get_ini_cond(self, x):
        rho_0 = self.get_rho(x, torch.zeros_like(x))
        rho_0_target = torch.ones_like(x) * (x<0.55) * (x>0.45)

        return (rho_0-rho_0_target)**2

    def get_c(self, x):
        return torch.ones_like(x) * (x>0.5) - torch.ones_like(x) * (x<0.5)

    def get_rho(self, x, t):
        return self.rho_net(x, t) * x * (1-x)#(self.rho_net(x, t) * self.bc_mul_t(x, t) + self.bc_add_t(x))# * self.bc_mul_x(x)
    
    def bc_mul_t(self, x, t):
        return  t * x * (x - 1)
    
    def bc_add_t(self, x):
        return torch.ones_like(x) * (x<0.55) * (x>0.45 )
    
    def bc_mul_x(self, x):
        return x * (x - 1)

    @staticmethod
    def random(min_value, max_value, shape, requires_grad=False, device=device):
        random_numbers = torch.rand(
            shape, device=device, dtype=torch.double, requires_grad=requires_grad
        )
        return min_value + (max_value - min_value) * random_numbers

    def make_collocation(self, n_collocation):
        shape = (n_collocation, 1)

        self.x_collocation = self.random(
            self.x_min**2, self.x_max**2, shape, requires_grad=True
        )
        self.t_collocation = self.random(
            self.t_min, self.t_max, shape, requires_grad=True
        )

    def train(self, **kwargs):
        # nombre de pas de descente
        epochs = kwargs.get("epochs", 500)
        # nombre de pts tirés pour monte-carlo
        n_collocation = kwargs.get("n_collocation", 10000)

        plot_history = kwargs.get("plot_history", False)

        # trucs de sauvegarde
        try:
            best_loss_value = self.loss.item()
        except AttributeError:
            best_loss_value = 1e10

        # boucle principale de la descnet ede gradient
        tps1 = time.time()
        for epoch in range(epochs):
            # mise à 0 du gradient
            self.rho_optimizer.zero_grad()

            # mise à 0 de la loss
            self.loss = torch.tensor([0.0], device=device)

            # Loss based on PDE
            if n_collocation > 0:
                self.make_collocation(n_collocation)

                res = self.get_res(self.x_collocation, self.t_collocation)
                ini_cond = self.get_ini_cond(self.x_collocation)
                self.loss = (res + 1e2 * ini_cond).sum() / n_collocation
                # self.loss = res.sum() / n_collocation

            self.loss.backward()
            self.rho_optimizer.step()

            self.loss_history.append(self.loss.item())

            if epoch % 500 == 0:
                print(f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}")
                try:
                    self.save(
                        self.file_name,
                        epoch,
                        best_rho_net,
                        best_rho_optimizer,
                        best_loss,
                        self.loss_history,
                    )
                except NameError:
                    pass

            if self.loss.item() < best_loss_value:
                print(f"epoch {epoch: 5d}:    best loss = {self.loss.item():5.2e}")
                best_loss = self.loss.clone()
                best_loss_value = best_loss.item()
                best_rho_net = copy.deepcopy(self.rho_net.state_dict())
                best_rho_optimizer = copy.deepcopy(self.rho_optimizer.state_dict())
        tps2 = time.time()

        print(f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}")

        try:
            self.save(
                self.file_name,
                epoch,
                best_rho_net,
                best_rho_optimizer,
                best_loss,
                self.loss_history,
            )
            self.load(self.file_name)
        except UnboundLocalError:
            pass

        if plot_history:
            self.plot_result()

        return tps2 - tps1

    def plot_result(self):

        import matplotlib.pyplot as plt

        n = 101
        plt.figure()
        for i in range(n):
            
            x = torch.linspace(0, 1, 1000, dtype=torch.float64)[:,None]
            rho_0_target = torch.ones_like(x) * (x<0.55) * (x>0.45)
            t = i/(n-1) * torch.ones_like(x)
            rho = self.get_rho(x, t)

            plt.title(f"densité à t = {i/(n-1):3.2e}")
            plt.plot(x.detach().cpu(), rho.detach().cpu())
            plt.plot(x.detach().cpu(), rho_0_target.detach().cpu(), "--")
            plt.pause(0.1)
            plt.clf()

        plt.close()
        plt.title(f"Loss function")
        plt.plot(self.loss_history)
        plt.show()
