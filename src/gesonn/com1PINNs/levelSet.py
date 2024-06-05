"""
Author:
    A BELIERES FRENDO (IRMA)
Date:
    05/05/2023

Solve poisson EDP in a level-set function
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
from gesonn.com1PINNs import sourceTerms

# local imports
from gesonn.out1Plot import makePlots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is poisson.py")

# ----------------------------------------------------------------------
#   CLASSE NET - HERITIERE DE NN.DATAPARALLEL
# ----------------------------------------------------------------------


class PDE_Forward(nn.DataParallel):
    # constructeur
    def __init__(self, layer_sizes, activation=torch.tanh):
        super(PDE_Forward, self).__init__(nn.Module())

        self.hidden_layers = []
        for l1, l2 in zip(layer_sizes[:-1], layer_sizes[+1:]):
            self.hidden_layers.append(nn.Linear(l1, l2).double())
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.output_layer = nn.Linear(layer_sizes[-1], 1).double()

        self.activation = activation

    # forward function -> defines the network structure
    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)

        layer_output = self.activation(self.hidden_layers[0](inputs))

        for hidden_layer in self.hidden_layers[1:]:
            layer_output = self.activation(hidden_layer(layer_output))

        return self.output_layer(layer_output)


# ----------------------------------------------------------------------
#   CLASSE NETWORK - RESEAU DE NEURONES
# ----------------------------------------------------------------------


class PINNs:
    DEFAULT_PINNS_DICT = {
        "learning_rate": 5e-3,
        "layer_sizes": [2, 10, 20, 20, 10, 1],
        "lx": 5,
        "ly": 5,
        "file_name": "default",
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "dirichlet_homgene",
    }

    # constructeur
    def __init__(self, **kwargs):
        PINNsDict = kwargs.get("PINNsDict", self.DEFAULT_PINNS_DICT)

        if PINNsDict.get("learning_rate") is None:
            PINNsDict["learning_rate"] = self.DEFAULT_PINNS_DICT["learning_rate"]
        if PINNsDict.get("layer_sizes") is None:
            PINNsDict["layer_sizes"] = self.DEFAULT_PINNS_DICT["layer_sizes"]
        if PINNsDict.get("lx") is None:
            PINNsDict["lx"] = self.DEFAULT_PINNS_DICT["lx"]
        if PINNsDict.get("ly") is None:
            PINNsDict["ly"] = self.DEFAULT_PINNS_DICT["ly"]
        if PINNsDict.get("file_name") is None:
            PINNsDict["file_name"] = self.DEFAULT_PINNS_DICT["file_name"]
        if PINNsDict.get("source_term") is None:
            PINNsDict["source_term"] = self.DEFAULT_PINNS_DICT["source_term"]
        if PINNsDict.get("boundary_condition") is None:
            PINNsDict["boundary_condition"] = self.DEFAULT_PINNS_DICT[
                "boundary_condition"
            ]
        if PINNsDict.get("to_be_trained") is None:
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
        self.lx, self.ly = PINNsDict["lx"], PINNsDict["ly"]
        self.theta_min, self.theta_max = 0, 2 * torch.pi
        self.vol_D = self.lx * self.ly
        # Source term of the Poisson problem
        self.source_term = PINNsDict["source_term"]
        # Boundary condition of the Poisson problem
        self.boundary_condition = PINNsDict["boundary_condition"]
        # thikhonov regularization epsilon parameter

        self.create_network()
        self.load(self.file_name)

        self.to_be_trained = PINNsDict["to_be_trained"]

        # flag for saving results
        self.save_results = kwargs.get("save_results", False)

    def create_network(self):
        # on utilise l'optimizer Adam

        # réseau associé à la solution de l'EDP
        self.u_net = nn.DataParallel(PDE_Forward(self.layer_sizes)).to(device)
        self.u_optimizer = torch.optim.Adam(
            self.u_net.parameters(), lr=self.learning_rate
        )

    def load(self, file_name):
        self.loss_history = []

        try:
            try:
                checkpoint = torch.load(file_name)
            except RuntimeError:
                checkpoint = torch.load(file_name, map_location=torch.device("cpu"))

            self.u_net.load_state_dict(checkpoint["u_model_state_dict"])
            self.u_optimizer.load_state_dict(checkpoint["u_optimizer_state_dict"])

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
            "lx": self.lx,
            "ly": self.ly,
        }

    @staticmethod
    def save(
        file_name,
        epoch,
        u_net_state,
        u_optimizer_state,
        loss,
        loss_history,
    ):
        torch.save(
            {
                epoch: epoch,
                "u_model_state_dict": u_net_state,
                "u_optimizer_state_dict": u_optimizer_state,
                "loss": loss,
                "loss_history": loss_history,
            },
            file_name,
        )

    def get_fv(self, x, y):
        u = self.get_u(x, y)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        grad_u_2 = dx_u**2 + dy_u**2

        # terme source
        f = sourceTerms.get_f(x, y, name=self.source_term)
        fu = f * u

        return 0.5 * grad_u_2 - fu

    def get_u(self, x, y):
        # return (
        #     self.u_net(x, y)
        #     * (x - 0.5 * self.lx)
        #     * (x + 0.5 * self.lx)
        #     * (y - 0.5 * self.ly)
        #     * (y + 0.5 * self.ly)
        # )
        return self.u_net(x, y) * self.get_phi(x, y)

    @staticmethod
    def get_phi(x, y):
        # x = x - 0.5 * y**2 + 0.3 * torch.sin(1 / 0.5 * y) - 0.2 * torch.sin(8.0 * y)
        # y = y + 0.2 * 0.5 * x + 0.12 * torch.cos(x)
        return 1 - x**2 - y**2

    @staticmethod
    def random(min_value, max_value, shape, requires_grad=False, device=device):
        random_numbers = torch.rand(
            shape, device=device, dtype=torch.double, requires_grad=requires_grad
        )
        return min_value + (max_value - min_value) * random_numbers

    def make_collocation(self, n_collocation):
        shape = (n_collocation, 1)

        x = self.random(-0.5 * self.lx, 0.5 * self.lx, shape, requires_grad=True)
        y = self.random(-0.5 * self.ly, 0.5 * self.ly, shape, requires_grad=True)

        in_bool = self.get_phi(x, y) >= 0
        out_bool = self.get_phi(x, y) < 0

        self.x_in, self.y_in = x[in_bool][:, None], y[in_bool][:, None]
        self.x_out, self.y_out = x[out_bool][:, None], y[out_bool][:, None]

        self.n_in = self.x_in.size()[0]
        self.n_out = self.x_out.size()[0]
        frac_vol_in = self.n_in / n_collocation
        self.vol_in = frac_vol_in * self.vol_D
        self.vol_out = self.vol_D - self.vol_in

    def train(self, **kwargs):
        # nombre de pas de descente
        epochs = kwargs.get("epochs", 500)
        # nombre de pts tirés pour monte-carlo
        n_collocation = kwargs.get("n_collocation", 10000)

        plot_history = kwargs.get("plot_history", False)
        save_plots = kwargs.get("save_plots", False)

        # trucs de sauvegarde
        try:
            best_loss_value = self.loss.item()
        except AttributeError:
            best_loss_value = 1e10

        # boucle principale de la descnet ede gradient
        tps1 = time.time()
        for epoch in range(epochs):
            # mise à 0 du gradient
            self.u_optimizer.zero_grad()

            # mise à 0 de la loss
            self.loss = torch.tensor([0.0], device=device)

            # Loss based on PDE
            if n_collocation > 0:
                self.make_collocation(n_collocation)

                integrand = self.get_fv(
                    self.x_in,
                    self.y_in,
                )

                # u_out = self.get_u(self.x_out, self.y_out) ** 2

                self.dirichlet_loss = integrand.sum() * self.vol_in / self.n_in
                # loss_out = u_out.sum() * self.vol_out / self.n_out
                self.loss = self.dirichlet_loss #+ 1e3 * loss_out

            self.loss.backward()
            self.u_optimizer.step()

            self.loss_history.append(self.loss.item())

            if epoch % 500 == 0:
                print(f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}")
                try:
                    self.save(
                        self.file_name,
                        epoch,
                        best_u_net,
                        best_u_optimizer,
                        best_loss,
                        self.loss_history,
                    )
                except NameError:
                    pass

            if self.loss.item() < best_loss_value:
                print(f"epoch {epoch: 5d}:    best loss = {self.loss.item():5.2e}")
                best_loss = self.loss.clone()
                best_loss_value = best_loss.item()
                best_u_net = copy.deepcopy(self.u_net.state_dict())
                best_u_optimizer = copy.deepcopy(self.u_optimizer.state_dict())
        tps2 = time.time()

        print(f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}")

        try:
            self.save(
                self.file_name,
                epoch,
                best_u_net,
                best_u_optimizer,
                best_loss,
                self.loss_history,
            )
        except UnboundLocalError:
            pass

        self.load(self.file_name)

        if plot_history:
            self.plot_result(save_plots)

        return tps2 - tps1

    def plot_result(self, save_plots):
        import matplotlib.pyplot as plt

        makePlots.loss(self.loss_history, save_plots, self.fig_storage)

        n_visu = 500_000
        self.make_collocation(n_visu)

        u = self.get_u(self.x_in, self.y_in)

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.scatter(
            self.x_in.detach().cpu(),
            self.y_in.detach().cpu(),
            s=1,
            c=u.detach().cpu(),
            cmap="gist_ncar",
        )
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax)

        plt.plot()
