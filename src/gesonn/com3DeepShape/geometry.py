"""
Author:
    A BELIERES FRENDO (ENSTA Paris)
Date:
    05/05/2023

To run:
    python3 deepDirichlet.py

ML for shape optimization
Inspired from a code given by V MICHEL DANSAC (INRIA)
"""


# %%


# ----------------------------------------------------------------------
#   IMPORTS - MACHINE CONFIGURATION
# ----------------------------------------------------------------------

# imports
import os
import copy
import time
import torch
import torch.nn as nn
import pandas as pd

# local imports
from gesonn.com1PINNs import poisson
from gesonn.com2SympNets import G
from gesonn.out1Plot import makePlots
from gesonn.com1PINNs import boundary_conditions as bc
from gesonn.com1PINNs import sourceTerms

try:
    import torchinfo

    no_torchinfo = False
except ModuleNotFoundError:
    no_torchinfo = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is deepGeometry.py")


# ----------------------------------------------------------------------
#   CLASSE NETWORK - RESEAU DE NEURONES
# ----------------------------------------------------------------------


class Geo_Net:
    DEFAULT_DEEP_GEO_DICT = {
        "pde_learning_rate": 1e-2,
        "sympnet_learning_rate": 1e-2,
        "layer_sizes": [2, 10, 20, 10, 1],
        "nb_of_networks": 2,
        "networks_size": 5,
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "default",
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "dirichlet_homogene",
    }

    # constructeur
    def __init__(self, **kwargs):
        deepGeoDict = kwargs.get("deepGeoDict", self.DEFAULT_DEEP_GEO_DICT)

        if deepGeoDict.get("pde_learning_rate") == None:
            deepGeoDict["pde_learning_rate"] = self.DEFAULT_DEEP_GEO_DICT["pde_learning_rate"]
        if deepGeoDict.get("sympnet_learning_rate") == None:
            deepGeoDict["sympnet_learning_rate"] = self.DEFAULT_DEEP_GEO_DICT["sympnet_learning_rate"]
        if deepGeoDict.get("layer_sizes") == None:
            deepGeoDict["layer_sizes"] = self.DEFAULT_DEEP_GEO_DICT["layer_sizes"]
        if deepGeoDict.get("nb_of_networks") == None:
            deepGeoDict["nb_of_networks"] = self.DEFAULT_DEEP_GEO_DICT[
                "nb_of_networks"
            ]
        if deepGeoDict.get("networks_size") == None:
            deepGeoDict["networks_size"] = self.DEFAULT_DEEP_GEO_DICT["networks_size"]
        if deepGeoDict.get("rho_min") == None:
            deepGeoDict["rho_min"] = self.DEFAULT_DEEP_GEO_DICT["rho_min"]
        if deepGeoDict.get("rho_max") == None:
            deepGeoDict["rho_max"] = self.DEFAULT_DEEP_GEO_DICT["rho_max"]
        if deepGeoDict.get("file_name") == None:
            deepGeoDict["file_name"] = self.DEFAULT_DEEP_GEO_DICT["file_name"]
        if deepGeoDict.get("source_term") == None:
            deepGeoDict["source_term"] = self.DEFAULT_DEEP_GEO_DICT["source_term"]
        if deepGeoDict.get("boundary_condition") == None:
            deepGeoDict["boundary_condition"] = self.DEFAULT_DEEP_GEO_DICT[
                "boundary_condition"
            ]
        if deepGeoDict.get("to_be_trained") == None:
            deepGeoDict["to_be_trained"] = self.DEFAULT_DEEP_GEO_DICT["to_be_trained"]


        # Storage file
        self.file_name = (
            "./../../../outputs/deepShape/net/" + deepGeoDict["file_name"] + ".pth"
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = os.path.join(script_dir, self.file_name)
        # Learning rate
        self.pde_learning_rate = deepGeoDict["pde_learning_rate"]
        self.sympnet_learning_rate = deepGeoDict["sympnet_learning_rate"]
        # Layer parameters
        self.layer_sizes = deepGeoDict["layer_sizes"]
        self.nb_of_networks = deepGeoDict["nb_of_networks"]
        self.networks_size = deepGeoDict["networks_size"]
        # Geometry of the shape
        self.rho_min, self.rho_max = deepGeoDict["rho_min"], deepGeoDict["rho_max"]
        self.theta_min, self.theta_max = 0, 2 * torch.pi
        self.Vol = torch.pi * self.rho_max**2
        # Source term of the Poisson problem
        self.source_term = deepGeoDict["source_term"]
        # Boundary condition of the Poisson problem
        self.boundary_condition = deepGeoDict["boundary_condition"]

        self.create_networks()
        self.load(self.file_name)

        self.to_be_trained = deepGeoDict["to_be_trained"]

    def sympnet_layer_append(self, nets, optims, i):
        nets.append(nn.DataParallel(G.Symp_Net_Forward(self.networks_size)).to(device))
        optims.append(
            torch.optim.Adam(nets[i].parameters(), lr=self.sympnet_learning_rate)
        )

    def create_networks(self):
        # réseau relatif au symplecto
        self.up_nets = []
        self.down_nets = []
        self.up_optimizers = []
        self.down_optimizers = []
        for i in range(self.nb_of_networks):
            self.sympnet_layer_append(self.up_nets, self.up_optimizers, i)
            self.sympnet_layer_append(self.down_nets, self.down_optimizers, i)
        # réseau associé à la solution de l'EDP
        self.u_net = nn.DataParallel(poisson.PDE_Forward(self.layer_sizes)).to(device)
        self.u_optimizer = torch.optim.Adam(
            self.u_net.parameters(), lr=self.pde_learning_rate
        )

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
            print("network was not loaded from file: training needed")

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
        u_net_state,
        u_optimizer_state,
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
                "u_model_state_dict": u_net_state,
                "u_optimizer_state_dict": u_optimizer_state,
                "loss": loss,
                "loss_history": loss_history,
                "nb_of_networks": nb_of_networks,
                "networks_size": networks_size,
            },
            file_name,
        )

    def get_metric_tensor(self, x, y):
        # il faut calculer :
        # A = J_T^{-t}*J_T^{-1}

        T = self.apply_symplecto(x, y)

        J_a = torch.autograd.grad(T[0].sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(T[0].sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(T[1].sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(T[1].sum(), y, create_graph=True)[0]

        fac = (J_a * J_d - J_b * J_c) ** 2
        A_a = (J_d**2 + J_b**2) / fac
        A_b = -(J_c * J_d + J_a * J_b) / fac
        A_c = -(J_c * J_d + J_a * J_b) / fac
        A_d = (J_c**2 + J_a**2) / fac

        return A_a, A_b, A_c, A_d

    def get_dn_u(self, x, y):
        xT, yT = self.apply_symplecto(x, y)

        J_a = torch.autograd.grad(xT.sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(xT.sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(yT.sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(yT.sum(), y, create_graph=True)[0]

        det = J_a * J_d - J_b * J_c
        a, b, c, d = det * J_d, -det * J_c, -det * J_b, det * J_a

        u = self.get_u(x, y)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        Jt_dx_u = a * dx_u + b * dy_u
        Jt_dy_u = c * dx_u + d * dy_u

        nx, ny = self.get_n(x, y)

        return Jt_dx_u * nx + Jt_dy_u * ny, nx, ny

    def get_n(self, x, y):
        tx, ty = -y, x

        xT, yT = self.apply_symplecto(x, y)
        J_a = torch.autograd.grad(xT.sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(xT.sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(yT.sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(yT.sum(), y, create_graph=True)[0]
        txT, tyT = J_a * tx + J_b * ty, J_c * tx + J_d * ty
        nxT, nyT = tyT, -txT
        normT = torch.sqrt(nxT**2 + nyT**2)
        nxT, nyT = nxT / normT, nyT / normT

        return nxT, nyT

    def get_avg_dn_u(self, x, y, n):
        dn_u, _, _ = self.get_dn_u(x, y)
        avg_dn_u = dn_u.sum() / n
        return avg_dn_u

    def left_hand_term(self, x, y):
        u = self.get_u(x, y)
        a, b, c, d = self.get_metric_tensor(x, y)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        A_grad_u_grad_u = (a * dx_u + b * dy_u) * dx_u + (c * dx_u + d * dy_u) * dy_u

        return A_grad_u_grad_u

    def right_hand_term(self, x, y):
        u = self.get_u(x, y)

        # terme source
        f = sourceTerms.get_f(*self.apply_symplecto(x, y), name=self.source_term)

        return f * u

    def apply_symplecto(self, x, y):
        x_net, y_net = x, y
        for i in range(self.nb_of_networks):
            x_net, y_net = x_net + self.up_nets[i](y_net), y_net
            x_net, y_net = x_net, y_net + self.down_nets[i](x_net)
        return x_net, y_net

    def get_u(self, x, y):
        return bc.apply_BC(
            self.u_net(*self.apply_symplecto(x, y)),
            x,
            y,
            self.rho_min,
            self.rho_max,
            name =self.boundary_condition,
        )

    @staticmethod
    def random(min_value, max_value, shape, requires_grad=False, device=device):
        random_numbers = torch.rand(
            shape, device=device, dtype=torch.double, requires_grad=requires_grad
        )
        return min_value + (max_value - min_value) * random_numbers

    def make_collocation(self, n_collocation):
        shape = (n_collocation, 1)

        rho_collocation = torch.sqrt(
            self.random(self.rho_min**2, self.rho_max**2, shape, requires_grad=True)
        )
        self.theta_collocation = self.random(
            self.theta_min, self.theta_max, shape, requires_grad=True
        )

        self.x_collocation = rho_collocation * torch.cos(self.theta_collocation)
        self.y_collocation = rho_collocation * torch.sin(self.theta_collocation)

    def get_mes_border(self):
        n = 10_000
        theta = torch.linspace(self.theta_min, self.theta_max, n, requires_grad=True)[
            :, None
        ]
        x = self.rho_max * torch.cos(theta)
        y = self.rho_max * torch.sin(theta)
        x, y = self.apply_symplecto(x, y)
        rho = torch.sqrt(x * x + y * y)
        lenghts = torch.sqrt(
            rho[:-1] ** 2 + rho[1:] ** 2 - 2 * (x[:-1] * x[1:] + y[:-1] * y[1:])
        )

        return lenghts.sum()

    def train(self, **kwargs):
        # nombre de pas de descente
        epochs = kwargs.get("epochs", 500)
        # nombre de pts tirés pour monte-carlo
        n_collocation = kwargs.get("n_collocation", 10_000)

        plot_history = kwargs.get("plot_history", False)

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
            self.u_optimizer.zero_grad()

            # mise à 0 de la loss
            self.loss = torch.tensor([0.0], device=device)

            # Loss based on PDE
            if n_collocation > 0:
                self.make_collocation(n_collocation)
                grad_u_2 = self.left_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                )
                fu = self.right_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                )
                dirichlet_loss = 0.5 * grad_u_2 - fu

                self.loss = dirichlet_loss.sum() / n_collocation * self.Vol

            self.loss.backward()
            for i in range(self.nb_of_networks):
                self.up_optimizers[i].step()
                self.down_optimizers[i].step()
            self.u_optimizer.step()

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
                        best_u_net,
                        best_u_optimizer,
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

                best_u_net = copy.deepcopy(self.u_net.state_dict())
                best_u_optimizer = copy.deepcopy(self.u_optimizer.state_dict())
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
                best_u_net,
                best_u_optimizer,
                best_loss,
                self.loss_history,
                self.get_physical_parameters(),
                self.nb_of_networks,
            )
            self.load(self.file_name)
        except UnboundLocalError:
            pass

        if plot_history:
            self.plot_result(epoch)

        return tps2 - tps1

    @staticmethod
    def copy_sympnet(to_be_copied):
        return [copy.deepcopy(copie.state_dict()) for copie in to_be_copied]

    def plot_result(self, derivative=False, random=False):

        makePlots.loss(self.loss_history)

        n_visu = 50_000
        self.make_collocation(n_visu)

        xT, yT = self.apply_symplecto(self.x_collocation, self.y_collocation)
        u_pred = self.get_u(self.x_collocation, self.y_collocation)

        x_border = self.rho_max * torch.cos(self.theta_collocation)
        y_border = self.rho_max * torch.sin(self.theta_collocation)
        xT_border, yT_border = self.apply_symplecto(x_border, y_border)
        dn_u, _, _ = self.get_dn_u(x_border, y_border)


        makePlots.edp(
            xT_border.detach().cpu(), 
            yT_border.detach().cpu(),
            dn_u.detach().cpu(),
            "gradient normal",
        )
        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            sourceTerms.get_f(
                *self.apply_symplecto(self.x_collocation, self.y_collocation),
                name=self.source_term
            ).detach().cpu(),
            "terme source",
        )
        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            u_pred.detach().cpu(),
            "EDP",
        )