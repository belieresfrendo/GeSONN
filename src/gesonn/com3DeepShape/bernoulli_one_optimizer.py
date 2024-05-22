"""
Authors:
    A. BELIERES--FRENDO (IRMA)
    V. MICHEL-DANSAC    (INRIA)
Date:
    2023 - 2024
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
from torch.optim.lr_scheduler import ExponentialLR

# local imports
from gesonn.com1PINNs import boundary_conditions as bc
from gesonn.com1PINNs import poisson
from gesonn.com2SympNets import G
from gesonn.out1Plot import makePlots

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
        "rho_min": 0.5,
        "rho_max": 1,
        "file_name": "default",
        "to_be_trained": True,
        "boundary_condition": "bernoulli",
        "a": 0.6,
        "tikhonov": 0,
    }

    # constructeur
    def __init__(self, **kwargs):
        deepGeoDict = kwargs.get("deepGeoDict", self.DEFAULT_DEEP_GEO_DICT)

        if deepGeoDict.get("pde_learning_rate") == None:
            deepGeoDict["pde_learning_rate"] = self.DEFAULT_DEEP_GEO_DICT[
                "pde_learning_rate"
            ]
        if deepGeoDict.get("sympnet_learning_rate") == None:
            deepGeoDict["sympnet_learning_rate"] = self.DEFAULT_DEEP_GEO_DICT[
                "sympnet_learning_rate"
            ]
        if deepGeoDict.get("layer_sizes") == None:
            deepGeoDict["layer_sizes"] = self.DEFAULT_DEEP_GEO_DICT["layer_sizes"]
        if deepGeoDict.get("nb_of_networks") == None:
            deepGeoDict["nb_of_networks"] = self.DEFAULT_DEEP_GEO_DICT["nb_of_networks"]
        if deepGeoDict.get("networks_size") == None:
            deepGeoDict["networks_size"] = self.DEFAULT_DEEP_GEO_DICT["networks_size"]
        if deepGeoDict.get("rho_min") == None:
            deepGeoDict["rho_min"] = self.DEFAULT_DEEP_GEO_DICT["rho_min"]
        if deepGeoDict.get("rho_max") == None:
            deepGeoDict["rho_max"] = self.DEFAULT_DEEP_GEO_DICT["rho_max"]
        if deepGeoDict.get("file_name") == None:
            deepGeoDict["file_name"] = self.DEFAULT_DEEP_GEO_DICT["file_name"]
        if deepGeoDict.get("boundary_condition") == None:
            deepGeoDict["boundary_condition"] = self.DEFAULT_DEEP_GEO_DICT[
                "boundary_condition"
            ]
        if deepGeoDict.get("a") == None:
            deepGeoDict["a"] = self.DEFAULT_DEEP_GEO_DICT["a"]
        if deepGeoDict.get("tikhonov") == None:
            deepGeoDict["tikhonov"] = self.DEFAULT_PINNS_DICT["tikhonov"]
        if deepGeoDict.get("to_be_trained") == None:
            deepGeoDict["to_be_trained"] = self.DEFAULT_DEEP_GEO_DICT["to_be_trained"]

        # Storage file
        self.file_name = (
            "./../../../outputs/deepShape/net/" + deepGeoDict["file_name"] + ".pth"
        )
        self.fig_storage = "./../outputs/deepShape/img/" + deepGeoDict["file_name"]
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
        # Boundary condition of the Poisson problem
        self.boundary_condition = deepGeoDict["boundary_condition"]
        # thikhonov regularization epsilon parameter
        self.pen_tikhonov = deepGeoDict["tikhonov"]

        # Parameters of the compact set K
        self.a = deepGeoDict["a"]
        self.b = self.rho_min**2 / self.a

        self.create_networks()
        self.load(self.file_name)

        self.to_be_trained = deepGeoDict["to_be_trained"]

    def sympnet_layer_append(self, nets, optims, i):
        nets.append(
            nn.DataParallel(G.Symp_Net_Forward(self.networks_size)).to(device)
        )
        optims.append(
            {"params": nets[i].parameters(), "lr": self.sympnet_learning_rate}
        )

    def create_networks(self):
        # réseau relatif au symplecto
        self.up_nets = []
        self.down_nets = []
        up_optimizers = []
        down_optimizers = []
        for i in range(self.nb_of_networks):
            self.sympnet_layer_append(self.up_nets, up_optimizers, i)
            self.sympnet_layer_append(self.down_nets, down_optimizers, i)
        # réseau associé à la solution de l'EDP
        self.u_net = nn.DataParallel(poisson.PDE_Forward(self.layer_sizes)).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.u_net.parameters(), "lr": self.pde_learning_rate},
                *up_optimizers,
                *down_optimizers,
            ]
        )

    def load_sympnet_layer(self, nets, checkpoint_nets, checkpoint):
        nets_state_dicts = checkpoint[checkpoint_nets]
        i = 0
        for _, state_dict in zip(nets, nets_state_dicts):
            nets[i].load_state_dict(state_dict)
            i += 1

    def load(self, file_name):
        self.loss_history = []

        try:
            try:
                checkpoint = torch.load(file_name)
            except RuntimeError:
                checkpoint = torch.load(file_name, map_location=torch.device("cpu"))

            self.load_sympnet_layer(self.up_nets, "up_models_state_dict", checkpoint)
            self.load_sympnet_layer(
                self.down_nets, "down_models_state_dict", checkpoint
            )

            self.u_net.load_state_dict(checkpoint["u_model_state_dict"])
            # print(self.u_net(torch.tensor([0.0], dtype=float)[:, None], torch.tensor([0.0], dtype=float)[:, None]).item())

            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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
            "a": self.a,
            "b": self.b,
        }

    @staticmethod
    def save_sympnet_layer(state_dicts):
        return [state_dict for state_dict in state_dicts]

    def save(
        self,
        file_name,
        epoch,
        up_nets_state_dict,
        down_nets_state_dict,
        u_net_state,
        optimizer_state,
        loss,
        loss_history,
        nb_of_networks,
        networks_size,
    ):
        torch.save(
            {
                epoch: epoch,
                "up_models_state_dict": self.save_sympnet_layer(up_nets_state_dict),
                "down_models_state_dict": self.save_sympnet_layer(down_nets_state_dict),
                "u_model_state_dict": u_net_state,
                "optimizer_state_dict": optimizer_state,
                "loss": loss,
                "loss_history": loss_history,
                "nb_of_networks": nb_of_networks,
                "networks_size": networks_size,
            },
            file_name,
        )

    def get_jacobian_matrix(self, x, y):
        xT, yT = self.apply_symplecto(x, y)

        J_a = torch.autograd.grad(xT.sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(xT.sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(yT.sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(yT.sum(), y, create_graph=True)[0]

        return J_a, J_b, J_c, J_d

    def get_metric_tensor(self, x, y):
        # il faut calculer :
        # A = J_T^{-t}*J_T^{-1}

        J_a, J_b, J_c, J_d = self.get_jacobian_matrix(x, y)

        fac = (J_a * J_d - J_b * J_c) ** 2
        A_a = (J_d**2 + J_b**2) / fac
        A_b = -(J_c * J_d + J_a * J_b) / fac
        A_c = -(J_c * J_d + J_a * J_b) / fac
        A_d = (J_c**2 + J_a**2) / fac

        return A_a, A_b, A_c, A_d

    def get_dn_u(self, x, y):
        J_a, J_b, J_c, J_d = self.get_jacobian_matrix(x, y)

        det = J_a * J_d - J_b * J_c
        a, b, c, d = det * J_d, -det * J_c, -det * J_b, det * J_a

        u = self.get_u(x, y)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        Jt_dx_u = a * dx_u + b * dy_u
        Jt_dy_u = c * dx_u + d * dy_u

        return torch.sqrt(Jt_dx_u**2 + Jt_dy_u**2)

    def get_optimality_condition(self, n=10_000):
        theta = torch.linspace(
            0, 2 * torch.pi, n, requires_grad=True, device=device, dtype=torch.float64
        )[:, None]
        x = self.rho_max * torch.cos(theta)
        y = self.rho_max * torch.sin(theta)
        xT, yT = self.apply_symplecto(x, y)
        J_a, J_b, J_c, J_d = self.get_jacobian_matrix(x, y)
        d_sigma = torch.sqrt((J_b * x - J_a * y) ** 2 + (J_d * x - J_c * y) ** 2)
        dn_u = self.get_dn_u(x, y)

        return dn_u, xT, yT

        avg_dn_u = (dn_u * d_sigma).sum() / n
        return torch.abs(avg_dn_u - dn_u), xT, yT

    def left_hand_term(self, x, y):
        u = self.get_u(x, y)
        a, b, c, d = self.get_metric_tensor(x, y)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        A_grad_u_grad_u = (a * dx_u + b * dy_u) * dx_u + (c * dx_u + d * dy_u) * dy_u

        return A_grad_u_grad_u
    
    def get_res(self, x, y):
        u = self.get_u(x, y)
        a, b, c, d = self.get_metric_tensor(x, y)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        A_grad_u = (a * dx_u + b * dy_u)
        A_grad_u = (c * dx_u + d * dy_u)
        dx_A_grad_u = torch.autograd.grad(A_grad_u.sum(), x, create_graph=True)[0]
        dy_A_grad_u = torch.autograd.grad(A_grad_u.sum(), y, create_graph=True)[0]

        return - dx_A_grad_u - dy_A_grad_u

    def apply_symplecto(self, x, y):
        x, y = x, y
        for i in range(self.nb_of_networks):
            x, y = x + self.up_nets[i](y), y
            x, y = x, y + self.down_nets[i](x)
        return x, y

    def apply_inverse_symplecto(self, x, y):
        for i in range(self.nb_of_networks):
            y = y - self.down_nets[self.nb_of_networks - 1 - i](x)
            x = x - self.up_nets[self.nb_of_networks - 1 - i](y)
        return x, y

    def get_u(self, x, y):
        xT, yT = self.apply_symplecto(x, y)

        return bc.apply_BC(
            self.u_net(xT, yT),
            x,
            y,
            self.rho_min,
            self.rho_max,
            name=self.boundary_condition,
            xT=xT,
            yT=yT,
            a=self.a,
            b=self.b,
        )

    def apply_rejet_kompact(self, x, y):
        xT, yT = self.apply_symplecto(x, y)
        condition = (xT / self.a) ** 2 + (yT / self.b) ** 2 >= 1
        xT, yT = (
            xT[condition][:, None],
            yT[condition][:, None],
        )
        return self.apply_inverse_symplecto(xT, yT)

    @staticmethod
    def random(min_value, max_value, shape, requires_grad=False, device=device):
        random_numbers = torch.rand(
            shape, device=device, dtype=torch.double, requires_grad=requires_grad
        )
        return min_value + (max_value - min_value) * random_numbers

    def make_collocation(self, n_collocation):
        shape = (n_collocation, 1)

        rho_collocation = torch.sqrt(
            self.random(0, self.rho_max**2, shape, requires_grad=True)
        )
        theta_collocation = self.random(
            self.theta_min, self.theta_max, shape, requires_grad=True
        )

        self.x_collocation = rho_collocation * torch.cos(theta_collocation)
        self.y_collocation = rho_collocation * torch.sin(theta_collocation)

        self.x_collocation, self.y_collocation = self.apply_rejet_kompact(
            self.x_collocation, self.y_collocation
        )

    def make_border_collocation(self, n_collocation):
        shape = (n_collocation, 1)

        theta_collocation = self.random(
            self.theta_min, self.theta_max, shape, requires_grad=True
        )
        self.x_gamma_collocation = self.rho_max * torch.cos(theta_collocation)
        self.y_gamma_collocation = self.rho_max * torch.sin(theta_collocation)

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
        save_plots = kwargs.get("save_plots", False)

        # trucs de sauvegarde ?
        try:
            best_loss_value = self.loss.item()
        except AttributeError:
            best_loss_value = 1e10

        self.scheduler = ExponentialLR(self.optimizer, 0.9999)

        # boucle principale de la descnet ede gradient
        tps1 = time.time()
        for epoch in range(epochs):
            # mise à 0 du gradient
            self.optimizer.zero_grad()

            # mise à 0 de la loss
            self.loss = torch.tensor([0.0], device=device)

            # Loss based on PDE
            if n_collocation > 0:
                self.make_collocation(n_collocation)
                n_pts = self.x_collocation.size()[0]

                grad_u_2 = self.left_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                )

                self.dirichlet_loss = 0.5 * grad_u_2.sum() * self.Vol / n_pts
                loss = self.dirichlet_loss

                if self.pen_tikhonov != 0:
                    self.tikhonov = (grad_u_2 **2).sum() * self.Vol / n_pts
                    loss += self.pen_tikhonov * self.tikhonov

                self.loss = loss

            self.loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.loss_history.append(self.loss.item())

            if epoch % 500 == 0:
                string = f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}, current lr = {self.scheduler.get_lr()[0]:5.2e}"
                if self.pen_tikhonov != 0:
                    string += f", Dirichlet = {self.dirichlet_loss:3.2e}, Tikhonov = {self.tikhonov:3.2e}"
                print(string)

            if self.loss.item() < best_loss_value:
                print(f"epoch {epoch: 5d}:    best loss = {self.loss.item():5.2e}, current lr = {self.scheduler.get_lr()[0]:5.2e}")
                best_loss = self.loss.clone()
                best_loss_value = best_loss.item()
                best_up_nets = self.copy_sympnet(self.up_nets).copy()
                best_down_nets = self.copy_sympnet(self.down_nets).copy()

                best_u_net = copy.deepcopy(self.u_net.state_dict())

                best_optimizer = copy.deepcopy(self.optimizer.state_dict())
                # print(self.u_net(torch.tensor([0.0], dtype=float)[:, None], torch.tensor([0.0], dtype=float)[:, None]).item())

        tps2 = time.time()

        print(f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}, current lr = {self.scheduler.get_lr()[0]:5.2e}")

        try:
            self.save(
                self.file_name,
                epoch,
                best_up_nets,
                best_down_nets,
                best_u_net,
                best_optimizer,
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

        return tps2 - tps1

    @staticmethod
    def copy_sympnet(to_be_copied):
        return [copy.deepcopy(copie.state_dict()) for copie in to_be_copied]

    def plot_result(self, save_plots):

        makePlots.loss(self.loss_history, save_plots, self.fig_storage)

        n_visu = 768
        if device == "cuda":
            n_visu = 124

        makePlots.edp_contour_bernoulli(
            self.rho_max,
            self.a,
            self.b,
            self.get_u,
            lambda x, y: self.apply_symplecto(x, y),
            lambda x, y: self.apply_inverse_symplecto(x, y),
            save_plots,
            self.fig_storage,
            n_visu=n_visu,
        )

        makePlots.optimality_condition(
            self.get_optimality_condition,
            save_plots,
            self.fig_storage,
        )
