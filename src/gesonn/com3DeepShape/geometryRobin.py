"""
Author:
    A BELIERES FRENDO (ENSTA Paris)
Date:
    05/05/2023

ML for shape optimization
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

# local imports
from gesonn.com1PINNs import boundary_conditions as bc
from gesonn.com1PINNs import poisson, sourceTerms
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
        "rho_max": 1,
        "file_name": "default",
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "homogeneous_dirichlet",
        "sympnet_activation": torch.sigmoid,
        "pinn_activation": torch.tanh,
    }

    # constructeur
    def __init__(self, **kwargs):
        deepGeoDict = kwargs.get("deepGeoDict", self.DEFAULT_DEEP_GEO_DICT)

        if deepGeoDict.get("pde_learning_rate") is None:
            deepGeoDict["pde_learning_rate"] = self.DEFAULT_DEEP_GEO_DICT[
                "pde_learning_rate"
            ]
        if deepGeoDict.get("sympnet_learning_rate") is None:
            deepGeoDict["sympnet_learning_rate"] = self.DEFAULT_DEEP_GEO_DICT[
                "sympnet_learning_rate"
            ]
        if deepGeoDict.get("layer_sizes") is None:
            deepGeoDict["layer_sizes"] = self.DEFAULT_DEEP_GEO_DICT["layer_sizes"]
        if deepGeoDict.get("nb_of_networks") is None:
            deepGeoDict["nb_of_networks"] = self.DEFAULT_DEEP_GEO_DICT["nb_of_networks"]
        if deepGeoDict.get("networks_size") is None:
            deepGeoDict["networks_size"] = self.DEFAULT_DEEP_GEO_DICT["networks_size"]
        if deepGeoDict.get("rho_max") is None:
            deepGeoDict["rho_max"] = self.DEFAULT_DEEP_GEO_DICT["rho_max"]
        if deepGeoDict.get("file_name") is None:
            deepGeoDict["file_name"] = self.DEFAULT_DEEP_GEO_DICT["file_name"]
        if deepGeoDict.get("source_term") is None:
            deepGeoDict["source_term"] = self.DEFAULT_DEEP_GEO_DICT["source_term"]
        if deepGeoDict.get("boundary_condition") is None:
            deepGeoDict["boundary_condition"] = self.DEFAULT_DEEP_GEO_DICT[
                "boundary_condition"
            ]
        if deepGeoDict.get("to_be_trained") is None:
            deepGeoDict["to_be_trained"] = self.DEFAULT_DEEP_GEO_DICT["to_be_trained"]
        if deepGeoDict.get("sympnet_activation") is None:
            deepGeoDict["sympnet_activation"] = self.DEFAULT_DEEP_GEO_DICT[
                "sympnet_activation"
            ]
        if deepGeoDict.get("pinn_activation") is None:
            deepGeoDict["pinn_activation"] = self.DEFAULT_DEEP_GEO_DICT[
                "pinn_activation"
            ]

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
        self.rho_min, self.rho_max = 0, deepGeoDict["rho_max"]
        self.theta_min, self.theta_max = 0, 2 * torch.pi
        self.Vol_Omega = torch.pi * self.rho_max**2
        self.Vol_Gamma = 2 * torch.pi * self.rho_max
        # Source term of the Poisson problem
        self.source_term = deepGeoDict["source_term"]
        # Boundary condition of the Poisson problem
        self.boundary_condition = deepGeoDict["boundary_condition"]
        # activation functions
        self.pinn_activation = deepGeoDict["pinn_activation"]
        self.sympnet_activation = deepGeoDict["sympnet_activation"]

        self.create_networks()
        self.load(self.file_name)

        self.to_be_trained = deepGeoDict["to_be_trained"]

    def sympnet_layer_append(self, nets, optims, i):
        nets.append(
            nn.DataParallel(
                G.Symp_Net_Forward(self.networks_size, self.sympnet_activation)
            ).to(device)
        )
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
        self.u_net = nn.DataParallel(
            poisson.PDE_Forward(self.layer_sizes, self.pinn_activation)
        ).to(device)
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

    @staticmethod
    def try_to_load(checkpoint, key):
        try:
            return checkpoint[key]
        except KeyError:
            return None

    def load(self, file_name):
        self.loss_history = {}

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

            self.loss = self.try_to_load(checkpoint, "loss")
            self.optimality_condition = self.try_to_load(
                checkpoint, "optimality_condition"
            )
            self.loss_history = self.try_to_load(checkpoint, "loss_history")

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
        optimality_condition,
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
                "optimality_condition": optimality_condition,
                "loss_history": loss_history,
                "nb_of_networks": nb_of_networks,
                "networks_size": networks_size,
            },
            file_name,
        )

    def get_metric_tensor(self, x, y):
        # il faut calculer :
        # A = J_T^{-t}*J_T^{-1}

        J_a, J_b, J_c, J_d = self.get_jacobian_T(x, y)
        A_a = J_d**2 + J_b**2
        A_b = -(J_c * J_d + J_a * J_b)
        A_c = A_b
        A_d = J_c**2 + J_a**2

        return A_a, A_b, A_c, A_d

    def get_jacobian_T(self, x, y):
        T = self.apply_symplecto(x, y)

        J_a = torch.autograd.grad(T[0].sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(T[0].sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(T[1].sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(T[1].sum(), y, create_graph=True)[0]

        return J_a, J_b, J_c, J_d

    def get_tangential_jacobian(self, x, y):
        J_a, J_b, J_c, J_d = self.get_jacobian_T(x, y)
        nx, ny = self.get_n(x, y)

        Jac_tan_x = J_d * nx - J_c * ny
        Jac_tan_y = -J_b * nx + J_a * ny

        return torch.sqrt(Jac_tan_x**2 + Jac_tan_y**2)

    def get_t(self, x, y):
        return -y, x

    def get_nT(self, x, y):
        J_a, J_b, J_c, J_d = self.get_jacobian_T(x, y)
        tx, ty = self.get_t(x, y)

        tTx = J_a * tx + J_b * ty
        tTy = J_c * tx + J_d * ty
        norm = torch.sqrt(tTx**2 + tTy**2)

        return -tTy / norm, tTx / norm

    def get_n(self, x, y):
        nx, ny = x, y
        return nx, ny

    def get_mean_curvature(self, x, y):
        xT, yT = self.apply_symplecto(x, y)
        xm, ym = self.apply_inverse_symplecto(xT, yT)
        nTx, nTy = self.get_nT(xm, ym)

        dx_nTx = torch.autograd.grad(nTx.sum(), xT, create_graph=True)[0]
        dy_nTy = torch.autograd.grad(nTy.sum(), yT, create_graph=True)[0]

        H = dx_nTx + dy_nTy

        # nx, ny = self.get_n(x, y)
        # J_a, J_b, J_c, J_d = self.get_jacobian_T(x, y)
        # J_inv_trans_n_x = J_d * nx - J_c * ny
        # J_inv_trans_n_y = -J_b * nx + J_a * ny

        # dx_J_inv_trans_n_x = torch.autograd.grad(J_inv_trans_n_x.sum(), x, create_graph=True)[0]
        # dy_J_inv_trans_n_y = torch.autograd.grad(J_inv_trans_n_y.sum(), y, create_graph=True)[0]

        # H = dx_J_inv_trans_n_x + dy_J_inv_trans_n_y

        return H

    def left_hand_term(self, x, y, x_border, y_border):
        u = self.get_u(x, y)
        a, b, c, d = self.get_metric_tensor(x, y)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        A_grad_u_grad_u = (a * dx_u + b * dy_u) * dx_u + (c * dx_u + d * dy_u) * dy_u

        gamma_0_u = self.get_u(x_border, y_border)
        Jac_tan = self.get_tangential_jacobian(x_border, y_border)

        return A_grad_u_grad_u + u**2, Jac_tan * gamma_0_u**2

    def right_hand_term(self, x, y):
        u = self.get_u(x, y)

        # terme source
        f = sourceTerms.get_f(
            *self.apply_symplecto(x, y),
            name=self.source_term,
        )

        return f * u

    def get_u(self, x, y):
        return bc.apply_BC(
            self.u_net(*self.apply_symplecto(x, y)),
            x,
            y,
            self.rho_min,
            self.rho_max,
            name=self.boundary_condition,
        )

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

    def get_optimality_condition(self, x, y):
        f = sourceTerms.get_f(
            *self.apply_symplecto(x, y),
            name=self.source_term,
        )

        u = self.get_u(x, y)

        kappa = 1

        J_a, J_b, J_c, J_d = self.get_jacobian_T(x, y)
        a, b, c, d = J_d, -J_c, -J_b, J_a
        u = self.get_u(x, y)
        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        Jt_dx_u = a * dx_u + b * dy_u
        Jt_dy_u = c * dx_u + d * dy_u
        grad_u_2 = Jt_dx_u**2 + Jt_dy_u**2

        H = self.get_mean_curvature(x, y)

        return 0.5 * grad_u_2 + 0.5 * u**2 * (1 + kappa * H - 2*kappa**2) - f * u

    @staticmethod
    def random(min_value, max_value, shape, requires_grad=False, device=device):
        random_numbers = torch.rand(
            shape, device=device, dtype=torch.double, requires_grad=requires_grad
        )
        return min_value + (max_value - min_value) * random_numbers

    def make_collocation(self, n_collocation):
        shape = (n_collocation, 1)

        self.rho_collocation = torch.sqrt(
            self.random(self.rho_min**2, self.rho_max**2, shape, requires_grad=True)
        )
        self.theta_collocation = self.random(
            0, 2 * torch.math.pi, shape, requires_grad=True
        )

        self.x_collocation = self.rho_collocation * torch.cos(self.theta_collocation)
        self.y_collocation = self.rho_collocation * torch.sin(self.theta_collocation)

        self.x_border_collocation = self.rho_max * torch.cos(self.theta_collocation)
        self.y_border_collocation = self.rho_max * torch.sin(self.theta_collocation)

    def append_to_history(self, keys, values):
        for key, value in zip(keys, values):
            try:
                self.loss_history[key].append(value)
            except KeyError:
                self.loss_history[key] = [value]

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
            best_optimality_condition_value = self.optimality_condition.item()
        except AttributeError:
            best_loss_value = 1e10
            best_optimality_condition_value = 1e10

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

                auu_Omega, auu_Gamma = self.left_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                    self.x_border_collocation,
                    self.y_border_collocation,
                )
                lu = self.right_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                )

                loss_Omega = 0.5 * auu_Omega - lu
                loss_Gamma = 0.5 * auu_Gamma
                self.loss = (
                    loss_Omega.sum() * self.Vol_Omega / n_collocation
                    + loss_Gamma.sum() * self.Vol_Gamma / n_collocation
                )

                self.optimality_condition = self.get_optimality_condition(
                    self.x_border_collocation,
                    self.y_border_collocation,
                ).var()
                # self.optimality_condition = torch.tensor([0.0], device=device)

            self.loss.backward()
            for i in range(self.nb_of_networks):
                self.up_optimizers[i].step()
                self.down_optimizers[i].step()
            self.u_optimizer.step()

            self.append_to_history(
                ("loss", "optimality_condition"),
                (self.loss.item(), self.optimality_condition.item()),
            )

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
                        best_optimality_condition,
                        self.loss_history,
                        self.get_physical_parameters(),
                        self.nb_of_networks,
                    )
                except NameError:
                    pass

            if (
                self.optimality_condition.item() < best_optimality_condition_value
                and epoch > 500
            ):
                print(
                    f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}, best optimality condition = {self.optimality_condition.item():5.2e}"
                )
                # best_loss = self.loss.clone()
                best_optimality_condition = self.optimality_condition.clone()
                best_optimality_condition_value = best_optimality_condition.item()
                # best_up_nets = self.copy_sympnet(self.up_nets)
                # best_up_optimizers = self.copy_sympnet(self.up_optimizers)
                # best_down_nets = self.copy_sympnet(self.down_nets)
                # best_down_optimizers = self.copy_sympnet(self.down_optimizers)

                # best_u_net = copy.deepcopy(self.u_net.state_dict())
                # best_u_optimizer = copy.deepcopy(self.u_optimizer.state_dict())

            if self.loss.item() < best_loss_value:
                print(
                    f"epoch {epoch: 5d}:    best loss = {self.loss.item():5.2e}, current optimality condition = {self.optimality_condition.item():5.2e}"
                )
                best_loss_value = self.loss.item()
                best_loss = self.loss.clone()
                # best_optimality_condition = self.optimality_condition.clone()
                # best_optimality_condition_value = best_optimality_condition.item()
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
                best_optimality_condition,
                self.loss_history,
                self.get_physical_parameters(),
                self.nb_of_networks,
            )
        except UnboundLocalError:
            pass

        self.load(self.file_name)

        if plot_history:
            self.plot_result(save_plots)

        return tps2 - tps1

    @staticmethod
    def copy_sympnet(to_be_copied):
        return [copy.deepcopy(copie.state_dict()) for copie in to_be_copied]

    def plot_result(self, save_plots):
        makePlots.loss(self.loss_history, save_plots, self.fig_storage)

        makePlots.edp_contour(
            self.rho_min,
            self.rho_max,
            self.get_u,
            lambda x, y: self.apply_symplecto(x, y),
            lambda x, y: self.apply_inverse_symplecto(x, y),
            save_plots,
            f"{self.fig_storage}_solution",
        )

        makePlots.edp_contour(
            self.rho_min,
            self.rho_max,
            lambda x, y: sourceTerms.get_f(
                *self.apply_symplecto(x, y), name=self.source_term
            ),
            lambda x, y: self.apply_symplecto(x, y),
            lambda x, y: self.apply_inverse_symplecto(x, y),
            save_plots,
            f"{self.fig_storage}_solution",
        )

        n_visu = 50_000
        self.make_collocation(n_visu)
        optimaity_condition = self.get_optimality_condition(
            self.x_border_collocation, self.y_border_collocation
        )

        import matplotlib.pyplot as plt

        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            """Add a vertical color bar to an image plot."""
            from mpl_toolkits import axes_grid1

            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            current_ax = plt.gca()
            cax = divider.append_axes("right", size=width, pad=pad)
            plt.sca(current_ax)
            return im.axes.figure.colorbar(im, cax=cax, **kwargs)

        xT, yT = self.apply_symplecto(
            self.x_border_collocation, self.y_border_collocation
        )
        _, ax = plt.subplots(figsize=(5, 5))
        im = ax.scatter(
            xT.detach().cpu(),
            yT.detach().cpu(),
            s=1,
            c=optimaity_condition.detach().cpu(),
            cmap="turbo",
        )
        add_colorbar(im)
        ax.set_aspect("equal")
