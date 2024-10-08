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
from gesonn.com1PINNs import poissonParam, sourceTerms
from gesonn.com2SympNets import GParam
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
        "layer_sizes": [3, 10, 20, 10, 1],
        "nb_of_networks": 2,
        "networks_size": 5,
        "rho_max": 1,
        "kappa_min": 0.5,
        "kappa_max": 1.5,
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
        if deepGeoDict.get("kappa_min") is None:
            deepGeoDict["kappa_min"] = self.DEFAULT_DEEP_GEO_DICT["kappa_min"]
        if deepGeoDict.get("kappa_max") is None:
            deepGeoDict["kappa_max"] = self.DEFAULT_DEEP_GEO_DICT["kappa_max"]
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
        self.kappa_min, self.kappa_max = (
            deepGeoDict["kappa_min"],
            deepGeoDict["kappa_max"],
        )
        self.theta_min, self.theta_max = 0, 2 * torch.pi
        self.Vol_Omega = torch.pi * self.rho_max**2
        self.Vol_Gamma = 2 * torch.pi * self.rho_max
        self.Vol_Param = self.kappa_max - self.kappa_min
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
                GParam.Symp_Net_Forward(self.networks_size, self.sympnet_activation)
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
            poissonParam.PDE_Forward(self.layer_sizes, self.pinn_activation)
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
            "kappa_min": self.kappa_min,
            "kappa_max": self.kappa_max,
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

    def get_metric_tensor(self, x, y, kappa):
        # il faut calculer :
        # A = J_T^{-t}*J_T^{-1}

        J_a, J_b, J_c, J_d = (
            self.J_a_Omega,
            self.J_b_Omega,
            self.J_c_Omega,
            self.J_d_Omega,
        )
        A_a = J_d**2 + J_b**2
        A_b = -(J_c * J_d + J_a * J_b)
        A_c = A_b
        A_d = J_c**2 + J_a**2

        return A_a, A_b, A_c, A_d

    def get_jacobian_T(self, x, y, kappa):
        T = self.apply_symplecto(x, y, kappa)

        J_a = torch.autograd.grad(T[0].sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(T[0].sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(T[1].sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(T[1].sum(), y, create_graph=True)[0]

        return J_a, J_b, J_c, J_d

    def get_tangential_jacobian(self, x, y, kappa):
        J_a, J_b, J_c, J_d = (
            self.J_a_Gamma,
            self.J_b_Gamma,
            self.J_c_Gamma,
            self.J_d_Gamma,
        )
        nx, ny = self.get_n(x, y)

        Jac_tan_x = J_d * nx - J_c * ny
        Jac_tan_y = -J_b * nx + J_a * ny

        return torch.sqrt(Jac_tan_x**2 + Jac_tan_y**2)

    def get_t(self, x, y):
        return y, -x

    def get_nT(self, x, y, kappa):
        J_a, J_b, J_c, J_d = self.get_jacobian_T(x, y, kappa)
        tx, ty = self.get_t(x, y)

        tTx = J_a * tx + J_b * ty
        tTy = J_c * tx + J_d * ty
        norm = torch.sqrt(tTx**2 + tTy**2)

        return -tTy / norm, tTx / norm

    def get_n(self, x, y):
        nx, ny = x, y
        return nx, ny

    def get_mean_curvature(self, x, y, kappa):
        xT, yT = self.apply_symplecto(x, y, kappa)
        xm, ym = self.apply_inverse_symplecto(xT, yT, kappa)
        nTx, nTy = self.get_nT(xm, ym, kappa)

        dx_nTx = torch.autograd.grad(nTx.sum(), xT, create_graph=True)[0]
        dy_nTy = torch.autograd.grad(nTy.sum(), yT, create_graph=True)[0]

        H = dx_nTx + dy_nTy

        return H

    def left_hand_term(self, x, y, x_border, y_border, kappa):
        u = self.u_Omega
        a, b, c, d = self.A_a_Omega, self.A_b_Omega, self.A_c_Omega, self.A_d_Omega

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        A_grad_u_grad_u = (a * dx_u + b * dy_u) * dx_u + (c * dx_u + d * dy_u) * dy_u

        gamma_0_u = self.u_Gamma
        Jac_tan = self.get_tangential_jacobian(x_border, y_border, kappa)

        return A_grad_u_grad_u, kappa * Jac_tan * gamma_0_u**2

    def right_hand_term(self, x, y, kappa):
        u = self.u_Omega

        # terme source
        f = sourceTerms.get_f(
            *self.apply_symplecto(x, y, kappa),
            name=self.source_term,
        )

        return f * u

    def get_u(self, x, y, kappa):
        return bc.apply_BC(
            self.u_net(*self.apply_symplecto(x, y, kappa), kappa),
            x,
            y,
            self.rho_min,
            self.rho_max,
            name=self.boundary_condition,
        )

    def apply_symplecto(self, x, y, kappa):
        for i in range(self.nb_of_networks):
            x = x + self.up_nets[i](y, kappa)
            y = y + self.down_nets[i](x, kappa)
        return x, y

    def apply_inverse_symplecto(self, x, y, kappa):
        for i in range(self.nb_of_networks):
            y = y - self.down_nets[self.nb_of_networks - 1 - i](x, kappa)
            x = x - self.up_nets[self.nb_of_networks - 1 - i](y, kappa)
        return x, y

    def get_optimality_condition_visu(self, kappa, n=10_000):
        """
        Be careful : kappa is a scalar
        """
        theta = torch.linspace(
            0, 2 * torch.pi, n, requires_grad=True, dtype=torch.float64, device=device
        )[:, None]
        x = self.rho_max * torch.cos(theta)
        y = self.rho_max * torch.sin(theta)
        kappa = kappa * torch.ones_like(x)
        xT, yT = self.apply_symplecto(x, y, kappa)

        f = sourceTerms.get_f(
            xT,
            yT,
            name=self.source_term,
        )

        u = self.get_u(x, y, kappa)

        J_a, J_b, J_c, J_d = self.get_jacobian_T(x, y, kappa)
        a, b, c, d = J_d, -J_c, -J_b, J_a
        u = self.get_u(x, y, kappa)
        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        Jt_dx_u = a * dx_u + b * dy_u
        Jt_dy_u = c * dx_u + d * dy_u
        grad_u_2 = Jt_dx_u**2 + Jt_dy_u**2

        H = self.get_mean_curvature(x, y, kappa)
        condition = 0.5 * grad_u_2 + 0.5 * u**2 * (kappa * H - 2 * kappa) - f * u

        return condition - condition.mean(), xT, yT

    def get_optimality_condition(self, x, y, kappa):
        f = sourceTerms.get_f(
            *self.apply_symplecto(x, y, kappa),
            name=self.source_term,
        )

        u = self.get_u(x, y, kappa)

        J_a, J_b, J_c, J_d = self.get_jacobian_T(x, y, kappa)
        a, b, c, d = J_d, -J_c, -J_b, J_a
        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        Jt_dx_u = a * dx_u + b * dy_u
        Jt_dy_u = c * dx_u + d * dy_u
        grad_u_2 = Jt_dx_u**2 + Jt_dy_u**2

        H = self.get_mean_curvature(x, y, kappa)
        return 0.5 * grad_u_2 + 0.5 * u**2 * (kappa * H - 2 * kappa) - f * u

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

        self.kappa_collocation = self.random(
            self.kappa_min, self.kappa_max, shape, requires_grad=True
        )

    def make_Omega_calls(self, x, y, kappa):
        self.u_Omega = self.get_u(x, y, kappa)
        self.J_a_Omega, self.J_b_Omega, self.J_c_Omega, self.J_d_Omega = (
            self.get_jacobian_T(x, y, kappa)
        )
        self.A_a_Omega, self.A_b_Omega, self.A_c_Omega, self.A_d_Omega = (
            self.get_metric_tensor(x, y, kappa)
        )

    def make_Gamma_calls(self, x, y, kappa):
        self.u_Gamma = self.get_u(x, y, kappa)
        self.J_a_Gamma, self.J_b_Gamma, self.J_c_Gamma, self.J_d_Gamma = (
            self.get_jacobian_T(x, y, kappa)
        )

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
        n_kappa_for_optimality_condition = kwargs.get(
            "n_kappa_for_optimality_condition", 10
        )

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
                self.make_Omega_calls(
                    self.x_collocation, self.y_collocation, self.kappa_collocation
                )
                self.make_Gamma_calls(
                    self.x_border_collocation,
                    self.y_border_collocation,
                    self.kappa_collocation,
                )

                auu_Omega, auu_Gamma = self.left_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                    self.x_border_collocation,
                    self.y_border_collocation,
                    self.kappa_collocation,
                )
                lu = self.right_hand_term(
                    self.x_collocation, self.y_collocation, self.kappa_collocation
                )

                loss_Omega = 0.5 * auu_Omega - lu
                loss_Gamma = 0.5 * auu_Gamma
                self.loss = (
                    loss_Omega.sum() * self.Vol_Omega / n_collocation
                    + loss_Gamma.sum() * self.Vol_Gamma / n_collocation
                ) * self.Vol_Param

                self.optimality_condition = torch.tensor([0.0], device=device)
                random_kappas = self.random(
                    self.kappa_min,
                    self.kappa_max,
                    n_kappa_for_optimality_condition,
                )
                for kappa in random_kappas:
                    self.optimality_condition += (
                        self.get_optimality_condition(
                            self.x_border_collocation,
                            self.y_border_collocation,
                            kappa.item() * torch.ones_like(self.x_border_collocation),
                        ).var()
                        / n_kappa_for_optimality_condition
                    )

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
                and epoch > 999
            ):
                print(
                    f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}, best optimality condition = {self.optimality_condition.item():5.2e}"
                )
                best_loss = self.loss.clone()
                best_optimality_condition = self.optimality_condition.clone()
                best_optimality_condition_value = best_optimality_condition.item()
                best_up_nets = self.copy_sympnet(self.up_nets)
                best_up_optimizers = self.copy_sympnet(self.up_optimizers)
                best_down_nets = self.copy_sympnet(self.down_nets)
                best_down_optimizers = self.copy_sympnet(self.down_optimizers)

                best_u_net = copy.deepcopy(self.u_net.state_dict())
                best_u_optimizer = copy.deepcopy(self.u_optimizer.state_dict())

            if self.loss.item() < best_loss_value:
                print(
                    f"epoch {epoch: 5d}:    best loss = {self.loss.item():5.2e}, current optimality condition = {self.optimality_condition.item():5.2e}"
                )
                best_loss_value = self.loss.item()

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

    def get_hausdorff_distance(
        self, approximate_symplecto, exact_symplecto, n_pts=10_000
    ):
        import numpy as np
        import scipy.spatial.distance as dist

        self.make_collocation(n_pts)

        x_ex, y_ex = exact_symplecto(self.x_collocation, self.y_collocation)
        x_net, y_net = approximate_symplecto(self.x_collocation, self.y_collocation)

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

    def plot_result(self, save_plots, n_kappas=5, n_kappas_for_optimality_condition=10):
        makePlots.loss(self.loss_history, save_plots, self.fig_storage)

        kappa_list_solution = list(
            self.random(self.kappa_min, self.kappa_max, n_kappas)
        )
        print(kappa_list_solution)

        kappa_list_optimality = list(
            self.random(
                self.kappa_min, self.kappa_max, n_kappas_for_optimality_condition
            )
        )

        makePlots.edp_contour_param_source(
            self.rho_min,
            self.rho_max,
            self.kappa_min,
            self.kappa_max,
            self.get_u,
            lambda x, y, kappa: self.apply_symplecto(x, y, kappa),
            lambda x, y, kappa: self.apply_inverse_symplecto(x, y, kappa),
            save_plots,
            f"{self.fig_storage}_solution",
            mu_list=kappa_list_solution,
        )

        makePlots.optimality_condition_param(
            self.kappa_min,
            self.kappa_max,
            self.get_optimality_condition_visu,
            save_plots,
            f"{self.fig_storage}_optimality",
            mu_list=kappa_list_optimality,
        )

        if self.source_term == "one":

            def exact_solution(x, y, kappa):
                n_pts = 10_000
                theta = torch.linspace(
                    0,
                    2 * torch.pi,
                    n_pts,
                    requires_grad=True,
                    dtype=torch.float64,
                    device=device,
                )[:, None]
                x_ = self.rho_max * torch.cos(theta)
                y_ = self.rho_max * torch.sin(theta)
                kappa_ = kappa[0].item() * torch.ones_like(x_)
                xT, yT = self.apply_symplecto(x_, y_, kappa_)

                x0 = xT.sum() / n_pts
                y0 = yT.sum() / n_pts

                print(x0.item(), y0.item())

                x, y = self.apply_symplecto(x, y, kappa)
                return (2 + kappa) / (4 * kappa) - 0.25 * (
                    (x - x0) ** 2 + (y - y0) ** 2
                )

            def error(x, y, kappa):
                return torch.abs(exact_solution(x, y, kappa) - self.get_u(x, y, kappa))

            makePlots.edp_contour_param_source(
                self.rho_min,
                self.rho_max,
                self.kappa_min,
                self.kappa_max,
                error,
                lambda x, y, kappa: self.apply_symplecto(x, y, kappa),
                lambda x, y, kappa: self.apply_inverse_symplecto(x, y, kappa),
                save_plots,
                f"{self.fig_storage}_solution_error",
                mu_list=kappa_list_solution,
                colormap="gist_heat",
            )

    def get_hausdorff_distances_to_disk(self, kappas, n_pts=10_000):
        import numpy as np
        import scipy.spatial.distance as dist

        self.make_collocation(n_pts)

        hausdorff_distances = torch.zeros_like(kappas)

        for i_kappa, kappa in enumerate(kappas):
            if i_kappa % 100 == 0:
                print(f"Hausdorff: {i_kappa}/{len(kappas)} done")

            theta = torch.linspace(
                0, 2 * torch.pi, n_pts, requires_grad=True, dtype=torch.float64
            )[:, None]
            x = self.rho_max * torch.cos(theta)
            y = self.rho_max * torch.sin(theta)
            xT, yT = self.apply_symplecto(x, y, kappa.item() * torch.ones_like(x))

            x0 = xT.sum() / n_pts
            y0 = yT.sum() / n_pts

            x_ex, y_ex = self.x_collocation + x0, self.y_collocation + y0
            x_net, y_net = self.apply_symplecto(
                self.x_collocation,
                self.y_collocation,
                kappa.item() * torch.ones_like(self.x_collocation),
            )

            X_net = []
            X_ex = []
            for x, y in zip(x_net.flatten().tolist(), y_net.flatten().tolist()):
                X_net.append((x, y))
            for x, y in zip(x_ex.flatten().tolist(), y_ex.flatten().tolist()):
                X_ex.append((x, y))

            X_net = np.array(X_net)
            X_ex = np.array(X_ex)

            hausdorff_distances[i_kappa] = max(
                dist.directed_hausdorff(X_net, X_ex)[0],
                dist.directed_hausdorff(X_ex, X_net)[0],
            )

        print(f"Hausdorff: {len(kappas)}/{len(kappas)} done")

        return hausdorff_distances

    def get_L2_error_on_disk(self, kappas, n_pts=10_000):
        def exact_solution(x, y, kappa):
            n_pts = 10_000
            theta = torch.linspace(
                0,
                2 * torch.pi,
                n_pts,
                requires_grad=True,
                dtype=torch.float64,
                device=device,
            )[:, None]
            x_ = self.rho_max * torch.cos(theta)
            y_ = self.rho_max * torch.sin(theta)
            kappa_ = kappa[0].item() * torch.ones_like(x_)
            xT, yT = self.apply_symplecto(x_, y_, kappa_)

            x0 = xT.sum() / n_pts
            y0 = yT.sum() / n_pts

            x, y = self.apply_symplecto(x, y, kappa)
            return (2 + kappa) / (4 * kappa) - 0.25 * ((x - x0) ** 2 + (y - y0) ** 2)

        def error(x, y, kappa):
            return torch.abs(exact_solution(x, y, kappa) - self.get_u(x, y, kappa))

        L2_errors = torch.zeros_like(kappas)

        for i_kappa, kappa in enumerate(kappas):
            if i_kappa % 100 == 0:
                print(f"L2: {i_kappa}/{len(kappas)} done")

            self.make_collocation(n_pts)
            MSE = (
                error(
                    self.x_collocation,
                    self.y_collocation,
                    kappa.item() * torch.ones_like(self.x_collocation),
                )[:, 0]
                .detach()
                .cpu()
                ** 2
            )
            L2_errors[i_kappa] = torch.sqrt(MSE.sum() / n_pts)

        print(f"L2: {len(kappas)}/{len(kappas)} done")

        return L2_errors

    def stats_on_optimality_condition(self, kappas, n_pts=10_000):
        optimality_conditions = torch.zeros_like(kappas)

        for i_kappa, kappa in enumerate(kappas):
            if i_kappa % 100 == 0:
                print(f"optimality: {i_kappa}/{len(kappas)} done")

            self.make_collocation(n_pts)

            optim_cond = self.get_optimality_condition(
                self.x_border_collocation,
                self.y_border_collocation,
                kappa.item() * torch.ones_like(self.x_border_collocation),
            )[:, 0]

            optimality_conditions[i_kappa] = optim_cond.var().detach().cpu()

            self.x_border_collocation = None
            self.y_border_collocation = None

        print(f"optimality: {len(kappas)}/{len(kappas)} done")

        return optimality_conditions

    def compute_stats_constant_source(self, n_pts=50_000, n_kappa=1_000):
        # compute a set of random kappas
        random_kappas = self.random(self.kappa_min, self.kappa_max, n_kappa)

        # compute Hausdorff distances between the solution and the disk for each kappa
        hausdorff_distances = self.get_hausdorff_distances_to_disk(random_kappas, n_pts)

        # compute L2 error for each kappa
        L2_errors = self.get_L2_error_on_disk(random_kappas, n_pts)

        # compute optimality_conditions for each kappa
        optimality_conditions = self.stats_on_optimality_condition(random_kappas, n_pts)

        print(f"\nMean Haussdorf distance: {hausdorff_distances.mean():3.2e}")
        print(f"Max Haussdorf distance: {hausdorff_distances.max():3.2e}")
        print(f"Min Haussdorf distance: {hausdorff_distances.min():3.2e}")
        print(f"Variance Haussdorf distance: {hausdorff_distances.var():3.2e}")

        print(f"\nMean L2 error: {L2_errors.mean():3.2e}")
        print(f"Max L2 error: {L2_errors.max():3.2e}")
        print(f"Min L2 error: {L2_errors.min():3.2e}")
        print(f"Variance L2 error: {L2_errors.var():3.2e}")

        print(f"\nMean optimality: {optimality_conditions.mean():3.2e}")
        print(f"Max optimality: {optimality_conditions.max():3.2e}")
        print(f"Min optimality: {optimality_conditions.min():3.2e}")
        print(f"Variance optimality: {optimality_conditions.var():3.2e}")

        return random_kappas, hausdorff_distances, L2_errors, optimality_conditions

    def compute_stats(self, n_pts=50_000, n_random=1_000):
        assert isinstance(n_pts, int) and n_pts > 0
        assert isinstance(n_random, int) and n_random > 0

        random_kappas = self.random(self.kappa_min, self.kappa_max, n_random)
        optimality_conditions = self.stats_on_optimality_condition(random_kappas, n_pts)

        print(f"\nMean optimality: {optimality_conditions.mean():3.2e}")
        print(f"Max optimality: {optimality_conditions.max():3.2e}")
        print(f"Min optimality: {optimality_conditions.min():3.2e}")
        print(f"Variance optimality: {optimality_conditions.var():3.2e}")

        return optimality_conditions
