"""
Author:
    A BELIERES FRENDO (ENSTA Paris)
Date:
    05/05/2023

ML for parametric shape optimization
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

# local imports
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
        "nb_of_networks": 4,
        "networks_size": 10,
        "rho_min": 0,
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
        if deepGeoDict.get("rho_min") is None:
            deepGeoDict["rho_min"] = self.DEFAULT_DEEP_GEO_DICT["rho_min"]
        if deepGeoDict.get("rho_max") is None:
            deepGeoDict["rho_max"] = self.DEFAULT_DEEP_GEO_DICT["rho_max"]
        if deepGeoDict.get("mu_min") is None:
            deepGeoDict["mu_min"] = self.DEFAULT_DEEP_GEO_DICT["mu_min"]
        if deepGeoDict.get("mu_max") is None:
            deepGeoDict["mu_max"] = self.DEFAULT_DEEP_GEO_DICT["mu_max"]
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
            "./../../../outputs/deepShape/net/param_"
            + deepGeoDict["file_name"]
            + ".pth"
        )
        self.fig_storage = (
            "./../outputs/deepShape/img/param_" + deepGeoDict["file_name"]
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
        self.mu_min, self.mu_max = deepGeoDict["mu_min"], deepGeoDict["mu_max"]
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
            "mu_min": self.mu_min,
            "mu_max": self.mu_max,
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

    def get_jacobian_matrix(self, x, y, mu):
        xT, yT = self.apply_symplecto(x, y, mu)

        J_a = torch.autograd.grad(xT.sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(xT.sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(yT.sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(yT.sum(), y, create_graph=True)[0]

        return J_a, J_b, J_c, J_d

    def get_metric_tensor(self, x, y, mu):
        # il faut calculer :
        # A = J_T^{-t}*J_T^{-1}

        J_a, J_b, J_c, J_d = self.get_jacobian_matrix(x, y, mu)

        fac = (J_a * J_d - J_b * J_c) ** 2
        A_a = (J_d**2 + J_b**2) / fac
        A_b = -(J_c * J_d + J_a * J_b) / fac
        A_c = -(J_c * J_d + J_a * J_b) / fac
        A_d = (J_c**2 + J_a**2) / fac

        return A_a, A_b, A_c, A_d

    def get_dn_u(self, x, y, mu):
        J_a, J_b, J_c, J_d = self.get_jacobian_matrix(x, y, mu)

        det = J_a * J_d - J_b * J_c
        a, b, c, d = det * J_d, -det * J_c, -det * J_b, det * J_a

        u = self.get_u(x, y, mu)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        Jt_dx_u = a * dx_u + b * dy_u
        Jt_dy_u = c * dx_u + d * dy_u

        return torch.sqrt(Jt_dx_u**2 + Jt_dy_u**2)

    def get_optimality_condition(self, mu, n_pts=10_000):
        """
        Be careful : mu is a scalar
        """
        theta = torch.linspace(0, 2 * torch.pi, n_pts, requires_grad=True)[:, None]
        x = self.rho_max * torch.cos(theta)
        y = self.rho_max * torch.sin(theta)
        mu = mu * torch.ones_like(x)
        xT, yT = self.apply_symplecto(x, y, mu)
        dn_u = self.get_dn_u(x, y, mu)

        return dn_u - dn_u.mean(), xT, yT

    def left_hand_term(self, x, y, mu):
        u = self.get_u(x, y, mu)
        a, b, c, d = self.get_metric_tensor(x, y, mu)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        A_grad_u_grad_u = (a * dx_u + b * dy_u) * dx_u + (c * dx_u + d * dy_u) * dy_u

        return A_grad_u_grad_u

    def right_hand_term(self, x, y, mu):
        u = self.get_u(x, y, mu)

        # terme source
        f = sourceTerms.get_f(
            *self.apply_symplecto(x, y, mu),
            mu=mu,
            name=self.source_term,
        )
        return f * u

    def apply_inverse_symplecto(self, x, y, mu):
        for i in range(self.nb_of_networks):
            y = y - self.down_nets[self.nb_of_networks - 1 - i](x, mu)
            x = x - self.up_nets[self.nb_of_networks - 1 - i](y, mu)
        return x, y

    def apply_symplecto(self, x, y, mu):
        for i in range(self.nb_of_networks):
            x = x + self.up_nets[i](y, mu)
            y = y + self.down_nets[i](x, mu)
        return x, y

    def get_u(self, x, y, mu):
        return bc.apply_BC(
            self.u_net(*self.apply_symplecto(x, y, mu), mu),
            x,
            y,
            self.rho_min,
            self.rho_max,
            name=self.boundary_condition,
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
        self.mu_collocation = self.random(
            self.mu_min, self.mu_max, shape, requires_grad=True
        )

    def make_border_collocation(self, n_collocation):
        shape = (n_collocation, 1)

        rho_border_collocation = self.rho_max * torch.ones(
            shape, dtype=torch.double, device=device
        )
        theta_border_collocation = self.random(
            self.theta_min, self.theta_max, shape, requires_grad=True
        )

        self.x_border_collocation = rho_border_collocation * torch.cos(
            theta_border_collocation
        )
        self.y_border_collocation = rho_border_collocation * torch.sin(
            theta_border_collocation
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

        n_mu_for_optimality_condition = kwargs.get("n_mu_for_optimality_condition", 10)

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

                grad_u_2 = self.left_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                    self.mu_collocation,
                )
                fu = self.right_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                    self.mu_collocation,
                )
                dirichlet_loss = 0.5 * grad_u_2 - fu

                self.loss = dirichlet_loss.sum() / n_collocation * self.Vol

                self.make_border_collocation(n_collocation)
                self.optimality_condition = 0

                random_mus = self.random(
                    self.mu_min, self.mu_max, n_mu_for_optimality_condition
                )

                for mu in random_mus:
                    self.optimality_condition += (
                        self.get_dn_u(
                            self.x_border_collocation,
                            self.y_border_collocation,
                            mu.item() * torch.ones_like(self.x_border_collocation),
                        ).var()
                        / n_mu_for_optimality_condition
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

            if self.optimality_condition.item() < best_optimality_condition_value:
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
            self.load(self.file_name)
        except UnboundLocalError:
            pass

        if plot_history:
            self.plot_result(save_plots)

        return tps2 - tps1

    @staticmethod
    def copy_sympnet(to_be_copied):
        return [copy.deepcopy(copie.state_dict()) for copie in to_be_copied]

    def plot_result(self, save_plots, n_mus=2, n_mus_for_optimality_condition=10):
        makePlots.loss(self.loss_history, save_plots, self.fig_storage)

        mu_list_solution = list(self.random(self.mu_min, self.mu_max, n_mus))
        mu_list_optimality = list(
            self.random(self.mu_min, self.mu_max, n_mus_for_optimality_condition)
        )

        makePlots.edp_contour_param_source(
            self.rho_min,
            self.rho_max,
            self.mu_min,
            self.mu_max,
            self.get_u,
            lambda x, y, mu: self.apply_symplecto(x, y, mu),
            lambda x, y, mu: self.apply_inverse_symplecto(x, y, mu),
            save_plots,
            f"{self.fig_storage}_solution",
            mu_list=mu_list_solution,
        )

        makePlots.optimality_condition_param(
            self.mu_min,
            self.mu_max,
            self.get_optimality_condition,
            save_plots,
            f"{self.fig_storage}_optimality",
            mu_list=mu_list_optimality,
        )

        if self.source_term == "one":

            def exact_solution(x, y, mu):
                return mu * 0.25 * (1 - x**2 - y**2)

            def error(x, y, mu):
                return torch.abs(exact_solution(x, y, mu) - self.get_u(x, y, mu))

            makePlots.edp_contour_param_source(
                self.rho_min,
                self.rho_max,
                self.mu_min,
                self.mu_max,
                error,
                lambda x, y, mu: self.apply_symplecto(x, y, mu),
                lambda x, y, mu: self.apply_inverse_symplecto(x, y, mu),
                save_plots,
                f"{self.fig_storage}_solution_error",
                mu_list=mu_list_solution,
                colormap="gist_heat",
            )

    def get_hausdorff_distances_to_disk(self, mus, n_pts=10_000):
        import numpy as np
        import scipy.spatial.distance as dist

        self.make_collocation(n_pts)

        hausdorff_distances = torch.zeros_like(mus)

        for i_mu, mu in enumerate(mus):
            if i_mu % 100 == 0:
                print(f"Hausdorff: {i_mu}/{len(mus)} done")

            theta = torch.linspace(
                0, 2 * torch.pi, n_pts, requires_grad=True, dtype=torch.float64
            )[:, None]
            x = self.rho_max * torch.cos(theta)
            y = self.rho_max * torch.sin(theta)
            xT, yT = self.apply_symplecto(x, y, mu.item() * torch.ones_like(x))

            x0 = xT.sum() / n_pts
            y0 = yT.sum() / n_pts

            x_ex, y_ex = self.x_collocation + x0, self.y_collocation + y0
            x_net, y_net = self.apply_symplecto(
                self.x_collocation,
                self.y_collocation,
                mu.item() * torch.ones_like(self.x_collocation),
            )

            X_net = []
            X_ex = []
            for x, y in zip(x_net.flatten().tolist(), y_net.flatten().tolist()):
                X_net.append((x, y))
            for x, y in zip(x_ex.flatten().tolist(), y_ex.flatten().tolist()):
                X_ex.append((x, y))

            X_net = np.array(X_net)
            X_ex = np.array(X_ex)

            hausdorff_distances[i_mu] = max(
                dist.directed_hausdorff(X_net, X_ex)[0],
                dist.directed_hausdorff(X_ex, X_net)[0],
            )

        print(f"Hausdorff: {len(mus)}/{len(mus)} done")

        return hausdorff_distances

    def get_L2_error_on_disk(self, mus, n_pts=10_000):
        def exact_solution(x, y, mu):
            return mu * 0.25 * (1 - x**2 - y**2)

        def error(x, y, mu):
            return torch.abs(exact_solution(x, y, mu) - self.get_u(x, y, mu))

        L2_errors = torch.zeros_like(mus)

        for i_mu, mu in enumerate(mus):
            if i_mu % 100 == 0:
                print(f"L2: {i_mu}/{len(mus)} done")

            self.make_collocation(n_pts)
            MSE = (
                error(
                    self.x_collocation,
                    self.y_collocation,
                    mu.item() * torch.ones_like(self.x_collocation),
                )[:, 0]
                .detach()
                .cpu()
                ** 2
            )
            L2_errors[i_mu] = torch.sqrt(MSE.sum() / n_pts)

        print(f"L2: {len(mus)}/{len(mus)} done")

        return L2_errors

    def stats_on_optimality_condition(self, mus, n_pts=10_000):
        optimality_conditions = torch.zeros_like(mus)

        for i_mu, mu in enumerate(mus):
            if i_mu % 100 == 0:
                print(f"optimality: {i_mu}/{len(mus)} done")

            self.make_border_collocation(n_pts)

            dn_u = self.get_dn_u(
                self.x_border_collocation,
                self.y_border_collocation,
                mu.item() * torch.ones_like(self.x_border_collocation),
            )[:, 0]

            optimality_conditions[i_mu] = dn_u.var().detach().cpu()

            self.x_border_collocation = None
            self.y_border_collocation = None

        print(f"optimality: {len(mus)}/{len(mus)} done")

        return optimality_conditions

    def compute_stats_constant_source(self, n_pts=10_000, n_mu=10):
        # compute a set of random mus
        random_mus = self.random(self.mu_min, self.mu_max, n_mu)

        # compute Hausdorff distances between the solution and the disk for each mu
        hausdorff_distances = self.get_hausdorff_distances_to_disk(random_mus, n_pts)

        # compute L2 error for each mu
        L2_errors = self.get_L2_error_on_disk(random_mus, n_pts)

        # compute optimality_conditions for each mu
        optimality_conditions = self.stats_on_optimality_condition(random_mus, n_pts)

        return random_mus, hausdorff_distances, L2_errors, optimality_conditions

    def get_fv_with_random_function(self, n_pts=50_000):
        assert isinstance(n_pts, int) and n_pts > 0
        self.make_collocation(n_pts)
        x, y, mu = self.x_collocation, self.y_collocation, self.mu_collocation

        u = self.get_u(x, y, mu)
        a, b, c, d = self.get_metric_tensor(x, y, mu)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        alpha = bc.compute_bc_mul(
            x, y, self.rho_min, self.rho_max, name=self.boundary_condition
        )

        coeff = torch.rand(6)
        constant = coeff[0]
        linear = coeff[1] * x + coeff[2] * y
        quadratic = coeff[3] * x**2 + coeff[4] * x * y + coeff[5] * y**2
        polynomial = constant + linear + quadratic

        phi = polynomial * alpha

        dx_phi = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        dy_phi = torch.autograd.grad(phi.sum(), y, create_graph=True)[0]

        term_x = (a * dx_u + b * dy_u) * dx_phi
        term_y = (c * dx_u + d * dy_u) * dy_phi
        A_grad_u_grad_phi = term_x + term_y

        f = sourceTerms.get_f(
            *self.apply_symplecto(x, y, mu),
            mu=mu,
            name=self.source_term,
        )

        return (A_grad_u_grad_phi - f * phi).sum().item() / x.shape[0] * self.Vol

    def compute_stats(self, n_pts=50_000, n_random=1_000):
        assert isinstance(n_pts, int) and n_pts > 0
        assert isinstance(n_random, int) and n_random > 0

        residuals = torch.zeros(n_random)
        for i in range(n_random):
            if i % 100 == 0:
                print(f"Computing residuals... {int(100 * i / n_random)}% done")
            residuals[i] = self.get_fv_with_random_function(n_pts)

        residuals = torch.abs(residuals)

        print(f"\nMean residual: {residuals.mean():3.2e}")
        print(f"Max residual: {residuals.max():3.2e}")
        print(f"Min residual: {residuals.min():3.2e}")
        print(f"Variance residual: {residuals.var():3.2e}\n")

        random_mus = self.random(self.mu_min, self.mu_max, n_random)
        optimality_conditions = self.stats_on_optimality_condition(random_mus, n_pts)

        print(f"\nMean optimality: {optimality_conditions.mean():3.2e}")
        print(f"Max optimality: {optimality_conditions.max():3.2e}")
        print(f"Min optimality: {optimality_conditions.min():3.2e}")
        print(f"Variance optimality: {optimality_conditions.var():3.2e}")

        return residuals, optimality_conditions
