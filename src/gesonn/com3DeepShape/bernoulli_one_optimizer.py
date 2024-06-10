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
        if deepGeoDict.get("file_name") is None:
            deepGeoDict["file_name"] = self.DEFAULT_DEEP_GEO_DICT["file_name"]
        if deepGeoDict.get("boundary_condition") is None:
            deepGeoDict["boundary_condition"] = self.DEFAULT_DEEP_GEO_DICT[
                "boundary_condition"
            ]
        if deepGeoDict.get("a") is None:
            deepGeoDict["a"] = self.DEFAULT_DEEP_GEO_DICT["a"]
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
        self.rho_min, self.rho_max = deepGeoDict["rho_min"], deepGeoDict["rho_max"]
        self.theta_min, self.theta_max = 0, 2 * torch.pi
        self.Vol = torch.pi * self.rho_max**2
        # Boundary condition of the Poisson problem
        self.boundary_condition = deepGeoDict["boundary_condition"]
        # Parameters of the compact set K
        self.a = deepGeoDict["a"]
        self.b = self.rho_min**2 / self.a
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
        # optims.append(
        #     torch.optim.Adam(nets[i].parameters(), lr=self.sympnet_learning_rate)
        # )
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
        self.u_net = nn.DataParallel(
            poisson.PDE_Forward(self.layer_sizes, self.pinn_activation)
        ).to(device)
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

            self.load_sympnet_layer(self.up_nets, "up_models_state_dict", checkpoint)
            self.load_sympnet_layer(
                self.down_nets, "down_models_state_dict", checkpoint
            )

            self.u_net.load_state_dict(checkpoint["u_model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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
        optimality_condition,
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
                "optimality_condition": optimality_condition,
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
        self.make_border_collocation(n)
        xT, yT = self.apply_symplecto(
            self.x_gamma_collocation, self.y_gamma_collocation
        )
        dn_u = self.get_dn_u(self.x_gamma_collocation, self.y_gamma_collocation)

        opt = dn_u - dn_u.mean()

        return opt, xT, yT

    def left_hand_term(self, x, y):
        u = self.get_u(x, y)
        a, b, c, d = self.get_metric_tensor(x, y)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        A_grad_u_grad_u = (a * dx_u + b * dy_u) * dx_u + (c * dx_u + d * dy_u) * dy_u

        return A_grad_u_grad_u

    def apply_symplecto(self, x, y):
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

    def make_collocation(self, n_collocation, rho_min=0):
        shape = (n_collocation, 1)

        rho_collocation = torch.sqrt(
            self.random(rho_min, self.rho_max**2, shape, requires_grad=True)
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

        # trucs de sauvegarde
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
                dirichlet_loss = 0.5 * grad_u_2.sum() * self.Vol / n_pts

                self.loss = dirichlet_loss

                self.make_border_collocation(n_collocation)
                self.optimality_condition = self.get_dn_u(
                    self.x_gamma_collocation, self.y_gamma_collocation
                ).var()

            self.loss.backward()
            self.optimizer.step()

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
                        best_down_nets,
                        best_u_net,
                        best_optimizer,
                        best_loss,
                        best_optimality_condition,
                        self.loss_history,
                        self.get_physical_parameters(),
                        self.nb_of_networks,
                    )
                except NameError:
                    pass

            # if self.loss.item() < best_loss_value:
            if self.optimality_condition.item() < best_optimality_condition_value:
                print(
                    f"epoch {epoch: 5d}:    current loss = {self.loss.item():5.2e}, best optimality condition = {self.optimality_condition.item():5.2e}"
                )
                best_loss = self.loss.clone()
                best_optimality_condition = self.optimality_condition.clone()
                # best_loss_value = best_loss.item()
                best_optimality_condition_value = best_optimality_condition.item()
                best_up_nets = self.copy_sympnet(self.up_nets).copy()
                best_down_nets = self.copy_sympnet(self.down_nets).copy()

                best_u_net = copy.deepcopy(self.u_net.state_dict())

                best_optimizer = copy.deepcopy(self.optimizer.state_dict())

            if self.loss.item() < best_loss_value:
                print(
                    f"epoch {epoch: 5d}:    best loss = {self.loss.item():5.2e}, current optimality condition = {self.optimality_condition.item():5.2e}"
                )
                best_loss_value = self.loss.item()

        tps2 = time.time()

        print(f"epoch {epoch: 5d}: current loss = {self.loss.item():3.2e}")

        try:
            self.save(
                self.file_name,
                epoch,
                best_up_nets,
                best_down_nets,
                best_u_net,
                best_optimizer,
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

    def plot_result(self, save_plots):
        makePlots.loss_bernoulli(self.loss_history, save_plots, self.fig_storage)

        makePlots.edp_contour_bernoulli(
            self.rho_max,
            self.a,
            self.b,
            self.get_u,
            lambda x, y: self.apply_symplecto(x, y),
            lambda x, y: self.apply_inverse_symplecto(x, y),
            save_plots,
            f"{self.fig_storage}_solution",
        )

        makePlots.optimality_condition(
            self.get_optimality_condition,
            save_plots,
            f"{self.fig_storage}_optimality",
        )

        if self.a == 0.5:
            n_pts = 10_000
            theta = torch.linspace(
                0, 2 * torch.pi, n_pts, requires_grad=True, dtype=torch.float64
            )[:, None]
            x = self.rho_max * torch.cos(theta)
            y = self.rho_max * torch.sin(theta)
            xT, yT = self.apply_symplecto(x, y)

            x0 = xT.sum() / n_pts
            y0 = yT.sum() / n_pts

            makePlots.shape_error(
                self.rho_max,
                lambda x, y: self.apply_symplecto(x, y),
                lambda x, y: (x + x0, y + y0),
                self.get_hausdorff_distance,
                save_plots,
                f"{self.fig_storage}_shape_error",
            )

            print(f"error to disk: {((xT - x0)**2 + (yT - y0)**2 - 1).sum() / n_pts}")

    def get_fv_with_random_function(self, n_pts=50_000):
        assert isinstance(n_pts, int) and n_pts > 0
        self.make_collocation(n_pts)
        x, y = self.x_collocation, self.y_collocation

        u = self.get_u(x, y)
        a, b, c, d = self.get_metric_tensor(x, y)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        xT, yT = self.apply_symplecto(x, y)

        alpha = bc.compute_bc_mul(
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

        beta = bc.compute_bc_add(
            x,
            y,
            self.rho_max,
            name=self.boundary_condition,
            xT=xT,
            yT=yT,
            a=self.a,
            b=self.b,
        )

        coeff = torch.rand(6)
        constant = coeff[0]
        linear = coeff[1] * x + coeff[2] * y
        quadratic = coeff[3] * x**2 + coeff[4] * x * y + coeff[5] * y**2
        polynomial = constant + linear + quadratic

        phi = polynomial * alpha + beta

        dx_phi = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        dy_phi = torch.autograd.grad(phi.sum(), y, create_graph=True)[0]

        term_x = (a * dx_u + b * dy_u) * dx_phi
        term_y = (c * dx_u + d * dy_u) * dy_phi
        A_grad_u_grad_phi = term_x + term_y

        return A_grad_u_grad_phi.sum().item() / x.shape[0] * self.Vol

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
        print(f"Variance residual: {residuals.var():3.2e}")

        return residuals
