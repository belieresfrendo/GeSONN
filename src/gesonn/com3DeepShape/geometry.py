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
        "rho_min": 0,
        "rho_max": 1,
        "a": 0.6,
        "file_name": "default",
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "homogeneous_dirichlet",
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
        if deepGeoDict.get("a") is None:
            deepGeoDict["a"] = self.DEFAULT_DEEP_GEO_DICT["a"]
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
        # Source term of the Poisson problem
        self.source_term = deepGeoDict["source_term"]
        # Boundary condition of the Poisson problem
        self.boundary_condition = deepGeoDict["boundary_condition"]

        # Parameters of the compact set K
        self.a = deepGeoDict["a"]
        self.b = self.rho_min**2 / self.a

        self.create_networks()
        self.load(self.file_name)

        self.to_be_trained = deepGeoDict["to_be_trained"]

        # LBFGS parameters
        self.switch_to_LBFGS = kwargs.get("switch_to_LBFGS", False)
        self.switch_to_LBFGS_at = kwargs.get("switch_to_LBFGS_at", 1_000)
        self.LBFGS_history_size = kwargs.get("LBFGS_history_size", 15)
        self.LBFGS_max_iter = kwargs.get("LBFGS_max_iter", 5)

    def sympnet_layer_append(self, nets, optims, i):
        nets.append(
            nn.DataParallel(G.Symp_Net_Forward_No_Bias(self.networks_size)).to(device)
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
        self.u_net = nn.DataParallel(poisson.PDE_Forward(self.layer_sizes)).to(device)
        self.u_optimizer = torch.optim.Adam(
            self.u_net.parameters(), lr=self.pde_learning_rate
        )

        self.LBFGS_u_optimizer = None
        self.LBFGS_up_optimizers = None
        self.LBFGS_down_optimizers = None
        self.LBFGS_is_active = False

    def create_LBFGS_optimizers(self):
        print("Le Big Fucking Gradient Soviet est loaded")
        self.LBFGS_u_optimizer = torch.optim.LBFGS(
            self.u_net.parameters(),
            # self.best_u_optimizer,
            history_size=self.LBFGS_history_size,
            max_iter=self.LBFGS_max_iter,
            line_search_fn="strong_wolfe",
        )
        self.LBFGS_up_optimizers = []
        self.LBFGS_down_optimizers = []
        for i in range(self.nb_of_networks):
            self.LBFGS_up_optimizers.append(
                torch.optim.LBFGS(
                    self.up_nets[i].parameters(),
                    # self.best_up_optimizers[i],
                    history_size=self.LBFGS_history_size,
                    max_iter=self.LBFGS_max_iter,
                    line_search_fn="strong_wolfe",
                )
            )
            self.LBFGS_down_optimizers.append(
                torch.optim.LBFGS(
                    self.down_nets[i].parameters(),
                    # self.best_down_optimizers[i],
                    history_size=self.LBFGS_history_size,
                    max_iter=self.LBFGS_max_iter,
                    line_search_fn="strong_wolfe",
                )
            )
        self.LBFGS_is_active = True

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

            if checkpoint["LBFGS_u_optimizer_state_dict"] is not None:
                self.create_LBFGS_optimizers()
                self.LBFGS_u_optimizer.load_state_dict(
                    checkpoint["LBFGS_u_optimizer_state_dict"]
                )

                optimizers_state_dicts = checkpoint["LBFGS_down_optimizers_state_dict"]
                i = 0
                for _, state_dict in zip(
                    self.LBFGS_down_optimizers, optimizers_state_dicts
                ):
                    self.LBFGS_down_optimizers[i].load_state_dict(state_dict)
                    i += 1

                optimizers_state_dicts = checkpoint["LBFGS_up_optimizers_state_dict"]
                i = 0
                for _, state_dict in zip(
                    self.LBFGS_up_optimizers, optimizers_state_dicts
                ):
                    self.LBFGS_up_optimizers[i].load_state_dict(state_dict)
                    i += 1

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
        if state_dicts is not None:
            return [state_dict for state_dict in state_dicts]
        else:
            return None

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
        LBFGS_u_optimizer_state=None,
        LBFGS_up_optimizers_state_dict=None,
        LBFGS_down_optimizers_state_dict=None,
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
                "LBFGS_u_optimizer_state_dict": LBFGS_u_optimizer_state,
                "LBFGS_up_optimizers_state_dict": self.save_sympnet_layer(
                    LBFGS_up_optimizers_state_dict
                ),
                "LBFGS_down_optimizers_state_dict": self.save_sympnet_layer(
                    LBFGS_down_optimizers_state_dict
                ),
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

        rho_2 = x**2 + y**2
        xT, yT = self.apply_symplecto(x, y)
        rhoT_2 = (xT / self.a) ** 2 + (yT / self.b) ** 2
        bc_add = (self.rho_max**2 - rho_2) / (rhoT_2 - rho_2)

        beta_x = torch.autograd.grad(
            bc_add.sum(), x, create_graph=True, allow_unused=True
        )[0]
        beta_y = torch.autograd.grad(
            bc_add.sum(), y, create_graph=True, allow_unused=True
        )[0]

        T = self.apply_symplecto(x, y)
        J_a = torch.autograd.grad(T[0].sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(T[0].sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(T[1].sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(T[1].sum(), y, create_graph=True)[0]
        inv_det = 1/(J_a*J_d - J_b*J_c)
        Jt_beta_x = inv_det * (J_d * beta_x - J_c * beta_y)
        Jt_beta_y = inv_det * (-J_b * beta_x + J_a * beta_y)
        Jt_beta_grad_u = Jt_beta_x * dx_u + Jt_beta_y * dy_u

        return A_grad_u_grad_u + 2 * Jt_beta_grad_u

    def right_hand_term(self, x, y):
        u = self.get_u(x, y)

        # terme source
        f = sourceTerms.get_f(*self.apply_symplecto(x, y), name=self.source_term)

        return f * u

    def apply_symplecto(self, x, y):
        x, y = x, y
        for i in range(self.nb_of_networks):
            x = x + self.up_nets[i](y)
            y = y + self.down_nets[i](x)
        return x, y

    def apply_inverse_symplecto(self, x, y):
        for i in range(self.nb_of_networks):
            y = y - self.down_nets[self.nb_of_networks - 1 - i](x)
            x = x - self.up_nets[self.nb_of_networks - 1 - i](y)
        return x, y

    def apply_rejet_kompact(self, x, y):
        xT, yT = self.apply_symplecto(x, y)
        condition = (xT / self.a) ** 2 + (yT / self.b) ** 2 >= 1
        xT, yT = (
            xT[condition][:, None],
            yT[condition][:, None],
        )
        return self.apply_inverse_symplecto(xT, yT)

    def get_u(self, x, y):
        if self.poisson_or_bernoulli == "poisson":
            return bc.apply_BC(
                self.u_net(*self.apply_symplecto(x, y)),
                x,
                y,
                self.rho_min,
                self.rho_max,
                name=self.boundary_condition,
            )
        elif self.poisson_or_bernoulli == "bernoulli":
            rho_2 = x**2 + y**2
            xT, yT = self.apply_symplecto(x, y)
            rhoT_2 = (xT / self.a) ** 2 + (yT / self.b) ** 2
            bc_mul = (rho_2 - self.rho_max**2) * (rhoT_2 - 1)
            bc_add = (self.rho_max**2 - rho_2) / (rhoT_2 - rho_2)

            u = self.u_net(xT, yT) * bc_mul

            # SOurce term
            # beta_x = torch.autograd.grad(
            #     bc_add.sum(), x, create_graph=True, allow_unused=True
            # )[0]
            # beta_y = torch.autograd.grad(
            #     bc_add.sum(), y, create_graph=True, allow_unused=True
            # )[0]
            # beta_xx = torch.autograd.grad(
            #     beta_x.sum(), x, create_graph=True, allow_unused=True
            # )[0]
            # beta_yy = torch.autograd.grad(
            #     beta_y.sum(), y, create_graph=True, allow_unused=True
            # )[0]
            # source_term = + beta_xx + beta_yy
            
            # beta = bc_add
            # alpha = 1/(rhoT_2-rho_2)
            # gamma_x = 

            # self.fu = source_term * u

            return u

        raise NameError("Attention poisson_or_bernoulli")

    @staticmethod
    def random(min_value, max_value, shape, requires_grad=False, device=device):
        random_numbers = torch.rand(
            shape, device=device, dtype=torch.double, requires_grad=requires_grad
        )
        return min_value + (max_value - min_value) * random_numbers

    def make_collocation(self, n_collocation):
        shape = (n_collocation, 1)

        if self.poisson_or_bernoulli == "poisson":
            rho_collocation = torch.sqrt(
                self.random(self.rho_min**2, self.rho_max**2, shape, requires_grad=True)
            )
        if self.poisson_or_bernoulli == "bernoulli":
            rho_collocation = torch.sqrt(
                self.random(0, self.rho_max**2, shape, requires_grad=True)
            )

        self.theta_collocation = self.random(
            self.theta_min, self.theta_max, shape, requires_grad=True
        )

        self.x_collocation = rho_collocation * torch.cos(self.theta_collocation)
        self.y_collocation = rho_collocation * torch.sin(self.theta_collocation)

        if self.poisson_or_bernoulli == "bernoulli":
            self.x_gamma_collocation = self.rho_max * torch.cos(self.theta_collocation)
            self.y_gamma_collocation = self.rho_max * torch.sin(self.theta_collocation)
            self.x_collocation, self.y_collocation = self.apply_rejet_kompact(
                self.x_collocation, self.y_collocation
            )

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

    def train(self, poisson_or_bernoulli, **kwargs):
        self.poisson_or_bernoulli = poisson_or_bernoulli

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

        # boucle principale de la descnet ede gradient
        tps1 = time.time()
        for epoch in range(epochs):
            LBFGS_activated = self.LBFGS_u_optimizer is not None

            def closure():
                # mise à 0 du gradient
                if LBFGS_activated:
                    self.LBFGS_u_optimizer.zero_grad()
                    for i in range(self.nb_of_networks):
                        self.LBFGS_up_optimizers[i].zero_grad()
                        self.LBFGS_down_optimizers[i].zero_grad()
                else:
                    self.u_optimizer.zero_grad()
                    for i in range(self.nb_of_networks):
                        self.up_optimizers[i].zero_grad()
                        self.down_optimizers[i].zero_grad()

                # mise à 0 de la loss
                self.loss = torch.tensor([0.0], device=device)

                # Loss based on PDE
                if n_collocation > 0:
                    self.make_collocation(n_collocation)

                    if poisson_or_bernoulli == "poisson":
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

                    elif poisson_or_bernoulli == "bernoulli":
                        n_pts = self.x_collocation.size()[0]
                        grad_u_2 = self.left_hand_term(
                            self.x_collocation, self.y_collocation
                        )
                        dirichlet_loss = 0.5 * grad_u_2# - self.fu
                        self.loss = dirichlet_loss.sum() / n_pts * self.Vol

                        # dn_u_pred, _, _ = self.get_dn_u(
                        #     self.x_gamma_collocation, self.y_gamma_collocation
                        # )
                        # coeff = 0.1
                        # optimality_condition = dn_u_pred - torch.mean(dn_u_pred)
                        # self.loss += coeff * torch.sum(optimality_condition**2)

                self.loss.backward(retain_graph=True)
                return self.loss

            if LBFGS_activated:
                self.LBFGS_u_optimizer.step(closure)
                for i in range(self.nb_of_networks):
                    self.LBFGS_up_optimizers[i].step(closure)
                    self.LBFGS_down_optimizers[i].step(closure)
            else:
                closure()
                self.u_optimizer.step()
                for i in range(self.nb_of_networks):
                    self.up_optimizers[i].step()
                    self.down_optimizers[i].step()

            self.loss_history.append(self.loss.item())

            if (
                epoch == self.switch_to_LBFGS_at
                and not LBFGS_activated
                and self.switch_to_LBFGS
            ):
                self.create_LBFGS_optimizers()

            if epoch % 500 == 0 or (epoch % 10 == 0 and LBFGS_activated):
                print(f"epoch {epoch: 5d}: current loss = {self.loss.item():5.2e}")

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

                if LBFGS_activated:
                    best_LBFGS_u_optimizer = copy.deepcopy(
                        self.LBFGS_u_optimizer.state_dict()
                    )
                    best_LBFGS_up_optimizers = self.copy_sympnet(
                        self.LBFGS_up_optimizers
                    )
                    best_LBFGS_down_optimizers = self.copy_sympnet(
                        self.LBFGS_down_optimizers
                    )
                else:
                    best_LBFGS_u_optimizer = None
                    best_LBFGS_up_optimizers = None
                    best_LBFGS_down_optimizers = None

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
                best_LBFGS_u_optimizer,
                best_LBFGS_up_optimizers,
                best_LBFGS_down_optimizers,
            )
            self.load(self.file_name)
        except UnboundLocalError:
            pass

        if plot_history:
            if self.poisson_or_bernoulli == "poisson":
                self.plot_result_poisson(save_plots)
            if self.poisson_or_bernoulli == "bernoulli":
                self.plot_result_bernoulli(save_plots)
        return tps2 - tps1

    @staticmethod
    def copy_sympnet(to_be_copied):
        return [copy.deepcopy(copie.state_dict()) for copie in to_be_copied]

    def plot_result_poisson(self, save_plots):
        makePlots.loss(self.loss_history, save_plots, self.fig_storage)

        n_visu = 50_000
        self.make_collocation(n_visu)

        x, y = self.x_collocation, self.y_collocation
        xT, yT = self.apply_symplecto(x, y)
        u_pred = self.get_u(x, y)

        x_border = self.rho_max * torch.cos(self.theta_collocation)
        y_border = self.rho_max * torch.sin(self.theta_collocation)
        xT_border, yT_border = self.apply_symplecto(x_border, y_border)
        dn_u, _, _ = self.get_dn_u(x_border, y_border)

        makePlots.edp(
            xT_border.detach().cpu(),
            yT_border.detach().cpu(),
            dn_u.detach().cpu(),
            save_plots,
            self.fig_storage + "_gradn",
            title="gradient normal",
        )

        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            sourceTerms.get_f(xT, yT, name=self.source_term).detach().cpu(),
            save_plots,
            self.fig_storage + "_source",
            title="terme source",
        )

        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            u_pred.detach().cpu(),
            save_plots,
            self.fig_storage + "_pde",
            title="EDP",
        )

        a, b, c, d = self.get_metric_tensor(x, y)

        dx_u = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u_pred.sum(), y, create_graph=True)[0]

        A_grad_u_1 = a * dx_u + b * dy_u
        A_grad_u_2 = c * dx_u + d * dy_u

        dx_A_grad_u_1 = torch.autograd.grad(A_grad_u_1.sum(), x, create_graph=True)[0]
        dy_A_grad_u_2 = torch.autograd.grad(A_grad_u_2.sum(), y, create_graph=True)[0]

        residual = (
            dx_A_grad_u_1
            + dy_A_grad_u_2
            + sourceTerms.get_f(xT, yT, name=self.source_term)
        )

        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            residual.detach().cpu(),
            False,
            None,
            title="residual",
        )

    def plot_result_bernoulli(self, save_plots):
        import matplotlib.pyplot as plt

        makePlots.loss(self.loss_history, save_plots, self.fig_storage)

        n_visu = 25_000
        self.make_collocation(n_visu)
        # u_pred = self.get_u(self.x_collocation, self.y_collocation)
        x, y = self.x_collocation, self.y_collocation
        xT, yT = self.apply_symplecto(self.x_collocation, self.y_collocation)
        xT_gamma, yT_gamma = self.apply_symplecto(
            self.x_gamma_collocation, self.y_gamma_collocation
        )

        rho_2 = x**2 + y**2
        xT, yT = self.apply_symplecto(x, y)
        rhoT_2 = (xT / self.a) ** 2 + (yT / self.b) ** 2
        bc_mul = (rho_2 - self.rho_max**2) * (rhoT_2 - 1)
        bc_add = (self.rho_max**2 - rho_2) / (rhoT_2 - rho_2)

        u_pred = self.u_net(xT, yT) * bc_mul + bc_add
        dn_u_pred, _, _ = self.get_dn_u(
            self.x_gamma_collocation, self.y_gamma_collocation
        )

        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            u_pred.detach().cpu(),
            save_plots,
            self.fig_storage + "_pde",
            title="EDP",
        )
        makePlots.shape(
            x.detach().cpu(),
            y.detach().cpu(),
            save_plots,
            self.fig_storage + "_collocation_points",
        )
        makePlots.edp(
            xT_gamma.detach().cpu(),
            yT_gamma.detach().cpu(),
            dn_u_pred.detach().cpu(),
            save_plots,
            self.fig_storage + "_condition_optimalite",
            title="$\partial_n u_{pred}$",
        )

        a, b, c, d = self.get_metric_tensor(x, y)

        dx_u = torch.autograd.grad(u_pred.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u_pred.sum(), y, create_graph=True)[0]

        A_grad_u_1 = a * dx_u + b * dy_u
        A_grad_u_2 = c * dx_u + d * dy_u

        dx_A_grad_u_1 = torch.autograd.grad(A_grad_u_1.sum(), x, create_graph=True)[0]
        dy_A_grad_u_2 = torch.autograd.grad(A_grad_u_2.sum(), y, create_graph=True)[0]

        residual = dx_A_grad_u_1 + dy_A_grad_u_2

        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            residual.detach().cpu(),
            False,
            None,
            title="residual",
        )

        plt.show()
