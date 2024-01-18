"""
Author:
    A BELIERES FRENDO (ENSTA Paris)
Date:
    05/05/2023

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
import pandas as pd
import math

# local imports
from soviets.com1PINNs import poissonParam
from soviets.com2SympNets import GParam
from soviets.out1Plot import colormaps

try:
    import torchinfo

    no_torchinfo = False
except ModuleNotFoundError:
    no_torchinfo = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is bernoulli.py")


# ----------------------------------------------------------------------
#   CLASSE NETWORK - RESEAU DE NEURONES
# ----------------------------------------------------------------------


class Geo_Net:
    # constructeur
    def __init__(self, deepGeoDict, **kwargs):
        self.rho_min, self.rho_max = deepGeoDict["rho_min"], deepGeoDict["rho_max"]
        self.theta_min, self.theta_max = kwargs.get("theta_min", 0), kwargs.get(
            "theta_max", 2 * torch.pi
        )
        self.mu_min, self.mu_max = kwargs.get("mu_min", 0.5), kwargs.get("mu_max", 2)

        self.Vol = torch.math.pi * (self.rho_max**2 - self.rho_min**2)

        self.file_name = "./../data/deepShape/net/" + deepGeoDict["file_name"] + ".pth"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = os.path.join(script_dir, self.file_name)

        self.pde_learning_rate = deepGeoDict["pde_learning_rate"]
        self.sympnet_learning_rate = deepGeoDict["sympnet_learning_rate"]

        # taille des différentes couches du réseau de neurones
        self.layer_sizes = deepGeoDict["layer_sizes"]
        self.nb_of_networks = deepGeoDict["nb_of_networks"]
        self.networks_size = deepGeoDict["networks_size"]

        self.create_networks()
        self.load(self.file_name)

        self.to_be_trained = deepGeoDict["to_be_trained"]

    def sympnet_layer_append(self, nets, optims, i):
        nets.append(
            nn.DataParallel(GParam.Symp_Net_Forward(self.networks_size)).to(device)
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
        self.u_net = nn.DataParallel(poissonParam.PDE_Forward(self.layer_sizes)).to(
            device
        )
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

    def get_metric_tensor(self, x, y, mu):
        """
        Calcule le tenseur métrique dans TC, évalué dans un point du cercle C

        Parameters
        ----------
        self :
            instance de la classe network
        x : float
            1ère coordonnée cartésienne dans le cercle
        y : float
            2ème coordonnée cartésienne dans le cercle

        Returns
        -------
        A_a, A_b, A_c, A_d : 4-uple de floats
            A = J_T^{-t}*J_T^{-1}(x, y)

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        T = self.apply_symplecto(x, y, mu)

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

    def left_hand_term(self, x, y, mu):
        """
        Retourne l'intégrande du terme de gauche dans TC, calculé à partir d'un point du cercle C

        Parameters
        ----------
        self :
            instance de la classe network
        x : float
            1ère coordonnée cartésienne dans le cercle
        y : float
            2ème coordonnée cartésienne dans le cercle
        rho : float
            1ère coordonnée polaire dans le cercle

        Returns
        -------
        A_grad_u_grad_u : float
            intégrande

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        u = self.get_u(x, y, mu)
        a, b, c, d = self.get_metric_tensor(x, y, mu)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        A_grad_u_grad_u = (a * dx_u + b * dy_u) * dx_u + (c * dx_u + d * dy_u) * dy_u

        return A_grad_u_grad_u

    def apply_symplecto(self, x, y, mu):
        for i in range(self.nb_of_networks):
            x = x + self.up_nets[i](y, mu)
            y = y + self.down_nets[i](x, mu)
        return x, y

    def network_BC_mul(self, x, y):
        rho_2 = x**2 + y**2
        bc_mul = (rho_2 - self.rho_max**2) * (self.rho_min**2 - rho_2)
        return bc_mul

    def network_BC_add(self, x, y):
        rho_2 = x**2 + y**2
        bc_add = 1 - (rho_2 - self.rho_min**2) / (
            self.rho_max**2 - self.rho_min**2
        )
        return bc_add

    def get_u(self, x, y, mu):
        return self.u_net(*self.apply_symplecto(x, y, mu), mu) * self.network_BC_mul(
            x, y
        ) + self.network_BC_add(x, y)

    def get_pen(self, x, y, mu):
        xT, yT = self.apply_symplecto(x, y, mu)
        x_goal = mu * x
        y_goal = 1 / mu * y

        return (xT - x_goal) ** 2 + (yT - y_goal) ** 2

    @staticmethod
    def random(min_value, max_value, shape, requires_grad=False, device=device):
        random_numbers = torch.rand(
            shape, device=device, dtype=torch.double, requires_grad=requires_grad
        )
        return min_value + (max_value - min_value) * random_numbers

    def make_collocation(self, n_collocation, plotting=False):
        shape = (n_collocation, 1)

        rho_collocation = torch.sqrt(
            self.random(self.rho_min**2, self.rho_max**2, shape, requires_grad=True)
        )
        theta_collocation = self.random(
            self.theta_min, self.theta_max, shape, requires_grad=True
        )

        self.x_collocation = rho_collocation * torch.cos(theta_collocation)
        self.y_collocation = rho_collocation * torch.sin(theta_collocation)
        self.mu_collocation = self.random(
            self.mu_min, self.mu_max, shape, requires_grad=True
        )

        if plotting:
            self.x_collocation_max = self.rho_max * torch.cos(theta_collocation)
            self.y_collocation_max = self.rho_max * torch.sin(theta_collocation)
        self.x_collocation_min = self.rho_min * torch.cos(theta_collocation)
        self.y_collocation_min = self.rho_min * torch.sin(theta_collocation)

        self.ones = torch.ones(shape, device=device)

    def get_mes_border(self):
        n = 10_000
        theta = torch.linspace(
            self.theta_min, self.theta_max, n, requires_grad=True, device=device
        )[:, None]
        x = self.rho_max * torch.cos(theta)
        y = self.rho_max * torch.sin(theta)
        x, y = self.apply_symplecto(x, y, mu)
        lenghts = torch.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)

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

        # boucle principale de la descnete de gradient
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
                    self.x_collocation, self.y_collocation, self.mu_collocation
                )
                dirichlet_loss = 0.5 * grad_u_2

                pen = self.get_pen(
                    self.x_collocation_min, self.y_collocation_min, self.mu_collocation
                )

                alpha = 50
                # beta = 1/10
                # if epoch > 200:
                #     alpha = 0.1
                #     if epoch >500:
                #         alpha = 1

                # a_D = 250 * math.log(2 + epoch / 500) / math.log(2)
                # a_P = 2.5 * math.log(2 + epoch / 500) / math.log(2)
                # a_N = 1

                D = dirichlet_loss.sum() / n_collocation * self.Vol
                P = alpha * pen.sum() / n_collocation * self.Vol
                # N = beta * optimality_condition.sum() / n_collocation * self.get_mes_border()
                if epoch == 0:
                    self.D0 = copy.deepcopy(D.item())
                    self.P0 = copy.deepcopy(P.item())
                    # self.N0 = copy.deepcopy(N.item())

                # self.loss = (
                #     (dirichlet_loss + alpha * pen).sum() / n_collocation * self.Vol
                # )  + 0.01*optimality_condition.sum() / n_collocation * self.get_mes_border()

                # try:
                #     self.loss = (
                #         a_D * D / self.D0 + a_P * P / self.P0 #+ a_N * N / self.N0
                #     )
                #     print(
                #         f"{a_D * D.item() / self.D0:5.2e}",
                #         f"{a_P * P.item() / self.P0:5.2e}",
                #         # f"{a_N * N.item() / self.N0:5.2e}",
                #     )
                # except KeyboardInterrupt:
                #     self.plot_result(epoch)
                #     raise

                self.loss = D + P  # + N

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

    @staticmethod
    def copy_sympnet(to_be_copied):
        return [copy.deepcopy(copie.state_dict()) for copie in to_be_copied]

    def plot_result(self, derivative=False, random=False):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=[12.8, 9.6])

        ax[0, 0].plot(self.loss_history)
        ax[0, 0].set_yscale("symlog", linthresh=1e-4)
        ax[0, 0].set_title("loss history")

        n_visu = 10_000
        self.make_collocation(n_visu, plotting=True)

        mu_visu1 = torch.ones((n_visu, 1), requires_grad=True, device=device) * (
            0.25 * self.mu_max + 0.75 * self.mu_min
        )
        mu_visu2 = torch.ones((n_visu, 1), requires_grad=True, device=device) * (
            0.5 * self.mu_max + 0.5 * self.mu_min
        )
        mu_visu3 = torch.ones((n_visu, 1), requires_grad=True, device=device) * (
            0.75 * self.mu_max + 0.25 * self.mu_min
        )

        xT_max1, yT_max1 = self.apply_symplecto(
            self.x_collocation_max, self.y_collocation_max, mu_visu1
        )
        xT_min1, yT_min1 = self.apply_symplecto(
            self.x_collocation_min, self.y_collocation_min, mu_visu1
        )
        xT_max2, yT_max2 = self.apply_symplecto(
            self.x_collocation_max, self.y_collocation_max, mu_visu2
        )
        xT_min2, yT_min2 = self.apply_symplecto(
            self.x_collocation_min, self.y_collocation_min, mu_visu2
        )
        xT_max3, yT_max3 = self.apply_symplecto(
            self.x_collocation_max, self.y_collocation_max, mu_visu3
        )
        xT_min3, yT_min3 = self.apply_symplecto(
            self.x_collocation_min, self.y_collocation_min, mu_visu3
        )

        xT, yT = self.apply_symplecto(self.x_collocation, self.y_collocation, mu_visu2)
        u_pred = self.get_u(self.x_collocation, self.y_collocation, mu_visu2).detach().cpu()

        xT = xT.detach().cpu()
        yT = yT.detach().cpu()

        ax[1, 0].scatter(
            xT_min1.detach().cpu(),
            yT_min1.detach().cpu(),
            s=1,
            label="1",
        )
        ax[1, 0].scatter(
            xT_max1.detach().cpu(),
            yT_max1.detach().cpu(),
            s=1,
            label="1",
        )
        ax[1, 0].scatter(
            xT_min2.detach().cpu(),
            yT_min2.detach().cpu(),
            s=1,
            label="2",
        )
        ax[1, 0].scatter(
            xT_max2.detach().cpu(),
            yT_max2.detach().cpu(),
            s=1,
            label="2",
        )
        ax[1, 0].scatter(
            xT_min3.detach().cpu(),
            yT_min3.detach().cpu(),
            s=1,
            label="3",
        )
        ax[1, 0].scatter(
            xT_max3.detach().cpu(),
            yT_max3.detach().cpu(),
            s=1,
            label="3",
        )
        ax[1, 0].legend()
        ax[1, 0].set_title("forme optimale")
        ax[1, 0].set_aspect("equal")

        im = ax[1, 1].scatter(
            xT,
            yT,
            s=1,
            c=u_pred,
            cmap=colormaps.make_colormap(),
        )
        fig.colorbar(im, ax=ax[1, 1])
        ax[1, 1].set_title("solution approchée de l'EDP")
        ax[1, 1].set_aspect("equal")

        # Chargement de la solution FreeFem++
        fem_path = "./../fem/data/bernoulli_u.csv"
        if os.path.isfile(fem_path):
            dict = pd.read_csv(fem_path, delimiter=";")
        else:
            fem_path = "./../../fem/data/bernoulli_u.csv"
            if os.path.isfile(fem_path):
                dict = pd.read_csv(fem_path, delimiter=";")
            else:
                raise FileNotFoundError("Could not find fem solution storage file")

        X = torch.tensor(dict["x"], requires_grad=True, device=device)[:, None]
        Y = torch.tensor(dict["y"], requires_grad=True, device=device)[:, None]
        Ufem = (torch.tensor(dict["u"], device=device)[:, None]).detach().cpu()

        # Affichage

        im = ax[0, 1].scatter(
            X.detach().cpu(),
            Y.detach().cpu(),
            s=1,
            c=Ufem,
            cmap=colormaps.make_colormap(),
            label="u_{fem}",
        )
        ax[0, 1].set_title("FEM")
        ax[0, 1].set_aspect("equal")
        fig.colorbar(im, ax=ax[0, 1])
        # ax[0, 1].scatter(
        #     xT,
        #     yT,
        #     s=1,
        #     c=u_pred,
        #     cmap=colormaps.make_colormap(),
        #     label="forme optimale",
        #     color="green",
        # )
        # ax[0, 1].scatter(
        #     xT_max.detach().cpu(),
        #     yT_max.detach().cpu(),
        #     s=1,
        #     color="green",
        # )
        # ax[0, 1].legend()
        # ax[0, 1].scatter(
        #     xT_min.detach().cpu(),
        #     yT_min.detach().cpu(),
        #     s=1,
        #     color="green",
        # )
        # ax[0, 1].legend()

        # print(self.get_mes_border())

        plt.show()
