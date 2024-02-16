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


class Bernoulli_Net:
    DEFAULT_DEEP_BERN_DICT = {
        "pde_learning_rate": 1e-2,
        "sympnet_learning_rate": 1e-2,
        "layer_sizes": [2, 10, 20, 10, 1],
        "nb_of_networks": 2,
        "networks_size": 5,
        "rho_min": 0.5,
        "rho_max": 1,
        "file_name": "default",
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "bernoulli",
    }

    # constructeur
    def __init__(self, deepDict, **kwargs):

        deepDict = kwargs.get("deepDict", self.DEFAULT_DEEP_BERN_DICT)

        if deepDict.get("pde_learning_rate") == None:
            deepDict["pde_learning_rate"] = self.DEFAULT_DEEP_BERN_DICT[
                "pde_learning_rate"
            ]
        if deepDict.get("sympnet_learning_rate") == None:
            deepDict["sympnet_learning_rate"] = self.DEFAULT_DEEP_BERN_DICT[
                "sympnet_learning_rate"
            ]
        if deepDict.get("layer_sizes") == None:
            deepDict["layer_sizes"] = self.DEFAULT_DEEP_BERN_DICT["layer_sizes"]
        if deepDict.get("nb_of_networks") == None:
            deepDict["nb_of_networks"] = self.DEFAULT_DEEP_BERN_DICT["nb_of_networks"]
        if deepDict.get("networks_size") == None:
            deepDict["networks_size"] = self.DEFAULT_DEEP_BERN_DICT["networks_size"]
        if deepDict.get("rho_min") == None:
            deepDict["rho_min"] = self.DEFAULT_DEEP_BERN_DICT["rho_min"]
        if deepDict.get("rho_max") == None:
            deepDict["rho_max"] = self.DEFAULT_DEEP_BERN_DICT["rho_max"]
        if deepDict.get("file_name") == None:
            deepDict["file_name"] = self.DEFAULT_DEEP_BERN_DICT["file_name"]
        if deepDict.get("source_term") == None:
            deepDict["source_term"] = self.DEFAULT_DEEP_BERN_DICT["source_term"]
        if deepDict.get("boundary_condition") == None:
            deepDict["boundary_condition"] = self.DEFAULT_DEEP_BERN_DICT[
                "boundary_condition"
            ]
        if deepDict.get("to_be_trained") == None:
            deepDict["to_be_trained"] = self.DEFAULT_DEEP_BERN_DICT["to_be_trained"]

        # Storage file
        self.file_name = (
            "./../../../outputs/deepShape/net/" + deepDict["file_name"] + ".pth"
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = os.path.join(script_dir, self.file_name)
        # Learning rate
        self.pde_learning_rate = deepDict["pde_learning_rate"]
        self.sympnet_learning_rate = deepDict["sympnet_learning_rate"]
        # Layer parameters
        self.layer_sizes = deepDict["layer_sizes"]
        self.nb_of_networks = deepDict["nb_of_networks"]
        self.networks_size = deepDict["networks_size"]
        # Geometry of the shape
        self.rho_min, self.rho_max = deepDict["rho_min"], deepDict["rho_max"]
        self.theta_min, self.theta_max = 0, 2 * torch.pi
        self.Vol = torch.pi * self.rho_max**2
        # Source term of the Poisson problem
        self.source_term = deepDict["source_term"]
        # Boundary condition of the Poisson problem
        self.boundary_condition = deepDict["boundary_condition"]

        self.create_networks()
        self.load(self.file_name)

        self.to_be_trained = deepDict["to_be_trained"]

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
        """
        Retourne le gradient normal dans TC, calculé à partir d'un point du cercle C

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
        dn_u : float
            gradient normal
        nxT : float
            1ère coordonnée cartésienne de la normale extérieure à TC
        nyT : float
            2ème coordonnée cartésienne de la normale extérieure à TC

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

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

        nxT, nyT = self.get_n(x, y)

        return Jt_dx_u * nxT + Jt_dy_u * nyT, nxT, nyT

    def get_n(self, x, y):
        """
        Retourne la normale dans TC, calculé à partir d'un point du cercle C

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
        nxT : float
            1ère coordonnée cartésienne de la normale extérieure à TC
        nyT : float
            2ème coordonnée cartésienne de la normale extérieure à TC

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        tx, ty = -y, x

        xT, yT = self.apply_symplecto(x, y)
        J_a = torch.autograd.grad(xT.sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(xT.sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(yT.sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(yT.sum(), y, create_graph=True)[0]
        txT, tyT = J_a * tx + J_b * ty, J_c * tx + J_d * ty
        nxT, nyT = tyT, -txT
        norm_nT = torch.sqrt(nxT**2 + nyT**2)
        nxT, nyT = nxT / norm_nT, nyT / norm_nT

        return nxT, nyT

    def get_avg_dn_u(self, x, y, n):
        """
        Retourne le gradient normal dans TC, calculé à partir d'un point du cercle C

        Parameters
        ----------
        self :
            instance de la classe network
        x : float
            1ère coordonnée cartésienne dans le cercle
        y : float
            2ème coordonnée cartésienne dans le cercle
        n : int
            nombre de pts de collocation

        Returns
        -------
        avg_dn_u : float
            moyenne du gradient normal sur le bord

        Raises
        ------
        KeyError
            when a key error
        OtherError
            when an other error
        """

        dn_u, _, _ = self.get_dn_u(x, y)
        avg_dn_u = dn_u.sum() / n * self.get_mes_border()
        return avg_dn_u

    def left_hand_term(self, x, y):
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
        return bc.apply_BC(
            self.u_net(*self.apply_symplecto(x, y)),
            x,
            y,
            self.rho_min,
            self.rho_max,
            name=self.boundary_condition,
        )

    def apply_rejet_kompact(self, xT, yT, a=0.8):
        b = self.rho_min**2 / a
        xT, yT = self.apply_symplecto(xT, yT)
        bool_rejet = (xT / a) ** 2 + (yT / b) ** 2 >= 1
        xT, yT = bool_rejet * xT, bool_rejet * yT

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
            self.y_collocation, self.y_collocation
        )

    def get_mes_border(self):
        n = 10_000
        theta = torch.linspace(
            self.theta_min, self.theta_max, n, requires_grad=True, device=device
        )[:, None]
        x = self.rho_max * torch.cos(theta)
        y = self.rho_max * torch.sin(theta)
        x, y = self.apply_symplecto(x, y)
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
                grad_u_2 = self.left_hand_term(self.x_collocation, self.y_collocation)
                dirichlet_loss = 0.5 * grad_u_2

                D = dirichlet_loss.sum() / n_collocation * self.Vol

                self.loss = D

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
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=[12.8, 9.6])

        ax[0, 0].plot(self.loss_history)
        ax[0, 0].set_yscale("symlog", linthresh=1e-4)
        ax[0, 0].set_title("loss history")

        n_visu = 25_000
        self.make_collocation(n_visu)

        xT, yT = self.apply_symplecto(self.x_collocation, self.y_collocation)
        xT_inv, yT_inv = self.apply_inverse_symplecto(xT, yT)
        u_pred = self.get_u(self.x_collocation, self.y_collocation).detach().cpu()
        # dn_u, _, _ = self.get_dn_u(self.x_collocation_max, self.y_collocation_max)

        self.x_collocation, self.y_collocation = (
            self.x_collocation.detach().cpu(),
            self.y_collocation.detach().cpu(),
        )
        xT, yT = xT.detach().cpu(), yT.detach().cpu()
        xT_inv, yT_inv = xT_inv.detach().cpu(), yT_inv.detach().cpu()

        im = ax[1, 1].scatter(
            xT,
            yT,
            s=1,
            c=u_pred,
            cmap="gist_ncar",
        )
        fig.colorbar(im, ax=ax[1, 1])
        ax[1, 1].set_title("solution approchée de l'EDP")
        ax[1, 1].set_aspect("equal")

        im = ax[1, 0].scatter(
            xT_inv,
            yT_inv,
            s=1,
        )
        ax[1, 0].set_title("solution approchée de l'EDP")
        ax[1, 0].set_aspect("equal")

        plt.show()

    def make_movie(self, epoch):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=[20, 15])

        n_visu = 1000
        self.make_collocation(n_visu)

        xT_max, yT_max = self.apply_symplecto(
            self.x_collocation_max, self.y_collocation_max
        )
        xT_min, yT_min = self.apply_symplecto(
            self.x_collocation_min, self.y_collocation_min
        )
        x_min_goal = self.a * self.x_collocation_min
        y_min_goal = 1 / self.a * self.y_collocation_min

        dn_u, _, _ = self.get_dn_u(self.x_collocation_max, self.y_collocation_max)

        im = ax.scatter(
            xT_max.detach().cpu(),
            yT_max.detach().cpu(),
            s=10,
            c=dn_u.detach().cpu(),
            cmap="gist_ncar",
            label="$|\partial_n u|$ sur la surface libre",
        )

        ax.scatter(
            x_min_goal.detach().cpu(),
            y_min_goal.detach().cpu(),
            marker="^",
            s=10,
            color="black",
            label="bord exact",
        )
        ax.scatter(
            xT_min.detach().cpu(),
            yT_min.detach().cpu(),
            s=5,
            color="red",
            label="bord pénalisé",
        )
        fig.colorbar(im, ax=ax)
        ax.legend()
        ax.set_aspect("equal")
        plt.title("epoch :" + str(epoch))

        plt.savefig("../data/deepShape/img/" + str(epoch) + ".png")  # , dpi=1200)
