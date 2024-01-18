"""
Author:
    A BELIERES FRENDO (ENSTA Paris)
Date:
    05/05/2023

To run:
    python3 poisson_symplec.py

Solve poisson EDP in a shape genereted by a symplectomorphism
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

# local imports
from gesonn.out1Plot import colormaps
from gesonn.com1PINNs import metricTensors

try:
    import torchinfo

    no_torchinfo = False
except ModuleNotFoundError:
    no_torchinfo = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is poisson.py")

# ----------------------------------------------------------------------
#   CLASSE NET - HERITIERE DE NN.DATAPARALLEL
# ----------------------------------------------------------------------


class PDE_Forward(nn.DataParallel):
    # constructeur
    def __init__(self, layer_sizes):
        super(PDE_Forward, self).__init__(nn.Module())

        self.hidden_layers = []
        for l1, l2 in zip(layer_sizes[:-1], layer_sizes[+1:]):
            self.hidden_layers.append(nn.Linear(l1, l2).double())
        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.output_layer = nn.Linear(layer_sizes[-1], 1).double()

    # forward function -> defines the network structure
    def forward(self, x, y, e):
        inputs = torch.cat([x, y, e], axis=1)

        layer_output = torch.tanh(self.hidden_layers[0](inputs))

        for hidden_layer in self.hidden_layers[1:]:
            layer_output = torch.tanh(hidden_layer(layer_output))

        # return torch.sigmoid(self.output_layer(layer_output))
        return self.output_layer(layer_output)


# ----------------------------------------------------------------------
#   CLASSE NETWORK - RESEAU DE NEURONES
#   approximation d'un symplectomorphisme
# ----------------------------------------------------------------------


class PINNs:
    DEFAULT_PINNS_DICT = {
        "learning_rate": 1e-3,
        "layer_sizes": [3, 10, 20, 20, 10, 1],
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "test",
        "symplecto_name": None,
        "SympNet": None,
        "to_be_trained": True,
    }

    # constructeur
    def __init__(self, **kwargs):
        PINNsDict = kwargs.get("PINNsDict", self.DEFAULT_PINNS_DICT)

        self.rho_min, self.rho_max = PINNsDict["rho_min"], PINNsDict["rho_max"]
        self.theta_min, self.theta_max = 0, 2 * torch.pi
        self.e_min, self.e_max = kwargs.get("e_min", 0.5), kwargs.get("e_max", 10)

        self.Vol = torch.pi * self.rho_max**2

        self.file_name = (
            "./../../../outputs/PINNs/net/poisson_" + PINNsDict["file_name"] + ".pth"
        )
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = os.path.join(script_dir, self.file_name)

        self.learning_rate = PINNsDict["learning_rate"]

        # taille des différentes couches du réseau de neurones
        self.layer_sizes = PINNsDict["layer_sizes"]

        self.name_symplecto = PINNsDict["symplecto_name"]
        self.SympNet = PINNsDict["SympNet"]

        self.create_network()
        self.load(self.file_name)

        self.to_be_trained = PINNsDict["to_be_trained"]

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
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "e_min": self.e_min,
            "e_max": self.e_max,
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

    def get_metric_tensor(self, x, y):
        # il faut calculer :
        # A = J_T^{-t}*J_T^{-1}

        if self.name_symplecto != None or self.SympNet != None:
            T = metricTensors.apply_symplecto(
                x, y, name=self.name_symplecto, SympNet=self.SympNet
            )

            J_a = torch.autograd.grad(T[0].sum(), x, create_graph=True)[0]
            J_b = torch.autograd.grad(T[0].sum(), y, create_graph=True)[0]
            J_c = torch.autograd.grad(T[1].sum(), x, create_graph=True)[0]
            J_d = torch.autograd.grad(T[1].sum(), y, create_graph=True)[0]

            fac = (J_a * J_d - J_b * J_c) ** 2
            A_a = (J_d**2 + J_b**2) / fac
            A_b = -(J_c * J_d + J_a * J_b) / fac
            A_c = -(J_c * J_d + J_a * J_b) / fac
            A_d = (J_c**2 + J_a**2) / fac

        elif self.name_symplecto == None and self.SympNet == None:
            A_a = 1
            A_b = 0
            A_c = 0
            A_d = 1

        return A_a, A_b, A_c, A_d

    def left_hand_term(self, x, y, e):
        u = self.get_u(x, y, e)
        a, b, c, d = self.get_metric_tensor(x, y)

        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]

        A_grad_u_grad_u = (a * dx_u + b * dy_u) * dx_u + (c * dx_u + d * dy_u) * dy_u

        return A_grad_u_grad_u

    def right_hand_term(self, x, y, e):
        u = self.get_u(x, y, e)

        # terme source
        f = self.get_f(x, y, e)

        return f * u

    def get_dn_u(self, x, y, e):
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

        xT, yT = metricTensors.apply_symplecto(
            x, y, name=self.name_symplecto, SympNet=self.SympNet
        )

        J_a = torch.autograd.grad(xT.sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(xT.sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(yT.sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(yT.sum(), y, create_graph=True)[0]

        det = J_a * J_d - J_b * J_c
        a, b, c, d = det * J_d, -det * J_c, -det * J_b, det * J_a

        u = self.get_u(x, y, e)

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

        xT, yT = metricTensors.apply_symplecto(
            x, y, name=self.name_symplecto, SympNet=self.SympNet
        )
        J_a = torch.autograd.grad(xT.sum(), x, create_graph=True)[0]
        J_b = torch.autograd.grad(xT.sum(), y, create_graph=True)[0]
        J_c = torch.autograd.grad(yT.sum(), x, create_graph=True)[0]
        J_d = torch.autograd.grad(yT.sum(), y, create_graph=True)[0]
        txT, tyT = J_a * tx + J_b * ty, J_c * tx + J_d * ty
        nxT, nyT = tyT, -txT
        norm_nT = torch.sqrt(nxT**2 + nyT**2)
        nxT, nyT = nxT / norm_nT, nyT / norm_nT

        return nxT, nyT

    def network_BC_mul(self, x, y):
        rho_2 = x**2 + y**2
        bc_mul = (rho_2 - self.rho_max**2)
        if self.rho_min > 0:
            bc_mul = bc_mul * (self.rho_min**2 - rho_2)
        return bc_mul

    def network_BC_add(self, x, y):
        return 0
        rho_2 = x**2 + y**2
        bc_add = 1 - (rho_2 - self.rho_min**2) / (self.rho_max**2 - self.rho_min**2)
        return bc_add

    def get_u(self, x, y, e):
        return self.u_net(
            *metricTensors.apply_symplecto(
                x, y, name=self.name_symplecto, SympNet=self.SympNet
            ),
            e,
        ) * self.network_BC_mul(x, y) + self.network_BC_add(x,y)

    def get_f(self, x, y, e):
        x, y = metricTensors.apply_symplecto(
            x, y, name=self.name_symplecto, SympNet=self.SympNet
        )
        # return e
        r2 = (x / e) ** 2 + (e * y) ** 2
        return torch.exp(1 - r2)

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
        theta_collocation = self.random(
            0, 2 * torch.math.pi, shape, requires_grad=True
        )

        self.x_collocation = rho_collocation * torch.cos(theta_collocation)
        self.y_collocation = rho_collocation * torch.sin(theta_collocation)
        self.e_collocation = self.random(
            self.e_min, self.e_max, shape, requires_grad=True
        )

    def train(self, **kwargs):
        # nombre de pas de descente
        epochs = kwargs.get("epochs", 500)
        # nombre de pts tirés pour monte-carlo
        n_collocation = kwargs.get("n_collocation", 10000)

        plot_history = kwargs.get("plot_history", False)

        # trucs de sauvegarde
        try:
            best_loss_value = self.loss.item()
        except AttributeError:
            best_loss_value = 1e10

        # boucle principale de la descnet ede gradient
        for epoch in range(epochs):
            # mise à 0 du gradient
            self.u_optimizer.zero_grad()

            # mise à 0 de la loss
            self.loss = torch.tensor([0.0], device=device)

            # Loss based on PDE
            if n_collocation > 0:
                self.make_collocation(n_collocation)

                grad_u_2 = self.left_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                    self.e_collocation,
                )
                fu = self.right_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                    self.e_collocation,
                )

                dirichlet_loss = 0.5 * grad_u_2 - fu
                self.loss = dirichlet_loss.sum() / n_collocation * self.Vol #* (self.e_max - self.e_min)

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
            self.load(self.file_name)
        except UnboundLocalError:
            pass

        if plot_history:
            self.plot_result()

    def plot_result(self, derivative=False, random=False):
        import matplotlib.pyplot as plt
        from matplotlib import rc

        rc("font", **{"family": "serif", "serif": ["fontenc"], "size": 15})
        rc("text", usetex=True)

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].plot(self.loss_history)
        ax[0, 0].set_yscale("symlog", linthresh=1e-4)

        n_visu = 50_000

        self.ones = torch.ones((n_visu, 1), requires_grad=True, device=device)
        e_visu = (self.e_max) * self.ones

        self.make_collocation(n_visu)
        u_pred = self.get_u(self.x_collocation, self.y_collocation, e_visu)

        xT, yT = metricTensors.apply_symplecto(
            self.x_collocation,
            self.y_collocation,
            name=self.name_symplecto,
            SympNet=self.SympNet,
        )
        xT = xT.detach().cpu()
        yT = yT.detach().cpu()

        im = ax[0, 1].scatter(
            xT,
            yT,
            s=1,
            c=u_pred.detach().cpu(),
            cmap=colormaps.make_colormap(),
        )
        fig.colorbar(im, ax=ax[0, 1])
        ax[0, 1].set_aspect("equal")
        ax[0, 1].set_title("Solution de l'EDP tensorisée")
        ax[0, 1].legend()

        # u_ex = 0.25 * (1 - self.x_collocation**2 - self.y_collocation**2)
        
        # im = ax[1, 0].scatter(
        #     xT,
        #     yT,
        #     s=1,
        #     c=u_ex.detach().cpu(),
        #     cmap=colormaps.make_colormap(),
        # )
        # fig.colorbar(im, ax=ax[1, 0])
        # ax[1, 0].set_aspect("equal")
        # ax[1, 0].set_title("u_exact")
        # ax[1, 0].legend()
        
        # err = (u_pred - u_ex)**2
        # im = ax[1, 1].scatter(
        #     xT,
        #     yT,
        #     s=1,
        #     c=err.detach().cpu(),
        #     cmap=colormaps.make_colormap(),
        # )
        # fig.colorbar(im, ax=ax[1, 1])
        # ax[1, 1].set_aspect("equal")
        # ax[1, 1].set_title("erreur")
        # ax[1, 1].legend()

        # n_border = 10_000
        # theta_border = self.random(
        #     self.theta_min, self.theta_max, n_border, requires_grad=True, device=device
        # )[:-1, None]

        # x_max = self.rho_max * torch.cos(theta_border)
        # y_max = self.rho_max * torch.sin(theta_border)
        # xT_max, yT_max = metricTensors.apply_symplecto(
        #     x_max,
        #     y_max,
        #     name=self.name_symplecto,
        #     SympNet=self.SympNet,
        # )
        # dn_u, _, _ = self.get_dn_u(x_max, y_max, e_visu)

        # im = ax[1, 0].scatter(
        #     xT_max.detach().cpu(),
        #     yT_max.detach().cpu(),
        #     s=1,
        #     c=dn_u.detach().cpu(),
        #     cmap=colormaps.make_colormap(),
        # )
        # fig.colorbar(im, ax=ax[1, 0])
        # ax[1, 0].set_aspect("equal")
        # ax[1, 0].set_title("Solution de l'EDP tensorisée")
        # ax[1, 0].legend()

        plt.show()

