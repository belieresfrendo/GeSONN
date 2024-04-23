"""
Author:
    A BELIERES FRENDO (IRMA)
Date:
    05/05/2023

Solve poisson EDP in a shape genereted by a symplectomorphism
-Delta u = f
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

# local imports
from gesonn.out1Plot import makePlots
from gesonn.com1PINNs import boundary_conditions as bc
from gesonn.com1PINNs import metricTensors
from gesonn.com1PINNs import sourceTerms

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
    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)

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
        "learning_rate": 5e-3,
        "layer_sizes": [2, 10, 20, 20, 10, 1],
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "default",
        "symplecto_name": None,
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "dirichlet_homgene",
    }

    # constructeur
    def __init__(self, **kwargs):
        PINNsDict = kwargs.get("PINNsDict", self.DEFAULT_PINNS_DICT)

        if PINNsDict.get("learning_rate") == None:
            PINNsDict["learning_rate"] = self.DEFAULT_PINNS_DICT["learning_rate"]
        if PINNsDict.get("layer_sizes") == None:
            PINNsDict["layer_sizes"] = self.DEFAULT_PINNS_DICT["layer_sizes"]
        if PINNsDict.get("rho_min") == None:
            PINNsDict["rho_min"] = self.DEFAULT_PINNS_DICT["rho_min"]
        if PINNsDict.get("rho_max") == None:
            PINNsDict["rho_max"] = self.DEFAULT_PINNS_DICT["rho_max"]
        if PINNsDict.get("a") is None:
            PINNsDict["a"] = self.DEFAULT_DEEP_GEO_DICT["a"]
        if PINNsDict.get("file_name") == None:
            PINNsDict["file_name"] = self.DEFAULT_PINNS_DICT["file_name"]
        if PINNsDict.get("symplecto_name") == None:
            PINNsDict["symplecto_name"] = self.DEFAULT_PINNS_DICT["symplecto_name"]
        if PINNsDict.get("source_term") == None:
            PINNsDict["source_term"] = self.DEFAULT_PINNS_DICT["source_term"]
        if PINNsDict.get("boundary_condition") == None:
            PINNsDict["boundary_condition"] = self.DEFAULT_PINNS_DICT[
                "boundary_condition"
            ]
        if PINNsDict.get("to_be_trained") == None:
            PINNsDict["to_be_trained"] = self.DEFAULT_PINNS_DICT["to_be_trained"]

        # Storage file
        self.file_name = (
            "./../../../outputs/PINNs/net/" + PINNsDict["file_name"] + ".pth"
        )
        self.fig_storage = "./../outputs/PINNs/img/" + PINNsDict["file_name"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = os.path.join(script_dir, self.file_name)
        # Learning rate
        self.learning_rate = PINNsDict["learning_rate"]
        # Layer sizes
        self.layer_sizes = PINNsDict["layer_sizes"]
        # Geometry of the shape
        self.rho_min, self.rho_max = PINNsDict["rho_min"], PINNsDict["rho_max"]
        self.theta_min, self.theta_max = 0, 2 * torch.pi
        self.Vol = torch.pi * self.rho_max**2
        # Parameters of the compact set K
        self.a = PINNsDict["a"]
        self.b = self.rho_min**2 / self.a
        self.name_symplecto = PINNsDict["symplecto_name"]
        # Source term of the Poisson problem
        self.source_term = PINNsDict["source_term"]
        # Boundary condition of the Poisson problem
        self.boundary_condition = PINNsDict["boundary_condition"]

        self.create_network()
        self.load(self.file_name)

        self.to_be_trained = PINNsDict["to_be_trained"]

        # flag for saving results
        self.save_results = kwargs.get("save_results", False)

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

        if self.name_symplecto is not None:
            T = metricTensors.apply_symplecto(x, y, name=self.name_symplecto)

            J_a = torch.autograd.grad(T[0].sum(), x, create_graph=True)[0]
            J_b = torch.autograd.grad(T[0].sum(), y, create_graph=True)[0]
            J_c = torch.autograd.grad(T[1].sum(), x, create_graph=True)[0]
            J_d = torch.autograd.grad(T[1].sum(), y, create_graph=True)[0]

            fac = (J_a * J_d - J_b * J_c) ** 2
            A_a = (J_d**2 + J_b**2) / fac
            A_b = -(J_c * J_d + J_a * J_b) / fac
            A_c = -(J_c * J_d + J_a * J_b) / fac
            A_d = (J_c**2 + J_a**2) / fac

        else:
            A_a = 1
            A_b = 0
            A_c = 0
            A_d = 1

        return A_a, A_b, A_c, A_d

    def left_hand_term(self, x, y):
        u = self.get_u(x, y)
        a, b, c, d = self.get_metric_tensor(x, y)
        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        A_grad_u_grad_u = (a * dx_u + b * dy_u) * dx_u + (c * dx_u + d * dy_u) * dy_u

        g = self.get_g(x, y)
        dx_g = torch.autograd.grad(g.sum(), x, create_graph=True, allow_unused=True)[0]
        dy_g = torch.autograd.grad(g.sum(), y, create_graph=True, allow_unused=True)[0]
        
        # A_grad_g_grad_g = (a * dx_g + b * dy_g) * dx_g + (c * dx_g + d * dy_g) * dy_g

        # return A_grad_u_grad_u - A_grad_g_grad_g

        T = metricTensors.apply_symplecto(x, y)
        # J_a = torch.autograd.grad(T[0].sum(), x, create_graph=True, allow_unused=True)[0]
        # J_b = torch.autograd.grad(T[0].sum(), y, create_graph=True, allow_unused=True)[0]
        # J_c = torch.autograd.grad(T[1].sum(), x, create_graph=True, allow_unused=True)[0]
        # J_d = torch.autograd.grad(T[1].sum(), y, create_graph=True, allow_unused=True)[0]
        J_a, J_b, J_c, J_d = 1, 0, 0, 1
        inv_det = 1/(J_a*J_d - J_b*J_c)
        Jt_dx_g = inv_det * (J_d * dx_g - J_c * dy_g)
        Jt_dy_g = inv_det * (-J_b * dx_g + J_a * dy_g)
        Jt_grad_g_grad_u = Jt_dx_g * dx_u + Jt_dy_g * dy_u

        return A_grad_u_grad_u + 2 * Jt_grad_g_grad_u

    def right_hand_term(self, x, y):
        u = self.get_u(x, y)

        # terme source
        f = sourceTerms.get_f(
            *metricTensors.apply_symplecto(x, y, name=self.name_symplecto),
            name=self.source_term,
        )

        return f * u

    def get_g(self, x, y):
        rho_2 = x**2 + y**2
        xT, yT = metricTensors.apply_symplecto(x, y, name=self.name_symplecto)
        rhoT_2 = (xT / self.a) ** 2 + (yT / self.b) ** 2
        g = (self.rho_max**2 - rho_2) / (rhoT_2 - rho_2)
        return g

    def get_res(self, x, y):

        # LHS
        u = self.get_u(x, y) + self.get_g(x, y)
        a, b, c, d = self.get_metric_tensor(x, y)
        dx_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dy_u = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
        A_grad_u_x = a * dx_u + b * dy_u
        A_grad_u_y = c * dx_u + d * dy_u
        dx_A_grad_u_x = torch.autograd.grad(A_grad_u_x.sum(), x, create_graph=True)[0]
        dy_A_grad_u_y = torch.autograd.grad(A_grad_u_y.sum(), y, create_graph=True)[0]
        div_A_grad_u = dx_A_grad_u_x + dy_A_grad_u_y

        # RHS
        f = sourceTerms.get_f(
            *metricTensors.apply_symplecto(x, y, name=self.name_symplecto),
            name=self.source_term,
        )

        return div_A_grad_u + f

    def get_u(self, x, y):
        xT, yT = metricTensors.apply_symplecto(x, y, name=self.name_symplecto)
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

    def train(self, **kwargs):
        # nombre de pas de descente
        epochs = kwargs.get("epochs", 500)
        # nombre de pts tirés pour monte-carlo
        n_collocation = kwargs.get("n_collocation", 10000)

        plot_history = kwargs.get("plot_history", False)
        save_plots = kwargs.get("save_plots", False)

        # trucs de sauvegarde
        try:
            best_loss_value = self.loss.item()
        except AttributeError:
            best_loss_value = 1e10

        # boucle principale de la descnet ede gradient
        tps1 = time.time()
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
                )
                fu = self.right_hand_term(
                    self.x_collocation,
                    self.y_collocation,
                )

                dirichlet_loss = 0.5 * grad_u_2 - fu
                self.loss = dirichlet_loss.sum() / n_collocation * self.Vol

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
        tps2 = time.time()

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
            self.plot_result(save_plots)

        return tps2 - tps1

    def plot_result(self, save_plots):

        makePlots.loss(self.loss_history, save_plots, self.fig_storage)

        n_visu = 100_000
        self.make_collocation(n_visu)

        u_pred = self.get_u(self.x_collocation, self.y_collocation)
        x, y = self.x_collocation, self.y_collocation
        xT, yT = metricTensors.apply_symplecto(
            self.x_collocation,
            self.y_collocation,
            name=self.name_symplecto,
        )
        g = self.get_g(x, y)
        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            (u_pred + g).detach().cpu(),
            save_plots,
            self.fig_storage,
            title="EDP",
        )

        abs_res = torch.abs(self.get_res(self.x_collocation, self.y_collocation))

        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            torch.log(abs_res).detach().cpu(),
            save_plots,
            self.fig_storage,
            title="RES",
        )

        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            abs_res.detach().cpu(),
            save_plots,
            self.fig_storage,
            title="RES",
        )

        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            torch.minimum(torch.ones_like(abs_res), abs_res).detach().cpu(),
            save_plots,
            self.fig_storage,
            title="RES",
        )

        makePlots.edp(
            xT.detach().cpu(),
            yT.detach().cpu(),
            self.get_g(x, y).detach().cpu(),
            save_plots,
            self.fig_storage,
            title="gé",
        )