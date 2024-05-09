# imports
import os

import torch

# local imports
from gesonn.com1PINNs import poisson
from gesonn.com1PINNs import metricTensors

from gesonn.ana1Tests import PINNs

from gesonn.out1Plot import makePlots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

if __name__ == "__main__":

    # ==============================================================
    # Parameters to be modified freely by the user
    # ==============================================================

    train = True
    # train = False

    PINNsDict = {
        "learning_rate": 1e-2,
        "layer_sizes": [2, 10, 20, 40, 40, 20, 10, 1],
        "rho_min": 0.2,
        "rho_max": 1,
        "file_name": "PINNS_compare_ritz_pinns",
        "symplecto_name": "bizaroid",
        "to_be_trained": train,
        "source_term": "one",
        "boundary_condition": "homogeneous_dirichlet",
    }

    DeepRitzDict = {
        "learning_rate": 1e-2,
        "layer_sizes": [2, 10, 20, 40, 40, 20, 10, 1],
        "rho_min": 0.2,
        "rho_max": 1,
        "file_name": "DeepRitz_compare_ritz_pinns",
        "symplecto_name": "bizaroid",
        "to_be_trained": train,
        "source_term": "one",
        "boundary_condition": "homogeneous_dirichlet",
    }

    epochs = 5_000
    n_collocation = 10_000
    new_training = False
    # new_training = True
    save_plots = False
    save_plots = True

    # ==============================================================
    # End of the modifiable area
    # ==============================================================

    if train:
        if new_training:
            try:
                os.remove("./../outputs/PINNs/net/" + PINNsDict["file_name"] + ".pth")
                os.remove(
                    "./../outputs/PINNs/net/" + DeepRitzDict["file_name"] + ".pth"
                )
            except FileNotFoundError:
                pass

        PINNS_network = PINNs.PINNs(PINNsDict=PINNsDict)
        DeepRitz_network = poisson.PINNs(PINNsDict=DeepRitzDict)

        _ = PINNS_network.train(
            epochs=epochs,
            n_collocation=n_collocation,
            plot_history=False,
            save_plots=save_plots,
        )
        _ = DeepRitz_network.train(
            epochs=epochs,
            n_collocation=n_collocation,
            plot_history=False,
            save_plots=save_plots,
        )

        makePlots.loss(PINNS_network.loss_history, False, None)
        makePlots.loss(DeepRitz_network.loss_history, False, None)


        makePlots.edp_contour(
            PINNS_network.rho_min,
            PINNS_network.rho_max,
            PINNS_network.get_u,
            lambda x, y: metricTensors.apply_symplecto(
                x, y, name=PINNS_network.name_symplecto
            ),
            lambda x, y: metricTensors.apply_symplecto(
                x, y, name=f"inverse_{PINNS_network.name_symplecto}"
            ),
            False,
            None,
        )

        makePlots.edp_contour(
            DeepRitz_network.rho_min,
            DeepRitz_network.rho_max,
            DeepRitz_network.get_u,
            lambda x, y: metricTensors.apply_symplecto(
                x, y, name=DeepRitz_network.name_symplecto
            ),
            lambda x, y: metricTensors.apply_symplecto(
                x, y, name=f"inverse_{DeepRitz_network.name_symplecto}"
            ),
            False,
            None,
        )

        makePlots.edp_contour(
            DeepRitz_network.rho_min,
            DeepRitz_network.rho_max,
            DeepRitz_network.get_res,
            lambda x, y: metricTensors.apply_symplecto(
                x, y, name=DeepRitz_network.name_symplecto
            ),
            lambda x, y: metricTensors.apply_symplecto(
                x, y, name=f"inverse_{DeepRitz_network.name_symplecto}"
            ),
            False,
            None,
        )

        def get_pointwise_err(x, y):
            return DeepRitz_network.get_u(x, y) - PINNS_network.get_u(x, y)


        makePlots.edp_contour(
            DeepRitz_network.rho_min,
            DeepRitz_network.rho_max,
            get_pointwise_err,
            lambda x, y: metricTensors.apply_symplecto(
                x, y, name=DeepRitz_network.name_symplecto
            ),
            lambda x, y: metricTensors.apply_symplecto(
                x, y, name=f"inverse_{DeepRitz_network.name_symplecto}"
            ),
            False,
            None,
        )
