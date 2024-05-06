# imports
import os
import torch

# local imports
from gesonn.com2SympNets import GParam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}")

if __name__ == "__main__":

    # ==============================================================
    # Parameters to be modified freely by the user
    # ==============================================================

    train = True
    train = False

    SympNetsDict = {
        "learning_rate": 5e-3,
        "nb_of_networks": 4,
        "networks_size": 5,
        "rho_min": 0,
        "rho_max": 1,
        "mu_min": 0.5,
        "mu_max": 2,
        "file_name": "SIAM",
        "symplecto_name": "bizaroid",
        "to_be_trained": True,
    }

    epochs = 100_000
    n_collocation = 10_000
    new_training = False
    new_training = True
    save_plots = False
    save_plots = True

    # ==============================================================
    # End of the modifiable area
    # ==============================================================

    if train:
        if new_training:
            try:
                os.remove(
                    "./../outputs/SympNets/net/param_" + SympNetsDict["file_name"] + ".pth"
                )
            except FileNotFoundError:
                pass

        network = GParam.Symp_Net(SympNetsDict=SympNetsDict)

        if device.type == "cpu":
            tps = network.train(
                epochs=epochs,
                n_collocation=n_collocation,
                plot_history=True,
                save_plots=save_plots,
            )
        else:
            tps = network.train(
                epochs=epochs,
                n_collocation=n_collocation,
                plot_history=True,
                save_plots=save_plots,
            )
        print(f"Computational time: {str(tps)[:4]} sec.")

    else:
        network = GParam.Symp_Net(SympNetsDict=SympNetsDict)
        network.plot_result(save_plots)
