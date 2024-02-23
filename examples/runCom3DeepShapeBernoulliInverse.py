# imports
import os

import torch

# local imports
from gesonn.com3DeepShape import bernoulliInverse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch loaded; device is {device}; script is runCom3DeepShapebernoulliInverse.py")

#==============================================================
# Parameters to be modified freely by the user
#==============================================================

train = True
# train = False

deepBernDict = {
    "pde_learning_rate": 1e-3,
    "sympnet_learning_rate": 1e-3,
    "layer_sizes": [2, 10, 20, 10, 1],
    "nb_of_networks": 2,
    "networks_size": 4,
    "rho_min": 0.5,
    "rho_max": 1,
    "file_name": "victor",
    "to_be_trained": True,
    "source_term": "one",
    "boundary_condition": "bernoulli",
}

epochs = 10_000
n_collocation = 1000
new_training = False
# new_training = True

#==============================================================
# End of the modifiable area
#==============================================================

if train:
    if new_training:
        try:
            os.remove("./../outputs/deepShape/net/" + deepBernDict["file_name"] + ".pth")
        except FileNotFoundError:
            pass

    network = bernoulliInverse.Bernoulli_Net(deepDict=deepBernDict)


    if device.type == "cpu":
        tps = network.train(epochs=epochs, n_collocation=n_collocation, plot_history=True)
    else:
        tps = network.train(epochs=epochs, n_collocation=n_collocation, plot_history=True)
    print(f"Computational time: {str(tps)[:4]} sec.")

else:
    network = bernoulliInverse.Bernoulli_Net(deepDict=deepBernDict)
