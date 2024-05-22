SIAM_one_backup.pth

    deepGeoDict = {
        "pde_learning_rate": 5e-3,
        "sympnet_learning_rate": 5e-3,
        "layer_sizes": [2, 10, 20, 20, 10, 1],
        "nb_of_networks": 2,
        "networks_size": 4,
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "SIAM_one_backup",
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "homogeneous_dirichlet",
        "pinn_activation": torch.tanh,
    }

    epochs = 750
    n_collocation = 5_000

SIAM_exp_backup.pth

    deepGeoDict = {
        "pde_learning_rate": 5e-3,
        "sympnet_learning_rate": 5e-3,
        "layer_sizes": [2, 10, 20, 40, 20, 10, 1],
        "nb_of_networks": 4,
        "networks_size": 4,
        "rho_min": 0,
        "rho_max": 1,
        "file_name": "SIAM_exp_backup",
        "to_be_trained": True,
        "source_term": "exp",
        "boundary_condition": "homogeneous_dirichlet",
        "pinn_activation": torch.tanh,
    }

    epochs = 9000
    n_collocation = 5_000

param_SIAM_constant_backup.pth

    deepGeoDict = {
        "pde_learning_rate": 5e-3,
        "sympnet_learning_rate": 5e-3,
        "layer_sizes": [3, 10, 20, 40, 20, 10, 1],
        "nb_of_networks": 4,
        "networks_size": 4,
        "rho_min": 0,
        "rho_max": 1,
        "mu_min": 0.5,
        "mu_max": 2.0,
        "file_name": "SIAM_constant_backup",
        "to_be_trained": True,
        "source_term": "one",
        "boundary_condition": "homogeneous_dirichlet",
        "pinn_activation": torch.tanh,
        "sympnet_activation": torch.tanh,
    }

    epochs = 1_000
    n_collocation = 10_000

param_SIAM_ellipse_backup.pth

    deepGeoDict = {
        "pde_learning_rate": 5e-3,
        "sympnet_learning_rate": 5e-3,
        "layer_sizes": [3, 10, 20, 40, 20, 10, 1],
        "nb_of_networks": 4,
        "networks_size": 8,
        "rho_min": 0,
        "rho_max": 1,
        "mu_min": 0.5,
        "mu_max": 2.0,
        "file_name": "SIAM_ellipse_backup",
        "to_be_trained": True,
        "source_term": "ellipse",
        "boundary_condition": "homogeneous_dirichlet",
        "pinn_activation": torch.tanh,
        "sympnet_activation": torch.tanh,
    }

    epochs = 1_000
    n_collocation = 10_000

param_SIAM_ellipse_backup.pth

    deepGeoDict = {
        "pde_learning_rate": 5e-3,
        "sympnet_learning_rate": 5e-3,
        "layer_sizes": [3, 10, 20, 40, 20, 10, 1],
        "nb_of_networks": 4,
        "networks_size": 8,
        "rho_min": 0,
        "rho_max": 1,
        "mu_min": 0.5,
        "mu_max": 2.0,
        "file_name": "SIAM_ellipse_backup",
        "to_be_trained": True,
        "source_term": "ellipse",
        "boundary_condition": "homogeneous_dirichlet",
        "pinn_activation": torch.tanh,
        "sympnet_activation": torch.tanh,
    }

    epochs = 1_000
    n_collocation = 10_000

param_SIAM_bizaroid_backup.pth

    deepGeoDict = {
        "pde_learning_rate": 5e-3,
        "sympnet_learning_rate": 5e-3,
        "layer_sizes": [3, 20, 40, 40, 20, 1],
        "nb_of_networks": 6,
        "networks_size": 8,
        "rho_min": 0,
        "rho_max": 1,
        "mu_min": 0.5,
        "mu_max": 2.0,
        "file_name": "SIAM_bizaroid_backup",
        "to_be_trained": True,
        "source_term": "bizaroid",
        "boundary_condition": "homogeneous_dirichlet",
        "pinn_activation": torch.tanh,
        "sympnet_activation": torch.tanh,
    }

    epochs = 1_000
    n_collocation = 10_000
