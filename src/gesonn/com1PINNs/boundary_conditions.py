def apply_BC(u, x, y, rho_min, rho_max, name=None):

    bc_mul = 0
    bc_add = 0
    rho_2 = x**2 + y**2

    if name == "homogeneous_dirichlet":
        bc_mul = rho_2 - rho_max**2
        if rho_min > 0:
            bc_mul = -bc_mul * (rho_min**2 - rho_2)
        bc_add = 0

    elif name == "bernoulli" and rho_min > 0:
        bc_mul = (rho_2 - rho_max**2) * (rho_min**2 - rho_2)
        bc_add = 1 - (rho_2 - rho_min**2) / (rho_max**2 - rho_min**2)

    return u * bc_mul + bc_add
